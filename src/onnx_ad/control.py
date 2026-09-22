"""Rules for the control-flow operations: If, Scan and Loop.

All three differentiate their subgraphs *in place* and replace the primal node with an
extended one, rather than computing a derivative beside it:

* forward, the extended node computes the primal outputs and their tangents at once -- an
  `If` still evaluates only the taken branch, a `Scan` still makes one pass;
* reverse, the adjoint of a subgraph recomputes the subgraph's primal from the values it was
  entered with (for a loop, a per-iteration tape of its state), then runs the adjoint.

A subgraph reads outer values by capture, and so does the derivative of one: the tangent or
adjoint of an outer tensor needs no plumbing in, it is in scope. What does need plumbing is
the way *out* -- an adjoint contribution to a captured value is an extra output of the
subgraph, summed into the outer adjoint by the node that carries it.
"""
from onnx import GraphProto, TensorProto, helper

from ._build import attribute
from ._graph import captures, reachable
from ._passes import Pairs, forward_nodes, reverse_nodes
from .rules import forward_rule, reverse_rule


def _graph(template, child, nodes, inputs=None, outputs=None, name=None):
    """A copy of `template` with new nodes, the child scope's constants, and optionally new
    inputs, outputs and name."""
    graph = GraphProto()
    graph.name = name or template.name
    graph.node.extend(nodes)
    graph.input.extend(template.input if inputs is None else inputs)
    graph.output.extend(template.output if outputs is None else outputs)
    graph.initializer.extend(template.initializer)
    if child is not None:
        graph.initializer.extend(child.b.initializers)
    graph.sparse_initializer.extend(template.sparse_initializer)
    graph.value_info.extend(template.value_info)
    return graph


def _node(template, inputs, outputs, **graphs_and_attributes):
    """A copy of a control-flow node with new operands and replaced attributes."""
    node = helper.make_node(template.op_type, list(inputs), list(outputs),
                            name=template.name, domain=template.domain)
    replaced = set(graphs_and_attributes)
    node.attribute.extend(a for a in template.attribute if a.name not in replaced)
    for key, value in graphs_and_attributes.items():
        node.attribute.append(helper.make_attribute(key, value))
    return node


def _nonempty(**attributes):
    """Only the list attributes that have entries."""
    return {key: value for key, value in attributes.items() if value}


def _bind(child, name, value, primal, seeded=True):
    """End a subgraph with `value` under a fresh name; returns its value_info."""
    out = child.b.name(name)
    child.b.alias(value, out)
    return child.value_info(out, primal, seeded)


# ============================================================================== If =====
# The branches have no inputs and the same output types, which is what makes this the
# gentle case: nothing is carried from one evaluation to the next.

BRANCHES = ("then_branch", "else_branch")


@forward_rule("If")
def _if_forward(ctx, node, tangents):
    built = []
    for key in BRANCHES:
        graph = attribute(node, key)
        child = ctx.child()
        nodes = forward_nodes(child, list(graph.node))
        built.append((key, graph, child, nodes,
                      [child.derivative.get(value.name) for value in graph.output]))
    # a tangent output exists if *either* branch produces one; the other then emits zeros,
    # so both branches keep the same output signature
    needed = [i for i in range(len(node.output)) if any(b[4][i] is not None for b in built)]
    if not needed:
        return [None]*len(node.output)
    branches = {}
    for key, graph, child, nodes, produced in built:
        outputs = list(graph.output)
        for i in needed:
            primal = graph.output[i].name
            tangent = produced[i]
            tangent = child.zeros(primal) if tangent is None else child.full(tangent, primal)
            outputs.append(_bind(child, "t_" + primal, tangent, primal))
        branches[key] = _graph(graph, child, nodes + child.b.nodes, outputs=outputs)
    names = [ctx.b.name("t_" + node.output[i]) for i in needed]
    ctx.replace(_node(node, node.input, list(node.output) + names, **branches))
    result = [None]*len(node.output)
    for i, name in zip(needed, names):
        result[i] = name
    return result


@reverse_rule("If")
def _if_reverse(ctx, node, grads):
    built = []
    for key in BRANCHES:
        graph = attribute(node, key)
        child = ctx.child()
        seeds = {graph.output[i].name: grads[i] for i in range(len(graph.output))}
        differentiated = [name for name in captures(graph) if name in ctx.depends]
        primal, adjoint, results = reverse_nodes(child, list(graph.node), seeds,
                                                 differentiated)
        built.append((key, graph, child, primal + adjoint, results))
    # one output per captured value that either branch contributes to
    union = sorted({name for b in built for name in b[4]
                    if name in ctx.depends and ctx.asked_for(name)})
    if not union:
        return Pairs()
    branches = {}
    for key, graph, child, nodes, results in built:
        outputs = []
        for name in union:
            value = results.get(name)
            value = child.zeros(name) if value is None else child.full(value, name)
            outputs.append(_bind(child, "a_" + name, value, name))
        branches[key] = _graph(graph, child, nodes + child.b.nodes, outputs=outputs,
                               name=graph.name + "_adjoint")
    names = [ctx.b.name("a_" + name) for name in union]
    ctx.b.nodes.append(_node(node, node.input[:1], names, **branches))
    return Pairs(zip(union, names))


# ============================================================================= Scan ====
# body: (state..., slice...) -> (state..., output slice...), with the trip count the length
# of the scan axis. A tangent rides along as extra state and extra scan inputs; an adjoint is
# a second Scan running backwards over a tape of the per-iteration input state.

def _scan_layout(node):
    """The operand counts and the axis/direction attributes, defaults filled in."""
    body = attribute(node, "body")
    m = attribute(node, "num_scan_inputs")
    n = len(node.input) - m
    k = len(node.output) - n
    return (body, n, m, k,
            list(attribute(node, "scan_input_axes") or [0]*m),
            list(attribute(node, "scan_input_directions") or [0]*m),
            list(attribute(node, "scan_output_axes") or [0]*k),
            list(attribute(node, "scan_output_directions") or [0]*k))


def _carried(body, n, in_first, out_first, live, seeds):
    """Which of the `n` loop-carried values carry a derivative.

    A fixed point: a state initialised to a constant becomes differentiated as soon as the
    body mixes a differentiated value into it, and then carries that into the next
    iteration. The carried values are body inputs `in_first...` and body outputs
    `out_first...` -- 0 and 0 for a Scan, 2 and 1 for a Loop, whose body also takes the
    iteration number and condition and returns the condition first.
    """
    inputs = [value.name for value in body.input]
    outputs = [value.name for value in body.output]
    carried = list(live)
    while True:
        reached = reachable(body.node, set(seeds) | {inputs[in_first + j] for j in range(n)
                                                     if carried[j]})
        grown = [carried[j] or outputs[out_first + j] in reached for j in range(n)]
        if grown == carried:
            return carried
        carried = grown


@forward_rule("Scan")
def _scan_forward(ctx, node, tangents):
    body, n, m, k, in_axes, in_dirs, out_axes, out_dirs = _scan_layout(node)
    inputs = [value.name for value in body.input]
    outputs = [value.name for value in body.output]
    outer = {name for name in captures(body) if ctx.derivative.get(name)}
    sliced = [tangents[n + j] is not None for j in range(m)]
    seeds = outer | {inputs[n + j] for j in range(m) if sliced[j]}
    carried = _carried(body, n, 0, 0, [tangents[j] is not None for j in range(n)], seeds)

    child = ctx.child()
    state_in, slice_in = [], []
    for j in range(n):
        if carried[j]:
            name = child.b.name("t_" + inputs[j])
            child.derivative[inputs[j]] = name
            state_in.append((j, child.value_info(name, inputs[j])))
    for j in range(m):
        if sliced[j]:
            name = child.b.name("t_" + inputs[n + j])
            child.derivative[inputs[n + j]] = name
            slice_in.append((j, child.value_info(name, inputs[n + j])))
    nodes = forward_nodes(child, list(body.node))
    state_out, slice_out = [], []
    for j, _ in state_in:
        tangent = child.derivative.get(outputs[j])
        tangent = child.zeros(outputs[j]) if tangent is None else child.full(tangent, outputs[j])
        state_out.append(_bind(child, "t_" + outputs[j], tangent, outputs[j]))
    for j in range(k):
        tangent = child.derivative.get(outputs[n + j])
        if tangent is not None:
            slice_out.append((j, _bind(child, "t_" + outputs[n + j],
                                       child.full(tangent, outputs[n + j]), outputs[n + j])))
    new_body = _graph(
        body, child, nodes + child.b.nodes,
        inputs=list(body.input[:n]) + [v for _, v in state_in] + list(body.input[n:])
        + [v for _, v in slice_in],
        outputs=list(body.output[:n]) + state_out + list(body.output[n:])
        + [v for _, v in slice_out])

    # the tangents going in have to be exactly the shape the body expects every iteration
    initial = [ctx.zeros(node.input[j]) if tangents[j] is None
               else ctx.full(tangents[j], node.input[j]) for j, _ in state_in]
    scanned = [ctx.full(tangents[n + j], node.input[n + j]) for j, _ in slice_in]
    finals = [ctx.b.name("t_" + node.output[j]) for j, _ in state_in]
    stacked = [ctx.b.name("t_" + node.output[n + j]) for j, _ in slice_out]
    # a tangent has a trailing seed axis, so an axis counted from the back is normalized
    ctx.replace(_node(
        node, list(node.input[:n]) + initial + list(node.input[n:]) + scanned,
        list(node.output[:n]) + finals + list(node.output[n:]) + stacked,
        body=new_body, num_scan_inputs=m + len(slice_in),
        scan_input_axes=in_axes + [in_axes[j] % ctx.rank(node.input[n + j])
                                   for j, _ in slice_in],
        scan_input_directions=in_dirs + [in_dirs[j] for j, _ in slice_in],
        # a Scan with no scan outputs -- the reverse sweep of a Loop is one -- has none
        **_nonempty(scan_output_axes=out_axes + [out_axes[j] % ctx.rank(node.output[n + j])
                                                 for j, _ in slice_out],
                    scan_output_directions=out_dirs + [out_dirs[j] for j, _ in slice_out])))
    result = [None]*len(node.output)
    for (j, _), name in zip(state_in, finals):
        result[j] = name
    for (j, _), name in zip(slice_out, stacked):
        result[n + j] = name
    return result


@reverse_rule("Scan")
def _scan_reverse(ctx, node, grads):
    body, n, m, k, in_axes, in_dirs, out_axes, out_dirs = _scan_layout(node)
    inputs = [value.name for value in body.input]
    outputs = [value.name for value in body.output]
    outer = [name for name in sorted(captures(body)) if name in ctx.depends]
    sliced = [node.input[n + j] in ctx.depends for j in range(m)]
    carried = _carried(body, n, 0, 0, [node.input[j] in ctx.depends for j in range(n)],
                       set(outer) | {inputs[n + j] for j in range(m) if sliced[j]})

    # 1. The primal Scan, also taping each iteration's *input* state: with the slices, which
    #    the outer graph still has, that is everything the body needs to be recomputed.
    tapes = []
    if n:
        extra_nodes, extra_outputs = [], []
        for j in range(n):
            inner = ctx.b.name("tape_" + inputs[j])
            extra_nodes.append(helper.make_node("Identity", [inputs[j]], [inner]))
            extra_outputs.append(ctx.value_info(inner, inputs[j], seeded=False))
            tapes.append(ctx.b.name("tape_" + node.input[j]))
        taped = _graph(body, None, list(body.node) + extra_nodes,
                       outputs=list(body.output) + extra_outputs)
        ctx.replace_primal(node, _node(node, node.input, list(node.output) + tapes, body=taped,
                                       scan_output_axes=out_axes + [0]*n,
                                       scan_output_directions=out_dirs + [0]*n))

    # 2. A Scan running the other way: recompute the body from the taped state and the
    #    slice, then its adjoint. Captured values accumulate in the *state* -- a weight used
    #    every iteration costs O(|w|), not O(T |w|) as a scan output would.
    child = ctx.child()
    state_adj = [(j, child.b.name("a_" + outputs[j])) for j in range(n) if carried[j]]
    accumulated = [(name, child.b.name("acc_" + name)) for name in outer
                   if ctx.asked_for(name)]
    output_adj = [(j, child.b.name("a_" + outputs[n + j])) for j in range(k)
                  if grads[n + j] is not None]
    seeds = {outputs[j]: name for j, name in state_adj}
    seeds.update({outputs[n + j]: name for j, name in output_adj})
    differentiated = ([inputs[j] for j, _ in state_adj]
                      + [inputs[n + j] for j in range(m) if sliced[j]] + outer)
    primal, adjoint, results = reverse_nodes(child, list(body.node), seeds, differentiated)

    state_out = []
    for j, _ in state_adj:
        value = results.get(inputs[j])
        value = child.zeros(inputs[j]) if value is None else child.full(value, inputs[j])
        state_out.append(_bind(child, "a_" + inputs[j], value, inputs[j]))
    acc_out = []
    for name, running in accumulated:
        value = results.get(name)
        total = running if value is None else \
            child.b.op("Add", [running, child.full(value, name)])
        acc_out.append(_bind(child, "acc_" + name, total, name))
    slice_out = []
    for j in range(m):
        if sliced[j]:
            value = results.get(inputs[n + j])
            value = child.zeros(inputs[n + j]) if value is None else \
                child.full(value, inputs[n + j])
            slice_out.append((j, _bind(child, "a_" + inputs[n + j], value, inputs[n + j])))
    swept = _graph(
        body, child, primal + adjoint + child.b.nodes,
        inputs=[child.value_info(name, outputs[j]) for j, name in state_adj]
        + [child.value_info(running, name) for name, running in accumulated]
        + list(body.input)  # tape slices and scan slices, under the body's own names
        + [child.value_info(name, outputs[n + j]) for j, name in output_adj],
        outputs=state_out + acc_out + [value for _, value in slice_out],
        name=body.name + "_adjoint")

    # Every scan input is read backwards relative to how the primal read it, and every
    # adjoint slice is written back where the primal read that slice from.
    initial = [ctx.zeros(node.output[j]) if grads[j] is None
               else ctx.full(grads[j], node.output[j]) for j, _ in state_adj]
    zeros = [ctx.zeros(name) for name, _ in accumulated]
    scanned = tapes + list(node.input[n:]) + [ctx.full(grads[n + j], node.output[n + j])
                                              for j, _ in output_adj]
    adj_initial = [ctx.b.name("a_" + node.input[j]) for j, _ in state_adj]
    totals = [ctx.b.name("acc_" + name) for name, _ in accumulated]
    adj_slices = [ctx.b.name("a_" + node.input[n + j]) for j, _ in slice_out]
    ctx.b.nodes.append(helper.make_node(
        "Scan", initial + zeros + scanned, adj_initial + totals + adj_slices,
        body=swept, num_scan_inputs=n + m + len(output_adj),
        scan_input_axes=[0]*n + in_axes + [out_axes[j] % ctx.rank(node.output[n + j])
                                           for j, _ in output_adj],
        scan_input_directions=[1]*n + [1 - d for d in in_dirs]
        + [1 - out_dirs[j] for j, _ in output_adj],
        # with nothing scanned differentiated there are no scan outputs, and an empty list
        # has no attribute type onnx could infer
        **_nonempty(scan_output_axes=[in_axes[j] % ctx.rank(node.input[n + j])
                                      for j, _ in slice_out],
                    scan_output_directions=[1 - in_dirs[j] for j, _ in slice_out])))

    pairs = Pairs((node.input[j], name) for (j, _), name in zip(state_adj, adj_initial))
    pairs.extend((name, total) for (name, _), total in zip(accumulated, totals))
    pairs.extend((node.input[n + j], name) for (j, _), name in zip(slice_out, adj_slices))
    return pairs


# ============================================================================= Loop ====
# body: (iteration, condition, carried...) -> (condition, carried..., output slice...). The
# condition can stop the loop early, so the trip count is only known once the primal has run
# -- and then it is simply the length of a tape. That is what makes the reverse sweep a
# Scan: it no longer needs to decide when to stop, and reading the tape backwards removes all
# index arithmetic.

def _runs_at_least_once(ctx, node):
    """True when the first iteration is certain: a positive constant trip count, and no
    condition input or a constant true one. (The body's condition only affects iterations
    after the first.)"""
    trips = ctx.values.get(node.input[0]) if node.input[0] else None
    if trips is None or int(trips.reshape(-1)[0]) <= 0:
        return False
    if len(node.input) > 1 and node.input[1]:
        condition = ctx.values.get(node.input[1])
        return condition is not None and bool(condition.reshape(-1)[0])
    return True


@forward_rule("Loop")
def _loop_forward(ctx, node, tangents):
    # the trip count and condition are primal only: termination depends on primal values,
    # so a tangent never changes it (where the count does change with the input, the
    # derivative is one-sided -- the convention already taken at Relu(0))
    body = attribute(node, "body")
    n = len(node.input) - 2
    k = len(node.output) - n
    inputs = [value.name for value in body.input]
    outputs = [value.name for value in body.output]
    outer = {name for name in captures(body) if ctx.derivative.get(name)}
    carried = _carried(body, n, 2, 1, [tangents[2 + j] is not None for j in range(n)], outer)

    child = ctx.child()
    state_in = []
    for j in range(n):
        if carried[j]:
            name = child.b.name("t_" + inputs[2 + j])
            child.derivative[inputs[2 + j]] = name
            state_in.append((j, child.value_info(name, inputs[2 + j])))
    nodes = forward_nodes(child, list(body.node))
    state_out, slice_out = [], []
    for j, _ in state_in:
        primal = outputs[1 + j]
        tangent = child.derivative.get(primal)
        tangent = child.zeros(primal) if tangent is None else child.full(tangent, primal)
        state_out.append(_bind(child, "t_" + primal, tangent, primal))
    for j in range(k):
        primal = outputs[1 + n + j]
        tangent = child.derivative.get(primal)
        if tangent is not None:
            slice_out.append((j, _bind(child, "t_" + primal, child.full(tangent, primal),
                                       primal)))
    new_body = _graph(
        body, child, nodes + child.b.nodes,
        inputs=list(body.input) + [v for _, v in state_in],
        outputs=list(body.output[:1 + n]) + state_out + list(body.output[1 + n:])
        + [v for _, v in slice_out])
    initial = [ctx.zeros(node.input[2 + j]) if tangents[2 + j] is None
               else ctx.full(tangents[2 + j], node.input[2 + j]) for j, _ in state_in]
    finals = [ctx.b.name("t_" + node.output[j]) for j, _ in state_in]
    stacked = [ctx.b.name("t_" + node.output[n + j]) for j, _ in slice_out]
    ctx.replace(_node(node, list(node.input) + initial,
                      list(node.output[:n]) + finals + list(node.output[n:]) + stacked,
                      body=new_body))
    result = [None]*len(node.output)
    for (j, _), name in zip(state_in, finals):
        result[j] = name
    for (j, _), name in zip(slice_out, stacked):
        result[n + j] = name
    return result


@reverse_rule("Loop")
def _loop_reverse(ctx, node, grads):
    body = attribute(node, "body")
    n = len(node.input) - 2
    k = len(node.output) - n
    inputs = [value.name for value in body.input]
    outputs = [value.name for value in body.output]
    outer = [name for name in sorted(captures(body)) if name in ctx.depends]
    carried = _carried(body, n, 2, 1, [node.input[2 + j] in ctx.depends for j in range(n)],
                       set(outer))

    # 1. The primal Loop, taping what the body was entered with each iteration: the
    #    iteration number, the condition and every carried value. The tape's length is the
    #    trip count.
    extra_nodes, extra_outputs, tapes = [], [], []
    for j in range(2 + n):
        inner = ctx.b.name("tape_" + inputs[j])
        extra_nodes.append(helper.make_node("Identity", [inputs[j]], [inner]))
        extra_outputs.append(ctx.value_info(inner, inputs[j], seeded=False))
        tapes.append(ctx.b.name("tape_" + (node.input[j] if j >= 2 and node.input[j]
                                           else inputs[j])))
    taped = _graph(body, None, list(body.node) + extra_nodes,
                   outputs=list(body.output) + extra_outputs)
    ctx.replace_primal(node, _node(node, node.input, list(node.output) + tapes, body=taped))

    # 2. The reverse sweep: exactly Scan's, over the tape.
    child = ctx.child()
    state_adj = [(j, child.b.name("a_" + outputs[1 + j])) for j in range(n) if carried[j]]
    accumulated = [(name, child.b.name("acc_" + name)) for name in outer
                   if ctx.asked_for(name)]
    output_adj = [(j, child.b.name("a_" + outputs[1 + n + j])) for j in range(k)
                  if grads[n + j] is not None]
    seeds = {outputs[1 + j]: name for j, name in state_adj}
    seeds.update({outputs[1 + n + j]: name for j, name in output_adj})
    differentiated = [inputs[2 + j] for j, _ in state_adj] + outer
    primal, adjoint, results = reverse_nodes(child, list(body.node), seeds, differentiated)

    state_out = []
    for j, _ in state_adj:
        value = results.get(inputs[2 + j])
        value = child.zeros(inputs[2 + j]) if value is None else \
            child.full(value, inputs[2 + j])
        state_out.append(_bind(child, "a_" + inputs[2 + j], value, inputs[2 + j]))
    acc_out = []
    for name, running in accumulated:
        value = results.get(name)
        total = running if value is None else \
            child.b.op("Add", [running, child.full(value, name)])
        acc_out.append(_bind(child, "acc_" + name, total, name))
    if not state_out and not acc_out:
        return Pairs()
    swept = _graph(
        body, child, primal + adjoint + child.b.nodes,
        inputs=[child.value_info(name, outputs[1 + j]) for j, name in state_adj]
        + [child.value_info(running, name) for name, running in accumulated]
        + list(body.input)  # the taped iteration, condition and carried values
        + [child.value_info(name, outputs[1 + n + j]) for j, name in output_adj],
        outputs=state_out + acc_out, name=body.name + "_adjoint")

    initial = [ctx.zeros(node.output[j]) if grads[j] is None
               else ctx.full(grads[j], node.output[j]) for j, _ in state_adj]
    zeros = [ctx.zeros(name) for name, _ in accumulated]
    scanned = tapes + [ctx.full(grads[n + j], node.output[n + j]) for j, _ in output_adj]
    count = len(scanned)
    sweep_out = [ctx.b.name("a_" + node.input[2 + j]) for j, _ in state_adj] + \
        [ctx.b.name("acc_" + name) for name, _ in accumulated]
    sweep = helper.make_node("Scan", initial + zeros + scanned, sweep_out, body=swept,
                             num_scan_inputs=count, scan_input_axes=[0]*count,
                             scan_input_directions=[1]*count)
    primals = [node.input[2 + j] for j, _ in state_adj] + [name for name, _ in accumulated]

    # 3. ONNX Runtime rejects a Scan over a zero-length axis, and a Loop may legitimately
    #    run zero times -- in which case the adjoint passes straight through.
    if _runs_at_least_once(ctx, node):
        ctx.b.nodes.append(sweep)
        results_ = sweep_out
    else:
        trips = ctx.b.op("Size", [tapes[0]], stem="trips")
        ran = ctx.b.op("Greater", [trips, ctx.b.constant(0, TensorProto.INT64, ())],
                       stem="ran")
        then = helper.make_graph([sweep], "swept", [], [
            ctx.value_info(name, primal) for name, primal in zip(sweep_out, primals)])
        skipped = [ctx.b.name(name + "_skipped") for name in sweep_out]
        otherwise = helper.make_graph(
            [helper.make_node("Identity", [value], [name])
             for value, name in zip(initial + zeros, skipped)], "skipped", [],
            [ctx.value_info(name, primal) for name, primal in zip(skipped, primals)])
        results_ = [ctx.b.name(name + "_guarded") for name in sweep_out]
        ctx.b.nodes.append(helper.make_node("If", [ran], results_, then_branch=then,
                                            else_branch=otherwise))
    return Pairs(zip(primals, results_))
