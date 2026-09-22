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
from onnx import GraphProto, helper

from ._build import attribute
from ._graph import captures, reachable
from ._passes import forward_nodes, reverse_nodes
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
        return {}
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
    return dict(zip(union, names))
