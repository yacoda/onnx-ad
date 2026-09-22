"""Unroll loops whose trip count is known before the model runs.

A `Scan` over a statically sized axis, or a `Loop` with a constant trip count and no
condition input, is the same computation as its body copied that many times. The copy is a
flat graph that the rest of the rule table already handles, which makes this useful in two
ways: on its own, for short fixed-length recurrences (an integrator with a few stages, a
short RNN) at the cost of a graph that grows with the trip count; and as an oracle, since a
derivative of the unrolled graph comes from entirely different code than the derivative of
the loop.

It is never the answer for a trip count only known at run time; such loops are left as they
are.
"""
from onnx import TensorProto, helper

from ._build import Builder, Shapes, attribute
from ._graph import all_constants, all_names, defines, subgraphs, walk


class UnrollTooLarge(Exception):
    """Unrolling would exceed the node budget."""


def unroll(model, max_nodes=100_000):
    """A copy of `model` with every statically bounded `Scan` and `Loop` unrolled.

    Loops nested inside a body are unrolled too, as are loops inside `If` branches. A loop
    whose trip count is only known at run time is left in place. `max_nodes` bounds the size
    of the result; exceeding it raises `UnrollTooLarge` rather than producing a graph nobody
    can load.
    """
    result = type(model)()
    result.CopyFrom(model)
    graph = result.graph
    state = _State(model, max_nodes)
    nodes = state.nodes(list(graph.node))
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.initializer.extend(state.b.initializers)
    graph.ClearField("value_info")
    return result


def inline_constant_ifs(model):
    """A copy of `model` with every `If` whose condition is a constant replaced by the
    branch it would take.

    Differentiating the result needs no recursion at all, and the branch not taken is not
    kept around to be differentiated for nothing. Branches are inlined at any depth.
    """
    state = _State(model, max_nodes=float("inf"))
    if not any(node.op_type == "If" and node.input[0] in state.values
               for scope in walk(model.graph) for node in scope.node):
        return model
    result = type(model)()
    result.CopyFrom(model)
    graph = result.graph
    nodes = state.inline(list(graph.node))
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.initializer.extend(state.b.initializers)
    return result


class _State:
    def __init__(self, model, max_nodes):
        self.shapes = Shapes(model)
        self.values = all_constants(model.graph)
        self.b = Builder(all_names(model.graph))
        self.budget = max_nodes
        self._hoisted = {}

    def spend(self, count):
        self.budget -= count
        if self.budget < 0:
            raise UnrollTooLarge("unrolling exceeds the node budget; raise max_nodes or "
                                 "leave this loop rolled")

    # ------------------------------------------------------------------ traversal ----
    def nodes(self, nodes):
        out = []
        for node in nodes:
            if node.op_type == "Scan" and self.scan_trips(node) is not None:
                out.extend(self.nodes(self.unroll_scan(node)))
            elif node.op_type == "Loop" and self.loop_trips(node) is not None:
                out.extend(self.nodes(self.unroll_loop(node)))
            elif subgraphs(node):
                out.append(self.descend(node))
            else:
                out.append(node)
        return out

    def inline(self, nodes):
        out = []
        for node in nodes:
            condition = self.values.get(node.input[0]) if node.op_type == "If" else None
            if condition is not None:
                taken = attribute(node, "then_branch" if bool(condition.reshape(-1)[0])
                                  else "else_branch")
                copied, outputs = self.instantiate(taken, {})
                out.extend(self.inline(copied))
                out.extend(helper.make_node("Identity", [value], [name])
                           for value, name in zip(outputs, node.output))
            elif subgraphs(node):
                out.append(self.descend(node, self.inline))
            else:
                out.append(node)
        return out

    def descend(self, node, into=None):
        """Rewrite inside the subgraphs of a node that is not itself rewritten."""
        into = into or self.nodes
        copy = helper.make_node(node.op_type, list(node.input), list(node.output),
                                name=node.name, domain=node.domain)
        for attribute_ in node.attribute:
            if attribute_.type == attribute_.GRAPH:
                inner = type(attribute_.g)()
                inner.CopyFrom(attribute_.g)
                body = into(list(inner.node))
                inner.ClearField("node")
                inner.node.extend(body)
                copy.attribute.append(helper.make_attribute(attribute_.name, inner))
            else:
                copy.attribute.append(attribute_)
        return copy

    # ------------------------------------------------------------------ trip counts --
    def scan_trips(self, node):
        count = attribute(node, "num_scan_inputs")
        first = node.input[len(node.input) - count]
        axis = (attribute(node, "scan_input_axes") or [0])[0]
        shape = self.shapes.shape(first)
        if shape is None:
            return None
        extent = shape[axis % len(shape)]
        return extent if extent else None

    def loop_trips(self, node):
        # (trip_count, "") is a for-loop: the body's condition output is ignored
        if not node.input[0] or (len(node.input) > 1 and node.input[1]):
            return None
        value = self.values.get(node.input[0])
        if value is None or int(value.reshape(-1)[0]) <= 0:
            return None
        return int(value.reshape(-1)[0])

    # ------------------------------------------------------------------ copying ------
    def hoisted(self, body):
        """A body's own initializers, moved to the outer graph under fresh names -- once
        per body, since every copy of it reads the same constants.

        The cache holds the body itself as well as its mapping. A protobuf submessage is a
        proxy nothing else keeps alive, and once one is freed a different body can be handed
        the same `id` -- and would silently inherit the wrong constants.
        """
        key = id(body)
        if key not in self._hoisted or self._hoisted[key][0] is not body:
            names = {}
            for tensor in body.initializer:
                fresh = self.b.name(tensor.name + "_h")
                copy = type(tensor)()
                copy.CopyFrom(tensor)
                copy.name = fresh
                self.b.initializers.append(copy)
                self.shapes.declare(fresh, tuple(tensor.dims), tensor.data_type)
                self.values[fresh] = self.values.get(tensor.name)
                names[tensor.name] = fresh
            self._hoisted[key] = (body, names)
        return self._hoisted[key][1]

    def instantiate(self, body, bindings):
        """One copy of a body with its inputs bound; returns the nodes and output names."""
        mapping = dict(self.hoisted(body))
        mapping.update(bindings)
        copied = []
        for node in body.node:
            copied.append(self.rename(node, mapping))
        self.spend(len(copied))
        return copied, [mapping.get(value.name, value.name) for value in body.output]

    def rename(self, node, mapping):
        """A copy of a body node with fresh output names and inputs mapped."""
        outputs = []
        for name in node.output:
            if not name:
                outputs.append("")
                continue
            fresh = self.b.name(name + "_u")
            shape = self.shapes.shape(name)
            if shape is not None:
                self.shapes.declare(fresh, shape, self.shapes.dtype(name))
            mapping[name] = fresh
            outputs.append(fresh)
        copy = helper.make_node(node.op_type, [mapping.get(n, n) for n in node.input],
                                outputs, name=node.name, domain=node.domain)
        for attribute_ in node.attribute:
            if attribute_.type == attribute_.GRAPH:
                copy.attribute.append(helper.make_attribute(
                    attribute_.name, _substitute(attribute_.g, mapping)))
            else:
                copy.attribute.append(attribute_)
        return copy

    def index(self, i):
        return self.b.constant(i, TensorProto.INT64, ())

    # ------------------------------------------------------------------ Scan ---------
    def unroll_scan(self, node):
        body = attribute(node, "body")
        m = attribute(node, "num_scan_inputs")
        n = len(node.input) - m
        k = len(node.output) - n
        in_axes = list(attribute(node, "scan_input_axes") or [0]*m)
        in_dirs = list(attribute(node, "scan_input_directions") or [0]*m)
        out_axes = list(attribute(node, "scan_output_axes") or [0]*k)
        out_dirs = list(attribute(node, "scan_output_directions") or [0]*k)
        trips = self.scan_trips(node)
        nodes, states, slices = [], list(node.input[:n]), [[] for _ in range(k)]
        for i in range(trips):
            bindings = {body.input[j].name: states[j] for j in range(n)}
            for j in range(m):
                source = node.input[n + j]
                position = trips - 1 - i if in_dirs[j] else i
                sliced = self.b.name(source + "_slice")
                nodes.append(helper.make_node("Gather", [source, self.index(position)],
                                              [sliced], axis=in_axes[j]))
                bindings[body.input[n + j].name] = sliced
            copied, outputs = self.instantiate(body, bindings)
            nodes.extend(copied)
            states = outputs[:n]
            for j in range(k):
                slices[j].append(outputs[n + j])
        for j in range(n):
            nodes.append(helper.make_node("Identity", [states[j]], [node.output[j]]))
        for j in range(k):
            ordered = slices[j][::-1] if out_dirs[j] else slices[j]
            nodes.extend(self.stack(ordered, out_axes[j], node.output[n + j]))
        return nodes

    # ------------------------------------------------------------------ Loop ---------
    def unroll_loop(self, node):
        body = attribute(node, "body")
        n = len(node.input) - 2
        k = len(node.output) - n
        trips = self.loop_trips(node)
        true = self.b.constant(True, TensorProto.BOOL, ())
        nodes, states, slices = [], list(node.input[2:]), [[] for _ in range(k)]
        for i in range(trips):
            bindings = {body.input[0].name: self.index(i), body.input[1].name: true}
            bindings.update({body.input[2 + j].name: states[j] for j in range(n)})
            copied, outputs = self.instantiate(body, bindings)
            nodes.extend(copied)
            states = outputs[1:1 + n]  # outputs[0] is the condition, ignored in a for-loop
            for j in range(k):
                slices[j].append(outputs[1 + n + j])
        for j in range(n):
            nodes.append(helper.make_node("Identity", [states[j]], [node.output[j]]))
        for j in range(k):
            nodes.extend(self.stack(slices[j], 0, node.output[n + j]))
        return nodes

    def stack(self, slices, axis, name):
        """Stack per-iteration slices along a new axis, as Scan and Loop outputs are."""
        if axis < 0:
            rank = self.shapes.shape(slices[0])
            axis += (len(rank) + 1) if rank is not None else 0
        unsqueezed, nodes = [], []
        for piece in slices:
            lifted = self.b.name(piece + "_row")
            nodes.append(helper.make_node("Unsqueeze", [piece, self.b.constant([axis], TensorProto.INT64, (1,))],
                                          [lifted]))
            unsqueezed.append(lifted)
        nodes.append(helper.make_node("Concat", unsqueezed, [name], axis=axis))
        return nodes


def _substitute(graph, mapping):
    """A copy of a nested graph whose references to outer names follow `mapping`."""
    local = defines(graph)
    visible = {name: fresh for name, fresh in mapping.items() if name not in local}
    copy = type(graph)()
    copy.CopyFrom(graph)
    copy.ClearField("node")
    for node in graph.node:
        inner = helper.make_node(node.op_type, [visible.get(n, n) for n in node.input],
                                 list(node.output), name=node.name, domain=node.domain)
        for attribute_ in node.attribute:
            if attribute_.type == attribute_.GRAPH:
                inner.attribute.append(helper.make_attribute(
                    attribute_.name, _substitute(attribute_.g, visible)))
            else:
                inner.attribute.append(attribute_)
        copy.node.append(inner)
    return copy
