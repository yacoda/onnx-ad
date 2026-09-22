"""Graph structure the passes need once a node can carry a subgraph.

A subgraph of `If`, `Scan` or `Loop` reads outer-scope tensors by name, without listing
them as inputs. Two consequences run through everything here: a node's real operands are
its explicit inputs *plus* whatever its subgraphs capture, and a derivative of an outer value
needs no plumbing into a subgraph at all -- it is simply in scope there.
"""
from onnx import AttributeProto, numpy_helper, helper


def subgraphs(node):
    """The graphs a node carries as attributes (`then_branch`, `body`, ...)."""
    found = []
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            found.append(attribute.g)
        elif attribute.type == AttributeProto.GRAPHS:
            found.extend(attribute.graphs)
    return found


def defines(graph):
    """Names a graph binds itself: its inputs, initializers and node outputs."""
    names = {value.name for value in graph.input}
    names.update(tensor.name for tensor in graph.initializer)
    names.update(name for node in graph.node for name in node.output if name)
    return names


def captures(graph):
    """Outer-scope names a graph reads, through any depth of nested subgraphs."""
    reads = set()
    for node in graph.node:
        reads |= node_reads(node)
    reads.update(value.name for value in graph.output)  # an output may be an outer name
    return reads - defines(graph) - {""}


def node_reads(node):
    """Everything a node depends on: its explicit inputs and its subgraphs' captures."""
    reads = {name for name in node.input if name}
    for graph in subgraphs(node):
        reads |= captures(graph)
    return reads


def reachable(nodes, live):
    """Values downstream of `live` within a node list (a forward dependency sweep)."""
    live = set(live)
    for node in nodes:
        if node_reads(node) & live:
            live.update(name for name in node.output if name)
    return live


def walk(graph):
    """This graph and every graph nested inside it."""
    yield graph
    for node in graph.node:
        for inner in subgraphs(node):
            yield from walk(inner)


def all_names(graph):
    """Every tensor name bound anywhere in a graph, subgraphs included.

    New names must avoid all of them: a subgraph tensor whose name collides with an outer
    one shadows it, silently.
    """
    names = set()
    for scope in walk(graph):
        names.update(value.name for value in
                     list(scope.input) + list(scope.output) + list(scope.value_info))
        names.update(tensor.name for tensor in scope.initializer)
        names.update(name for node in scope.node for name in node.output if name)
    return names


def all_constants(graph):
    """Initializers and Constant-node values anywhere in a graph, as numpy arrays."""
    values = {}
    for scope in walk(graph):
        for tensor in scope.initializer:
            values[tensor.name] = numpy_helper.to_array(tensor)
        for node in scope.node:
            if node.op_type == "Constant":
                tensor = next((helper.get_attribute_value(a) for a in node.attribute
                               if a.name == "value"), None)
                if tensor is not None:
                    values[node.output[0]] = numpy_helper.to_array(tensor)
    return values
