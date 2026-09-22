"""The two walks, over a node list in a scope, so they can recurse into subgraphs.

`forward_nodes` interleaves: each node's tangent nodes come right after it. Emitting them
all after the primal graph, as a flat graph allows, stops working once a node can produce
primal *and* derivative outputs -- the extended `If` or `Scan` that replaces a primal one
consumes tangents computed earlier and feeds primal values to nodes computed later, so it
has to sit where the primal node sat.

`reverse_nodes` keeps the primal nodes first and the adjoint after them, since every adjoint
rule may read any primal value. A rule may substitute a primal node (a `Scan` that also
tapes its state), and what is left pending when the walk ends -- contributions to the
scope's inputs and captured outer values -- is handed back to the caller.
"""
from ._build import UnsupportedOperator
from ._graph import node_reads, reachable
from .rules import FORWARD, REVERSE


class Pairs(list):
    """Adjoint contributions as (name, value) pairs.

    For a rule whose operands include captured values, contributions cannot be matched to
    `node.input` by position. Nor can they be a dict: the same outer tensor may feed a node
    twice -- as a loop's initial state and as a scan input -- and both contributions count.
    """


def forward_nodes(ctx, nodes):
    """The node list with tangent nodes interleaved; `ctx.derivative` gains the tangents."""
    emitted = []
    for node in nodes:
        if not any(ctx.derivative.get(name) for name in node_reads(node)):
            emitted.append(node)  # nothing differentiated reaches it; part of the primal
            continue
        if node.op_type not in FORWARD:
            raise UnsupportedOperator(
                "no forward rule for %s (node '%s'); an operation only needs one when a "
                "differentiated value reaches it"
                % (node.op_type, node.name or node.output[0]))
        ctx.b.nodes = []
        ctx._replacement = None
        tangents = [ctx.derivative.get(name) if name else None for name in node.input]
        produced = FORWARD[node.op_type](ctx, node, tangents)
        if produced is None or isinstance(produced, str):
            produced = [produced]
        for name, tangent in zip(node.output, produced):
            if name and tangent is not None:
                ctx.derivative[name] = tangent
        if ctx._replacement is not None:
            # helpers the rule emitted (outer tangents fed into a subgraph) come first
            emitted.extend(ctx.b.nodes)
            emitted.append(ctx._replacement)
        else:
            emitted.append(node)
            emitted.extend(ctx.b.nodes)
    ctx.b.nodes = []
    ctx._replacement = None
    return emitted


def reverse_nodes(ctx, nodes, seeds, differentiated):
    """Reverse-differentiate a node list.

    `seeds` maps values to their incoming adjoint and `differentiated` names the values the
    derivative is taken with respect to. Returns the primal node list (with any substitutions
    a rule asked for), the adjoint node list, and the summed contributions to values the
    list does not itself define -- its inputs and captured outer values.
    """
    depends = reachable(nodes, differentiated)
    ctx.depends = depends
    ctx._replacements = {}
    ctx.b.nodes = []
    pending = {}

    def accumulate(name, contribution):
        pending.setdefault(name, []).append(contribution)

    def take(name):
        """Every contribution to `name` is in hand by the time its own node is reached."""
        return ctx.sum(pending.pop(name, []))

    for name, seed in seeds.items():
        if seed is not None:
            accumulate(name, seed)

    for node in reversed(nodes):
        reads = node_reads(node)
        if not reads & depends:
            continue  # constant with respect to the differentiated inputs
        grads = [take(name) if name else None for name in node.output]
        if all(grad is None for grad in grads):
            continue  # this node's results do not reach a seeded output
        if node.op_type not in REVERSE:
            raise UnsupportedOperator(
                "no reverse rule for %s (node '%s'); an operation only needs one when it "
                "lies between a differentiated input and a seeded output"
                % (node.op_type, node.name or node.output[0]))
        ctx.wanted = {name for name in reads if name in depends}
        contributions = REVERSE[node.op_type](ctx, node, grads)
        if isinstance(contributions, str):  # a rule must return one entry per operand
            raise TypeError("the reverse rule for %s returned a tensor, not a list"
                            % node.op_type)
        items = contributions if isinstance(contributions, Pairs) else \
            zip(node.input, contributions)
        for name, contribution in items:
            if name and contribution is not None and name in depends:
                accumulate(name, contribution)
        ctx.wanted = None

    results = {name: take(name) for name in list(pending)}
    primal = []
    for node in nodes:
        original, replacement = ctx._replacements.get(id(node), (None, None))
        primal.append(replacement if original is node else node)
    adjoint = ctx.b.nodes
    ctx.b.nodes = []
    ctx._replacements = {}
    return primal, adjoint, results
