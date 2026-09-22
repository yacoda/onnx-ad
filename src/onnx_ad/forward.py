"""Forward mode: emit a model computing `fwd_y = J . fwd_x`.

One walk over the graph in topological order, carrying a tangent per value. A value whose
tangent is absent is constant with respect to the seeds -- which is the whole reason the
emitted graph stays small, since every weight in a network is such a value and costs
nothing. The primal graph is kept verbatim and the tangent nodes are appended after it, so
the primal outputs remain available and the model can be differentiated again.
"""
from ._build import (Context, UnsupportedOperator, assemble, conventions, rename,
                     seeded_value_info, select)
from .rules import FORWARD


def forward(model, inputs=None, outputs=None, prefix=None, dim=None, layout="casadi"):
    """The forward-derivative model of `model`.

    `inputs` names the graph inputs to seed (default: every floating-point one) and
    `outputs` the outputs to differentiate (default: every floating-point one). The result
    keeps the original signature as a prefix: after it come one seed input `<prefix><x>` per
    seeded input and one output `<prefix><y>` per differentiated output, so several
    directions ride in one evaluation.

    `prefix` and `dim` default to CasADi's own convention, derived from the names the model
    already carries: `fwd_`/`nfwd`, then `fwd2_`/`nfwd2` on a model that has been through a
    forward pass before. `layout` decides how the seeds are shaped -- see `reverse`.
    """
    prefix, dim = conventions(model, "fwd", prefix, dim)
    result = type(model)()
    result.CopyFrom(model)
    graph = result.graph
    ctx = Context(model)
    seed_inputs = []
    for value in select(graph.input, inputs, "input"):
        name = rename(ctx.b, prefix + value.name)
        seed_inputs.append(seeded_value_info(value, name, dim, layout))
        tangent = ctx.unpack(name, value.name) if layout == "casadi" else name
        ctx.derivative[value.name] = tangent
        ctx.add_seed(tangent)

    for node in graph.node:
        tangents = [ctx.derivative.get(name) if name else None for name in node.input]
        if all(tangent is None for tangent in tangents):
            continue  # nothing differentiated reaches this node; it is part of the primal
        if node.op_type not in FORWARD:
            raise UnsupportedOperator(
                "no forward rule for %s (node '%s'); an operation only needs one when a "
                "differentiated value reaches it"
                % (node.op_type, node.name or node.output[0]))
        produced = FORWARD[node.op_type](ctx, node, tangents)
        if produced is None or isinstance(produced, str):
            produced = [produced]
        for name, tangent in zip(node.output, produced):
            if name and tangent is not None:
                ctx.derivative[name] = tangent

    derivative_outputs = []
    for value in select(graph.output, outputs, "output"):
        tangent = ctx.derivative.get(value.name)
        seeded = ctx.zeros(value.name) if tangent is None else ctx.full(tangent, value.name)
        if layout == "casadi":
            seeded = ctx.pack(seeded, value.name)
        name = rename(ctx.b, prefix + value.name)
        ctx.b.alias(seeded, name)
        derivative_outputs.append(seeded_value_info(value, name, dim, layout))
    return assemble(result, ctx, seed_inputs, derivative_outputs)
