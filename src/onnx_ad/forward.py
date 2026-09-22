"""Forward mode: emit a model computing `fwd_y = J . fwd_x`.

One walk over the graph in topological order, carrying a tangent per value. A value whose
tangent is absent is constant with respect to the seeds -- which is the whole reason the
emitted graph stays small, since every weight in a network is such a value and costs
nothing. The primal graph is kept verbatim and the tangent nodes are appended after it, so
the primal outputs remain available and the model can be differentiated again.
"""
from . import control  # noqa: F401 -- registers the If/Scan/Loop rules
from ._build import (Context, assemble, conventions, rename, seeded_value_info, select)
from .expand import expand_functions
from .unroll import inline_constant_ifs
from ._passes import forward_nodes


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
    model = expand_functions(model)     # spec-defined functions, and constant folding
    model = inline_constant_ifs(model)  # a constant condition needs no subgraph at all
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

    unpacking = ctx.b.nodes  # seed unpacking reads graph inputs only, so it can go first
    ctx.b.nodes = []
    # reserve the output names before any rule draws from the shared allocator
    chosen = select(graph.output, outputs, "output")
    reserved = {value.name: rename(ctx.b, prefix + value.name) for value in chosen}
    body = forward_nodes(ctx, list(graph.node))

    derivative_outputs = []
    for value in chosen:
        tangent = ctx.derivative.get(value.name)
        seeded = ctx.zeros(value.name) if tangent is None else ctx.full(tangent, value.name)
        if layout == "casadi":
            seeded = ctx.pack(seeded, value.name)
        name = reserved[value.name]
        ctx.b.alias(seeded, name)
        derivative_outputs.append(seeded_value_info(value, name, dim, layout))
    return assemble(result, ctx, unpacking + body + ctx.b.nodes, seed_inputs,
                    derivative_outputs)
