"""Reverse mode: emit a model computing `adj_x = J^T . adj_y`.

Two walks. The first, forward, marks the values that depend on the differentiated inputs --
nothing else can carry an adjoint, and a node touching none of them is skipped entirely.
The second walks the nodes backwards, handing each node the adjoints of its outputs and
accumulating what it returns onto its operands; a value with several consumers collects
several contributions and they are summed when the value's own node is reached.

The primal graph is kept, so a nonlinear rule reads the primal tensors it needs straight
from it rather than taking them as extra inputs. The emitted model is therefore a plain
function of `(x, adj_y)` -- no `uses_output` convention, and it can be differentiated again.
"""
from . import control  # noqa: F401 -- registers the If/Scan/Loop rules
from ._build import (Context, assemble, conventions, rename, seeded_value_info, select)
from .unroll import inline_constant_ifs
from ._passes import reverse_nodes


def reverse(model, inputs=None, outputs=None, prefix=None, dim=None, layout="casadi"):
    """The reverse-derivative (adjoint) model of `model`.

    `outputs` names the graph outputs to seed (default: every floating-point one) and
    `inputs` the inputs to differentiate (default: every floating-point one). The result
    keeps the original signature as a prefix: after it come one seed input `<prefix><y>` per
    seeded output and one output `<prefix><x>` per differentiated input.

    `prefix` and `dim` default to CasADi's convention (`adj_`/`nadj`, then `adj2_`/`nadj2`),
    derived from the names the model already carries.

    `layout` decides how a seed tensor is shaped. The default, `"casadi"`, presents it the
    way CasADi reads an ONNX tensor -- as a matrix whose column count is multiplied by the
    seed count, so the adjoint of a vector `x` of length `n` is an `n`-by-`nadj` matrix.
    `"onnx"` instead appends the seed count as a trailing axis of the primal's own shape,
    which is the internal layout and the natural one for a consumer that handles rank
    directly.
    """
    prefix, dim = conventions(model, "adj", prefix, dim)
    model = inline_constant_ifs(model)  # a constant condition needs no subgraph at all
    result = type(model)()
    result.CopyFrom(model)
    graph = result.graph
    ctx = Context(model)
    differentiated = select(graph.input, inputs, "input")

    seeds, seed_inputs = {}, []
    for value in select(graph.output, outputs, "output"):
        name = rename(ctx.b, prefix + value.name)
        seed_inputs.append(seeded_value_info(value, name, dim, layout))
        seed = ctx.unpack(name, value.name) if layout == "casadi" else name
        ctx.add_seed(seed)
        seeds[value.name] = seed
    # unpacking may read the shape of a primal output, so it goes after the primal graph
    unpacking = ctx.b.nodes
    ctx.b.nodes = []
    # reserve the output names now: the allocator is shared with every rule, and a rule
    # that took `adj_<x>` first would leave the convention nothing to name the output
    reserved = {value.name: rename(ctx.b, prefix + value.name) for value in differentiated}

    primal, adjoint, results = reverse_nodes(ctx, list(graph.node), seeds,
                                             [value.name for value in differentiated])

    derivative_outputs = []
    for value in differentiated:
        adjoint_ = results.get(value.name)
        seeded = ctx.zeros(value.name) if adjoint_ is None else ctx.full(adjoint_, value.name)
        if layout == "casadi":
            seeded = ctx.pack(seeded, value.name)
        name = reserved[value.name]
        ctx.b.alias(seeded, name)
        derivative_outputs.append(seeded_value_info(value, name, dim, layout))
    return assemble(result, ctx, primal + unpacking + adjoint + ctx.b.nodes, seed_inputs,
                    derivative_outputs)
