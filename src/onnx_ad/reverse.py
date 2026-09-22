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
from ._build import (Context, UnsupportedOperator, assemble, conventions, rename,
                     seeded_value_info, select)
from .rules import REVERSE


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
    result = type(model)()
    result.CopyFrom(model)
    graph = result.graph
    ctx = Context(model)
    differentiated = select(graph.input, inputs, "input")
    depends = {value.name for value in differentiated}
    for node in graph.node:
        if any(name in depends for name in node.input if name):
            depends.update(name for name in node.output if name)

    pending = {}

    def accumulate(name, contribution):
        pending.setdefault(name, []).append(contribution)

    def take(name):
        """Every contribution to `name` is in hand by the time its own node is reached."""
        return ctx.sum(pending.pop(name, []))

    seed_inputs = []
    for value in select(graph.output, outputs, "output"):
        name = rename(ctx.b, prefix + value.name)
        seed_inputs.append(seeded_value_info(value, name, dim, layout))
        seed = ctx.unpack(name, value.name) if layout == "casadi" else name
        ctx.add_seed(seed)
        accumulate(value.name, seed)

    for node in reversed(graph.node):
        if not any(name in depends for name in node.input if name):
            continue  # constant with respect to the differentiated inputs
        grads = [take(name) if name else None for name in node.output]
        if all(grad is None for grad in grads):
            continue  # this node's results do not reach a seeded output
        if node.op_type not in REVERSE:
            raise UnsupportedOperator(
                "no reverse rule for %s (node '%s'); an operation only needs one when it "
                "lies between a differentiated input and a seeded output"
                % (node.op_type, node.name or node.output[0]))
        ctx.wanted = {name for name in node.input if name and name in depends}
        contributions = REVERSE[node.op_type](ctx, node, grads)
        if isinstance(contributions, str):  # a rule must return one entry per operand
            raise TypeError("the reverse rule for %s returned a tensor, not a list"
                            % node.op_type)
        for name, contribution in zip(node.input, contributions):
            if name and contribution is not None and name in depends:
                accumulate(name, contribution)

    derivative_outputs = []
    for value in differentiated:
        adjoint = take(value.name)
        seeded = ctx.zeros(value.name) if adjoint is None else ctx.full(adjoint, value.name)
        if layout == "casadi":
            seeded = ctx.pack(seeded, value.name)
        name = rename(ctx.b, prefix + value.name)
        ctx.b.alias(seeded, name)
        derivative_outputs.append(seeded_value_info(value, name, dim, layout))
    return assemble(result, ctx, seed_inputs, derivative_outputs)
