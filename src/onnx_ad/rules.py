"""Per-operation differentiation rules, forward and reverse, over the same table.

A rule sees the *primal* node exactly as it was authored -- the pass keeps the whole primal
graph -- so a nonlinear rule can read the node's own output instead of recomputing it:
the tangent of `Tanh` is `(1 - y*y) * t`, with `y` the tensor the primal `Tanh` already
produced. That is what keeps a derivative graph roughly the size of the primal one, and it
is why the reverse model stays a plain function of `(x, adj_y)` with no extra inputs.

Forward rules return the tangent of the node's output, shaped like the primal output with a
trailing seed axis (possibly under-broadcast, see `_build`). Reverse rules return one
adjoint contribution per operand, each already reduced to that operand's own shape.
"""
from onnx import TensorProto

from ._build import (FLOAT_TYPES, INT64_MAX, UnsupportedOperator, attribute)

FORWARD = {}
REVERSE = {}
#: Unary elementwise operations, as `d/dx` built from the primal input `x` and output `y`.
DERIVATIVE = {}


def forward_rule(*op_types):
    def register(fn):
        for op_type in op_types:
            FORWARD[op_type] = fn
        return fn
    return register


def reverse_rule(*op_types):
    def register(fn):
        for op_type in op_types:
            REVERSE[op_type] = fn
        return fn
    return register


def elementwise(*op_types):
    """Register a unary elementwise operation by its derivative; both modes follow."""
    def register(fn):
        for op_type in op_types:
            DERIVATIVE[op_type] = fn
            FORWARD[op_type] = _elementwise_forward
            REVERSE[op_type] = _elementwise_reverse
        return fn
    return register


def _elementwise_forward(ctx, node, tangents):
    d = DERIVATIVE[node.op_type](ctx, node, node.input[0], node.output[0],
                                 ctx.dtype(node.input[0], node.output[0]))
    return ctx.b.op("Mul", [tangents[0], ctx.lift(d)], stem="t_" + node.output[0])


def _elementwise_reverse(ctx, node, grads):
    d = DERIVATIVE[node.op_type](ctx, node, node.input[0], node.output[0],
                                 ctx.dtype(node.input[0], node.output[0]))
    return [ctx.b.op("Mul", [grads[0], ctx.lift(d)], stem="a_" + node.input[0])]


# --- unary elementwise -------------------------------------------------------------------

@elementwise("Exp")
def _d_exp(ctx, node, x, y, dtype):
    return y


@elementwise("Log")
def _d_log(ctx, node, x, y, dtype):
    return ctx.b.op("Reciprocal", [x])


@elementwise("Sqrt")
def _d_sqrt(ctx, node, x, y, dtype):
    return ctx.b.op("Div", [ctx.constant(0.5, dtype), y])


@elementwise("Reciprocal")
def _d_reciprocal(ctx, node, x, y, dtype):
    return ctx.b.op("Neg", [ctx.b.op("Mul", [y, y])])


@elementwise("Tanh")
def _d_tanh(ctx, node, x, y, dtype):
    return ctx.b.op("Sub", [ctx.constant(1.0, dtype), ctx.b.op("Mul", [y, y])])


@elementwise("Sigmoid")
def _d_sigmoid(ctx, node, x, y, dtype):
    return ctx.b.op("Mul", [y, ctx.b.op("Sub", [ctx.constant(1.0, dtype), y])])


@elementwise("Softplus")
def _d_softplus(ctx, node, x, y, dtype):
    return ctx.b.op("Sigmoid", [x])


@elementwise("Erf")
def _d_erf(ctx, node, x, y, dtype):
    square = ctx.b.op("Mul", [x, x])
    return ctx.b.op("Mul", [ctx.constant(1.1283791670955126, dtype),
                            ctx.b.op("Exp", [ctx.b.op("Neg", [square])])])


@elementwise("Sin")
def _d_sin(ctx, node, x, y, dtype):
    return ctx.b.op("Cos", [x])


@elementwise("Cos")
def _d_cos(ctx, node, x, y, dtype):
    return ctx.b.op("Neg", [ctx.b.op("Sin", [x])])


# Piecewise operations take the derivative of the branch the primal took; at the kink
# itself the convention below is the one every AD framework ships (Relu'(0) = 0).
@elementwise("Relu")
def _d_relu(ctx, node, x, y, dtype):
    positive = ctx.b.op("Greater", [x, ctx.constant(0.0, dtype)])
    return ctx.b.op("Cast", [positive], to=dtype)


@elementwise("LeakyRelu")
def _d_leaky_relu(ctx, node, x, y, dtype):
    alpha = attribute(node, "alpha", 0.01)
    positive = ctx.b.op("Greater", [x, ctx.constant(0.0, dtype)])
    return ctx.b.op("Where", [positive, ctx.constant(1.0, dtype), ctx.constant(alpha, dtype)])


@elementwise("Elu")
def _d_elu(ctx, node, x, y, dtype):
    alpha = attribute(node, "alpha", 1.0)
    positive = ctx.b.op("Greater", [x, ctx.constant(0.0, dtype)])
    # below zero y = alpha*(exp(x) - 1), so the derivative alpha*exp(x) is y + alpha
    return ctx.b.op("Where", [positive, ctx.constant(1.0, dtype),
                              ctx.b.op("Add", [y, ctx.constant(alpha, dtype)])])


@elementwise("Abs")
def _d_abs(ctx, node, x, y, dtype):
    return ctx.b.op("Sign", [x])


# --- linear operations without a factor ---------------------------------------------------

@forward_rule("Identity")
def _identity_forward(ctx, node, tangents):
    return tangents[0]


@reverse_rule("Identity")
def _identity_reverse(ctx, node, grads):
    return [grads[0]]


@forward_rule("Neg")
def _neg_forward(ctx, node, tangents):
    return ctx.b.op("Neg", [tangents[0]])


@reverse_rule("Neg")
def _neg_reverse(ctx, node, grads):
    return [ctx.b.op("Neg", [grads[0]])]


# --- binary elementwise -------------------------------------------------------------------

@forward_rule("Add", "Sub")
def _add_forward(ctx, node, tangents):
    ta, tb = tangents
    if tb is None:
        return ta
    if ta is None:
        return ctx.b.op("Neg", [tb]) if node.op_type == "Sub" else tb
    return ctx.b.op(node.op_type, [ta, tb])


@reverse_rule("Add", "Sub")
def _add_reverse(ctx, node, grads):
    g, rank = grads[0], ctx.rank(node.output[0])
    second = ctx.b.op("Neg", [g]) if node.op_type == "Sub" else g
    return [ctx.unbroadcast(g, rank, node.input[0]),
            ctx.unbroadcast(second, rank, node.input[1])]


@forward_rule("Mul")
def _mul_forward(ctx, node, tangents):
    a, b = node.input[0], node.input[1]
    terms = []
    if tangents[0] is not None:
        terms.append(ctx.b.op("Mul", [tangents[0], ctx.lift(b)]))
    if tangents[1] is not None:
        terms.append(ctx.b.op("Mul", [ctx.lift(a), tangents[1]]))
    return ctx.sum(terms)


@reverse_rule("Mul")
def _mul_reverse(ctx, node, grads):
    a, b = node.input[0], node.input[1]
    g, rank = grads[0], ctx.rank(node.output[0])
    return [ctx.unbroadcast(ctx.b.op("Mul", [g, ctx.lift(b)]), rank, a),
            ctx.unbroadcast(ctx.b.op("Mul", [g, ctx.lift(a)]), rank, b)]


@forward_rule("Div")
def _div_forward(ctx, node, tangents):
    b, y = node.input[1], node.output[0]
    # d(a/b) = (da - (a/b) db)/b, written through the primal quotient y = a/b
    numerator = tangents[0]
    if tangents[1] is not None:
        term = ctx.b.op("Mul", [ctx.lift(y), tangents[1]])
        numerator = ctx.b.op("Neg", [term]) if numerator is None else \
            ctx.b.op("Sub", [numerator, term])
    return ctx.b.op("Div", [numerator, ctx.lift(b)])


@reverse_rule("Div")
def _div_reverse(ctx, node, grads):
    a, b, y = node.input[0], node.input[1], node.output[0]
    g, rank = grads[0], ctx.rank(node.output[0])
    over_b = ctx.b.op("Div", [g, ctx.lift(b)])
    return [ctx.unbroadcast(over_b, rank, a),
            ctx.unbroadcast(ctx.b.op("Neg", [ctx.b.op("Mul", [over_b, ctx.lift(y)])]), rank, b)]


@forward_rule("Pow")
def _pow_forward(ctx, node, tangents):
    terms = []
    if tangents[0] is not None:
        terms.append(ctx.b.op("Mul", [tangents[0], ctx.lift(_pow_d_base(ctx, node))]))
    if tangents[1] is not None:
        terms.append(ctx.b.op("Mul", [tangents[1], ctx.lift(_pow_d_exponent(ctx, node))]))
    return ctx.sum(terms)


@reverse_rule("Pow")
def _pow_reverse(ctx, node, grads):
    g, rank = grads[0], ctx.rank(node.output[0])
    return [ctx.unbroadcast(ctx.b.op("Mul", [g, ctx.lift(_pow_d_base(ctx, node))]),
                            rank, node.input[0]),
            ctx.unbroadcast(ctx.b.op("Mul", [g, ctx.lift(_pow_d_exponent(ctx, node))]),
                            rank, node.input[1])]


def _pow_d_base(ctx, node):
    """b*a^(b-1) -- not b*y/a, which would be undefined at a = 0 for b > 1."""
    a, b = node.input[0], node.input[1]
    dtype = ctx.dtype(node.output[0], a)
    exponent = b if ctx.shapes.dtype(b) == dtype else ctx.b.op("Cast", [b], to=dtype)
    lowered = ctx.b.op("Sub", [exponent, ctx.constant(1.0, dtype)])
    return ctx.b.op("Mul", [exponent, ctx.b.op("Pow", [a, lowered])])


def _pow_d_exponent(ctx, node):
    """y*log(a); the exponent must be of the same floating type for this to mean anything."""
    a, y = node.input[0], node.output[0]
    if ctx.dtype(node.input[1]) != ctx.dtype(a):
        raise UnsupportedOperator(
            "Pow with a differentiated integer exponent has no derivative in that operand")
    return ctx.b.op("Mul", [y, ctx.b.op("Log", [a])])


# --- MatMul -------------------------------------------------------------------------------
# The seed axis is trailing, but MatMul contracts the last two axes, so a seeded operand is
# rotated to put the seed axis first (where MatMul treats it as one more batch dimension)
# and rotated back afterwards. Rank-1 operands are promoted to matrices first, exactly as
# numpy.matmul does internally, and the promoted axis is dropped from the result.

def _seed_front(ctx, value, rank):
    return ctx.b.op("Transpose", [value], perm=[rank] + list(range(rank)), stem="seed_first")


def _seed_back(ctx, value, rank):
    return ctx.b.op("Transpose", [value], perm=list(range(1, rank + 1)) + [0], stem="seed_last")


def _pad_batch(ctx, value, rank, target_rank):
    """Insert singleton batch axes after the leading seed axis, up to the result's rank."""
    missing = target_rank - rank
    return value if missing <= 0 else ctx.unsqueeze(value, list(range(1, missing + 1)))


def _promote(ctx, value, rank, axis):
    """A rank-1 operand as a matrix: `axis` is 0 for the left operand, 1 for the right."""
    if rank >= 2:
        return value, rank
    promoted = ctx.unsqueeze(value, [axis])
    shape = list(ctx.shapes.shape(value))
    shape.insert(axis, 1)
    ctx.shapes.declare(promoted, shape, ctx.shapes.dtype(value))
    return promoted, 2


def _matmul_shapes(ctx, node):
    """Both operands as matrices, with the rank the result has before the promoted axes go."""
    a, b = node.input[0], node.input[1]
    if ctx.rank(a) == 0 or ctx.rank(b) == 0:
        raise UnsupportedOperator("MatMul needs operands of rank 1 or more")
    a2, ra2 = _promote(ctx, a, ctx.rank(a), 0)
    b2, rb2 = _promote(ctx, b, ctx.rank(b), 1)
    return a2, ra2, b2, rb2, max(ra2, rb2)


@forward_rule("MatMul")
def _matmul_forward(ctx, node, tangents):
    a, b, z = node.input[0], node.input[1], node.output[0]
    a2, ra2, b2, rb2, rz2 = _matmul_shapes(ctx, node)
    terms = []
    if tangents[0] is not None:
        seeded = ctx.full(tangents[0], a)
        if ra2 > ctx.rank(a):
            seeded = ctx.unsqueeze(seeded, [0])
        rotated = _pad_batch(ctx, _seed_front(ctx, seeded, ra2), ra2, rz2)
        terms.append(_seed_back(ctx, ctx.b.op("MatMul", [rotated, b2]), rz2))
    if tangents[1] is not None:
        seeded = ctx.full(tangents[1], b)
        if rb2 > ctx.rank(b):
            seeded = ctx.unsqueeze(seeded, [1])
        rotated = _pad_batch(ctx, _seed_front(ctx, seeded, rb2), rb2, rz2)
        terms.append(_seed_back(ctx, ctx.b.op("MatMul", [a2, rotated]), rz2))
    total = ctx.sum(terms)
    return total if rz2 == ctx.rank(z) else ctx.reshape_like(total, z)


@reverse_rule("MatMul")
def _matmul_reverse(ctx, node, grads):
    a, b, z = node.input[0], node.input[1], node.output[0]
    ra, rb = ctx.rank(a), ctx.rank(b)
    a2, ra2, b2, rb2, rz2 = _matmul_shapes(ctx, node)
    seeded = ctx.full(grads[0], z)
    axes = ([rz2 - 1] if rb == 1 else []) + ([rz2 - 2] if ra == 1 else [])
    if axes:  # put back the axes numpy.matmul dropped, so the result is shaped like z2
        seeded = ctx.unsqueeze(seeded, sorted(axes))
    front = _seed_front(ctx, seeded, rz2)

    def transposed(value, rank):  # the matrix transpose, leaving batch axes alone
        return ctx.b.op("Transpose", [value],
                        perm=list(range(rank - 2)) + [rank - 1, rank - 2])

    left = ctx.b.op("MatMul", [front, transposed(b2, rb2)])
    right = ctx.b.op("MatMul", [transposed(a2, ra2), front])
    contributions = []
    for value, promoted, original in ((left, a2, a), (right, b2, b)):
        reduced = ctx.unbroadcast(_seed_back(ctx, value, rz2), rz2, promoted)
        contributions.append(reduced if promoted == original
                             else ctx.reshape_like(reduced, original))
    return contributions


# --- structural operations ----------------------------------------------------------------
# These move or combine values without changing them, so both modes are the same operation
# applied to the derivative tensor -- the only care needed is the trailing seed axis: axes
# and permutations counted from the front are unaffected, and ones counted from the back are
# normalized to the front first. A structural rule needs its operand's derivative at the
# operand's exact shape, so it calls `full` rather than accepting an under-broadcast tensor.

def _axes(ctx, node, index, rank, name="axes"):
    """The axes/perm operand as non-negative positions in a tensor of the given rank."""
    if len(node.input) > index and node.input[index]:
        values = ctx.integers(node.input[index])
        if values is None:
            raise UnsupportedOperator(
                "%s is differentiable only with a constant '%s' operand" % (node.op_type, name))
    else:
        values = attribute(node, name)
    return None if values is None else [axis % rank for axis in values]


@forward_rule("Reshape", "Flatten")
def _reshape_forward(ctx, node, tangents):
    # the seed axis rides along as the trailing -1: the element count is a multiple of it
    return ctx.reshape_like(ctx.full(tangents[0], node.input[0]), node.output[0])


@reverse_rule("Reshape", "Flatten")
def _reshape_reverse(ctx, node, grads):
    return [ctx.reshape_like(ctx.full(grads[0], node.output[0]), node.input[0])]


def _permutation(ctx, node):
    rank = ctx.rank(node.input[0])
    perm = attribute(node, "perm")
    return list(reversed(range(rank))) if perm is None else list(perm), rank


@forward_rule("Transpose")
def _transpose_forward(ctx, node, tangents):
    perm, rank = _permutation(ctx, node)
    return ctx.b.op("Transpose", [ctx.full(tangents[0], node.input[0])], perm=perm + [rank])


@reverse_rule("Transpose")
def _transpose_reverse(ctx, node, grads):
    perm, rank = _permutation(ctx, node)
    inverse = [0]*rank
    for position, axis in enumerate(perm):
        inverse[axis] = position
    return [ctx.b.op("Transpose", [ctx.full(grads[0], node.output[0])],
                     perm=inverse + [rank])]


@forward_rule("Unsqueeze")
def _unsqueeze_forward(ctx, node, tangents):
    axes = _axes(ctx, node, 1, ctx.rank(node.output[0]))
    return ctx.unsqueeze(ctx.full(tangents[0], node.input[0]), axes)


@reverse_rule("Unsqueeze")
def _unsqueeze_reverse(ctx, node, grads):
    axes = _axes(ctx, node, 1, ctx.rank(node.output[0]))
    return [ctx.squeeze(ctx.full(grads[0], node.output[0]), axes)]


def _squeezed_axes(ctx, node):
    rank = ctx.rank(node.input[0])
    axes = _axes(ctx, node, 1, rank)
    if axes is None:  # no axes given: every dimension of size one goes
        shape = ctx.shapes.static(node.input[0])
        if shape is None:
            raise UnsupportedOperator(
                "Squeeze without an axes operand needs a declared shape to be differentiated")
        axes = [i for i, d in enumerate(shape) if d == 1]
    return axes


@forward_rule("Squeeze")
def _squeeze_forward(ctx, node, tangents):
    return ctx.squeeze(ctx.full(tangents[0], node.input[0]), _squeezed_axes(ctx, node))


@reverse_rule("Squeeze")
def _squeeze_reverse(ctx, node, grads):
    return [ctx.unsqueeze(ctx.full(grads[0], node.output[0]), _squeezed_axes(ctx, node))]


@forward_rule("Expand")
def _expand_forward(ctx, node, tangents):
    # an under-broadcast tangent of the operand is already one of the result: Expand only
    # widens dimensions, and the seed axis stays trailing through the widening
    return tangents[0]


@reverse_rule("Expand")
def _expand_reverse(ctx, node, grads):
    return [ctx.unbroadcast(ctx.full(grads[0], node.output[0]),
                            ctx.rank(node.output[0]), node.input[0]), None]


@forward_rule("Sum", "Mean")
def _sum_forward(ctx, node, tangents):
    total = ctx.sum([t for t in tangents if t is not None])
    if node.op_type == "Mean":
        count = ctx.constant(1.0/len(node.input), ctx.dtype(node.output[0]))
        return ctx.b.op("Mul", [total, count])
    return total


@reverse_rule("Sum", "Mean")
def _sum_reverse(ctx, node, grads):
    g, rank = grads[0], ctx.rank(node.output[0])
    if node.op_type == "Mean":
        count = ctx.constant(1.0/len(node.input), ctx.dtype(node.output[0]))
        g = ctx.b.op("Mul", [g, count])
    return [ctx.unbroadcast(g, rank, name) for name in node.input]


def _concat_axis(ctx, node):
    return attribute(node, "axis") % ctx.rank(node.output[0])


@forward_rule("Concat")
def _concat_forward(ctx, node, tangents):
    pieces = [ctx.zeros(name) if tangent is None else ctx.full(tangent, name)
              for name, tangent in zip(node.input, tangents)]
    return ctx.b.op("Concat", pieces, axis=_concat_axis(ctx, node))


@reverse_rule("Concat")
def _concat_reverse(ctx, node, grads):
    axis = _concat_axis(ctx, node)
    seeded = ctx.full(grads[0], node.output[0])
    contributions, offset = [], 0
    for name in node.input:
        shape = ctx.shapes.shape(name)
        if shape is None or shape[axis] is None:
            raise UnsupportedOperator(
                "Concat is differentiable in reverse only where the concatenated extent of "
                "each operand is declared; '%s' has none" % name)
        contributions.append(ctx.b.op("Slice", [
            seeded, ctx.b.ints([offset]), ctx.b.ints([offset + shape[axis]]),
            ctx.b.ints([axis])]))
        offset += shape[axis]
    return contributions


def _reduction_axes(ctx, node):
    rank = ctx.rank(node.input[0])
    axes = _axes(ctx, node, 1, rank)
    if axes is None and not attribute(node, "noop_with_empty_axes", 0):
        axes = list(range(rank))
    return axes or [], int(attribute(node, "keepdims", 1))


@forward_rule("ReduceSum", "ReduceMean")
def _reduce_forward(ctx, node, tangents):
    axes, keepdims = _reduction_axes(ctx, node)
    seeded = ctx.full(tangents[0], node.input[0])
    if not axes:
        return seeded
    if node.op_type == "ReduceSum":
        return ctx.reduce_sum(seeded, axes, keepdims=keepdims)
    return ctx.b.op("ReduceMean", [seeded], axes=axes, keepdims=keepdims) \
        if ctx.opset < 18 else \
        ctx.b.op("ReduceMean", [seeded, ctx.b.ints(axes)], keepdims=keepdims)


@reverse_rule("ReduceSum", "ReduceMean")
def _reduce_reverse(ctx, node, grads):
    axes, keepdims = _reduction_axes(ctx, node)
    x, y = node.input[0], node.output[0]
    seeded = ctx.full(grads[0], y)
    if axes and not keepdims:  # put the reduced axes back before broadcasting over them
        seeded = ctx.unsqueeze(seeded, axes)
    contribution = ctx.full(seeded, x)
    if node.op_type == "ReduceMean" and axes:
        dtype = ctx.dtype(x, y)
        count = ctx.b.op("Cast", [ctx.b.op("Div", [ctx.b.op("Size", [x]),
                                                   ctx.b.op("Size", [y])])], to=dtype)
        contribution = ctx.b.op("Div", [contribution, count])
    return [contribution]


# --- Gemm ---------------------------------------------------------------------------------
# Y = alpha . A' . B' + beta . C, with A' and B' optionally transposed. This is what
# torch.nn.Linear exports to, so it is the one operation no network can do without. The
# operands are always matrices, which makes the seed rotation simpler than MatMul's.

def _gemm_transpose(ctx, value):
    """Swap the two primal axes of a seeded rank-2 tensor, leaving the seed axis last."""
    return ctx.b.op("Transpose", [value], perm=[1, 0, 2])


def _scaled(ctx, value, factor, dtype):
    return value if factor == 1.0 else \
        ctx.b.op("Mul", [value, ctx.constant(factor, dtype)])


def _gemm_operands(ctx, node):
    a, b = node.input[0], node.input[1]
    left = ctx.b.op("Transpose", [a], perm=[1, 0]) if attribute(node, "transA", 0) else a
    right = ctx.b.op("Transpose", [b], perm=[1, 0]) if attribute(node, "transB", 0) else b
    return left, right


@forward_rule("Gemm")
def _gemm_forward(ctx, node, tangents):
    y = node.output[0]
    dtype = ctx.dtype(y, *node.input)
    alpha, beta = attribute(node, "alpha", 1.0), attribute(node, "beta", 1.0)
    left, right = _gemm_operands(ctx, node)
    terms = []
    for index, other, on_left in ((0, right, True), (1, left, False)):
        if tangents[index] is None:
            continue
        seeded = ctx.full(tangents[index], node.input[index])
        if attribute(node, "transA" if index == 0 else "transB", 0):
            seeded = _gemm_transpose(ctx, seeded)
        front = _seed_front(ctx, seeded, 2)
        product = ctx.b.op("MatMul", [front, other] if on_left else [other, front])
        terms.append(_scaled(ctx, _seed_back(ctx, product, 2), alpha, dtype))
    if len(tangents) > 2 and tangents[2] is not None:
        terms.append(_scaled(ctx, tangents[2], beta, dtype))
    return ctx.sum(terms)


@reverse_rule("Gemm")
def _gemm_reverse(ctx, node, grads):
    y = node.output[0]
    dtype = ctx.dtype(y, *node.input)
    alpha, beta = attribute(node, "alpha", 1.0), attribute(node, "beta", 1.0)
    left, right = _gemm_operands(ctx, node)
    front = _seed_front(ctx, ctx.full(grads[0], y), 2)
    contributions = []
    for index, other, on_left in ((0, right, True), (1, left, False)):
        operands = [front, ctx.b.op("Transpose", [other], perm=[1, 0])] if on_left else \
            [ctx.b.op("Transpose", [other], perm=[1, 0]), front]
        value = _scaled(ctx, _seed_back(ctx, ctx.b.op("MatMul", operands), 2), alpha, dtype)
        if attribute(node, "transA" if index == 0 else "transB", 0):
            value = _gemm_transpose(ctx, value)
        contributions.append(value)
    if len(node.input) > 2 and node.input[2]:
        contributions.append(_scaled(ctx, ctx.unbroadcast(grads[0], 2, node.input[2]),
                                     beta, dtype))
    return contributions


# --- operations a derivative does not flow through ----------------------------------------
# Shape queries, comparisons, index-producing and piecewise-constant operations all have a
# zero derivative. They need rules all the same: without one, a differentiated value merely
# *reaching* such a node -- a Relu written as Where(Greater(x, 0), x, 0), say -- would be
# rejected. Absent is how this table spells zero, so the rules produce nothing.

ZERO_DERIVATIVE = (
    "Shape", "Size", "NonZero", "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
    "And", "Or", "Xor", "Not", "BitwiseAnd", "BitwiseOr", "BitwiseXor", "BitwiseNot",
    "ArgMax", "ArgMin", "IsNaN", "IsInf", "IsFinite", "Sign", "Floor", "Ceil", "Round",
    "Hardmax", "OneHot", "Bernoulli", "Multinomial", "Det",
    # outputs that depend on an operand's shape or type but not its values
    "ConstantOfShape", "EyeLike", "RandomNormalLike", "RandomUniformLike",
    # integer or boolean results, or a reinterpretation of the bits
    "NonMaxSuppression", "QuantizeLinear", "BitCast",
)


@forward_rule(*ZERO_DERIVATIVE)
def _zero_forward(ctx, node, tangents):
    return [None]*len(node.output)


@reverse_rule(*ZERO_DERIVATIVE)
def _zero_reverse(ctx, node, grads):
    return [None]*len(node.input)


def _cast_target(ctx, node):
    if node.op_type == "CastLike":
        return ctx.shapes.dtype(node.input[1])
    return attribute(node, "to")


@forward_rule("Cast", "CastLike")
def _cast_forward(ctx, node, tangents):
    # float to float carries the tangent across; anything through an integer type loses it.
    # CastLike's second operand only names a type: a tangent reaching it -- `CastLike(2.0, x)`
    # with x differentiated, which is how exporters type a scalar -- carries nothing through.
    if tangents[0] is None:
        return None
    target = _cast_target(ctx, node)
    if target not in FLOAT_TYPES or ctx.shapes.dtype(node.input[0]) not in FLOAT_TYPES:
        return None
    return ctx.b.op("Cast", [tangents[0]], to=target)


@reverse_rule("Cast", "CastLike")
def _cast_reverse(ctx, node, grads):
    source = ctx.shapes.dtype(node.input[0])
    rest = [None]*(len(node.input) - 1)
    if not ctx.asked_for(node.input[0]) or _cast_target(ctx, node) not in FLOAT_TYPES \
            or source not in FLOAT_TYPES:
        return [None] + rest
    return [ctx.b.op("Cast", [grads[0]], to=source)] + rest


# Inference-mode operations that are the identity on their data operand
@forward_rule("Dropout")
def _dropout_forward(ctx, node, tangents):
    _reject_training(ctx, node)
    return [tangents[0], None]


@reverse_rule("Dropout")
def _dropout_reverse(ctx, node, grads):
    _reject_training(ctx, node)
    return [grads[0]] + [None]*(len(node.input) - 1)


def _reject_training(ctx, node):
    """Dropout is the identity unless training_mode is on, which is out of scope."""
    if len(node.input) > 2 and node.input[2]:
        value = ctx.values.get(node.input[2])
        if value is None or bool(value.reshape(-1)[0]):
            raise UnsupportedOperator(
                "Dropout with training_mode is not differentiated; export in inference mode")


# --- the rest of the elementwise family ---------------------------------------------------
# Each entry is d/dx, written through the primal input `x` and output `y` wherever the
# output already carries the expensive part -- `Tan` is `1 + y*y`, not `1/cos(x)^2`.

@elementwise("Tan")
def _d_tan(ctx, node, x, y, dtype):
    return ctx.b.op("Add", [ctx.constant(1.0, dtype), ctx.b.op("Mul", [y, y])])


@elementwise("Sinh")
def _d_sinh(ctx, node, x, y, dtype):
    return ctx.b.op("Cosh", [x])


@elementwise("Cosh")
def _d_cosh(ctx, node, x, y, dtype):
    return ctx.b.op("Sinh", [x])


def _inverse_root(ctx, x, offset, dtype, negate=False):
    """1/sqrt(offset + sign . x^2), the derivative shared by the inverse trigonometrics."""
    square = ctx.b.op("Mul", [x, x])
    one = ctx.constant(offset, dtype)
    inner = ctx.b.op("Sub", [one, square]) if not negate else ctx.b.op("Add", [square, one])
    return ctx.b.op("Reciprocal", [ctx.b.op("Sqrt", [inner])])


@elementwise("Asin")
def _d_asin(ctx, node, x, y, dtype):
    return _inverse_root(ctx, x, 1.0, dtype)


@elementwise("Acos")
def _d_acos(ctx, node, x, y, dtype):
    return ctx.b.op("Neg", [_inverse_root(ctx, x, 1.0, dtype)])


@elementwise("Atan")
def _d_atan(ctx, node, x, y, dtype):
    return ctx.b.op("Reciprocal", [ctx.b.op("Add", [ctx.constant(1.0, dtype),
                                                    ctx.b.op("Mul", [x, x])])])


@elementwise("Asinh")
def _d_asinh(ctx, node, x, y, dtype):
    return _inverse_root(ctx, x, 1.0, dtype, negate=True)


@elementwise("Acosh")
def _d_acosh(ctx, node, x, y, dtype):
    return _inverse_root(ctx, x, -1.0, dtype, negate=True)


@elementwise("Atanh")
def _d_atanh(ctx, node, x, y, dtype):
    return ctx.b.op("Reciprocal", [ctx.b.op("Sub", [ctx.constant(1.0, dtype),
                                                    ctx.b.op("Mul", [x, x])])])


@elementwise("Selu")
def _d_selu(ctx, node, x, y, dtype):
    # the spec's defaults are these exact float32 values, not the mathematical constants
    alpha = attribute(node, "alpha", 1.67326319217681884765625)
    gamma = attribute(node, "gamma", 1.05070102214813232421875)
    positive = ctx.b.op("Greater", [x, ctx.constant(0.0, dtype)])
    # below zero y = gamma.alpha.(exp(x) - 1), so gamma.alpha.exp(x) is y + gamma.alpha
    return ctx.b.op("Where", [positive, ctx.constant(gamma, dtype),
                              ctx.b.op("Add", [y, ctx.constant(gamma*alpha, dtype)])])


@elementwise("Celu")
def _d_celu(ctx, node, x, y, dtype):
    alpha = attribute(node, "alpha", 1.0)
    positive = ctx.b.op("Greater", [x, ctx.constant(0.0, dtype)])
    # below zero y = alpha.(exp(x/alpha) - 1), so exp(x/alpha) is y/alpha + 1
    negative = ctx.b.op("Add", [ctx.b.op("Div", [y, ctx.constant(alpha, dtype)]),
                                ctx.constant(1.0, dtype)])
    return ctx.b.op("Where", [positive, ctx.constant(1.0, dtype), negative])


@elementwise("ThresholdedRelu")
def _d_thresholded_relu(ctx, node, x, y, dtype):
    alpha = attribute(node, "alpha", 1.0)
    return ctx.b.op("Cast", [ctx.b.op("Greater", [x, ctx.constant(alpha, dtype)])], to=dtype)


@elementwise("Shrink")
def _d_shrink(ctx, node, x, y, dtype):
    lambd = attribute(node, "lambd", 0.5)
    outside = ctx.b.op("Greater", [ctx.b.op("Abs", [x]), ctx.constant(lambd, dtype)])
    return ctx.b.op("Cast", [outside], to=dtype)


@elementwise("Softsign")
def _d_softsign(ctx, node, x, y, dtype):
    scale = ctx.b.op("Add", [ctx.constant(1.0, dtype), ctx.b.op("Abs", [x])])
    return ctx.b.op("Reciprocal", [ctx.b.op("Mul", [scale, scale])])


def _hard_sigmoid(ctx, x, alpha, beta, dtype):
    """The clipped line alpha.x + beta, and the mask of where it is not clipped."""
    line = ctx.b.op("Add", [ctx.b.op("Mul", [x, ctx.constant(alpha, dtype)]),
                            ctx.constant(beta, dtype)])
    zero, one = ctx.constant(0.0, dtype), ctx.constant(1.0, dtype)
    inside = ctx.b.op("And", [ctx.b.op("Greater", [line, zero]),
                              ctx.b.op("Less", [line, one])])
    value = ctx.b.op("Min", [ctx.b.op("Max", [line, zero]), one])
    return value, ctx.b.op("Where", [inside, ctx.constant(alpha, dtype), zero])


@elementwise("HardSigmoid")
def _d_hard_sigmoid(ctx, node, x, y, dtype):
    return _hard_sigmoid(ctx, x, attribute(node, "alpha", 0.2),
                         attribute(node, "beta", 0.5), dtype)[1]


@elementwise("HardSwish")
def _d_hard_swish(ctx, node, x, y, dtype):
    gate, slope = _hard_sigmoid(ctx, x, 1.0/6, 0.5, dtype)
    return ctx.b.op("Add", [gate, ctx.b.op("Mul", [x, slope])])


@elementwise("Mish")
def _d_mish(ctx, node, x, y, dtype):
    # y = x.tanh(softplus(x));  y' = tanh(sp) + x.(1 - tanh(sp)^2).sigmoid(x)
    tanh = ctx.b.op("Tanh", [ctx.b.op("Softplus", [x])])
    gain = ctx.b.op("Sub", [ctx.constant(1.0, dtype), ctx.b.op("Mul", [tanh, tanh])])
    return ctx.b.op("Add", [tanh, ctx.b.op("Mul", [ctx.b.op("Mul", [x, gain]),
                                                   ctx.b.op("Sigmoid", [x])])])


@elementwise("Gelu")
def _d_gelu(ctx, node, x, y, dtype):
    half, one = ctx.constant(0.5, dtype), ctx.constant(1.0, dtype)
    if attribute(node, "approximate", "none") == "tanh":
        # y = 0.5x(1 + tanh(c(x + a x^3))), c = sqrt(2/pi), a = 0.044715
        c, a = 0.7978845608028654, 0.044715
        cube = ctx.b.op("Mul", [ctx.b.op("Mul", [x, x]), x])
        inner = ctx.b.op("Mul", [ctx.b.op("Add", [x, ctx.b.op("Mul", [
            cube, ctx.constant(a, dtype)])]), ctx.constant(c, dtype)])
        tanh = ctx.b.op("Tanh", [inner])
        gain = ctx.b.op("Sub", [one, ctx.b.op("Mul", [tanh, tanh])])
        # d inner/dx = c(1 + 3a x^2)
        slope = ctx.b.op("Mul", [ctx.b.op("Add", [one, ctx.b.op("Mul", [
            ctx.b.op("Mul", [x, x]), ctx.constant(3*a, dtype)])]), ctx.constant(c, dtype)])
        return ctx.b.op("Mul", [half, ctx.b.op("Add", [
            ctx.b.op("Add", [one, tanh]),
            ctx.b.op("Mul", [ctx.b.op("Mul", [x, gain]), slope])])])
    # exact: y = x.Phi(x);  y' = Phi(x) + x.phi(x)
    cumulative = ctx.b.op("Mul", [half, ctx.b.op("Add", [one, ctx.b.op("Erf", [
        ctx.b.op("Mul", [x, ctx.constant(0.7071067811865476, dtype)])])])])
    density = ctx.b.op("Mul", [ctx.constant(0.3989422804014327, dtype), ctx.b.op("Exp", [
        ctx.b.op("Mul", [ctx.b.op("Mul", [x, x]), ctx.constant(-0.5, dtype)])])])
    return ctx.b.op("Add", [cumulative, ctx.b.op("Mul", [x, density])])


# --- selection -----------------------------------------------------------------------------
# The derivative follows the branch the primal took. At a tie or a kink the convention below
# is the usual one: the first operand wins a Min/Max tie, and a value sitting exactly on a
# Clip bound is treated as clipped.

def _mask(ctx, condition):
    """A primal-shaped boolean, lifted so it selects across the seed axis."""
    return ctx.lift(condition)


@forward_rule("Where")
def _where_forward(ctx, node, tangents):
    dtype = ctx.dtype(node.output[0], node.input[1], node.input[2])
    zero = ctx.constant(0.0, dtype)
    branches = [zero if t is None else t for t in tangents[1:3]]
    return ctx.b.op("Where", [_mask(ctx, node.input[0])] + branches)


@reverse_rule("Where")
def _where_reverse(ctx, node, grads):
    dtype = ctx.dtype(node.output[0], node.input[1], node.input[2])
    zero, rank = ctx.constant(0.0, dtype), ctx.rank(node.output[0])
    condition = _mask(ctx, node.input[0])
    return [None,
            ctx.unbroadcast(ctx.b.op("Where", [condition, grads[0], zero]), rank,
                            node.input[1]),
            ctx.unbroadcast(ctx.b.op("Where", [condition, zero, grads[0]]), rank,
                            node.input[2])]


def _clip_masks(ctx, node):
    """(inside, below, above) as primal-shaped booleans; below/above are None when absent."""
    x = node.input[0]
    below = above = None
    inside = None
    for index, comparison in ((1, "Less"), (2, "Greater")):
        if len(node.input) <= index or not node.input[index]:
            continue
        outside = ctx.b.op(comparison, [x, node.input[index]])
        if index == 1:
            below = outside
        else:
            above = outside
        kept = ctx.b.op("Not", [outside])
        inside = kept if inside is None else ctx.b.op("And", [inside, kept])
    return inside, below, above


@forward_rule("Clip")
def _clip_forward(ctx, node, tangents):
    dtype = ctx.dtype(node.output[0], node.input[0])
    zero = ctx.constant(0.0, dtype)
    inside, below, above = _clip_masks(ctx, node)
    terms = []
    if tangents[0] is not None:
        terms.append(tangents[0] if inside is None else
                     ctx.b.op("Where", [_mask(ctx, inside), tangents[0], zero]))
    for tangent, mask in ((tangents[1] if len(tangents) > 1 else None, below),
                          (tangents[2] if len(tangents) > 2 else None, above)):
        if tangent is not None and mask is not None:
            terms.append(ctx.b.op("Where", [_mask(ctx, mask), tangent, zero]))
    return ctx.sum(terms)


@reverse_rule("Clip")
def _clip_reverse(ctx, node, grads):
    dtype = ctx.dtype(node.output[0], node.input[0])
    zero, rank = ctx.constant(0.0, dtype), ctx.rank(node.output[0])
    inside, below, above = _clip_masks(ctx, node)
    contributions = [grads[0] if inside is None else
                     ctx.b.op("Where", [_mask(ctx, inside), grads[0], zero])]
    for index, mask in ((1, below), (2, above)):
        if len(node.input) <= index or not node.input[index] or mask is None:
            contributions.append(None)
            continue
        contributions.append(ctx.unbroadcast(
            ctx.b.op("Where", [_mask(ctx, mask), grads[0], zero]), rank, node.input[index]))
    return contributions


def _extremum_fold(ctx, node):
    """Fold the operands pairwise, keeping the mask that says the running value won."""
    comparison = "GreaterOrEqual" if node.op_type == "Max" else "LessOrEqual"
    running, masks = node.input[0], []
    for other in node.input[1:]:
        mask = ctx.b.op(comparison, [running, other])
        masks.append(mask)
        running = ctx.b.op("Where", [mask, running, other])
    return masks


@forward_rule("Max", "Min")
def _extremum_forward(ctx, node, tangents):
    dtype = ctx.dtype(node.output[0], *node.input)
    zero = ctx.constant(0.0, dtype)
    masks = _extremum_fold(ctx, node)
    running = tangents[0]
    for mask, other in zip(masks, tangents[1:]):
        if running is None and other is None:
            continue
        running = ctx.b.op("Where", [_mask(ctx, mask),
                                     zero if running is None else running,
                                     zero if other is None else other])
    return running


@reverse_rule("Max", "Min")
def _extremum_reverse(ctx, node, grads):
    dtype = ctx.dtype(node.output[0], *node.input)
    zero, rank = ctx.constant(0.0, dtype), ctx.rank(node.output[0])
    masks = _extremum_fold(ctx, node)
    contributions = [None]*len(node.input)
    running = grads[0]
    for index in range(len(masks) - 1, -1, -1):
        mask = _mask(ctx, masks[index])
        contributions[index + 1] = ctx.unbroadcast(
            ctx.b.op("Where", [mask, zero, running]), rank, node.input[index + 1])
        running = ctx.b.op("Where", [mask, running, zero])
    contributions[0] = ctx.unbroadcast(running, rank, node.input[0])
    return contributions


def _prelu_parts(ctx, node):
    """d y/d x and d y/d slope, both of the result's shape."""
    x, slope = node.input[0], node.input[1]
    dtype = ctx.dtype(node.output[0], x)
    positive = ctx.b.op("GreaterOrEqual", [x, ctx.constant(0.0, dtype)])
    return (ctx.b.op("Where", [positive, ctx.constant(1.0, dtype), slope]),
            ctx.b.op("Where", [positive, ctx.constant(0.0, dtype), x]))


@forward_rule("PRelu")
def _prelu_forward(ctx, node, tangents):
    gain, negative = _prelu_parts(ctx, node)
    terms = []
    if tangents[0] is not None:
        terms.append(ctx.b.op("Mul", [tangents[0], ctx.lift(gain)]))
    if tangents[1] is not None:
        terms.append(ctx.b.op("Mul", [tangents[1], ctx.lift(negative)]))
    return ctx.sum(terms)


@reverse_rule("PRelu")
def _prelu_reverse(ctx, node, grads):
    gain, negative = _prelu_parts(ctx, node)
    rank = ctx.rank(node.output[0])
    return [ctx.unbroadcast(ctx.b.op("Mul", [grads[0], ctx.lift(gain)]), rank, node.input[0]),
            ctx.unbroadcast(ctx.b.op("Mul", [grads[0], ctx.lift(negative)]), rank,
                            node.input[1])]


# --- softmax family -------------------------------------------------------------------------
# The Jacobian of Softmax is diag(s) - s s^T, which is symmetric -- so forward and reverse are
# the same expression. LogSoftmax has Jacobian I - 1 s^T, which is not.

def _softmax_axis(ctx, node):
    if ctx.opset < 13:
        raise UnsupportedOperator(
            "%s before opset 13 flattens the input to 2-D; re-export at opset 13 or later"
            % node.op_type)
    return attribute(node, "axis", -1) % ctx.rank(node.input[0])


@forward_rule("Softmax")
def _softmax_forward(ctx, node, tangents):
    axis, y = _softmax_axis(ctx, node), node.output[0]
    seeded = ctx.full(tangents[0], node.input[0])
    weighted = ctx.b.op("Mul", [ctx.lift(y), seeded])
    total = ctx.reduce_sum(weighted, [axis], keepdims=1)
    return ctx.b.op("Mul", [ctx.lift(y), ctx.b.op("Sub", [seeded, total])])


@reverse_rule("Softmax")
def _softmax_reverse(ctx, node, grads):
    return [_softmax_forward(ctx, node, grads)]  # the Jacobian is its own transpose


@forward_rule("LogSoftmax")
def _log_softmax_forward(ctx, node, tangents):
    axis, y = _softmax_axis(ctx, node), node.output[0]
    seeded = ctx.full(tangents[0], node.input[0])
    probability = ctx.lift(ctx.b.op("Exp", [y]))
    total = ctx.reduce_sum(ctx.b.op("Mul", [probability, seeded]), [axis], keepdims=1)
    return ctx.b.op("Sub", [seeded, total])


@reverse_rule("LogSoftmax")
def _log_softmax_reverse(ctx, node, grads):
    axis, y = _softmax_axis(ctx, node), node.output[0]
    seeded = ctx.full(grads[0], node.output[0])
    probability = ctx.lift(ctx.b.op("Exp", [y]))
    total = ctx.reduce_sum(seeded, [axis], keepdims=1)
    return [ctx.b.op("Sub", [seeded, ctx.b.op("Mul", [probability, total])])]


# --- indexing and rearrangement -------------------------------------------------------------
# These move values without changing them, so the tangent is the same operation and the
# adjoint is its inverse: Split undoes into Concat, Slice into Pad, Pad into Slice, Gather
# into a scatter-add, Tile into a sum over the copies. The seed axis is trailing, so an axis
# counted from the front is used as it stands and one counted from the back is normalized.

def _constant_operand(ctx, node, index, what):
    if len(node.input) <= index or not node.input[index]:
        return None
    values = ctx.integers(node.input[index])
    if values is None:
        raise UnsupportedOperator(
            "%s is differentiable only with a constant '%s' operand" % (node.op_type, what))
    return values


@forward_rule("Split")
def _split_forward(ctx, node, tangents):
    axis = attribute(node, "axis", 0) % ctx.rank(node.input[0])
    seeded = ctx.full(tangents[0], node.input[0])
    operands = [seeded] + ([node.input[1]] if len(node.input) > 1 and node.input[1] else [])
    extra = {"num_outputs": len(node.output)} if len(operands) == 1 and \
        attribute(node, "num_outputs") is not None else {}
    return ctx.b.op("Split", operands, outputs=len(node.output), axis=axis, stem="t_split",
                    **extra)


@reverse_rule("Split")
def _split_reverse(ctx, node, grads):
    axis = attribute(node, "axis", 0) % ctx.rank(node.input[0])
    pieces = [ctx.zeros(name) if grad is None else ctx.full(grad, name)
              for name, grad in zip(node.output, grads)]
    return [ctx.b.op("Concat", pieces, axis=axis)] + [None]*(len(node.input) - 1)


def _slice_extent(ctx, node):
    """The per-axis (before, after) zero padding that undoes the slice."""
    starts = _constant_operand(ctx, node, 1, "starts")
    ends = _constant_operand(ctx, node, 2, "ends")
    axes = _constant_operand(ctx, node, 3, "axes")
    steps = _constant_operand(ctx, node, 4, "steps")
    if starts is None or ends is None:
        raise UnsupportedOperator("Slice is differentiable only with constant bounds")
    if steps is not None and any(step != 1 for step in steps):
        raise UnsupportedOperator(
            "Slice with a step other than 1 is not differentiated in reverse mode")
    rank = ctx.rank(node.input[0])
    shape = ctx.shapes.static(node.input[0])
    if shape is None:
        raise UnsupportedOperator(
            "Slice needs a declared shape to be differentiated in reverse mode")
    axes = list(range(len(starts))) if axes is None else [a % rank for a in axes]
    before, after = [0]*rank, [0]*rank
    for axis, start, end in zip(axes, starts, ends):
        extent = shape[axis]
        start = min(max(start + extent if start < 0 else start, 0), extent)
        end = min(max(end + extent if end < 0 else end, 0), extent)
        before[axis], after[axis] = start, max(extent - end, 0)
    return before, after


@forward_rule("Slice")
def _slice_forward(ctx, node, tangents):
    seeded = ctx.full(tangents[0], node.input[0])
    operands = [seeded] + list(node.input[1:])
    axes = _constant_operand(ctx, node, 3, "axes")
    if axes is not None:  # a negative axis would land on the seed axis
        operands[3] = ctx.b.ints([a % ctx.rank(node.input[0]) for a in axes])
    return ctx.b.op("Slice", operands, stem="t_slice")


@reverse_rule("Slice")
def _slice_reverse(ctx, node, grads):
    before, after = _slice_extent(ctx, node)
    padded = _pad(ctx, ctx.full(grads[0], node.output[0]), before + [0], after + [0])
    return [padded] + [None]*(len(node.input) - 1)


def _pad(ctx, value, before, after):
    """Zero-pad, with the trailing seed axis left alone (its entries are the last 0s)."""
    pads = ctx.b.ints(list(before) + list(after))
    if ctx.opset >= 11:
        return ctx.b.op("Pad", [value, pads], mode="constant", stem="padded")
    return ctx.b.op("Pad", [value], pads=list(before) + list(after), mode="constant",
                    stem="padded")


def _pad_widths(ctx, node):
    rank = ctx.rank(node.input[0])
    pads = _constant_operand(ctx, node, 1, "pads")
    if pads is None:
        pads = attribute(node, "pads")
    if pads is None:
        raise UnsupportedOperator("Pad is differentiable only with constant pads")
    axes = _constant_operand(ctx, node, 3, "axes")
    before, after = [0]*rank, [0]*rank
    chosen = list(range(rank)) if axes is None else [a % rank for a in axes]
    for position, axis in enumerate(chosen):
        before[axis], after[axis] = pads[position], pads[len(chosen) + position]
    return before, after


@forward_rule("Pad")
def _pad_forward(ctx, node, tangents):
    mode = attribute(node, "mode", "constant")
    if len(node.input) > 2 and node.input[2] and ctx.derivative.get(node.input[2]):
        raise UnsupportedOperator(
            "Pad with a differentiated constant_value is not supported; the padded region "
            "would need a seeded fill")
    before, after = _pad_widths(ctx, node)
    seeded = ctx.full(tangents[0], node.input[0])
    # reflect and edge are linear, so the tangent uses the same mode; constant pads with zero
    pads = ctx.b.ints(before + [0] + after + [0])
    if ctx.opset >= 11:
        return ctx.b.op("Pad", [seeded, pads], mode=mode, stem="t_pad")
    return ctx.b.op("Pad", [seeded], pads=before + [0] + after + [0], mode=mode, stem="t_pad")


@reverse_rule("Pad")
def _pad_reverse(ctx, node, grads):
    if attribute(node, "mode", "constant") != "constant":
        raise UnsupportedOperator(
            "only constant-mode Pad is differentiated in reverse; reflect and edge would "
            "have to accumulate onto the values they copy")
    before, after = _pad_widths(ctx, node)
    rank = ctx.rank(node.input[0])
    seeded = ctx.full(grads[0], node.output[0])
    starts = ctx.b.ints([max(b, 0) for b in before])
    ends = ctx.b.ints([INT64_MAX if a <= 0 else -a for a in after])
    axes = ctx.b.ints(list(range(rank)))
    return [ctx.b.op("Slice", [seeded, starts, ends, axes], stem="a_pad")] + \
        [None]*(len(node.input) - 1)


@forward_rule("Tile")
def _tile_forward(ctx, node, tangents):
    seeded = ctx.full(tangents[0], node.input[0])
    repeats = ctx.b.op("Concat", [node.input[1], ctx.b.ints([1])], axis=0, stem="t_repeats")
    return ctx.b.op("Tile", [seeded, repeats], stem="t_tile")


@reverse_rule("Tile")
def _tile_reverse(ctx, node, grads):
    repeats = _constant_operand(ctx, node, 1, "repeats")
    shape = ctx.shapes.static(node.input[0])
    if repeats is None or shape is None:
        raise UnsupportedOperator(
            "Tile is differentiable in reverse only with constant repeats and a declared "
            "input shape, so the copies can be summed")
    # split each tiled axis into (repeat, extent) and sum the repeats away
    split = []
    for repeat, extent in zip(repeats, shape):
        split += [repeat, extent]
    seeded = ctx.full(grads[0], node.output[0])
    reshaped = ctx.b.op("Reshape", [seeded, ctx.b.ints(split + [-1])], stem="a_tile")
    reduced = ctx.reduce_sum(reshaped, list(range(0, 2*len(shape), 2)), keepdims=0)
    return [ctx.reshape_like(reduced, node.input[0]), None]


@forward_rule("CumSum")
def _cumsum_forward(ctx, node, tangents):
    seeded = ctx.full(tangents[0], node.input[0])
    return ctx.b.op("CumSum", [seeded, _cumsum_axis(ctx, node)],
                    exclusive=attribute(node, "exclusive", 0),
                    reverse=attribute(node, "reverse", 0), stem="t_cumsum")


@reverse_rule("CumSum")
def _cumsum_reverse(ctx, node, grads):
    # the adjoint of a prefix sum is a suffix sum
    seeded = ctx.full(grads[0], node.output[0])
    return [ctx.b.op("CumSum", [seeded, _cumsum_axis(ctx, node)],
                     exclusive=attribute(node, "exclusive", 0),
                     reverse=1 - attribute(node, "reverse", 0), stem="a_cumsum"), None]


def _cumsum_axis(ctx, node):
    axis = ctx.integers(node.input[1])
    if axis is None:
        raise UnsupportedOperator("CumSum is differentiable only with a constant axis")
    return ctx.b.constant(axis[0] % ctx.rank(node.input[0]), TensorProto.INT64, ())


@forward_rule("Gather")
def _gather_forward(ctx, node, tangents):
    axis = attribute(node, "axis", 0) % ctx.rank(node.input[0])
    seeded = ctx.full(tangents[0], node.input[0])
    return ctx.b.op("Gather", [seeded, node.input[1]], axis=axis, stem="t_gather")


@reverse_rule("Gather")
def _gather_reverse(ctx, node, grads):
    """The adjoint of a gather is a scatter-add into a zero tensor."""
    if ctx.opset < 16:
        raise UnsupportedOperator(
            "Gather needs ScatterND with reduction='add' (opset 16) to be differentiated "
            "in reverse; re-export at opset 16 or later")
    data, indices, y = node.input[0], node.input[1], node.output[0]
    rank = ctx.rank(data)
    axis = attribute(node, "axis", 0) % rank
    seeded = ctx.full(grads[0], y)
    zeros = ctx.zeros(data)
    if axis:  # ScatterND indexes the leading axes, so bring the gathered one to the front
        order = [axis] + [a for a in range(rank) if a != axis]
        zeros = ctx.b.op("Transpose", [zeros], perm=order + [rank], stem="a_zeros")
        seeded = ctx.b.op("Transpose", [seeded], perm=_gather_order(ctx, node, rank, axis),
                          stem="a_seeded")
    # ScatterND rejects negative indices, which Gather accepts. Every constant here is a
    # scalar: a [1]-shaped one would broadcast a scalar index -- what `x[2]` exports to -- up
    # to rank 1, and ScatterND would then expect updates with an axis they do not have.
    indices = ctx.b.op("Cast", [indices], to=TensorProto.INT64, stem="indices")
    zero = ctx.b.constant(0, TensorProto.INT64, ())
    extent = ctx.b.op("Gather", [ctx.shape_of(data),
                                 ctx.b.constant(axis, TensorProto.INT64, ())], axis=0,
                      stem="extent")
    normalized = ctx.b.op("Where", [ctx.b.op("Less", [indices, zero]),
                                    ctx.b.op("Add", [indices, extent]), indices],
                          stem="positive_indices")
    scattered = ctx.b.op("ScatterND", [zeros, ctx.unsqueeze(normalized, [-1]), seeded],
                         reduction="add", stem="a_gather")
    if axis:
        order = [axis] + [a for a in range(rank) if a != axis]
        inverse = [0]*rank
        for position, source in enumerate(order):
            inverse[source] = position
        scattered = ctx.b.op("Transpose", [scattered], perm=inverse + [rank], stem="a_back")
    return [scattered, None]


def _gather_order(ctx, node, rank, axis):
    """Move the gathered block of axes to the front of the result, seed axis last."""
    indices_rank = ctx.rank(node.input[1])
    block = list(range(axis, axis + indices_rank))
    rest = [a for a in range(rank - 1 + indices_rank) if a not in block]
    return block + rest + [rank - 1 + indices_rank]


# --- the rest of the reductions -------------------------------------------------------------
# Every one of these is "weight each element, then sum over the reduced axes": the weight is
# what distinguishes them, and both modes use it -- forward sums the weighted tangent, reverse
# broadcasts the adjoint back and weights it. ReduceSum and ReduceMean are above; they are the
# two whose weight is a constant.

def _reference(ctx, node, axes, keepdims):
    """The result, shaped so it broadcasts against the input."""
    y = node.output[0]
    return y if keepdims else ctx.unsqueeze(y, axes)


def _weight_extremum(ctx, node, axes, keepdims, dtype):
    """The winners, sharing the derivative equally when several elements tie."""
    mask = ctx.b.op("Cast", [ctx.b.op("Equal", [node.input[0],
                                                _reference(ctx, node, axes, keepdims)])],
                    to=dtype)
    return ctx.b.op("Div", [mask, ctx.reduce_sum(mask, axes, keepdims=1)])


def _weight_log_sum_exp(ctx, node, axes, keepdims, dtype):
    return ctx.b.op("Exp", [ctx.b.op("Sub", [node.input[0],
                                             _reference(ctx, node, axes, keepdims)])])


def _weight_l1(ctx, node, axes, keepdims, dtype):
    return ctx.b.op("Sign", [node.input[0]])


def _weight_l2(ctx, node, axes, keepdims, dtype):
    return ctx.b.op("Div", [node.input[0], _reference(ctx, node, axes, keepdims)])


def _weight_sum_square(ctx, node, axes, keepdims, dtype):
    return ctx.b.op("Mul", [node.input[0], ctx.constant(2.0, dtype)])


def _weight_product(ctx, node, axes, keepdims, dtype):
    # y/x, which is the product of the other factors wherever x is not zero
    return ctx.b.op("Div", [_reference(ctx, node, axes, keepdims), node.input[0]])


WEIGHTED_REDUCTIONS = {
    "ReduceMax": _weight_extremum, "ReduceMin": _weight_extremum,
    "ReduceLogSumExp": _weight_log_sum_exp, "ReduceL1": _weight_l1,
    "ReduceL2": _weight_l2, "ReduceSumSquare": _weight_sum_square,
    "ReduceProd": _weight_product,
}


@forward_rule(*WEIGHTED_REDUCTIONS)
def _weighted_reduce_forward(ctx, node, tangents):
    axes, keepdims = _reduction_axes(ctx, node)
    seeded = ctx.full(tangents[0], node.input[0])
    if not axes:
        return seeded
    weight = WEIGHTED_REDUCTIONS[node.op_type](ctx, node, axes, keepdims,
                                               ctx.dtype(node.output[0], node.input[0]))
    return ctx.reduce_sum(ctx.b.op("Mul", [ctx.lift(weight), seeded]), axes,
                          keepdims=keepdims)


@reverse_rule(*WEIGHTED_REDUCTIONS)
def _weighted_reduce_reverse(ctx, node, grads):
    axes, keepdims = _reduction_axes(ctx, node)
    seeded = ctx.full(grads[0], node.output[0])
    if not axes:
        return [seeded]
    if not keepdims:
        seeded = ctx.unsqueeze(seeded, axes)
    weight = WEIGHTED_REDUCTIONS[node.op_type](ctx, node, axes, keepdims,
                                               ctx.dtype(node.output[0], node.input[0]))
    return [ctx.full(ctx.b.op("Mul", [ctx.lift(weight), seeded]), node.input[0])]


# --- normalization --------------------------------------------------------------------------

def _channel_shape(ctx, rank):
    """[1, -1, 1, ...]: a per-channel vector reshaped to broadcast along axis 1."""
    return ctx.b.ints([1, -1] + [1]*(rank - 2))


@forward_rule("BatchNormalization")
def _batch_norm_forward(ctx, node, tangents):
    gain, standardized = _batch_norm_parts(ctx, node, tangents)
    terms = []
    if tangents[0] is not None:
        terms.append(ctx.b.op("Mul", [tangents[0], ctx.lift(gain)]))
    if len(tangents) > 1 and tangents[1] is not None:
        terms.append(ctx.b.op("Mul", [_as_channel(ctx, tangents[1], node), ctx.lift(
            standardized)]))
    if len(tangents) > 2 and tangents[2] is not None:
        terms.append(_as_channel(ctx, tangents[2], node))
    return [ctx.sum(terms)] + [None]*(len(node.output) - 1)


@reverse_rule("BatchNormalization")
def _batch_norm_reverse(ctx, node, grads):
    gain, standardized = _batch_norm_parts(ctx, node, [None]*len(node.input))
    rank = ctx.rank(node.output[0])
    outside = [a for a in range(rank) if a != 1]
    seeded = ctx.full(grads[0], node.output[0])

    def per_channel(value):
        return ctx.reshape_like(ctx.reduce_sum(value, outside, keepdims=0), node.input[1])

    return [ctx.b.op("Mul", [seeded, ctx.lift(gain)]),
            per_channel(ctx.b.op("Mul", [seeded, ctx.lift(standardized)])),
            per_channel(seeded), None, None]


def _batch_norm_parts(ctx, node, tangents):
    """Inference-mode batch norm is affine: y = gain.(x - mean) + B, gain per channel."""
    if attribute(node, "training_mode", 0):
        raise UnsupportedOperator(
            "BatchNormalization with training_mode is not differentiated; export in "
            "inference mode, where it is an affine map")
    for index in (3, 4):
        if len(tangents) > index and tangents[index] is not None:
            raise UnsupportedOperator(
                "the running mean and variance of BatchNormalization are statistics, not "
                "differentiable operands")
    x, scale, mean, variance = node.input[0], node.input[1], node.input[3], node.input[4]
    rank, dtype = ctx.rank(x), ctx.dtype(node.output[0], x)
    epsilon = attribute(node, "epsilon", 1e-5)
    deviation = ctx.b.op("Sqrt", [ctx.b.op("Add", [variance, ctx.constant(epsilon, dtype)])])
    gain = ctx.b.op("Reshape", [ctx.b.op("Div", [scale, deviation]),
                                _channel_shape(ctx, rank)], stem="gain")
    centered = ctx.b.op("Sub", [x, ctx.b.op("Reshape", [mean, _channel_shape(ctx, rank)])])
    standardized = ctx.b.op("Div", [centered, ctx.b.op("Reshape", [
        deviation, _channel_shape(ctx, rank)])], stem="standardized")
    return gain, standardized


def _as_channel(ctx, seeded, node):
    """A seeded per-channel vector (`[C, nseed]`) reshaped to broadcast along axis 1."""
    rank = ctx.rank(node.output[0])
    shape = ctx.b.op("Concat", [ctx.b.ints([1, -1] + [1]*(rank - 2)), ctx.count()],
                     axis=0, stem="channel_shape")
    return ctx.b.op("Reshape", [seeded, shape], stem="channel")


@forward_rule("LayerNormalization")
def _layer_norm_forward(ctx, node, tangents):
    axes, standardized, invstd = _layer_norm_parts(ctx, node)
    x = node.input[0]
    terms = []
    if tangents[0] is not None:
        seeded = ctx.full(tangents[0], x)
        # d xhat = invstd . (t - mean(t) - xhat . mean(xhat . t)) over the normalized axes
        centered = ctx.b.op("Sub", [seeded, ctx.reduce_mean(seeded, axes)])
        projection = ctx.b.op("Mul", [ctx.lift(standardized), ctx.reduce_mean(
            ctx.b.op("Mul", [ctx.lift(standardized), seeded]), axes)])
        direction = ctx.b.op("Mul", [ctx.lift(invstd),
                                     ctx.b.op("Sub", [centered, projection])])
        terms.append(direction if len(node.input) < 2 or not node.input[1] else
                     ctx.b.op("Mul", [direction, ctx.lift(node.input[1])]))
    if len(tangents) > 1 and tangents[1] is not None:
        terms.append(ctx.b.op("Mul", [tangents[1], ctx.lift(standardized)]))
    if len(tangents) > 2 and tangents[2] is not None:
        terms.append(tangents[2])
    return [ctx.sum(terms)] + [None]*(len(node.output) - 1)


@reverse_rule("LayerNormalization")
def _layer_norm_reverse(ctx, node, grads):
    if any(grad is not None for grad in grads[1:]):
        raise UnsupportedOperator(
            "the mean and inverse standard deviation LayerNormalization reports are not "
            "differentiated; seed only its first output")
    axes, standardized, invstd = _layer_norm_parts(ctx, node)
    rank = ctx.rank(node.output[0])
    seeded = ctx.full(grads[0], node.output[0])
    scaled = seeded if len(node.input) < 2 or not node.input[1] else \
        ctx.b.op("Mul", [seeded, ctx.lift(node.input[1])])
    centered = ctx.b.op("Sub", [scaled, ctx.reduce_mean(scaled, axes)])
    projection = ctx.b.op("Mul", [ctx.lift(standardized), ctx.reduce_mean(
        ctx.b.op("Mul", [ctx.lift(standardized), scaled]), axes)])
    contributions = [ctx.b.op("Mul", [ctx.lift(invstd),
                                      ctx.b.op("Sub", [centered, projection])])]
    if len(node.input) > 1 and node.input[1]:
        contributions.append(ctx.unbroadcast(
            ctx.b.op("Mul", [seeded, ctx.lift(standardized)]), rank, node.input[1]))
    if len(node.input) > 2 and node.input[2]:
        contributions.append(ctx.unbroadcast(seeded, rank, node.input[2]))
    return contributions


def _layer_norm_parts(ctx, node):
    """The normalized axes, the standardized input, and 1/sqrt(var + eps)."""
    x = node.input[0]
    rank, dtype = ctx.rank(x), ctx.dtype(node.output[0], x)
    axis = attribute(node, "axis", -1) % rank
    axes = list(range(axis, rank))
    epsilon = attribute(node, "epsilon", 1e-5)
    centered = ctx.b.op("Sub", [x, ctx.reduce_mean(x, axes)])
    variance = ctx.reduce_mean(ctx.b.op("Mul", [centered, centered]), axes)
    invstd = ctx.b.op("Reciprocal", [ctx.b.op("Sqrt", [ctx.b.op("Add", [
        variance, ctx.constant(epsilon, dtype)])])], stem="invstd")
    return axes, ctx.b.op("Mul", [centered, invstd], stem="standardized"), invstd


@forward_rule("GatherND")
def _gather_nd_forward(ctx, node, tangents):
    _reject_batch_dims(node)
    seeded = ctx.full(tangents[0], node.input[0])
    return ctx.b.op("GatherND", [seeded, node.input[1]], stem="t_gathernd")


@reverse_rule("GatherND")
def _gather_nd_reverse(ctx, node, grads):
    """ScatterND is exactly GatherND's adjoint, once the indices are made non-negative."""
    _reject_batch_dims(node)
    if ctx.opset < 16:
        raise UnsupportedOperator(
            "GatherND needs ScatterND with reduction='add' (opset 16) to be differentiated "
            "in reverse; re-export at opset 16 or later")
    data, indices = node.input[0], node.input[1]
    depth = ctx.shapes.static(indices)
    if depth is None:
        raise UnsupportedOperator("GatherND needs a declared indices shape in reverse mode")
    extents = ctx.b.op("Slice", [ctx.shape_of(data), ctx.b.ints([0]),
                                 ctx.b.ints([depth[-1]])], stem="extents")
    positive = ctx.b.op("Where", [ctx.b.op("Less", [indices, ctx.b.ints([0])]),
                                  ctx.b.op("Add", [indices, extents]), indices],
                        stem="positive_indices")
    return [ctx.b.op("ScatterND", [ctx.zeros(data), positive,
                                   ctx.full(grads[0], node.output[0])],
                     reduction="add", stem="a_gathernd"), None]


def _reject_batch_dims(node):
    if attribute(node, "batch_dims", 0):
        raise UnsupportedOperator(
            "GatherND with batch_dims is not differentiated; ScatterND, its adjoint, has no "
            "matching attribute")


# --- convolution ----------------------------------------------------------------------------
# Conv has no room for a seed axis: its operands are [batch, channel, spatial...]. But a
# convolution is independent across the batch, so the seed axis folds into the batch for the
# input's tangent, and into the output channels for the weight's. The adjoint with respect to
# the input is a ConvTranspose with the very same weight and attributes.

def _conv_attributes(ctx, node):
    auto_pad = attribute(node, "auto_pad", "NOTSET")
    if auto_pad not in ("NOTSET", "VALID"):
        raise UnsupportedOperator(
            "Conv with auto_pad=%s is not differentiated; re-export with explicit pads"
            % auto_pad)
    spatial = ctx.rank(node.input[0]) - 2
    attrs = {"group": attribute(node, "group", 1),
             "strides": list(attribute(node, "strides", [1]*spatial)),
             "dilations": list(attribute(node, "dilations", [1]*spatial)),
             "pads": list(attribute(node, "pads", [0]*(2*spatial)))}
    kernel = attribute(node, "kernel_shape")
    if kernel is not None:
        attrs["kernel_shape"] = list(kernel)
    return attrs, spatial


def _merge_leading(ctx, value, tail=None):
    """[a, b, rest...] -> [a*b, rest...].

    `tail` is `rest` when it is statically known, which makes the target a constant any
    version of ONNX shape inference can see through -- so a later pass differentiating this
    model again still knows the rank. Without it the target is computed at run time.

    Not `Reshape` with zeros: a 0 in a reshape target copies the input's dimension at the
    *same* index, which after a merge is the wrong one -- it silently produces a tensor of
    the right size and the wrong shape.
    """
    if tail is not None and None not in tail:
        target = ctx.b.ints([-1] + list(tail))
    else:
        tail_ = ctx.b.op("Slice", [ctx.b.op("Shape", [value], stem="merge_shape"),
                                   ctx.b.ints([2]), ctx.b.ints([INT64_MAX])], stem="tail")
        target = ctx.b.op("Concat", [ctx.b.ints([-1]), tail_], axis=0, stem="merged_shape")
    return ctx.b.op("Reshape", [value, target], stem="merged")


def _fold_seed_into_batch(ctx, value, primal):
    """[batch, ...spatial, nseed] -> [nseed*batch, ...spatial], one image per seed."""
    static = ctx.shapes.static(primal)
    return _merge_leading(ctx, _seed_front(ctx, value, ctx.rank(primal)),
                          None if static is None else static[1:])


def _unfold_batch_into_seed(ctx, value, reference, rank):
    """The inverse, with `reference` the primal tensor whose shape the result must have.

    A constant target where the shape is declared, for the same reason as in
    `_merge_leading`: older shape inference cannot see through a computed one.
    """
    static = ctx.shapes.static(reference)
    target = ctx.b.ints([-1] + list(static)) if static is not None else ctx.b.op(
        "Concat", [ctx.b.ints([-1]), ctx.shape_of(reference)], axis=0, stem="unfold")
    return _seed_back(ctx, ctx.b.op("Reshape", [value, target], stem="unfolded"), rank)


@forward_rule("Conv")
def _conv_forward(ctx, node, tangents):
    attrs, spatial = _conv_attributes(ctx, node)
    x, w, y = node.input[0], node.input[1], node.output[0]
    terms = []
    if tangents[0] is not None:
        folded = _fold_seed_into_batch(ctx, ctx.full(tangents[0], x), x)
        product = ctx.b.op("Conv", [folded, w], stem="t_conv", **attrs)
        terms.append(_unfold_batch_into_seed(ctx, product, y, ctx.rank(y)))
    if tangents[1] is not None:
        if attrs["group"] != 1:
            raise UnsupportedOperator(
                "a grouped Conv is not differentiated with respect to its weight; the seed "
                "axis would have to fold into the output channels, which the groups own")
        seeded = ctx.full(tangents[1], w)
        # fold the seeds into the output channels: one filter bank per seed
        static_w = ctx.shapes.static(w)
        stacked = _merge_leading(ctx, _seed_front(ctx, seeded, ctx.rank(w)),
                                 None if static_w is None else static_w[1:])
        product = ctx.b.op("Conv", [x, stacked], stem="t_conv_w", **attrs)
        # [batch, nseed*out, spatial...] -> [batch, out, spatial..., nseed]
        split = ctx.b.op("Reshape", [product, ctx.b.op("Concat", [
            ctx.b.ints([0, -1]), ctx.b.op("Slice", [ctx.shape_of(y), ctx.b.ints([1]),
                                                    ctx.b.ints([INT64_MAX])])],
            axis=0, stem="t_split_shape")], stem="t_split")
        terms.append(ctx.b.op("Transpose", [split],
                              perm=[0, 2] + list(range(3, 3 + spatial)) + [1],
                              stem="t_conv_seeded"))
    if len(tangents) > 2 and tangents[2] is not None:
        terms.append(_as_channel(ctx, tangents[2], node))
    return ctx.sum(terms)


@reverse_rule("Conv")
def _conv_reverse(ctx, node, grads):
    attrs, spatial = _conv_attributes(ctx, node)
    x, w, y = node.input[0], node.input[1], node.output[0]
    rank = ctx.rank(x)
    seeded = ctx.full(grads[0], y)
    contributions = [None]
    if ctx.asked_for(x):
        shape = ctx.shapes.static(x)
        if shape is None:
            raise UnsupportedOperator(
                "Conv needs a declared input shape in reverse mode, so the transposed "
                "convolution knows how much of its output to keep")
        transposed = dict(attrs, output_shape=list(shape[2:]))
        transposed.pop("kernel_shape", None)
        folded = _fold_seed_into_batch(ctx, seeded, y)
        contributions[0] = _unfold_batch_into_seed(
            ctx, ctx.b.op("ConvTranspose", [folded, w], stem="a_conv", **transposed), x, rank)
    contributions.append(_conv_weight_adjoint(ctx, node, attrs, spatial, seeded))
    if len(node.input) > 2 and node.input[2] and ctx.asked_for(node.input[2]):
        outside = [a for a in range(ctx.rank(y)) if a != 1]
        contributions.append(ctx.reshape_like(
            ctx.reduce_sum(seeded, outside, keepdims=0), node.input[2]))
    return contributions


def _conv_weight_adjoint(ctx, node, attrs, spatial, seeded):
    """d L/d W: a convolution of the padded input with the adjoint as the filter bank.

    Swapping the batch and channel axes of both turns the weight gradient into an ordinary
    convolution whose strides are the original dilations and whose dilations are the
    original strides. The seeds ride in the filter bank's output channels.
    """
    x, w = node.input[0], node.input[1]
    if not ctx.asked_for(w):
        return None                     # a constant weight: nothing to contribute to
    if attrs["group"] != 1:
        raise UnsupportedOperator(
            "a grouped Conv is not differentiated with respect to its weight")
    kernel = ctx.shapes.static(w)
    if kernel is None:
        raise UnsupportedOperator("Conv needs a declared weight shape in reverse mode")
    rank = ctx.rank(x)
    pads = attrs["pads"]
    padded = x if not any(pads) else _pad(
        ctx, x, [0, 0] + list(pads[:spatial]), [0, 0] + list(pads[spatial:]))
    swap = [1, 0] + list(range(2, rank))
    # [batch, out, spatial..., nseed] -> [out*nseed, batch, spatial...]
    static_y = ctx.shapes.static(node.output[0])
    filters = _merge_leading(ctx, ctx.b.op(
        "Transpose", [seeded], perm=[1, rank] + [0] + list(range(2, rank)),
        stem="a_filters"), None if static_y is None else [static_y[0]] + list(static_y[2:]))
    product = ctx.b.op("Conv", [ctx.b.op("Transpose", [padded], perm=swap), filters],
                       strides=attrs["dilations"], dilations=attrs["strides"],
                       pads=[0]*(2*spatial), group=1, stem="a_conv_w")
    # crop any overhang, then split the seeds back out of the output channels
    product = ctx.b.op("Slice", [product, ctx.b.ints([0]*spatial),
                                 ctx.b.ints(list(kernel[2:])),
                                 ctx.b.ints(list(range(2, rank)))], stem="a_crop")
    split = ctx.b.op("Reshape", [product, ctx.b.op("Concat", [
        ctx.b.ints([0, kernel[0], -1]), ctx.b.ints(list(kernel[2:]))], axis=0,
        stem="a_split_shape")], stem="a_split")
    # [in, out, nseed, spatial...] -> [out, in, spatial..., nseed]
    return ctx.b.op("Transpose", [split],
                    perm=[1, 0] + list(range(3, 3 + spatial)) + [2], stem="a_conv_weight")


# --- operations the passes emit, so that their output differentiates again --------------
# The reverse of Gather and GatherND is a ScatterND, and the reverse of Conv a
# ConvTranspose. Without rules for those two, forward(reverse(model)) -- the second-order
# file `family` writes -- fails on any network that gathers or convolves.

def _scatter_reduction(node):
    reduction = attribute(node, "reduction", "none")
    if reduction not in ("none", "add"):
        raise UnsupportedOperator(
            "ScatterND with reduction='%s' is not differentiated; only 'none' and 'add' are "
            "linear" % reduction)
    return reduction


@forward_rule("ScatterND")
def _scatter_nd_forward(ctx, node, tangents):
    reduction = _scatter_reduction(node)
    data, indices, updates = node.input[0], node.input[1], node.input[2]
    seeded = [ctx.zeros(name) if tangent is None else ctx.full(tangent, name)
              for name, tangent in ((data, tangents[0]), (updates, tangents[2]))]
    extra = {"reduction": reduction} if reduction != "none" else {}
    return ctx.b.op("ScatterND", [seeded[0], indices, seeded[1]], stem="t_scatternd", **extra)


@reverse_rule("ScatterND")
def _scatter_nd_reverse(ctx, node, grads):
    reduction = _scatter_reduction(node)
    indices, updates = node.input[1], node.input[2]
    seeded = ctx.full(grads[0], node.output[0])
    # with 'none' the scattered positions were overwritten, so none of it reaches the data
    to_data = seeded if reduction == "add" else ctx.b.op(
        "ScatterND", [seeded, indices, ctx.zeros(updates)], stem="a_scatternd")
    to_updates = ctx.b.op("GatherND", [seeded, indices], stem="a_updates")
    return [to_data, None, to_updates]


def _conv_transpose_pads(ctx, node, attrs, spatial):
    """The padding a ConvTranspose actually uses: an `output_shape` overrides `pads`."""
    shape = attribute(node, "output_shape")
    if shape is None:
        return list(attrs["pads"])
    x = ctx.shapes.static(node.input[0])
    w = ctx.shapes.static(node.input[1])
    if x is None or w is None:
        raise UnsupportedOperator(
            "ConvTranspose with output_shape needs declared input and weight shapes to be "
            "differentiated in reverse mode")
    shape = list(shape)[-spatial:]
    padding = list(attribute(node, "output_padding", [0]*spatial))
    begin, end = [], []
    for i in range(spatial):
        total = (attrs["strides"][i]*(x[2 + i] - 1) + padding[i]
                 + (w[2 + i] - 1)*attrs["dilations"][i] + 1 - shape[i])
        begin.append(total - total//2)   # the spec's split when auto_pad is not SAME_UPPER
        end.append(total//2)
    return begin + end


@forward_rule("ConvTranspose")
def _conv_transpose_forward(ctx, node, tangents):
    attrs, spatial = _conv_attributes(ctx, node)
    if tangents[1] is not None:
        raise UnsupportedOperator(
            "ConvTranspose is not differentiated with respect to its weight")
    x, w, y = node.input[0], node.input[1], node.output[0]
    terms = []
    if tangents[0] is not None:
        folded = _fold_seed_into_batch(ctx, ctx.full(tangents[0], x), x)
        extra = {key: attribute(node, key) for key in ("output_shape", "output_padding")
                 if attribute(node, key) is not None}
        product = ctx.b.op("ConvTranspose", [folded, w], stem="t_convt", **attrs, **extra)
        terms.append(_unfold_batch_into_seed(ctx, product, y, ctx.rank(y)))
    if len(tangents) > 2 and tangents[2] is not None:
        terms.append(_as_channel(ctx, tangents[2], node))
    return ctx.sum(terms)


@reverse_rule("ConvTranspose")
def _conv_transpose_reverse(ctx, node, grads):
    """The adjoint of a transposed convolution is the convolution with the same weight."""
    attrs, spatial = _conv_attributes(ctx, node)
    x, w, y = node.input[0], node.input[1], node.output[0]
    if ctx.asked_for(w):
        raise UnsupportedOperator(
            "ConvTranspose is not differentiated with respect to its weight")
    seeded = ctx.full(grads[0], y)
    contributions = [None, None]
    if ctx.asked_for(x):
        convolution = dict(attrs, pads=_conv_transpose_pads(ctx, node, attrs, spatial))
        folded = _fold_seed_into_batch(ctx, seeded, y)
        contributions[0] = _unfold_batch_into_seed(
            ctx, ctx.b.op("Conv", [folded, w], stem="a_convt", **convolution), x, ctx.rank(x))
    if len(node.input) > 2 and node.input[2] and ctx.asked_for(node.input[2]):
        outside = [a for a in range(ctx.rank(y)) if a != 1]
        contributions.append(ctx.reshape_like(
            ctx.reduce_sum(seeded, outside, keepdims=0), node.input[2]))
    return contributions


# --- element-wise indexing ------------------------------------------------------------------
# GatherElements and ScatterElements address every element through an index tensor of the
# data's own rank, so the seed axis needs an index of its own: the same index, repeated along
# it. Both accept negative indices, so no normalization is needed.

def _seeded_indices(ctx, indices):
    """`indices` with a trailing seed axis, each seed direction addressed identically."""
    return ctx.b.op("Expand", [ctx.unsqueeze(indices, [-1]), ctx.seeded_shape(indices)],
                    stem="seeded_indices")


@forward_rule("GatherElements")
def _gather_elements_forward(ctx, node, tangents):
    data, indices = node.input[0], node.input[1]
    axis = attribute(node, "axis", 0) % ctx.rank(data)
    return ctx.b.op("GatherElements", [ctx.full(tangents[0], data),
                                       _seeded_indices(ctx, indices)], axis=axis,
                    stem="t_gatherelements")


@reverse_rule("GatherElements")
def _gather_elements_reverse(ctx, node, grads):
    """A scatter-add of the adjoint into a zero tensor: repeated indices accumulate."""
    if ctx.opset < 16:
        raise UnsupportedOperator(
            "GatherElements needs ScatterElements with reduction='add' (opset 16) to be "
            "differentiated in reverse; re-export at opset 16 or later")
    data, indices = node.input[0], node.input[1]
    axis = attribute(node, "axis", 0) % ctx.rank(data)
    return [ctx.b.op("ScatterElements", [ctx.zeros(data), _seeded_indices(ctx, indices),
                                         ctx.full(grads[0], node.output[0])],
                     axis=axis, reduction="add", stem="a_gatherelements"), None]


@forward_rule("ScatterElements", "Scatter")
def _scatter_elements_forward(ctx, node, tangents):
    reduction = _scatter_reduction(node)
    data, indices, updates = node.input[0], node.input[1], node.input[2]
    axis = attribute(node, "axis", 0) % ctx.rank(data)
    seeded = [ctx.zeros(name) if tangent is None else ctx.full(tangent, name)
              for name, tangent in ((data, tangents[0]), (updates, tangents[2]))]
    extra = {"reduction": reduction} if reduction != "none" else {}
    return ctx.b.op(node.op_type, [seeded[0], _seeded_indices(ctx, indices), seeded[1]],
                    axis=axis, stem="t_scatterelements", **extra)


@reverse_rule("ScatterElements", "Scatter")
def _scatter_elements_reverse(ctx, node, grads):
    reduction = _scatter_reduction(node)
    data, indices, updates = node.input[0], node.input[1], node.input[2]
    axis = attribute(node, "axis", 0) % ctx.rank(data)
    seeded = ctx.full(grads[0], node.output[0])
    index = _seeded_indices(ctx, indices)
    # with 'none' the scattered positions were overwritten, so none of it reaches the data
    to_data = seeded if reduction == "add" else ctx.b.op(
        node.op_type, [seeded, index, ctx.zeros(updates)], axis=axis, stem="a_scatter")
    to_updates = ctx.b.op("GatherElements", [seeded, index], axis=axis, stem="a_updates")
    return [to_data, None, to_updates]


# --- Range -------------------------------------------------------------------------------
# start + delta * i for i = 0, 1, ...: linear in start and delta. Most Ranges are integer and
# never differentiated; the rule matters because without one a Range would be expanded into
# its spec body -- a Loop -- which is slower, and not a faithful drop-in when `start` is a
# one-element tensor rather than a scalar.

def _range_index(ctx, node):
    y = node.output[0]
    dtype = ctx.dtype(y)
    count = ctx.b.op("Size", [y], stem="range_count")
    steps = ctx.b.op("Range", [ctx.b.constant(0, TensorProto.INT64, ()), count,
                               ctx.b.constant(1, TensorProto.INT64, ())], stem="range_steps")
    return ctx.b.op("Cast", [steps], to=dtype, stem="range_index")


@forward_rule("Range")
def _range_forward(ctx, node, tangents):
    terms = []
    if tangents[0] is not None:  # d/d start: every element moves with it
        terms.append(tangents[0])
    if tangents[2] is not None:  # d/d delta: element i moves i times as far
        terms.append(ctx.b.op("Mul", [ctx.lift(_range_index(ctx, node)), tangents[2]]))
    total = ctx.sum(terms)
    return None if total is None else ctx.full(total, node.output[0])


@reverse_rule("Range")
def _range_reverse(ctx, node, grads):
    seeded = ctx.full(grads[0], node.output[0])
    to_start = ctx.reduce_sum(seeded, [0], keepdims=0) if ctx.asked_for(node.input[0]) else None
    to_delta = None
    if len(node.input) > 2 and ctx.asked_for(node.input[2]):
        to_delta = ctx.reduce_sum(ctx.b.op("Mul", [seeded, ctx.lift(_range_index(ctx, node))]),
                                  [0], keepdims=0)
    return [to_start, None, to_delta]
