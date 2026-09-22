"""Lower operations the spec defines by prose alone into primitives that have rules.

`LRN`, `GridSample`, `STFT` and `TensorScatter` come without a function body, yet each is a
short composition of operations the rule table covers: a padded channel-window sum, a
weighted sum of gathers, a gather of frames followed by a DFT, a scatter at computed
positions. Writing that composition out -- as `expand_functions` does for the operations the
spec *does* give a body -- gives both derivative modes, to any order, without a rule of
their own; the lowered primal is the spec's definition, evaluated with primitive kernels.

GridSample's lowering follows the ONNX reference implementation step by step: unnormalize
the coordinate, round it (nearest), fold it back inside the image (border, reflection),
then interpolate over the taps each mode reads -- per tap an index, folded back again for
the padding mode, and a weight. Its derivative with respect to the grid flows through the
weights; floors and roundings contribute nothing, as they should.
"""
import itertools

from onnx import AttributeProto, TensorProto, helper

from ._build import INT64_MAX, Builder, Shapes, UnsupportedOperator, attribute
from ._graph import all_constants, all_names, subgraphs, walk

LOWERINGS = {}


def lowering(op_type):
    def register(fn):
        LOWERINGS[op_type] = fn
        return fn
    return register


def lower(model):
    """A copy of `model` with every LRN, GridSample, STFT and TensorScatter node lowered."""
    if not any(node.op_type in LOWERINGS and node.domain in ("", "ai.onnx")
               for graph in walk(model.graph) for node in graph.node):
        return model
    opset = max([o.version for o in model.opset_import if o.domain in ("", "ai.onnx")] or [18])
    state = _Lowering(model, opset)
    result = type(model)()
    result.CopyFrom(model)
    result.graph.CopyFrom(state.graph(model.graph, state.b))
    return result


class _Lowering:
    def __init__(self, model, opset):
        self.opset = opset
        self.shapes = Shapes(model)
        self.values = all_constants(model.graph)
        self.b = Builder(all_names(model.graph))

    def graph(self, graph, builder):
        """The graph with its nodes lowered; new constants become its own initializers, since
        shape inference inside a subgraph does not see through outer ones."""
        nodes = []
        for node in graph.node:
            if node.op_type in LOWERINGS and node.domain in ("", "ai.onnx"):
                builder.nodes = []
                self.b_ = builder
                LOWERINGS[node.op_type](self, node)
                nodes.extend(builder.nodes)
            elif subgraphs(node):
                nodes.append(self.descend(node, builder))
            else:
                nodes.append(node)
        builder.nodes = []
        copy = type(graph)()
        copy.CopyFrom(graph)
        copy.ClearField("node")
        copy.node.extend(nodes)
        copy.initializer.extend(builder.initializers)
        builder.initializers = []
        return copy

    def descend(self, node, builder):
        copy = helper.make_node(node.op_type, list(node.input), list(node.output),
                                name=node.name, domain=node.domain)
        for a in node.attribute:
            if a.type == AttributeProto.GRAPH:
                copy.attribute.append(helper.make_attribute(
                    a.name, self.graph(a.g, builder.child())))
            else:
                copy.attribute.append(a)
        return copy

    # ---------------------------------------------------------------- spelling ---------
    def op(self, op_type, inputs, **attrs):
        return self.b_.op(op_type, inputs, stem=op_type.lower() + "_l", **attrs)

    def ints(self, values):
        return self.b_.ints(values)

    def scalar(self, value, dtype):
        return self.b_.constant(value, dtype, ())

    def int_scalar(self, value):
        return self.scalar(value, TensorProto.INT64)

    def unsqueeze(self, value, axes):
        if self.opset >= 13:
            return self.op("Unsqueeze", [value, self.ints(axes)])
        return self.op("Unsqueeze", [value], axes=list(axes))

    def slice(self, value, start, end, axis):
        if self.opset >= 10:
            return self.op("Slice", [value, self.ints([start]), self.ints([end]),
                                     self.ints([axis])])
        return self.op("Slice", [value], starts=[start], ends=[end], axes=[axis])

    def dim(self, value, axis):
        """Dimension `axis` of `value` as an int64 scalar."""
        return self.op("Gather", [self.op("Shape", [value]), self.int_scalar(axis)], axis=0)

    def range(self, limit):
        return self.op("Range", [self.int_scalar(0), limit, self.int_scalar(1)])

    def int64(self, value):
        return self.op("Cast", [value], to=TensorProto.INT64)

    def finish(self, node, value, index=0):
        self.b_.alias(value, node.output[index])


# ------------------------------------------------------------------------ LRN ------------
# y = x / (bias + alpha/size * S)^beta, with S the sum of x^2 over a window of `size`
# channels -- floor((size-1)/2) before the channel, ceil((size-1)/2) after it.

@lowering("LRN")
def _lower_lrn(s, node):
    x = node.input[0]
    dtype = s.shapes.dtype(x)
    rank = s.shapes.rank(x)
    size = attribute(node, "size")
    alpha, beta = attribute(node, "alpha", 1e-4), attribute(node, "beta", 0.75)
    bias = attribute(node, "bias", 1.0)
    before, after = (size - 1)//2, size - 1 - (size - 1)//2
    square = s.op("Mul", [x, x])
    pads = [0]*(2*rank)
    pads[1], pads[rank + 1] = before, after
    if s.opset >= 11:
        padded = s.op("Pad", [square, s.ints(pads)], mode="constant")
    else:
        padded = s.op("Pad", [square], pads=pads, mode="constant")
    channels = (s.shapes.shape(x) or [None, None])[1]
    # a nonnegative end where the channel count is known: older shape inference gives up
    # on a negative one, and everything downstream would lose its shape
    windows = [s.slice(padded, i, i + channels if channels is not None else
                       i - (size - 1) if i < size - 1 else INT64_MAX, 1)
               for i in range(size)]
    total = windows[0] if size == 1 else s.op("Sum", windows)
    scaled = s.op("Add", [s.scalar(bias, dtype),
                          s.op("Mul", [s.scalar(alpha/size, dtype), total])])
    s.finish(node, s.op("Mul", [x, s.op("Pow", [scaled, s.scalar(-beta, dtype)])]))


# ------------------------------------------------------------------------ STFT -----------
# Frames of the signal gathered at [f*step + k], windowed, then a DFT along the frame axis.

@lowering("STFT")
def _lower_stft(s, node):
    signal, step = node.input[0], node.input[1]
    window = node.input[2] if len(node.input) > 2 and node.input[2] else None
    length = node.input[3] if len(node.input) > 3 and node.input[3] else None
    if window is not None:
        width = s.dim(window, 0)
    elif length is not None:
        width = s.int64(length)
    else:
        raise UnsupportedOperator("STFT needs a window or a frame_length")
    step = s.int64(step)
    frames = s.op("Add", [s.op("Div", [s.op("Sub", [s.dim(signal, 1), width]), step]),
                          s.int_scalar(1)])
    starts = s.op("Mul", [s.range(frames), step])
    indices = s.op("Add", [s.unsqueeze(starts, [1]), s.unsqueeze(s.range(width), [0])])
    gathered = s.op("Gather", [signal, indices], axis=1)  # [batch, frames, width, 1|2]
    if window is not None:
        gathered = s.op("Mul", [gathered, s.unsqueeze(window, [0, 1, 3])])
    onesided = attribute(node, "onesided", 1)
    if s.opset >= 20:
        result = s.op("DFT", [gathered, "", s.int_scalar(2)], onesided=onesided)
    else:
        result = s.op("DFT", [gathered], axis=2, onesided=onesided)
    s.finish(node, result)


# ------------------------------------------------------------------------ TensorScatter --
# The update written into the cache along `axis`, at write_indices[b] + j (modulo the cache
# length when circular): a ScatterElements at computed positions.

@lowering("TensorScatter")
def _lower_tensor_scatter(s, node):
    cache, update = node.input[0], node.input[1]
    written = node.input[2] if len(node.input) > 2 and node.input[2] else None
    rank = s.shapes.rank(cache)
    axis = attribute(node, "axis", -2) % rank
    if written is None:
        written = s.op("ConstantOfShape", [s.slice(s.op("Shape", [cache]), 0, 1, 0)],
                       value=helper.make_tensor("zero", TensorProto.INT64, [1], [0]))
    positions = s.op("Add", [s.unsqueeze(s.int64(written), [1]),
                             s.unsqueeze(s.range(s.dim(update, axis)), [0])])
    if attribute(node, "mode", "linear") == "circular":
        positions = s.op("Mod", [positions, s.dim(cache, axis)])
    positions = s.unsqueeze(positions, [a for a in range(rank) if a not in (0, axis)])
    positions = s.op("Expand", [positions, s.op("Shape", [update])])
    s.finish(node, s.op("ScatterElements", [cache, positions, update], axis=axis))


# ------------------------------------------------------------------------ GridSample -----
CUBIC_ALPHA = -0.75


@lowering("GridSample")
def _lower_grid_sample(s, node):
    x, grid = node.input[0], node.input[1]
    static = s.shapes.static(x)
    if static is None or None in static[2:]:
        raise UnsupportedOperator("GridSample needs the input's spatial shape declared")
    dims = [int(d) for d in static[2:]]
    mode = {"bilinear": "linear", "bicubic": "cubic"}.get(
        attribute(node, "mode", "linear" if s.opset >= 20 else "bilinear"),
        attribute(node, "mode", "linear" if s.opset >= 20 else "bilinear"))
    padding = attribute(node, "padding_mode", "zeros")
    aligned = attribute(node, "align_corners", 0)
    dtype = s.shapes.dtype(grid)
    c = lambda v: s.scalar(float(v), dtype)
    count = len(dims)

    taps = []  # per spatial axis: [(int64 index [N, 1, P], weight or None), ...]
    for i, d in enumerate(dims):
        low, high = (0.0, d - 1.0) if aligned else (-0.5, d - 0.5)
        n = s.op("Reshape", [s.op("Gather", [grid, s.int_scalar(count - 1 - i)], axis=-1),
                             s.ints([0, 1, -1])])
        shifted = s.op("Add", [n, c(1)])
        if aligned:
            coordinate = s.op("Mul", [shifted, c((d - 1)/2)])
        else:
            coordinate = s.op("Div", [s.op("Sub", [s.op("Mul", [shifted, c(d)]), c(1)]), c(2)])
        if mode == "nearest":
            coordinate = s.op("Round", [coordinate])
        outside = s.op("Or", [s.op("Less", [coordinate, c(low)]),
                              s.op("Greater", [coordinate, c(high)])])
        # ONNX Runtime and PyTorch fold only the taps of a cubic, not its coordinate
        if padding == "border" and mode != "cubic":
            coordinate = s.op("Where", [outside, s.op("Clip", [coordinate, c(0), c(d - 1)]),
                                        coordinate])
        elif padding == "reflection" and mode != "cubic":
            coordinate = s.op("Where", [outside, _reflect(s, coordinate, low, high, c),
                                        coordinate])
        taps.append([_tap(s, index, weight, d, low, high, padding, c, dtype)
                     for index, weight in _taps(s, coordinate, mode, c)])

    # one gather per combination of taps, weighted by the product of the tap weights
    flat = s.op("Reshape", [x, s.ints([0, 0, -1])])  # [N, C, prod(dims)]
    strides = [int(v) for v in reversed(list(itertools.accumulate(
        [1] + dims[::-1][:-1], lambda a, b: a*b)))]
    target = None
    terms = []
    for combination in itertools.product(*taps):
        position, weight = None, None
        for (index, tap_weight), stride in zip(combination, strides):
            term = s.op("Mul", [index, s.int_scalar(stride)]) if stride != 1 else index
            position = term if position is None else s.op("Add", [position, term])
            if tap_weight is not None:
                weight = tap_weight if weight is None else s.op("Mul", [weight, tap_weight])
        if target is None:
            target = s.op("Concat", [s.slice(s.op("Shape", [x]), 0, 2, 0),
                                     s.slice(s.op("Shape", [position]), 2, 3, 0)], axis=0)
        value = s.op("GatherElements", [flat, s.op("Expand", [position, target])], axis=2)
        if weight is not None:
            if s.shapes.dtype(x) != dtype:
                weight = s.op("Cast", [weight], to=s.shapes.dtype(x))
            value = s.op("Mul", [value, weight])
        terms.append(value)
    total = terms[0] if len(terms) == 1 else s.op("Sum", terms)
    shape = s.op("Concat", [s.slice(s.op("Shape", [x]), 0, 2, 0),
                            s.slice(s.op("Shape", [grid]), 1, -1, 0)], axis=0)
    s.finish(node, s.op("Reshape", [total, shape]))


def _reflect(s, value, low, high, c):
    """Fold a coordinate back inside [low, high] by reflecting at the borders, as often as
    it takes -- the reference implementation's `_gs_reflect`."""
    extent = high - low

    def fold(distance, near, far):
        periods = s.op("Floor", [s.op("Div", [distance, c(extent)])])
        rest = s.op("Sub", [distance, s.op("Mul", [periods, c(extent)])])
        even = s.op("Equal", [s.op("Sub", [periods, s.op("Mul", [c(2), s.op(
            "Floor", [s.op("Div", [periods, c(2)])])])]), c(0)])
        toward = s.op("Add", [c(near), rest]) if near < far else s.op("Sub", [c(near), rest])
        back = s.op("Sub", [c(far), rest]) if near < far else s.op("Add", [c(far), rest])
        return s.op("Where", [even, toward, back])

    below = fold(s.op("Sub", [c(low), value]), low, high)
    above = fold(s.op("Sub", [value, c(high)]), high, low)
    return s.op("Where", [s.op("Less", [value, c(low)]), below,
                          s.op("Where", [s.op("Greater", [value, c(high)]), above, value])])


def _taps(s, coordinate, mode, c):
    """(float index, weight) for each tap the interpolation mode reads along one axis."""
    if mode == "nearest":
        return [(coordinate, None)]
    base = s.op("Floor", [coordinate])
    t = s.op("Sub", [coordinate, base])
    at = lambda k: s.op("Add", [base, c(k)]) if k else base
    if mode == "linear":
        return [(base, s.op("Sub", [c(1), t])), (at(1), t)]
    if mode != "cubic":
        raise UnsupportedOperator("GridSample mode %r is not supported" % mode)
    a = CUBIC_ALPHA

    def outer(u):   # |u| in [1, 2]: ((a u - 5a) u + 8a) u - 4a
        return s.op("Sub", [s.op("Mul", [s.op("Add", [s.op("Mul", [s.op(
            "Sub", [s.op("Mul", [c(a), u]), c(5*a)]), u]), c(8*a)]), u]), c(4*a)])

    def inner(u):   # |u| in [0, 1]: ((a + 2) u - (a + 3)) u^2 + 1
        return s.op("Add", [s.op("Mul", [s.op("Sub", [s.op("Mul", [c(a + 2), u]), c(a + 3)]),
                                         s.op("Mul", [u, u])]), c(1)])

    one_minus = s.op("Sub", [c(1), t])
    return [(at(-1), outer(s.op("Add", [t, c(1)]))), (base, inner(t)),
            (at(1), inner(one_minus)), (at(2), outer(s.op("Sub", [c(2), t])))]


def _tap(s, index, weight, d, low, high, padding, c, dtype):
    """A tap's (int64 index inside the axis, weight). Out of range, `border` reads the edge,
    `reflection` the mirrored position, and `zeros` nothing: its weight is masked instead."""
    if padding == "reflection":
        index = _reflect(s, index, low, high, c)
    integer = s.int64(s.op("Clip", [index, c(0), c(d - 1)]))
    if padding == "zeros":
        inside = s.op("And", [s.op("GreaterOrEqual", [index, c(0)]),
                              s.op("LessOrEqual", [index, c(d - 1)])])
        mask = s.op("Cast", [inside], to=dtype)
        weight = mask if weight is None else s.op("Mul", [weight, mask])
    return integer, weight
