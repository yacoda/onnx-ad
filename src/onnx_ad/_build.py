"""Emission helpers and the per-graph tape shared by the forward and reverse passes.

Both passes carry one derivative tensor per differentiated value, laid out as the primal
tensor with **one extra trailing axis holding the seed directions** -- the tangent of a
value of shape `[3, 4]` has shape `[3, 4, nfwd]`. A trailing axis is what CasADi's 2-D
reading of an ONNX tensor expects (`fwd_x` of a vector input is an `n`-by-`nfwd` matrix),
and it is also what ordinary right-aligned broadcasting wants: `[3, 4, nfwd]` combines with
the primal `[3, 4]` as soon as the primal is unsqueezed to `[3, 4, 1]`, which is what
`Context.lift` does and caches.

Derivative tensors are allowed to stay *under-broadcast* while they travel: an operand that
was itself broadcast contributes a tangent of its own (smaller) shape, and only the graph
outputs are materialized to the full shape with `Context.full`. That keeps the emitted
graph close to the size of the primal one.
"""
from collections import ChainMap
from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ._graph import all_constants, all_names, walk

# Element types we differentiate; everything else rides along as a constant
FLOAT_TYPES = (TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16, TensorProto.BFLOAT16)
NUMPY_OF = {TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64,
            TensorProto.FLOAT16: np.float16, TensorProto.INT64: np.int64,
            TensorProto.INT32: np.int32, TensorProto.BOOL: np.bool_}
INT64_MAX = np.iinfo(np.int64).max


class UnsupportedOperator(Exception):
    """A differentiated value reached an operation with no rule."""


def attribute(node, name, default=None) -> Any:
    """An attribute's value, with string attributes decoded -- onnx hands those back as
    bytes, and a rule comparing one against "tanh" or "SAME_UPPER" would silently never
    match, taking a wrong branch rather than failing."""
    value = next((helper.get_attribute_value(a) for a in node.attribute if a.name == name),
                 default)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, list) and value and isinstance(value[0], bytes):
        return [v.decode() for v in value]
    return value


class Builder:
    """Accumulates nodes and initializers with unique names.

    A child builder, for a subgraph, shares the name allocator with its parent -- a subgraph
    name that collides with an outer one shadows it silently -- but keeps its own nodes and
    its own constants. Constants have to be *local*: ONNX shape inference does not treat an
    outer-scope initializer as constant data inside a subgraph, so an `Unsqueeze` whose axes
    come from outside gets no inferred shape there, and a later pass differentiating this
    model again could not place its seed axis.
    """

    def __init__(self, taken=(), parent=None):
        self.nodes = []
        self.initializers = []
        self._constants = {}
        if parent is None:
            self._taken = set(taken)
            self._counter = [0]
        else:
            self._taken = parent._taken
            self._counter = parent._counter

    def child(self):
        return Builder(parent=self)

    def name(self, stem):
        if stem not in self._taken:  # keep the natural name when it is free
            self._taken.add(stem)
            return stem
        while True:
            self._counter[0] += 1
            candidate = "%s_%d" % (stem, self._counter[0])
            if candidate not in self._taken:
                self._taken.add(candidate)
                return candidate

    def op(self, op_type, inputs, outputs=1, stem=None, **attrs):
        """Append a node; returns its output name, or the list when outputs > 1."""
        names = [self.name(stem or op_type.lower()) for _ in range(outputs)]
        self.nodes.append(helper.make_node(op_type, list(inputs), names, **attrs))
        return names[0] if outputs == 1 else names

    def alias(self, value, name):
        """Bind an exact, already reserved output name to an existing tensor."""
        self.nodes.append(helper.make_node("Identity", [value], [name]))
        return name

    def constant(self, value, dtype, shape=None):
        """An initializer, shared between uses."""
        array = np.asarray(value, dtype=NUMPY_OF[dtype])
        if shape is not None:
            array = array.reshape(shape)
        key = (dtype, array.shape, array.tobytes())
        if key not in self._constants:
            name = self.name("const")
            self.initializers.append(numpy_helper.from_array(array, name))
            self._constants[key] = name
        return self._constants[key]

    def ints(self, values):
        values = list(values)
        return self.constant(values, TensorProto.INT64, (len(values),))


class Shapes:
    """Element type, rank and (where declared) static shape of every value in a graph."""

    def __init__(self, model):
        try:
            from onnx import shape_inference
            inferred = shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:  # shape inference is a convenience, never a requirement
            inferred = model
        self._type = {}
        self._shape = {}
        for tensor in model.graph.initializer:
            self._type[tensor.name] = tensor.data_type
            self._shape[tensor.name] = tuple(tensor.dims)
        for top in (inferred.graph, model.graph):
            for graph in walk(top):  # subgraph bodies declare their own values
                for tensor in graph.initializer:
                    self._type.setdefault(tensor.name, tensor.data_type)
                    self._shape.setdefault(tensor.name, tuple(tensor.dims))
                for value in list(graph.input) + list(graph.output) + list(graph.value_info):
                    self._absorb(value)

    def _absorb(self, value):
        tensor_type = value.type.tensor_type
        if not tensor_type.elem_type:
            return
        self._type.setdefault(value.name, tensor_type.elem_type)
        if not tensor_type.HasField("shape") or value.name in self._shape:
            return
        self._shape[value.name] = tuple(
            d.dim_value if d.WhichOneof("value") == "dim_value" else None for d in
            tensor_type.shape.dim)

    def dtype(self, name):
        return self._type.get(name, TensorProto.DOUBLE)

    def shape(self, name):
        """The declared shape, with None for symbolic dimensions; None when unknown."""
        return self._shape.get(name)

    def rank(self, name):
        shape = self._shape.get(name)
        if shape is None:
            raise UnsupportedOperator(
                "the rank of '%s' is unknown; run onnx.shape_inference on the model, or "
                "declare the shape, so the seed axis can be placed" % name)
        return len(shape)

    def declare(self, name, shape, dtype=None):
        """Record the shape of a tensor the pass itself emits (promoted operands)."""
        self._shape[name] = tuple(shape)
        if dtype is not None:
            self._type[name] = dtype

    def static(self, name):
        """The declared shape when every dimension is known, else None."""
        shape = self._shape.get(name)
        return shape if shape is not None and None not in shape else None


class Context:
    """The tape: derivative tensor per value, plus the seed-axis algebra the rules use."""

    def __init__(self, model):
        graph = model.graph
        self.root = self
        self.parent = None
        self.b = Builder(all_names(graph))
        self.shapes = Shapes(model)
        self.opset = max([o.version for o in model.opset_import
                          if o.domain in ("", "ai.onnx")] or [18])
        self.values = all_constants(graph)  # initializers and Constants, any depth
        self.derivative = {}        # value name -> derivative tensor name
        self.wanted = None          # operands a reverse rule's caller will actually use
        self.depends = set()        # reverse: values that depend on differentiated inputs
        self._seeds = []            # seed tensors, any one of which carries the seed count
        self._count = None
        self._lift = {}
        self._full = {}
        self._shape_of = {}
        self._replacement = None    # forward: a node this step's rule substitutes
        self._replacements = {}     # reverse: primal nodes a rule substitutes, by identity

    def child(self):
        """A scope for a subgraph.

        Reads fall through to the enclosing scopes -- an outer tensor, derivative or cached
        helper is in scope inside a subgraph -- but writes stay local, because a tensor
        defined inside a subgraph does not exist outside it.
        """
        scope = object.__new__(Context)
        scope.root = self.root
        scope.parent = self
        scope.b = self.b.child()
        scope.shapes = self.shapes
        scope.opset = self.opset
        scope.values = self.values
        scope.derivative = ChainMap({}, self.derivative)
        scope.wanted = None
        scope.depends = set()
        scope._lift = ChainMap({}, self._lift)
        scope._full = ChainMap({}, self._full)
        scope._shape_of = ChainMap({}, self._shape_of)
        scope._replacement = None
        scope._replacements = {}
        return scope

    # --- substituting a node (control flow extends the primal node it differentiates) ---
    def replace(self, node):
        """Forward: emit `node` in place of the primal node being differentiated."""
        self._replacement = node

    def replace_primal(self, original, node):
        """Reverse: have the primal pass run `node` instead of `original` (to tape it).

        Keyed by identity, with the original kept alongside: protobuf proxies are only
        stable while something holds them, and a reused `id` must not match."""
        self._replacements[id(original)] = (original, node)

    def value_info(self, name, primal, seeded=True):
        """A subgraph input or output typed like `primal`, plus a seed axis if seeded."""
        shape = self.shapes.shape(primal)
        if shape is not None:
            shape = list(shape) + ([None] if seeded else [])
        return helper.make_tensor_value_info(name, self.shapes.dtype(primal), shape)

    # --- value queries ---
    def dtype(self, *names):
        for name in names:
            if name and self.shapes.dtype(name) in FLOAT_TYPES:
                return self.shapes.dtype(name)
        return TensorProto.DOUBLE

    def rank(self, name):
        return self.shapes.rank(name)

    def constant(self, value, dtype):
        return self.b.constant(value, dtype)

    def asked_for(self, name):
        """Whether a reverse contribution to `name` is wanted at all.

        The driver discards a contribution to a value that does not depend on the
        differentiated inputs, but an expensive rule should not build it in the first
        place -- a Conv over a constant weight is a whole extra convolution.
        """
        return self.wanted is None or name in self.wanted

    def integers(self, name):
        """A constant integer operand (axes, perm, split sizes) as a list, else None."""
        array = self.values.get(name)
        return None if array is None else [int(v) for v in array.reshape(-1)]

    # --- opset-dependent spellings ---
    def unsqueeze(self, value, axes):
        if self.opset >= 13:
            return self.b.op("Unsqueeze", [value, self.b.ints(axes)])
        return self.b.op("Unsqueeze", [value], axes=list(axes))

    def squeeze(self, value, axes):
        if self.opset >= 13:
            return self.b.op("Squeeze", [value, self.b.ints(axes)])
        return self.b.op("Squeeze", [value], axes=list(axes))

    def reduce_sum(self, value, axes, keepdims=1):
        """ReduceSum with a possibly dynamic axes tensor (an empty one means: reduce nothing)."""
        if self.opset >= 13:
            if not isinstance(axes, str):
                axes = self.b.ints(axes)
            return self.b.op("ReduceSum", [value, axes], keepdims=keepdims,
                             noop_with_empty_axes=1)
        return self.b.op("ReduceSum", [value], axes=list(axes), keepdims=keepdims)

    def reduce_mean(self, value, axes, keepdims=1):
        """ReduceMean took axes as an input only from opset 18."""
        if self.opset >= 18:
            return self.b.op("ReduceMean", [value, self.b.ints(axes)], keepdims=keepdims)
        return self.b.op("ReduceMean", [value], axes=list(axes), keepdims=keepdims)

    # --- the seed axis ---
    def add_seed(self, name):
        self.root._seeds.append(name)

    def count(self):
        """An int64 [1] tensor holding the seed count, read off a seed tensor's last axis.

        Always built in the outermost graph, from a top-level seed: every subgraph then
        reads the same tensor by capture.
        """
        root = self.root
        if root._count is None:
            if not root._seeds:
                raise UnsupportedOperator("no seed tensors: nothing is being differentiated")
            shape = root.b.op("Shape", [root._seeds[0]], stem="seed_shape")
            root._count = root.b.op(
                "Slice", [shape, root.b.ints([-1]), root.b.ints([INT64_MAX])], stem="nseed")
        return root._count

    def shape_of(self, name):
        if name not in self._shape_of:
            self._shape_of[name] = self.b.op("Shape", [name], stem=name + "_shape")
        return self._shape_of[name]

    def seeded_shape(self, name):
        """An int64 tensor holding shape(name) ++ [seed count]."""
        return self.b.op("Concat", [self.shape_of(name), self.count()], axis=0, stem="seeded")

    def lift(self, name):
        """The primal value with a trailing singleton axis, so it broadcasts against seeds."""
        if name not in self._lift:
            self._lift[name] = self.unsqueeze(name, [-1])
        return self._lift[name]

    def full(self, value, primal):
        """Materialize a derivative tensor to exactly shape(primal) ++ [seed count]."""
        key = (value, primal)
        if key in self._full:
            return self._full[key]
        static = self.shapes.static(primal)
        target = (self.b.ints(list(static) + [1]) if static is not None else
                  self.b.op("Concat", [self.shape_of(primal), self.b.ints([1])], axis=0,
                            stem="target"))
        self._full[key] = self.b.op("Expand", [value, target], stem="full")
        return self._full[key]

    def reshape_like(self, value, primal):
        """Reinterpret a derivative tensor as shape(primal) ++ [seed count]."""
        static = self.shapes.static(primal)
        target = (self.b.ints(list(static) + [-1]) if static is not None else
                  self.b.op("Concat", [self.shape_of(primal), self.b.ints([-1])], axis=0,
                            stem="like"))
        return self.b.op("Reshape", [value, target])

    def zeros(self, primal):
        """A zero derivative of shape(primal) ++ [seed count]."""
        dtype = self.dtype(primal)
        value = helper.make_tensor("zero", dtype, [1], [0])
        return self.b.op("ConstantOfShape", [self.seeded_shape(primal)], value=value,
                         stem="zeros")

    # --- CasADi's packed seed layout ---
    # CasADi reads an ONNX tensor as a matrix (rank 0/1 as a column, rank 2 directly, higher
    # ranks flattened to a column) and wants the seeds of an `r`-by-`c` value as a single
    # `r`-by-`(nseed*c)` matrix, seed-major. That is our `shape ++ [nseed]` with the seed axis
    # moved next to the column axis -- a transpose and a reshape, and nothing at all for the
    # rank-1 values that a CasADi-facing model should use at its boundary anyway.

    def _rows_and_columns(self, primal):
        """The two shape tensors (or None when static) CasADi's matrix reading uses."""
        shape = self.shapes.shape(primal)
        if shape is None:
            raise UnsupportedOperator(
                "the CasADi layout needs the declared shape of '%s'" % primal)
        return shape

    def pack(self, value, primal):
        """Our layout -> CasADi's matrix of seeds."""
        rank = self.rank(primal)
        if rank == 1:
            return value                       # [n, nseed] is already what CasADi wants
        if rank == 0:
            return self.b.op("Reshape", [value, self.b.ints([1, -1])], stem="packed")
        if rank > 2:
            target = self.b.op("Concat", [self.b.ints([-1]), self.count()], axis=0,
                               stem="flat")
            return self.b.op("Reshape", [value, target], stem="packed")
        rows = self.shapes.shape(primal)[0]
        target = (self.b.ints([rows, -1]) if rows is not None else self.b.op(
            "Concat", [self.b.op("Slice", [self.shape_of(primal), self.b.ints([0]),
                                           self.b.ints([1])]), self.b.ints([-1])],
            axis=0, stem="rows"))
        return self.b.op("Reshape", [self.b.op("Transpose", [value], perm=[0, 2, 1]), target],
                         stem="packed")

    def unpack(self, value, primal):
        """CasADi's matrix of seeds -> our layout."""
        rank = self.rank(primal)
        if rank == 1:
            return value
        if rank == 0:
            return self.b.op("Reshape", [value, self.b.ints([-1])], stem="unpacked")
        if rank > 2:
            return self.reshape_like(value, primal)  # numel*nseed / numel infers nseed
        rows, columns = self.shapes.shape(primal)
        if rows is not None and columns is not None:
            target = self.b.ints([rows, -1, columns])
        else:
            shape = self.shape_of(primal)
            slice_ = lambda a, b: self.b.op("Slice", [shape, self.b.ints([a]),
                                                      self.b.ints([b])])
            target = self.b.op("Concat", [slice_(0, 1), self.b.ints([-1]), slice_(1, 2)],
                               axis=0, stem="unpacked_shape")
        return self.b.op("Transpose", [self.b.op("Reshape", [value, target])],
                         perm=[0, 2, 1], stem="unpacked")

    def sum(self, values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return self.b.op("Sum", values, stem="acc")

    # --- broadcasting ---
    def unbroadcast(self, value, source_rank, target):
        """Reduce an adjoint contribution of rank `source_rank` (+ seed) to target's shape.

        Reverse mode is where broadcasting bites: a contribution arrives shaped like the
        *result* of the operation and must be summed back over the axes the operand was
        broadcast along. Where the operand's shape is declared this is a static axis list;
        where it is symbolic the axes are computed at run time, so dynamic dimensions that
        happen to be 1 are still handled.
        """
        target_rank = self.rank(target)
        lead = source_rank - target_rank
        if lead < 0:
            raise UnsupportedOperator(
                "'%s' has rank %d, more than the rank %d of the value it feeds"
                % (target, target_rank, source_rank))
        static = self.shapes.static(target)
        if static is not None:
            axes = list(range(lead)) + [lead + i for i, d in enumerate(static) if d == 1]
            if not axes:
                return value
            reduced = self.reduce_sum(value, axes, keepdims=1)
            if lead == 0:
                return reduced
            return self.b.op("Reshape", [reduced, self.b.ints(list(static) + [-1])])
        shape = self.shape_of(target)
        padded = shape if lead == 0 else self.b.op(
            "Concat", [self.b.ints([1]*lead), shape], axis=0, stem="padded")
        mask = self.b.op("Equal", [padded, self.b.ints([1])])
        axes = self.squeeze(self.b.op("NonZero", [mask]), [0])
        reduced = self.reduce_sum(value, axes, keepdims=1)
        return self.b.op("Reshape", [reduced, self.b.op(
            "Concat", [shape, self.b.ints([-1])], axis=0, stem="restore")])


def diff_prefix(names, kind):
    """CasADi's derivative-prefix rule, applied to a model's own tensor names.

    `FunctionInternal::diff_prefix` scans the existing input and output names for the
    highest `<kind><n>_` index and returns the next one: `fwd_`, then `fwd2_`, `fwd3_`. A
    model carrying `fwd_x` therefore yields `fwd2_`, so repeated differentiation names
    itself, and `forward(reverse(m))` still yields `fwd_` because `adj_` is a different
    family. The matching seed dimension is the prefix with `n` in front: `nfwd`, `nfwd2`.
    """
    highest = 0
    for name in names:
        end = name.find("_")
        if end < len(kind) or not name.startswith(kind):
            continue
        if end == len(kind):
            index = 1
        else:
            try:
                index = int(name[len(kind):end])
            except ValueError:
                continue
        highest = max(highest, index)
    return kind + "_" if highest == 0 else "%s%d_" % (kind, highest + 1)


def conventions(model, kind, prefix, dim):
    """The prefix and seed-dimension name for a pass, defaulting to CasADi's."""
    graph = model.graph
    if prefix is None:
        prefix = diff_prefix([v.name for v in list(graph.input) + list(graph.output)], kind)
    return prefix, dim if dim is not None else "n" + prefix[:-1]


def select(values, names, kind):
    """The values to differentiate: the ones named, else every floating-point one."""
    by_name = {value.name: value for value in values}
    if names is None:
        chosen = [v for v in values if v.type.tensor_type.elem_type in FLOAT_TYPES]
        if not chosen:
            raise ValueError("the graph has no floating-point %s to differentiate" % kind)
        return chosen
    chosen = []
    for name in names:
        if name not in by_name:
            raise ValueError("'%s' is not a graph %s" % (name, kind))
        if by_name[name].type.tensor_type.elem_type not in FLOAT_TYPES:
            raise ValueError("%s '%s' is not of a floating-point type" % (kind, name))
        chosen.append(by_name[name])
    return chosen


def rename(builder, stem):
    """A name that must come out exactly as asked: the conventions depend on it."""
    name = builder.name(stem)
    if name != stem:
        raise ValueError("'%s' is already used in the graph; it is the name the derivative "
                         "convention needs" % stem)
    return name


def assemble(result, ctx, nodes, seed_inputs, derivative_outputs):
    """Install the node list, and append initializers, seed inputs and derivative outputs."""
    graph = result.graph
    graph.ClearField("node")
    graph.node.extend(nodes)
    graph.initializer.extend(ctx.b.initializers)
    graph.input.extend(seed_inputs)
    graph.output.extend(derivative_outputs)
    graph.ClearField("value_info")  # shapes changed wherever a seed axis was added
    return result


def seeded_value_info(value, name, dim, layout="onnx"):
    """The type of a seed tensor: `shape ++ [dim]`, or CasADi's packed matrix.

    In the CasADi layout the seed count multiplies the value's *column* count, and a product
    of two symbolic extents is not expressible in ONNX -- so the packed axis gets a composed
    parameter name (`nfwd_nadj`), which is exactly the "packed columns such as nadj * nfwd"
    that CasADi's derivative wrapper infers a binding for.
    """
    out = type(value)()
    out.CopyFrom(value)
    out.name = name
    dims = [(d.dim_value if d.WhichOneof("value") == "dim_value" else
             (d.dim_param or None)) for d in out.type.tensor_type.shape.dim]
    if layout == "onnx":
        out.type.tensor_type.shape.dim.add().dim_param = dim
        return out
    if len(dims) == 0:
        packed = [1, dim]
    elif len(dims) == 1:
        packed = [dims[0], dim]
    elif len(dims) == 2:
        columns = dims[1]
        packed = [dims[0], dim if columns == 1 else
                  "%s_%s" % (dim, columns if isinstance(columns, str) else "x%d" % columns)]
    else:
        known = [d for d in dims if isinstance(d, int)]
        rows = None
        if len(known) == len(dims):
            rows = 1
            for d in dims:
                rows *= d
        packed = [rows if rows is not None else "%s_rows" % name, dim]
    out.type.tensor_type.ClearField("shape")
    for value_ in packed:
        added = out.type.tensor_type.shape.dim.add()
        if isinstance(value_, int):
            added.dim_value = value_
        else:
            added.dim_param = value_
    return out
