"""Structure, pooling and normalization rules added after 0.2.0.

Each case runs through the shared harness: forward against reverse, finite differences of
the primal, and the closure check that every derivative model uses only operations with
rules. Where ONNX Runtime has no double kernel the case drops to float32.
"""
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from onnx_ad.lower import LOWERINGS, lower
from test_ad import IR_VERSION, MAX_OPSET, JacobianCase, run

RNG = np.random.default_rng(17)
OLD_ORT = tuple(int(v) for v in ort.__version__.split(".")[:2]) < (1, 19)
try:
    CUMPROD = onnx.defs.get_schema("CumProd").since_version
except Exception:
    CUMPROD = 10**6  # not in this onnx


def build(nodes, x_shape, y_shape, initializers=(), opset=18, dtype=TensorProto.DOUBLE,
          outputs=None):
    outputs = outputs or [helper.make_tensor_value_info("y", dtype, y_shape)]
    graph = helper.make_graph(nodes if isinstance(nodes, list) else [nodes], "g",
                              [helper.make_tensor_value_info("x", dtype, x_shape)], outputs,
                              list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def arr(name, value, dtype=np.float64):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def runs(model, feeds):
    try:
        run(model, feeds)
        return True
    except Exception:
        return False


class Case(JacobianCase):
    """The shared harness; test classes derive from it rather than from each other."""

    def case(self, nodes, x_shape, y_shape, initializers=(), opset=18, x=None, reference=None,
             outputs=None):
        """Double where the runtime can, float32 with a coarse step where it cannot."""
        if opset > MAX_OPSET:
            self.skipTest("opset %d is newer than this onnx" % opset)
        x = RNG.standard_normal(x_shape) if x is None else x
        model = build(nodes, x_shape, y_shape, initializers, opset, outputs=outputs)
        if runs(model, {"x": x}):
            return self.check(model, {"x": x}, reference, fd_tol=1e-5)
        single = [numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)
                  if numpy_helper.to_array(t).dtype == np.float64 else t for t in initializers]
        model = build(nodes, x_shape, y_shape, single, opset, dtype=TensorProto.FLOAT,
                      outputs=[helper.make_tensor_value_info(o.name, TensorProto.FLOAT if
                                                             o.type.tensor_type.elem_type ==
                                                             TensorProto.DOUBLE else
                                                             o.type.tensor_type.elem_type,
                                                             [d.dim_value for d in
                                                              o.type.tensor_type.shape.dim])
                               for o in outputs] if outputs else None)
        x = x.astype(np.float32)
        if not runs(model, {"x": x}):
            self.skipTest("does not run in this ONNX Runtime")
        return self.check(model, {"x": x}, None, rtol=5e-5, step=1e-3, fd_tol=5e-3)


class MoreOpTests(Case):
    def test_trilu(self):
        for upper in (0, 1):
            with self.subTest(upper=upper):
                self.case(helper.make_node("Trilu", ["x", "k"], ["y"], upper=upper),
                          [2, 3, 3], [2, 3, 3], [arr("k", 1, np.int64)])

    def test_reverse_sequence(self):
        self.case(helper.make_node("ReverseSequence", ["x", "lens"], ["y"], batch_axis=1,
                                   time_axis=0),
                  [4, 2, 3], [4, 2, 3], [arr("lens", [4, 2], np.int64)])

    def test_mod(self):
        x = RNG.standard_normal((2, 3))*3
        self.case(helper.make_node("Mod", ["x", "b"], ["y"], fmod=1), [2, 3], [2, 3],
                  [arr("b", [1.3, -0.7, 2.1])], x=x)

    def test_mod_divisor(self):
        nodes = [helper.make_node("Mod", ["a", "x"], ["y"], fmod=1)]
        self.case(nodes, [3], [3], [arr("a", [4.3, -2.9, 5.5])], x=np.array([1.3, 0.7, 2.1]))

    def test_cumprod(self):
        x = np.abs(RNG.standard_normal(4)) + 0.5
        for reverse in (0, 1):
            with self.subTest(reverse=reverse):
                self.case(helper.make_node("CumProd", ["x", "a"], ["y"], reverse=reverse),
                          [4], [4], [arr("a", 0, np.int64)], x=x, opset=CUMPROD)

    def test_topk(self):
        outputs = [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [2, 2]),
                   helper.make_tensor_value_info("i", TensorProto.INT64, [2, 2])]
        self.case(helper.make_node("TopK", ["x", "k"], ["y", "i"], axis=1),
                  [2, 4], [2, 2], [arr("k", [2], np.int64)], outputs=outputs)

    def test_compress(self):
        for axis in (None, 1):
            with self.subTest(axis=axis):
                extra = {} if axis is None else {"axis": axis}
                shape = [3] if axis is None else [2, 2]  # the condition keeps 3 elements
                self.case(helper.make_node("Compress", ["x", "c"], ["y"], **extra),
                          [2, 3], shape, [arr("c", [True, False, True, True] if axis is None
                                              else [True, False, True], np.bool_)])

    def test_depth_to_space_and_back(self):
        for op, mode, x_shape, y_shape in (("DepthToSpace", "DCR", [1, 8, 2, 3], [1, 2, 4, 6]),
                                           ("DepthToSpace", "CRD", [1, 8, 2, 3], [1, 2, 4, 6]),
                                           ("SpaceToDepth", None, [1, 2, 4, 6], [1, 8, 2, 3])):
            with self.subTest(op=op, mode=mode):
                extra = {"mode": mode} if mode else {}
                self.case(helper.make_node(op, ["x"], ["y"], blocksize=2, **extra),
                          x_shape, y_shape)

    def test_global_pools(self):
        for op in ("GlobalAveragePool", "GlobalMaxPool"):
            with self.subTest(op=op):
                self.case(helper.make_node(op, ["x"], ["y"]), [2, 3, 4, 5], [2, 3, 1, 1])

    def test_global_lp_pool(self):
        x = RNG.standard_normal((2, 3, 4)) + 0.3
        for p in (1, 2, 3):
            with self.subTest(p=p):
                self.case(helper.make_node("GlobalLpPool", ["x"], ["y"], p=p), [2, 3, 4],
                          [2, 3, 1], x=x)

    def test_lp_normalization(self):
        for p in (1, 2):
            with self.subTest(p=p):
                self.case(helper.make_node("LpNormalization", ["x"], ["y"], axis=1, p=p),
                          [2, 4], [2, 4])

    def test_instance_normalization(self):
        self.case(helper.make_node("InstanceNormalization", ["x", "s", "b"], ["y"]),
                  [2, 3, 5], [2, 3, 5], [arr("s", RNG.standard_normal(3)),
                                         arr("b", RNG.standard_normal(3))])


class PoolingTests(Case):
    def test_max_pool(self):
        for strides, pads in (([2, 2], [0, 0, 0, 0]), ([1, 1], [1, 1, 1, 1])):
            with self.subTest(strides=strides, pads=pads):
                shape = [1, 2, 2, 2] if strides == [2, 2] else [1, 2, 4, 4]
                self.case(helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2] if
                                           strides == [2, 2] else [3, 3], strides=strides,
                                           pads=pads), [1, 2, 4, 4], shape)

    def test_max_pool_with_its_indices(self):
        outputs = [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [1, 2, 2, 2]),
                   helper.make_tensor_value_info("i", TensorProto.INT64, [1, 2, 2, 2])]
        self.case(helper.make_node("MaxPool", ["x"], ["y", "i"], kernel_shape=[2, 2],
                                   strides=[2, 2]), [1, 2, 4, 4], [1, 2, 2, 2], outputs=outputs)

    def test_average_pool(self):
        for include, pads in ((0, [1, 1, 1, 1]), (1, [1, 1, 1, 1]), (0, [0, 0, 0, 0])):
            with self.subTest(count_include_pad=include, pads=pads):
                out = [1, 2, 4, 4] if any(pads) else [1, 2, 2, 2]
                self.case(helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3]
                                           if any(pads) else [2, 2], strides=[1, 1] if
                                           any(pads) else [2, 2], pads=pads,
                                           count_include_pad=include), [1, 2, 4, 4], out)

    def test_average_pool_one_dimensional(self):
        self.case(helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3], strides=[2]),
                  [2, 3, 7], [2, 3, 3])

    def test_lp_pool(self):
        x = RNG.standard_normal((1, 2, 4, 4)) + 0.2
        for p in (1, 2, 3):
            with self.subTest(p=p):
                self.case(helper.make_node("LpPool", ["x"], ["y"], kernel_shape=[2, 2],
                                           strides=[2, 2], p=p), [1, 2, 4, 4], [1, 2, 2, 2],
                          x=x)

    def test_a_small_cnn(self):
        w = RNG.standard_normal((4, 2, 3, 3))
        nodes = [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1],
                                  kernel_shape=[3, 3]),
                 helper.make_node("Relu", ["c"], ["r"]),
                 helper.make_node("MaxPool", ["r"], ["p"], kernel_shape=[2, 2], strides=[2, 2]),
                 helper.make_node("GlobalAveragePool", ["p"], ["y"])]
        self.case(nodes, [1, 2, 4, 4], [1, 4, 1, 1], [arr("w", w)])


class EinsumTests(Case):
    CASES = [
        # equation, shape of x, shape of the other operand (None: x alone), output shape
        ("ij,jk->ik", [2, 3], [3, 4], [2, 4]),
        ("bij,bjk->bik", [2, 3, 4], [2, 4, 2], [2, 3, 2]),
        ("...ij,...jk->...ik", [2, 3, 4], [2, 4, 2], [2, 3, 2]),
        ("bhqd,bhkd->bhqk", [1, 2, 3, 4], [1, 2, 5, 4], [1, 2, 3, 5]),  # attention scores
        ("i,j->ij", [3], [4], [3, 4]),                                   # outer product
        ("i,i->", [3], [3], []),                                         # dot product
        ("ij->ji", [2, 3], None, [3, 2]),                                # transpose
        ("ij->i", [2, 3], None, [2]),                                    # summed index
        ("ij", [2, 3], None, [2, 3]),                                    # implicit output
        ("ij,jk", [2, 3], [3, 4], [2, 4]),                               # implicit, contracted
    ]

    def test_equations(self):
        for equation, x_shape, other, y_shape in self.CASES:
            with self.subTest(equation=equation):
                inputs = ["x"] + (["w"] if other is not None else [])
                initializers = [arr("w", RNG.standard_normal(other))] if other else []
                self.case(helper.make_node("Einsum", inputs, ["y"], equation=equation),
                          x_shape, y_shape, initializers, opset=12)

    def test_second_operand(self):
        self.case(helper.make_node("Einsum", ["w", "x"], ["y"], equation="ij,jk->ik"),
                  [3, 4], [2, 4], [arr("w", RNG.standard_normal((2, 3)))], opset=12)

    def test_both_operands_are_the_input(self):
        self.case(helper.make_node("Einsum", ["x", "x"], ["y"], equation="ij,kj->ik"),
                  [2, 3], [2, 2], opset=12)

    def test_broadcast_ellipsis(self):
        # x's ellipsis is one batch dimension of size 1 against the other operand's 2
        self.case(helper.make_node("Einsum", ["x", "w"], ["y"], equation="...ij,...jk->...ik"),
                  [1, 3, 4], [2, 3, 2], [arr("w", RNG.standard_normal((2, 4, 2)))], opset=12)


class ResizeTests(Case):
    """The reverse rule measures each axis's interpolation matrix with the node's own
    resize, so every mode and coordinate transform is exercised the same way."""

    def resize(self, mode, scales=None, sizes=None, **attrs):
        inputs = ["x", "", "scales"] if scales is not None else ["x", "", "", "sizes"]
        initializers = [arr("scales", scales, np.float32)] if scales is not None else \
            [arr("sizes", sizes, np.int64)]
        return helper.make_node("Resize", inputs, ["y"], mode=mode, **attrs), initializers

    def test_modes_by_scale(self):
        for mode, coordinates in (("nearest", "asymmetric"), ("linear", "half_pixel"),
                                  ("linear", "align_corners"), ("cubic", "half_pixel"),
                                  ("linear", "pytorch_half_pixel")):
            with self.subTest(mode=mode, coordinates=coordinates):
                node, initializers = self.resize(mode, scales=[1, 1, 2, 1.5],
                                                 coordinate_transformation_mode=coordinates)
                self.case(node, [1, 2, 3, 4], [1, 2, 6, 6], initializers)

    def test_by_size(self):
        node, initializers = self.resize("linear", sizes=[1, 2, 5, 7],
                                         coordinate_transformation_mode="half_pixel")
        self.case(node, [1, 2, 3, 4], [1, 2, 5, 7], initializers)

    def test_downsampling_with_antialias(self):
        node, initializers = self.resize("linear", scales=[1, 1, 0.5, 0.5], antialias=1)
        self.case(node, [1, 1, 6, 6], [1, 1, 3, 3], initializers)

    def test_by_size_in_a_batch(self):
        # `sizes` names the batch extent too, which the forward rule has to scale by the seeds
        node, initializers = self.resize("nearest", sizes=[2, 1, 4, 4])
        self.case(node, [2, 1, 2, 2], [2, 1, 4, 4], initializers)

    def test_upsample(self):
        node = helper.make_node("Upsample", ["x", "scales"], ["y"], mode="nearest")
        self.case(node, [1, 1, 2, 3], [1, 1, 4, 6],
                  [arr("scales", [1, 1, 2, 2], np.float32)], opset=9)


class DFTTests(Case):
    """Adjoint of the unnormalized DFT is N times the inverse; checked through every case
    that changes what the adjoint has to undo."""

    def dft(self, x_shape, y_shape, opset=17, length=None, **attrs):
        inputs = ["x"] + (["n"] if length is not None else [])
        initializers = [arr("n", length, np.int64)] if length is not None else []
        self.case(helper.make_node("DFT", inputs, ["y"], axis=1, **attrs), x_shape, y_shape,
                  initializers, opset=opset)

    def test_complex(self):
        self.dft([2, 6, 2], [2, 6, 2])

    def test_inverse(self):
        # ONNX Runtime's double inverse DFT is itself only ~3e-8 accurate, so finite
        # differences need a large step -- harmless, a DFT being linear
        model = build(helper.make_node("DFT", ["x"], ["y"], axis=1, inverse=1), [2, 6, 2],
                      [2, 6, 2], opset=17)
        self.check(model, {"x": RNG.standard_normal((2, 6, 2))}, None, step=0.5, fd_tol=1e-6)

    def test_real_signal(self):
        self.dft([2, 6, 1], [2, 6, 2])

    def test_onesided(self):
        self.dft([2, 6, 1], [2, 4, 2], onesided=1)

    def test_zero_padded_length(self):
        self.dft([2, 5, 2], [2, 8, 2], length=8)

    def test_truncated_length(self):
        self.dft([2, 7, 2], [2, 4, 2], length=4)

    def test_axis_as_an_input(self):
        if MAX_OPSET < 20:
            self.skipTest("DFT takes its axis as an input from opset 20")
        node = helper.make_node("DFT", ["x", "", "a"], ["y"])
        self.case(node, [2, 6, 2], [2, 6, 2], [arr("a", 1, np.int64)], opset=20)


class GatherLikeTests(Case):
    """Operations that move elements: Unique, MaxUnpool, Col2Im -- and DequantizeLinear,
    differentiated in its scale."""

    def test_unique(self):
        for attrs, y_shape in (({}, [6]), ({"axis": 1}, [2, 3]), ({"sorted": 0}, [6])):
            with self.subTest(**attrs):
                self.case(helper.make_node("Unique", ["x"], ["y"], **attrs), [2, 3], y_shape)

    def test_unique_with_duplicates(self):
        # the first occurrence carries the derivative; finite differences cannot see that,
        # as perturbing a duplicate changes the output's size
        x = np.array([3.0, 1.0, 3.0, 2.0, 1.0])
        model = build(helper.make_node("Unique", ["x"], ["y", "i", "inv", "c"]), [5], [3],
                      outputs=[helper.make_tensor_value_info("y", TensorProto.DOUBLE, [3])])
        expected = np.zeros((3, 5))
        expected[0, 1] = expected[1, 3] = expected[2, 0] = 1.0
        self.check(model, {"x": x}, expected, differences=False)

    def test_max_unpool(self):
        indices = arr("i", [[[[0, 3], [9, 14]]]], np.int64)
        self.case(helper.make_node("MaxUnpool", ["x", "i"], ["y"], kernel_shape=[2, 2],
                                   strides=[2, 2]), [1, 1, 2, 2], [1, 1, 4, 4], [indices])

    def test_max_unpool_after_overlapping_windows(self):
        # windows of 3 with stride 1 on a peaked input report the peak more than once
        x = np.array([[[[0.0, 1.0, 0.5, 0.2], [0.1, 9.0, 0.3, 0.4], [0.2, 0.1, 0.6, 8.0],
                        [0.3, 0.2, 0.7, 0.8]]]])
        pool = helper.make_node("MaxPool", ["x"], ["p", "i"], kernel_shape=[3, 3])
        unpool = helper.make_node("MaxUnpool", ["p", "i", "s"], ["y"], kernel_shape=[3, 3])
        self.case([pool, unpool], [1, 1, 4, 4], [1, 1, 4, 4],
                  [arr("s", [1, 1, 4, 4], np.int64)], x=x)

    def test_col2im(self):
        if MAX_OPSET < 18:
            self.skipTest("Col2Im is opset 18")
        for attrs, image, cols in (({}, [4, 5], 9), ({"strides": [2, 1]}, [5, 5], 6),
                                   ({"pads": [1, 0, 1, 1], "dilations": [1, 2]}, [4, 5], 10)):
            with self.subTest(**attrs):
                self.case(helper.make_node("Col2Im", ["x", "img", "blk"], ["y"], **attrs),
                          [1, 2*6, cols], [1, 2] + image,
                          [arr("img", image, np.int64), arr("blk", [2, 3], np.int64)])

    def test_dequantize_linear(self):
        q = arr("q", RNG.integers(-100, 100, (2, 3, 4)), np.int8)
        zp = arr("zp", [3, -2, 0], np.int8)
        with self.subTest("per axis"):
            self.case(helper.make_node("DequantizeLinear", ["q", "x", "zp"], ["y"], axis=1),
                      [3], [2, 3, 4], [q, zp], opset=13, x=np.array([0.1, 0.2, 0.05]))
        with self.subTest("per tensor"):
            self.case(helper.make_node("DequantizeLinear", ["q", "x"], ["y"]), [], [2, 3, 4],
                      [q], x=np.array(0.1))


class LoweredCase(Case):
    """Lowered operations: the lowered primal against the runtime's own kernel (or the ONNX
    reference implementation where the runtime has none), then the derivatives in double,
    through the lowering, which runs in double whatever kernels the runtime lacks."""

    def lowered(self, nodes, x_shape, y_shape, initializers=(), opset=18, x=None,
                primal_tol=1e-12, expected=None):
        if opset > MAX_OPSET:
            self.skipTest("opset %d is newer than this onnx" % opset)
        x = RNG.standard_normal(x_shape) if x is None else x
        model = build(nodes, x_shape, y_shape, initializers, opset)
        lowered = lower(model)
        self.assertFalse({n.op_type for n in lowered.graph.node} & set(LOWERINGS))
        if not runs(lowered, {"x": x}):
            self.skipTest("this ONNX Runtime does not load opset %d" % opset)
        got = run(lowered, {"x": x})["y"]
        single = build(nodes, x_shape, y_shape, [
            numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name)
            if numpy_helper.to_array(t).dtype == np.float64 else t
            for t in initializers], opset, dtype=TensorProto.FLOAT)
        if expected is not None:
            pass
        elif runs(model, {"x": x}):
            expected = run(model, {"x": x})["y"]
        elif runs(single, {"x": x.astype(np.float32)}):
            expected = run(single, {"x": x.astype(np.float32)})["y"]
            primal_tol = max(primal_tol, 2e-5)
        else:  # the reference implementation, where no runtime kernel exists
            from onnx.reference import ReferenceEvaluator
            expected = ReferenceEvaluator(model).run(None, {"x": x})[0]
        np.testing.assert_allclose(got, expected, rtol=primal_tol, atol=primal_tol)
        return self.check(lowered, {"x": x}, fd_tol=1e-5)


class LRNTests(LoweredCase):
    def test_sizes(self):
        for size in (1, 3, 5):
            with self.subTest(size=size):
                self.lowered(helper.make_node("LRN", ["x"], ["y"], size=size, alpha=0.3,
                                              beta=0.6, bias=1.5), [2, 6, 3, 2], [2, 6, 3, 2],
                             primal_tol=1e-6)

    def test_even_size(self):
        # ONNX Runtime refuses even sizes and the reference implementation centres the window
        # differently from the spec, so the spec's own formula is the reference: the window
        # is [c - floor((size-1)/2), c + ceil((size-1)/2)], here [c, c + 1]
        x = RNG.standard_normal((2, 5, 3, 2))
        square = np.pad(x**2, [(0, 0), (0, 1), (0, 0), (0, 0)])
        total = square[:, :-1] + square[:, 1:]
        self.lowered(helper.make_node("LRN", ["x"], ["y"], size=2, alpha=0.3, beta=0.6,
                                      bias=1.5), [2, 5, 3, 2], [2, 5, 3, 2], x=x,
                     expected=x*(1.5 + 0.3/2*total)**-0.6, primal_tol=1e-6)

    def test_old_opset(self):
        self.lowered(helper.make_node("LRN", ["x"], ["y"], size=3), [1, 4, 5, 2], [1, 4, 5, 2],
                     opset=9, primal_tol=1e-6)


class GridSampleTests(LoweredCase):
    IMAGE = RNG.standard_normal((2, 3, 4, 5))

    def sample(self, grid_input, opset=20, **attrs):
        """Differentiated in the grid (the image a constant), or the other way round."""
        if opset > MAX_OPSET:  # the same modes, under their opset 16 names
            opset = 16
            attrs["mode"] = {"linear": "bilinear", "cubic": "bicubic"}.get(
                attrs.get("mode"), attrs.get("mode"))
            attrs = {k: v for k, v in attrs.items() if v is not None}
        grid = RNG.uniform(-1.3, 1.3, (2, 3, 2, 2))
        if grid_input:
            node = helper.make_node("GridSample", ["img", "x"], ["y"], **attrs)
            return self.lowered(node, [2, 3, 2, 2], [2, 3, 3, 2], [arr("img", self.IMAGE)],
                                opset=opset, x=grid, primal_tol=2e-5)
        node = helper.make_node("GridSample", ["x", "grid"], ["y"], **attrs)
        return self.lowered(node, [2, 3, 4, 5], [2, 3, 3, 2], [arr("grid", grid)],
                            opset=opset, x=self.IMAGE, primal_tol=2e-5)

    def test_modes(self):
        for mode in ("linear", "nearest", "cubic"):
            for padding in ("zeros", "border", "reflection"):
                for aligned in (0, 1):
                    for grid_input in (False, True):
                        with self.subTest(mode=mode, padding=padding, aligned=aligned,
                                          grid=grid_input):
                            if (mode, padding) == ("cubic", "border") and OLD_ORT:
                                self.skipTest("ONNX Runtime before 1.19 clamps a cubic's "
                                              "coordinate as well as its taps; later "
                                              "versions and PyTorch clamp only the taps")
                            self.sample(grid_input, mode=mode, padding_mode=padding,
                                        align_corners=aligned)

    def test_opset_16_names(self):
        for mode in ("bilinear", "bicubic"):
            with self.subTest(mode=mode):
                self.sample(True, opset=16, mode=mode)

    def test_spatial_transformer(self):
        # AffineGrid expands to a body that branches on a condition folding to a constant;
        # the branch taken must fold in turn, or the grid loses its shape
        if MAX_OPSET < 20:
            self.skipTest("AffineGrid is opset 20")
        theta = np.eye(2, 3)*1.3 + 0.2*RNG.standard_normal((2, 2, 3))
        nodes = [helper.make_node("AffineGrid", ["x", "size"], ["grid"]),
                 helper.make_node("GridSample", ["img", "grid"], ["y"], mode="cubic",
                                  padding_mode="reflection")]
        self.lowered(nodes, [2, 2, 3], [2, 3, 6, 7],
                     [arr("size", [2, 3, 6, 7], np.int64), arr("img", self.IMAGE)], opset=20,
                     x=theta, primal_tol=2e-5)

    def test_volumetric(self):
        grid = RNG.uniform(-1.1, 1.1, (1, 2, 3, 2, 3))
        node = helper.make_node("GridSample", ["img", "x"], ["y"], mode="linear")
        self.lowered(node, [1, 2, 3, 2, 3], [1, 2, 2, 3, 2],
                     [arr("img", RNG.standard_normal((1, 2, 3, 4, 3)))], opset=20, x=grid,
                     primal_tol=2e-5)


class STFTTests(LoweredCase):
    def test_windowed(self):
        window = arr("w", np.hanning(6))
        for onesided, bins in ((1, 4), (0, 6)):
            with self.subTest(onesided=onesided):
                self.lowered(helper.make_node("STFT", ["x", "step", "w"], ["y"],
                                              onesided=onesided),
                             [2, 16, 1], [2, 6, bins, 2],
                             [arr("step", 2, np.int64), window], opset=17, primal_tol=1e-9)

    def test_frame_length_and_complex_signal(self):
        # against numpy: ONNX Runtime 1.19's STFT scrambles the frames of a complex signal
        x = RNG.standard_normal((1, 10, 2))
        signal = x[..., 0] + 1j*x[..., 1]
        frames = np.stack([np.fft.fft(signal[:, 3*f:3*f + 4]) for f in range(3)], axis=1)
        self.lowered(helper.make_node("STFT", ["x", "step", "", "n"], ["y"], onesided=0),
                     [1, 10, 2], [1, 3, 4, 2],
                     [arr("step", 3, np.int64), arr("n", 4, np.int64)], opset=17, x=x,
                     primal_tol=1e-9, expected=np.stack([frames.real, frames.imag], -1))


class TensorScatterTests(LoweredCase):
    def scatter(self, mode, written):
        cache = arr("cache", RNG.standard_normal((2, 3, 5, 4)))
        indices = [arr("wi", written, np.int64)] if written is not None else []
        node = helper.make_node("TensorScatter", ["cache", "x"] + (["wi"] if indices else []),
                                ["y"], mode=mode)
        self.lowered(node, [2, 3, 2, 4], [2, 3, 5, 4], [cache] + indices, opset=24)

    def test_linear(self):
        self.scatter("linear", [1, 3])

    def test_circular(self):
        self.scatter("circular", [4, 2])

    def test_from_the_start(self):
        self.scatter("linear", None)


if __name__ == "__main__":
    unittest.main()
