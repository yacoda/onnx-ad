"""Structure, pooling and normalization rules added after 0.2.0.

Each case runs through the shared harness: forward against reverse, finite differences of
the primal, and the closure check that every derivative model uses only operations with
rules. Where ONNX Runtime has no double kernel the case drops to float32.
"""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from test_ad import IR_VERSION, MAX_OPSET, JacobianCase, run

RNG = np.random.default_rng(17)
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


if __name__ == "__main__":
    unittest.main()
