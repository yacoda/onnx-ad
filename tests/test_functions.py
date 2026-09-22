"""Operations the ONNX specification defines as a function of simpler ones.

They have no rule of their own: they are expanded into the specification's body and that is
differentiated. So each test exercises the expansion (and the spec's body) as much as the
rules, against the same references as everywhere else.
"""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from test_ad import IR_VERSION, MAX_OPSET, JacobianCase, run

RNG = np.random.default_rng(5)


def runs(model, feeds):
    """Whether this ONNX Runtime can load and run the primal model at all."""
    try:
        run(model, feeds)
        return True
    except Exception:
        return False


def build(node, x_shape, y_shape, initializers=(), opset=18, dtype=TensorProto.DOUBLE,
          y_dtype=None):
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", dtype, x_shape)],
                              [helper.make_tensor_value_info("y", y_dtype or dtype, y_shape)],
                              list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def arr(name, value, dtype=np.float64):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


class FunctionOpTests(JacobianCase):
    def run_case(self, node, x_shape, y_shape, initializers=(), opset=18, x=None, **options):
        """Double precision where ONNX Runtime can run the operation that way, else float32
        with a coarse finite-difference step -- some of these have no double kernel, and
        ORT's fallback to the spec's own body is not always type-correct either."""
        if opset > MAX_OPSET:
            self.skipTest("opset %d is newer than this onnx" % opset)
        x = RNG.standard_normal(x_shape) if x is None else x
        model = build(node, x_shape, y_shape, initializers, opset=opset)
        if runs(model, {"x": x}):
            self.check(model, {"x": x}, None, fd_tol=options.pop("fd_tol", 1e-5), **options)
            return
        single = [numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32)
                                          if numpy_helper.to_array(t).dtype == np.float64
                                          else numpy_helper.to_array(t), t.name)
                  for t in initializers]
        model = build(node, x_shape, y_shape, single, opset=opset, dtype=TensorProto.FLOAT)
        x = x.astype(np.float32)
        if not runs(model, {"x": x}):
            self.skipTest("%s does not run in this ONNX Runtime" % node.op_type)
        self.check(model, {"x": x}, None, rtol=5e-5, step=1e-3, fd_tol=5e-3)

    def test_swish(self):
        self.run_case(helper.make_node("Swish", ["x"], ["y"], alpha=1.7), [4], [4], opset=24)

    def test_swish_default_alpha(self):
        self.run_case(helper.make_node("Swish", ["x"], ["y"]), [4], [4], opset=24)

    def test_mean_variance_normalization(self):
        self.run_case(helper.make_node("MeanVarianceNormalization", ["x"], ["y"], axes=[0, 2]),
                      [2, 3, 4], [2, 3, 4])

    def test_reduce_log_sum(self):
        x = np.abs(RNG.standard_normal((2, 3))) + 0.5
        self.run_case(helper.make_node("ReduceLogSum", ["x", "a"], ["y"], keepdims=0),
                      [2, 3], [2], [arr("a", [1], np.int64)], x=x)

    def test_group_normalization(self):
        node = helper.make_node("GroupNormalization", ["x", "s", "b"], ["y"], num_groups=2,
                                stash_type=TensorProto.DOUBLE)
        self.run_case(node, [2, 4, 3], [2, 4, 3],
                      [arr("s", RNG.standard_normal(4)), arr("b", RNG.standard_normal(4))],
                      opset=21)

    def test_rms_normalization(self):
        node = helper.make_node("RMSNormalization", ["x", "s"], ["y"], axis=-1,
                                stash_type=TensorProto.DOUBLE)
        self.run_case(node, [2, 5], [2, 5], [arr("s", RNG.standard_normal(5))], opset=23)

    def test_softmax_cross_entropy_loss(self):
        node = helper.make_node("SoftmaxCrossEntropyLoss", ["x", "labels"], ["y"],
                                reduction="mean")
        self.run_case(node, [3, 4], [], [arr("labels", [1, 3, 0], np.int64)])

    def test_negative_log_likelihood_loss(self):
        node = helper.make_node("NegativeLogLikelihoodLoss", ["x", "labels"], ["y"],
                                reduction="sum")
        self.run_case(node, [3, 4], [], [arr("labels", [1, 3, 0], np.int64)])

    def test_gather_elements_and_scatter_elements(self):
        indices = np.array([[1, 0, 1], [0, 0, 1]], dtype=np.int64)
        self.run_case(helper.make_node("GatherElements", ["x", "i"], ["y"], axis=0),
                      [2, 3], [2, 3], [arr("i", indices, np.int64)])
        for reduction in ("none", "add"):
            extra = {} if reduction == "none" else {"reduction": reduction}
            self.run_case(helper.make_node("ScatterElements", ["x", "i", "u"], ["y"], axis=1,
                                           **extra),
                          [2, 3], [2, 3], [arr("i", [[2], [0]], np.int64),
                                           arr("u", [[5.0], [7.0]])])

    def test_range_is_differentiable_in_start_and_delta(self):
        # y = start + delta*i;  here x feeds both, so dy_i/dx = 1 + i
        nodes = [helper.make_node("Range", ["x", "limit", "x"], ["y"])]
        graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [])],
                                  [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [4])],
                                  [arr("limit", 3.5)])
        m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = IR_VERSION
        self.check(m, {"x": np.array(0.8)}, (1 + np.arange(4.0)).reshape(4, 1), fd_tol=1e-5)

    def test_center_crop_pad(self):
        for shape in ([6, 3], [2, 3]):
            with self.subTest(target=shape):
                node = helper.make_node("CenterCropPad", ["x", "s"], ["y"])
                self.run_case(node, [4, 3], shape, [arr("s", shape, np.int64)])

    def test_attention(self):
        node = helper.make_node("Attention", ["x", "k", "v"], ["y"])
        k = RNG.standard_normal((1, 2, 5, 4))
        v = RNG.standard_normal((1, 2, 5, 4))
        self.run_case(node, [1, 2, 3, 4], [1, 2, 3, 4], [arr("k", k), arr("v", v)], opset=23)

    def test_rotary_embedding(self):
        node = helper.make_node("RotaryEmbedding", ["x", "cos", "sin"], ["y"])
        cos = np.cos(RNG.standard_normal((1, 3, 2)))
        sin = np.sin(RNG.standard_normal((1, 3, 2)))
        self.run_case(node, [1, 2, 3, 4], [1, 2, 3, 4], [arr("cos", cos), arr("sin", sin)],
                      opset=23)


class Opset27Tests(JacobianCase):
    """Operations new in opset 27, which no released ONNX Runtime loads yet. Their expansion
    uses only operations unchanged since opset 21, so the derivative models are run stamped
    as opset 21 -- the finite differences take the expanded primal, stamped likewise."""

    def setUp(self):
        if MAX_OPSET < 27:
            self.skipTest("opset 27 is newer than this onnx")
        import test_ad

        def stamped(model, feeds, _run=test_ad.run):
            copy = type(model)()
            copy.CopyFrom(model)
            for entry in copy.opset_import:
                if entry.domain in ("", "ai.onnx"):
                    entry.version = min(entry.version, 21)
            return _run(copy, feeds)
        self._run = test_ad.run
        test_ad.run = stamped

    def tearDown(self):
        import test_ad
        test_ad.run = getattr(self, "_run", test_ad.run)

    def expanded(self, node, x_shape, y_shape, initializers, dtype=TensorProto.DOUBLE):
        from onnx_ad.expand import expand_functions
        model = build(node, x_shape, y_shape, initializers, opset=27, dtype=dtype)
        return model, expand_functions(model)

    def test_causal_conv_with_state(self):
        # float32: ONNX Runtime's Conv has no double kernel
        f = lambda name, shape: arr(name, RNG.standard_normal(shape), np.float32)
        for activation in ("none", "silu"):
            with self.subTest(activation=activation):
                node = helper.make_node("CausalConvWithState", ["x", "w", "b", "p"],
                                        ["y", "s"], activation=activation)
                model, expanded = self.expanded(
                    node, [2, 3, 5], [2, 3, 5],
                    [f("w", (3, 1, 3)), f("b", (3,)), f("p", (2, 3, 2))], TensorProto.FLOAT)
                x = {"x": RNG.standard_normal((2, 3, 5)).astype(np.float32)}
                fwd = self.check(model, x, None, rtol=5e-5, differences=False)
                import test_ad
                np.testing.assert_allclose(fwd, test_ad.jacobian_differences(
                    expanded, x, "x", "y", 1e-3), rtol=5e-3, atol=5e-3)

    def test_linear_attention(self):
        # the body computes in float32 whatever the input type; the output is linear in the
        # query, so a large finite-difference step is exact up to that rounding
        import test_ad
        for rule in ("linear", "gated", "delta", "gated_delta"):
            with self.subTest(update_rule=rule):
                initializers = [arr("k", 0.5*RNG.standard_normal((2, 3, 4))),
                                arr("v", RNG.standard_normal((2, 3, 4)))]
                extra = ["", "", ""]
                if "gated" in rule:
                    initializers.append(arr("g", -0.3*np.abs(RNG.standard_normal((2, 3, 4)))))
                    extra[1] = "g"
                if "delta" in rule:
                    initializers.append(arr("beta", RNG.uniform(0.1, 0.9, (2, 3, 2))))
                    extra[2] = "beta"
                node = helper.make_node("LinearAttention", ["x", "k", "v"] + extra,
                                        ["y", "state"], q_num_heads=2, kv_num_heads=2,
                                        update_rule=rule)
                model, expanded = self.expanded(node, [2, 3, 4], [2, 3, 4], initializers)
                x = {"x": RNG.standard_normal((2, 3, 4))}
                fwd = self.check(model, x, None, differences=False)
                np.testing.assert_allclose(fwd, test_ad.jacobian_differences(
                    expanded, x, "x", "y", 0.5), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
