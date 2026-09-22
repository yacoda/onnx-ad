"""RNN, GRU and LSTM: expanded into a Scan, then differentiated as control flow.

Two separate things are checked. That the expansion *is* the operation -- its primal output
against ONNX Runtime's own fused kernel -- and that its derivative is right, against the
usual references.
"""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_ad.recurrent import expand_recurrent
from test_ad import IR_VERSION, JacobianCase, available, run

RNG = np.random.default_rng(23)
D = TensorProto.DOUBLE


def recurrent(op, steps=3, batch=2, width=3, hidden=4, direction="forward", layout=0,
              bias=True, initial=True, peepholes=False, dtype=D, **attrs):
    directions = 2 if direction == "bidirectional" else 1
    gates = {"RNN": 1, "GRU": 3, "LSTM": 4}[op]
    kind = np.float64 if dtype == D else np.float32
    values = {"W": RNG.standard_normal((directions, gates*hidden, width))*0.5,
              "R": RNG.standard_normal((directions, gates*hidden, hidden))*0.5}
    inputs = ["x", "W", "R", "B" if bias else ""]
    if bias:
        values["B"] = RNG.standard_normal((directions, 2*gates*hidden))*0.3
    if initial or peepholes:
        inputs += ["", "h0"]
        state = (directions, batch, hidden) if layout == 0 else (batch, directions, hidden)
        values["h0"] = RNG.standard_normal(state)*0.3
        if op == "LSTM":
            inputs.append("c0")
            values["c0"] = RNG.standard_normal(state)*0.3
    if peepholes:
        inputs.append("P")
        values["P"] = RNG.standard_normal((directions, 3*hidden))*0.3
    outputs = ["y", "yh"] + (["yc"] if op == "LSTM" else [])
    node = helper.make_node(op, inputs, outputs, hidden_size=hidden, direction=direction,
                            layout=layout, **attrs)
    x_shape = [steps, batch, width] if layout == 0 else [batch, steps, width]
    y_shape = [steps, directions, batch, hidden] if layout == 0 else \
        [batch, steps, directions, hidden]
    h_shape = [directions, batch, hidden] if layout == 0 else [batch, directions, hidden]
    graph = helper.make_graph(
        [node], "g", [helper.make_tensor_value_info("x", dtype, x_shape)],
        [helper.make_tensor_value_info("y", dtype, y_shape),
         helper.make_tensor_value_info("yh", dtype, h_shape)]
        + ([helper.make_tensor_value_info("yc", dtype, h_shape)] if op == "LSTM" else []),
        [numpy_helper.from_array(v.astype(kind), k) for k, v in values.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model, {"x": (RNG.standard_normal(x_shape)*0.8).astype(kind)}


def sequence_first(model):
    """The same recurrent model with layout 0: states and data sequence-first."""
    copy = type(model)()
    copy.CopyFrom(model)
    node = copy.graph.node[0]
    for attribute in node.attribute:
        if attribute.name == "layout":
            attribute.i = 0
    for tensor in copy.graph.initializer:
        if tensor.name in ("h0", "c0"):
            value = numpy_helper.to_array(tensor).transpose(1, 0, 2)
            tensor.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(value), tensor.name))

    def reorder(value, perm):
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        value.type.tensor_type.shape.ClearField("dim")
        for axis in perm:
            value.type.tensor_type.shape.dim.add().dim_value = dims[axis]

    reorder(copy.graph.input[0], [1, 0, 2])
    reorder(copy.graph.output[0], [1, 2, 0, 3])
    for output in copy.graph.output[1:]:
        reorder(output, [1, 0, 2])
    return copy


class RecurrentTests(JacobianCase):
    CASES = [("RNN", {}), ("GRU", {}), ("GRU", {"linear_before_reset": 1}), ("LSTM", {}),
             ("LSTM", {"peepholes": True}), ("LSTM", {"direction": "reverse"}),
             ("LSTM", {"direction": "bidirectional"}), ("GRU", {"layout": 1}),
             ("RNN", {"bias": False, "initial": False}),
             ("RNN", {"activations": ["Relu"]}),
             ("LSTM", {"activations": ["HardSigmoid", "Tanh", "Softsign"],
                       "activation_alpha": [0.25], "activation_beta": [0.4]})]

    def test_the_expansion_is_the_operation(self):
        # ONNX Runtime's fused kernels are single precision only, so compare there
        for op, options in self.CASES:
            with self.subTest(op=op, **{k: str(v) for k, v in options.items()}):
                model, feeds = recurrent(op, dtype=TensorProto.FLOAT, **options)
                expanded = run(expand_recurrent(model), feeds)
                if options.get("layout") == 1:
                    # ORT implements no batch-first kernel: check against its sequence-first
                    # kernel on the same weights, with the data transposed in and out
                    fused = run(sequence_first(model), {"x": feeds["x"].transpose(1, 0, 2)})
                    fused = {"y": fused["y"].transpose(2, 0, 1, 3),
                             "yh": fused["yh"].transpose(1, 0, 2)}
                else:
                    fused = run(model, feeds)
                for name, value in fused.items():
                    np.testing.assert_allclose(expanded[name], value, rtol=2e-5, atol=2e-6)

    def test_derivatives_in_double(self):
        # through the expanded model, which runs in double where the fused kernel cannot
        for op, options in self.CASES:
            for y in ("y", "yh") + (("yc",) if op == "LSTM" else ()):
                with self.subTest(op=op, output=y, **{k: str(v) for k, v in options.items()}):
                    model, feeds = recurrent(op, **options)
                    expanded = expand_recurrent(model)
                    if not available(expanded, feeds):
                        continue  # an activation with no double kernel in this runtime
                    self.check(expanded, feeds, None, y=y, fd_tol=1e-5,
                               inputs=["x"], outputs=[y])

    def test_derivatives_end_to_end(self):
        # the fused operation itself, expanded inside the passes, in single precision
        for op, options in self.CASES:
            if options.get("layout") == 1:
                continue  # no batch-first kernel in ONNX Runtime to take differences of
            with self.subTest(op=op, **{k: str(v) for k, v in options.items()}):
                model, feeds = recurrent(op, dtype=TensorProto.FLOAT, **options)
                self.check(model, feeds, None, y="yh", rtol=5e-5, step=1e-3, fd_tol=5e-3,
                           inputs=["x"], outputs=["yh"])

    def test_a_clip_is_refused(self):
        from onnx_ad import UnsupportedOperator, reverse
        model, _ = recurrent("LSTM", clip=1.0)
        with self.assertRaises(UnsupportedOperator):
            reverse(model)


if __name__ == "__main__":
    unittest.main()
