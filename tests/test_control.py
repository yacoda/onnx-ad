"""Control flow: If, Scan and Loop, differentiated and checked like every other rule.

The references are the same ones the rest of the suite uses -- finite differences of the
primal, analytic Jacobians where they are short, forward against reverse -- plus one that
only exists here: a loop with a static trip count can be *unrolled* into a flat graph, and
the Jacobian of the unrolled graph comes from entirely different code than the Jacobian of
the loop.
"""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx_ad import forward, reverse
from test_ad import (IR_VERSION, JacobianCase, jacobian_forward, jacobian_reverse, run)

RNG = np.random.default_rng(11)
D = TensorProto.DOUBLE


def vi(name, shape, dtype=D):
    return helper.make_tensor_value_info(name, dtype, shape)


def model(nodes, inputs, outputs, initializers=(), opset=18):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    result = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    result.ir_version = IR_VERSION
    onnx.checker.check_model(result)
    return result


def const(name, value):
    return numpy_helper.from_array(np.asarray(value), name)


def branch(nodes, outputs, name):
    return helper.make_graph(nodes, name, [], outputs)


class IfTests(JacobianCase):
    """Differentiated inside the branch, so only the taken branch computes a derivative.

    No `Sin` anywhere, even in a branch a test never takes: older ONNX Runtime has no
    double-precision kernel for `Cos` (its derivative), and it resolves kernels for *both*
    branches when the session loads.
    """

    W = np.array([2.0, -1.5, 0.5])

    def two_branch_model(self, condition):
        then = branch([helper.make_node("Mul", ["x", "w"], ["a"]),
                       helper.make_node("Tanh", ["a"], ["ta"])], [vi("ta", [3])], "then")
        otherwise = branch([helper.make_node("Exp", ["x"], ["e"])], [vi("e", [3])], "else")
        nodes = [helper.make_node("If", ["c"], ["y"], then_branch=then, else_branch=otherwise)]
        return model(nodes, [vi("x", [3])], [vi("y", [3])],
                     [const("w", self.W), const("c", np.array(condition))])

    def test_then_branch(self):
        x = np.array([0.3, -0.7, 1.1])
        expected = np.diag(self.W*(1 - np.tanh(self.W*x)**2))
        self.check(self.two_branch_model(True), {"x": x}, expected)

    def test_else_branch(self):
        x = np.array([0.3, -0.7, 1.1])
        self.check(self.two_branch_model(False), {"x": x}, np.diag(np.exp(x)))

    def test_condition_is_a_runtime_input(self):
        then = branch([helper.make_node("Mul", ["x", "x"], ["s"])], [vi("s", [3])], "then")
        otherwise = branch([helper.make_node("Neg", ["x"], ["n"])], [vi("n", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])])
        x = np.array([0.3, -0.7, 1.1])
        for condition, expected in ((True, np.diag(2*x)), (False, -np.eye(3))):
            with self.subTest(condition=condition):
                self.check(m, {"x": x, "c": np.array(condition)}, expected)

    def test_a_branch_that_is_constant(self):
        # the else branch does not depend on x: it must emit a zero tangent of its own,
        # so both branches keep the same output signature
        then = branch([helper.make_node("Tanh", ["x"], ["s"])], [vi("s", [3])], "then")
        otherwise = branch([helper.make_node("Identity", ["w"], ["k"])], [vi("k", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])],
                  [const("w", self.W)])
        x = np.array([0.3, -0.7, 1.1])
        self.check(m, {"x": x, "c": np.array(False)}, np.zeros((3, 3)))

    def test_a_branch_that_passes_an_outer_value_through(self):
        # the checker insists a branch output be produced inside the branch, so a pure
        # pass-through is an Identity on the captured value
        then = branch([helper.make_node("Identity", ["x"], ["xi"])], [vi("xi", [3])], "then")
        otherwise = branch([helper.make_node("Mul", ["x", "w"], ["m"])], [vi("m", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])],
                  [const("w", self.W)])
        x = np.array([0.3, -0.7, 1.1])
        for condition, expected in ((True, np.eye(3)), (False, np.diag(self.W))):
            with self.subTest(condition=condition):
                self.check(m, {"x": x, "c": np.array(condition)}, expected)

    def test_captured_value_is_the_differentiated_one(self):
        # w is a model input read only inside a branch: its adjoint has to leave the If
        then = branch([helper.make_node("Mul", ["a", "w"], ["m"])], [vi("m", [3])], "then")
        otherwise = branch([helper.make_node("Add", ["a", "w"], ["p"])], [vi("p", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("w", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])],
                  [const("a", np.array([0.5, 1.5, -2.0]))])
        w = np.array([0.3, -0.7, 1.1])
        for condition, expected in ((True, np.diag([0.5, 1.5, -2.0])), (False, np.eye(3))):
            with self.subTest(condition=condition):
                self.check(m, {"w": w, "c": np.array(condition)}, expected, x="w")

    def test_values_flowing_in_and_out(self):
        # x -> Exp -> into the If; the If's output -> Mul afterwards
        then = branch([helper.make_node("Mul", ["e", "e"], ["s"])], [vi("s", [3])], "then")
        otherwise = branch([helper.make_node("Neg", ["e"], ["n"])], [vi("n", [3])], "else")
        nodes = [helper.make_node("Exp", ["x"], ["e"]),
                 helper.make_node("If", ["c"], ["b"], then_branch=then, else_branch=otherwise),
                 helper.make_node("Mul", ["b", "x"], ["y"])]
        m = model(nodes, [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])])
        x = np.array([0.3, -0.7, 1.1])
        # then: y = e^{2x} x, y' = e^{2x}(2x + 1);  else: y = -e^x x, y' = -e^x(x + 1)
        for condition, expected in ((True, np.diag(np.exp(2*x)*(2*x + 1))),
                                    (False, np.diag(-np.exp(x)*(x + 1)))):
            with self.subTest(condition=condition):
                self.check(m, {"x": x, "c": np.array(condition)}, expected)

    def test_several_outputs(self):
        then = branch([helper.make_node("Exp", ["x"], ["e"]),
                       helper.make_node("Identity", ["w"], ["k"])],
                      [vi("e", [3]), vi("k", [3])], "then")
        otherwise = branch([helper.make_node("Tanh", ["x"], ["s"]),
                            helper.make_node("Mul", ["x", "w"], ["m"])],
                           [vi("s", [3]), vi("m", [3])], "else")
        nodes = [helper.make_node("If", ["c"], ["y1", "y2"], then_branch=then,
                                  else_branch=otherwise),
                 helper.make_node("Add", ["y1", "y2"], ["y"])]
        m = model(nodes, [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])],
                  [const("w", self.W)])
        x = np.array([0.3, -0.7, 1.1])
        self.check(m, {"x": x, "c": np.array(True)}, np.diag(np.exp(x)))

    def test_nested(self):
        inner_then = branch([helper.make_node("Exp", ["x"], ["e"])], [vi("e", [3])], "it")
        inner_else = branch([helper.make_node("Neg", ["x"], ["n"])], [vi("n", [3])], "ie")
        outer_then = branch([helper.make_node("If", ["d"], ["z"], then_branch=inner_then,
                                              else_branch=inner_else),
                             helper.make_node("Mul", ["z", "x"], ["zx"])],
                            [vi("zx", [3])], "ot")
        outer_else = branch([helper.make_node("Identity", ["x"], ["i"])], [vi("i", [3])], "oe")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=outer_then,
                                    else_branch=outer_else)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL), vi("d", [], TensorProto.BOOL)],
                  [vi("y", [3])])
        x = np.array([0.3, -0.7, 1.1])
        cases = [((True, True), np.diag(np.exp(x)*(x + 1))),
                 ((True, False), np.diag(-2*x)),
                 ((False, True), np.eye(3))]
        for (c, d), expected in cases:
            with self.subTest(c=c, d=d):
                self.check(m, {"x": x, "c": np.array(c), "d": np.array(d)}, expected)

    def test_forward_over_adjoint_through_an_if(self):
        then = branch([helper.make_node("Mul", ["x", "x"], ["s"]),
                       helper.make_node("Mul", ["s", "x"], ["cube"])], [vi("cube", [3])], "then")
        otherwise = branch([helper.make_node("Tanh", ["x"], ["s2"])], [vi("s2", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])])
        second = forward(reverse(m), inputs=["x"], outputs=["adj_x"])
        onnx.checker.check_model(second)
        x = np.array([0.3, -0.7, 1.1])
        # y_i = x_i^3, so with unit weights the Hessian of sum(y) is diag(6x)
        got = run(second, {"x": x, "c": np.array(True), "adj_y": np.ones((3, 1)),
                           "fwd_x": np.eye(3)})["fwd_adj_x"]
        np.testing.assert_allclose(got.reshape(3, 3), np.diag(6*x), rtol=1e-12, atol=1e-12)

    def test_only_the_taken_branch_is_evaluated(self):
        # the derivative of the untaken branch would divide by zero; it must never run
        then = branch([helper.make_node("Exp", ["x"], ["e"])], [vi("e", [3])], "then")
        otherwise = branch([helper.make_node("Reciprocal", ["x"], ["r"])], [vi("r", [3])], "else")
        m = model([helper.make_node("If", ["c"], ["y"], then_branch=then,
                                    else_branch=otherwise)],
                  [vi("x", [3]), vi("c", [], TensorProto.BOOL)], [vi("y", [3])])
        x = np.array([0.0, 0.0, 0.0])
        fwd = jacobian_forward(m, {"x": x, "c": np.array(True)}, "x", "y")
        rev = jacobian_reverse(m, {"x": x, "c": np.array(True)}, "x", "y")
        self.assertTrue(np.all(np.isfinite(fwd)) and np.all(np.isfinite(rev)))
        np.testing.assert_allclose(fwd, np.eye(3))
        np.testing.assert_allclose(rev, np.eye(3))


if __name__ == "__main__":
    unittest.main()
