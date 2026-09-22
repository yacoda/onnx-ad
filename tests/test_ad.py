"""Every rule executed through ONNX Runtime and compared against an independent reference.

Three references, deliberately unrelated to the rules under test:

* central finite differences of the *primal* model, for the shape of the answer;
* the analytic Jacobian, written in numpy, wherever one is short enough to write;
* forward against reverse -- `J` and `J^T` come from separate walks over separate rules, so
  their agreement to machine precision is a real check and not a tautology.

Nothing here imports a framework: the models are built by hand from the protobuf.
"""
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from math import erf as _erf

erf = np.vectorize(_erf)

from onnx_ad import UnsupportedOperator, forward, reverse

RNG = np.random.default_rng(7)


#: An ONNX model carries two independent version numbers, and both have a ceiling here:
#: the IR version must not exceed what the installed `onnx` checker knows, nor what the
#: installed ONNX Runtime will load (10, for every release up to 1.22). The opset version
#: must not exceed what the installed `onnx` has operator definitions for.
IR_VERSION = min(onnx.IR_VERSION, 10)
MAX_OPSET = onnx.defs.onnx_opset_version()


def build(nodes, inputs, outputs, initializers=(), opset=18, dtype=TensorProto.DOUBLE):
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(n, dtype, s) for n, s in inputs],
        [helper.make_tensor_value_info(n, dtype, s) for n, s in outputs],
        list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def run(model, feeds):
    """ONNX Runtime with its extended fusions off: FusedMatMul has no double kernel."""
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session = ort.InferenceSession(model.SerializeToString(), options,
                                   providers=["CPUExecutionProvider"])
    names = [v.name for v in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds)))


def available(model, feeds):
    """Whether this ONNX Runtime has a kernel for the model at its declared precision."""
    try:
        run(model, feeds)
    except Exception as failure:
        if "NOT_IMPLEMENTED" in str(failure) or "Could not find an implementation" in str(
                failure):
            return False
        raise
    return True


def shape_of(model, name):
    return _value(model, name)[0]


def numpy_of(model, name):
    return {TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64}[
        _value(model, name)[1]]


def _value(model, name):
    for value in list(model.graph.input) + list(model.graph.output):
        if value.name == name:
            return (tuple(d.dim_value for d in value.type.tensor_type.shape.dim),
                    value.type.tensor_type.elem_type)
    raise KeyError(name)


def matrix_shape(shape):
    """How CasADi reads an ONNX tensor: rank 0/1 as a column, rank 2 directly, higher
    ranks flattened to a column."""
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], 1
    if len(shape) == 2:
        return shape
    return int(np.prod(shape, dtype=int)), 1


def pack(seeds, shape):
    """`shape ++ [nseed]` -> CasADi's `rows`-by-`nseed*columns` matrix, seed-major."""
    rows, columns = matrix_shape(shape)
    count = seeds.shape[-1]
    if len(shape) == 2:
        seeds = seeds.transpose(0, 2, 1)
    return np.ascontiguousarray(seeds.reshape(rows, count*columns))


def unpack(packed, shape, count):
    """The inverse of `pack`."""
    rows, columns = matrix_shape(shape)
    if len(shape) == 2:
        return packed.reshape(rows, count, columns).transpose(0, 2, 1)
    return packed.reshape(tuple(shape) + (count,))


def jacobian_forward(model, feeds, x, y, **options):
    """The dense Jacobian d y/d x, from one evaluation with an identity seed matrix."""
    in_shape, out_shape = shape_of(model, x), shape_of(model, y)
    n = int(np.prod(in_shape, dtype=int))
    seeds = np.eye(n, dtype=numpy_of(model, x)).reshape(in_shape + (n,))
    derivative = forward(model, **options)
    layout = options.get("layout", "casadi")
    fed = pack(seeds, in_shape) if layout == "casadi" else seeds
    out = run(derivative, dict(feeds, **{"fwd_" + x: fed}))["fwd_" + y]
    if layout == "casadi":
        out = unpack(out, out_shape, n)
    return out.reshape(-1, n)


def jacobian_reverse(model, feeds, x, y, **options):
    """The same Jacobian, transposed out of one evaluation with an identity adjoint seed."""
    in_shape, out_shape = shape_of(model, x), shape_of(model, y)
    m = int(np.prod(out_shape, dtype=int))
    seeds = np.eye(m, dtype=numpy_of(model, y)).reshape(out_shape + (m,))
    derivative = reverse(model, **options)
    layout = options.get("layout", "casadi")
    fed = pack(seeds, out_shape) if layout == "casadi" else seeds
    out = run(derivative, dict(feeds, **{"adj_" + y: fed}))["adj_" + x]
    if layout == "casadi":
        out = unpack(out, in_shape, m)
    return out.reshape(-1, m).T


def jacobian_differences(model, feeds, x, y, step=1e-6):
    """Central differences of the primal model, the reference that knows nothing of rules."""
    base = np.asarray(feeds[x])
    columns = []
    for index in range(base.size):
        shift = np.zeros(base.size, dtype=base.dtype)
        shift[index] = step
        shift = np.asarray(shift.reshape(base.shape))
        plus = run(model, dict(feeds, **{x: np.asarray(base + shift)}))[y]
        minus = run(model, dict(feeds, **{x: np.asarray(base - shift)}))[y]
        columns.append(((plus - minus)/(2*step)).ravel())
    return np.stack(columns, axis=1)


class JacobianCase(unittest.TestCase):
    """Asserts forward == reverse == the reference, for one model and one point."""

    def check(self, model, feeds, reference=None, x="x", y="y", rtol=1e-11, fd_tol=2e-6,
              differences=True, step=1e-6, **options):
        fwd = jacobian_forward(model, feeds, x, y, **options)
        rev = jacobian_reverse(model, feeds, x, y, **options)
        np.testing.assert_allclose(fwd, rev, rtol=rtol, atol=rtol)
        if reference is not None:
            np.testing.assert_allclose(fwd, reference, rtol=rtol, atol=rtol)
        if differences:
            np.testing.assert_allclose(fwd, jacobian_differences(model, feeds, x, y, step),
                                       rtol=fd_tol, atol=fd_tol)
        return fwd


class ElementwiseTests(JacobianCase):
    """One unary rule per case, against its analytic derivative written in numpy."""

    CASES = {
        "Exp": np.exp,
        "Log": lambda x: 1/x,
        "Sqrt": lambda x: 0.5/np.sqrt(x),
        "Reciprocal": lambda x: -1/x**2,
        "Tanh": lambda x: 1 - np.tanh(x)**2,
        "Sigmoid": lambda x: 1/(1 + np.exp(-x))*(1 - 1/(1 + np.exp(-x))),
        "Softplus": lambda x: 1/(1 + np.exp(-x)),
        "Erf": lambda x: 2/np.sqrt(np.pi)*np.exp(-x**2),
        "Sin": np.cos,
        "Cos": lambda x: -np.sin(x),
        "Neg": lambda x: -np.ones_like(x),
        "Identity": np.ones_like,
        "Relu": lambda x: (x > 0).astype(float),
        "LeakyRelu": lambda x: np.where(x > 0, 1.0, 0.01),
        "Elu": lambda x: np.where(x > 0, 1.0, np.exp(x)),
        "Abs": np.sign,
        "Tan": lambda x: 1 + np.tan(x)**2,
        "Sinh": np.cosh,
        "Cosh": np.sinh,
        "Asin": lambda x: 1/np.sqrt(1 - x**2),
        "Acos": lambda x: -1/np.sqrt(1 - x**2),
        "Atan": lambda x: 1/(1 + x**2),
        "Asinh": lambda x: 1/np.sqrt(1 + x**2),
        "Acosh": lambda x: 1/np.sqrt(x**2 - 1),
        "Atanh": lambda x: 1/(1 - x**2),
        "Selu": lambda x: 1.05070102214813232421875*np.where(
            x > 0, 1.0, 1.67326319217681884765625*np.exp(x)),
        "Celu": lambda x: np.where(x > 0, 1.0, np.exp(x)),
        "ThresholdedRelu": lambda x: (x > 1).astype(float),
        "Shrink": lambda x: (np.abs(x) > 0.5).astype(float),
        "Softsign": lambda x: 1/(1 + np.abs(x))**2,
        "HardSigmoid": lambda x: np.where((0.2*x + 0.5 > 0) & (0.2*x + 0.5 < 1), 0.2, 0.0),
        "HardSwish": lambda x: np.clip(x/6 + 0.5, 0, 1) + x*np.where(
            (x/6 + 0.5 > 0) & (x/6 + 0.5 < 1), 1/6, 0.0),
        "Mish": lambda x: (np.tanh(np.log1p(np.exp(x)))
                           + x*(1 - np.tanh(np.log1p(np.exp(x)))**2)/(1 + np.exp(-x))),
        "Gelu": lambda x: (0.5*(1 + erf(x/np.sqrt(2)))
                           + x*np.exp(-x**2/2)/np.sqrt(2*np.pi)),
    }
    #: Arguments have to stay inside each rule's domain
    DOMAIN = {"Log": lambda x: np.abs(x) + 0.2, "Sqrt": lambda x: np.abs(x) + 0.2,
              "Asin": lambda x: 0.4*x, "Acos": lambda x: 0.4*x, "Atanh": lambda x: 0.4*x,
              "Acosh": lambda x: np.abs(x) + 1.5}
    #: ONNX Runtime has no double kernel for these, or computes them in single precision
    #: anyway (Shrink rounds a double tensor to float), so they are tested in float32 and
    #: against the analytic derivative alone -- finite differences are meaningless there.
    SINGLE = {"Erf", "Sin", "Cos", "Tan", "Sinh", "Cosh", "Asin", "Acos", "Atan",
              "Asinh", "Acosh", "Atanh", "Gelu", "Mish", "Celu", "Shrink"}
    #: Operations introduced after opset 18
    OPSET = {"Gelu": 20}

    def argument(self, op, dtype=float):
        x = np.array([0.7, -1.3, 0.25, 2.1])
        return np.asarray(self.DOMAIN.get(op, lambda v: v)(x), dtype=dtype)

    def test_rules(self):
        for op, derivative in self.CASES.items():
            if op in self.SINGLE:
                continue
            with self.subTest(op=op):
                node = helper.make_node(op, ["x"], ["y"])
                model = build([node], [("x", [4])], [("y", [4])])
                x = self.argument(op)
                if available(model, {"x": x}):
                    self.check(model, {"x": x}, np.diag(derivative(x)))
                    continue
                # older runtimes lack a double kernel for some activations; drop to float32
                # and to the analytic derivative alone, since differences are then useless
                model = build([node], [("x", [4])], [("y", [4])], dtype=TensorProto.FLOAT)
                x = self.argument(op, np.float32)
                self.check(model, {"x": x}, np.diag(derivative(x)), rtol=3e-6,
                           differences=False)

    def test_rules_without_a_double_kernel(self):
        for op in sorted(self.SINGLE):
            with self.subTest(op=op):
                opset = self.OPSET.get(op, 18)
                if opset > MAX_OPSET:
                    continue  # this onnx has no definition for the operation yet
                model = build([helper.make_node(op, ["x"], ["y"])], [("x", [4])],
                              [("y", [4])], opset=opset, dtype=TensorProto.FLOAT)
                x = self.argument(op, np.float32)
                self.check(model, {"x": x}, np.diag(self.CASES[op](x)), rtol=3e-6,
                           differences=False)

    @unittest.skipIf(MAX_OPSET < 20, "Gelu arrived in opset 20")
    def test_gelu_tanh_approximation(self):
        model = build([helper.make_node("Gelu", ["x"], ["y"], approximate="tanh")],
                      [("x", [4])], [("y", [4])], opset=20, dtype=TensorProto.FLOAT)
        x = self.argument("Gelu", np.float32)
        c, a = np.sqrt(2/np.pi), 0.044715
        inner = c*(x + a*x**3)
        expected = 0.5*(1 + np.tanh(inner)) + 0.5*x*(1 - np.tanh(inner)**2)*c*(1 + 3*a*x**2)
        self.check(model, {"x": x}, np.diag(expected), rtol=3e-6, differences=False)


class ArithmeticTests(JacobianCase):
    def test_binary(self):
        cases = {"Add": lambda a, b: a + b, "Sub": lambda a, b: a - b,
                 "Mul": lambda a, b: a*b, "Div": lambda a, b: a/b,
                 "Pow": lambda a, b: a**b}
        a, b = np.array([1.3, 0.7, 2.2]), np.array([0.4, 1.9, 0.8])
        for op, reference in cases.items():
            with self.subTest(op=op):
                model = build([helper.make_node(op, ["x", "b"], ["y"])],
                              [("x", [3])], [("y", [3])],
                              [numpy_helper.from_array(b, "b")])
                step = 1e-6
                expected = np.diag((reference(a + step, b) - reference(a - step, b))/(2*step))
                self.check(model, {"x": a}, expected, rtol=1e-5)

    def test_both_operands_differentiated(self):
        # x feeds a Mul twice: the reverse walk must sum the two contributions
        model = build([helper.make_node("Mul", ["x", "x"], ["y"])], [("x", [3])], [("y", [3])])
        x = np.array([1.3, -0.7, 2.2])
        self.check(model, {"x": x}, np.diag(2*x))

    def test_pow_with_differentiated_exponent(self):
        model = build([helper.make_node("Pow", ["a", "x"], ["y"])],
                      [("x", [3])], [("y", [3])],
                      [numpy_helper.from_array(np.array([1.3, 0.7, 2.2]), "a")])
        a, x = np.array([1.3, 0.7, 2.2]), np.array([0.4, 1.9, 0.8])
        self.check(model, {"x": x}, np.diag(a**x*np.log(a)))

    def test_fan_out_accumulates(self):
        # y = exp(x) + x*x: two paths to x, and both must arrive
        nodes = [helper.make_node("Exp", ["x"], ["e"]),
                 helper.make_node("Mul", ["x", "x"], ["s"]),
                 helper.make_node("Add", ["e", "s"], ["y"])]
        model = build(nodes, [("x", [3])], [("y", [3])])
        x = np.array([0.3, -1.1, 0.8])
        self.check(model, {"x": x}, np.diag(np.exp(x) + 2*x))


class BroadcastTests(JacobianCase):
    """Reverse mode's sharp edge: a contribution must be summed back over broadcast axes."""

    def test_row_vector_bias(self):
        b = RNG.standard_normal(3)
        model = build([helper.make_node("Add", ["x", "b"], ["y"])],
                      [("x", [2, 3])], [("y", [2, 3])], [numpy_helper.from_array(b, "b")])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, np.eye(6))

    def test_broadcast_operand_is_the_differentiated_one(self):
        # x of shape [3] against a [4, 3] constant: the adjoint of x sums over the 4 rows
        a = RNG.standard_normal((4, 3))
        model = build([helper.make_node("Mul", ["x", "a"], ["y"])],
                      [("x", [3])], [("y", [4, 3])], [numpy_helper.from_array(a, "a")])
        x = RNG.standard_normal(3)
        expected = np.zeros((12, 3))
        for row in range(4):
            expected[row*3:(row + 1)*3] = np.diag(a[row])
        self.check(model, {"x": x}, expected)

    def test_scalar_against_matrix(self):
        a = RNG.standard_normal((2, 3))
        model = build([helper.make_node("Div", ["a", "x"], ["y"])],
                      [("x", [])], [("y", [2, 3])], [numpy_helper.from_array(a, "a")])
        x = np.array(1.7)
        self.check(model, {"x": x}, (-a/x**2).reshape(6, 1))

    def test_singleton_axis(self):
        a = RNG.standard_normal((4, 3))
        model = build([helper.make_node("Mul", ["x", "a"], ["y"])],
                      [("x", [1, 3])], [("y", [4, 3])], [numpy_helper.from_array(a, "a")])
        self.check(model, {"x": RNG.standard_normal((1, 3))}, None)


class MatMulTests(JacobianCase):
    RANKS = [((3, 4), (4, 2)), ((4,), (4, 2)), ((3, 4), (4,)), ((4,), (4,)),
             ((5, 3, 4), (4, 2)), ((3, 4), (5, 4, 2)), ((5, 3, 4), (5, 4, 2))]

    def test_left_operand(self):
        for left, right in self.RANKS:
            with self.subTest(shapes=(left, right)):
                b = RNG.standard_normal(right)
                out = np.matmul(np.zeros(left), b).shape
                model = build([helper.make_node("MatMul", ["x", "b"], ["y"])],
                              [("x", list(left))], [("y", list(out))],
                              [numpy_helper.from_array(b, "b")])
                x = RNG.standard_normal(left)
                self.check(model, {"x": x}, None, fd_tol=1e-5)

    def test_right_operand(self):
        for left, right in self.RANKS:
            with self.subTest(shapes=(left, right)):
                a = RNG.standard_normal(left)
                out = np.matmul(a, np.zeros(right)).shape
                model = build([helper.make_node("MatMul", ["a", "x"], ["y"])],
                              [("x", list(right))], [("y", list(out))],
                              [numpy_helper.from_array(a, "a")])
                x = RNG.standard_normal(right)
                self.check(model, {"x": x}, None, fd_tol=1e-5)

    def test_both_operands(self):
        # y = x @ x, so the rule must handle a seeded tensor on either side at once
        model = build([helper.make_node("MatMul", ["x", "x"], ["y"])],
                      [("x", [3, 3])], [("y", [3, 3])])
        self.check(model, {"x": RNG.standard_normal((3, 3))}, None, fd_tol=1e-5)


class StructuralTests(JacobianCase):
    """Moving values around: the only care the seed axis needs is not to be counted from
    the back. These operations are also the ones the passes themselves emit, so the rules
    below are what makes a derivative model differentiable again."""

    def test_reshape(self):
        model = build([helper.make_node("Reshape", ["x", "s"], ["y"])],
                      [("x", [2, 3])], [("y", [3, 2])],
                      [helper.make_tensor("s", TensorProto.INT64, [2], [3, 2])])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, np.eye(6))

    def test_flatten(self):
        model = build([helper.make_node("Flatten", ["x"], ["y"], axis=1)],
                      [("x", [2, 3, 2])], [("y", [2, 6])])
        self.check(model, {"x": RNG.standard_normal((2, 3, 2))}, np.eye(12))

    def test_transpose(self):
        model = build([helper.make_node("Transpose", ["x"], ["y"], perm=[2, 0, 1])],
                      [("x", [2, 3, 4])], [("y", [4, 2, 3])])
        x = RNG.standard_normal((2, 3, 4))
        expected = np.eye(24).reshape((2, 3, 4, 24)).transpose(2, 0, 1, 3).reshape(24, 24)
        self.check(model, {"x": x}, expected)

    def test_default_transpose(self):
        model = build([helper.make_node("Transpose", ["x"], ["y"])],
                      [("x", [2, 3])], [("y", [3, 2])])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, None)

    def test_unsqueeze_and_squeeze(self):
        nodes = [helper.make_node("Unsqueeze", ["x", "a"], ["u"]),
                 helper.make_node("Exp", ["u"], ["e"]),
                 helper.make_node("Squeeze", ["e", "a"], ["y"])]
        model = build(nodes, [("x", [3])], [("y", [3])],
                      [helper.make_tensor("a", TensorProto.INT64, [1], [-1])])
        x = np.array([0.4, -0.9, 1.2])
        self.check(model, {"x": x}, np.diag(np.exp(x)))

    def test_expand(self):
        model = build([helper.make_node("Expand", ["x", "s"], ["y"])],
                      [("x", [1, 3])], [("y", [4, 3])],
                      [helper.make_tensor("s", TensorProto.INT64, [2], [4, 3])])
        x = RNG.standard_normal((1, 3))
        self.check(model, {"x": x}, np.tile(np.eye(3), (4, 1)))

    def test_concat(self):
        c = RNG.standard_normal((2, 2))
        model = build([helper.make_node("Concat", ["x", "c"], ["y"], axis=-1)],
                      [("x", [2, 3])], [("y", [2, 5])],
                      [numpy_helper.from_array(c, "c")])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, None)

    def test_concat_of_one_value_with_itself(self):
        model = build([helper.make_node("Concat", ["x", "x"], ["y"], axis=0)],
                      [("x", [3])], [("y", [6])])
        self.check(model, {"x": RNG.standard_normal(3)}, np.vstack([np.eye(3), np.eye(3)]))

    def test_reduce_sum(self):
        for keepdims, shape in ((1, [2, 1]), (0, [2])):
            with self.subTest(keepdims=keepdims):
                model = build([helper.make_node("ReduceSum", ["x", "a"], ["y"],
                                                keepdims=keepdims)],
                              [("x", [2, 3])], [("y", shape)],
                              [helper.make_tensor("a", TensorProto.INT64, [1], [-1])])
                expected = np.zeros((2, 6))
                expected[0, :3] = expected[1, 3:] = 1
                self.check(model, {"x": RNG.standard_normal((2, 3))}, expected)

    def test_reduce_sum_over_everything(self):
        model = build([helper.make_node("ReduceSum", ["x"], ["y"], keepdims=0)],
                      [("x", [2, 3])], [("y", [])])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, np.ones((1, 6)))

    def test_reduce_mean(self):
        model = build([helper.make_node("ReduceMean", ["x", "a"], ["y"], keepdims=0)],
                      [("x", [2, 3])], [("y", [2])],
                      [helper.make_tensor("a", TensorProto.INT64, [1], [1])])
        expected = np.zeros((2, 6))
        expected[0, :3] = expected[1, 3:] = 1/3
        self.check(model, {"x": RNG.standard_normal((2, 3))}, expected)

    def test_sum_of_several(self):
        c = RNG.standard_normal(3)
        model = build([helper.make_node("Sum", ["x", "x", "c"], ["y"])],
                      [("x", [3])], [("y", [3])], [numpy_helper.from_array(c, "c")])
        self.check(model, {"x": RNG.standard_normal(3)}, 2*np.eye(3))


class SelectionTests_(JacobianCase):
    """Rules whose derivative follows the branch the primal took."""

    def test_where(self):
        c = np.array([True, False, True, False])
        b = RNG.standard_normal(4)
        model = build([helper.make_node("Where", ["c", "x", "b"], ["y"])],
                      [("x", [4])], [("y", [4])],
                      [numpy_helper.from_array(c, "c"),
                       numpy_helper.from_array(b, "b")])
        self.check(model, {"x": RNG.standard_normal(4)}, np.diag(c.astype(float)))

    def test_where_on_the_false_branch(self):
        c = np.array([True, False, True, False])
        a = RNG.standard_normal(4)
        model = build([helper.make_node("Where", ["c", "a", "x"], ["y"])],
                      [("x", [4])], [("y", [4])],
                      [numpy_helper.from_array(c, "c"), numpy_helper.from_array(a, "a")])
        self.check(model, {"x": RNG.standard_normal(4)}, np.diag(1.0 - c))

    def test_where_broadcasts(self):
        c = np.array([[True], [False]])
        model = build([helper.make_node("Where", ["c", "x", "x"], ["y"])],
                      [("x", [2, 3])], [("y", [2, 3])], [numpy_helper.from_array(c, "c")])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, np.eye(6))

    def test_clip(self):
        model = build([helper.make_node("Clip", ["x", "lo", "hi"], ["y"])],
                      [("x", [5])], [("y", [5])],
                      [numpy_helper.from_array(np.array(-0.5), "lo"),
                       numpy_helper.from_array(np.array(0.5), "hi")])
        x = np.array([-1.2, -0.2, 0.1, 0.9, 0.3])
        self.check(model, {"x": x}, np.diag(((x > -0.5) & (x < 0.5)).astype(float)))

    def test_clip_bound_is_differentiated(self):
        model = build([helper.make_node("Clip", ["a", "x"], ["y"])],
                      [("x", [])], [("y", [4])],
                      [numpy_helper.from_array(np.array([-1.2, -0.2, 0.1, 0.9]), "a")])
        a = np.array([-1.2, -0.2, 0.1, 0.9])
        self.check(model, {"x": np.array(0.0)}, (a < 0.0).astype(float).reshape(4, 1))

    def test_clip_with_one_bound(self):
        model = build([helper.make_node("Clip", ["x", "lo"], ["y"])],
                      [("x", [4])], [("y", [4])],
                      [numpy_helper.from_array(np.array(0.0), "lo")])
        x = np.array([-1.2, -0.2, 0.1, 0.9])
        self.check(model, {"x": x}, np.diag((x > 0).astype(float)))

    def test_max_and_min(self):
        b = np.array([0.5, -0.5, 1.5, -1.5])
        for op in ("Max", "Min"):
            with self.subTest(op=op):
                model = build([helper.make_node(op, ["x", "b"], ["y"])],
                              [("x", [4])], [("y", [4])], [numpy_helper.from_array(b, "b")])
                x = np.array([0.7, -1.3, 0.25, 2.1])
                wins = (x >= b) if op == "Max" else (x <= b)
                self.check(model, {"x": x}, np.diag(wins.astype(float)))

    def test_max_of_three(self):
        b, c = np.array([0.5, -0.5, 1.5, -1.5]), np.array([0.6, -2.0, 3.0, 0.0])
        model = build([helper.make_node("Max", ["x", "b", "c"], ["y"])],
                      [("x", [4])], [("y", [4])],
                      [numpy_helper.from_array(b, "b"), numpy_helper.from_array(c, "c")])
        x = np.array([0.7, -1.3, 0.25, 2.1])
        wins = (x >= b) & (x >= c)
        self.check(model, {"x": x}, np.diag(wins.astype(float)))

    def test_prelu(self):
        slope = np.array([0.1, 0.2, 0.3, 0.4])
        model = build([helper.make_node("PRelu", ["x", "s"], ["y"])],
                      [("x", [4])], [("y", [4])], [numpy_helper.from_array(slope, "s")])
        x = np.array([0.7, -1.3, 0.25, -2.1])
        self.check(model, {"x": x}, np.diag(np.where(x >= 0, 1.0, slope)))

    def test_prelu_slope_is_differentiated(self):
        a = np.array([0.7, -1.3, 0.25, -2.1])
        model = build([helper.make_node("PRelu", ["a", "x"], ["y"])],
                      [("x", [4])], [("y", [4])], [numpy_helper.from_array(a, "a")])
        self.check(model, {"x": np.array([0.1, 0.2, 0.3, 0.4])},
                   np.diag(np.where(a >= 0, 0.0, a)))


class SoftmaxTests(JacobianCase):
    @staticmethod
    def softmax(x, axis=-1):
        shifted = np.exp(x - x.max(axis=axis, keepdims=True))
        return shifted/shifted.sum(axis=axis, keepdims=True)

    def test_softmax(self):
        for axis in (-1, 0, 1):
            with self.subTest(axis=axis):
                model = build([helper.make_node("Softmax", ["x"], ["y"], axis=axis)],
                              [("x", [2, 3])], [("y", [2, 3])])
                x = RNG.standard_normal((2, 3))
                s = self.softmax(x, axis)
                expected = np.zeros((6, 6))
                for i in range(6):
                    for j in range(6):
                        same = (i//3 == j//3) if axis in (-1, 1) else (i % 3 == j % 3)
                        if same:
                            expected[i, j] = s.ravel()[i]*((i == j) - s.ravel()[j])
                self.check(model, {"x": x}, expected)

    def test_log_softmax(self):
        model = build([helper.make_node("LogSoftmax", ["x"], ["y"], axis=-1)],
                      [("x", [2, 3])], [("y", [2, 3])])
        x = RNG.standard_normal((2, 3))
        s = self.softmax(x)
        expected = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                if i//3 == j//3:
                    expected[i, j] = (i == j) - s.ravel()[j]
        self.check(model, {"x": x}, expected)

    def test_softmax_in_a_network(self):
        w = RNG.standard_normal((4, 3))
        nodes = [helper.make_node("MatMul", ["x", "w"], ["h"]),
                 helper.make_node("Softmax", ["h"], ["p"], axis=-1),
                 helper.make_node("Log", ["p"], ["y"])]
        model = build(nodes, [("x", [2, 4])], [("y", [2, 3])],
                      [numpy_helper.from_array(w, "w")])
        self.check(model, {"x": RNG.standard_normal((2, 4))}, None, fd_tol=1e-5)


class GemmTests(JacobianCase):
    """What torch.nn.Linear exports to, in all four transpose combinations."""

    def test_transposes_and_scaling(self):
        for transA in (0, 1):
            for transB in (0, 1):
                with self.subTest(transA=transA, transB=transB):
                    a = RNG.standard_normal((4, 3) if transA else (3, 4))
                    c = RNG.standard_normal(2)
                    shape = [2, 4] if transB else [4, 2]
                    node = helper.make_node("Gemm", ["a", "x", "c"], ["y"], alpha=0.7,
                                            beta=1.3, transA=transA, transB=transB)
                    model = build([node], [("x", shape)], [("y", [3, 2])],
                                  [numpy_helper.from_array(a, "a"),
                                   numpy_helper.from_array(c, "c")])
                    self.check(model, {"x": RNG.standard_normal(shape)}, None, fd_tol=1e-5)

    def test_first_operand_and_bias(self):
        b = RNG.standard_normal((4, 2))
        node = helper.make_node("Gemm", ["x", "b", "c"], ["y"])
        model = build([node], [("x", [3, 4])], [("y", [3, 2])],
                      [numpy_helper.from_array(b, "b"),
                       numpy_helper.from_array(RNG.standard_normal(2), "c")])
        x = RNG.standard_normal((3, 4))
        expected = np.zeros((6, 12))
        for row in range(3):
            for column in range(2):
                expected[row*2 + column, row*4:(row + 1)*4] = b[:, column]
        self.check(model, {"x": x}, expected, fd_tol=1e-5)

    def test_bias_is_the_differentiated_operand(self):
        a, b = RNG.standard_normal((3, 4)), RNG.standard_normal((4, 2))
        node = helper.make_node("Gemm", ["a", "b", "x"], ["y"], beta=0.5)
        model = build([node], [("x", [2])], [("y", [3, 2])],
                      [numpy_helper.from_array(a, "a"), numpy_helper.from_array(b, "b")])
        self.check(model, {"x": RNG.standard_normal(2)}, np.tile(0.5*np.eye(2), (3, 1)))

    def test_gemm_without_a_bias(self):
        b = RNG.standard_normal((4, 2))
        model = build([helper.make_node("Gemm", ["x", "b"], ["y"])],
                      [("x", [3, 4])], [("y", [3, 2])], [numpy_helper.from_array(b, "b")])
        self.check(model, {"x": RNG.standard_normal((3, 4))}, None, fd_tol=1e-5)


class IndexingTests(JacobianCase):
    """Split undoes into Concat, Slice into Pad, Pad into Slice, Gather into a scatter-add."""

    def test_split(self):
        nodes = [helper.make_node("Split", ["x"], ["a", "b"], axis=0, num_outputs=2),
                 helper.make_node("Exp", ["a"], ["ea"]),
                 helper.make_node("Concat", ["ea", "b"], ["y"], axis=0)]
        model = build(nodes, [("x", [4])], [("y", [4])])
        x = np.array([0.4, -0.9, 1.2, 0.3])
        self.check(model, {"x": x}, np.diag([np.exp(x[0]), np.exp(x[1]), 1.0, 1.0]))

    def test_split_with_sizes(self):
        nodes = [helper.make_node("Split", ["x", "s"], ["a", "b"], axis=1),
                 helper.make_node("Concat", ["b", "a"], ["y"], axis=1)]
        model = build(nodes, [("x", [2, 5])], [("y", [2, 5])],
                      [helper.make_tensor("s", TensorProto.INT64, [2], [2, 3])])
        self.check(model, {"x": RNG.standard_normal((2, 5))}, None)

    def test_slice(self):
        model = build([helper.make_node("Slice", ["x", "s", "e", "a"], ["y"])],
                      [("x", [2, 5])], [("y", [2, 2])],
                      [helper.make_tensor("s", TensorProto.INT64, [1], [1]),
                       helper.make_tensor("e", TensorProto.INT64, [1], [3]),
                       helper.make_tensor("a", TensorProto.INT64, [1], [-1])])
        expected = np.zeros((4, 10))
        for row in range(2):
            for k in range(2):
                expected[row*2 + k, row*5 + 1 + k] = 1
        self.check(model, {"x": RNG.standard_normal((2, 5))}, expected)

    def test_slice_with_negative_bounds(self):
        model = build([helper.make_node("Slice", ["x", "s", "e"], ["y"])],
                      [("x", [6])], [("y", [3])],
                      [helper.make_tensor("s", TensorProto.INT64, [1], [-4]),
                       helper.make_tensor("e", TensorProto.INT64, [1], [-1])])
        expected = np.zeros((3, 6))
        expected[0, 2] = expected[1, 3] = expected[2, 4] = 1
        self.check(model, {"x": RNG.standard_normal(6)}, expected)

    def test_pad(self):
        model = build([helper.make_node("Pad", ["x", "p"], ["y"], mode="constant")],
                      [("x", [3])], [("y", [6])],
                      [helper.make_tensor("p", TensorProto.INT64, [2], [2, 1])])
        expected = np.zeros((6, 3))
        expected[2, 0] = expected[3, 1] = expected[4, 2] = 1
        self.check(model, {"x": RNG.standard_normal(3)}, expected)

    def test_pad_two_dimensions(self):
        model = build([helper.make_node("Pad", ["x", "p"], ["y"], mode="constant")],
                      [("x", [2, 3])], [("y", [4, 4])],
                      [helper.make_tensor("p", TensorProto.INT64, [4], [1, 0, 1, 1])])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, None)

    def test_tile(self):
        model = build([helper.make_node("Tile", ["x", "r"], ["y"])],
                      [("x", [2, 3])], [("y", [4, 3])],
                      [helper.make_tensor("r", TensorProto.INT64, [2], [2, 1])])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, np.vstack([np.eye(6),
                                                                        np.eye(6)]))

    def test_cumsum(self):
        model = build([helper.make_node("CumSum", ["x", "a"], ["y"])],
                      [("x", [4])], [("y", [4])],
                      [helper.make_tensor("a", TensorProto.INT64, [], [0])])
        self.check(model, {"x": RNG.standard_normal(4)}, np.tril(np.ones((4, 4))))

    def test_cumsum_reversed(self):
        model = build([helper.make_node("CumSum", ["x", "a"], ["y"], reverse=1)],
                      [("x", [4])], [("y", [4])],
                      [helper.make_tensor("a", TensorProto.INT64, [], [0])])
        self.check(model, {"x": RNG.standard_normal(4)}, np.triu(np.ones((4, 4))))

    def test_gather(self):
        indices = np.array([2, 0, 2], dtype=np.int64)
        model = build([helper.make_node("Gather", ["x", "i"], ["y"], axis=0)],
                      [("x", [4])], [("y", [3])], [numpy_helper.from_array(indices, "i")])
        expected = np.zeros((3, 4))
        for row, index in enumerate(indices):
            expected[row, index] = 1
        # index 2 is gathered twice: the adjoint has to accumulate, not overwrite
        self.check(model, {"x": RNG.standard_normal(4)}, expected)

    def test_gather_rows(self):
        indices = np.array([[1, 0], [1, 1]], dtype=np.int64)
        model = build([helper.make_node("Gather", ["x", "i"], ["y"], axis=0)],
                      [("x", [2, 3])], [("y", [2, 2, 3])],
                      [numpy_helper.from_array(indices, "i")])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, None)

    def test_gather_on_a_later_axis(self):
        indices = np.array([2, 0], dtype=np.int64)
        model = build([helper.make_node("Gather", ["x", "i"], ["y"], axis=1)],
                      [("x", [2, 3])], [("y", [2, 2])],
                      [numpy_helper.from_array(indices, "i")])
        expected = np.zeros((4, 6))
        for row in range(2):
            for k, index in enumerate(indices):
                expected[row*2 + k, row*3 + index] = 1
        self.check(model, {"x": RNG.standard_normal((2, 3))}, expected)

    def test_gather_with_negative_indices(self):
        indices = np.array([-1, 0], dtype=np.int64)
        model = build([helper.make_node("Gather", ["x", "i"], ["y"], axis=0)],
                      [("x", [3])], [("y", [2])], [numpy_helper.from_array(indices, "i")])
        expected = np.zeros((2, 3))
        expected[0, 2] = expected[1, 0] = 1
        self.check(model, {"x": RNG.standard_normal(3)}, expected)


class ReductionTests(JacobianCase):
    """Every one of these weights the elements and sums; the weight is the rule.

    Checked against the analytic weight rather than against finite differences: several of
    these kernels are single-precision inside ONNX Runtime even on a double tensor, which
    makes a 1e-6 difference quotient meaningless. Forward against reverse still applies.
    """

    #: op -> (primal, d/dx), both along the reduced axis
    CASES = {
        "ReduceMax": (lambda x: x.max(1), lambda x: (x == x.max(1, keepdims=True))*1.0),
        "ReduceMin": (lambda x: x.min(1), lambda x: (x == x.min(1, keepdims=True))*1.0),
        "ReduceLogSumExp": (lambda x: np.log(np.exp(x).sum(1)),
                            lambda x: np.exp(x)/np.exp(x).sum(1, keepdims=True)),
        "ReduceL1": (lambda x: np.abs(x).sum(1), np.sign),
        "ReduceL2": (lambda x: np.sqrt((x**2).sum(1)),
                     lambda x: x/np.sqrt((x**2).sum(1, keepdims=True))),
        "ReduceSumSquare": (lambda x: (x**2).sum(1), lambda x: 2*x),
        "ReduceProd": (lambda x: x.prod(1), lambda x: x.prod(1, keepdims=True)/x),
    }

    def test_reductions(self):
        x = RNG.standard_normal((2, 3)) + 1.5  # away from zero, where ReduceProd's y/x is not
        for op, (primal, derivative) in self.CASES.items():
            for keepdims in (0, 1):
                with self.subTest(op=op, keepdims=keepdims):
                    shape = [2, 1] if keepdims else [2]
                    model = build([helper.make_node(op, ["x", "a"], ["y"],
                                                    keepdims=keepdims)],
                                  [("x", [2, 3])], [("y", shape)],
                                  [helper.make_tensor("a", TensorProto.INT64, [1], [1])])
                    if not available(model, {"x": x}):
                        self.skipTest("%s has no double kernel in this ONNX Runtime" % op)
                    weights = derivative(x)
                    expected = np.zeros((2, 6))
                    for row in range(2):
                        expected[row, row*3:(row + 1)*3] = weights[row]
                    self.check(model, {"x": x}, expected, rtol=1e-7, differences=False)

    def test_reduce_max_shares_a_tie(self):
        # both entries win, so each takes half
        model = build([helper.make_node("ReduceMax", ["x", "a"], ["y"], keepdims=0)],
                      [("x", [2])], [("y", [])],
                      [helper.make_tensor("a", TensorProto.INT64, [1], [0])])
        self.check(model, {"x": np.array([1.0, 1.0])}, np.array([[0.5, 0.5]]),
                   differences=False)


class NormalizationTests(JacobianCase):
    def test_layer_normalization(self):
        scale, bias = RNG.standard_normal(4), RNG.standard_normal(4)
        model = build([helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"],
                                        axis=-1, epsilon=1e-5)],
                      [("x", [2, 4])], [("y", [2, 4])],
                      [numpy_helper.from_array(scale, "s"),
                       numpy_helper.from_array(bias, "b")])
        self.check(model, {"x": RNG.standard_normal((2, 4))}, None, fd_tol=1e-5)

    def test_layer_normalization_over_two_axes(self):
        model = build([helper.make_node("LayerNormalization", ["x", "s"], ["y"], axis=1)],
                      [("x", [2, 3, 4])], [("y", [2, 3, 4])],
                      [numpy_helper.from_array(RNG.standard_normal((3, 4)), "s")])
        self.check(model, {"x": RNG.standard_normal((2, 3, 4))}, None, fd_tol=1e-4)

    def test_layer_normalization_scale_and_bias(self):
        for index, name in ((1, "s"), (2, "b")):
            with self.subTest(operand=name):
                operands = ["x0", "s", "b"]
                operands[index] = "x"
                initializers = [numpy_helper.from_array(RNG.standard_normal((2, 4)), "x0")]
                for other, label in ((1, "s"), (2, "b")):
                    if other != index:
                        initializers.append(
                            numpy_helper.from_array(RNG.standard_normal(4), label))
                model = build([helper.make_node("LayerNormalization", operands, ["y"],
                                                axis=-1)],
                              [("x", [4])], [("y", [2, 4])], initializers)
                self.check(model, {"x": RNG.standard_normal(4)}, None, fd_tol=1e-5)

    def test_batch_normalization(self):
        scale, bias = RNG.standard_normal(3), RNG.standard_normal(3)
        mean, variance = RNG.standard_normal(3), np.abs(RNG.standard_normal(3)) + 0.5
        model = build([helper.make_node("BatchNormalization",
                                        ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-5)],
                      [("x", [2, 3, 4])], [("y", [2, 3, 4])],
                      [numpy_helper.from_array(v, n) for v, n in
                       [(scale, "s"), (bias, "b"), (mean, "m"), (variance, "v")]])
        gain = scale/np.sqrt(variance + 1e-5)
        expected = np.diag(np.tile(np.repeat(gain, 4), 2))
        self.check(model, {"x": RNG.standard_normal((2, 3, 4))}, expected, fd_tol=1e-5)

    def test_batch_normalization_scale(self):
        a = RNG.standard_normal((2, 3, 4))
        mean, variance = RNG.standard_normal(3), np.abs(RNG.standard_normal(3)) + 0.5
        model = build([helper.make_node("BatchNormalization",
                                        ["a", "x", "b", "m", "v"], ["y"])],
                      [("x", [3])], [("y", [2, 3, 4])],
                      [numpy_helper.from_array(v, n) for v, n in
                       [(a, "a"), (RNG.standard_normal(3), "b"), (mean, "m"),
                        (variance, "v")]])
        self.check(model, {"x": RNG.standard_normal(3)}, None, fd_tol=1e-5)


class ConvolutionTests(JacobianCase):
    """The seed axis folds into the batch for the input and into the output channels for
    the weight; the adjoint with respect to the input is the very same ConvTranspose."""

    #: ONNX Runtime has no double-precision Conv kernel, so these run in float32 -- which
    #: also means the finite-difference reference needs a coarse step and a loose tolerance.
    #: The sharp check here is forward against reverse: one uses Conv, the other
    #: ConvTranspose, so a layout mistake in either cannot survive their agreement.
    TOLERANCES = dict(rtol=2e-5, step=1e-3, fd_tol=5e-3)

    def conv(self, x_shape, w_shape, out_shape, differentiate="x", **attrs):
        weight = RNG.standard_normal(w_shape).astype(np.float32)
        bias = RNG.standard_normal(w_shape[0]).astype(np.float32)
        operands = {"x": ["x", "w", "b"], "w": ["a", "x", "b"], "b": ["a", "w", "x"]}
        names = operands[differentiate]
        given = {"a": RNG.standard_normal(x_shape).astype(np.float32),
                 "w": weight, "b": bias}
        shapes = {"x": x_shape, "w": w_shape, "b": [w_shape[0]]}
        initializers = [numpy_helper.from_array(given[n], n) for n in names if n != "x"]
        model = build([helper.make_node("Conv", names, ["y"], **attrs)],
                      [("x", list(shapes[differentiate]))], [("y", list(out_shape))],
                      initializers, dtype=TensorProto.FLOAT)
        return model, {"x": RNG.standard_normal(shapes[differentiate]).astype(np.float32)}

    def test_input(self):
        model, feeds = self.conv([1, 2, 5, 5], [3, 2, 3, 3], [1, 3, 5, 5],
                                 pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_input_strided_and_dilated(self):
        model, feeds = self.conv([1, 2, 9, 9], [3, 2, 3, 3], [1, 3, 4, 4],
                                 pads=[1, 1, 1, 1], strides=[2, 2], dilations=[2, 2],
                                 kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_input_grouped(self):
        model, feeds = self.conv([1, 4, 5, 5], [4, 2, 3, 3], [1, 4, 5, 5],
                                 pads=[1, 1, 1, 1], group=2, kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_input_one_dimensional(self):
        model, feeds = self.conv([2, 2, 7], [3, 2, 3], [2, 3, 7],
                                 pads=[1, 1], kernel_shape=[3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_weight(self):
        model, feeds = self.conv([2, 2, 5, 5], [3, 2, 3, 3], [2, 3, 5, 5],
                                 differentiate="w", pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_weight_strided_and_dilated(self):
        model, feeds = self.conv([2, 2, 9, 9], [3, 2, 3, 3], [2, 3, 4, 4],
                                 differentiate="w", pads=[1, 1, 1, 1], strides=[2, 2],
                                 dilations=[2, 2], kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)

    def test_bias(self):
        model, feeds = self.conv([2, 2, 5, 5], [3, 2, 3, 3], [2, 3, 5, 5],
                                 differentiate="b", pads=[1, 1, 1, 1], kernel_shape=[3, 3])
        self.check(model, feeds, None, **self.TOLERANCES)


class GatherNDTests(JacobianCase):
    def test_gather_nd(self):
        indices = np.array([[0, 1], [1, 0], [0, 1]], dtype=np.int64)
        model = build([helper.make_node("GatherND", ["x", "i"], ["y"])],
                      [("x", [2, 3])], [("y", [3])], [numpy_helper.from_array(indices, "i")])
        expected = np.zeros((3, 6))
        for row, (a, b) in enumerate(indices):
            expected[row, a*3 + b] = 1
        self.check(model, {"x": RNG.standard_normal((2, 3))}, expected)

    def test_gather_nd_slices(self):
        indices = np.array([[1], [0], [1]], dtype=np.int64)
        model = build([helper.make_node("GatherND", ["x", "i"], ["y"])],
                      [("x", [2, 3])], [("y", [3, 3])],
                      [numpy_helper.from_array(indices, "i")])
        self.check(model, {"x": RNG.standard_normal((2, 3))}, None)


class NetworkTests(JacobianCase):
    """An assembled network, which is what the rules exist for."""

    @staticmethod
    def mlp(dtype=TensorProto.DOUBLE, opset=18, batch=2):
        kind = np.float32 if dtype == TensorProto.FLOAT else np.float64
        w1, b1 = RNG.standard_normal((4, 5)).astype(kind), RNG.standard_normal(5).astype(kind)
        w2, b2 = RNG.standard_normal((5, 3)).astype(kind), RNG.standard_normal(3).astype(kind)
        nodes = [helper.make_node("MatMul", ["x", "w1"], ["h1"]),
                 helper.make_node("Add", ["h1", "b1"], ["p1"]),
                 helper.make_node("Tanh", ["p1"], ["a1"]),
                 helper.make_node("MatMul", ["a1", "w2"], ["h2"]),
                 helper.make_node("Add", ["h2", "b2"], ["y"])]
        initializers = [numpy_helper.from_array(v, n) for v, n in
                        [(w1, "w1"), (b1, "b1"), (w2, "w2"), (b2, "b2")]]
        model = build(nodes, [("x", [batch, 4])], [("y", [batch, 3])], initializers,
                      opset=opset, dtype=dtype)
        return model, (w1, b1, w2, b2)

    def test_mlp(self):
        model, (w1, b1, w2, b2) = self.mlp()
        x = RNG.standard_normal((2, 4))
        # d y/d x, row by row: w1 . diag(1 - tanh^2) . w2
        blocks = []
        for row in x:
            gain = 1 - np.tanh(row @ w1 + b1)**2
            blocks.append(w1 @ np.diag(gain) @ w2)
        expected = np.zeros((6, 8))
        for i, block in enumerate(blocks):
            expected[i*3:(i + 1)*3, i*4:(i + 1)*4] = block.T
        self.check(model, {"x": x}, expected, fd_tol=1e-5)

    def test_older_opset(self):
        # opset 12 spells Unsqueeze/Squeeze/ReduceSum axes as attributes, not inputs
        model, _ = self.mlp(opset=12)
        self.check(model, {"x": RNG.standard_normal((2, 4))}, None, fd_tol=1e-5)

    def test_single_precision(self):
        model, _ = self.mlp(dtype=TensorProto.FLOAT)
        x = RNG.standard_normal((2, 4)).astype(np.float32)
        fwd = jacobian_forward(model, {"x": x}, "x", "y")
        rev = jacobian_reverse(model, {"x": x}, "x", "y")
        np.testing.assert_allclose(fwd, rev, rtol=1e-6, atol=1e-6)

    def test_primal_outputs_are_kept(self):
        model, _ = self.mlp()
        x = RNG.standard_normal((2, 4))
        primal = run(model, {"x": x})["y"]
        np.testing.assert_allclose(
            run(forward(model), {"x": x, "fwd_x": np.zeros((2, 4))})["y"], primal)
        np.testing.assert_allclose(
            run(reverse(model), {"x": x, "adj_y": np.zeros((2, 3))})["y"], primal)

    def test_several_seeds_in_one_evaluation(self):
        model, _ = self.mlp()
        x = RNG.standard_normal((2, 4))
        directions = RNG.standard_normal((2, 4, 3))
        packed = run(forward(model), {"x": x, "fwd_x": pack(directions, (2, 4))})["fwd_y"]
        got = unpack(packed, (2, 3), 3)
        jacobian = jacobian_forward(model, {"x": x}, "x", "y")
        for k in range(3):
            np.testing.assert_allclose(got[..., k].ravel(),
                                       jacobian @ directions[..., k].ravel(),
                                       rtol=1e-11, atol=1e-11)


class SymbolicShapeTests(JacobianCase):
    """A dynamic batch dimension: the broadcast reduction has to be computed at run time."""

    def test_dynamic_batch(self):
        b = RNG.standard_normal(3)
        w = RNG.standard_normal((4, 3))
        nodes = [helper.make_node("MatMul", ["x", "w"], ["h"]),
                 helper.make_node("Add", ["h", "b"], ["y"])]
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("x", TensorProto.DOUBLE, ["batch", 4])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, ["batch", 3])],
            [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = IR_VERSION
        onnx.checker.check_model(model)
        x = RNG.standard_normal((2, 4))
        seeds = pack(np.eye(8).reshape(2, 4, 8), (2, 4))
        got = unpack(run(forward(model), {"x": x, "fwd_x": seeds})["fwd_y"], (2, 3), 8)
        adjoints = pack(np.eye(6).reshape(2, 3, 6), (2, 3))
        adj = unpack(run(reverse(model), {"x": x, "adj_y": adjoints})["adj_x"], (2, 4), 6)
        np.testing.assert_allclose(got.reshape(6, 8), adj.reshape(8, 6).T,
                                   rtol=1e-12, atol=1e-12)

    def test_dynamic_broadcast_operand(self):
        # x is [1, 3] against a [batch, 3] input: the adjoint sums over a run-time batch
        nodes = [helper.make_node("Mul", ["x", "a"], ["y"])]
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("x", TensorProto.DOUBLE, ["one", 3]),
             helper.make_tensor_value_info("a", TensorProto.DOUBLE, ["batch", 3])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, ["batch", 3])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = IR_VERSION
        onnx.checker.check_model(model)
        x, a = RNG.standard_normal((1, 3)), RNG.standard_normal((4, 3))
        seeds = pack(np.eye(12).reshape(4, 3, 12), (4, 3))
        packed = run(reverse(model, inputs=["x"]),
                     {"x": x, "a": a, "adj_y": seeds})["adj_x"]
        adj = unpack(packed, (1, 3), 12)
        expected = np.zeros((3, 12))
        for row in range(4):
            for column in range(3):
                expected[column, row*3 + column] = a[row, column]
        np.testing.assert_allclose(adj.reshape(3, 12), expected, rtol=1e-12, atol=1e-12)


class CompositionTests(JacobianCase):
    """A derivative model is an ONNX model, so it differentiates again."""

    def test_forward_over_adjoint_is_the_hessian(self):
        # f(x) = sum_i tanh(x_i)^2 . a_i: the adjoint of a scalar output is the gradient,
        # and the forward derivative of that adjoint model is the Hessian
        a = RNG.standard_normal(3)
        nodes = [helper.make_node("Tanh", ["x"], ["t"]),
                 helper.make_node("Mul", ["t", "t"], ["q"]),
                 helper.make_node("Mul", ["q", "a"], ["w"]),
                 helper.make_node("ReduceSum", ["w"], ["y"], keepdims=0)]
        model = build(nodes, [("x", [3])], [("y", [])],
                      [numpy_helper.from_array(a, "a")])
        x = np.array([0.4, -0.9, 1.2])
        adjoint = reverse(model)
        hessian_model = forward(adjoint, inputs=["x"], outputs=["adj_x"])
        # only x is seeded, so the adjoint weight stays a plain input of the Hessian model
        got = run(hessian_model, {"x": x, "adj_y": np.ones((1, 1)),
                                  "fwd_x": np.eye(3)})["fwd_adj_x"]
        t, s = np.tanh(x), 1 - np.tanh(x)**2
        np.testing.assert_allclose(got.reshape(3, 3), np.diag(a*(2*s*s - 4*t*t*s)),
                                   rtol=1e-10, atol=1e-10)

    def test_a_derivative_model_still_checks(self):
        model, _ = NetworkTests.mlp()
        for derivative in (forward(model), reverse(model)):
            onnx.checker.check_model(derivative)
            onnx.shape_inference.infer_shapes(derivative, strict_mode=True)

    def test_repeated_differentiation_names_itself(self):
        # CasADi's rule: a model already carrying `fwd_x` gets `fwd2_`/`nfwd2` next
        nodes = [helper.make_node("Exp", ["x"], ["y"])]
        model = build(nodes, [("x", [2])], [("y", [2])])
        first = forward(model)
        second = forward(first, inputs=["x"], outputs=["fwd_y"])
        self.assertIn("fwd2_x", [v.name for v in second.graph.input])
        self.assertIn("fwd2_fwd_y", [v.name for v in second.graph.output])
        seed = next(v for v in second.graph.input if v.name == "fwd2_x")
        self.assertEqual(seed.type.tensor_type.shape.dim[-1].dim_param, "nfwd2")
        x = np.array([0.3, -0.8])
        got = run(second, {"x": x, "fwd_x": np.ones((2, 1)), "fwd2_x": np.ones((2, 1))})
        np.testing.assert_allclose(got["fwd2_fwd_y"].reshape(2), np.exp(x), rtol=1e-12)


try:
    from onnx_complex2real import complex_step
except ImportError:  # optional: the two packages check each other when both are present
    complex_step = None


@unittest.skipIf(complex_step is None, "onnx-complex2real is not installed")
class ComplexStepTests(unittest.TestCase):
    """The independent reference with no truncation error: f'(x)v = Im f(x + i h v)/h.

    Neither package knows anything about the other's method -- one differentiates the graph
    by a rule table, the other evaluates the untouched graph off the real axis -- so their
    agreement to machine precision is a real check on both.
    """

    @staticmethod
    def polynomial():
        """A network of Add/Mul/MatMul only: the complex lowering of a transcendental
        needs Sin/Cos, which ONNX Runtime registers in single precision only."""
        w, b = RNG.standard_normal((4, 3)), RNG.standard_normal(3)
        nodes = [helper.make_node("MatMul", ["x", "w"], ["h"]),
                 helper.make_node("Add", ["h", "b"], ["p"]),
                 helper.make_node("Mul", ["p", "p"], ["q"]),
                 helper.make_node("Mul", ["q", "p"], ["y"])]
        return build(nodes, [("x", [2, 4])], [("y", [2, 3])],
                     [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])

    def test_network(self):
        model = self.polynomial()
        x = RNG.standard_normal((2, 4))
        direction = RNG.standard_normal((2, 4))
        step = 1e-20
        twin = complex_step(model)
        reference = run(twin, {"x": x, "im_x": step*direction})["im_y"]/step
        seeds = pack(direction[..., None], (2, 4))
        got = unpack(run(forward(model), {"x": x, "fwd_x": seeds})["fwd_y"], (2, 3), 1)
        np.testing.assert_allclose(got[..., 0], reference, rtol=1e-12, atol=1e-14)

    def test_hessian_against_the_complex_step_of_the_adjoint(self):
        # the complex step of an adjoint model is a forward-over-adjoint product too
        model = self.polynomial()
        x, direction = RNG.standard_normal((2, 4)), RNG.standard_normal((2, 4))
        weights = RNG.standard_normal((2, 3))
        adjoint = reverse(model)
        step = 1e-20
        twin = complex_step(adjoint, ["x"])
        seeded_weights = pack(weights[..., None], (2, 3))
        reference = run(twin, {"x": x, "im_x": step*direction,
                               "adj_y": seeded_weights})["im_adj_x"]/step
        hessian = forward(adjoint, inputs=["x"], outputs=["adj_x"])
        got = run(hessian, {"x": x, "adj_y": seeded_weights,
                            "fwd_x": pack(direction[..., None], (2, 4))})["fwd_adj_x"]
        np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-12)


class SelectionTests(JacobianCase):
    def test_constant_output_gets_a_zero_derivative(self):
        nodes = [helper.make_node("Exp", ["x"], ["y"]),
                 helper.make_node("Identity", ["c"], ["z"])]
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [2])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [2]),
             helper.make_tensor_value_info("z", TensorProto.DOUBLE, [3])],
            [numpy_helper.from_array(np.ones(3), "c")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = IR_VERSION
        out = run(forward(model), {"x": np.zeros(2), "fwd_x": np.ones((2, 4))})
        np.testing.assert_allclose(out["fwd_z"], np.zeros((3, 4)))
        self.assertEqual(out["fwd_z"].shape, (3, 4))  # rank-1 values need no repacking

    def test_selected_inputs_only(self):
        nodes = [helper.make_node("Mul", ["x", "u"], ["y"])]
        model = build(nodes, [("x", [3]), ("u", [3])], [("y", [3])])
        derivative = forward(model, inputs=["x"])
        self.assertEqual([v.name for v in derivative.graph.input], ["x", "u", "fwd_x"])
        out = run(derivative, {"x": np.ones(3), "u": np.arange(3.0),
                               "fwd_x": np.eye(3)})
        np.testing.assert_allclose(out["fwd_y"], np.diag(np.arange(3.0)))

    def test_unknown_input_is_rejected(self):
        model = build([helper.make_node("Exp", ["x"], ["y"])], [("x", [2])], [("y", [2])])
        with self.assertRaises(ValueError):
            forward(model, inputs=["nope"])

    def test_operation_without_a_rule(self):
        nodes = [helper.make_node("Resize", ["x", "", "s"], ["y"], mode="linear")]
        model = build(nodes, [("x", [3])], [("y", [6])],
                      [numpy_helper.from_array(np.array([2.0], dtype=np.float32), "s")])
        with self.assertRaises(UnsupportedOperator):
            forward(model)
        with self.assertRaises(UnsupportedOperator):
            reverse(model)

    def test_operation_without_a_rule_is_fine_off_the_path(self):
        # Resize has no rule, but nothing differentiated reaches it
        nodes = [helper.make_node("Resize", ["c", "", "r"], ["s"], mode="linear"),
                 helper.make_node("Exp", ["x"], ["y"])]
        graph = helper.make_graph(
            nodes, "g",
            [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [3])],
            [helper.make_tensor_value_info("y", TensorProto.DOUBLE, [3]),
             helper.make_tensor_value_info("s", TensorProto.DOUBLE, [3])],
            [numpy_helper.from_array(np.arange(3.0), "c"),
             numpy_helper.from_array(np.array([1.0], dtype=np.float32), "r")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        model.ir_version = IR_VERSION
        derivative = forward(model, outputs=["y"])
        self.assertIn("fwd_y", [v.name for v in derivative.graph.output])


if __name__ == "__main__":
    unittest.main()
