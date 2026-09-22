"""Consume the family from CasADi and check it against the PyTorch reference. No torch here.

    CASADI_ONNXRUNTIME_LIB=/path/to/libonnxruntime.so \
        python examples/casadi_side.py generated
"""
import sys

import casadi as ca
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "generated"
reference = np.load(folder + "/reference.npz")
x, w = reference["x"].astype(float), reference["w"].astype(float)
jacobian, hessian = reference["jacobian"].astype(float), reference["hessian"].astype(float)


def report(label, got, expected):
    got, expected = np.array(got).squeeze(), np.array(expected).squeeze()
    error = np.abs(got - expected).max()
    print("%-34s max|casadi - torch| = %.2e   %s"
          % (label, error, "OK" if error < 2e-6 else "MISMATCH"))
    return error < 2e-6


# CasADi finds adj_f.onnx beside f.onnx, and fwd_adj_f.onnx beside that
f = ca.GraphBuilder(folder + "/f.onnx").create("f")
checks = [report("f(x)", f(x), reference["y"]),
          report("reverse(1): J^T w", f.reverse(1)(x, f(x), ca.DM(w)), jacobian.T @ w),
          report("reverse(2): J^T", f.reverse(2)(x, f(x), ca.DM.eye(2)), jacobian.T)]

symbol = ca.MX.sym("v", 2)
checks.append(report("jacobian (built from adj_f)",
                     ca.Function("jf", [symbol], [ca.jacobian(f(symbol), symbol)])(x),
                     jacobian))
checks.append(report("hessian of w.f (fwd_adj_f)",
                     ca.Function("hf", [symbol],
                                 [ca.hessian(ca.dot(f(symbol), ca.DM(w)), symbol)[0]])(x),
                     hessian))
print("\n" + ("PASS" if all(checks) else "FAIL"))
sys.exit(0 if all(checks) else 1)
