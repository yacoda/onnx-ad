"""Export a PyTorch model's primal to ONNX, then build the whole derivative family here.

No PyTorch AD is involved: `torch.onnx.export` produces `f.onnx`, and `onnx_ad.family`
produces `adj_f.onnx` and `fwd_adj_f.onnx` from that graph. CasADi's ONNX backend discovers
the siblings by name and can then build gradients, Jacobians and exact Hessians.

    python examples/torch_to_casadi.py generated

Needs torch to export. The CasADi half needs a build with WITH_ONNX and WITH_ONNX_RUNTIME,
and CASADI_ONNXRUNTIME_LIB pointing at libonnxruntime.so.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from onnx_ad import family

torch.manual_seed(0)
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "generated")

network = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(),
                        nn.Linear(16, 2)).eval()
for parameter in network.parameters():
    parameter.requires_grad_(False)


class Primal(nn.Module):
    """A rank-1 boundary: x of length 2 in, y of length 2 out.

    CasADi reads a rank-1 ONNX tensor as an n-by-1 column and its seeds as n-by-nseed, which
    is exactly what onnx-ad emits, so nothing has to be repacked. A rank-2 boundary works
    too -- the seeds are then packed into the column count, as CasADi expects.
    """

    def forward(self, x):
        return network(x.reshape(1, -1)).reshape(-1)


folder.mkdir(parents=True, exist_ok=True)
torch.onnx.export(Primal(), (torch.zeros(2),), str(folder / "f.onnx"),
                  input_names=["x"], output_names=["y"], dynamo=True, opset_version=18,
                  # without this the weights land in f.onnx.data, which CasADi cannot follow:
                  # it hands the model to ONNX Runtime as bytes
                  external_data=False)

for path in family(onnx.load(folder / "f.onnx"), str(folder / "f.onnx")):
    model = onnx.load(path)
    print("%-16s %s -> %s" % (
        Path(path).name,
        [v.name for v in model.graph.input], [v.name for v in model.graph.output]))

point = np.array([0.2, -0.1], dtype=np.float32)
weight = np.array([1.0, -0.5], dtype=np.float32)
x = torch.tensor(point)
np.savez(folder / "reference.npz", x=point, w=weight, y=Primal()(x).numpy(),
         jacobian=torch.autograd.functional.jacobian(Primal(), x).numpy(),
         hessian=torch.autograd.functional.hessian(
             lambda v: (Primal()(v)*torch.tensor(weight)).sum(), x).numpy())
print("\nPyTorch reference written to", folder / "reference.npz")
print("Now run:  python examples/casadi_side.py", folder)
