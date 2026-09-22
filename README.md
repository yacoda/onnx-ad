# onnx-ad

Automatic differentiation **of ONNX graphs**, by source-code transformation: given a model,
produce new ONNX models that compute its Jacobian-vector and vector-Jacobian products. Pure
Python over the `onnx` protobuf, no runtime dependency, no framework in the loop.

```sh
python -m pip install onnx-ad
```

```python
import onnx
from onnx_ad import forward, reverse, family

model = onnx.load("f.onnx")                      # inputs x -> outputs y
onnx.save(forward(model), "fwd_f.onnx")          # + fwd_x -> + fwd_y = J . fwd_x
onnx.save(reverse(model), "adj_f.onnx")          # + adj_y -> + adj_x = J^T . adj_y

family(model, "generated/f.onnx")                # the whole set CasADi discovers
```

A derivative model keeps the original signature as a prefix and appends the seeds, so the
primal outputs stay available. Any number of seed directions is evaluated in a single pass.

## Why differentiate ONNX

| Route | What it costs |
| --- | --- |
| Differentiate in PyTorch, then export | derivative graphs only for models that came from PyTorch; forward mode goes through `jvp`/`vmap`, which is where export breaks; every derivative order needs another trace |
| Complex step (`Im f(x + i h v)/h`) | forward mode only, one evaluation per direction, and a convention rather than an identity at piecewise operations |
| **This** | needs a rule per ONNX operation — but then any ONNX model has derivatives, from any producer, at any order |

The third route is the one with no ceiling. A Jacobian-vector-product graph is itself an
ONNX model, so it can be differentiated again, and consumers need no new capability.

## Composition: second derivatives for free

The passes keep the primal graph and emit only ordinary ONNX operations, so their own output
is differentiable. Forward-over-adjoint — the exact-Hessian building block — is the two
passes composed, with no special casing:

```python
adjoint = reverse(model)                 # x, adj_y -> adj_x
hessian = forward(adjoint)               # + fwd_x, fwd_adj_y -> + fwd_adj_x
```

Repeated differentiation names itself the way CasADi's `diff_prefix` does: a model that
already carries `fwd_x` gets `fwd2_`/`nfwd2` next, so `forward(forward(model))` needs no
arguments.

## CasADi

The conventions are CasADi's, by default, so an emitted family drops into its ONNX backend
with nothing to adapt:

* **names** — `fwd_<x>`, `adj_<y>`, then `fwd2_`, `adj2_`, from the same rule
  `FunctionInternal::diff_prefix` applies, and seed dimensions `nfwd`, `nadj`, `nfwd2`;
* **layout** — CasADi reads an ONNX tensor as a matrix (rank 0/1 as a column, rank 2
  directly, higher ranks flattened) and wants the seeds of an `r`-by-`c` value as one
  `r`-by-`(nseed*c)` matrix. That is what `layout="casadi"` emits. Pass `layout="onnx"` for
  the internal form instead, where the seed count is a trailing axis on the primal's own
  shape;
* **files** — `family` writes `f.onnx`, `adj_f.onnx` and `fwd_adj_f.onnx`, the
  `<kind>_<filename>` siblings the backend looks for beside a model.

```python
f = casadi.GraphBuilder("generated/f.onnx").create("f")
f.reverse(1)(x, f(x), w)                            # from adj_f.onnx
casadi.hessian(casadi.dot(f(v), w), v)              # from fwd_adj_f.onnx
```

`examples/torch_to_casadi.py` exports a PyTorch model's **primal only** and generates the
rest here; `examples/casadi_side.py` consumes it and checks gradients, Jacobians and exact
Hessians against PyTorch. Verified against a CasADi build with `WITH_ONNX=ON` and
`WITH_ONNX_RUNTIME=ON`, with `CASADI_ONNXRUNTIME_LIB` pointing at `libonnxruntime.so`.

Two traps when the primal comes from `torch.onnx.export`: pass `external_data=False`, or the
weights land in a sidecar `f.onnx.data` that CasADi cannot follow (it hands the model to ONNX
Runtime as bytes); and CasADi needs the *complete* forward signature of the adjoint, which is
why `family` seeds `adj_y` as well as `x`.

Deliberately **not** offered: a `jacobian` pass. CasADi builds dense Jacobians from the
adjoint itself, and a `jac_` sibling would only duplicate that.

## What it emits

Two walks over one rule table. Forward carries a tangent per value in graph order; reverse
walks backwards, accumulating a contribution per value and summing where a value has several
consumers. A value with no derivative is *absent* rather than zero, which is what keeps the
emitted graph the size of the primal one — every weight in a network is such a value, and
costs nothing.

Nonlinear rules read the primal tensors they need straight from the primal graph rather than
recomputing them: the tangent of `Tanh` is `(1 - y*y) * t`, with `y` the tensor the primal
`Tanh` already produced. The reverse model therefore stays a plain function of `(x, adj_y)` —
no "uses output" convention, nothing to wire up.

Reverse mode's sharp edge is broadcasting: a contribution arrives shaped like the *result*
and must be summed back over the axes the operand was broadcast along. Where the operand's
shape is declared this is a static axis list; where it is symbolic the axes are computed at
run time, so a dynamic batch dimension survives.

### Operations with rules

**Arithmetic** `Add` `Sub` `Mul` `Div` `Neg` `Pow` `Sum` `Mean` `Identity`

**Elementwise** `Exp` `Log` `Sqrt` `Reciprocal` `Abs` `Sign` `Sin` `Cos` `Tan` `Sinh`
`Cosh` `Asin` `Acos` `Atan` `Asinh` `Acosh` `Atanh` `Erf` `Tanh` `Sigmoid` `Relu`
`LeakyRelu` `Elu` `Selu` `Celu` `PRelu` `ThresholdedRelu` `Softplus` `Softsign` `Shrink`
`HardSigmoid` `HardSwish` `Mish` `Gelu` (exact and `tanh`)

**Linear algebra** `MatMul` `Gemm` `Conv` (strided, dilated, grouped, 1-D and up; weight and
bias too)

**Shape** `Reshape` `Flatten` `Transpose` `Squeeze` `Unsqueeze` `Expand` `Concat` `Split`
`Slice` `Pad` `Tile` `Gather` `GatherND` `CumSum`

**Reductions** `ReduceSum` `ReduceMean` `ReduceMax` `ReduceMin` `ReduceProd`
`ReduceLogSumExp` `ReduceL1` `ReduceL2` `ReduceSumSquare`

**Selection** `Where` `Clip` `Min` `Max`

**Networks** `Softmax` `LogSoftmax` `LayerNormalization` `BatchNormalization` (inference)
`Dropout` (inference) `Cast` `CastLike`

**Zero derivative, and allowed to consume differentiated values** `Shape` `Size` `NonZero`
`Equal` `Greater` `Less` `GreaterOrEqual` `LessOrEqual` `And` `Or` `Xor` `Not` the bitwise
family `ArgMax` `ArgMin` `IsNaN` `IsInf` `Floor` `Ceil` `Round` `Hardmax` `OneHot` `Det`

An operation without a rule is an error **only when a differentiated value reaches it** — a
`Resize` on a constant branch is fine.

Still missing: pooling (`MaxPool`, `AveragePool`, `GlobalAveragePool`), `ConvTranspose` as a
primal, `Einsum`, `Resize`, the scatter operations, `LpNormalization`,
`InstanceNormalization` and `GroupNormalization`. Control flow (`If`, `Scan`, `Loop`) has a
design but no rules yet — see [CONTROL-FLOW.md](CONTROL-FLOW.md). Out of scope: sparsity and
training-mode operations.

### Against PyTorch

Every one of these exports (`torch.onnx.export`, `dynamo=True`, opset 18) differentiates in
both modes, and the resulting Jacobian matches `torch.autograd.functional.jacobian` to
float32 precision:

| Model | Operations | forward | reverse |
| --- | --- | ---: | ---: |
| MLP | `Gemm` `Tanh` | 6e-8 | 6e-8 |
| GELU MLP | `Gemm` `Erf` `Mul` `Div` `Add` | 2e-7 | 1e-7 |
| SiLU MLP | `Gemm` `Sigmoid` `Mul` | 8e-8 | 8e-8 |
| Attention block | `Gemm` `MatMul` `Softmax` `LayerNormalization` `Reshape` `Transpose` | 2e-7 | 2e-7 |
| Convolutional net | `Conv` `Relu` `Gemm` `Reshape` | 1e-7 | 8e-8 |
| Indexing and reductions | `GatherND` `Slice` `Clip` `ReduceMax` `ReduceProd` `ReduceL2` `ReduceLogSumExp` | 1e-7 | 1e-7 |
| GRU cell | `Gemm` `Split` `Sigmoid` `Tanh` | 6e-8 | 6e-8 |

## Running the result

Reverse mode emits `Transpose` feeding `MatMul`, which ONNX Runtime's extended optimizer
fuses into `com.microsoft.FusedMatMul` — a kernel registered for `float` only. On a
double-precision model, load with the fusions off:

```python
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
session = ort.InferenceSession("adj_f.onnx", options)
```

## Command line

```sh
onnx-ad forward f.onnx fwd_f.onnx
onnx-ad reverse f.onnx adj_f.onnx --inputs x --outputs y
onnx-ad family  f.onnx generated/f.onnx
```

## Testing

Every rule is executed through ONNX Runtime and compared against references that know nothing
of the rule table: central finite differences of the primal model, an analytic Jacobian
written in numpy, and forward against reverse — `J` and `J^T` come from separate walks over
separate rules, so their agreement to machine precision is a real check.

No PyTorch is involved in the unit tests; comparisons against it belong in an integration
suite, not here.

```sh
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
```

## License

MIT. Releasing is documented in [RELEASE.md](RELEASE.md): PyPI Trusted Publishing, so no API
token exists, and every artifact carries a PEP 740 provenance attestation.
