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

That only works if the rule set is closed under its own output — the reverse of `Gather` is a
`ScatterND` and the reverse of `Conv` a `ConvTranspose`, so both have rules too.

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

### Operations

All but a handful of the operations in the default ONNX domain that can carry a
floating-point derivative are covered, in one of three ways; the handful is listed last. An
operation needs covering only **when a differentiated value
reaches it** — anything on a constant branch is left alone — and every derivative model is
itself closed under differentiation: the tests check that each operation a derivative
model uses has rules of its own, which is what makes `forward(reverse(model))` always work.

**By a rule** — 113 operations, both modes, plus 37 whose derivative is zero:

* *Arithmetic and elementwise* — `Add` `Sub` `Mul` `Div` `Neg` `Pow` `Mod` `Sum` `Mean`
  `Identity` `Exp` `Log` `Sqrt` `Reciprocal` `Abs` `Sin` `Cos` `Tan` `Sinh` `Cosh` `Asin`
  `Acos` `Atan` `Asinh` `Acosh` `Atanh` `Erf` `Tanh` `Sigmoid` `Relu` `LeakyRelu` `Elu`
  `Selu` `Celu` `PRelu` `ThresholdedRelu` `Softplus` `Softsign` `Shrink` `HardSigmoid`
  `HardSwish` `Mish` `Gelu` `Cast` `CastLike`
* *Linear algebra and convolution* — `MatMul` `Gemm` `Einsum` `Conv` `ConvTranspose`
  `Col2Im` `DFT`
* *Pooling and resampling* — `MaxPool` `AveragePool` `LpPool` `GlobalAveragePool`
  `GlobalMaxPool` `GlobalLpPool` `MaxUnpool` `Resize` `Upsample` `RoiAlign` (average mode,
  in the image)
* *Normalization* — `BatchNormalization` (inference) `LayerNormalization`
  `InstanceNormalization` `LpNormalization` `Softmax` `LogSoftmax` `Dropout` (inference)
* *Reductions and scans* — `ReduceSum` `ReduceMean` `ReduceMax` `ReduceMin` `ReduceProd`
  `ReduceLogSumExp` `ReduceL1` `ReduceL2` `ReduceSumSquare` `CumSum` `CumProd` `TopK`
* *Shape and indexing* — `Reshape` `Flatten` `Transpose` `Squeeze` `Unsqueeze` `Expand`
  `Concat` `Split` `Slice` `Pad` `Tile` `Trilu` `Range` `ReverseSequence` `DepthToSpace`
  `SpaceToDepth` `Gather` `GatherElements` `GatherND` `Scatter` `ScatterElements`
  `ScatterND` `Compress` `Unique` `Where` `Clip` `Min` `Max` `DequantizeLinear` (in the
  scale)
* *Control flow* — `If` `Scan` `Loop`, see below
* *Zero derivative* — the comparisons, logical and bitwise operations, `Shape` `Size`
  `NonZero` `ArgMax` `ArgMin` `IsNaN` `IsInf` `IsFinite` `Sign` `Floor` `Ceil` `Round` `Hardmax`
  `OneHot` `Det` `ConstantOfShape` `EyeLike` the random operations `QuantizeLinear`
  `NonMaxSuppression` `BitCast`

**By the specification's own function body** — `Attention` `RotaryEmbedding`
`RMSNormalization` `GroupNormalization` `MeanVarianceNormalization` `Swish`
`ReduceLogSum` `SoftmaxCrossEntropyLoss` `NegativeLogLikelihoodLoss` `AffineGrid`
`CenterCropPad`, and opset 27's `LinearAttention` and `CausalConvWithState`. The body is
expanded before differentiation, at the model's opset and, where the spec generates it per
call, for the node's actual types; shape arithmetic is folded to constants and `If`s on a
folded condition are inlined, so the rules see static axes and shapes.

**By lowering** — the spec defines these in prose only, so onnx-ad writes the composition
out: `LRN` (a padded channel-window sum), `GridSample` (every mode, padding mode and
alignment, following the reference implementation step by step), `DeformConv` (bilinear
taps gathered into columns, then a grouped `MatMul`), `STFT` (gathered frames and a `DFT`)
and `TensorScatter` (a `ScatterElements` at computed positions). `RNN`, `GRU` and `LSTM`
become a `Scan` over their equations. A lowered operation is differentiable in every
floating-point operand; the derivative model computes its primal with the lowered form too.
`GridSample`'s cubic mode folds its taps back inside the image, not its coordinate, as
PyTorch and ONNX Runtime 1.19 and later do; ONNX Runtime 1.16 differs near the border.

**Not covered** — `MaxRoiPool`; `RoiAlign` in max mode or in its boxes (PyTorch gives the
boxes no gradient either, but silently); the `Sequence` and `Optional` types; and
training-mode `BatchNormalization` and `Dropout`. Each is refused with a reason when a
differentiated value reaches it. Operations whose results are integers, strings or
booleans — `QLinearConv`, `MatMulInteger`, the string and text operations — never carry a
derivative and need nothing.

### Control flow

`If`, `Scan` and `Loop` are differentiated *in place*: the primal node is replaced by one whose
subgraph also computes the derivative, rather than a derivative being built beside it.

* **`If`** — only the taken branch computes its derivative. An `If` on a constant condition is
  inlined before either pass runs.
* **`Scan`** — forward mode is one `Scan` carrying the tangent as extra state: two times the
  body, any number of seeds, nothing stored. Reverse mode tapes each iteration's input state
  and appends a second `Scan` that reads everything backwards, recomputes the body and runs
  its adjoint. A captured weight accumulates in that scan's state, so its adjoint costs
  `O(|w|)`, not `O(T·|w|)`.
* **`Loop`** — the trip count may depend on the data, which is only a problem until the primal
  has run: then it is the length of the tape. The reverse sweep is therefore `Scan`'s, reading
  the tape backwards, behind an `If` for the loop that ran zero times.

Nesting works in both directions, and so does composition: `forward(reverse(model))` through
any of the three. The design, and what ONNX Runtime was checked to allow, is in
[CONTROL-FLOW.md](CONTROL-FLOW.md); [the slides](docs/slides/slides.pdf) draw it.

`unroll(model)` flattens every `Scan` and `Loop` whose trip count is known before the model
runs. That is useful on its own for short, fixed-length recurrences — at the price of a graph
that grows with the trip count — and it is what the control-flow rules are tested against: the
Jacobian of the unrolled graph comes from entirely different code.

### Against PyTorch

Every one of these exports (`torch.onnx.export`, `dynamo=True`, opset 18) differentiates in
both modes, the resulting Jacobian matches `torch.autograd.functional.jacobian` to float32
precision, and `family` builds its second-order file:

| Model | Operations | forward | reverse |
| --- | --- | ---: | ---: |
| MLP | `Gemm` `Tanh` | 6e-8 | 6e-8 |
| GELU MLP | `Gemm` `Erf` `Mul` `Div` `Add` | 2e-7 | 1e-7 |
| SiLU MLP | `Gemm` `Sigmoid` `Mul` | 8e-8 | 8e-8 |
| Attention block | `Gemm` `MatMul` `Softmax` `LayerNormalization` `Reshape` `Transpose` | 2e-7 | 2e-7 |
| Convolutional net | `Conv` `Relu` `Gemm` `Reshape` | 1e-7 | 8e-8 |
| Indexing and reductions | `GatherND` `Slice` `Clip` `ReduceMax` `ReduceProd` `ReduceL2` `ReduceLogSumExp` | 1e-7 | 1e-7 |
| GRU cell | `Gemm` `Split` `Sigmoid` `Tanh` | 6e-8 | 6e-8 |

Exported whole, `nn.LSTM` and `nn.GRU` (bidirectional, two layers) agree with autograd to
about 1e-7, Hessians included. A spatial transformer — `affine_grid` then `grid_sample`,
exported at opset 20 as `AffineGrid` and `GridSample` — agrees in double precision to 1e-13,
in the image and in `theta`, for every mode, padding mode and alignment.

## Running the result

Reverse mode emits `Transpose` feeding `MatMul`, which ONNX Runtime's extended optimizer
fuses into `com.microsoft.FusedMatMul`. Through ONNX Runtime 1.19 that kernel exists for
`float` only (1.30 has a double one), so on a double-precision model with an older runtime
load with the fusions off:

```python
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
session = ort.InferenceSession("adj_f.onnx", options)
```

**ONNX Runtime before 1.19 miscompiles some derivative models.** Its `EliminateIdentity`
pass changes the adjoint of an expanded GRU or LSTM by about 0.1 — silently. The model
itself is correct: unoptimized, every runtime agrees with finite differences, and 1.19 and
later agree at every optimization level. On an older runtime, disable that one pass:

```python
session = ort.InferenceSession("adj_f.onnx", options, disabled_optimizers=["EliminateIdentity"])
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
