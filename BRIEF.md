# onnx-ad — brief

*Status: implemented. This is the original design brief, kept for the reasoning
behind the choices; see README.md for what the package actually does.*

Source-code-transforming automatic differentiation **on ONNX graphs**, in pure Python over
the `onnx` protobuf. Given a model, produce new ONNX models that compute its forward and
reverse derivatives — what torch2casadi does today inside PyTorch, done from first
principles one level down, where the graph is already flat, typed and framework-neutral.

Planned repo: `yacoda/onnx-ad`. Pure `onnx` + `numpy`, no runtime dependency, MIT, per-op
rule table, CLI.

## Why this, and why not the existing routes

| Route | What it costs |
| --- | --- |
| PyTorch AD, then export (torch2casadi today) | derivative graphs only exist for models that came from PyTorch; forward mode goes through `jvp`/`vmap`, which is where the export breaks (seed-count specialisation in `slice_backward`, `gelu_backward`, `native_layer_norm_backward`); every derivative order needs another trace |
| Complex step | forward mode only, one graph evaluation per direction, and a *convention* rather than an identity at piecewise operations |
| **Differentiate the ONNX graph itself** | needs a rule per ONNX operation — but then any ONNX model has derivatives, from any producer, at any order, with no framework in the loop |

The third route is the one with no ceiling: a Jacobian-vector product graph is itself an
ONNX model, so it can be differentiated again for Hessians, and consumers (CasADi's ONNX
backend, ONNX Runtime, anything) need no new capability.

## Scope

- `forward(model, inputs=None, outputs=None)` → a model computing `fwd_y = J·fwd_x`, with
  seed inputs `fwd_<x>` and outputs `fwd_<y>`, seed count a symbolic dimension (`nfwd`).
- `reverse(model, ...)` → `adj_x = Jᵀ·adj_y`, inputs `adj_<y>`, outputs `adj_<x>`, seed
  count `nadj`.
- Composition: `forward(reverse(model))` gives forward-over-adjoint (`fwd_adj_*`), the
  exact-Hessian building block, with no special casing.
- `jacobian(model)` as a convenience over repeated seeds where a dense Jacobian is wanted.
- Naming and seed-dimension conventions to match what CasADi's ONNX backend already
  discovers (`fwd_`/`adj_`/`jac_` siblings, `nfwd`/`nadj` dimensions), so an exported family
  drops straight into the existing consumer.

Non-goals for a first release: control flow (`Loop`, `Scan`, `If`), sparsity exploitation,
graph optimisation beyond constant folding of the obvious (an optimiser pass belongs
downstream), training-mode operations (`Dropout`, batch-norm statistics).

## Design sketch

Two passes over the same rule table:

- A **tape**: walk `graph.node` in topological order, keeping a map from value name to the
  tangent (forward) or adjoint (reverse) tensor name, absent when the value is constant with
  respect to the differentiated inputs. Absent tangents are what keeps the emitted graph
  small — the same trick as the "no imaginary part" case in the complex pass, and the reason
  a constant weight costs nothing.
- **Forward** emits in graph order; **reverse** emits a second walk in reverse order,
  accumulating contributions per value (`Add` when a value has several consumers), and needs
  the primal values of any node whose rule is nonlinear — so the reverse model keeps the
  primal subgraph it needs rather than taking it as an input (`uses_output` in CasADi's
  convention is then unnecessary, and the model stays a plain function of `x`, `adj_y`).
- Rules per operation, both modes. Start with what a traced network contains: `Add`, `Sub`,
  `Mul`, `Div`, `MatMul`, `Gemm`, `Einsum`, `Conv`, `Pow`, `Reciprocal`, `Sqrt`, `Exp`,
  `Log`, `Tanh`, `Sigmoid`, `Relu`, `LeakyRelu`, `Elu`, `Softplus`, `Erf`, `Gelu`, `Softmax`,
  `LogSoftmax`, `ReduceSum`, `ReduceMean`, `Reshape`, `Transpose`, `Concat`, `Split`,
  `Slice`, `Gather`, `Squeeze`, `Unsqueeze`, `Expand`, `Pad`, `Tile`, `Where`, `Identity`.
  An operation with no rule is an error only when a differentiated value reaches it.
- Broadcasting is the sharp edge of reverse mode: a contribution must be reduced back to the
  operand's shape (`ReduceSum` over broadcast axes, then `Reshape`). Do it from the declared
  shapes where they exist and from `Shape`/`ReduceSum` nodes where they do not, so dynamic
  dimensions survive.
- Seeds: one symbolic dimension, appended as the last axis of the seed tensors, so several
  directions ride in one evaluation — unlike the complex step, which needs one pass per
  direction.

## Testing

Every rule executed through ONNX Runtime and compared against an independent reference, per
operation and on assembled networks. References come from finite differences for the shape
of the answer and from an analytic Jacobian for machine-precision agreement, so nothing
depends on PyTorch to be believed. PyTorch comparisons belong in torch2casadi's integration
tests, not here.

## Relationship to the other repos
- `torch2casadi` could, once this exists, export the primal only and generate the whole
  derivative family from it — no PyTorch AD in the pipeline at all. That is a later decision,
  not a promise; PyTorch's reverse AD is well tested and the ONNX-level rules must earn that
  trust first.
- CasADi needs nothing new: the emitted families use the sibling conventions its ONNX
  backend already discovers.
