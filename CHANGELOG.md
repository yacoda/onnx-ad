# Changelog

## Unreleased

Control flow, and second-order families that 0.1.0 could not build.

- `If`, `Scan` and `Loop` differentiate in both modes, nested in either direction and through
  `forward(reverse(model))`. Forward mode extends the node in place; reverse mode tapes a
  loop's per-iteration state and sweeps it backwards with a `Scan` — for `Loop`, behind an
  `If` for the zero-iteration case, which ONNX Runtime's `Scan` cannot express.
- An `If` on a constant condition is inlined before differentiation.
- `unroll(model)` flattens every `Scan` and `Loop` whose trip count is known ahead of time.
- **Fixed:** `family()` could not build `fwd_adj_*` for any model that convolves or gathers,
  because the reverse of `Conv` emits `ConvTranspose` and the reverse of `Gather`/`GatherND`
  emits `ScatterND`, and neither had a rule. Both now do, in both modes.
- **Fixed:** `Gather` with a scalar index — what `x[2]` exports to — failed in reverse mode.
- Outputs of control-flow nodes that ONNX shape inference leaves unshaped are filled in, so
  values downstream of a `Loop` keep their rank.

## 0.1.0 — 2026-09-22

First release: both passes, CasADi's conventions, and 109 operation rules.

- `forward(model)` emits `fwd_y = J . fwd_x`; `reverse(model)` emits `adj_x = J^T . adj_y`.
  Several seed directions ride in one evaluation.
- The primal graph is kept, so a derivative model is a plain function of its original inputs
  and the seeds, and is itself differentiable: `forward(reverse(model))` is
  forward-over-adjoint, the exact-Hessian building block.
- CasADi's conventions throughout, by default: derivative prefixes and seed dimensions from
  its `diff_prefix` rule (`fwd_`/`nfwd`, then `fwd2_`/`nfwd2`), the seed layout its 2-D
  reading of an ONNX tensor expects, and `<kind>_<filename>` sibling files. `family` writes
  the whole set; `layout="onnx"` opts out of the packing.
- Rules for arithmetic, the elementwise family, `MatMul`, `Gemm`, `Conv`, the shape and
  indexing operations, the reductions, `Where`/`Clip`/`Min`/`Max`, `Softmax`,
  `LogSoftmax`, `LayerNormalization`, inference-mode `BatchNormalization` and `Dropout`, and
  the zero-derivative operations a differentiated value may pass through.
- `onnx-ad forward|reverse|family` on the command line.
