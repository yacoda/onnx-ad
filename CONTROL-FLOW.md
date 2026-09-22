# Differentiating control flow — design brief

*Status: design only. `If`, `Scan` and `Loop` have no rules yet.*

The three operations are worth taking in the order `If`, `Scan`, `Loop`, because each adds
exactly one capability the previous one did not need, and the third gets most of its answer
from the second.

| | new capability needed | reverse mode needs |
| --- | --- | --- |
| `If` | recursion into subgraphs, outer-scope capture | nothing iterative |
| `Scan` | a tape, and a sweep in the opposite direction | trip count known **before** the primal runs |
| `Loop` | a trip count known only **after** the primal runs | the `Scan` machinery, fed by that trip count |

The claim that makes `Loop` cheap: **the adjoint of a `Loop` is a `Scan`.** Once the primal
loop has run, its trip count is data, so the reverse sweep no longer needs data-dependent
termination — and `Scan` can read its inputs backwards, which removes all index arithmetic.

## What ONNX Runtime actually allows

Checked against ONNX Runtime 1.30 before committing to any of this:

| Assumption | Result |
| --- | --- |
| a subgraph reads an outer-scope tensor by name | works, and through two nesting levels |
| `Scan` with `scan_input_directions=[1]`, `scan_output_directions=[1]` | works — reversed cumulative sum came back exactly |
| a `Scan` state carrying a trailing seed axis | works |
| `Loop` with a data-dependent trip count, taped via a scan output, count recovered as `Shape(tape)[0]` | works |
| **`Scan` over a zero-length axis** | **rejected**: *"Invalid dim0_offset of 0. Dimension 0 is 0"* |
| `Loop` with trip count 0 | works, returns the initial state |
| an `If` guarding a possibly-empty reverse `Scan` | works |

Only the zero-trip-count case needs care, and the `If` guard is the fix.

## Shared infrastructure

Most of the work is here, and all three operations spend it.

**A scope chain.** `forward` and `reverse` become functions over a `GraphProto` plus a parent
`Context`, with the model-level entry points a thin wrapper. Derivative-map and shape lookups
walk up the chain; nodes are emitted into the current graph's own list. Because a subgraph
reads outer names directly, *the tangent or adjoint of an outer value needs no plumbing at
all* — it is simply in scope. That is the single biggest reason this stays small.

**One name allocator for the whole model.** A subgraph tensor whose name collides with an
outer one shadows it, silently. The allocator has to be shared down the chain even though the
node lists are not. This splits today's `Builder` into a shared namer and a per-scope node
list.

**Constants stay in the outermost graph.** Subgraphs capture them, so `Builder.constant`
keeps deduplicating globally and no constant is rebuilt per iteration.

**Capture analysis.** `captures(graph)` — names read by the subgraph but produced neither
inside it nor by its own inputs or initializers, computed recursively. Reverse mode needs it
to know which outer values a branch or body must route adjoint contributions out to; it is
intersected with `ctx.asked_for` so a constant weight costs nothing.

## If

Branches have no inputs and the same output types, which makes this the gentle case.

**Forward.** Differentiate each branch *in place*: run the forward pass on the branch
subgraph with a child context, and append the resulting tangents to that branch's outputs.
The `If` node then carries `k` primal outputs followed by the tangents. Only the taken branch
computes its derivative — which is the whole point of `If`, and the reason not to compute both
branches and select with `Where`. Take the union of outputs that got a tangent in *either*
branch, and materialize zeros inside the branch that did not, shaped from that branch's own
output tensor.

**Reverse.** Seed each branch with the adjoints of the `If` outputs — outer tensors, visible
inside — run the reverse pass, and append one output per captured value in the union across
branches. The outer pass accumulates those onto the captured values' adjoints. A branch that
does not touch a captured value emits zeros shaped from the captured tensor itself.

**Shortcut worth having.** When `cond` is a constant, inline the taken branch into the parent
and drop the `If` entirely. Traced Python `if`s on static flags export to exactly this, and
inlining leaves a flat graph that the existing rules handle with no recursion at all.

## Scan

Fixed trip count, taken from the scan axis. `body: (state..., slice...) -> (state..., out...)`.

**Forward is one `Scan` and no tape.** The tangent of a scan is a scan of the same length
whose state is the pair `(state, tangent)` and whose scan inputs are the pair
`(slice, tangent)`:

```
Scan(state…, t_state…, scan…, t_scan…) -> state_f, t_state_f, out…, t_out…
body: (s, t_s, x, t_x) -> (s', t_s', y, t_y)
```

`num_scan_inputs` doubles and each axis and direction attribute is duplicated — with the axes
**normalized non-negative first**, since a tangent carries a trailing seed axis and a negative
axis would land on it. Cost is about twice the primal body, for any number of seeds. Nothing
is stored per iteration.

**Reverse is two nodes.**

1. *The primal scan, taping its state.* Add one scan output per state carrying that
   iteration's **input** state. That is the minimal tape: the body's inputs are the state and
   the slice, the slices are already in the outer graph, so the input state is exactly the
   root from which the whole body can be recomputed. Nothing else needs storing — and because
   the reverse pass keeps the primal subgraph anyway, that recomputation is not extra work to
   implement, it is what the pass already emits.
2. *A reversed scan.* Its scan inputs — the tape, the original scan inputs, and the adjoints
   of the scan outputs — all read with `direction=1`. Its states are the adjoints of the
   loop-carried state, plus one accumulator per captured value. Its body recomputes the primal
   from the taped state and the slice, then runs the adjoint: the adjoint of the input state
   becomes the next state, the adjoint of each slice becomes a scan output written with
   `direction=1` so it lands in the original order, and each captured value's contribution is
   added to its accumulator.

Accumulating captured values in the **state** rather than as scan outputs is what keeps a
weight's adjoint at `O(|w|)` instead of `O(T·|w|)`.

Memory is `O(T · Σ|state|)`; time is one extra body evaluation per iteration. That is the
ordinary trade, and checkpointing — taping every `k`-th state and recomputing between — is the
refinement to add later, not now.

## Loop

`body: (iter, cond_in, v…) -> (cond_out, v…, out…)`, with `cond_out` able to stop early.

**Forward** is `Scan`'s, with the condition left alone: termination depends only on primal
values. Where the trip count changes with the input the derivative is one-sided, the same
convention already taken at `Relu(0)`.

**Reverse** is where the ordering pays off:

1. Run the primal `Loop`, taping the per-iteration input state as a scan output.
2. Recover the trip count as `Shape(tape)[0]` — it is simply how many rows the tape has.
3. Run **the `Scan` reverse sweep from the previous section**, unchanged. The trip count is
   now the tape's length, and `direction=1` gives the reverse order for free.
4. Guard the sweep with `If(T > 0)`, whose else branch returns the zero-initialized
   accumulators, because ONNX Runtime rejects a zero-length `Scan`. Skip the guard when the
   loop provably runs at least once.

Using a reverse `Loop` instead would tolerate `T = 0` natively but would need
`Gather(tape, T-1-i)` index arithmetic *and* a final flip of the scan outputs, which come out
in iteration order. The guarded `Scan` is the better shape.

## Composition survives

The property that a derivative model is itself differentiable has to hold through all of this,
or second derivatives stop at the first loop:

- forward of `If` is an `If`; reverse of `If` is an `If`.
- forward of `Scan` is a `Scan`; reverse of `Scan` is two `Scan`s.
- reverse of `Loop` is a `Loop` plus an `If` around a `Scan` — every one of which has rules by
  the time `Loop` is reached.

So `forward(reverse(model))` still gives forward-over-adjoint, and the implementation order is
also the order in which the set of rules closes under its own output.

## Testing: unrolling is the oracle

For a static trip count, a `Scan` or `Loop` can be unrolled into a flat graph that the
existing, heavily tested rules already differentiate. Differentiating the unrolled graph and
the control-flow graph then gives two Jacobians from two independent implementations, on top
of finite differences of the primal and forward against reverse.

That makes `unroll(model)` worth shipping **first**, as a small, self-contained pass: it is
the test oracle, and on its own it already unblocks fixed-length recurrences — a four-stage
integrator, a short RNN — at the cost of a graph that grows with the trip count. It is a
stopgap with a size guard, never the answer for a dynamic trip count.

## Staging

0. `unroll` for static trip counts — small, useful on its own, and the oracle for everything after.
1. Scope chain, shared name allocator, capture analysis; then `If`, with constant-condition inlining.
2. `Scan` forward, then `Scan` reverse.
3. `Loop` forward, then `Loop` reverse on top of `Scan`'s.

## Not in scope

Checkpointing; detecting a linear body and skipping its tape; sequence and optional types;
`SequenceAt`/`SequenceInsert`; a body that carries state other than through its declared
loop-carried values.
