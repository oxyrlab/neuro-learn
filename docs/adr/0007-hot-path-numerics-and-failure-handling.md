# ADR 0007: Hot-path numerics and failure handling

- **Status:** accepted
- **Date:** 2026-11-23

## Context

Phase 3 measured allocation per call (`npm run check:allocation`) and fuzzed every learner. Findings:

- V8 boxed a number per element whenever a small helper returned a fractional value (`clampMagnitude`, per-element `activation.apply`), and per weight when a clamp merged an integer-valued bound and a computed value into one variable. Result: 256–800 bytes of garbage per `learn`/`forward` call.
- A huge finite input could overflow `Float32` in an unbounded (`relu`/`linear`) network; `Infinity * 0` then turned every weight into `NaN`.
- `restore` accepted states a learner could never have produced.
- Recurrent networks could keep a non-finite hidden state.

## Decision

1. Activations expose buffer-level methods (`applyInPlace`, `scaleByDerivative`); networks and backprop use them. The scalar `apply`/`derivativeFromOutput` remain for tests and callers.
2. Clamps store the bound or value inside each branch (`if (x > limit) w[i] = limit; else if …`) and `clampInPlace` replaces `clampMagnitude`.
3. `MlpSgdLearner.learn` skips an update whose loss or gradient is non-finite (documented), instead of throwing, because a host simulation should not crash on one extreme input.
4. `RecurrentNetwork.forward` throws the new `NumericInstabilityError` and leaves its state unchanged when the hidden state or output would become non-finite.
5. `restore` validates against the learner's own invariants (weights within `maxStrength`/`maxWeight`, non-negative trace). Bounds must be <= the Float32 maximum.
6. Sizes that cannot be allocated raise `InvalidParameterError` (`createFloat32Array`), not a raw `RangeError`. `createSeededRng` requires an integer seed.
7. `eligibility-trace` rejects negative cue strengths in `observe` and `learn`; a replacing trace cannot represent them and silently ignoring them was a bug.

## Rejected options

- Throwing from `MlpSgdLearner.learn` on overflow: crashes a host over one extreme input.
- Clamping features at the input: hides caller errors and changes semantics.
- A flush-to-zero option for denormals: measured no slowdown (values are widened to float64 on read), so it would be dead complexity.

## Consequences

Hot paths measure ~0 bytes per call. New hot-path code must keep that property; `check:allocation` covers integer and fractional bounds because the two take different engine paths.
