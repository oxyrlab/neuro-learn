# Status

- **Current phase:** 3 — realistic tests, debugging and release preparation (complete, awaiting review and tag approval).
- **Maturity:** initial implementation (`0.1.0`). Everything planned is implemented and tested; improvements and extensions are expected in later sessions.
- **Version:** `0.1.0` prepared; tag `v0.1.0` is pushed.
- **Next task:** integrate into EcoSim (phase 0 of EcoSim): thin `src/brain/learning` layer over this package; amend EcoSim ADR 0008.

## Done

- Phase 0: tooling, `src/core` (Rng, errors), docs skeleton, ADRs 0001–0004.
- Phase 1: `Learner`, validation, `linear-math`, `runLearnerContract` (`/testing`), `rescorla-wagner`, `eligibility-trace`, `hebbian`.
- Phase 2: activations, `Network`, `FeedforwardNetwork`, `Backpropagator`, `MlpSgdLearner`, `RecurrentNetwork`.
- Phase 3: scenario tests (fear conditioning, extinction, generalization, reward/pain, long runs, function fitting, recurrent evolution), model-based fuzzing, allocation check, benchmarks, code-review fixes, ADR 0007.

## Open issues / debt

- `check:allocation` runs in CI on Node 24 only; thresholds were verified on Node 24 and may need tuning on other versions.
- Features of `predict` are not range-checked: extreme finite inputs (near 3e38) can produce infinite outputs (state stays safe).

## Open design questions

- None for this package. (Neuromodulator semantics are an EcoSim decision.)
