# Status

- **Current phase:** 1 — core and linear learners (complete, awaiting review).
- **Next task:** Phase 2 — `mlp-sgd` (with numerical gradient check), then feedforward and recurrent networks on `Float32Array`, serialization and determinism tests.

## Done

- Phase 0: tooling, `src/core` (Rng, errors), docs skeleton, ADRs 0001–0004.
- Phase 1: `core/validation`, `Learner` interface, `linear-math`, `runLearnerContract` (`/testing` entry), `rescorla-wagner`, `eligibility-trace`, `hebbian`; concept docs; ADR 0005.

## Open issues / debt

- "No allocation after construction" is only checked structurally; a measured benchmark is planned for phase 3.

## Open design questions

- None for this package. (Neuromodulator semantics are an EcoSim decision.)
