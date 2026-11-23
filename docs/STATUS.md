# Status

- **Current phase:** 2 — networks and `mlp-sgd` (complete, awaiting review).
- **Next task:** Phase 3 — realistic scenario tests (conditioning, extinction, generalization, XOR/regression), property tests, allocation benchmark, debugging, release `0.1.0`.

## Done

- Phase 0: tooling, `src/core` (Rng, errors), docs skeleton, ADRs 0001–0004.
- Phase 1: `core/validation`, `Learner` interface, `linear-math`, `runLearnerContract` (`/testing` entry), `rescorla-wagner`, `eligibility-trace`, `hebbian`; concept docs; ADR 0005.
- Phase 2: `nn/activations`, `Network`, `FeedforwardNetwork`, `Backpropagator` (numerical gradient check), `MlpSgdLearner`, `RecurrentNetwork`; contract option `initialState`; ADR 0006.

## Open issues / debt

- "No allocation after construction" is only checked structurally; a measured benchmark is planned for phase 3.

## Open design questions

- None for this package. (Neuromodulator semantics are an EcoSim decision.)
