# Changelog

## Unreleased

- Benchmark against TensorFlow.js in `benchmarks/tfjs-comparison` (speed, resources, accuracy; separate package, not published); results summarized in `docs/performance.md`.

## 0.1.0

- Learners: `RescorlaWagnerLearner`, `EligibilityTraceLearner`, `HebbianLearner`, `MlpSgdLearner`; shared `Learner` interface and `runLearnerContract` (`@oxyrlab/neuro-learn/testing`).
- Networks: `FeedforwardNetwork`, `RecurrentNetwork` (Elman), `Backpropagator`, activations.
- Core: `Rng` + `createSeededRng`, `InvalidParameterError`, `SizeMismatchError`, `NumericInstabilityError`.
- Phase 3 fixes: `eligibility-trace` rejects negative cues; MLP skips non-finite updates; `restore` validates bounds; `RecurrentNetwork` raises `NumericInstabilityError`; bounds limited to Float32 range; impossible sizes raise `InvalidParameterError`; `createSeededRng` validates the seed.
- Performance: hot paths allocate ~0 bytes per call (was 256–800); `npm run check:allocation`, `npm run bench`; see `docs/performance.md`; ADR 0007.
- Scenario and fuzz tests; EcoSim `threat-memory` criteria reproduced (see `tests/scenarios/threat-memory.test.ts`).

## Earlier (pre-release)

- Phase 2: activations, `FeedforwardNetwork`, `RecurrentNetwork`, `Backpropagator`, `MlpSgdLearner`; contract option `initialState`; ADR 0006.

- Phase 1: `Learner` interface, `RescorlaWagnerLearner`, `EligibilityTraceLearner`, `HebbianLearner`, `runLearnerContract` (`@oxyrlab/neuro-learn/testing`), validation helpers, ADR 0005.
- Phase 0: project scaffolding, `src/core` (Rng, errors), docs skeleton, ADRs 0001–0004.
