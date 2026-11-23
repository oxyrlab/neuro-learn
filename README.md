# @oxyrlab/neuro-learn

Small, deterministic, allocation-free learning algorithms and neural networks on `Float32Array`.
Built as the learning core for [EcoSim](https://github.com/oxyrlab), but usable on its own.

> **Status:** `0.1.0`: all planned components are implemented and tested (see [docs/STATUS.md](docs/STATUS.md)).

## Contents

| Area     | Items                                                                                |
| -------- | ------------------------------------------------------------------------------------ |
| Learners | `rescorla-wagner`, `eligibility-trace`, `hebbian`, `mlp-sgd`                         |
| Networks | `FeedforwardNetwork`, `RecurrentNetwork` (Elman), `Backpropagator` on `Float32Array` |
| Core     | `Rng` interface, seeded RNG, descriptive errors                                      |

## Install

Not published to npm yet. Install a tagged release straight from GitHub (it builds on install):

```bash
npm install github:oxyrlab/neuro-learn#v0.1.0
```

## Quick start

```ts
import { createSeededRng, EligibilityTraceLearner, MlpSgdLearner } from '@oxyrlab/neuro-learn';

// Fear conditioning with a delayed outcome (cue strengths must be >= 0).
const fear = new EligibilityTraceLearner({
  inputSize: 4,
  outputSize: 1,
  traceTimeConstantSeconds: 2,
});
const seen = Float32Array.of(0.9, 0.1, 0.8, 0.1);
fear.decay(0.1); // time passes
fear.observe(seen); // the creature sees something
fear.learn(new Float32Array(4), Float32Array.of(1), 0.4); // pain arrives: credit flows to recent cues
const danger = new Float32Array(1);
fear.predict(seen, danger); // ≈ 0.4

// A small network trained online.
const mlp = new MlpSgdLearner({ layerSizes: [2, 6, 1], rng: createSeededRng(1), maxWeight: 10 });
mlp.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.1);
```

Every learner implements `Learner` (`predict`, `learn`, `decay`, `serialize`, `restore`); see [docs/concepts/learner.md](docs/concepts/learner.md). Test your own learner with the shared contract:

```ts
import { runLearnerContract } from '@oxyrlab/neuro-learn/testing'; // needs vitest
runLearnerContract('my-learner', () => new MyLearner({/* non-zero forgetting */}));
```

## Design rules

- Zero runtime dependencies; strict TypeScript; named exports only.
- Deterministic: randomness only through an injected `Rng`.
- Fixed state size and no allocation after construction.
- Every component is unit-tested; coverage thresholds are enforced in CI.
- Hot paths allocate nothing (`npm run check:allocation`); see [docs/performance.md](docs/performance.md).

## Development

```bash
npm install
npm run typecheck && npm run lint && npm run test:coverage && npm run build && npm run docs:check
```

Contributors and AI assistants: start with [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE)
