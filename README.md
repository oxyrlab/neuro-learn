# @oxyrlab/neuro-learn

Small, deterministic, allocation-free learning algorithms and neural networks on `Float32Array`.
Built as the learning core for [EcoSim](https://github.com/oxyrlab), but usable on its own.

> **Status:** under development (phases 0–2 complete: scaffolding, core, learners, networks). See [docs/STATUS.md](docs/STATUS.md).

## Planned contents

| Area     | Items                                                                                |
| -------- | ------------------------------------------------------------------------------------ |
| Learners | `rescorla-wagner`, `eligibility-trace`, `hebbian`, `mlp-sgd`                         |
| Networks | `FeedforwardNetwork`, `RecurrentNetwork` (Elman), `Backpropagator` on `Float32Array` |
| Core     | `Rng` interface, seeded RNG, descriptive errors                                      |

## Design rules

- Zero runtime dependencies; strict TypeScript; named exports only.
- Deterministic: randomness only through an injected `Rng`.
- Fixed state size and no allocation after construction.
- Every component is unit-tested; coverage thresholds are enforced in CI.

## Development

```bash
npm install
npm run typecheck && npm run lint && npm run test:coverage && npm run build && npm run docs:check
```

Contributors and AI assistants: start with [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE)
