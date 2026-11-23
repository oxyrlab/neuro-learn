# neuro-learn docs

Small learning library (Learners and neural networks) on `Float32Array`. Deterministic, allocation-free,
no runtime dependencies. Consumed by EcoSim through the `Learner` interface.

## Reading order by task

| Task                 | Read                                                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Understand terms     | [glossary](glossary.md)                                                                                                                                 |
| Add a learner        | [glossary](glossary.md), [ADR 0002](adr/0002-float32array-state.md), [ADR 0003](adr/0003-injected-rng.md), [component template](templates/component.md) |
| Touch randomness     | [ADR 0003](adr/0003-injected-rng.md)                                                                                                                    |
| Change architecture  | [ADR 0001](adr/0001-typescript-no-runtime-deps.md), [ADR 0004](adr/0004-standalone-package.md), [ADR template](templates/adr.md)                        |
| Know what to do next | [STATUS](STATUS.md)                                                                                                                                     |

## Files

- [STATUS.md](STATUS.md) — current state
- [CHANGELOG.md](CHANGELOG.md) — notable changes
- [glossary.md](glossary.md) — definitions
- `adr/` — architecture decisions; `templates/` — doc templates
