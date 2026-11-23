---
id: core/learner
kind: core
status: experimental
files:
  [
    src/learning/learner.ts,
    src/learning/linear-math.ts,
    src/core/validation.ts,
    src/testing/learner-contract.ts,
  ]
tests:
  [
    src/learning/linear-math.test.ts,
    src/core/validation.test.ts,
    src/testing/learner-contract.test.ts,
  ]
---

# Learner interface and contract

**Purpose:** common shape of every learner so hosts (EcoSim) can swap them and reuse one test suite.

## Interface

`inputSize`, `outputSize`, `stateSize`, `predict(features, out)`, `learn(features, target, rate)`, `decay(dt)`, `serialize()`, `restore(state)`. Full TSDoc in `src/learning/learner.ts`.

## Conventions

| Topic       | Rule                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------ |
| Features    | Finite numbers, normalized by the caller (typically [0, 1])                                                  |
| `rate`      | Effective rate for this call, in [0, 1]; callers scale it by a modulator                                     |
| `dt`        | Seconds, finite, >= 0                                                                                        |
| Allocation  | None in `predict`/`learn`/`decay`; `serialize` returns a copy                                                |
| Errors      | Wrong buffer length → `SizeMismatchError`; non-finite value or out-of-range scalar → `InvalidParameterError` |
| Inheritance | New instance = empty memory; never copy learned state to offspring                                           |

## Contract suite

`runLearnerContract(name, factory, options?)` from `@oxyrlab/neuro-learn/testing` (needs `vitest`). The factory must enable a non-zero forgetting rate. Checks: constant state size, empty start, finite and bounded state, determinism, serialize/restore round-trip, independent serialized copy, input immutability, forgetting, validation errors.

## Gotchas

- Absence of allocation is checked structurally only; measured benchmarks come in phase 3.
- Delta-rule stability needs `rate · Σ cue² < 2`; weights are clamped regardless, so state never diverges.
