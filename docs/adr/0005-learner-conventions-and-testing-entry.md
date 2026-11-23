# ADR 0005: Learner conventions and the `/testing` entry point

- **Status:** accepted
- **Date:** 2026-11-23

## Context

EcoSim needs to plug its own learners in and verify them against the same rules. Delayed credit assignment needs a "recent cues" memory that the generic `Learner` interface (EcoSim spec 5.3.1) does not express.

## Decision

1. `Learner` adds `inputSize` and `outputSize` to the spec interface; `rate` is the effective per-call rate in [0, 1] (the host applies any modulator scaling).
2. `eligibility-trace` exposes an extra `observe(features)` method (not part of `Learner`). Time passes only through `decay(dt)`, which also decays the trace; `predict` never changes state. Typical tick: `decay`, `observe`, `predict`.
3. The shared test suite `runLearnerContract` is published as `@oxyrlab/neuro-learn/testing`. `vitest` is an optional peer dependency, so the main entry never imports it.
4. Learned state is never inherited: a new instance always starts with empty memory.

## Rejected options

- `predict` updating the trace: surprising side effect.
- `observe(features, dt)`: risks double-counting time when the host also calls `decay(dt)`.
- Putting the contract inside the main entry: would force a test framework on every consumer.

## Consequences

Hosts can verify custom learners with one call. Absence of allocation is only structurally verified until phase 3 benchmarks.
