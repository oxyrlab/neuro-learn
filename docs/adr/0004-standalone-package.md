# ADR 0004: neuro-learn is a standalone package

- **Status:** accepted
- **Date:** 2026-11-23

## Context

EcoSim spec (ADR 0008 there) places learners in `src/brain/learning` and allows an external library only through the `Learner` interface. The learners depend only on `Float32Array`, not on ECS, bus or physics.

## Decision

Develop the learners and the small neural networks here as `@oxyrlab/neuro-learn` (MIT, public). EcoSim's `src/brain/learning` becomes a thin layer over this package. Learning _modules_ (e.g. `threat-memory`) that need the signal bus stay in EcoSim.

## Rejected options

- Keeping everything inside EcoSim: couples reusable math to the simulator and slows isolated testing.
- A multi-backend adapter: out of scope (EcoSim ADR 0008).

## Consequences

EcoSim ADR 0008 needs an amendment when EcoSim starts. The `Learner` interface is this package's public contract; changes follow semver.
