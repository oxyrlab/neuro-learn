# ADR 0002: Fixed-size Float32Array state, no allocation in hot paths

- **Status:** accepted
- **Date:** 2026-11-23

## Context

Learners run for hundreds of creatures every tick; GC pressure and variable-size state hurt speed and serialization.

## Decision

Every component stores its state in `Float32Array`s sized at construction. `predict`, `learn` and `decay` write into caller-provided buffers and allocate nothing. Weights are clamped to stay bounded.

## Rejected options

- Plain `number[]` or objects: slower, GC-heavy, harder to snapshot.

## Consequences

Cheap `serialize`/`restore`; contract tests can assert fixed state size and no allocation. Float32 precision is accepted.
