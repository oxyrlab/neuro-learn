# ADR 0003: Injected Rng for determinism

- **Status:** accepted
- **Date:** 2026-11-23

## Context

Simulations must be exactly reproducible from a seed.

## Decision

Components needing randomness (e.g. weight initialization) receive an `Rng` (`next()` in [0, 1)). `Math.random`, `Date.now` and `performance.now` are banned in `src/` by ESLint. A small seeded generator (`createSeededRng`, mulberry32) is provided for tests and standalone use.

## Rejected options

- Global seeded RNG: hidden state breaks isolation.

## Consequences

The host (EcoSim) supplies its own per-entity `Rng`; identical seeds give identical results.
