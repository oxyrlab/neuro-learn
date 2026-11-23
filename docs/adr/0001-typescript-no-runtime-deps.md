# ADR 0001: Strict TypeScript, zero runtime dependencies

- **Status:** accepted
- **Date:** 2026-11-23

## Context

The library is a numerical core that must be easy to audit, embed in a Web Worker or Node, and reuse.

## Decision

Write in strict TypeScript (`noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`). Ship with no runtime dependencies. Dev tooling only: Vitest, fast-check, ESLint, Prettier, tsup.

## Rejected options

- ML frameworks (TensorFlow.js etc.): too heavy, non-deterministic, unnecessary for small online learners.

## Consequences

All algorithms are implemented in-house and fully tested; bundle stays tiny.
