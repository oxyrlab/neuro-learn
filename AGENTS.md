# AGENTS.md

Entry point for any AI assistant working on `@oxyrlab/neuro-learn`.

## Read first

1. [docs/README.md](docs/README.md) — index and reading order.
2. [docs/STATUS.md](docs/STATUS.md) — current phase, next task, open issues.

## Non-negotiable rules

- English for all code, comments, identifiers and docs.
- One concept per file (< 300 lines); named exports only; TSDoc on every public export.
- Strict TypeScript, no `any`. Validate inputs and throw descriptive errors (`src/core/errors.ts`).
- No `Math.random`, `Date.now`, `performance.now` in `src/` (lint-enforced). Inject an `Rng`.
- State lives in fixed-size `Float32Array`s; no allocation after construction.
- No hard-coded numeric limits: expose them as constructor parameters with documented ranges.
- Every component ships with complete unit tests next to it (`x.ts` + `x.test.ts`).
- Docs are updated in the same change: `docs/STATUS.md`, `docs/CHANGELOG.md`, ADR if architecture changes.
- Commit each file separately with a Conventional Commit message. Never bundle many files in one commit.

## Definition of done

`npm run typecheck`, `npm run lint`, `npm run test:coverage`, `npm run build`, `npm run docs:check` all green.

## Phases

0 scaffolding · 1 core + linear learners · 2 networks · 3 realistic tests, debugging, release.
