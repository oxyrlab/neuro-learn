import { InvalidParameterError } from './errors';

/**
 * Source of randomness injected into every component that needs it.
 * Components never call `Math.random()`; this keeps results reproducible (docs/adr/0003).
 */
export interface Rng {
  /** Returns a uniformly distributed float in the half-open interval [0, 1). */
  next(): number;
}

/**
 * Creates a small, fast, seeded pseudo-random generator (mulberry32).
 * The same seed always yields the same sequence.
 *
 * @param seed - Any integer. Only its low 32 bits matter: seeds that differ by a multiple of
 *   2^32 give the same stream.
 * @returns A deterministic {@link Rng}.
 * @throws InvalidParameterError if `seed` is not an integer (NaN, Infinity and fractions are
 *   rejected instead of silently collapsing to a fixed stream).
 *
 * @example
 * const rng = createSeededRng(42);
 * const a = rng.next(); // always the same value for seed 42
 */
export function createSeededRng(seed: number): Rng {
  if (!Number.isInteger(seed)) throw new InvalidParameterError('seed', seed, 'an integer');
  let state = seed >>> 0;
  return {
    next(): number {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    },
  };
}
