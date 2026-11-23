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
 * @param seed - Any integer; it is coerced to an unsigned 32-bit value.
 * @returns A deterministic {@link Rng}.
 *
 * @example
 * const rng = createSeededRng(42);
 * const a = rng.next(); // always the same value for seed 42
 */
export function createSeededRng(seed: number): Rng {
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
