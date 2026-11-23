import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { createSeededRng } from './rng';

describe('createSeededRng', () => {
  it('produces the same sequence for the same seed', () => {
    const a = createSeededRng(42);
    const b = createSeededRng(42);
    for (let i = 0; i < 100; i += 1) {
      expect(a.next()).toBe(b.next());
    }
  });

  it('produces different sequences for different seeds', () => {
    expect(createSeededRng(1).next()).not.toBe(createSeededRng(2).next());
  });

  it('always returns values in [0, 1) (property)', () => {
    fc.assert(
      fc.property(fc.integer(), (seed) => {
        const rng = createSeededRng(seed);
        for (let i = 0; i < 50; i += 1) {
          const value = rng.next();
          if (!(value >= 0 && value < 1)) return false;
        }
        return true;
      }),
    );
  });

  it('is roughly uniform', () => {
    const rng = createSeededRng(7);
    let sum = 0;
    const count = 10000;
    for (let i = 0; i < count; i += 1) sum += rng.next();
    expect(sum / count).toBeGreaterThan(0.48);
    expect(sum / count).toBeLessThan(0.52);
  });
});
