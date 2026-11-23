import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { applyDeltaRule, clampInPlace, multiplyMatrixVector, scaleInPlace } from './linear-math';

describe('multiplyMatrixVector', () => {
  it('multiplies a 2x3 matrix by a vector', () => {
    const out = new Float32Array(2);
    multiplyMatrixVector(Float32Array.of(1, 2, 3, 4, 5, 6), 2, 3, Float32Array.of(1, 0, -1), out);
    expect(Array.from(out)).toEqual([-2, -2]);
  });
});

describe('clampInPlace', () => {
  it('clamps both sides and leaves values inside untouched', () => {
    const buffer = Float32Array.of(5, -5, 1, -1);
    clampInPlace(buffer, 2);
    expect(Array.from(buffer)).toEqual([2, -2, 1, -1]);
  });
});

describe('scaleInPlace', () => {
  it('scales every element', () => {
    const buffer = Float32Array.of(1, -2, 4);
    scaleInPlace(buffer, 0.5);
    expect(Array.from(buffer)).toEqual([0.5, -1, 2]);
  });
});

describe('applyDeltaRule', () => {
  it('moves a single weight toward the target by rate * error', () => {
    const weights = new Float32Array(1);
    applyDeltaRule(weights, 1, 1, Float32Array.of(1), Float32Array.of(1), 0.5, 10);
    expect(weights[0]).toBeCloseTo(0.5);
    applyDeltaRule(weights, 1, 1, Float32Array.of(1), Float32Array.of(1), 0.5, 10);
    expect(weights[0]).toBeCloseTo(0.75);
  });

  it('does not change weights of absent cues', () => {
    const weights = new Float32Array(2);
    applyDeltaRule(weights, 1, 2, Float32Array.of(1, 0), Float32Array.of(1), 0.5, 10);
    expect(weights[1]).toBe(0);
  });

  it('clamps weights to the limit', () => {
    const weights = new Float32Array(1);
    applyDeltaRule(weights, 1, 1, Float32Array.of(1), Float32Array.of(100), 1, 0.5);
    expect(weights[0]).toBe(0.5);
  });

  it('keeps weights finite and bounded for any inputs (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: 0, max: 1, noNaN: true }), { minLength: 3, maxLength: 3 }),
        fc.float({ min: -5, max: 5, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (inputs, target, rate) => {
          const weights = new Float32Array(3);
          for (let step = 0; step < 20; step += 1) {
            applyDeltaRule(
              weights,
              1,
              3,
              Float32Array.from(inputs),
              Float32Array.of(target),
              rate,
              1,
            );
          }
          return weights.every((w) => Number.isFinite(w) && Math.abs(w) <= 1);
        },
      ),
    );
  });
});
