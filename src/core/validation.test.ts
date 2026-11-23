import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from './errors';
import {
  assertBound,
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertNonNegativeBuffer,
  assertPositive,
  assertPositiveInteger,
  assertSize,
  assertWithinMagnitude,
  createFloat32Array,
  FLOAT32_MAX,
  isFiniteBuffer,
} from './validation';

describe('assertSize', () => {
  it('accepts a matching length', () => {
    expect(() => assertSize('x', new Float32Array(3), 3)).not.toThrow();
  });
  it('rejects a different length with a descriptive error', () => {
    expect(() => assertSize('x', new Float32Array(2), 3)).toThrow(SizeMismatchError);
    expect(() => assertSize('x', new Float32Array(2), 3)).toThrow(/"x".*2.*3/);
  });
});

describe('assertFiniteBuffer', () => {
  it('accepts finite values', () => {
    expect(() => assertFiniteBuffer('x', Float32Array.of(0, -1, 2.5))).not.toThrow();
  });
  it.each([NaN, Infinity, -Infinity])('rejects %s and names the index', (bad) => {
    expect(() => assertFiniteBuffer('x', Float32Array.of(0, bad))).toThrow(/x\[1\]/);
    expect(() => assertFiniteBuffer('x', Float32Array.of(0, bad))).toThrow(InvalidParameterError);
  });
});

describe('assertPositiveInteger', () => {
  it('accepts integers >= 1', () => {
    expect(() => assertPositiveInteger('n', 1)).not.toThrow();
  });
  it.each([0, -1, 1.5, NaN, Infinity])('rejects %s', (bad) => {
    expect(() => assertPositiveInteger('n', bad)).toThrow(InvalidParameterError);
  });
});

describe('assertPositive', () => {
  it('accepts positive finite numbers', () => {
    expect(() => assertPositive('p', 0.001)).not.toThrow();
  });
  it.each([0, -1, NaN, Infinity])('rejects %s', (bad) => {
    expect(() => assertPositive('p', bad)).toThrow(InvalidParameterError);
  });
});

describe('assertNonNegative', () => {
  it('accepts zero and positives', () => {
    expect(() => assertNonNegative('p', 0)).not.toThrow();
    expect(() => assertNonNegative('p', 3)).not.toThrow();
  });
  it.each([-0.1, NaN, Infinity])('rejects %s', (bad) => {
    expect(() => assertNonNegative('p', bad)).toThrow(InvalidParameterError);
  });
});

describe('assertFiniteInRange', () => {
  it('accepts values on and inside the bounds', () => {
    expect(() => assertFiniteInRange('r', 0, 0, 1)).not.toThrow();
    expect(() => assertFiniteInRange('r', 1, 0, 1)).not.toThrow();
  });
  it.each([-0.01, 1.01, NaN, Infinity])('rejects %s', (bad) => {
    expect(() => assertFiniteInRange('r', bad, 0, 1)).toThrow(/\[0, 1\]/);
  });
});

describe('assertNonNegativeBuffer', () => {
  it('accepts zeros and positives', () => {
    expect(() => assertNonNegativeBuffer('x', Float32Array.of(0, 1, 2.5))).not.toThrow();
  });
  it.each([-0.5, NaN, Infinity])('rejects %s and names the index', (bad) => {
    expect(() => assertNonNegativeBuffer('x', Float32Array.of(0, bad))).toThrow(/x\[1\]/);
    expect(() => assertNonNegativeBuffer('x', Float32Array.of(0, bad))).toThrow(
      InvalidParameterError,
    );
  });
});

describe('assertBound', () => {
  it('accepts positive finite values up to the Float32 maximum', () => {
    expect(() => assertBound('b', 1)).not.toThrow();
    expect(() => assertBound('b', FLOAT32_MAX)).not.toThrow();
  });
  it.each([0, -1, NaN, Infinity, 1e39])('rejects %s', (bad) => {
    expect(() => assertBound('b', bad)).toThrow(InvalidParameterError);
  });
});

describe('assertWithinMagnitude', () => {
  it('accepts values inside the bound, including the Float32-rounded bound itself', () => {
    expect(() => assertWithinMagnitude('s', Float32Array.of(-0.5, 0, 0.5), 0.5)).not.toThrow();
    expect(() => assertWithinMagnitude('s', Float32Array.of(0.1), 0.1)).not.toThrow();
  });
  it.each([0.51, -0.51, NaN, Infinity])('rejects %s and names the index', (bad) => {
    expect(() => assertWithinMagnitude('s', Float32Array.of(0, bad), 0.5)).toThrow(/s\[1\]/);
  });
});

describe('isFiniteBuffer', () => {
  it('reports finiteness', () => {
    expect(isFiniteBuffer(Float32Array.of(0, 1, -2))).toBe(true);
    expect(isFiniteBuffer(Float32Array.of(0, NaN))).toBe(false);
    expect(isFiniteBuffer(Float32Array.of(Infinity))).toBe(false);
  });
});

describe('createFloat32Array', () => {
  it('allocates a zero-filled array', () => {
    expect(Array.from(createFloat32Array('n', 3))).toEqual([0, 0, 0]);
  });
  it.each([1e10, 2 ** 60, 1e300, 1.5, -1, NaN])(
    'rejects the size %s with a descriptive error',
    (bad) => {
      expect(() => createFloat32Array('inputSize * outputSize', bad)).toThrow(
        InvalidParameterError,
      );
      expect(() => createFloat32Array('inputSize * outputSize', bad)).toThrow(
        /inputSize \* outputSize/,
      );
    },
  );
});
