import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from './errors';
import {
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertPositive,
  assertPositiveInteger,
  assertSize,
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
