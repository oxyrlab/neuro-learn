import { describe, expect, it } from 'vitest';
import { InvalidParameterError } from '../core/errors';
import { ACTIVATIONS, getActivation } from './activations';
import type { ActivationId } from './activations';

const ids: ActivationId[] = ['tanh', 'relu', 'sigmoid', 'linear'];

describe('activations', () => {
  it('computes known values', () => {
    expect(ACTIVATIONS.tanh.apply(0)).toBe(0);
    expect(ACTIVATIONS.sigmoid.apply(0)).toBe(0.5);
    expect(ACTIVATIONS.relu.apply(-3)).toBe(0);
    expect(ACTIVATIONS.relu.apply(2)).toBe(2);
    expect(ACTIVATIONS.linear.apply(-7)).toBe(-7);
  });

  it('keeps sigmoid and tanh inside their ranges even for huge inputs', () => {
    for (const x of [-1e6, 1e6]) {
      expect(ACTIVATIONS.sigmoid.apply(x)).toBeGreaterThanOrEqual(0);
      expect(ACTIVATIONS.sigmoid.apply(x)).toBeLessThanOrEqual(1);
      expect(Math.abs(ACTIVATIONS.tanh.apply(x))).toBeLessThanOrEqual(1);
    }
  });

  it.each(ids)('%s: derivativeFromOutput matches a numerical derivative', (id) => {
    const activation = ACTIVATIONS[id];
    for (const x of [-2, -0.5, 0.3, 1.7]) {
      const eps = 1e-6;
      const numeric = (activation.apply(x + eps) - activation.apply(x - eps)) / (2 * eps);
      expect(activation.derivativeFromOutput(activation.apply(x))).toBeCloseTo(numeric, 4);
    }
  });

  it('relu derivative is 0 at and below zero', () => {
    expect(ACTIVATIONS.relu.derivativeFromOutput(0)).toBe(0);
  });

  it('each entry carries its own id', () => {
    for (const id of ids) expect(ACTIVATIONS[id].id).toBe(id);
  });

  it('getActivation returns the registered activation', () => {
    expect(getActivation('tanh')).toBe(ACTIVATIONS.tanh);
  });

  it('getActivation rejects unknown ids listing the valid ones', () => {
    expect(() => getActivation('swish', 'hiddenActivation')).toThrow(InvalidParameterError);
    expect(() => getActivation('swish', 'hiddenActivation')).toThrow(/hiddenActivation.*tanh/);
    expect(() => getActivation('toString')).toThrow(InvalidParameterError);
  });
});
