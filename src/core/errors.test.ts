import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from './errors';

describe('InvalidParameterError', () => {
  it('describes the parameter, value and expectation', () => {
    const error = new InvalidParameterError('learningRate', 2, 'a number in (0, 1]');
    expect(error.message).toBe(
      'Invalid parameter "learningRate": got 2, expected a number in (0, 1].',
    );
    expect(error.name).toBe('InvalidParameterError');
    expect(error.parameter).toBe('learningRate');
    expect(error.actual).toBe(2);
    expect(error).toBeInstanceOf(Error);
  });
});

describe('SizeMismatchError', () => {
  it('describes the buffer and both lengths', () => {
    const error = new SizeMismatchError('features', 3, 4);
    expect(error.message).toBe('Size mismatch for "features": got length 3, expected length 4.');
    expect(error.name).toBe('SizeMismatchError');
    expect(error.buffer).toBe('features');
    expect(error.actual).toBe(3);
    expect(error.expected).toBe(4);
    expect(error).toBeInstanceOf(Error);
  });
});
