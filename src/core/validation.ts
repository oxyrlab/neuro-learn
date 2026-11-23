import { InvalidParameterError, SizeMismatchError } from './errors';

/**
 * Throws {@link SizeMismatchError} unless `buffer.length === expected`.
 *
 * @param name - Buffer name used in the error message.
 * @param buffer - Buffer to check.
 * @param expected - Required length.
 */
export function assertSize(name: string, buffer: Float32Array, expected: number): void {
  if (buffer.length !== expected) {
    throw new SizeMismatchError(name, buffer.length, expected);
  }
}

/**
 * Throws {@link InvalidParameterError} unless every element of `buffer` is a finite number.
 * Allocation-free.
 *
 * @param name - Buffer name used in the error message.
 * @param buffer - Buffer to check.
 */
export function assertFiniteBuffer(name: string, buffer: Float32Array): void {
  for (let index = 0; index < buffer.length; index += 1) {
    const value = buffer[index] as number;
    if (!Number.isFinite(value)) {
      throw new InvalidParameterError(`${name}[${index}]`, value, 'a finite number');
    }
  }
}

/**
 * Throws {@link InvalidParameterError} unless `value` is an integer >= 1.
 *
 * @param name - Parameter name.
 * @param value - Value to check.
 */
export function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(name, value, 'an integer >= 1');
  }
}

/**
 * Throws {@link InvalidParameterError} unless `value` is finite and strictly greater than zero.
 *
 * @param name - Parameter name.
 * @param value - Value to check.
 */
export function assertPositive(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(name, value, 'a finite number > 0');
  }
}

/**
 * Throws {@link InvalidParameterError} unless `value` is finite and >= 0.
 *
 * @param name - Parameter name.
 * @param value - Value to check.
 */
export function assertNonNegative(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(name, value, 'a finite number >= 0');
  }
}

/**
 * Throws {@link InvalidParameterError} unless `value` is finite and inside `[min, max]`.
 *
 * @param name - Parameter name.
 * @param value - Value to check.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 */
export function assertFiniteInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new InvalidParameterError(name, value, `a finite number in [${min}, ${max}]`);
  }
}
