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

/**
 * Throws {@link InvalidParameterError} unless every element of `buffer` is finite and >= 0.
 * Allocation-free.
 *
 * @param name - Buffer name used in the error message.
 * @param buffer - Buffer to check.
 */
export function assertNonNegativeBuffer(name: string, buffer: Float32Array): void {
  for (let index = 0; index < buffer.length; index += 1) {
    const value = buffer[index] as number;
    if (!Number.isFinite(value) || value < 0) {
      throw new InvalidParameterError(`${name}[${index}]`, value, 'a finite number >= 0');
    }
  }
}

/** Largest finite value a `Float32Array` element can hold. */
export const FLOAT32_MAX = 3.4028234663852886e38;

/**
 * Throws {@link InvalidParameterError} unless `value` is a usable magnitude bound: finite,
 * strictly positive and not larger than the largest `Float32` (a larger bound could never be
 * enforced because stored values would overflow to Infinity).
 *
 * @param name - Parameter name.
 * @param value - Value to check.
 */
export function assertBound(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0 || value > FLOAT32_MAX) {
    throw new InvalidParameterError(
      name,
      value,
      'a finite number > 0 and <= 3.4028234663852886e38',
    );
  }
}

/**
 * Throws {@link InvalidParameterError} unless every element of `buffer` is finite and within
 * `[-limit, limit]` as stored in a `Float32Array` (the limit is rounded to Float32 first, because
 * clamped values are stored that way). Allocation-free.
 *
 * @param name - Buffer name used in the error message.
 * @param buffer - Buffer to check.
 * @param limit - Magnitude bound the values must respect.
 */
export function assertWithinMagnitude(name: string, buffer: Float32Array, limit: number): void {
  const stored = Math.fround(limit);
  for (let index = 0; index < buffer.length; index += 1) {
    const value = buffer[index] as number;
    if (!Number.isFinite(value) || value > stored || value < -stored) {
      throw new InvalidParameterError(
        `${name}[${index}]`,
        value,
        `a finite number in [-${limit}, ${limit}]`,
      );
    }
  }
}

/**
 * Returns true if every element of `buffer` is finite. Allocation-free.
 *
 * @param buffer - Buffer to check.
 */
export function isFiniteBuffer(buffer: Float32Array): boolean {
  for (let index = 0; index < buffer.length; index += 1) {
    if (!Number.isFinite(buffer[index] as number)) return false;
  }
  return true;
}

/**
 * Allocates a zero-filled `Float32Array`, turning an impossible size into a descriptive error
 * instead of a raw `RangeError`.
 *
 * @param name - Description of what the length comes from, e.g. `"inputSize * outputSize"`.
 * @param length - Requested element count.
 * @throws InvalidParameterError if `length` is not a safe integer or cannot be allocated.
 */
export function createFloat32Array(name: string, length: number): Float32Array {
  if (!Number.isSafeInteger(length)) {
    throw new InvalidParameterError(name, length, 'a size small enough to allocate');
  }
  try {
    return new Float32Array(length);
  } catch {
    // For a safe integer length the only possible failure is a RangeError (size too large).
    throw new InvalidParameterError(name, length, 'a size small enough to allocate');
  }
}
