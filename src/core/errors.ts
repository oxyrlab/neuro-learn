/**
 * Thrown when a constructor or function receives a parameter outside its documented domain.
 *
 * The message always states which parameter is wrong, the offending value and the allowed range,
 * so callers can fix the problem without reading the source.
 *
 * @example
 * throw new InvalidParameterError('learningRate', 2, 'a number in (0, 1]');
 */
export class InvalidParameterError extends Error {
  public readonly parameter: string;
  public readonly actual: unknown;

  /**
   * @param parameter - Name of the offending parameter, e.g. `"learningRate"`.
   * @param actual - The value that was received.
   * @param expected - Human-readable description of what is allowed.
   */
  public constructor(parameter: string, actual: unknown, expected: string) {
    super(`Invalid parameter "${parameter}": got ${String(actual)}, expected ${expected}.`);
    this.name = 'InvalidParameterError';
    this.parameter = parameter;
    this.actual = actual;
  }
}

/**
 * Thrown when a Float32Array (or other buffer) has a length that does not match what the
 * receiving component was created for.
 *
 * @example
 * throw new SizeMismatchError('features', 3, 4);
 */
export class SizeMismatchError extends Error {
  public readonly buffer: string;
  public readonly actual: number;
  public readonly expected: number;

  /**
   * @param buffer - Name of the buffer, e.g. `"features"`.
   * @param actual - Received length.
   * @param expected - Required length.
   */
  public constructor(buffer: string, actual: number, expected: number) {
    super(`Size mismatch for "${buffer}": got length ${actual}, expected length ${expected}.`);
    this.name = 'SizeMismatchError';
    this.buffer = buffer;
    this.actual = actual;
    this.expected = expected;
  }
}
