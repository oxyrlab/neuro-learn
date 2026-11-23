/**
 * Writes `weights · vector` into `out`, where `weights` is a row-major
 * `rowCount × columnCount` matrix. Allocation-free.
 *
 * @param weights - Row-major matrix, length `rowCount * columnCount`.
 * @param rowCount - Number of rows (outputs).
 * @param columnCount - Number of columns (inputs).
 * @param vector - Input of length `columnCount`.
 * @param out - Output of length `rowCount`, overwritten.
 */
export function multiplyMatrixVector(
  weights: Float32Array,
  rowCount: number,
  columnCount: number,
  vector: Float32Array,
  out: Float32Array,
): void {
  for (let row = 0; row < rowCount; row += 1) {
    const base = row * columnCount;
    let sum = 0;
    for (let column = 0; column < columnCount; column += 1) {
      sum += (weights[base + column] as number) * (vector[column] as number);
    }
    out[row] = sum;
  }
}

/**
 * Clamps `value` to `[-limit, limit]`.
 *
 * @param value - Value to clamp.
 * @param limit - Non-negative magnitude bound.
 */
export function clampMagnitude(value: number, limit: number): number {
  return value > limit ? limit : value < -limit ? -limit : value;
}

/**
 * Multiplies every element of `buffer` by `factor`, in place.
 *
 * @param buffer - Buffer to scale.
 * @param factor - Multiplier.
 */
export function scaleInPlace(buffer: Float32Array, factor: number): void {
  for (let index = 0; index < buffer.length; index += 1) {
    buffer[index] = (buffer[index] as number) * factor;
  }
}

/**
 * Delta rule on a row-major weight matrix: for each output `j`,
 * `error = target[j] - Σ weights[j][i] * inputs[i]` and
 * `weights[j][i] += rate * inputs[i] * error`, clamped to `[-limit, limit]`.
 * Allocation-free. This is the Rescorla–Wagner update (`ΔV = α(λ − ΣV)`).
 *
 * @param weights - Row-major matrix, modified in place.
 * @param rowCount - Number of outputs.
 * @param columnCount - Number of inputs.
 * @param inputs - Cue strengths of length `columnCount`.
 * @param targets - Outcome strengths λ of length `rowCount`.
 * @param rate - Learning rate α.
 * @param limit - Maximum absolute weight.
 */
export function applyDeltaRule(
  weights: Float32Array,
  rowCount: number,
  columnCount: number,
  inputs: Float32Array,
  targets: Float32Array,
  rate: number,
  limit: number,
): void {
  for (let row = 0; row < rowCount; row += 1) {
    const base = row * columnCount;
    let predicted = 0;
    for (let column = 0; column < columnCount; column += 1) {
      predicted += (weights[base + column] as number) * (inputs[column] as number);
    }
    const step = rate * ((targets[row] as number) - predicted);
    for (let column = 0; column < columnCount; column += 1) {
      const next = (weights[base + column] as number) + step * (inputs[column] as number);
      weights[base + column] = clampMagnitude(next, limit);
    }
  }
}
