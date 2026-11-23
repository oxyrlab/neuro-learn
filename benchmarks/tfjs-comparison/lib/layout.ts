/**
 * Weight layouts. neuro-learn stores each dense layer as `fanOut x fanIn` row-major (one row per
 * output unit) followed by `fanOut` biases; TensorFlow.js stores kernels as `fanIn x fanOut`.
 */

export function transpose(source: ArrayLike<number>, rows: number, columns: number): Float32Array {
  const result = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      result[column * rows + row] = source[row * columns + column] as number;
    }
  }
  return result;
}

export interface DenseParams {
  readonly fanIn: number;
  readonly fanOut: number;
  /** tf layout: fanIn x fanOut, row-major. */
  readonly kernel: Float32Array;
  readonly bias: Float32Array;
}

export function mlpToDense(sizes: readonly number[], weights: ArrayLike<number>): DenseParams[] {
  const params: DenseParams[] = [];
  let offset = 0;
  for (let layer = 0; layer < sizes.length - 1; layer += 1) {
    const fanIn = sizes[layer] as number;
    const fanOut = sizes[layer + 1] as number;
    const matrix = Float32Array.from(
      { length: fanOut * fanIn },
      (_, k) => weights[offset + k] as number,
    );
    const bias = Float32Array.from(
      { length: fanOut },
      (_, k) => weights[offset + fanOut * fanIn + k] as number,
    );
    params.push({ fanIn, fanOut, kernel: transpose(matrix, fanOut, fanIn), bias });
    offset += fanOut * (fanIn + 1);
  }
  return params;
}

export function denseToMlp(params: readonly DenseParams[]): Float32Array {
  const parts = params.flatMap((p) => [transpose(p.kernel, p.fanIn, p.fanOut), p.bias]);
  const all = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    all.set(part, offset);
    offset += part.length;
  }
  return all;
}

export interface RnnParams {
  /** All in tf layout (fanIn x fanOut). */
  readonly inputKernel: Float32Array;
  readonly recurrentKernel: Float32Array;
  readonly hiddenBias: Float32Array;
  readonly outputKernel: Float32Array;
  readonly outputBias: Float32Array;
}

export function rnnToParams(sizes: readonly number[], weights: Float32Array): RnnParams {
  const [inputs, hidden, outputs] = sizes as [number, number, number];
  let offset = 0;
  const take = (count: number): Float32Array => {
    const part = weights.slice(offset, offset + count);
    offset += count;
    return part;
  };
  const inputToHidden = take(hidden * inputs);
  const hiddenToHidden = take(hidden * hidden);
  const hiddenBias = take(hidden);
  const hiddenToOutput = take(outputs * hidden);
  const outputBias = take(outputs);
  return {
    inputKernel: transpose(inputToHidden, hidden, inputs),
    recurrentKernel: transpose(hiddenToHidden, hidden, hidden),
    hiddenBias,
    outputKernel: transpose(hiddenToOutput, outputs, hidden),
    outputBias,
  };
}

export function rnnFromParams(sizes: readonly number[], p: RnnParams): Float32Array {
  const [inputs, hidden, outputs] = sizes as [number, number, number];
  const parts = [
    transpose(p.inputKernel, inputs, hidden),
    transpose(p.recurrentKernel, hidden, hidden),
    p.hiddenBias,
    transpose(p.outputKernel, hidden, outputs),
    p.outputBias,
  ];
  const all = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    all.set(part, offset);
    offset += part.length;
  }
  return all;
}
