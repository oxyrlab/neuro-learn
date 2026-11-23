/**
 * Independent float64 reference implementations (plain loops, no library). They define the "true"
 * answer so accuracy is judged neutrally instead of one library being graded against the other.
 */

export function mlpOutputs(
  sizes: readonly number[],
  w: ArrayLike<number>,
  x: ArrayLike<number>,
): Float64Array[] {
  const outs: Float64Array[] = [Float64Array.from(x)];
  let offset = 0;
  for (let layer = 0; layer < sizes.length - 1; layer += 1) {
    const fanIn = sizes[layer] as number;
    const fanOut = sizes[layer + 1] as number;
    const previous = outs[layer] as Float64Array;
    const next = new Float64Array(fanOut);
    for (let row = 0; row < fanOut; row += 1) {
      let sum = w[offset + fanOut * fanIn + row] as number;
      for (let col = 0; col < fanIn; col += 1)
        sum += (w[offset + row * fanIn + col] as number) * (previous[col] as number);
      next[row] = layer < sizes.length - 2 ? Math.tanh(sum) : sum;
    }
    outs.push(next);
    offset += fanOut * (fanIn + 1);
  }
  return outs;
}

/** One online SGD step on 0.5 * sum((y - t)^2), tanh hidden layers, linear output. Mutates `w`. */
export function mlpStep(
  sizes: readonly number[],
  w: Float64Array,
  x: ArrayLike<number>,
  t: ArrayLike<number>,
  rate: number,
): void {
  const outs = mlpOutputs(sizes, w, x);
  const layerCount = sizes.length - 1;
  const last = outs[layerCount] as Float64Array;
  let delta = Float64Array.from(last, (y, k) => y - (t[k] as number));
  const offsets: number[] = [];
  let total = 0;
  for (let layer = 0; layer < layerCount; layer += 1) {
    offsets.push(total);
    total += (sizes[layer + 1] as number) * ((sizes[layer] as number) + 1);
  }
  for (let layer = layerCount - 1; layer >= 0; layer -= 1) {
    const fanIn = sizes[layer] as number;
    const fanOut = sizes[layer + 1] as number;
    const start = offsets[layer] as number;
    const inputs = outs[layer] as Float64Array;
    let previousDelta = new Float64Array(0);
    if (layer > 0) {
      previousDelta = new Float64Array(fanIn);
      for (let col = 0; col < fanIn; col += 1) {
        let sum = 0;
        for (let row = 0; row < fanOut; row += 1)
          sum += (w[start + row * fanIn + col] as number) * (delta[row] as number);
        const y = inputs[col] as number;
        previousDelta[col] = sum * (1 - y * y);
      }
    }
    for (let row = 0; row < fanOut; row += 1) {
      const d = delta[row] as number;
      for (let col = 0; col < fanIn; col += 1)
        w[start + row * fanIn + col] =
          (w[start + row * fanIn + col] as number) - rate * d * (inputs[col] as number);
      w[start + fanOut * fanIn + row] = (w[start + fanOut * fanIn + row] as number) - rate * d;
    }
    delta = previousDelta;
  }
}

/** Delta rule: w += rate * (target - w.x) * x, per output row. Mutates `w` (outputs x inputs). */
export function deltaStep(
  w: Float64Array,
  inputs: number,
  outputs: number,
  x: ArrayLike<number>,
  t: ArrayLike<number>,
  rate: number,
): void {
  for (let row = 0; row < outputs; row += 1) {
    let predicted = 0;
    for (let col = 0; col < inputs; col += 1)
      predicted += (w[row * inputs + col] as number) * (x[col] as number);
    const step = rate * ((t[row] as number) - predicted);
    for (let col = 0; col < inputs; col += 1)
      w[row * inputs + col] = (w[row * inputs + col] as number) + step * (x[col] as number);
  }
}

/** One Elman step. Mutates `hidden`, returns the outputs. */
export function rnnStep(
  sizes: readonly number[],
  w: ArrayLike<number>,
  hidden: Float64Array,
  x: ArrayLike<number>,
): Float64Array {
  const [inputs, units, outputs] = sizes as [number, number, number];
  const recurrentStart = units * inputs;
  const biasStart = recurrentStart + units * units;
  const outputStart = biasStart + units;
  const outputBiasStart = outputStart + outputs * units;
  const next = new Float64Array(units);
  for (let u = 0; u < units; u += 1) {
    let sum = w[biasStart + u] as number;
    for (let c = 0; c < inputs; c += 1) sum += (w[u * inputs + c] as number) * (x[c] as number);
    for (let c = 0; c < units; c += 1)
      sum += (w[recurrentStart + u * units + c] as number) * (hidden[c] as number);
    next[u] = Math.tanh(sum);
  }
  hidden.set(next);
  const out = new Float64Array(outputs);
  for (let r = 0; r < outputs; r += 1) {
    let sum = w[outputBiasStart + r] as number;
    for (let c = 0; c < units; c += 1)
      sum += (w[outputStart + r * units + c] as number) * (next[c] as number);
    out[r] = sum;
  }
  return out;
}
