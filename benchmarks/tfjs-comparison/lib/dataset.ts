/** Deterministic data and initial weights, identical for every implementation. */
import type { Scenario } from './spec';
import { BLOCKS, POOL } from './spec';

/** mulberry32: tiny seeded PRNG so the data does not depend on any library under test. */
export function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Dataset {
  readonly inputs: Float32Array[];
  readonly targets: Float32Array[];
  /** BLOCKS contiguous blocks of `batch` samples, for batched forward calls. */
  readonly blocks: Float32Array[];
  readonly heldInputs: Float32Array[];
  readonly heldTargets: Float32Array[];
}

function sampleInputs(count: number, size: number, rng: () => number): Float32Array[] {
  return Array.from({ length: count }, () => Float32Array.from({ length: size }, rng));
}

/** A fixed teacher gives learnable targets: tanh of a random projection (MLP) or a linear map (delta rule). */
function makeTeacher(sc: Scenario, rng: () => number): (x: Float32Array) => Float32Array {
  const inputs = sc.layers[0] as number;
  const outputs = sc.layers[sc.layers.length - 1] as number;
  const linear = sc.kind === 'delta-rule';
  const scale = linear ? 2 / inputs : 2 / Math.sqrt(inputs);
  const matrix = Array.from({ length: outputs }, () =>
    Float64Array.from({ length: inputs }, () => (rng() * 2 - 1) * scale),
  );
  const offsets = Float64Array.from({ length: outputs }, () => (linear ? 0 : rng() - 0.5));
  return (x) =>
    Float32Array.from(matrix, (row, k) => {
      let sum = offsets[k] as number;
      for (let j = 0; j < inputs; j += 1) sum += (row[j] as number) * (x[j] as number);
      return linear ? sum : 0.8 * Math.tanh(sum);
    });
}

export function makeDataset(sc: Scenario): Dataset {
  const inputSize = sc.layers[0] as number;
  const rng = createRng(20260101);
  const teacher = makeTeacher(sc, rng);
  const inputs = sampleInputs(POOL, inputSize, rng);
  const heldInputs = sampleInputs(POOL, inputSize, rng);
  const blocks = Array.from({ length: BLOCKS }, (_, block) => {
    const flat = new Float32Array(sc.batch * inputSize);
    for (let k = 0; k < sc.batch; k += 1) {
      flat.set(inputs[(block * sc.batch + k) & (POOL - 1)] as Float32Array, k * inputSize);
    }
    return flat;
  });
  const hasTargets = sc.kind === 'mlp-train' || sc.kind === 'delta-rule';
  return {
    inputs,
    blocks,
    heldInputs,
    targets: hasTargets ? inputs.map(teacher) : [],
    heldTargets: hasTargets ? heldInputs.map(teacher) : [],
  };
}

function xavier(count: number, fanSum: number, rng: () => number): Float32Array {
  const limit = Math.sqrt(6 / fanSum);
  return Float32Array.from({ length: count }, () => (rng() * 2 - 1) * limit);
}

/**
 * Initial weights in the neuro-learn layout (lib/layout.ts), Glorot-uniform with zero biases,
 * the same scheme neuro-learn uses. Both libraries start from exactly these numbers.
 */
export function makeInitialWeights(sc: Scenario): Float32Array {
  const rng = createRng(77);
  const sizes = sc.layers;
  if (sc.kind === 'delta-rule')
    return new Float32Array((sizes[0] as number) * (sizes[1] as number));
  if (sc.kind === 'rnn-forward') {
    const [inputs, hidden, outputs] = sizes as [number, number, number];
    const parts = [
      xavier(hidden * inputs, inputs + hidden, rng),
      xavier(hidden * hidden, 2 * hidden, rng),
      new Float32Array(hidden),
      xavier(outputs * hidden, hidden + outputs, rng),
      new Float32Array(outputs),
    ];
    const all = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
    let offset = 0;
    for (const part of parts) {
      all.set(part, offset);
      offset += part.length;
    }
    return all;
  }
  const networks = sc.kind === 'population-forward' ? sc.batch : 1; // every creature gets its own weights
  const parts: Float32Array[] = [];
  for (let network = 0; network < networks; network += 1) {
    for (let layer = 0; layer < sizes.length - 1; layer += 1) {
      const fanIn = sizes[layer] as number;
      const fanOut = sizes[layer + 1] as number;
      parts.push(xavier(fanOut * fanIn, fanIn + fanOut, rng), new Float32Array(fanOut));
    }
  }
  const all = new Float32Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    all.set(part, offset);
    offset += part.length;
  }
  return all;
}
