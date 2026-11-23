/** Function approximation with `mlp-sgd`: XOR across several seeds and a sine regression. */
import { describe, expect, it } from 'vitest';
import { createSeededRng, MlpSgdLearner } from '../../src';

const XOR: Array<[number[], number]> = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
];

function predict(learner: MlpSgdLearner, inputs: number[]): number {
  const out = new Float32Array(1);
  learner.predict(Float32Array.from(inputs), out);
  return out[0] as number;
}

describe('mlp-sgd function fitting', () => {
  it.each([1, 2, 3, 4, 5, 6, 7, 8])('learns XOR with seed %d', (seed) => {
    const learner = new MlpSgdLearner({
      layerSizes: [2, 6, 1],
      rng: createSeededRng(seed),
      maxWeight: 10,
    });
    for (let epoch = 0; epoch < 4000; epoch += 1) {
      for (const [inputs, label] of XOR) {
        learner.learn(Float32Array.from(inputs), Float32Array.of(label), 0.1);
      }
    }
    for (const [inputs, label] of XOR) {
      expect(Math.abs(predict(learner, inputs) - label)).toBeLessThan(0.25);
    }
  });

  it('fits a sine wave on [0, 1] with small mean squared error', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [1, 12, 1],
      rng: createSeededRng(3),
      maxWeight: 10,
    });
    const rng = createSeededRng(10);
    const curve = (x: number) => 0.5 + 0.4 * Math.sin(2 * Math.PI * x);
    for (let step = 0; step < 60_000; step += 1) {
      const x = rng.next();
      learner.learn(Float32Array.of(x), Float32Array.of(curve(x)), 0.05);
    }
    let squaredError = 0;
    const samples = 100;
    for (let i = 0; i < samples; i += 1) {
      const x = i / (samples - 1);
      squaredError += (predict(learner, [x]) - curve(x)) ** 2;
    }
    expect(squaredError / samples).toBeLessThan(0.01);
  });
});
