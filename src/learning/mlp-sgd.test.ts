import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from '../core/errors';
import { createSeededRng } from '../core/rng';
import { runLearnerContract } from '../testing/learner-contract';
import { MlpSgdLearner } from './mlp-sgd';

runLearnerContract(
  'mlp-sgd',
  () =>
    new MlpSgdLearner({
      layerSizes: [4, 5, 2],
      rng: createSeededRng(3),
      maxWeight: 2,
      forgetRatePerSecond: 0.1,
    }),
  { initialState: 'zero-prediction', maxAbsState: 2 },
);

function predictOne(learner: MlpSgdLearner, features: number[]): number {
  const out = new Float32Array(1);
  learner.predict(Float32Array.from(features), out);
  return out[0] as number;
}

const XOR: Array<[number[], number]> = [
  [[0, 0], 0],
  [[0, 1], 1],
  [[1, 0], 1],
  [[1, 1], 0],
];

describe('MlpSgdLearner', () => {
  it('learns XOR (needs the hidden layer)', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [2, 6, 1],
      rng: createSeededRng(1),
      maxWeight: 10,
    });
    for (let epoch = 0; epoch < 4000; epoch += 1) {
      for (const [inputs, label] of XOR) {
        learner.learn(Float32Array.from(inputs), Float32Array.of(label), 0.1);
      }
    }
    for (const [inputs, label] of XOR) {
      expect(Math.abs(predictOne(learner, inputs) - label)).toBeLessThan(0.2);
    }
  });

  it('learns a linearly separable function without hidden layers (sigmoid output)', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [2, 1],
      rng: createSeededRng(1),
      outputActivation: 'sigmoid',
    });
    const and: Array<[number[], number]> = [
      [[0, 0], 0],
      [[0, 1], 0],
      [[1, 0], 0],
      [[1, 1], 1],
    ];
    for (let epoch = 0; epoch < 3000; epoch += 1) {
      for (const [inputs, label] of and) {
        learner.learn(Float32Array.from(inputs), Float32Array.of(label), 0.5);
      }
    }
    for (const [inputs, label] of and) {
      expect(Math.abs(predictOne(learner, inputs) - label)).toBeLessThan(0.2);
    }
  });

  it('reduces the error on a repeated example', () => {
    const learner = new MlpSgdLearner({ layerSizes: [2, 4, 1], rng: createSeededRng(2) });
    const before = Math.abs(predictOne(learner, [1, 0]) - 1);
    for (let i = 0; i < 50; i += 1) learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.1);
    expect(Math.abs(predictOne(learner, [1, 0]) - 1)).toBeLessThan(before);
  });

  it('restore replaces all state, even on an instance with a different seed', () => {
    const trained = new MlpSgdLearner({ layerSizes: [2, 4, 1], rng: createSeededRng(1) });
    for (let i = 0; i < 100; i += 1) trained.learn(Float32Array.of(1, 1), Float32Array.of(1), 0.1);
    const other = new MlpSgdLearner({ layerSizes: [2, 4, 1], rng: createSeededRng(99) });
    other.restore(trained.serialize());
    expect(predictOne(other, [1, 1])).toBe(predictOne(trained, [1, 1]));
    expect(predictOne(other, [0.3, 0.6])).toBe(predictOne(trained, [0.3, 0.6]));
  });

  it('clamps weights to maxWeight even with a huge target', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [1, 2, 1],
      rng: createSeededRng(1),
      maxWeight: 0.5,
    });
    for (let i = 0; i < 200; i += 1) learner.learn(Float32Array.of(1), Float32Array.of(1000), 1);
    expect(Math.max(...Array.from(learner.serialize()).map(Math.abs))).toBeLessThanOrEqual(0.5);
  });

  it('keeps initial hidden weights within maxWeight', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [3, 8, 1],
      rng: createSeededRng(1),
      maxWeight: 0.01,
    });
    expect(Math.max(...Array.from(learner.serialize()).map(Math.abs))).toBeLessThanOrEqual(0.01);
  });

  it('forgets exponentially and not when the rate is zero', () => {
    const forgetting = new MlpSgdLearner({
      layerSizes: [1, 1],
      rng: createSeededRng(1),
      forgetRatePerSecond: 0.5,
    });
    forgetting.restore(Float32Array.of(1, 0.4));
    forgetting.decay(2);
    expect(forgetting.serialize()[0]).toBeCloseTo(Math.exp(-1), 5);
    const steady = new MlpSgdLearner({ layerSizes: [1, 1], rng: createSeededRng(1) });
    steady.restore(Float32Array.of(1, 0.4));
    steady.decay(50);
    expect(steady.serialize()[1]).toBeCloseTo(0.4, 6);
  });

  it('reports sizes', () => {
    const learner = new MlpSgdLearner({ layerSizes: [3, 4, 2], rng: createSeededRng(1) });
    expect([learner.inputSize, learner.outputSize, learner.stateSize]).toEqual([
      3,
      2,
      4 * 4 + 2 * 5,
    ]);
  });

  it('rejects a target of the wrong size', () => {
    const learner = new MlpSgdLearner({ layerSizes: [2, 1], rng: createSeededRng(1) });
    expect(() => learner.learn(new Float32Array(2), new Float32Array(2), 0.1)).toThrow(
      SizeMismatchError,
    );
  });

  it.each([
    ['layerSizes', { layerSizes: [2] }],
    ['maxWeight', { layerSizes: [2, 1], maxWeight: 0 }],
    ['forgetRatePerSecond', { layerSizes: [2, 1], forgetRatePerSecond: -1 }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new MlpSgdLearner({ rng: createSeededRng(1), ...options })).toThrow(
      InvalidParameterError,
    );
    expect(() => new MlpSgdLearner({ rng: createSeededRng(1), ...options })).toThrow(
      new RegExp(name),
    );
  });

  it('stays finite when huge inputs overflow Float32 in the forward pass (relu)', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [1, 3, 1],
      rng: createSeededRng(1),
      hiddenActivation: 'relu',
    });
    for (let i = 0; i < 10; i += 1) learner.learn(Float32Array.of(3e38), Float32Array.of(1), 0.5);
    expect(learner.serialize().every((value) => Number.isFinite(value))).toBe(true);
    expect(() => learner.restore(learner.serialize())).not.toThrow();
  });

  it('skips the update instead of corrupting weights, and still learns afterwards', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [1, 3, 1],
      rng: createSeededRng(1),
      hiddenActivation: 'relu',
    });
    // The first call is still finite (the output layer starts at zero) and pushes the output
    // weights to their bound; the second call then overflows in the forward pass and is skipped.
    learner.learn(Float32Array.of(3e38), Float32Array.of(1), 0.5);
    const before = Array.from(learner.serialize());
    learner.learn(Float32Array.of(3e38), Float32Array.of(1), 0.5);
    expect(Array.from(learner.serialize())).toEqual(before);
    for (let i = 0; i < 200; i += 1) learner.learn(Float32Array.of(0.5), Float32Array.of(1), 0.1);
    expect(predictOne(learner, [0.5])).toBeGreaterThan(0.5);
  });

  it('rejects restoring weights beyond maxWeight', () => {
    const learner = new MlpSgdLearner({
      layerSizes: [1, 1],
      rng: createSeededRng(1),
      maxWeight: 1,
    });
    expect(() => learner.restore(Float32Array.of(2, 0))).toThrow(InvalidParameterError);
    expect(() => learner.restore(Float32Array.of(0))).toThrow(SizeMismatchError);
  });

  it('rejects a bound larger than the Float32 range and impossible sizes', () => {
    expect(
      () => new MlpSgdLearner({ layerSizes: [1, 1], rng: createSeededRng(1), maxWeight: 1e39 }),
    ).toThrow(/maxWeight/);
    expect(() => new MlpSgdLearner({ layerSizes: [1e10, 1], rng: createSeededRng(1) })).toThrow(
      InvalidParameterError,
    );
  });
});
