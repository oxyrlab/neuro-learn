import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { InvalidParameterError } from '../core/errors';
import { runLearnerContract } from '../testing/learner-contract';
import { HebbianLearner } from './hebbian';

runLearnerContract(
  'hebbian',
  () => new HebbianLearner({ inputSize: 4, outputSize: 2, forgetRatePerSecond: 0.1 }),
  { maxAbsState: 1 },
);

describe('HebbianLearner', () => {
  it('strengthens only co-active pairs', () => {
    const learner = new HebbianLearner({ inputSize: 2, outputSize: 1, weightDecay: 0 });
    learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.2);
    expect(learner.serialize()[0]).toBeCloseTo(0.2, 6);
    expect(learner.serialize()[1]).toBe(0);
  });

  it('converges to post·pre / weightDecay when below maxWeight', () => {
    const learner = new HebbianLearner({
      inputSize: 1,
      outputSize: 1,
      maxWeight: 10,
      weightDecay: 0.5,
    });
    for (let i = 0; i < 200; i += 1) learner.learn(Float32Array.of(1), Float32Array.of(1), 0.3);
    expect(learner.serialize()[0]).toBeCloseTo(2, 3);
  });

  it('is anti-Hebbian for negative post-synaptic activity', () => {
    const learner = new HebbianLearner({ inputSize: 1, outputSize: 1, weightDecay: 0 });
    learner.learn(Float32Array.of(1), Float32Array.of(-1), 0.2);
    expect(learner.serialize()[0]).toBeCloseTo(-0.2, 6);
  });

  it('clamps to maxWeight even without decay', () => {
    const learner = new HebbianLearner({
      inputSize: 1,
      outputSize: 1,
      maxWeight: 0.5,
      weightDecay: 0,
    });
    for (let i = 0; i < 50; i += 1) learner.learn(Float32Array.of(1), Float32Array.of(1), 1);
    expect(learner.serialize()[0]).toBe(0.5);
  });

  it('predicts the weighted sum of inputs', () => {
    const learner = new HebbianLearner({ inputSize: 2, outputSize: 1 });
    learner.restore(Float32Array.of(0.5, -0.25));
    const out = new Float32Array(1);
    learner.predict(Float32Array.of(1, 1), out);
    expect(out[0]).toBeCloseTo(0.25, 6);
  });

  it('forgets exponentially and not when the rate is zero', () => {
    const forgetting = new HebbianLearner({
      inputSize: 1,
      outputSize: 1,
      forgetRatePerSecond: 0.5,
    });
    forgetting.restore(Float32Array.of(1));
    forgetting.decay(2);
    expect(forgetting.serialize()[0]).toBeCloseTo(Math.exp(-1), 5);
    const steady = new HebbianLearner({ inputSize: 1, outputSize: 1 });
    steady.restore(Float32Array.of(0.8));
    steady.decay(10);
    expect(steady.serialize()[0]).toBeCloseTo(0.8, 6);
  });

  it('keeps weights finite and within maxWeight for any inputs (property)', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(
            fc.float({ min: 0, max: 1, noNaN: true }),
            fc.float({ min: -1, max: 1, noNaN: true }),
            fc.float({ min: 0, max: 1, noNaN: true }),
          ),
          { maxLength: 60 },
        ),
        (steps) => {
          const learner = new HebbianLearner({ inputSize: 1, outputSize: 1, maxWeight: 0.7 });
          for (const [pre, post, rate] of steps) {
            learner.learn(Float32Array.of(pre), Float32Array.of(post), rate);
          }
          const weight = learner.serialize()[0] as number;
          return Number.isFinite(weight) && Math.abs(weight) <= 0.7 + 1e-6;
        },
      ),
    );
  });

  it('reports sizes', () => {
    const learner = new HebbianLearner({ inputSize: 3, outputSize: 2 });
    expect([learner.inputSize, learner.outputSize, learner.stateSize]).toEqual([3, 2, 6]);
  });

  it.each([
    ['inputSize', { inputSize: -1, outputSize: 1 }],
    ['outputSize', { inputSize: 1, outputSize: 0 }],
    ['maxWeight', { inputSize: 1, outputSize: 1, maxWeight: 0 }],
    ['weightDecay', { inputSize: 1, outputSize: 1, weightDecay: 1.5 }],
    ['forgetRatePerSecond', { inputSize: 1, outputSize: 1, forgetRatePerSecond: -1 }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new HebbianLearner(options)).toThrow(InvalidParameterError);
    expect(() => new HebbianLearner(options)).toThrow(new RegExp(name));
  });

  it('rejects restoring weights beyond maxWeight', () => {
    const learner = new HebbianLearner({ inputSize: 1, outputSize: 1, maxWeight: 1 });
    expect(() => learner.restore(Float32Array.of(-5))).toThrow(InvalidParameterError);
  });

  it('rejects a bound larger than the Float32 range and impossible sizes', () => {
    expect(() => new HebbianLearner({ inputSize: 1, outputSize: 1, maxWeight: 1e39 })).toThrow(
      /maxWeight/,
    );
    expect(() => new HebbianLearner({ inputSize: 1e10, outputSize: 1 })).toThrow(
      InvalidParameterError,
    );
  });
});
