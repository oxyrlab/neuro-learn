import { describe, expect, it } from 'vitest';
import { InvalidParameterError } from '../core/errors';
import { runLearnerContract } from '../testing/learner-contract';
import { RescorlaWagnerLearner } from './rescorla-wagner';

runLearnerContract(
  'rescorla-wagner',
  () => new RescorlaWagnerLearner({ inputSize: 4, outputSize: 2, forgetRatePerSecond: 0.1 }),
  { maxAbsState: 1 },
);

function predictOne(learner: RescorlaWagnerLearner, features: number[]): number {
  const out = new Float32Array(1);
  learner.predict(Float32Array.from(features), out);
  return out[0] as number;
}

describe('RescorlaWagnerLearner', () => {
  it('converges toward the outcome strength', () => {
    const learner = new RescorlaWagnerLearner({ inputSize: 4, outputSize: 1 });
    for (let i = 0; i < 100; i += 1) {
      learner.learn(Float32Array.of(1, 0, 0, 0), Float32Array.of(1), 0.2);
    }
    expect(predictOne(learner, [1, 0, 0, 0])).toBeCloseTo(1, 2);
    expect(predictOne(learner, [0, 1, 0, 0])).toBe(0);
  });

  it('extinguishes when the cue appears without the outcome', () => {
    const learner = new RescorlaWagnerLearner({ inputSize: 2, outputSize: 1 });
    for (let i = 0; i < 50; i += 1) learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.3);
    for (let i = 0; i < 50; i += 1) learner.learn(Float32Array.of(1, 0), Float32Array.of(0), 0.3);
    expect(predictOne(learner, [1, 0])).toBeLessThan(0.05);
  });

  it('shows blocking: a pre-trained cue prevents learning about a new cue', () => {
    const blocked = new RescorlaWagnerLearner({ inputSize: 2, outputSize: 1 });
    for (let i = 0; i < 60; i += 1) blocked.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.3);
    for (let i = 0; i < 60; i += 1) blocked.learn(Float32Array.of(1, 1), Float32Array.of(1), 0.3);
    const control = new RescorlaWagnerLearner({ inputSize: 2, outputSize: 1 });
    for (let i = 0; i < 60; i += 1) control.learn(Float32Array.of(1, 1), Float32Array.of(1), 0.3);
    expect(predictOne(blocked, [0, 1])).toBeLessThan(0.05);
    expect(predictOne(control, [0, 1])).toBeCloseTo(0.5, 1);
  });

  it('clamps strengths to maxStrength', () => {
    const learner = new RescorlaWagnerLearner({ inputSize: 1, outputSize: 1, maxStrength: 0.5 });
    for (let i = 0; i < 20; i += 1) learner.learn(Float32Array.of(1), Float32Array.of(10), 1);
    expect(predictOne(learner, [1])).toBe(0.5);
  });

  it('forgets exponentially with forgetRatePerSecond', () => {
    const learner = new RescorlaWagnerLearner({
      inputSize: 1,
      outputSize: 1,
      forgetRatePerSecond: 0.5,
    });
    learner.restore(Float32Array.of(1));
    learner.decay(2);
    expect(learner.serialize()[0]).toBeCloseTo(Math.exp(-1), 5);
  });

  it('does not forget when forgetRatePerSecond is zero', () => {
    const learner = new RescorlaWagnerLearner({ inputSize: 1, outputSize: 1 });
    learner.restore(Float32Array.of(0.8));
    learner.decay(100);
    expect(learner.serialize()[0]).toBeCloseTo(0.8, 6);
  });

  it('reports sizes', () => {
    const learner = new RescorlaWagnerLearner({ inputSize: 3, outputSize: 2 });
    expect([learner.inputSize, learner.outputSize, learner.stateSize]).toEqual([3, 2, 6]);
  });

  it.each([
    ['inputSize', { inputSize: 0, outputSize: 1 }],
    ['outputSize', { inputSize: 1, outputSize: 1.5 }],
    ['maxStrength', { inputSize: 1, outputSize: 1, maxStrength: 0 }],
    ['forgetRatePerSecond', { inputSize: 1, outputSize: 1, forgetRatePerSecond: -1 }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new RescorlaWagnerLearner(options)).toThrow(InvalidParameterError);
    expect(() => new RescorlaWagnerLearner(options)).toThrow(new RegExp(name));
  });
});
