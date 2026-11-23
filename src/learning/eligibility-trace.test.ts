import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from '../core/errors';
import { runLearnerContract } from '../testing/learner-contract';
import { EligibilityTraceLearner } from './eligibility-trace';

const base = { inputSize: 4, outputSize: 1, traceTimeConstantSeconds: 1 };

runLearnerContract(
  'eligibility-trace',
  () => new EligibilityTraceLearner({ ...base, outputSize: 2, forgetRatePerSecond: 0.1 }),
  { maxAbsState: 1 },
);

function strengthOfCue(learner: EligibilityTraceLearner, cue: number): number {
  return learner.serialize()[cue] as number;
}

describe('EligibilityTraceLearner', () => {
  it('observe records cues as a replacing trace (max)', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.observe(Float32Array.of(1, 0.5, 0, 0));
    learner.observe(Float32Array.of(0, 0.2, 0.7, 0));
    expect(Array.from(learner.serialize().subarray(4))).toEqual([1, 0.5, 0.7, 0].map(Math.fround));
  });

  it('trace decays with the time constant', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.observe(Float32Array.of(1, 0, 0, 0));
    learner.decay(1);
    expect(learner.serialize()[4]).toBeCloseTo(Math.exp(-1), 5);
  });

  it('credits a cue for an outcome that arrives later', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.observe(Float32Array.of(1, 0, 0, 0));
    for (let i = 0; i < 5; i += 1) learner.decay(0.2);
    learner.learn(new Float32Array(4), Float32Array.of(1), 0.5);
    expect(strengthOfCue(learner, 0)).toBeGreaterThan(0.1);
    expect(strengthOfCue(learner, 1)).toBe(0);
  });

  it('gives (almost) no credit after a very long delay', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.observe(Float32Array.of(1, 0, 0, 0));
    learner.decay(30);
    learner.learn(new Float32Array(4), Float32Array.of(1), 1);
    expect(strengthOfCue(learner, 0)).toBeLessThan(1e-6);
  });

  it('acts like the delta rule when the cue and outcome coincide', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.learn(Float32Array.of(1, 0, 0, 0), Float32Array.of(1), 0.5);
    expect(strengthOfCue(learner, 0)).toBeCloseTo(0.5, 6);
  });

  it('predict uses current cues and does not change state', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.learn(Float32Array.of(1, 0, 0, 0), Float32Array.of(1), 0.5);
    const before = Array.from(learner.serialize());
    const out = new Float32Array(1);
    learner.predict(Float32Array.of(1, 0, 0, 0), out);
    expect(out[0]).toBeCloseTo(0.5, 6);
    learner.predict(new Float32Array(4), out);
    expect(out[0]).toBe(0);
    expect(Array.from(learner.serialize())).toEqual(before);
  });

  it('forgets strengths with forgetRatePerSecond but not when it is zero', () => {
    const forgetting = new EligibilityTraceLearner({ ...base, forgetRatePerSecond: 0.5 });
    forgetting.restore(Float32Array.of(1, 0, 0, 0, 0, 0, 0, 0));
    forgetting.decay(2);
    expect(strengthOfCue(forgetting, 0)).toBeCloseTo(Math.exp(-1), 5);
    const steady = new EligibilityTraceLearner(base);
    steady.restore(Float32Array.of(0.8, 0, 0, 0, 0, 0, 0, 0));
    steady.decay(10);
    expect(strengthOfCue(steady, 0)).toBeCloseTo(0.8, 6);
  });

  it('reports sizes (strengths plus trace)', () => {
    const learner = new EligibilityTraceLearner({ ...base, outputSize: 2 });
    expect(learner.stateSize).toBe(4 * 2 + 4);
  });

  it('observe validates its input', () => {
    const learner = new EligibilityTraceLearner(base);
    expect(() => learner.observe(new Float32Array(3))).toThrow(SizeMismatchError);
    expect(() => learner.observe(new Float32Array(4).fill(NaN))).toThrow(InvalidParameterError);
  });

  it('rejects negative cue strengths instead of silently ignoring them', () => {
    const learner = new EligibilityTraceLearner(base);
    expect(() => learner.observe(Float32Array.of(0, -0.1, 0, 0))).toThrow(/features\[1\]/);
    expect(() => learner.learn(Float32Array.of(0, -0.1, 0, 0), Float32Array.of(1), 0.5)).toThrow(
      InvalidParameterError,
    );
  });

  it('a rejected learn leaves the state untouched', () => {
    const learner = new EligibilityTraceLearner(base);
    learner.observe(Float32Array.of(1, 0, 0, 0));
    const before = Array.from(learner.serialize());
    expect(() => learner.learn(Float32Array.of(0, -1, 0, 0), Float32Array.of(1), 0.5)).toThrow();
    expect(Array.from(learner.serialize())).toEqual(before);
  });

  it.each([
    ['inputSize', { ...base, inputSize: 0 }],
    ['outputSize', { ...base, outputSize: 0 }],
    ['traceTimeConstantSeconds', { ...base, traceTimeConstantSeconds: 0 }],
    ['maxStrength', { ...base, maxStrength: -1 }],
    ['forgetRatePerSecond', { ...base, forgetRatePerSecond: -1 }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new EligibilityTraceLearner(options)).toThrow(new RegExp(name));
  });

  it('rejects restoring a negative trace or strengths beyond maxStrength', () => {
    const learner = new EligibilityTraceLearner({ ...base, inputSize: 1 });
    expect(() => learner.restore(Float32Array.of(0, -1))).toThrow(/state\.trace\[0\]/);
    expect(() => learner.restore(Float32Array.of(100, 0))).toThrow(/state\[0\]/);
    expect(() => learner.restore(Float32Array.of(0.5, 3))).not.toThrow(); // a trace may exceed maxStrength
  });

  it('rejects a bound larger than the Float32 range and impossible sizes', () => {
    expect(() => new EligibilityTraceLearner({ ...base, maxStrength: 1e39 })).toThrow(
      /maxStrength/,
    );
    expect(() => new EligibilityTraceLearner({ ...base, inputSize: 1e10 })).toThrow(
      InvalidParameterError,
    );
  });
});
