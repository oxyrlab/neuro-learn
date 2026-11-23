/**
 * Long runs: hundreds of thousands of learn/decay steps must keep every learner finite,
 * bounded and deterministic.
 */
import { describe, expect, it } from 'vitest';
import {
  createSeededRng,
  EligibilityTraceLearner,
  HebbianLearner,
  MlpSgdLearner,
  RescorlaWagnerLearner,
} from '../../src';
import type { Learner } from '../../src';

const STEPS = 200_000;
const MAX_STRENGTH = 1.5;

const factories: Array<[string, () => Learner]> = [
  [
    'rescorla-wagner',
    () =>
      new RescorlaWagnerLearner({
        inputSize: 6,
        outputSize: 3,
        maxStrength: MAX_STRENGTH,
        forgetRatePerSecond: 0.01,
      }),
  ],
  [
    'eligibility-trace',
    () =>
      new EligibilityTraceLearner({
        inputSize: 6,
        outputSize: 3,
        traceTimeConstantSeconds: 2,
        maxStrength: MAX_STRENGTH,
        forgetRatePerSecond: 0.01,
      }),
  ],
  [
    'hebbian',
    () =>
      new HebbianLearner({
        inputSize: 6,
        outputSize: 3,
        maxWeight: MAX_STRENGTH,
        forgetRatePerSecond: 0.01,
      }),
  ],
  [
    'mlp-sgd',
    () =>
      new MlpSgdLearner({
        layerSizes: [6, 8, 3],
        rng: createSeededRng(4),
        maxWeight: MAX_STRENGTH,
        forgetRatePerSecond: 0.01,
      }),
  ],
];

function run(learner: Learner, seed: number): Float32Array {
  const rng = createSeededRng(seed);
  const features = new Float32Array(learner.inputSize);
  const target = new Float32Array(learner.outputSize);
  for (let step = 0; step < STEPS; step += 1) {
    for (let i = 0; i < features.length; i += 1) features[i] = rng.next();
    for (let i = 0; i < target.length; i += 1) target[i] = rng.next() * 2 - 1;
    learner.learn(features, target, rng.next());
    learner.decay(0.1);
  }
  return learner.serialize();
}

describe.each(factories)('%s over long runs', (_name, create) => {
  it('stays finite and within its weight bound', () => {
    const state = run(create(), 1);
    for (const value of state) {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value)).toBeLessThanOrEqual(MAX_STRENGTH + 1e-6);
    }
  });

  it('is deterministic across two identical runs', () => {
    expect(Array.from(run(create(), 2))).toEqual(Array.from(run(create(), 2)));
  });
});
