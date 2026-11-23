/**
 * Fear conditioning as EcoSim's `threat-memory` module needs it (EcoSim spec 9.4):
 * conditioning after 3 episodes, extinction after 20 sightings, generalization to similar
 * appearance, and no inheritance of learned state. Appearance features are non-negative
 * [0, 1] values (size, hue, lightness, shape, part count, ...).
 */
import { describe, expect, it } from 'vitest';
import { EligibilityTraceLearner } from '../../src';
import { predictClamped, runEpisode } from '../helpers/episodes';

const APPEARANCE_X = Float32Array.of(0.9, 0.1, 0.8, 0.1, 0.9, 0.1);
const SIMILAR_TO_X = Float32Array.of(0.7, 0.3, 0.6, 0.3, 0.7, 0.3);
const DIFFERENT_FROM_X = Float32Array.of(0.1, 0.9, 0.1, 0.9, 0.1, 0.9);

/** Learning rates the host would use: strong on pain, gentle on "seen without pain". */
const PAIN_RATE = 0.4;
const SAFE_RATE = 0.1;
const DANGER = Float32Array.of(1);
const SAFE = Float32Array.of(0);

function createLearner(): EligibilityTraceLearner {
  return new EligibilityTraceLearner({
    inputSize: 6,
    outputSize: 1,
    traceTimeConstantSeconds: 2,
    forgetRatePerSecond: 0.001,
  });
}

describe('threat-memory scenario (eligibility-trace)', () => {
  it('knows nothing before any pain', () => {
    expect(predictClamped(createLearner(), APPEARANCE_X)).toBe(0);
  });

  it('after 3 "seen X, then pain" episodes predictedDanger > 0.6', () => {
    const learner = createLearner();
    for (let episode = 0; episode < 3; episode += 1) {
      runEpisode(learner, APPEARANCE_X, DANGER, PAIN_RATE);
    }
    expect(predictClamped(learner, APPEARANCE_X)).toBeGreaterThan(0.6);
  });

  it('after 20 sightings without pain predictedDanger < 0.2 (extinction)', () => {
    const learner = createLearner();
    for (let episode = 0; episode < 3; episode += 1) {
      runEpisode(learner, APPEARANCE_X, DANGER, PAIN_RATE);
    }
    for (let sighting = 0; sighting < 20; sighting += 1) {
      runEpisode(learner, APPEARANCE_X, SAFE, SAFE_RATE);
    }
    expect(predictClamped(learner, APPEARANCE_X)).toBeLessThan(0.2);
  });

  it('responds partially to a similar-looking creature and barely to a different one', () => {
    const learner = createLearner();
    for (let episode = 0; episode < 3; episode += 1) {
      runEpisode(learner, APPEARANCE_X, DANGER, PAIN_RATE);
    }
    const toX = predictClamped(learner, APPEARANCE_X);
    const toSimilar = predictClamped(learner, SIMILAR_TO_X);
    const toDifferent = predictClamped(learner, DIFFERENT_FROM_X);
    expect(toSimilar).toBeGreaterThan(0.5 * toX);
    expect(toSimilar).toBeLessThan(toX);
    expect(toDifferent).toBeLessThan(0.35 * toX);
  });

  it('gives no credit for an outcome that arrives long after the sighting', () => {
    const learner = createLearner();
    for (let tick = 0; tick < 10; tick += 1) {
      learner.decay(0.1);
      learner.observe(APPEARANCE_X);
    }
    learner.decay(30);
    learner.learn(new Float32Array(6), DANGER, PAIN_RATE);
    expect(predictClamped(learner, APPEARANCE_X)).toBeLessThan(0.01);
  });

  it('does not pass learned fear to offspring (a new instance starts empty)', () => {
    const parent = createLearner();
    for (let episode = 0; episode < 3; episode += 1) {
      runEpisode(parent, APPEARANCE_X, DANGER, PAIN_RATE);
    }
    const offspring = createLearner();
    expect(predictClamped(parent, APPEARANCE_X)).toBeGreaterThan(0.6);
    expect(predictClamped(offspring, APPEARANCE_X)).toBe(0);
    expect(offspring.serialize().every((v) => v === 0)).toBe(true);
  });
});
