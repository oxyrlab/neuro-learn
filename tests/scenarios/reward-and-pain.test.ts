/**
 * Two outcomes from distinct cues: a creature learns to approach a rewarding cue and to flee
 * a painful one. Outputs: index 0 = "approach", index 1 = "flee". Valence is expressed by which
 * output the outcome targets; the learning rate (modulator strength) is always non-negative.
 */
import { describe, expect, it } from 'vitest';
import { EligibilityTraceLearner } from '../../src';
import { predictClamped, runEpisode } from '../helpers/episodes';

const SWEET = Float32Array.of(1, 0, 0, 0);
const BITTER = Float32Array.of(0, 1, 0, 0);
const REWARD = Float32Array.of(1, 0);
const PAIN = Float32Array.of(0, 1);

function createLearner(): EligibilityTraceLearner {
  return new EligibilityTraceLearner({
    inputSize: 4,
    outputSize: 2,
    traceTimeConstantSeconds: 2,
  });
}

describe('reward and pain scenario', () => {
  it('learns to approach the rewarding cue and flee the painful one', () => {
    const learner = createLearner();
    for (let episode = 0; episode < 6; episode += 1) {
      runEpisode(learner, SWEET, REWARD, 0.4);
      runEpisode(learner, BITTER, PAIN, 0.4);
    }
    expect(predictClamped(learner, SWEET, 0)).toBeGreaterThan(0.8);
    expect(predictClamped(learner, SWEET, 1)).toBeLessThan(0.05);
    expect(predictClamped(learner, BITTER, 1)).toBeGreaterThan(0.8);
    expect(predictClamped(learner, BITTER, 0)).toBeLessThan(0.05);
  });

  it('a stronger modulator (higher rate) learns faster than a weaker one', () => {
    const strong = createLearner();
    const weak = createLearner();
    runEpisode(strong, SWEET, REWARD, 0.6);
    runEpisode(weak, SWEET, REWARD, 0.1);
    expect(predictClamped(strong, SWEET, 0)).toBeGreaterThan(predictClamped(weak, SWEET, 0));
  });

  it('a zero rate learns nothing', () => {
    const learner = createLearner();
    runEpisode(learner, SWEET, REWARD, 0);
    expect(predictClamped(learner, SWEET, 0)).toBe(0);
  });
});
