import type { EligibilityTraceLearner } from '../../src';

/** Brain tick length used by the scenarios: 10 Hz, the EcoSim default. */
export const TICK_SECONDS = 0.1;
/** Ticks a creature keeps seeing something before an outcome is evaluated. */
export const SIGHTING_TICKS = 10;
/** Ticks of quiet time between two episodes. */
export const GAP_TICKS = 20;

/**
 * Runs one episode: the creature sees `features` for `SIGHTING_TICKS` ticks, then the outcome
 * `target` arrives with learning rate `rate`, then `GAP_TICKS` quiet ticks pass.
 * Per tick the order is `decay`, `observe` (as documented for eligibility-trace).
 */
export function runEpisode(
  learner: EligibilityTraceLearner,
  features: Float32Array,
  target: Float32Array,
  rate: number,
): void {
  for (let tick = 0; tick < SIGHTING_TICKS; tick += 1) {
    learner.decay(TICK_SECONDS);
    learner.observe(features);
  }
  learner.learn(new Float32Array(learner.inputSize), target, rate);
  for (let tick = 0; tick < GAP_TICKS; tick += 1) learner.decay(TICK_SECONDS);
}

/** Current prediction for `features`, clamped to [0, 1] like a `predictedDanger` port. */
export function predictClamped(
  learner: { predict(f: Float32Array, o: Float32Array): void; outputSize: number },
  features: Float32Array,
  outputIndex = 0,
): number {
  const out = new Float32Array(learner.outputSize);
  learner.predict(features, out);
  return Math.min(1, Math.max(0, out[outputIndex] as number));
}
