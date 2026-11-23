import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from '../core/errors';
import { createSeededRng } from '../core/rng';
import type { Learner } from '../learning/learner';

/** Options for {@link runLearnerContract}. */
export interface LearnerContractOptions {
  /** If set, every state value must stay within `[-maxAbsState, maxAbsState]` during training. */
  readonly maxAbsState?: number;
  /** Seed for the generated training stimuli. Default 1234. */
  readonly seed?: number;
  /** Number of training steps per test. Default 200. */
  readonly steps?: number;
}

const DEFAULT_SEED = 1234;
const DEFAULT_STEPS = 200;
const TRAINING_RATE = 0.1;
const TRAINING_DT_SECONDS = 0.1;

function fillRandom(buffer: Float32Array, next: () => number): void {
  for (let index = 0; index < buffer.length; index += 1) buffer[index] = next();
}

function train(learner: Learner, seed: number, steps: number): void {
  const rng = createSeededRng(seed);
  const features = new Float32Array(learner.inputSize);
  const target = new Float32Array(learner.outputSize);
  for (let step = 0; step < steps; step += 1) {
    fillRandom(features, () => rng.next());
    fillRandom(target, () => rng.next());
    learner.learn(features, target, TRAINING_RATE);
    learner.decay(TRAINING_DT_SECONDS);
  }
}

function sumOfMagnitudes(buffer: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < buffer.length; index += 1) sum += Math.abs(buffer[index] as number);
  return sum;
}

function predictProbe(learner: Learner, seed: number): Float32Array {
  const rng = createSeededRng(seed);
  const features = new Float32Array(learner.inputSize);
  fillRandom(features, () => rng.next());
  const out = new Float32Array(learner.outputSize);
  learner.predict(features, out);
  return out;
}

/**
 * Registers the shared Vitest contract every {@link Learner} must satisfy: fixed state size,
 * empty start, finite and bounded state, determinism, serialize/restore round-trip, independent
 * serialized copies, input immutability, monotone forgetting, and descriptive input validation.
 *
 * The factory must return a fresh learner each call, configured with a **non-zero** forgetting
 * rate (so the strict-forgetting check is meaningful).
 *
 * @param name - Label used in the suite title.
 * @param create - Factory returning a new learner with empty memory.
 * @param options - Optional bounds and training settings.
 *
 * @example
 * runLearnerContract('hebbian', () => new HebbianLearner({ inputSize: 4, outputSize: 2, forgetRatePerSecond: 0.1 }));
 */
export function runLearnerContract(
  name: string,
  create: () => Learner,
  options: LearnerContractOptions = {},
): void {
  const seed = options.seed ?? DEFAULT_SEED;
  const steps = options.steps ?? DEFAULT_STEPS;

  describe(`Learner contract: ${name}`, () => {
    it('has positive sizes and a constant state length', () => {
      const learner = create();
      expect(learner.inputSize).toBeGreaterThan(0);
      expect(learner.outputSize).toBeGreaterThan(0);
      expect(learner.serialize().length).toBe(learner.stateSize);
      train(learner, seed, steps);
      expect(learner.serialize().length).toBe(learner.stateSize);
    });

    it('starts with empty (all-zero) memory', () => {
      expect(sumOfMagnitudes(create().serialize())).toBe(0);
    });

    it('keeps state and predictions finite and bounded under training', () => {
      const learner = create();
      train(learner, seed, steps);
      const state = learner.serialize();
      for (const value of state) {
        expect(Number.isFinite(value)).toBe(true);
        if (options.maxAbsState !== undefined) {
          expect(Math.abs(value)).toBeLessThanOrEqual(options.maxAbsState + 1e-6);
        }
      }
      for (const value of predictProbe(learner, seed + 1)) {
        expect(Number.isFinite(value)).toBe(true);
      }
    });

    it('is deterministic: same operations give identical state', () => {
      const first = create();
      const second = create();
      train(first, seed, steps);
      train(second, seed, steps);
      expect(Array.from(first.serialize())).toEqual(Array.from(second.serialize()));
    });

    it('round-trips through serialize and restore', () => {
      const trained = create();
      train(trained, seed, steps);
      const restored = create();
      restored.restore(trained.serialize());
      expect(Array.from(restored.serialize())).toEqual(Array.from(trained.serialize()));
      expect(Array.from(predictProbe(restored, 99))).toEqual(Array.from(predictProbe(trained, 99)));
    });

    it('returns an independent copy from serialize', () => {
      const learner = create();
      train(learner, seed, steps);
      const snapshot = learner.serialize();
      const expected = Array.from(snapshot);
      snapshot.fill(7);
      expect(Array.from(learner.serialize())).toEqual(expected);
    });

    it('does not modify the caller buffers', () => {
      const learner = create();
      const features = new Float32Array(learner.inputSize).fill(0.5);
      const target = new Float32Array(learner.outputSize).fill(0.25);
      learner.learn(features, target, TRAINING_RATE);
      learner.predict(features, new Float32Array(learner.outputSize));
      expect(Array.from(features)).toEqual(new Array(learner.inputSize).fill(0.5));
      expect(Array.from(target)).toEqual(new Array(learner.outputSize).fill(0.25));
    });

    it('forgets: decay never increases and strictly reduces non-zero state', () => {
      const learner = create();
      train(learner, seed, steps);
      const before = sumOfMagnitudes(learner.serialize());
      learner.decay(0);
      expect(sumOfMagnitudes(learner.serialize())).toBe(before);
      learner.decay(1);
      const after = sumOfMagnitudes(learner.serialize());
      expect(after).toBeLessThan(before);
      expect(after).toBeGreaterThanOrEqual(0);
    });

    it('rejects buffers of the wrong size with SizeMismatchError', () => {
      const learner = create();
      const features = new Float32Array(learner.inputSize);
      const target = new Float32Array(learner.outputSize);
      const out = new Float32Array(learner.outputSize);
      const longFeatures = new Float32Array(learner.inputSize + 1);
      const longTarget = new Float32Array(learner.outputSize + 1);
      expect(() => learner.predict(longFeatures, out)).toThrow(SizeMismatchError);
      expect(() => learner.predict(features, longTarget)).toThrow(SizeMismatchError);
      expect(() => learner.learn(longFeatures, target, 0.1)).toThrow(SizeMismatchError);
      expect(() => learner.learn(features, longTarget, 0.1)).toThrow(SizeMismatchError);
      expect(() => learner.restore(new Float32Array(learner.stateSize + 1))).toThrow(
        SizeMismatchError,
      );
    });

    it('rejects invalid rates, time steps and non-finite values with InvalidParameterError', () => {
      const learner = create();
      const features = new Float32Array(learner.inputSize);
      const target = new Float32Array(learner.outputSize);
      for (const rate of [NaN, -0.1, 1.1, Infinity]) {
        expect(() => learner.learn(features, target, rate)).toThrow(InvalidParameterError);
      }
      for (const dt of [NaN, -1, Infinity]) {
        expect(() => learner.decay(dt)).toThrow(InvalidParameterError);
      }
      const badFeatures = new Float32Array(learner.inputSize).fill(NaN);
      expect(() => learner.learn(badFeatures, target, 0.1)).toThrow(InvalidParameterError);
      expect(() => learner.predict(badFeatures, target)).toThrow(InvalidParameterError);
      const badTarget = new Float32Array(learner.outputSize).fill(NaN);
      expect(() => learner.learn(features, badTarget, 0.1)).toThrow(InvalidParameterError);
      const badState = new Float32Array(learner.stateSize).fill(NaN);
      expect(() => learner.restore(badState)).toThrow(InvalidParameterError);
    });
  });
}
