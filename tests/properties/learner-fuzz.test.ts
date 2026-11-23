/**
 * Model-based fuzzing of every learner: random sequences of operations with arbitrary finite
 * inputs (including values near the Float32 limit). Invariants after each operation:
 * the state is finite and within the weight bound, and a learner never throws anything other than
 * `InvalidParameterError` / `SizeMismatchError` (documented rejections).
 */
import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  createSeededRng,
  EligibilityTraceLearner,
  HebbianLearner,
  InvalidParameterError,
  MlpSgdLearner,
  RescorlaWagnerLearner,
  SizeMismatchError,
} from '../../src';
import type { Learner } from '../../src';

const INPUT_SIZE = 3;
const OUTPUT_SIZE = 2;
const BOUND = 2;

const smallValue = fc.float({ min: -2, max: 2, noNaN: true, noDefaultInfinity: true });
const extremeValue = fc.double({ min: -3e38, max: 3e38, noNaN: true, noDefaultInfinity: true });
const anyValue = fc.oneof(
  { weight: 4, arbitrary: smallValue },
  { weight: 1, arbitrary: extremeValue },
);
const vector = (size: number) => fc.array(anyValue, { minLength: size, maxLength: size });

type Operation =
  | { kind: 'learn'; features: number[]; target: number[]; rate: number }
  | { kind: 'decay'; dt: number }
  | { kind: 'observe'; features: number[] }
  | { kind: 'snapshot' };

const operation: fc.Arbitrary<Operation> = fc.oneof(
  fc.record({
    kind: fc.constant('learn' as const),
    features: vector(INPUT_SIZE),
    target: vector(OUTPUT_SIZE),
    rate: fc.float({ min: 0, max: 1, noNaN: true }),
  }),
  fc.record({
    kind: fc.constant('decay' as const),
    dt: fc.float({ min: 0, max: 1000, noNaN: true }),
  }),
  fc.record({ kind: fc.constant('observe' as const), features: vector(INPUT_SIZE) }),
  fc.constant({ kind: 'snapshot' as const }),
);

const learners: Array<[string, () => Learner & { observe?: (f: Float32Array) => void }]> = [
  [
    'rescorla-wagner',
    () =>
      new RescorlaWagnerLearner({
        inputSize: INPUT_SIZE,
        outputSize: OUTPUT_SIZE,
        maxStrength: BOUND,
        forgetRatePerSecond: 0.1,
      }),
  ],
  [
    'eligibility-trace',
    () =>
      new EligibilityTraceLearner({
        inputSize: INPUT_SIZE,
        outputSize: OUTPUT_SIZE,
        traceTimeConstantSeconds: 1,
        maxStrength: BOUND,
        forgetRatePerSecond: 0.1,
      }),
  ],
  [
    'hebbian',
    () =>
      new HebbianLearner({
        inputSize: INPUT_SIZE,
        outputSize: OUTPUT_SIZE,
        maxWeight: BOUND,
        forgetRatePerSecond: 0.1,
      }),
  ],
  [
    'mlp-sgd (relu)',
    () =>
      new MlpSgdLearner({
        layerSizes: [INPUT_SIZE, 4, OUTPUT_SIZE],
        rng: createSeededRng(2),
        hiddenActivation: 'relu',
        maxWeight: BOUND,
        forgetRatePerSecond: 0.1,
      }),
  ],
  [
    'mlp-sgd',
    () =>
      new MlpSgdLearner({
        layerSizes: [INPUT_SIZE, 4, OUTPUT_SIZE],
        rng: createSeededRng(1),
        maxWeight: BOUND,
        forgetRatePerSecond: 0.1,
      }),
  ],
];

function apply(learner: ReturnType<(typeof learners)[number][1]>, op: Operation): void {
  try {
    if (op.kind === 'learn') {
      learner.learn(Float32Array.from(op.features), Float32Array.from(op.target), op.rate);
    } else if (op.kind === 'decay') {
      learner.decay(op.dt);
    } else if (op.kind === 'observe') {
      learner.observe?.(Float32Array.from(op.features));
    } else {
      learner.restore(learner.serialize());
    }
  } catch (error) {
    if (!(error instanceof InvalidParameterError || error instanceof SizeMismatchError))
      throw error;
  }
}

/**
 * Weights must be finite and within BOUND. The eligibility-trace state ends with the cue trace
 * (INPUT_SIZE values), which only has to be finite and non-negative: it mirrors the caller's cues.
 */
function expectHealthy(name: string, learner: Learner): void {
  const state = learner.serialize();
  const weightCount = name === 'eligibility-trace' ? state.length - INPUT_SIZE : state.length;
  state.forEach((value, index) => {
    expect(Number.isFinite(value)).toBe(true);
    if (index < weightCount) expect(Math.abs(value)).toBeLessThanOrEqual(BOUND + 1e-6);
    else expect(value).toBeGreaterThanOrEqual(0);
  });
}

describe.each(learners)('%s fuzzing', (name, create) => {
  it('keeps state finite and bounded under arbitrary operation sequences', () => {
    fc.assert(
      fc.property(fc.array(operation, { maxLength: 40 }), (operations) => {
        const learner = create();
        for (const op of operations) {
          apply(learner, op);
          expectHealthy(name, learner);
        }
      }),
      { numRuns: 300 },
    );
  });

  it('is deterministic: two copies given the same operations end in the same state', () => {
    fc.assert(
      fc.property(fc.array(operation, { maxLength: 30 }), (operations) => {
        const first = create();
        const second = create();
        for (const op of operations) {
          apply(first, op);
          apply(second, op);
        }
        expect(Array.from(first.serialize())).toEqual(Array.from(second.serialize()));
      }),
      { numRuns: 150 },
    );
  });
});
