/**
 * Accuracy ("performance" in the ML sense): does an implementation compute the right numbers, and
 * does it learn the same thing? Everything is judged against the float64 reference, never against
 * the other library, and held-out loss is always evaluated with the same reference forward pass.
 */
import type { Dataset } from './dataset';
import { deltaStep, mlpOutputs, mlpStep, rnnStep } from './reference';
import type { Scenario, Subject } from './spec';
import { BLOCKS } from './spec';

export interface Checkpoint {
  readonly step: number;
  readonly loss: number;
  readonly referenceLoss: number;
}

export interface AccuracyResult {
  /** Forward kinds: max |output - float64 reference| over every output of the checked calls. */
  readonly maxAbsError?: number;
  /** Training kinds. */
  readonly checkpoints?: readonly Checkpoint[];
  readonly maxWeightDifference?: number;
  readonly finite: boolean;
}

const TRAIN_STEPS = 2000;
const CHECKPOINTS = [0, 50, 200, 1000, 2000];
const RNN_STEPS = 300;

function maxAbsDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let max = 0;
  for (let i = 0; i < a.length; i += 1)
    max = Math.max(max, Math.abs((a[i] as number) - (b[i] as number)));
  return max;
}

function isFinite32(values: ArrayLike<number>): boolean {
  for (let i = 0; i < values.length; i += 1) if (!Number.isFinite(values[i])) return false;
  return true;
}

/** Mean over the held-out set of 0.5 * sum of squared errors, always via the float64 reference. */
function heldOutLoss(sc: Scenario, data: Dataset, weights: ArrayLike<number>): number {
  let total = 0;
  data.heldInputs.forEach((x, k) => {
    const t = data.heldTargets[k] as Float32Array;
    const y =
      sc.kind === 'mlp-train'
        ? (mlpOutputs(sc.layers, weights, x).at(-1) as Float64Array)
        : (mlpOutputs(
            [sc.layers[0] as number, sc.layers[1] as number],
            withZeroBias(weights, sc),
            x,
          ).at(-1) as Float64Array);
    for (let i = 0; i < y.length; i += 1) total += 0.5 * ((y[i] as number) - (t[i] as number)) ** 2;
  });
  return total / data.heldInputs.length;
}

/** The delta rule has no biases; append zero biases so the MLP evaluator (one linear layer) applies. */
function withZeroBias(weights: ArrayLike<number>, sc: Scenario): Float64Array {
  const result = new Float64Array((weights.length as number) + (sc.layers[1] as number));
  for (let i = 0; i < weights.length; i += 1) result[i] = weights[i] as number;
  return result;
}

function checkForward(
  sc: Scenario,
  subject: Subject,
  data: Dataset,
  init: Float32Array,
): AccuracyResult {
  const inputSize = sc.layers[0] as number;
  const outputSize = sc.layers[sc.layers.length - 1] as number;
  let maxAbsError = 0;
  let finite = true;
  if (sc.kind === 'rnn-forward') {
    const hidden = new Float64Array(sc.layers[1] as number);
    for (let step = 0; step < RNN_STEPS; step += 1) {
      subject.run(step);
      const expected = rnnStep(sc.layers, init, hidden, data.inputs[step & 255] as Float32Array);
      maxAbsError = Math.max(maxAbsError, maxAbsDifference(subject.output, expected));
      finite &&= isFinite32(subject.output);
    }
    return { maxAbsError, finite };
  }
  const stateSize = sc.kind === 'population-forward' ? init.length / sc.batch : 0;
  for (let block = 0; block < BLOCKS; block += 1) {
    subject.run(block);
    const flat = data.blocks[block] as Float32Array;
    for (let k = 0; k < sc.batch; k += 1) {
      const weights =
        sc.kind === 'population-forward' ? init.subarray(k * stateSize, (k + 1) * stateSize) : init;
      const expected = mlpOutputs(
        sc.layers,
        weights,
        flat.subarray(k * inputSize, (k + 1) * inputSize),
      ).at(-1) as Float64Array;
      maxAbsError = Math.max(
        maxAbsError,
        maxAbsDifference(subject.output.subarray(k * outputSize, (k + 1) * outputSize), expected),
      );
    }
    finite &&= isFinite32(subject.output);
  }
  return { maxAbsError, finite };
}

function checkTraining(
  sc: Scenario,
  subject: Subject,
  data: Dataset,
  init: Float32Array,
): AccuracyResult {
  const reference = Float64Array.from(init);
  const checkpoints: Checkpoint[] = [];
  const record = (step: number): void => {
    checkpoints.push({
      step,
      loss: heldOutLoss(sc, data, subject.weights()),
      referenceLoss: heldOutLoss(sc, data, reference),
    });
  };
  record(0);
  for (let step = 0; step < TRAIN_STEPS; step += 1) {
    const j = step & 255;
    subject.run(step);
    const x = data.inputs[j] as Float32Array;
    const t = data.targets[j] as Float32Array;
    if (sc.kind === 'mlp-train') mlpStep(sc.layers, reference, x, t, sc.rate);
    else deltaStep(reference, sc.layers[0] as number, sc.layers[1] as number, x, t, sc.rate);
    if (CHECKPOINTS.includes(step + 1)) record(step + 1);
  }
  const finalWeights = subject.weights();
  return {
    checkpoints,
    maxWeightDifference: maxAbsDifference(finalWeights, reference),
    finite: isFinite32(finalWeights),
  };
}

export function measureAccuracy(
  sc: Scenario,
  subject: Subject,
  data: Dataset,
  init: Float32Array,
): AccuracyResult {
  return sc.kind === 'mlp-train' || sc.kind === 'delta-rule'
    ? checkTraining(sc, subject, data, init)
    : checkForward(sc, subject, data, init);
}
