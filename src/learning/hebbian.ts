import {
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertPositive,
  assertPositiveInteger,
  assertSize,
} from '../core/validation';
import type { Learner } from './learner';
import { multiplyMatrixVector, scaleInPlace } from './linear-math';

/** Default maximum absolute weight. */
const DEFAULT_MAX_WEIGHT = 1;
/** Default per-update weight decay factor. */
const DEFAULT_WEIGHT_DECAY = 0.1;
/** Default forgetting rate: no forgetting. */
const DEFAULT_FORGET_RATE_PER_SECOND = 0;

/** Construction options for {@link HebbianLearner}. */
export interface HebbianOptions {
  /** Number of pre-synaptic inputs, integer >= 1. */
  readonly inputSize: number;
  /** Number of post-synaptic outputs, integer >= 1. */
  readonly outputSize: number;
  /** Maximum absolute weight, > 0. Default 1. */
  readonly maxWeight?: number;
  /** Per-update decay `d` in `ΔW = rate · (post·pre − d·W)`, in [0, 1]. Default 0.1. */
  readonly weightDecay?: number;
  /** Exponential forgetting rate in 1/s, >= 0. Default 0. */
  readonly forgetRatePerSecond?: number;
}

/**
 * Hebbian association ("cells that fire together wire together") with weight decay:
 * `ΔW[j][i] = rate · (post[j] · pre[i] − weightDecay · W[j][i])`, clamped to `±maxWeight`.
 * The caller supplies the post-synaptic activity as `target`. Without decay a Hebbian weight
 * grows without bound; the decay term pulls it toward `post·pre / weightDecay`.
 *
 * State: weights `W`, row-major, length `outputSize * inputSize`.
 *
 * @example
 * const learner = new HebbianLearner({ inputSize: 2, outputSize: 1 });
 * learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.2); // inputs and output co-active
 */
export class HebbianLearner implements Learner {
  public readonly inputSize: number;
  public readonly outputSize: number;
  public readonly stateSize: number;
  private readonly weights: Float32Array;
  private readonly maxWeight: number;
  private readonly weightDecay: number;
  private readonly forgetRatePerSecond: number;

  /**
   * @param options - See {@link HebbianOptions}.
   * @throws InvalidParameterError if any option is outside its documented range.
   */
  public constructor(options: HebbianOptions) {
    assertPositiveInteger('inputSize', options.inputSize);
    assertPositiveInteger('outputSize', options.outputSize);
    this.maxWeight = options.maxWeight ?? DEFAULT_MAX_WEIGHT;
    this.weightDecay = options.weightDecay ?? DEFAULT_WEIGHT_DECAY;
    this.forgetRatePerSecond = options.forgetRatePerSecond ?? DEFAULT_FORGET_RATE_PER_SECOND;
    assertPositive('maxWeight', this.maxWeight);
    assertFiniteInRange('weightDecay', this.weightDecay, 0, 1);
    assertNonNegative('forgetRatePerSecond', this.forgetRatePerSecond);
    this.inputSize = options.inputSize;
    this.outputSize = options.outputSize;
    this.stateSize = options.inputSize * options.outputSize;
    this.weights = new Float32Array(this.stateSize);
  }

  /** @inheritdoc */
  public predict(features: Float32Array, out: Float32Array): void {
    assertSize('features', features, this.inputSize);
    assertSize('out', out, this.outputSize);
    assertFiniteBuffer('features', features);
    multiplyMatrixVector(this.weights, this.outputSize, this.inputSize, features, out);
  }

  /** @inheritdoc */
  public learn(features: Float32Array, target: Float32Array, rate: number): void {
    assertSize('features', features, this.inputSize);
    assertSize('target', target, this.outputSize);
    assertFiniteBuffer('features', features);
    assertFiniteBuffer('target', target);
    assertFiniteInRange('rate', rate, 0, 1);
    for (let row = 0; row < this.outputSize; row += 1) {
      const post = target[row] as number;
      const base = row * this.inputSize;
      for (let column = 0; column < this.inputSize; column += 1) {
        const weight = this.weights[base + column] as number;
        const pre = features[column] as number;
        const next = weight + rate * (post * pre - this.weightDecay * weight);
        if (next > this.maxWeight) this.weights[base + column] = this.maxWeight;
        else if (next < -this.maxWeight) this.weights[base + column] = -this.maxWeight;
        else this.weights[base + column] = next;
      }
    }
  }

  /** @inheritdoc */
  public decay(dt: number): void {
    assertNonNegative('dt', dt);
    if (this.forgetRatePerSecond === 0) return;
    scaleInPlace(this.weights, Math.exp(-this.forgetRatePerSecond * dt));
  }

  /** @inheritdoc */
  public serialize(): Float32Array {
    return this.weights.slice();
  }

  /** @inheritdoc */
  public restore(state: Float32Array): void {
    assertSize('state', state, this.stateSize);
    assertFiniteBuffer('state', state);
    this.weights.set(state);
  }
}
