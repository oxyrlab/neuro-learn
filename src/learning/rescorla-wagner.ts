import {
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertPositive,
  assertPositiveInteger,
  assertSize,
} from '../core/validation';
import type { Learner } from './learner';
import { applyDeltaRule, multiplyMatrixVector, scaleInPlace } from './linear-math';

/** Default maximum absolute associative strength. */
const DEFAULT_MAX_STRENGTH = 1;
/** Default forgetting rate: no forgetting. */
const DEFAULT_FORGET_RATE_PER_SECOND = 0;

/** Construction options for {@link RescorlaWagnerLearner}. */
export interface RescorlaWagnerOptions {
  /** Number of cues (input features), integer >= 1. */
  readonly inputSize: number;
  /** Number of outcomes (outputs), integer >= 1. */
  readonly outputSize: number;
  /** Maximum absolute associative strength `|V|`, > 0. Default 1. */
  readonly maxStrength?: number;
  /** Exponential forgetting rate in 1/s, >= 0. Default 0 (no forgetting). */
  readonly forgetRatePerSecond?: number;
}

/**
 * Rescorla–Wagner conditioning: associative strengths `V` (outcomes × cues) learned with the
 * delta rule `ΔV = α · cue · (λ − ΣV·cue)`. Extinction happens naturally when `λ = 0`.
 *
 * State: `V`, row-major, length `outputSize * inputSize`.
 *
 * @example
 * const learner = new RescorlaWagnerLearner({ inputSize: 4, outputSize: 1 });
 * learner.learn(Float32Array.of(1, 0, 0, 0), Float32Array.of(1), 0.2); // cue 0 paired with outcome
 */
export class RescorlaWagnerLearner implements Learner {
  public readonly inputSize: number;
  public readonly outputSize: number;
  public readonly stateSize: number;
  private readonly strengths: Float32Array;
  private readonly maxStrength: number;
  private readonly forgetRatePerSecond: number;

  /**
   * @param options - See {@link RescorlaWagnerOptions}.
   * @throws InvalidParameterError if any option is outside its documented range.
   */
  public constructor(options: RescorlaWagnerOptions) {
    assertPositiveInteger('inputSize', options.inputSize);
    assertPositiveInteger('outputSize', options.outputSize);
    this.maxStrength = options.maxStrength ?? DEFAULT_MAX_STRENGTH;
    this.forgetRatePerSecond = options.forgetRatePerSecond ?? DEFAULT_FORGET_RATE_PER_SECOND;
    assertPositive('maxStrength', this.maxStrength);
    assertNonNegative('forgetRatePerSecond', this.forgetRatePerSecond);
    this.inputSize = options.inputSize;
    this.outputSize = options.outputSize;
    this.stateSize = options.inputSize * options.outputSize;
    this.strengths = new Float32Array(this.stateSize);
  }

  /** @inheritdoc */
  public predict(features: Float32Array, out: Float32Array): void {
    assertSize('features', features, this.inputSize);
    assertSize('out', out, this.outputSize);
    assertFiniteBuffer('features', features);
    multiplyMatrixVector(this.strengths, this.outputSize, this.inputSize, features, out);
  }

  /** @inheritdoc */
  public learn(features: Float32Array, target: Float32Array, rate: number): void {
    assertSize('features', features, this.inputSize);
    assertSize('target', target, this.outputSize);
    assertFiniteBuffer('features', features);
    assertFiniteBuffer('target', target);
    assertFiniteInRange('rate', rate, 0, 1);
    applyDeltaRule(
      this.strengths,
      this.outputSize,
      this.inputSize,
      features,
      target,
      rate,
      this.maxStrength,
    );
  }

  /** @inheritdoc */
  public decay(dt: number): void {
    assertNonNegative('dt', dt);
    if (this.forgetRatePerSecond === 0) return;
    scaleInPlace(this.strengths, Math.exp(-this.forgetRatePerSecond * dt));
  }

  /** @inheritdoc */
  public serialize(): Float32Array {
    return this.strengths.slice();
  }

  /** @inheritdoc */
  public restore(state: Float32Array): void {
    assertSize('state', state, this.stateSize);
    assertFiniteBuffer('state', state);
    this.strengths.set(state);
  }
}
