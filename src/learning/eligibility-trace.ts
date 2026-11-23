import {
  assertBound,
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertNonNegativeBuffer,
  assertPositive,
  assertPositiveInteger,
  assertSize,
  assertWithinMagnitude,
  createFloat32Array,
} from '../core/validation';
import type { Learner } from './learner';
import { applyDeltaRule, multiplyMatrixVector, scaleInPlace } from './linear-math';

/** Default maximum absolute associative strength. */
const DEFAULT_MAX_STRENGTH = 1;
/** Default forgetting rate: no forgetting. */
const DEFAULT_FORGET_RATE_PER_SECOND = 0;

/** Construction options for {@link EligibilityTraceLearner}. */
export interface EligibilityTraceOptions {
  /** Number of cues (input features), integer >= 1. */
  readonly inputSize: number;
  /** Number of outcomes (outputs), integer >= 1. */
  readonly outputSize: number;
  /** Time constant τ in seconds: a cue's trace falls to 1/e after τ seconds. Must be > 0. */
  readonly traceTimeConstantSeconds: number;
  /** Maximum absolute associative strength, > 0. Default 1. */
  readonly maxStrength?: number;
  /** Exponential forgetting rate of the strengths in 1/s, >= 0. Default 0. */
  readonly forgetRatePerSecond?: number;
}

/**
 * Delayed credit assignment. Keeps a decaying trace of recently seen cues; when an outcome
 * arrives later, `learn` credits cues in proportion to their remaining trace (delta rule on the
 * trace instead of the instantaneous cue).
 *
 * Typical tick: `decay(dt)`, `observe(features)`, `predict(...)`; call `learn(...)` when an
 * outcome (reward or pain) arrives. `predict` uses the current cues and never changes state.
 *
 * State: strengths (`outputSize * inputSize`) followed by the trace (`inputSize`).
 *
 * @example
 * const learner = new EligibilityTraceLearner({ inputSize: 4, outputSize: 1, traceTimeConstantSeconds: 2 });
 * learner.observe(Float32Array.of(1, 0, 0, 0)); // cue seen
 * learner.decay(1);                             // time passes
 * learner.learn(new Float32Array(4), Float32Array.of(1), 0.5); // outcome arrives later
 */
export class EligibilityTraceLearner implements Learner {
  /** Number of cues (input features). */
  public readonly inputSize: number;
  /** Number of outcomes (outputs). */
  public readonly outputSize: number;
  /** Length of the serialized state: strengths plus trace, `inputSize * (outputSize + 1)`. */
  public readonly stateSize: number;
  private readonly state: Float32Array;
  private readonly strengths: Float32Array;
  private readonly trace: Float32Array;
  private readonly traceTimeConstantSeconds: number;
  private readonly maxStrength: number;
  private readonly forgetRatePerSecond: number;

  /**
   * @param options - See {@link EligibilityTraceOptions}.
   * @throws InvalidParameterError if any option is outside its documented range.
   */
  public constructor(options: EligibilityTraceOptions) {
    assertPositiveInteger('inputSize', options.inputSize);
    assertPositiveInteger('outputSize', options.outputSize);
    assertPositive('traceTimeConstantSeconds', options.traceTimeConstantSeconds);
    this.maxStrength = options.maxStrength ?? DEFAULT_MAX_STRENGTH;
    this.forgetRatePerSecond = options.forgetRatePerSecond ?? DEFAULT_FORGET_RATE_PER_SECOND;
    assertBound('maxStrength', this.maxStrength);
    assertNonNegative('forgetRatePerSecond', this.forgetRatePerSecond);
    this.inputSize = options.inputSize;
    this.outputSize = options.outputSize;
    this.traceTimeConstantSeconds = options.traceTimeConstantSeconds;
    const strengthCount = options.inputSize * options.outputSize;
    this.stateSize = strengthCount + options.inputSize;
    this.state = createFloat32Array('inputSize * (outputSize + 1)', this.stateSize);
    this.strengths = this.state.subarray(0, strengthCount);
    this.trace = this.state.subarray(strengthCount);
  }

  /**
   * Records currently visible cues: `trace[i] = max(trace[i], features[i])` (replacing trace).
   *
   * @param features - Cue strengths of length `inputSize`, finite and >= 0 (normally in [0, 1]).
   *   Negative values are rejected because a replacing trace cannot represent them.
   * @throws InvalidParameterError if any value is negative or non-finite.
   */
  public observe(features: Float32Array): void {
    assertSize('features', features, this.inputSize);
    assertNonNegativeBuffer('features', features);
    for (let index = 0; index < this.inputSize; index += 1) {
      const value = features[index] as number;
      if (value > (this.trace[index] as number)) this.trace[index] = value;
    }
  }

  /** @inheritdoc */
  public predict(features: Float32Array, out: Float32Array): void {
    assertSize('features', features, this.inputSize);
    assertSize('out', out, this.outputSize);
    assertFiniteBuffer('features', features);
    multiplyMatrixVector(this.strengths, this.outputSize, this.inputSize, features, out);
  }

  /**
   * Folds `features` into the trace, then credits cues by their trace strength.
   *
   * @inheritdoc
   */
  public learn(features: Float32Array, target: Float32Array, rate: number): void {
    assertSize('features', features, this.inputSize);
    assertSize('target', target, this.outputSize);
    assertFiniteBuffer('features', features);
    assertFiniteBuffer('target', target);
    assertFiniteInRange('rate', rate, 0, 1);
    this.observe(features);
    applyDeltaRule(
      this.strengths,
      this.outputSize,
      this.inputSize,
      this.trace,
      target,
      rate,
      this.maxStrength,
    );
  }

  /**
   * Advances time: the trace decays with time constant τ and the strengths forget at
   * `forgetRatePerSecond`.
   *
   * @inheritdoc
   */
  public decay(dt: number): void {
    assertNonNegative('dt', dt);
    scaleInPlace(this.trace, Math.exp(-dt / this.traceTimeConstantSeconds));
    if (this.forgetRatePerSecond === 0) return;
    scaleInPlace(this.strengths, Math.exp(-this.forgetRatePerSecond * dt));
  }

  /** @inheritdoc */
  public serialize(): Float32Array {
    return this.state.slice();
  }

  /** @inheritdoc */
  public restore(state: Float32Array): void {
    assertSize('state', state, this.stateSize);
    assertWithinMagnitude('state', state.subarray(0, this.strengths.length), this.maxStrength);
    assertNonNegativeBuffer('state.trace', state.subarray(this.strengths.length));
    this.state.set(state);
  }
}
