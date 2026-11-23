import type { Rng } from '../core/rng';
import {
  assertBound,
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertSize,
  assertWithinMagnitude,
  isFiniteBuffer,
} from '../core/validation';
import { Backpropagator } from '../nn/backprop';
import type { ActivationId } from '../nn/activations';
import { FeedforwardNetwork } from '../nn/feedforward-network';
import type { Learner } from './learner';
import { clampInPlace, scaleInPlace } from './linear-math';

/** Default maximum absolute weight. */
const DEFAULT_MAX_WEIGHT = 5;
/** Default forgetting rate: no forgetting. */
const DEFAULT_FORGET_RATE_PER_SECOND = 0;

/** Construction options for {@link MlpSgdLearner}. */
export interface MlpSgdOptions {
  /** Sizes of all layers including input and output, e.g. `[4, 6, 1]`. See `FeedforwardNetwork`. */
  readonly layerSizes: readonly number[];
  /** Source of randomness for hidden-layer initialization. */
  readonly rng: Rng;
  /** Activation of hidden layers. Default `"tanh"`. */
  readonly hiddenActivation?: ActivationId;
  /** Activation of the output layer. Default `"linear"`. */
  readonly outputActivation?: ActivationId;
  /** Maximum absolute weight, > 0. Default 5. */
  readonly maxWeight?: number;
  /** Exponential forgetting rate in 1/s, >= 0. Default 0 (no forgetting). */
  readonly forgetRatePerSecond?: number;
}

/**
 * Small multilayer perceptron trained by online stochastic gradient descent on the squared error
 * `½ Σ (output − target)²`. Use it only when the linear learners are not expressive enough.
 *
 * Hidden layers start with random (Xavier) weights so symmetry is broken; the output layer starts
 * at zero, so a new instance predicts zero for the `linear`, `tanh` and `relu` output activations
 * (a `sigmoid` output starts at 0.5). That is its "empty memory".
 *
 * State: all network weights, layout as in `FeedforwardNetwork`.
 *
 * @example
 * const learner = new MlpSgdLearner({ layerSizes: [2, 4, 1], rng: createSeededRng(1) });
 * learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.1);
 */
export class MlpSgdLearner implements Learner {
  /** Number of network inputs (`layerSizes[0]`). */
  public readonly inputSize: number;
  /** Number of network outputs (last entry of `layerSizes`). */
  public readonly outputSize: number;
  /** Length of the serialized state: the number of network weights. */
  public readonly stateSize: number;
  private readonly network: FeedforwardNetwork;
  private readonly backprop: Backpropagator;
  private readonly gradient: Float32Array;
  private readonly scratchOutput: Float32Array;
  private readonly maxWeight: number;
  private readonly forgetRatePerSecond: number;

  /**
   * @param options - See {@link MlpSgdOptions}.
   * @throws InvalidParameterError if any option is outside its documented range.
   */
  public constructor(options: MlpSgdOptions) {
    this.maxWeight = options.maxWeight ?? DEFAULT_MAX_WEIGHT;
    this.forgetRatePerSecond = options.forgetRatePerSecond ?? DEFAULT_FORGET_RATE_PER_SECOND;
    assertBound('maxWeight', this.maxWeight);
    assertNonNegative('forgetRatePerSecond', this.forgetRatePerSecond);
    this.network = new FeedforwardNetwork({ ...options, zeroOutputLayer: true });
    clampInPlace(this.network.weights, this.maxWeight);
    this.backprop = new Backpropagator(this.network);
    this.inputSize = this.network.inputSize;
    this.outputSize = this.network.outputSize;
    this.stateSize = this.network.stateSize;
    this.gradient = new Float32Array(this.stateSize);
    this.scratchOutput = new Float32Array(this.outputSize);
  }

  /** @inheritdoc */
  public predict(features: Float32Array, out: Float32Array): void {
    this.network.forward(features, out);
  }

  /** @inheritdoc */
  public learn(features: Float32Array, target: Float32Array, rate: number): void {
    assertSize('target', target, this.outputSize);
    assertFiniteBuffer('target', target);
    assertFiniteInRange('rate', rate, 0, 1);
    this.network.forward(features, this.scratchOutput);
    const loss = this.backprop.computeGradient(target, this.gradient);
    // A forward pass that overflowed Float32 (huge inputs with unbounded activations) yields a
    // non-finite loss or gradient; applying it would turn every weight into NaN, so skip the update.
    if (!Number.isFinite(loss) || !isFiniteBuffer(this.gradient)) return;
    const weights = this.network.weights;
    for (let index = 0; index < weights.length; index += 1) {
      const next = (weights[index] as number) - rate * (this.gradient[index] as number);
      if (next > this.maxWeight) weights[index] = this.maxWeight;
      else if (next < -this.maxWeight) weights[index] = -this.maxWeight;
      else weights[index] = next;
    }
  }

  /** @inheritdoc */
  public decay(dt: number): void {
    assertNonNegative('dt', dt);
    if (this.forgetRatePerSecond === 0) return;
    scaleInPlace(this.network.weights, Math.exp(-this.forgetRatePerSecond * dt));
  }

  /** @inheritdoc */
  public serialize(): Float32Array {
    return this.network.serialize();
  }

  /** @inheritdoc */
  public restore(state: Float32Array): void {
    assertSize('state', state, this.stateSize);
    assertWithinMagnitude('state', state, this.maxWeight);
    this.network.restore(state);
  }
}
