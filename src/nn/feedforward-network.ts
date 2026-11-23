import { InvalidParameterError } from '../core/errors';
import type { Rng } from '../core/rng';
import { assertFiniteBuffer, assertPositiveInteger, assertSize } from '../core/validation';
import type { Activation, ActivationId } from './activations';
import { getActivation } from './activations';
import type { Network } from './network';

/** Default activation of hidden layers. */
const DEFAULT_HIDDEN_ACTIVATION: ActivationId = 'tanh';
/** Default activation of the output layer. */
const DEFAULT_OUTPUT_ACTIVATION: ActivationId = 'linear';

/** Construction options for {@link FeedforwardNetwork}. */
export interface FeedforwardNetworkOptions {
  /** Sizes of all layers including input and output, e.g. `[8, 6, 3]`; at least 2 entries, each integer >= 1. */
  readonly layerSizes: readonly number[];
  /** Source of randomness for weight initialization. */
  readonly rng: Rng;
  /** Activation of hidden layers. Default `"tanh"`. */
  readonly hiddenActivation?: ActivationId;
  /** Activation of the output layer. Default `"linear"`. */
  readonly outputActivation?: ActivationId;
  /** If true, the output layer starts at zero so the initial prediction is zero. Default false. */
  readonly zeroOutputLayer?: boolean;
}

/**
 * Fully connected feedforward network.
 *
 * Weight layout: for each layer `l` (from `layerSizes[l]` to `layerSizes[l+1]`), the row-major
 * weight matrix `W[out][in]` followed by the bias vector `b[out]`. Initialization is Xavier
 * uniform for weights (limit `sqrt(6 / (in + out))`) and zero for biases.
 *
 * @example
 * const network = new FeedforwardNetwork({ layerSizes: [2, 3, 1], rng: createSeededRng(1) });
 * const out = new Float32Array(1);
 * network.forward(Float32Array.of(0.5, -0.5), out);
 */
export class FeedforwardNetwork implements Network {
  public readonly inputSize: number;
  public readonly outputSize: number;
  public readonly stateSize: number;
  public readonly weights: Float32Array;
  /** Sizes of all layers including input and output. */
  public readonly layerSizes: readonly number[];
  /** Offset of each layer's weight matrix inside `weights`; its bias follows the matrix. */
  public readonly layerOffsets: readonly number[];
  /** Activation of each weight layer (`layerSizes.length - 1` entries). */
  public readonly layerActivations: readonly Activation[];
  /**
   * Layer values from the most recent `forward`: index 0 is the input, the last is the output.
   * Used by backpropagation; contents are only meaningful right after `forward`.
   */
  public readonly layerOutputs: readonly Float32Array[];

  /**
   * @param options - See {@link FeedforwardNetworkOptions}.
   * @throws InvalidParameterError if `layerSizes` or an activation id is invalid.
   */
  public constructor(options: FeedforwardNetworkOptions) {
    const sizes = options.layerSizes;
    if (!Array.isArray(sizes) || sizes.length < 2) {
      throw new InvalidParameterError(
        'layerSizes',
        JSON.stringify(sizes),
        'an array with at least 2 entries (input and output)',
      );
    }
    sizes.forEach((size, index) => assertPositiveInteger(`layerSizes[${index}]`, size));
    const hidden = getActivation(
      options.hiddenActivation ?? DEFAULT_HIDDEN_ACTIVATION,
      'hiddenActivation',
    );
    const output = getActivation(
      options.outputActivation ?? DEFAULT_OUTPUT_ACTIVATION,
      'outputActivation',
    );
    const layerCount = sizes.length - 1;
    const offsets: number[] = [];
    let total = 0;
    for (let layer = 0; layer < layerCount; layer += 1) {
      offsets.push(total);
      total += (sizes[layer + 1] as number) * ((sizes[layer] as number) + 1);
    }
    this.layerSizes = [...sizes];
    this.layerOffsets = offsets;
    this.layerActivations = this.layerSizes
      .slice(1)
      .map((_, layer) => (layer === layerCount - 1 ? output : hidden));
    this.layerOutputs = this.layerSizes.map((size) => new Float32Array(size));
    this.inputSize = sizes[0] as number;
    this.outputSize = sizes[layerCount] as number;
    this.stateSize = total;
    this.weights = new Float32Array(total);
    const randomizedLayers = options.zeroOutputLayer === true ? layerCount - 1 : layerCount;
    for (let layer = 0; layer < randomizedLayers; layer += 1) {
      const fanIn = sizes[layer] as number;
      const fanOut = sizes[layer + 1] as number;
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      const start = offsets[layer] as number;
      for (let index = 0; index < fanOut * fanIn; index += 1) {
        this.weights[start + index] = (options.rng.next() * 2 - 1) * limit;
      }
    }
  }

  /** @inheritdoc */
  public forward(input: Float32Array, out: Float32Array): void {
    assertSize('input', input, this.inputSize);
    assertSize('out', out, this.outputSize);
    assertFiniteBuffer('input', input);
    (this.layerOutputs[0] as Float32Array).set(input);
    const layerCount = this.layerSizes.length - 1;
    for (let layer = 0; layer < layerCount; layer += 1) {
      const previous = this.layerOutputs[layer] as Float32Array;
      const next = this.layerOutputs[layer + 1] as Float32Array;
      const fanIn = this.layerSizes[layer] as number;
      const fanOut = this.layerSizes[layer + 1] as number;
      const start = this.layerOffsets[layer] as number;
      const biasStart = start + fanOut * fanIn;
      const activation = this.layerActivations[layer] as Activation;
      for (let row = 0; row < fanOut; row += 1) {
        const base = start + row * fanIn;
        let sum = this.weights[biasStart + row] as number;
        for (let column = 0; column < fanIn; column += 1) {
          sum += (this.weights[base + column] as number) * (previous[column] as number);
        }
        next[row] = sum;
      }
      activation.applyInPlace(next);
    }
    out.set(this.layerOutputs[layerCount] as Float32Array);
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
