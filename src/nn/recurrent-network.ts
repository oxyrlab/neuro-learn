import type { Rng } from '../core/rng';
import { assertFiniteBuffer, assertPositiveInteger, assertSize } from '../core/validation';
import type { ActivationId } from './activations';
import { getActivation } from './activations';
import type { Network } from './network';

/** Default activation of the hidden state. */
const DEFAULT_HIDDEN_ACTIVATION: ActivationId = 'tanh';
/** Default activation of the output. */
const DEFAULT_OUTPUT_ACTIVATION: ActivationId = 'linear';

/** Construction options for {@link RecurrentNetwork}. */
export interface RecurrentNetworkOptions {
  /** Number of inputs, integer >= 1. */
  readonly inputSize: number;
  /** Number of hidden units, integer >= 1. */
  readonly hiddenSize: number;
  /** Number of outputs, integer >= 1. */
  readonly outputSize: number;
  /** Source of randomness for weight initialization. */
  readonly rng: Rng;
  /** Activation of the hidden state. Default `"tanh"`. */
  readonly hiddenActivation?: ActivationId;
  /** Activation of the output. Default `"linear"`. */
  readonly outputActivation?: ActivationId;
}

/**
 * Elman recurrent network. Each `forward` call is one time step:
 * `h' = f(Wxh·x + Whh·h + bh)`, `out = g(Why·h' + by)`; `h'` is kept for the next call.
 *
 * Weight layout in `weights`: `Wxh` (hidden × input), `Whh` (hidden × hidden), `bh` (hidden),
 * `Why` (output × hidden), `by` (output). Weights use Xavier-uniform initialization, biases zero.
 * Serialized state is the weights followed by the hidden state. There is no gradient training;
 * weights are meant to be evolved from outside.
 *
 * @example
 * const network = new RecurrentNetwork({ inputSize: 2, hiddenSize: 4, outputSize: 1, rng: createSeededRng(1) });
 * network.forward(input, out); // depends on earlier calls
 * network.reset();             // forget the history
 */
export class RecurrentNetwork implements Network {
  public readonly inputSize: number;
  public readonly hiddenSize: number;
  public readonly outputSize: number;
  public readonly stateSize: number;
  public readonly weights: Float32Array;
  private readonly state: Float32Array;
  private readonly hidden: Float32Array;
  private readonly nextHidden: Float32Array;
  private readonly hiddenActivation: ReturnType<typeof getActivation>;
  private readonly outputActivation: ReturnType<typeof getActivation>;
  private readonly inputToHiddenStart = 0;
  private readonly hiddenToHiddenStart: number;
  private readonly hiddenBiasStart: number;
  private readonly hiddenToOutputStart: number;
  private readonly outputBiasStart: number;

  /**
   * @param options - See {@link RecurrentNetworkOptions}.
   * @throws InvalidParameterError if a size or activation id is invalid.
   */
  public constructor(options: RecurrentNetworkOptions) {
    assertPositiveInteger('inputSize', options.inputSize);
    assertPositiveInteger('hiddenSize', options.hiddenSize);
    assertPositiveInteger('outputSize', options.outputSize);
    this.hiddenActivation = getActivation(
      options.hiddenActivation ?? DEFAULT_HIDDEN_ACTIVATION,
      'hiddenActivation',
    );
    this.outputActivation = getActivation(
      options.outputActivation ?? DEFAULT_OUTPUT_ACTIVATION,
      'outputActivation',
    );
    const { inputSize, hiddenSize, outputSize } = options;
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.hiddenToHiddenStart = this.inputToHiddenStart + hiddenSize * inputSize;
    this.hiddenBiasStart = this.hiddenToHiddenStart + hiddenSize * hiddenSize;
    this.hiddenToOutputStart = this.hiddenBiasStart + hiddenSize;
    this.outputBiasStart = this.hiddenToOutputStart + outputSize * hiddenSize;
    const weightCount = this.outputBiasStart + outputSize;
    this.stateSize = weightCount + hiddenSize;
    this.state = new Float32Array(this.stateSize);
    this.weights = this.state.subarray(0, weightCount);
    this.hidden = this.state.subarray(weightCount);
    this.nextHidden = new Float32Array(hiddenSize);
    this.randomize(
      this.inputToHiddenStart,
      hiddenSize * inputSize,
      inputSize + hiddenSize,
      options.rng,
    );
    this.randomize(this.hiddenToHiddenStart, hiddenSize * hiddenSize, 2 * hiddenSize, options.rng);
    this.randomize(
      this.hiddenToOutputStart,
      outputSize * hiddenSize,
      hiddenSize + outputSize,
      options.rng,
    );
  }

  private randomize(start: number, count: number, fanSum: number, rng: Rng): void {
    const limit = Math.sqrt(6 / fanSum);
    for (let index = 0; index < count; index += 1) {
      this.weights[start + index] = (rng.next() * 2 - 1) * limit;
    }
  }

  /** Sets the hidden state back to zero (forgets the input history). Weights are unchanged. */
  public reset(): void {
    this.hidden.fill(0);
  }

  /** @inheritdoc */
  public forward(input: Float32Array, out: Float32Array): void {
    assertSize('input', input, this.inputSize);
    assertSize('out', out, this.outputSize);
    assertFiniteBuffer('input', input);
    for (let unit = 0; unit < this.hiddenSize; unit += 1) {
      let sum = this.weights[this.hiddenBiasStart + unit] as number;
      const inputBase = this.inputToHiddenStart + unit * this.inputSize;
      for (let column = 0; column < this.inputSize; column += 1) {
        sum += (this.weights[inputBase + column] as number) * (input[column] as number);
      }
      const recurrentBase = this.hiddenToHiddenStart + unit * this.hiddenSize;
      for (let column = 0; column < this.hiddenSize; column += 1) {
        sum += (this.weights[recurrentBase + column] as number) * (this.hidden[column] as number);
      }
      this.nextHidden[unit] = this.hiddenActivation.apply(sum);
    }
    this.hidden.set(this.nextHidden);
    for (let row = 0; row < this.outputSize; row += 1) {
      let sum = this.weights[this.outputBiasStart + row] as number;
      const base = this.hiddenToOutputStart + row * this.hiddenSize;
      for (let column = 0; column < this.hiddenSize; column += 1) {
        sum += (this.weights[base + column] as number) * (this.hidden[column] as number);
      }
      out[row] = this.outputActivation.apply(sum);
    }
  }

  /** @inheritdoc */
  public serialize(): Float32Array {
    return this.state.slice();
  }

  /** @inheritdoc */
  public restore(state: Float32Array): void {
    assertSize('state', state, this.stateSize);
    assertFiniteBuffer('state', state);
    this.state.set(state);
  }
}
