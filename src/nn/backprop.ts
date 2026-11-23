import { assertFiniteBuffer, assertSize } from '../core/validation';
import type { FeedforwardNetwork } from './feedforward-network';

/**
 * Computes the gradient of the squared error `½ Σ (output − target)²` with respect to every
 * weight of a {@link FeedforwardNetwork}. All buffers are allocated once in the constructor.
 *
 * Call `network.forward(...)` first: the gradient is computed from the layer values it left behind.
 *
 * @example
 * const backprop = new Backpropagator(network);
 * const gradient = new Float32Array(network.weights.length);
 * network.forward(input, out);
 * const loss = backprop.computeGradient(target, gradient);
 */
export class Backpropagator {
  private readonly network: FeedforwardNetwork;
  private readonly deltas: readonly Float32Array[];

  /**
   * @param network - Network whose gradient will be computed.
   */
  public constructor(network: FeedforwardNetwork) {
    this.network = network;
    this.deltas = network.layerSizes.slice(1).map((size) => new Float32Array(size));
  }

  /**
   * Writes `∂loss/∂weight` for every weight into `gradient` (same layout as `network.weights`).
   *
   * @param target - Desired output, length `network.outputSize`.
   * @param gradient - Output buffer, length `network.weights.length`, fully overwritten.
   * @returns The loss `½ Σ (output − target)²` for the last forward pass.
   */
  public computeGradient(target: Float32Array, gradient: Float32Array): number {
    const network = this.network;
    assertSize('target', target, network.outputSize);
    assertSize('gradient', gradient, network.weights.length);
    assertFiniteBuffer('target', target);
    const layerCount = network.layerSizes.length - 1;

    const outputValues = network.layerOutputs[layerCount] as Float32Array;
    const outputActivation = network.layerActivations[layerCount - 1];
    const outputDeltas = this.deltas[layerCount - 1] as Float32Array;
    let loss = 0;
    for (let index = 0; index < network.outputSize; index += 1) {
      const value = outputValues[index] as number;
      const error = value - (target[index] as number);
      loss += 0.5 * error * error;
      outputDeltas[index] = error;
    }
    outputActivation?.scaleByDerivative(outputValues, outputDeltas);

    for (let layer = layerCount - 1; layer >= 0; layer -= 1) {
      const fanIn = network.layerSizes[layer] as number;
      const fanOut = network.layerSizes[layer + 1] as number;
      const start = network.layerOffsets[layer] as number;
      const biasStart = start + fanOut * fanIn;
      const inputs = network.layerOutputs[layer] as Float32Array;
      const layerDeltas = this.deltas[layer] as Float32Array;
      for (let row = 0; row < fanOut; row += 1) {
        const delta = layerDeltas[row] as number;
        const base = start + row * fanIn;
        for (let column = 0; column < fanIn; column += 1) {
          gradient[base + column] = delta * (inputs[column] as number);
        }
        gradient[biasStart + row] = delta;
      }
      if (layer > 0) {
        const previousDeltas = this.deltas[layer - 1] as Float32Array;
        const previousActivation = network.layerActivations[layer - 1];
        for (let column = 0; column < fanIn; column += 1) {
          let sum = 0;
          for (let row = 0; row < fanOut; row += 1) {
            sum +=
              (network.weights[start + row * fanIn + column] as number) *
              (layerDeltas[row] as number);
          }
          previousDeltas[column] = sum;
        }
        previousActivation?.scaleByDerivative(inputs, previousDeltas);
      }
    }
    return loss;
  }
}
