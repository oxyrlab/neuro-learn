import { describe, expect, it } from 'vitest';
import { SizeMismatchError } from '../core/errors';
import { createSeededRng } from '../core/rng';
import type { ActivationId } from './activations';
import { Backpropagator } from './backprop';
import { FeedforwardNetwork } from './feedforward-network';

/** Central finite difference step. Weights are Float32, so a larger step keeps rounding noise small. */
const EPSILON = 5e-3;
/** Maximum relative error allowed between analytic and numerical gradients. */
const TOLERANCE = 2e-3;

function lossOf(network: FeedforwardNetwork, input: Float32Array, target: Float32Array): number {
  const out = new Float32Array(network.outputSize);
  network.forward(input, out);
  let loss = 0;
  for (let i = 0; i < out.length; i += 1) {
    const error = (out[i] as number) - (target[i] as number);
    loss += 0.5 * error * error;
  }
  return loss;
}

function numericalGradient(
  network: FeedforwardNetwork,
  input: Float32Array,
  target: Float32Array,
  index: number,
): number {
  const original = network.weights[index] as number;
  network.weights[index] = original + EPSILON;
  const plus = network.weights[index] as number;
  const lossPlus = lossOf(network, input, target);
  network.weights[index] = original - EPSILON;
  const minus = network.weights[index] as number;
  const lossMinus = lossOf(network, input, target);
  network.weights[index] = original;
  return (lossPlus - lossMinus) / (plus - minus);
}

const hiddenOptions: ActivationId[] = ['tanh', 'sigmoid', 'linear', 'relu'];
const outputOptions: ActivationId[] = ['linear', 'tanh', 'sigmoid'];

describe('Backpropagator gradient check', () => {
  for (const hiddenActivation of hiddenOptions) {
    for (const outputActivation of outputOptions) {
      it(`matches numerical gradients (hidden ${hiddenActivation}, output ${outputActivation})`, () => {
        const network = new FeedforwardNetwork({
          layerSizes: [3, 4, 3, 2],
          rng: createSeededRng(11),
          hiddenActivation,
          outputActivation,
        });
        // Non-zero biases make the check cover bias gradients and avoid relu kinks at zero.
        for (let i = 0; i < network.weights.length; i += 1) {
          network.weights[i] = (network.weights[i] as number) + 0.05 * Math.sin(i + 1);
        }
        const input = Float32Array.of(0.4, -0.7, 0.9);
        const target = Float32Array.of(0.3, 0.6);
        const out = new Float32Array(2);
        const gradient = new Float32Array(network.weights.length);
        network.forward(input, out);
        new Backpropagator(network).computeGradient(target, gradient);
        for (let index = 0; index < gradient.length; index += 1) {
          const analytic = gradient[index] as number;
          const numeric = numericalGradient(network, input, target, index);
          const scale = Math.max(1, Math.abs(analytic), Math.abs(numeric));
          expect(Math.abs(analytic - numeric) / scale).toBeLessThan(TOLERANCE);
        }
      });
    }
  }

  it('handles a single weight layer (no hidden layer)', () => {
    const network = new FeedforwardNetwork({ layerSizes: [2, 1], rng: createSeededRng(3) });
    const input = Float32Array.of(0.5, -0.25);
    const target = Float32Array.of(1);
    const gradient = new Float32Array(network.weights.length);
    network.forward(input, new Float32Array(1));
    new Backpropagator(network).computeGradient(target, gradient);
    for (let index = 0; index < gradient.length; index += 1) {
      expect(gradient[index]).toBeCloseTo(numericalGradient(network, input, target, index), 3);
    }
  });
});

describe('Backpropagator', () => {
  it('returns the loss of the last forward pass', () => {
    const network = new FeedforwardNetwork({ layerSizes: [1, 1], rng: createSeededRng(1) });
    network.weights.set([2, 0]);
    network.forward(Float32Array.of(1), new Float32Array(1));
    const loss = new Backpropagator(network).computeGradient(
      Float32Array.of(0),
      new Float32Array(2),
    );
    expect(loss).toBeCloseTo(2, 6); // output 2, target 0 → ½ · 2²
  });

  it('overwrites the whole gradient buffer', () => {
    const network = new FeedforwardNetwork({ layerSizes: [2, 2, 1], rng: createSeededRng(1) });
    const gradient = new Float32Array(network.weights.length).fill(123);
    network.forward(Float32Array.of(1, 1), new Float32Array(1));
    new Backpropagator(network).computeGradient(Float32Array.of(0.5), gradient);
    expect(gradient.every((g) => g !== 123)).toBe(true);
  });

  it('validates buffer sizes', () => {
    const network = new FeedforwardNetwork({ layerSizes: [2, 1], rng: createSeededRng(1) });
    const backprop = new Backpropagator(network);
    expect(() => backprop.computeGradient(new Float32Array(2), new Float32Array(3))).toThrow(
      SizeMismatchError,
    );
    expect(() => backprop.computeGradient(new Float32Array(1), new Float32Array(2))).toThrow(
      SizeMismatchError,
    );
  });
});
