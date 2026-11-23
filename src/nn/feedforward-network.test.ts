import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { InvalidParameterError, SizeMismatchError } from '../core/errors';
import { createSeededRng } from '../core/rng';
import { FeedforwardNetwork } from './feedforward-network';

const rng = () => createSeededRng(1);

function handBuilt(hiddenActivation: 'linear' | 'relu'): FeedforwardNetwork {
  const network = new FeedforwardNetwork({
    layerSizes: [2, 2, 1],
    rng: rng(),
    hiddenActivation,
    outputActivation: 'linear',
  });
  // layer 0: W = [[1, 2], [3, 4]], b = [0.5, -0.5]; layer 1: W = [[1, -1]], b = [0.25]
  network.weights.set([1, 2, 3, 4, 0.5, -0.5, 1, -1, 0.25]);
  return network;
}

describe('FeedforwardNetwork', () => {
  it('computes a hand-calculated forward pass', () => {
    const out = new Float32Array(1);
    handBuilt('linear').forward(Float32Array.of(1, 1), out);
    expect(out[0]).toBeCloseTo(-2.75, 6);
  });

  it('applies the hidden activation (relu zeroes negative pre-activations)', () => {
    const out = new Float32Array(1);
    handBuilt('relu').forward(Float32Array.of(-1, -1), out);
    expect(out[0]).toBeCloseTo(0.25, 6);
  });

  it('reports sizes and weight count', () => {
    const network = new FeedforwardNetwork({ layerSizes: [8, 6, 3], rng: rng() });
    expect(network.inputSize).toBe(8);
    expect(network.outputSize).toBe(3);
    expect(network.stateSize).toBe(6 * 9 + 3 * 7);
    expect(network.weights.length).toBe(network.stateSize);
  });

  it('initializes with Xavier-bounded weights and zero biases', () => {
    const network = new FeedforwardNetwork({ layerSizes: [4, 3], rng: rng() });
    const limit = Math.sqrt(6 / 7);
    for (let i = 0; i < 12; i += 1)
      expect(Math.abs(network.weights[i] as number)).toBeLessThanOrEqual(limit);
    for (let i = 12; i < 15; i += 1) expect(network.weights[i]).toBe(0);
    expect(network.weights.some((w) => w !== 0)).toBe(true);
  });

  it('is deterministic for a seed and differs between seeds', () => {
    const a = new FeedforwardNetwork({ layerSizes: [3, 4, 2], rng: createSeededRng(5) });
    const b = new FeedforwardNetwork({ layerSizes: [3, 4, 2], rng: createSeededRng(5) });
    const c = new FeedforwardNetwork({ layerSizes: [3, 4, 2], rng: createSeededRng(6) });
    expect(Array.from(a.weights)).toEqual(Array.from(b.weights));
    expect(Array.from(a.weights)).not.toEqual(Array.from(c.weights));
  });

  it('zeroOutputLayer leaves the output layer at zero and the prediction at zero', () => {
    const network = new FeedforwardNetwork({
      layerSizes: [3, 4, 2],
      rng: rng(),
      zeroOutputLayer: true,
    });
    expect(network.weights.slice(0, 16).some((w) => w !== 0)).toBe(true);
    expect(network.weights.slice(16).every((w) => w === 0)).toBe(true);
    const out = new Float32Array(2);
    network.forward(Float32Array.of(1, -1, 0.5), out);
    expect(Array.from(out)).toEqual([0, 0]);
  });

  it('does not modify its input', () => {
    const network = new FeedforwardNetwork({ layerSizes: [2, 2], rng: rng() });
    const input = Float32Array.of(0.3, 0.7);
    network.forward(input, new Float32Array(2));
    expect(Array.from(input)).toEqual([Math.fround(0.3), Math.fround(0.7)]);
  });

  it('serialize returns an independent copy and restore replaces the weights', () => {
    const a = new FeedforwardNetwork({ layerSizes: [2, 3, 1], rng: createSeededRng(1) });
    const b = new FeedforwardNetwork({ layerSizes: [2, 3, 1], rng: createSeededRng(2) });
    const snapshot = a.serialize();
    snapshot.fill(9);
    expect(a.weights.every((w) => w !== 9)).toBe(true);
    b.restore(a.serialize());
    const outA = new Float32Array(1);
    const outB = new Float32Array(1);
    a.forward(Float32Array.of(0.1, 0.2), outA);
    b.forward(Float32Array.of(0.1, 0.2), outB);
    expect(outB[0]).toBe(outA[0]);
  });

  it('keeps outputs finite for any finite input (property)', () => {
    const network = new FeedforwardNetwork({ layerSizes: [3, 5, 2], rng: rng() });
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -10, max: 10, noNaN: true }), { minLength: 3, maxLength: 3 }),
        (values) => {
          const out = new Float32Array(2);
          network.forward(Float32Array.from(values), out);
          return out.every((v) => Number.isFinite(v));
        },
      ),
    );
  });

  it('keeps sigmoid outputs inside [0, 1] (property)', () => {
    const network = new FeedforwardNetwork({
      layerSizes: [2, 4, 2],
      rng: rng(),
      outputActivation: 'sigmoid',
    });
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -100, max: 100, noNaN: true }), { minLength: 2, maxLength: 2 }),
        (values) => {
          const out = new Float32Array(2);
          network.forward(Float32Array.from(values), out);
          return out.every((v) => v >= 0 && v <= 1);
        },
      ),
    );
  });

  it('validates buffer sizes and values', () => {
    const network = new FeedforwardNetwork({ layerSizes: [2, 1], rng: rng() });
    expect(() => network.forward(new Float32Array(3), new Float32Array(1))).toThrow(
      SizeMismatchError,
    );
    expect(() => network.forward(new Float32Array(2), new Float32Array(2))).toThrow(
      SizeMismatchError,
    );
    expect(() => network.forward(Float32Array.of(NaN, 0), new Float32Array(1))).toThrow(
      InvalidParameterError,
    );
    expect(() => network.restore(new Float32Array(1))).toThrow(SizeMismatchError);
    expect(() => network.restore(new Float32Array(network.stateSize).fill(NaN))).toThrow(
      InvalidParameterError,
    );
  });

  it.each([
    ['layerSizes', { layerSizes: [3] }],
    ['layerSizes', { layerSizes: [] }],
    ['layerSizes\\[1\\]', { layerSizes: [2, 0, 1] }],
    ['layerSizes\\[0\\]', { layerSizes: [1.5, 1] }],
    ['hiddenActivation', { layerSizes: [2, 2, 1], hiddenActivation: 'swish' as never }],
    ['outputActivation', { layerSizes: [2, 1], outputActivation: 'swish' as never }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new FeedforwardNetwork({ rng: rng(), ...options })).toThrow(InvalidParameterError);
    expect(() => new FeedforwardNetwork({ rng: rng(), ...options })).toThrow(new RegExp(name));
  });
});
