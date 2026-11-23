import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { InvalidParameterError, NumericInstabilityError, SizeMismatchError } from '../core/errors';
import { createSeededRng } from '../core/rng';
import { RecurrentNetwork } from './recurrent-network';

const base = { inputSize: 1, hiddenSize: 1, outputSize: 1 };

function handBuilt(): RecurrentNetwork {
  const network = new RecurrentNetwork({
    ...base,
    rng: createSeededRng(1),
    hiddenActivation: 'linear',
    outputActivation: 'linear',
  });
  // Wxh = 1, Whh = 0.5, bh = 0, Why = 2, by = 0.1
  network.weights.set([1, 0.5, 0, 2, 0.1]);
  return network;
}

function step(network: RecurrentNetwork, value: number): number {
  const out = new Float32Array(network.outputSize);
  network.forward(Float32Array.of(value), out);
  return out[0] as number;
}

describe('RecurrentNetwork', () => {
  it('depends on history: the same input gives different outputs over time', () => {
    const network = handBuilt();
    expect(step(network, 1)).toBeCloseTo(2.1, 6); // h = 1, out = 2·1 + 0.1
    expect(step(network, 1)).toBeCloseTo(3.1, 6); // h = 1 + 0.5·1 = 1.5, out = 2·1.5 + 0.1
  });

  it('reset forgets the history but keeps the weights', () => {
    const network = handBuilt();
    step(network, 1);
    step(network, 1);
    network.reset();
    expect(step(network, 1)).toBeCloseTo(2.1, 6);
    expect(Array.from(network.weights)).toEqual([1, 0.5, 0, 2, 0.1].map(Math.fround));
  });

  it('includes the hidden state in the serialized state', () => {
    const network = handBuilt();
    step(network, 1);
    const state = network.serialize();
    expect(state.length).toBe(network.stateSize);
    expect(network.stateSize).toBe(5 + 1);
    expect(state[5]).toBeCloseTo(1, 6);
  });

  it('restores weights and history so a sequence continues identically', () => {
    const a = new RecurrentNetwork({
      inputSize: 2,
      hiddenSize: 3,
      outputSize: 2,
      rng: createSeededRng(1),
    });
    const b = new RecurrentNetwork({
      inputSize: 2,
      hiddenSize: 3,
      outputSize: 2,
      rng: createSeededRng(77),
    });
    const input = Float32Array.of(0.4, -0.2);
    a.forward(input, new Float32Array(2));
    a.forward(input, new Float32Array(2));
    b.restore(a.serialize());
    const outA = new Float32Array(2);
    const outB = new Float32Array(2);
    a.forward(input, outA);
    b.forward(input, outB);
    expect(Array.from(outB)).toEqual(Array.from(outA));
  });

  it('is deterministic for a seed and differs between seeds', () => {
    const make = (seed: number) =>
      new RecurrentNetwork({ ...base, hiddenSize: 4, rng: createSeededRng(seed) }).serialize();
    expect(Array.from(make(5))).toEqual(Array.from(make(5)));
    expect(Array.from(make(5))).not.toEqual(Array.from(make(6)));
  });

  it('keeps tanh hidden state within [-1, 1] and outputs finite (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.float({ min: -50, max: 50, noNaN: true }), { minLength: 1, maxLength: 30 }),
        (sequence) => {
          const network = new RecurrentNetwork({
            inputSize: 1,
            hiddenSize: 4,
            outputSize: 1,
            rng: createSeededRng(3),
          });
          for (const value of sequence) {
            if (!Number.isFinite(step(network, value))) return false;
          }
          const hidden = network.serialize().slice(network.weights.length);
          return hidden.every((h) => Math.abs(h) <= 1);
        },
      ),
    );
  });

  it('does not modify its input', () => {
    const network = new RecurrentNetwork({ ...base, rng: createSeededRng(1) });
    const input = Float32Array.of(0.3);
    network.forward(input, new Float32Array(1));
    expect(input[0]).toBe(Math.fround(0.3));
  });

  it('validates buffer sizes and values', () => {
    const network = new RecurrentNetwork({ ...base, rng: createSeededRng(1) });
    expect(() => network.forward(new Float32Array(2), new Float32Array(1))).toThrow(
      SizeMismatchError,
    );
    expect(() => network.forward(new Float32Array(1), new Float32Array(2))).toThrow(
      SizeMismatchError,
    );
    expect(() => network.forward(Float32Array.of(NaN), new Float32Array(1))).toThrow(
      InvalidParameterError,
    );
    expect(() => network.restore(new Float32Array(2))).toThrow(SizeMismatchError);
    expect(() => network.restore(new Float32Array(network.stateSize).fill(Infinity))).toThrow(
      InvalidParameterError,
    );
  });

  it.each([
    ['inputSize', { ...base, inputSize: 0 }],
    ['hiddenSize', { ...base, hiddenSize: 0 }],
    ['outputSize', { ...base, outputSize: 1.5 }],
    ['hiddenActivation', { ...base, hiddenActivation: 'swish' as never }],
    ['outputActivation', { ...base, outputActivation: 'swish' as never }],
  ])('rejects invalid %s', (name, options) => {
    expect(() => new RecurrentNetwork({ rng: createSeededRng(1), ...options })).toThrow(
      new RegExp(name),
    );
  });

  it('throws NumericInstabilityError instead of keeping an infinite hidden state (relu)', () => {
    const network = new RecurrentNetwork({
      inputSize: 1,
      hiddenSize: 1,
      outputSize: 1,
      rng: createSeededRng(1),
      hiddenActivation: 'relu',
    });
    network.weights.set([5, 5, 0, 1, 0]); // each step multiplies the hidden state by 5
    const out = new Float32Array(1);
    let thrown: unknown;
    let lastFiniteState = network.serialize();
    for (let step = 0; step < 200 && thrown === undefined; step += 1) {
      try {
        network.forward(Float32Array.of(1), out);
        lastFiniteState = network.serialize();
      } catch (error) {
        thrown = error;
      }
    }
    expect(thrown).toBeInstanceOf(NumericInstabilityError);
    expect(String((thrown as Error).message)).toMatch(/hidden state.*reset/);
    expect(Array.from(network.serialize())).toEqual(Array.from(lastFiniteState));
    expect(network.serialize().every((v) => Number.isFinite(v))).toBe(true);
    network.reset();
    expect(() => network.forward(Float32Array.of(0), out)).not.toThrow();
  });

  it('throws NumericInstabilityError when the output overflows and leaves the history untouched', () => {
    const network = new RecurrentNetwork({
      inputSize: 1,
      hiddenSize: 1,
      outputSize: 1,
      rng: createSeededRng(1),
      hiddenActivation: 'linear',
    });
    network.weights.set([1, 0, 0, 1e30, 0]); // hidden = input, output = 1e30 * hidden
    const before = Array.from(network.serialize());
    expect(() => network.forward(Float32Array.of(1e30), new Float32Array(1))).toThrow(/output/);
    expect(Array.from(network.serialize())).toEqual(before);
  });

  it('rejects sizes too large to allocate', () => {
    expect(
      () => new RecurrentNetwork({ ...base, hiddenSize: 1e10, rng: createSeededRng(1) }),
    ).toThrow(InvalidParameterError);
  });
});
