/**
 * Evolving a recurrent network through its live `weights` view, the way EcoSim's genome will:
 * a (1+1) hill climber mutates weights and keeps the child if it is no worse. Task: output the
 * previous input (a one-step memory), which a memoryless network cannot solve.
 */
import { describe, expect, it } from 'vitest';
import { createSeededRng, RecurrentNetwork } from '../../src';
import type { Rng } from '../../src';

const SEQUENCE_COUNT = 6;
const SEQUENCE_LENGTH = 20;
const GENERATIONS = 3000;
const MUTATION_SIZE = 0.15;

function gaussian(rng: Rng): number {
  const u = Math.max(rng.next(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng.next());
}

function makeSequences(seed: number): number[][] {
  const rng = createSeededRng(seed);
  return Array.from({ length: SEQUENCE_COUNT }, () =>
    Array.from({ length: SEQUENCE_LENGTH }, () => (rng.next() < 0.5 ? 0 : 1)),
  );
}

/** Mean squared error of "output = previous input" over all sequences (lower is better). */
function error(network: RecurrentNetwork, sequences: number[][]): number {
  const out = new Float32Array(1);
  let total = 0;
  let count = 0;
  for (const sequence of sequences) {
    network.reset();
    let previous = 0;
    for (const value of sequence) {
      network.forward(Float32Array.of(value), out);
      total += ((out[0] as number) - previous) ** 2;
      count += 1;
      previous = value;
    }
  }
  return total / count;
}

describe('recurrent network evolution', () => {
  it('a hill climber evolves a one-step memory from random weights', () => {
    const rng = createSeededRng(42);
    const sequences = makeSequences(7);
    let best = new RecurrentNetwork({ inputSize: 1, hiddenSize: 4, outputSize: 1, rng });
    let bestError = error(best, sequences);
    const startingError = bestError;
    for (let generation = 0; generation < GENERATIONS; generation += 1) {
      const child = new RecurrentNetwork({ inputSize: 1, hiddenSize: 4, outputSize: 1, rng });
      child.restore(best.serialize());
      for (let i = 0; i < child.weights.length; i += 1) {
        child.weights[i] = (child.weights[i] as number) + MUTATION_SIZE * gaussian(rng);
      }
      child.reset();
      const childError = error(child, sequences);
      if (childError <= bestError) {
        best = child;
        bestError = childError;
      }
    }
    expect(bestError).toBeLessThan(startingError);
    expect(bestError).toBeLessThan(0.05);

    // Generalizes to sequences never seen during evolution.
    expect(error(best, makeSequences(99))).toBeLessThan(0.1);
  });
});
