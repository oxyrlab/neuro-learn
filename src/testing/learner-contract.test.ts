import {
  assertFiniteBuffer,
  assertFiniteInRange,
  assertNonNegative,
  assertSize,
} from '../core/validation';
import type { Learner } from '../learning/learner';
import { runLearnerContract } from './learner-contract';

/** Minimal diagonal learner used only to prove the contract is satisfiable. */
class DiagonalLearner implements Learner {
  public readonly inputSize = 3;
  public readonly outputSize = 3;
  public readonly stateSize = 3;
  private readonly weights = new Float32Array(3);

  public predict(features: Float32Array, out: Float32Array): void {
    assertSize('features', features, 3);
    assertSize('out', out, 3);
    assertFiniteBuffer('features', features);
    for (let i = 0; i < 3; i += 1) out[i] = (this.weights[i] as number) * (features[i] as number);
  }

  public learn(features: Float32Array, target: Float32Array, rate: number): void {
    assertSize('features', features, 3);
    assertSize('target', target, 3);
    assertFiniteBuffer('features', features);
    assertFiniteBuffer('target', target);
    assertFiniteInRange('rate', rate, 0, 1);
    for (let i = 0; i < 3; i += 1) {
      const error = (target[i] as number) - (this.weights[i] as number) * (features[i] as number);
      this.weights[i] = Math.max(
        -1,
        Math.min(1, (this.weights[i] as number) + rate * (features[i] as number) * error),
      );
    }
  }

  public decay(dt: number): void {
    assertNonNegative('dt', dt);
    const factor = Math.exp(-0.5 * dt);
    for (let i = 0; i < 3; i += 1) this.weights[i] = (this.weights[i] as number) * factor;
  }

  public serialize(): Float32Array {
    return this.weights.slice();
  }

  public restore(state: Float32Array): void {
    assertSize('state', state, 3);
    assertFiniteBuffer('state', state);
    this.weights.set(state);
  }
}

runLearnerContract('diagonal reference learner', () => new DiagonalLearner(), { maxAbsState: 1 });
runLearnerContract(
  'diagonal reference learner (custom seed and steps)',
  () => new DiagonalLearner(),
  {
    seed: 5,
    steps: 20,
  },
);
