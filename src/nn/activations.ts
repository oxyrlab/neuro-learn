import { InvalidParameterError } from '../core/errors';

/** Identifier of a built-in activation function. */
export type ActivationId = 'tanh' | 'relu' | 'sigmoid' | 'linear';

/** An activation function together with its derivative expressed through its own output. */
export interface Activation {
  /** Identifier, e.g. `"tanh"`. */
  readonly id: ActivationId;
  /** Computes `f(x)`. */
  apply(x: number): number;
  /** Computes `f'(x)` given `y = f(x)`; this is all backpropagation needs. */
  derivativeFromOutput(y: number): number;
  /**
   * Applies `f` to every element of `values`, in place. Hot paths use this instead of the scalar
   * `apply`: a per-element call that returns a fractional number makes V8 allocate a boxed number
   * each time, whereas a loop inside one call allocates nothing.
   */
  applyInPlace(values: Float32Array): void;
  /**
   * Multiplies `deltas[i]` by `f'` computed from `outputs[i] = f(x)`, in place (backpropagation
   * through this activation). Allocation-free for the same reason as `applyInPlace`.
   */
  scaleByDerivative(outputs: Float32Array, deltas: Float32Array): void;
}

/**
 * Registry of built-in activations, keyed by id.
 * Output ranges: `tanh` (-1, 1), `sigmoid` (0, 1), `relu` [0, ∞), `linear` (-∞, ∞).
 */
export const ACTIVATIONS: Readonly<Record<ActivationId, Activation>> = {
  tanh: {
    id: 'tanh',
    apply: (x) => Math.tanh(x),
    derivativeFromOutput: (y) => 1 - y * y,
    applyInPlace(values) {
      for (let index = 0; index < values.length; index += 1) {
        values[index] = Math.tanh(values[index] as number);
      }
    },
    scaleByDerivative(outputs, deltas) {
      for (let index = 0; index < deltas.length; index += 1) {
        const y = outputs[index] as number;
        deltas[index] = (deltas[index] as number) * (1 - y * y);
      }
    },
  },
  sigmoid: {
    id: 'sigmoid',
    apply: (x) => 1 / (1 + Math.exp(-x)),
    derivativeFromOutput: (y) => y * (1 - y),
    applyInPlace(values) {
      for (let index = 0; index < values.length; index += 1) {
        values[index] = 1 / (1 + Math.exp(-(values[index] as number)));
      }
    },
    scaleByDerivative(outputs, deltas) {
      for (let index = 0; index < deltas.length; index += 1) {
        const y = outputs[index] as number;
        deltas[index] = (deltas[index] as number) * (y * (1 - y));
      }
    },
  },
  relu: {
    id: 'relu',
    apply: (x) => (x > 0 ? x : 0),
    derivativeFromOutput: (y) => (y > 0 ? 1 : 0),
    applyInPlace(values) {
      for (let index = 0; index < values.length; index += 1) {
        if ((values[index] as number) < 0) values[index] = 0;
      }
    },
    scaleByDerivative(outputs, deltas) {
      for (let index = 0; index < deltas.length; index += 1) {
        if ((outputs[index] as number) <= 0) deltas[index] = 0;
      }
    },
  },
  linear: {
    id: 'linear',
    apply: (x) => x,
    derivativeFromOutput: () => 1,
    applyInPlace() {
      // Identity: nothing to do.
    },
    scaleByDerivative() {
      // Derivative is 1: deltas are unchanged.
    },
  },
};

/**
 * Looks up an activation by id.
 *
 * @param id - One of `"tanh" | "relu" | "sigmoid" | "linear"`.
 * @param parameter - Parameter name used in the error message. Default `"activation"`.
 * @throws InvalidParameterError if `id` is not a built-in activation.
 */
export function getActivation(id: string, parameter = 'activation'): Activation {
  if (!Object.hasOwn(ACTIVATIONS, id)) {
    throw new InvalidParameterError(
      parameter,
      id,
      `one of ${Object.keys(ACTIVATIONS)
        .map((key) => `"${key}"`)
        .join(', ')}`,
    );
  }
  return ACTIVATIONS[id as ActivationId];
}
