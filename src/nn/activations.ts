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
  },
  sigmoid: {
    id: 'sigmoid',
    apply: (x) => 1 / (1 + Math.exp(-x)),
    derivativeFromOutput: (y) => y * (1 - y),
  },
  relu: {
    id: 'relu',
    apply: (x) => (x > 0 ? x : 0),
    derivativeFromOutput: (y) => (y > 0 ? 1 : 0),
  },
  linear: {
    id: 'linear',
    apply: (x) => x,
    derivativeFromOutput: () => 1,
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
