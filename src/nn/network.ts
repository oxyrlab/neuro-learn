/**
 * Common shape of the small neural networks in this package. Networks only run forward;
 * their weights are changed from outside (evolution/mutation, or a learner such as `mlp-sgd`).
 *
 * All methods allocate nothing except `serialize`, which returns an independent copy.
 */
export interface Network {
  /** Number of inputs expected by `forward`. */
  readonly inputSize: number;
  /** Number of outputs written by `forward`. */
  readonly outputSize: number;
  /**
   * Live view of the trainable weights (not a copy): writing to it changes the network.
   * Intended for mutation and crossover.
   */
  readonly weights: Float32Array;
  /** Length of the serialized state (weights plus any recurrent state). */
  readonly stateSize: number;

  /**
   * Runs the network on one input.
   *
   * @param input - Input of length `inputSize`, finite values.
   * @param out - Output buffer of length `outputSize`, overwritten.
   */
  forward(input: Float32Array, out: Float32Array): void;

  /** Returns an independent copy of the full state (length `stateSize`). */
  serialize(): Float32Array;

  /**
   * Replaces the full state with a copy of `state`.
   *
   * @param state - Buffer of length `stateSize` containing finite values.
   */
  restore(state: Float32Array): void;
}
