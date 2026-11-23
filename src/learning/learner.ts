/**
 * Contract shared by every learner: a fixed-size `Float32Array` state that can predict,
 * learn from one example, forget over time, and be snapshotted.
 *
 * Conventions:
 * - Feature values are finite and normalized by the caller, typically to [0, 1]. A learner may add
 *   constraints (`eligibility-trace` requires features >= 0); see its docs.
 * - `predict`, `learn` and `decay` allocate nothing; `serialize` returns an independent copy.
 * - Buffers with the wrong length throw `SizeMismatchError`; non-finite values or out-of-range
 *   scalars throw `InvalidParameterError`.
 * - A new instance always starts with empty (all-zero) memory.
 */
export interface Learner {
  /** Number of input features expected by `predict` and `learn`. */
  readonly inputSize: number;
  /** Number of outputs written by `predict` and expected as `target` in `learn`. */
  readonly outputSize: number;
  /** Length of the serialized state; fixed at construction. */
  readonly stateSize: number;

  /**
   * Computes the current prediction without changing state.
   *
   * @param features - Input of length `inputSize`.
   * @param out - Output buffer of length `outputSize`, overwritten.
   */
  predict(features: Float32Array, out: Float32Array): void;

  /**
   * Updates state from one example.
   *
   * @param features - Input of length `inputSize`.
   * @param target - Desired outcome of length `outputSize`.
   * @param rate - Effective learning rate for this call, in [0, 1]; callers scale it by a modulator.
   */
  learn(features: Float32Array, target: Float32Array, rate: number): void;

  /**
   * Forgets: moves state toward zero for an elapsed time.
   *
   * @param dt - Elapsed time in seconds, finite and >= 0.
   */
  decay(dt: number): void;

  /** Returns an independent copy of the full state (length `stateSize`). */
  serialize(): Float32Array;

  /**
   * Replaces the full state with a copy of `state`.
   *
   * @param state - Buffer of length `stateSize` containing finite values that respect the
   *   learner's own bounds (e.g. weights within `maxStrength`); anything this learner could not
   *   have produced is rejected with `InvalidParameterError`.
   */
  restore(state: Float32Array): void;
}
