/** Shared vocabulary of the benchmark: what is compared, on which problems, and how a subject looks. */

export type Kind =
  'mlp-forward' | 'population-forward' | 'mlp-train' | 'delta-rule' | 'rnn-forward';

export interface Scenario {
  readonly id: string;
  readonly title: string;
  readonly kind: Kind;
  /** MLP: layer sizes. delta-rule: [inputs, outputs]. rnn-forward: [inputs, hidden, outputs]. */
  readonly layers: readonly number[];
  /** Independent samples (or, for population-forward, creatures) processed by one call of `Subject.run`. */
  readonly batch: number;
  /** SGD learning rate (training kinds only). */
  readonly rate: number;
}

export const SCENARIOS: readonly Scenario[] = [
  {
    id: 'mlp-forward-small',
    title: 'MLP forward [16,16,4], 1 sample per call',
    kind: 'mlp-forward',
    layers: [16, 16, 4],
    batch: 1,
    rate: 0,
  },
  {
    id: 'mlp-forward-large',
    title: 'MLP forward [64,128,64,8], 1 sample per call',
    kind: 'mlp-forward',
    layers: [64, 128, 64, 8],
    batch: 1,
    rate: 0,
  },
  {
    id: 'mlp-forward-batch',
    title: 'MLP forward [64,128,64,8], 256 samples per call',
    kind: 'mlp-forward',
    layers: [64, 128, 64, 8],
    batch: 256,
    rate: 0,
  },
  {
    id: 'population-forward',
    title: 'Population: 500 creatures, each with its own MLP [16,16,4] (one tick)',
    kind: 'population-forward',
    layers: [16, 16, 4],
    batch: 500,
    rate: 0,
  },
  {
    id: 'mlp-train-small',
    title: 'MLP online SGD step [16,16,4]',
    kind: 'mlp-train',
    layers: [16, 16, 4],
    batch: 1,
    rate: 0.05,
  },
  {
    id: 'mlp-train-large',
    title: 'MLP online SGD step [64,128,64,8]',
    kind: 'mlp-train',
    layers: [64, 128, 64, 8],
    batch: 1,
    rate: 0.01,
  },
  {
    id: 'delta-rule-small',
    title: 'Delta rule (Rescorla-Wagner) 16 to 4',
    kind: 'delta-rule',
    layers: [16, 4],
    batch: 1,
    rate: 0.1,
  },
  {
    id: 'delta-rule-large',
    title: 'Delta rule (Rescorla-Wagner) 256 to 64',
    kind: 'delta-rule',
    layers: [256, 64],
    batch: 1,
    rate: 0.01,
  },
  {
    id: 'rnn-forward-small',
    title: 'Elman RNN step 16-16-4',
    kind: 'rnn-forward',
    layers: [16, 16, 4],
    batch: 1,
    rate: 0,
  },
  {
    id: 'rnn-forward-large',
    title: 'Elman RNN step 64-128-8',
    kind: 'rnn-forward',
    layers: [64, 128, 8],
    batch: 1,
    rate: 0,
  },
];

export type TfBackend = 'cpu' | 'wasm' | 'node';

export interface ImplSpec {
  readonly id: string;
  readonly family: 'neuro' | 'tfjs';
  readonly backend?: TfBackend;
  /** `layers` = tf.layers model API; `ops` = low-level tf ops (matMul, add, tanh, autograd). */
  readonly api?: 'layers' | 'ops';
}

const TF_BACKENDS: readonly TfBackend[] = ['cpu', 'wasm', 'node'];
export const IMPLS: readonly ImplSpec[] = [
  { id: 'neuro-learn', family: 'neuro' },
  ...TF_BACKENDS.flatMap((backend) =>
    (['layers', 'ops'] as const).map((api) => ({
      id: `tfjs-${backend}-${api}`,
      family: 'tfjs' as const,
      backend,
      api,
    })),
  ),
];

/** One configured model under test. `run` must do exactly the work a simulation would do per call. */
export interface Subject {
  /** Performs one call. For forward kinds the results land in `output`. */
  run(call: number): void;
  readonly output: Float32Array;
  /** Current weights in the neuro-learn layout (see lib/layout.ts); used for accuracy checks. */
  weights(): Float32Array;
  /** Live tensor count of the framework, if it tracks one (leak check). */
  tensorCount?(): number;
}

/** Number of distinct samples and input blocks the calls cycle through. Powers of two. */
export const POOL = 256;
export const BLOCKS = 8;
