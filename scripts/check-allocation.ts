/**
 * Allocation check (`npm run check:allocation`, needs `--expose-gc`, set by the npm script).
 * For each hot-path operation it forces a collection, runs a short window of calls and reads how
 * much `heapUsed` grew, repeated over many windows. The window is short enough that no
 * collection happens inside it, so growth / calls is the allocation per call. The median window
 * must stay at or below `MAX_BYTES_PER_CALL` (noise from reading `heapUsed` itself is far below it).
 * A deliberately allocating control operation must exceed `MIN_CONTROL_BYTES_PER_CALL`, which
 * proves the detector is not blind.
 */
import {
  createSeededRng,
  EligibilityTraceLearner,
  FeedforwardNetwork,
  HebbianLearner,
  MlpSgdLearner,
  RecurrentNetwork,
  RescorlaWagnerLearner,
} from '../src';
import type { Learner } from '../src';

const WINDOWS = 50;
const CALLS_PER_WINDOW = 4000;
const WARMUP_CALLS = 50_000;
const MAX_BYTES_PER_CALL = 2;
const MIN_CONTROL_BYTES_PER_CALL = 50;

declare const gc: (() => void) | undefined;
if (typeof gc !== 'function') {
  console.error('Run with --expose-gc (use `npm run check:allocation`).');
  process.exit(1);
}

const features = new Float32Array(8).fill(0.5);
const target = new Float32Array(2).fill(0.25);
const out = new Float32Array(2);

const operations: Array<[string, () => void]> = [];
/**
 * Each learner is checked with an integer bound (V8 stores it as a small integer) and a
 * fractional bound (stored as a double): the two take different code paths in the engine, and
 * an earlier version allocated only for the integer case.
 */
function createLearners(
  bound: number,
): Array<[string, Learner & { observe?: (f: Float32Array) => void }]> {
  const label = `bound ${bound}`;
  return [
    [
      `rescorla-wagner (${label})`,
      new RescorlaWagnerLearner({
        inputSize: 8,
        outputSize: 2,
        maxStrength: bound,
        forgetRatePerSecond: 0.1,
      }),
    ],
    [
      `eligibility-trace (${label})`,
      new EligibilityTraceLearner({
        inputSize: 8,
        outputSize: 2,
        traceTimeConstantSeconds: 2,
        maxStrength: bound,
        forgetRatePerSecond: 0.1,
      }),
    ],
    [
      `hebbian (${label})`,
      new HebbianLearner({
        inputSize: 8,
        outputSize: 2,
        maxWeight: bound,
        forgetRatePerSecond: 0.1,
      }),
    ],
    [
      `mlp-sgd (${label})`,
      new MlpSgdLearner({
        layerSizes: [8, 8, 2],
        rng: createSeededRng(1),
        maxWeight: bound,
        forgetRatePerSecond: 0.1,
      }),
    ],
  ];
}
const learners = [...createLearners(1), ...createLearners(0.9)];
for (const [name, learner] of learners) {
  operations.push([`${name}.predict`, () => learner.predict(features, out)]);
  operations.push([`${name}.learn`, () => learner.learn(features, target, 0.1)]);
  operations.push([`${name}.decay`, () => learner.decay(0.1)]);
  if (learner.observe) operations.push([`${name}.observe`, () => learner.observe?.(features)]);
}
const feedforward = new FeedforwardNetwork({ layerSizes: [8, 8, 2], rng: createSeededRng(1) });
const recurrent = new RecurrentNetwork({
  inputSize: 8,
  hiddenSize: 8,
  outputSize: 2,
  rng: createSeededRng(1),
});
operations.push(['feedforward.forward', () => feedforward.forward(features, out)]);
operations.push(['recurrent.forward', () => recurrent.forward(features, out)]);

/**
 * Median bytes allocated per call over several short windows. The median ignores the occasional
 * window in which the engine maps a fresh memory page (a ~1 MB jump unrelated to the operation);
 * genuine per-call allocation shows up in every window.
 */
function medianBytesPerCall(operation: () => void): number {
  for (let warmup = 0; warmup < WARMUP_CALLS; warmup += 1) operation();
  const samples: number[] = [];
  for (let window = 0; window < WINDOWS; window += 1) {
    gc?.();
    const before = process.memoryUsage().heapUsed;
    for (let call = 0; call < CALLS_PER_WINDOW; call += 1) operation();
    const growth = process.memoryUsage().heapUsed - before;
    samples.push(growth / CALLS_PER_WINDOW);
  }
  samples.sort((a, b) => a - b);
  return samples[Math.floor(samples.length / 2)] as number;
}

// Control: allocates a typed array per call, so the detector must see it.
let sink = 0;
const control = medianBytesPerCall(() => {
  sink += new Float32Array(16).length;
});
if (control < MIN_CONTROL_BYTES_PER_CALL || sink === 0) {
  console.error(
    `FAIL control: saw ${control.toFixed(1)} B/call, expected at least ${MIN_CONTROL_BYTES_PER_CALL}; the detector is blind.`,
  );
  process.exit(1);
}
console.log(`ok   control (allocating): ${control.toFixed(1)} B/call, detector works`);

let failed = false;
for (const [name, operation] of operations) {
  const bytesPerCall = medianBytesPerCall(operation);
  const ok = bytesPerCall <= MAX_BYTES_PER_CALL;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: median ${bytesPerCall.toFixed(2)} B/call`);
  if (!ok) failed = true;
}
process.exit(failed ? 1 : 0);
