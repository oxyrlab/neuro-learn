/**
 * Throughput of the hot-path operations (`npm run bench`). Prints nanoseconds per call and calls
 * per second. Numbers are machine-specific; they are recorded in docs/performance.md for
 * orientation and are not pass/fail thresholds.
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

const INPUTS = 16;
const OUTPUTS = 4;
const WARMUP_CALLS = 200_000;
const MEASURED_CALLS = 2_000_000;
const REPEATS = 5;

const features = new Float32Array(INPUTS).fill(0.5);
const target = new Float32Array(OUTPUTS).fill(0.25);
const out = new Float32Array(OUTPUTS);

/** Best (lowest) nanoseconds per call over several repeats, which filters scheduler noise. */
function nanosecondsPerCall(operation: () => void): number {
  for (let call = 0; call < WARMUP_CALLS; call += 1) operation();
  let best = Number.POSITIVE_INFINITY;
  for (let repeat = 0; repeat < REPEATS; repeat += 1) {
    const start = process.hrtime.bigint();
    for (let call = 0; call < MEASURED_CALLS; call += 1) operation();
    best = Math.min(best, Number(process.hrtime.bigint() - start) / MEASURED_CALLS);
  }
  return best;
}

const rows: Array<[string, number]> = [];
const learners: Array<[string, Learner]> = [
  ['rescorla-wagner 16→4', new RescorlaWagnerLearner({ inputSize: INPUTS, outputSize: OUTPUTS })],
  [
    'eligibility-trace 16→4',
    new EligibilityTraceLearner({
      inputSize: INPUTS,
      outputSize: OUTPUTS,
      traceTimeConstantSeconds: 2,
    }),
  ],
  ['hebbian 16→4', new HebbianLearner({ inputSize: INPUTS, outputSize: OUTPUTS })],
  [
    'mlp-sgd [16,16,4]',
    new MlpSgdLearner({ layerSizes: [INPUTS, 16, OUTPUTS], rng: createSeededRng(1) }),
  ],
];
for (const [name, learner] of learners) {
  rows.push([`${name} predict`, nanosecondsPerCall(() => learner.predict(features, out))]);
  rows.push([`${name} learn`, nanosecondsPerCall(() => learner.learn(features, target, 0.1))]);
  rows.push([`${name} decay`, nanosecondsPerCall(() => learner.decay(0.1))]);
}
const feedforward = new FeedforwardNetwork({
  layerSizes: [INPUTS, 16, OUTPUTS],
  rng: createSeededRng(1),
});
const recurrent = new RecurrentNetwork({
  inputSize: INPUTS,
  hiddenSize: 16,
  outputSize: OUTPUTS,
  rng: createSeededRng(1),
});
rows.push([
  'feedforward [16,16,4] forward',
  nanosecondsPerCall(() => feedforward.forward(features, out)),
]);
rows.push([
  'recurrent 16-16-4 forward',
  nanosecondsPerCall(() => recurrent.forward(features, out)),
]);

console.log(`Node ${process.version}, best of ${REPEATS} × ${MEASURED_CALLS} calls\n`);
console.log('| Operation | ns/call | calls/s |\n|---|---:|---:|');
for (const [name, nanoseconds] of rows) {
  console.log(
    `| ${name} | ${nanoseconds.toFixed(1)} | ${Math.round(1e9 / nanoseconds).toLocaleString('en-US')} |`,
  );
}
