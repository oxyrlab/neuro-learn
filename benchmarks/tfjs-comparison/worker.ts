/**
 * One measurement in a fresh process (so memory and JIT state are never shared between
 * implementations). Invoked by run.ts as: node --expose-gc --import tsx worker.ts '<json>'.
 * Prints a single line `@@RESULT@@ <json>`; everything else on stdout/stderr is library noise.
 */
import { measureAccuracy } from './lib/accuracy';
import { makeDataset, makeInitialWeights } from './lib/dataset';
import { measure, snapshot } from './lib/measure';
import type { ImplSpec } from './lib/spec';
import { IMPLS, SCENARIOS } from './lib/spec';
import { buildSubject } from './lib/subjects';

interface Job {
  readonly impl: string;
  readonly scenario: string;
  readonly mode: 'perf' | 'accuracy';
  readonly quick: boolean;
}

const job = JSON.parse(process.argv[2] as string) as Job;
const impl = IMPLS.find((candidate) => candidate.id === job.impl) as ImplSpec;
const scenario = SCENARIOS.find((candidate) => candidate.id === job.scenario);
if (!scenario) throw new Error(`Unknown scenario ${job.scenario}`);

const data = makeDataset(scenario);
const init = makeInitialWeights(scenario);
const baseline = snapshot();
const loadStart = process.hrtime.bigint();
const subject = await buildSubject(impl, scenario, init, data);
const loadMs = Number(process.hrtime.bigint() - loadStart) / 1e6;
const loaded = snapshot();
const load = {
  ms: loadMs,
  rssMb: loaded.rssMb - baseline.rssMb,
  heapMb: loaded.heapMb - baseline.heapMb,
  externalMb: loaded.externalMb - baseline.externalMb,
};

let payload: object;
if (job.mode === 'accuracy') {
  payload = { accuracy: measureAccuracy(scenario, subject, data, init) };
} else {
  const options = job.quick
    ? { repeatMs: 60, repeats: 5, warmupMs: 150 }
    : { repeatMs: 250, repeats: 9, warmupMs: 600 };
  const result = await measure(subject, scenario.batch, options);
  payload = { load, perf: result, peakRssMb: process.resourceUsage().maxRSS / 1024 };
}
console.log(
  `@@RESULT@@ ${JSON.stringify({ impl: job.impl, scenario: job.scenario, mode: job.mode, ...payload })}`,
);
process.exit(0);
