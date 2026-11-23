/**
 * Orchestrator: runs every (implementation x scenario x mode) in its own child process and writes
 * results/<name>.json and results/<name>.md.
 *
 *   npm run bench                         full run
 *   npm run bench -- --quick              short run (smoke test, noisier numbers)
 *   npm run bench -- --pin 2 --name pinned   pin every child to one CPU core (equal-resources run)
 *   npm run bench -- --only mlp-train-small --impl neuro-learn,tfjs-node-ops
 */
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderReport } from './report';
import type { RunRecord } from './report';
import { IMPLS, SCENARIOS } from './lib/spec';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name: string): string | undefined => {
  const index = args.indexOf(`--${name}`);
  return index === -1 ? undefined : (args[index + 1] ?? 'true');
};
const quick = args.includes('--quick');
const pin = flag('pin');
const name = flag('name') ?? (quick ? 'quick' : 'results');
const onlyScenarios = flag('only')?.split(',');
const onlyImpls = flag('impl')?.split(',');

function directorySize(directory: string): number {
  if (!fs.existsSync(directory)) return 0;
  let total = 0;
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    total += entry.isDirectory() ? directorySize(full) : fs.statSync(full).size;
  }
  return total;
}

function runChild(impl: string, scenario: string, mode: 'perf' | 'accuracy'): RunRecord | null {
  const job = JSON.stringify({ impl, scenario, mode, quick });
  const nodeArgs = [
    '--expose-gc',
    '--no-warnings',
    '--import',
    'tsx',
    path.join(here, 'worker.ts'),
    job,
  ];
  const [command, commandArgs] = pin
    ? ['taskset', ['-c', pin, process.execPath, ...nodeArgs]]
    : [process.execPath, nodeArgs];
  const child = spawnSync(command as string, commandArgs as string[], {
    cwd: here,
    encoding: 'utf8',
    timeout: 15 * 60_000,
    maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, TF_CPP_MIN_LOG_LEVEL: '2' },
  });
  const line = child.stdout.split('\n').find((candidate) => candidate.startsWith('@@RESULT@@ '));
  if (!line) {
    console.error(
      `  FAILED ${impl} / ${scenario} / ${mode}: ${(child.stderr || String(child.error)).slice(-400)}`,
    );
    return null;
  }
  return JSON.parse(line.slice('@@RESULT@@ '.length)) as RunRecord;
}

const impls = IMPLS.filter((impl) => !onlyImpls || onlyImpls.includes(impl.id));
const scenarios = SCENARIOS.filter((sc) => !onlyScenarios || onlyScenarios.includes(sc.id));
const records: RunRecord[] = [];
const started = Date.now();
for (const scenario of scenarios) {
  for (const impl of impls) {
    for (const mode of ['perf', 'accuracy'] as const) {
      const record = runChild(impl.id, scenario.id, mode);
      if (record) records.push(record);
      process.stdout.write(
        `${((Date.now() - started) / 60000).toFixed(1)} min  ${scenario.id}  ${impl.id}  ${mode}${record ? '' : '  (failed)'}\n`,
      );
    }
  }
}

const root = path.join(here, '..', '..');
const tfRoot = path.join(here, 'node_modules', '@tensorflow');
const sizeMb = (bytes: number): number => bytes / 1048576;
const lib = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8')) as {
  version: string;
};
const tfVersion = (pkg: string): string =>
  (
    JSON.parse(fs.readFileSync(path.join(tfRoot, pkg, 'package.json'), 'utf8')) as {
      version: string;
    }
  ).version;
const env = {
  date: new Date().toISOString(),
  node: process.version,
  cpu: os.cpus()[0]?.model ?? 'unknown',
  logicalCpus: os.cpus().length,
  memoryGb: os.totalmem() / 1073741824,
  platform: `${os.platform()} ${os.release()}`,
  pinnedCpu: pin ?? null,
  quick,
  neuroLearn: lib.version,
  tfjs: tfVersion('tfjs'),
  tfjsNode: tfVersion('tfjs-node'),
  tfjsWasm: tfVersion('tfjs-backend-wasm'),
  footprintMb: {
    'neuro-learn (dist)': sizeMb(directorySize(path.join(root, 'dist'))),
    'tfjs (core, layers, cpu backend, data, converter)': sizeMb(
      [
        'tfjs',
        'tfjs-core',
        'tfjs-layers',
        'tfjs-converter',
        'tfjs-data',
        'tfjs-backend-cpu',
      ].reduce((sum, pkg) => sum + directorySize(path.join(tfRoot, pkg)), 0),
    ),
    'tfjs-backend-wasm': sizeMb(directorySize(path.join(tfRoot, 'tfjs-backend-wasm'))),
    'tfjs-node (incl. native TensorFlow library)': sizeMb(
      directorySize(path.join(tfRoot, 'tfjs-node')),
    ),
  },
};

fs.mkdirSync(path.join(here, 'results'), { recursive: true });
fs.writeFileSync(
  path.join(here, 'results', `${name}.json`),
  JSON.stringify({ env, records }, null, 2),
);
fs.writeFileSync(path.join(here, 'results', `${name}.md`), renderReport(env, records));
console.log(
  `\nDone in ${((Date.now() - started) / 60000).toFixed(1)} min. Wrote results/${name}.json and results/${name}.md`,
);
