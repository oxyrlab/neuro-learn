/** Renders the collected records as Markdown tables. */
import { IMPLS, SCENARIOS } from './lib/spec';
import type { AccuracyResult } from './lib/accuracy';
import type { MeasureResult } from './lib/measure';

export interface RunRecord {
  readonly impl: string;
  readonly scenario: string;
  readonly mode: 'perf' | 'accuracy';
  readonly load?: { ms: number; rssMb: number; heapMb: number; externalMb: number };
  readonly perf?: MeasureResult;
  readonly peakRssMb?: number;
  readonly accuracy?: AccuracyResult;
}

export interface Environment {
  readonly [key: string]: unknown;
  readonly footprintMb: Readonly<Record<string, number>>;
}

const BASELINE = 'neuro-learn';

function formatTime(ns: number): string {
  if (ns < 1000) return `${ns.toFixed(0)} ns`;
  if (ns < 1e6) return `${(ns / 1000).toFixed(ns < 1e4 ? 2 : 1)} µs`;
  return `${(ns / 1e6).toFixed(2)} ms`;
}

function formatRatio(ratio: number): string {
  if (ratio >= 1.05) return `${ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1)}× slower`;
  if (ratio <= 0.95)
    return `${1 / ratio >= 10 ? (1 / ratio).toFixed(0) : (1 / ratio).toFixed(1)}× faster`;
  return 'same';
}

const scientific = (value: number): string => (value === 0 ? '0' : value.toExponential(1));
const find = (
  records: readonly RunRecord[],
  impl: string,
  scenario: string,
  mode: RunRecord['mode'],
): RunRecord | undefined =>
  records.find((r) => r.impl === impl && r.scenario === scenario && r.mode === mode);

function table(header: string[], rows: string[][]): string {
  return [
    `| ${header.join(' | ')} |`,
    `|${header.map((_, i) => (i === 0 ? '---' : '---:')).join('|')}|`,
    ...rows.map((r) => `| ${r.join(' | ')} |`),
  ].join('\n');
}

function speedSection(records: readonly RunRecord[], scenario: string): string {
  const base = find(records, BASELINE, scenario, 'perf')?.perf;
  const rows = IMPLS.flatMap((impl) => {
    const perf = find(records, impl.id, scenario, 'perf')?.perf;
    if (!perf) return [[impl.id, 'failed', '', '', '', '']];
    return [
      [
        impl.id === BASELINE ? `**${impl.id}**` : impl.id,
        formatTime(perf.nsPerSample.median),
        Math.round(1e9 / perf.nsPerSample.median).toLocaleString('en-US'),
        impl.id === BASELINE || !base
          ? 'baseline'
          : formatRatio(perf.nsPerSample.median / base.nsPerSample.median),
        `${(perf.nsPerSample.cv * 100).toFixed(1)}%`,
        perf.cpuUtilization.toFixed(2),
      ],
    ];
  });
  return table(
    [
      'Implementation',
      'Time / sample (median)',
      'Samples / s',
      'vs neuro-learn',
      'Noise (CV)',
      'CPU cores used',
    ],
    rows,
  );
}

function resourceSection(records: readonly RunRecord[], scenario: string, batch: number): string {
  const rows = IMPLS.map((impl) => {
    const record = find(records, impl.id, scenario, 'perf');
    const perf = record?.perf;
    if (!record || !perf || !record.load) return [impl.id, 'failed', '', '', '', '', '', ''];
    return [
      impl.id === BASELINE ? `**${impl.id}**` : impl.id,
      `${perf.heapBytesPerCall.toFixed(0)} B`,
      // gcPauseMsPer1MCalls is numerically the pause in ns per call.
      `${((perf.gcPauseMsPer1MCalls / (perf.nsPerSample.median * batch)) * 100).toFixed(1)}% (${(perf.gcCountPer1MCalls / 1000).toFixed(1)} per 1k calls)`,
      formatTime(perf.cpuNsPerSample),
      `${record.peakRssMb?.toFixed(0)} MB`,
      `${record.load.rssMb.toFixed(0)} MB / ${record.load.ms.toFixed(0)} ms`,
      `${perf.rssGrowthMb.toFixed(1)} MB`,
      perf.tensorGrowth === null ? 'n/a' : String(perf.tensorGrowth),
    ];
  });
  return table(
    [
      'Implementation',
      'JS heap / call',
      'GC share of time',
      'CPU time / sample',
      'Peak RSS',
      'Load: RSS / time',
      'RSS growth in run',
      'Tensor leak',
    ],
    rows,
  );
}

function accuracySection(records: readonly RunRecord[]): string {
  const ran = (id: string): boolean => records.some((r) => r.scenario === id);
  const forward = SCENARIOS.filter((sc) => sc.kind.endsWith('forward') && ran(sc.id));
  const forwardRows = IMPLS.map((impl) => [
    impl.id,
    ...forward.map((sc) => {
      const accuracy = find(records, impl.id, sc.id, 'accuracy')?.accuracy;
      return accuracy?.maxAbsError === undefined
        ? 'failed'
        : `${scientific(accuracy.maxAbsError)}${accuracy.finite ? '' : ' (NaN!)'}`;
    }),
  ]);
  const training = SCENARIOS.filter((sc) => !sc.kind.endsWith('forward') && ran(sc.id));
  const trainingRows = IMPLS.flatMap((impl) =>
    training.map((sc) => {
      const accuracy = find(records, impl.id, sc.id, 'accuracy')?.accuracy;
      const last = accuracy?.checkpoints?.at(-1);
      const first = accuracy?.checkpoints?.[0];
      if (!accuracy || !last || !first) return [impl.id, sc.id, 'failed', '', '', ''];
      return [
        impl.id,
        sc.id,
        first.loss.toFixed(4),
        last.loss.toFixed(6),
        last.referenceLoss.toFixed(6),
        scientific(accuracy.maxWeightDifference ?? NaN),
      ];
    }),
  );
  return [
    '### Forward pass: largest absolute error against the float64 reference',
    '',
    table(['Implementation', ...forward.map((sc) => sc.id)], forwardRows),
    '',
    '### Training: held-out loss after 2000 online SGD steps from identical initial weights and sample order',
    '',
    'The reference column is a float64 re-implementation trained on the same steps. "Max weight diff" is against that reference.',
    '',
    table(
      [
        'Implementation',
        'Scenario',
        'Initial loss',
        'Final loss',
        'Reference final loss',
        'Max weight diff',
      ],
      trainingRows,
    ),
  ].join('\n');
}

export function renderReport(env: Environment, records: readonly RunRecord[]): string {
  const lines: string[] = ['# neuro-learn vs TensorFlow.js benchmark', '', '## Environment', ''];
  for (const [key, value] of Object.entries(env)) {
    if (key !== 'footprintMb') lines.push(`- ${key}: ${String(value)}`);
  }
  lines.push(
    '',
    '## Install footprint (on disk)',
    '',
    table(
      ['Package', 'Size'],
      Object.entries(env.footprintMb).map(([k, v]) => [k, `${v.toFixed(1)} MB`]),
    ),
    '',
  );
  lines.push('## Speed and resources per scenario', '');
  for (const scenario of SCENARIOS) {
    if (!records.some((r) => r.scenario === scenario.id)) continue;
    lines.push(
      `### ${scenario.title} (\`${scenario.id}\`)`,
      '',
      speedSection(records, scenario.id),
      '',
      resourceSection(records, scenario.id, scenario.batch),
      '',
    );
  }
  lines.push('## Accuracy', '', accuracySection(records), '');
  return lines.join('\n');
}
