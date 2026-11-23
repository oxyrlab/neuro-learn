/**
 * Speed and resource measurement for one subject in one process. Reports time per sample, CPU time,
 * heap allocation per call, garbage-collector work, memory growth and tensor leaks.
 */
import { PerformanceObserver } from 'node:perf_hooks';
import type { Subject } from './spec';

declare const gc: (() => void) | undefined;

export interface MemorySnapshot {
  readonly rssMb: number;
  readonly heapMb: number;
  readonly externalMb: number;
}

export function snapshot(): MemorySnapshot {
  gc?.();
  const usage = process.memoryUsage();
  return {
    rssMb: usage.rss / 1048576,
    heapMb: usage.heapUsed / 1048576,
    externalMb: (usage.external + usage.arrayBuffers) / 1048576,
  };
}

export interface MeasureOptions {
  readonly repeatMs: number;
  readonly repeats: number;
  readonly warmupMs: number;
}

export interface MeasureResult {
  readonly nsPerSample: { median: number; min: number; max: number; cv: number };
  readonly callsPerRepeat: number;
  readonly cpuNsPerSample: number;
  /** CPU time divided by wall time: above 1 means the library used several threads. */
  readonly cpuUtilization: number;
  readonly heapBytesPerCall: number;
  readonly gcCount: number;
  readonly gcPauseMsPer1MCalls: number;
  readonly gcCountPer1MCalls: number;
  readonly rssGrowthMb: number;
  readonly externalGrowthMb: number;
  readonly tensorGrowth: number | null;
}

const nowNs = (): bigint => process.hrtime.bigint();
const yieldToEventLoop = (): Promise<void> => new Promise((resolve) => setImmediate(resolve));

function runCalls(subject: Subject, start: number, count: number): number {
  const run = subject.run;
  for (let k = 0; k < count; k += 1) run(start + k);
  return start + count;
}

/** Doubles the call count until one chunk takes at least `minMs`, returning calls per ms. */
function calibrate(subject: Subject, minMs: number): number {
  let counter = 0;
  for (let calls = 1; ; calls *= 2) {
    const start = nowNs();
    counter = runCalls(subject, counter, calls);
    const ms = Number(nowNs() - start) / 1e6;
    if (ms >= minMs || calls >= 1 << 24) return calls / ms;
  }
}

/** Heap growth per call over short windows right after a forced GC, minus the probe's own cost. */
function heapBytesPerCall(subject: Subject, counterStart: number): number {
  const windows = 60;
  const calls = 100;
  const growth: number[] = [];
  const baseline: number[] = [];
  let counter = counterStart;
  for (let w = 0; w < windows; w += 1) {
    gc?.();
    const emptyBefore = process.memoryUsage().heapUsed;
    baseline.push(process.memoryUsage().heapUsed - emptyBefore);
    gc?.();
    const before = process.memoryUsage().heapUsed;
    counter = runCalls(subject, counter, calls);
    growth.push(process.memoryUsage().heapUsed - before);
  }
  const median = (values: number[]): number =>
    [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] as number;
  return Math.max(0, (median(growth) - median(baseline)) / calls);
}

export async function measure(
  subject: Subject,
  samplesPerCall: number,
  options: MeasureOptions,
): Promise<MeasureResult> {
  const perMs = calibrate(subject, 5);
  const callsPerRepeat = Math.max(1, Math.round(perMs * options.repeatMs));
  let counter = 0;
  const warmupEnd = Date.now() + options.warmupMs;
  while (Date.now() < warmupEnd)
    counter = runCalls(subject, counter, Math.max(1, Math.round(callsPerRepeat / 20)));

  const before = snapshot(); // forces a GC, so it must happen before the observer starts counting
  let gcCount = 0;
  let gcPauseMs = 0;
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      gcCount += 1;
      gcPauseMs += entry.duration;
    }
  });
  observer.observe({ entryTypes: ['gc'] });
  await yieldToEventLoop();
  gcCount = 0;
  gcPauseMs = 0;

  const tensorsBefore = subject.tensorCount?.() ?? null;
  const cpuBefore = process.cpuUsage();
  const wallBefore = nowNs();
  const samples: number[] = [];
  for (let repeat = 0; repeat < options.repeats; repeat += 1) {
    const start = nowNs();
    counter = runCalls(subject, counter, callsPerRepeat);
    samples.push(Number(nowNs() - start) / callsPerRepeat / samplesPerCall);
    await yieldToEventLoop(); // lets the observer receive the GC entries of this repeat
  }
  const wallNs = Number(nowNs() - wallBefore);
  const cpu = process.cpuUsage(cpuBefore);
  await yieldToEventLoop();
  observer.disconnect();
  const gcCountMeasured = gcCount;
  const gcPauseMeasured = gcPauseMs;
  const after = snapshot();
  const totalCalls = callsPerRepeat * options.repeats;
  const sorted = [...samples].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] as number;
  const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
  const deviation = Math.sqrt(samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length);
  const tensorsAfter = subject.tensorCount?.() ?? null;
  return {
    nsPerSample: {
      median,
      min: sorted[0] as number,
      max: sorted.at(-1) as number,
      cv: deviation / mean,
    },
    callsPerRepeat,
    cpuNsPerSample: ((cpu.user + cpu.system) * 1000) / totalCalls / samplesPerCall,
    cpuUtilization: ((cpu.user + cpu.system) * 1000) / wallNs,
    heapBytesPerCall: heapBytesPerCall(subject, counter),
    gcCount: gcCountMeasured,
    gcPauseMsPer1MCalls: (gcPauseMeasured / totalCalls) * 1e6,
    gcCountPer1MCalls: (gcCountMeasured / totalCalls) * 1e6,
    rssGrowthMb: after.rssMb - before.rssMb,
    externalGrowthMb: after.externalMb - before.externalMb,
    tensorGrowth:
      tensorsBefore === null || tensorsAfter === null ? null : tensorsAfter - tensorsBefore,
  };
}
