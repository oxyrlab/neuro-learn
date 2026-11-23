# neuro-learn vs TensorFlow.js benchmark

Compares `@oxyrlab/neuro-learn` with TensorFlow.js on the features both libraries share, using the
same data, the same initial weights and the same sample order. It lives in its own folder with its
own `package.json` so the library itself stays dependency-free.

## What is compared

| Scenario             | neuro-learn                  | TensorFlow.js equivalent                          |
| -------------------- | ---------------------------- | ------------------------------------------------- |
| `mlp-forward-*`      | `FeedforwardNetwork.forward` | dense layers, tanh hidden, linear output          |
| `population-forward` | 500 separate networks        | 500 models (layers) / stacked `batchMatMul` (ops) |
| `mlp-train-*`        | `MlpSgdLearner.learn`        | SGD on 0.5 · sum of squared errors, batch size 1  |
| `delta-rule-*`       | `RescorlaWagnerLearner`      | linear dense layer without bias, same SGD loss    |
| `rnn-forward-*`      | `RecurrentNetwork.forward`   | stateful `simpleRNN` + dense / manual ops         |

`hebbian` and `eligibility-trace` have no TensorFlow.js counterpart, so they are not compared.

TensorFlow.js is run in six configurations: backends `cpu` (pure JS), `wasm` and `node` (native
TensorFlow C library) times two APIs, `layers` (`tf.layers`, what most people write) and `ops`
(`tf.matMul`/`tf.add`/`tf.tanh` plus autograd, the lowest-overhead path).

## Fairness rules

- Every measurement runs in a **fresh process** (`worker.ts`), so memory and JIT state are never shared.
- Identical data, identical Glorot-initialised weights, identical sample order for both libraries.
- Inputs and outputs are plain `Float32Array`s owned by the caller, because a simulation needs the
  results in its own buffers. For TensorFlow.js this includes creating the input tensor and calling
  `dataSync()` on every call; that overhead is part of using it from a simulation loop.
- Warm-up first, then 9 repeats of 250 ms; the median is reported, with the coefficient of variation.
- Accuracy is judged against an independent **float64 reference** (`lib/reference.ts`), never one
  library against the other. Held-out loss is always evaluated with that same reference forward pass.
- `--pin <cpu>` runs every process on one core (`taskset`), giving both libraries equal CPU resources.

## Run

```bash
npm run build                 # in the repository root: the benchmark imports ../../dist
cd benchmarks/tfjs-comparison
npm install
npm run bench                 # full run, writes results/results.{json,md}
npm run bench -- --pin 2 --name pinned-1core
npm run bench -- --quick --only mlp-train-small --impl neuro-learn,tfjs-node-ops
```

## Notes

- `@tensorflow/tfjs-node` 4.22 (the latest release) does not load on Node 23+ because Node removed
  `util.isNullOrUndefined` and similar helpers. `lib/subjects/tf-load.ts` restores them before loading.
  This affects loading only, not the measured kernels.
- Results depend on the machine, the CPU governor and background load. Compare ratios, not absolute numbers.
- Measured: time per sample, CPU time and cores used, JS heap bytes per call, GC share of time, RSS
  (peak, growth during the run, cost of loading the library), tensor leaks, install size, accuracy.
