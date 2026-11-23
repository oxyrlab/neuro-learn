# Performance

Measured on Node v24.11.1 (Linux) with `npm run bench`; numbers are orientation only and differ per machine.

## Throughput (16 inputs → 4 outputs; MLP/networks [16, 16, 4])

| Operation                    | ns/call | calls/s |
| ---------------------------- | ------: | ------: |
| rescorla-wagner predict      |      77 |  13.0 M |
| rescorla-wagner learn        |     232 |   4.3 M |
| eligibility-trace predict    |      80 |  12.4 M |
| eligibility-trace learn      |     271 |   3.7 M |
| eligibility-trace decay      |      21 |  47.8 M |
| hebbian predict              |      84 |  11.9 M |
| hebbian learn                |     210 |   4.8 M |
| mlp-sgd predict              |     524 |   1.9 M |
| mlp-sgd learn                |    1820 |  0.55 M |
| feedforward forward          |     531 |   1.9 M |
| recurrent forward            |     803 |   1.2 M |
| `decay` (RW / Hebbian / MLP) |     3–4 | > 270 M |

**EcoSim budget example:** 500 creatures, one `eligibility-trace` predict (80 ns) plus one `mlp-sgd` predict (524 ns) per brain tick at 10 Hz is about 0.3 ms per tick, about 4% of the 8 ms tick budget. `learn` runs only on learning events.

## Comparison with TensorFlow.js

A separate benchmark ([benchmarks/tfjs-comparison](../benchmarks/tfjs-comparison/README.md)) runs the shared features on identical data, initial weights and sample order, with each measurement in a fresh process. TensorFlow.js 4.22 is measured with three backends (`cpu`, `wasm`, `node`) and two APIs (`tf.layers`, low-level ops). Full tables: [results](../benchmarks/tfjs-comparison/results/results.md) and [pinned to one core](../benchmarks/tfjs-comparison/results/pinned-1core.md). Node v24.11.1, Intel i5-12500H.

Time per sample of TensorFlow.js relative to neuro-learn (range over the six TensorFlow.js configurations):

| Scenario                          | neuro-learn | TensorFlow.js                                                                    |
| --------------------------------- | ----------: | -------------------------------------------------------------------------------- |
| MLP forward [16,16,4]             |      0.5 µs | 17× to 262× slower                                                               |
| MLP online SGD step [16,16,4]     |        2 µs | 57× to 264× slower                                                               |
| Delta rule 16 → 4                 |     0.24 µs | 184× to 1126× slower                                                             |
| Elman RNN step 16-16-4            |     0.84 µs | 15× to 242× slower                                                               |
| MLP forward [64,128,64,8]         |       18 µs | 1.2× to 13× slower                                                               |
| MLP online SGD step [64,128,64,8] |       95 µs | 3.9× to 10× slower                                                               |
| One network, 256 samples per call |       19 µs | 5× to over 20× faster (wasm, node); 1.7× slower (cpu)                            |
| 500 creatures, one network each   |     0.65 µs | 1.2× faster (stacked `batchMatMul`); 52× to 238× slower (one model per creature) |

Resources: neuro-learn allocates 0 bytes per call and peaks at about 70 MB RSS, loading in about 6 ms and 0.3 MB on disk; TensorFlow.js allocates 24 KB to 1 MB of JS heap per call, peaks at 200 to 450 MB RSS, loads in 300 to 400 ms and takes 243 MB (cpu) to 1.1 GB (node) on disk. No tensor leaks were found. Accuracy is equal within Float32 rounding: both match an independent float64 reference (forward error about 1e-7 for neuro-learn, 1e-7 to 1.3e-6 for TensorFlow.js) and reach the same held-out loss after identical online training.

Conclusion: for small networks, batch size 1 and per-creature online learning, neuro-learn is faster by one to three orders of magnitude. TensorFlow.js only wins when many samples share one network, or with batched stacked weights, and only by a small factor at the population sizes tested. GPU backends were not measured.

## Allocation

`npm run check:allocation` (run with `--expose-gc`) forces a collection, runs 4000-call windows and reads the heap growth; the median over 50 windows must be <= 2 bytes/call. A deliberately allocating control must read >= 50 bytes/call, which proves the detector works. Every `predict`, `learn`, `decay`, `observe` and `forward` is checked with an integer bound (`1`) and a fractional bound (`0.9`), because V8 takes different paths for them.

Lessons (see [ADR 0007](adr/0007-hot-path-numerics-and-failure-handling.md)):

- A helper that returns a fractional number per element makes V8 box a heap number each call; work on whole buffers instead.
- Merging an integer-valued bound and a computed value into one variable (`next = limit`) boxes a number per weight; store inside each branch.
- Reading `heapUsed` inside the measured loop is unreliable (it jumps by whole memory pages); measure per short window.

## Denormals

Weights that decay into the Float32 denormal range (below about 1.2e-38) cause no measurable slowdown, because values are widened to float64 when read (`predict` ratio 1.05 for rescorla-wagner and 0.79 for the MLP, i.e. within noise). No flush-to-zero option is needed.
