# neuro-learn vs TensorFlow.js benchmark

## Environment

- date: 2026-11-23T15:25:15.925Z
- node: v24.11.1
- cpu: 12th Gen Intel(R) Core(TM) i5-12500H
- logicalCpus: 16
- memoryGb: 15.283103942871094
- platform: linux 7.0.0-43-generic
- pinnedCpu: null
- quick: false
- neuroLearn: 0.1.0
- tfjs: 4.22.0
- tfjsNode: 4.22.0
- tfjsWasm: 4.22.0

## Install footprint (on disk)

| Package                                           |      Size |
| ------------------------------------------------- | --------: |
| neuro-learn (dist)                                |    0.3 MB |
| tfjs (core, layers, cpu backend, data, converter) |  243.2 MB |
| tfjs-backend-wasm                                 |   12.6 MB |
| tfjs-node (incl. native TensorFlow library)       | 1149.9 MB |

## Speed and resources per scenario

### MLP forward [16,16,4], 1 sample per call (`mlp-forward-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 510 ns |   1,961,232 |       baseline |       0.8% |           1.00 |
| tfjs-cpu-layers  |                23.1 µs |      43,280 |     45× slower |      12.7% |           1.30 |
| tfjs-cpu-ops     |                8.67 µs |     115,385 |     17× slower |       4.9% |           1.05 |
| tfjs-wasm-layers |                27.5 µs |      36,423 |     54× slower |       7.4% |           1.28 |
| tfjs-wasm-ops    |                12.2 µs |      82,049 |     24× slower |       6.2% |           1.12 |
| tfjs-node-layers |               133.5 µs |       7,492 |    262× slower |       6.7% |           1.41 |
| tfjs-node-ops    |                52.8 µs |      18,940 |    104× slower |      13.6% |           2.13 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            511 ns |    70 MB |      1 MB / 6 ms |            0.4 MB |         n/a |
| tfjs-cpu-layers  |       107024 B | 2.4% (2.3 per 1k calls) |           31.7 µs |   213 MB |   87 MB / 300 ms |            0.1 MB |           0 |
| tfjs-cpu-ops     |        49455 B | 2.0% (0.8 per 1k calls) |           9.32 µs |   196 MB |   75 MB / 291 ms |            0.1 MB |           0 |
| tfjs-wasm-layers |        81182 B | 2.5% (2.3 per 1k calls) |           35.7 µs |   213 MB |   85 MB / 332 ms |            1.7 MB |           0 |
| tfjs-wasm-ops    |        28092 B | 1.2% (0.8 per 1k calls) |           14.0 µs |   213 MB |   97 MB / 320 ms |            0.0 MB |           0 |
| tfjs-node-layers |       104941 B | 0.0% (0.0 per 1k calls) |          192.5 µs |   275 MB |  157 MB / 360 ms |            0.3 MB |           0 |
| tfjs-node-ops    |        24303 B | 0.0% (0.0 per 1k calls) |          120.2 µs |   276 MB |  155 MB / 361 ms |            0.1 MB |           0 |

### MLP forward [64,128,64,8], 1 sample per call (`mlp-forward-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                18.2 µs |      54,874 |       baseline |       0.2% |           1.00 |
| tfjs-cpu-layers  |                68.7 µs |      14,556 |    3.8× slower |       8.2% |           1.19 |
| tfjs-cpu-ops     |                45.2 µs |      22,100 |    2.5× slower |       4.0% |           1.06 |
| tfjs-wasm-layers |                42.1 µs |      23,761 |    2.3× slower |       9.9% |           1.27 |
| tfjs-wasm-ops    |                21.6 µs |      46,396 |    1.2× slower |       5.3% |           1.10 |
| tfjs-node-layers |               231.8 µs |       4,315 |     13× slower |      17.5% |           1.80 |
| tfjs-node-ops    |                84.9 µs |      11,779 |    4.7× slower |      17.1% |           1.95 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |           18.2 µs |    72 MB |      1 MB / 8 ms |            0.1 MB |         n/a |
| tfjs-cpu-layers  |       226702 B | 1.5% (3.3 per 1k calls) |           84.9 µs |   205 MB |   80 MB / 321 ms |            3.0 MB |           0 |
| tfjs-cpu-ops     |       152362 B | 0.7% (1.1 per 1k calls) |           49.0 µs |   198 MB |   77 MB / 301 ms |            1.4 MB |           0 |
| tfjs-wasm-layers |       115499 B | 1.8% (3.2 per 1k calls) |           56.3 µs |   215 MB |   95 MB / 321 ms |            1.8 MB |           0 |
| tfjs-wasm-ops    |        42764 B | 0.9% (1.2 per 1k calls) |           24.2 µs |   205 MB |   89 MB / 328 ms |            0.1 MB |           0 |
| tfjs-node-layers |       146037 B | 0.0% (0.0 per 1k calls) |          449.0 µs |   272 MB |  156 MB / 364 ms |            0.1 MB |           0 |
| tfjs-node-ops    |        36980 B | 0.0% (0.0 per 1k calls) |          180.3 µs |   278 MB |  163 MB / 353 ms |            0.0 MB |           0 |

### MLP forward [64,128,64,8], 256 samples per call (`mlp-forward-batch`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                19.4 µs |      51,615 |       baseline |       0.2% |           1.00 |
| tfjs-cpu-layers  |                32.5 µs |      30,758 |    1.7× slower |       5.1% |           1.12 |
| tfjs-cpu-ops     |                32.2 µs |      31,052 |    1.7× slower |       7.3% |           1.08 |
| tfjs-wasm-layers |                3.94 µs |     253,593 |    4.9× faster |       2.1% |           1.16 |
| tfjs-wasm-ops    |                3.65 µs |     273,603 |    5.3× faster |       1.0% |           1.03 |
| tfjs-node-layers |                1.62 µs |     617,305 |     12× faster |      49.6% |           2.23 |
| tfjs-node-ops    |                 882 ns |   1,134,317 |     22× faster |       8.8% |           3.07 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |  0.0% (0.0 per 1k calls) |           19.4 µs |    76 MB |      0 MB / 8 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |        32936 B | 0.8% (95.2 per 1k calls) |           37.7 µs |   254 MB |   81 MB / 308 ms |            0.4 MB |           0 |
| tfjs-cpu-ops     |        48041 B | 0.4% (37.0 per 1k calls) |           35.7 µs |   222 MB |   79 MB / 350 ms |            0.9 MB |           0 |
| tfjs-wasm-layers |       115728 B | 0.8% (10.8 per 1k calls) |           4.65 µs |   222 MB |   88 MB / 339 ms |            0.4 MB |           0 |
| tfjs-wasm-ops    |        42859 B |  0.1% (1.3 per 1k calls) |           3.78 µs |   209 MB |   85 MB / 332 ms |           26.0 MB |           0 |
| tfjs-node-layers |       146219 B |  0.0% (0.0 per 1k calls) |           4.75 µs |   279 MB |  159 MB / 363 ms |            0.1 MB |           0 |
| tfjs-node-ops    |        36817 B |  0.0% (0.0 per 1k calls) |           2.84 µs |   291 MB |  165 MB / 357 ms |            0.1 MB |           0 |

### Population: 500 creatures, each with its own MLP [16,16,4] (one tick) (`population-forward`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 647 ns |   1,545,388 |       baseline |       0.1% |           1.00 |
| tfjs-cpu-layers  |                34.2 µs |      29,279 |     53× slower |      13.0% |           1.40 |
| tfjs-cpu-ops     |                 938 ns |   1,065,650 |    1.5× slower |       2.2% |           1.11 |
| tfjs-wasm-layers |                40.5 µs |      24,690 |     63× slower |      10.9% |           1.41 |
| tfjs-wasm-ops    |                 542 ns |   1,846,402 |    1.2× faster |       1.2% |           1.03 |
| tfjs-node-layers |               154.4 µs |       6,478 |    239× slower |       6.8% |           1.15 |
| tfjs-node-ops    |                 524 ns |   1,907,555 |    1.2× faster |       6.3% |           5.13 |

| Implementation   | JS heap / call |          GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ------------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |   0.0% (0.0 per 1k calls) |            648 ns |    76 MB |     2 MB / 14 ms |            0.4 MB |         n/a |
| tfjs-cpu-layers  |      1082110 B | 4.1% (638.9 per 1k calls) |           50.6 µs |   355 MB |  116 MB / 340 ms |           -0.0 MB |           0 |
| tfjs-cpu-ops     |       110323 B |   0.7% (5.0 per 1k calls) |           1.05 µs |   237 MB |   78 MB / 299 ms |            4.0 MB |           0 |
| tfjs-wasm-layers |       493520 B | 4.2% (629.6 per 1k calls) |           60.0 µs |   369 MB |  113 MB / 372 ms |            0.4 MB |           0 |
| tfjs-wasm-ops    |        28344 B |   0.2% (1.1 per 1k calls) |            566 ns |   211 MB |   86 MB / 345 ms |            3.1 MB |           0 |
| tfjs-node-layers |       539900 B | 2.1% (888.9 per 1k calls) |          181.4 µs |   444 MB |  199 MB / 411 ms |            1.1 MB |           0 |
| tfjs-node-ops    |        24419 B |   0.0% (0.0 per 1k calls) |           2.77 µs |   273 MB |  160 MB / 361 ms |            0.1 MB |           0 |

### MLP online SGD step [16,16,4] (`mlp-train-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                1.98 µs |     504,204 |       baseline |       0.4% |           1.00 |
| tfjs-cpu-layers  |               168.9 µs |       5,920 |     85× slower |      17.4% |           2.14 |
| tfjs-cpu-ops     |               113.1 µs |       8,844 |     57× slower |       8.4% |           1.66 |
| tfjs-wasm-layers |               208.8 µs |       4,789 |    105× slower |       5.8% |           1.85 |
| tfjs-wasm-ops    |               197.1 µs |       5,075 |     99× slower |      17.0% |           1.76 |
| tfjs-node-layers |               524.3 µs |       1,907 |    264× slower |       5.3% |           1.32 |
| tfjs-node-ops    |               393.4 µs |       2,542 |    198× slower |      14.0% |           1.40 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |           16 B | 0.1% (0.0 per 1k calls) |           1.99 µs |    72 MB |      1 MB / 6 ms |            0.8 MB |         n/a |
| tfjs-cpu-layers  |       385606 B | 4.3% (5.1 per 1k calls) |          409.3 µs |   313 MB |   81 MB / 298 ms |           -3.0 MB |           0 |
| tfjs-cpu-ops     |       320416 B | 3.4% (4.3 per 1k calls) |          191.7 µs |   293 MB |   79 MB / 287 ms |           -1.3 MB |           0 |
| tfjs-wasm-layers |       336680 B | 1.7% (3.3 per 1k calls) |          409.3 µs |   307 MB |   90 MB / 328 ms |           -2.5 MB |           0 |
| tfjs-wasm-ops    |       254270 B | 0.8% (2.6 per 1k calls) |          343.0 µs |   290 MB |   87 MB / 337 ms |           -0.9 MB |           0 |
| tfjs-node-layers |       341394 B | 0.0% (0.0 per 1k calls) |          718.2 µs |   362 MB |  152 MB / 362 ms |           -1.3 MB |           0 |
| tfjs-node-ops    |       232351 B | 0.0% (0.0 per 1k calls) |          592.0 µs |   364 MB |  161 MB / 362 ms |           -0.4 MB |           0 |

### MLP online SGD step [64,128,64,8] (`mlp-train-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                94.7 µs |      10,561 |       baseline |       1.5% |           1.00 |
| tfjs-cpu-layers  |               815.5 µs |       1,226 |    8.6× slower |      15.0% |           1.41 |
| tfjs-cpu-ops     |               690.4 µs |       1,448 |    7.3× slower |      14.2% |           1.42 |
| tfjs-wasm-layers |               476.0 µs |       2,101 |    5.0× slower |       7.0% |           1.81 |
| tfjs-wasm-ops    |               368.5 µs |       2,714 |    3.9× slower |      10.7% |           1.62 |
| tfjs-node-layers |               948.6 µs |       1,054 |     10× slower |       9.4% |           1.49 |
| tfjs-node-ops    |               625.9 µs |       1,598 |    6.6× slower |       9.6% |           1.49 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |           16 B |  0.0% (0.0 per 1k calls) |           95.3 µs |    76 MB |      1 MB / 7 ms |            0.1 MB |         n/a |
| tfjs-cpu-layers  |       255809 B | 2.5% (32.4 per 1k calls) |           1.21 ms |   283 MB |   81 MB / 312 ms |            1.2 MB |           0 |
| tfjs-cpu-ops     |       113282 B | 3.3% (64.5 per 1k calls) |           1.05 ms |   273 MB |   84 MB / 300 ms |            1.6 MB |           0 |
| tfjs-wasm-layers |       475731 B |  2.4% (6.9 per 1k calls) |          894.9 µs |   293 MB |   85 MB / 336 ms |           -0.6 MB |           0 |
| tfjs-wasm-ops    |       349355 B |  0.9% (4.4 per 1k calls) |          633.7 µs |   287 MB |   91 MB / 360 ms |           -0.4 MB |           0 |
| tfjs-node-layers |       491647 B |  0.0% (0.0 per 1k calls) |           1.44 ms |   346 MB |  166 MB / 389 ms |           -0.9 MB |           0 |
| tfjs-node-ops    |       322471 B | 5.4% (11.1 per 1k calls) |           1.04 ms |   303 MB |  173 MB / 375 ms |            1.0 MB |           0 |

### Delta rule (Rescorla-Wagner) 16 to 4 (`delta-rule-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 244 ns |   4,094,985 |       baseline |       0.8% |           1.00 |
| tfjs-cpu-layers  |                71.1 µs |      14,057 |    291× slower |       9.1% |           1.76 |
| tfjs-cpu-ops     |                45.0 µs |      22,205 |    184× slower |       9.1% |           1.70 |
| tfjs-wasm-layers |               101.3 µs |       9,876 |    415× slower |      12.0% |           1.71 |
| tfjs-wasm-ops    |                87.4 µs |      11,439 |    358× slower |      11.7% |           2.01 |
| tfjs-node-layers |               275.0 µs |       3,637 |   1126× slower |       5.5% |           1.35 |
| tfjs-node-ops    |               197.9 µs |       5,054 |    810× slower |      14.9% |           1.72 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            243 ns |    73 MB |      1 MB / 6 ms |            0.1 MB |         n/a |
| tfjs-cpu-layers  |       172560 B | 5.3% (2.5 per 1k calls) |          131.7 µs |   318 MB |   76 MB / 298 ms |           -3.1 MB |           0 |
| tfjs-cpu-ops     |       121381 B | 3.8% (1.7 per 1k calls) |           78.4 µs |   297 MB |   78 MB / 309 ms |            0.6 MB |           0 |
| tfjs-wasm-layers |       163742 B | 0.0% (0.0 per 1k calls) |          185.8 µs |   308 MB |   88 MB / 348 ms |           -3.1 MB |           0 |
| tfjs-wasm-ops    |       112815 B | 0.0% (0.0 per 1k calls) |          188.9 µs |   292 MB |   82 MB / 334 ms |           -1.1 MB |           0 |
| tfjs-node-layers |       165882 B | 0.0% (0.0 per 1k calls) |          375.0 µs |   363 MB |  154 MB / 366 ms |           -1.2 MB |           0 |
| tfjs-node-ops    |       101441 B | 0.0% (0.0 per 1k calls) |          379.8 µs |   359 MB |  154 MB / 368 ms |           -0.6 MB |           0 |

### Delta rule (Rescorla-Wagner) 256 to 64 (`delta-rule-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                55.5 µs |      18,016 |       baseline |       0.6% |           1.00 |
| tfjs-cpu-layers  |               625.9 µs |       1,598 |     11× slower |      15.9% |           1.18 |
| tfjs-cpu-ops     |               534.8 µs |       1,870 |    9.6× slower |       8.8% |           1.22 |
| tfjs-wasm-layers |               233.5 µs |       4,283 |    4.2× slower |       9.3% |           1.42 |
| tfjs-wasm-ops    |               183.7 µs |       5,444 |    3.3× slower |       9.9% |           1.35 |
| tfjs-node-layers |               362.0 µs |       2,762 |    6.5× slower |      12.1% |           1.59 |
| tfjs-node-ops    |               302.4 µs |       3,307 |    5.4× slower |      19.1% |           1.83 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |  0.0% (0.0 per 1k calls) |           55.7 µs |    76 MB |      1 MB / 6 ms |            0.6 MB |         n/a |
| tfjs-cpu-layers  |       523547 B | 3.9% (53.8 per 1k calls) |          810.1 µs |   275 MB |   83 MB / 301 ms |            0.4 MB |           0 |
| tfjs-cpu-ops     |       451635 B | 3.2% (50.0 per 1k calls) |          680.3 µs |   274 MB |   78 MB / 299 ms |            0.1 MB |           0 |
| tfjs-wasm-layers |       163821 B |  0.0% (0.0 per 1k calls) |          346.8 µs |   287 MB |   87 MB / 341 ms |           -0.5 MB |           0 |
| tfjs-wasm-ops    |       112917 B |  0.0% (0.0 per 1k calls) |          254.6 µs |   234 MB |   88 MB / 338 ms |            9.6 MB |           0 |
| tfjs-node-layers |       166126 B |  0.0% (0.0 per 1k calls) |          617.2 µs |   335 MB |  166 MB / 379 ms |           -1.2 MB |           0 |
| tfjs-node-ops    |       101429 B |  0.0% (0.0 per 1k calls) |          529.7 µs |   290 MB |  155 MB / 368 ms |            0.1 MB |           0 |

### Elman RNN step 16-16-4 (`rnn-forward-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 843 ns |   1,186,137 |       baseline |       0.3% |           1.00 |
| tfjs-cpu-layers  |                43.4 µs |      23,046 |     51× slower |      12.9% |           1.30 |
| tfjs-cpu-ops     |                13.1 µs |      76,510 |     16× slower |       3.5% |           1.02 |
| tfjs-wasm-layers |                46.1 µs |      21,684 |     55× slower |       7.4% |           1.25 |
| tfjs-wasm-ops    |                18.2 µs |      55,018 |     22× slower |       1.2% |           1.06 |
| tfjs-node-layers |               204.1 µs |       4,899 |    242× slower |       3.8% |           1.10 |
| tfjs-node-ops    |                60.7 µs |      16,475 |     72× slower |       4.4% |           1.01 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            844 ns |    73 MB |      1 MB / 6 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       172412 B | 2.5% (3.8 per 1k calls) |           59.2 µs |   211 MB |   96 MB / 319 ms |            1.1 MB |           0 |
| tfjs-cpu-ops     |        74946 B | 1.2% (1.1 per 1k calls) |           13.5 µs |   197 MB |   77 MB / 298 ms |            1.0 MB |           0 |
| tfjs-wasm-layers |       126742 B | 2.1% (3.6 per 1k calls) |           60.2 µs |   219 MB |   92 MB / 338 ms |            2.6 MB |           0 |
| tfjs-wasm-ops    |        36488 B | 0.9% (0.5 per 1k calls) |           19.6 µs |   270 MB |   90 MB / 331 ms |            0.0 MB |           0 |
| tfjs-node-layers |       157505 B | 0.8% (4.4 per 1k calls) |          226.3 µs |   303 MB |  197 MB / 411 ms |            1.5 MB |           0 |
| tfjs-node-ops    |        28367 B | 0.0% (0.0 per 1k calls) |           63.7 µs |   276 MB |  160 MB / 364 ms |            0.0 MB |           0 |

### Elman RNN step 64-128-8 (`rnn-forward-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                28.8 µs |      34,743 |       baseline |       0.2% |           1.00 |
| tfjs-cpu-layers  |               105.7 µs |       9,456 |    3.7× slower |      22.3% |           1.36 |
| tfjs-cpu-ops     |                61.0 µs |      16,389 |    2.1× slower |      15.7% |           1.19 |
| tfjs-wasm-layers |                50.5 µs |      19,808 |    1.8× slower |       3.7% |           1.13 |
| tfjs-wasm-ops    |                22.0 µs |      45,527 |    1.3× faster |       1.1% |           1.06 |
| tfjs-node-layers |               234.9 µs |       4,257 |    8.2× slower |       5.0% |           1.28 |
| tfjs-node-ops    |                73.4 µs |      13,622 |    2.6× slower |       5.9% |           1.36 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |           28.8 µs |    73 MB |      1 MB / 7 ms |            0.1 MB |         n/a |
| tfjs-cpu-layers  |       266721 B | 1.6% (5.1 per 1k calls) |          159.6 µs |   204 MB | 129 MB / 1134 ms |            0.9 MB |           0 |
| tfjs-cpu-ops     |       149074 B | 0.9% (1.0 per 1k calls) |           76.9 µs |   198 MB |   77 MB / 296 ms |            0.3 MB |           0 |
| tfjs-wasm-layers |       126579 B | 1.9% (3.6 per 1k calls) |           58.3 µs |   253 MB |  149 MB / 406 ms |            4.1 MB |           0 |
| tfjs-wasm-ops    |        36507 B | 1.0% (1.1 per 1k calls) |           23.2 µs |   208 MB |   94 MB / 335 ms |           -0.1 MB |           0 |
| tfjs-node-layers |       157726 B | 0.6% (4.4 per 1k calls) |          298.6 µs |   313 MB |  230 MB / 493 ms |            1.8 MB |           0 |
| tfjs-node-ops    |        28367 B | 0.0% (0.0 per 1k calls) |          103.7 µs |   275 MB |  159 MB / 366 ms |            0.1 MB |           0 |

## Accuracy

### Forward pass: largest absolute error against the float64 reference

| Implementation   | mlp-forward-small | mlp-forward-large | mlp-forward-batch | population-forward | rnn-forward-small | rnn-forward-large |
| ---------------- | ----------------: | ----------------: | ----------------: | -----------------: | ----------------: | ----------------: |
| neuro-learn      |            8.2e-8 |            8.2e-8 |            1.2e-7 |             1.2e-7 |            1.2e-7 |            1.9e-7 |
| tfjs-cpu-layers  |            8.2e-8 |            9.2e-8 |            1.5e-7 |             1.2e-7 |            1.4e-7 |            2.4e-7 |
| tfjs-cpu-ops     |            8.2e-8 |            9.2e-8 |            1.5e-7 |             1.2e-7 |            1.4e-7 |            2.4e-7 |
| tfjs-wasm-layers |            1.4e-7 |            4.0e-7 |            5.3e-7 |             3.7e-7 |            3.5e-7 |            1.3e-6 |
| tfjs-wasm-ops    |            1.4e-7 |            4.0e-7 |            5.3e-7 |             3.7e-7 |            3.5e-7 |            1.3e-6 |
| tfjs-node-layers |            2.0e-7 |            3.3e-7 |            4.6e-7 |             4.3e-7 |            3.8e-7 |            1.3e-6 |
| tfjs-node-ops    |            2.0e-7 |            3.3e-7 |            4.6e-7 |             4.3e-7 |            3.8e-7 |            1.3e-6 |

### Training: held-out loss after 2000 online SGD steps from identical initial weights and sample order

The reference column is a float64 re-implementation trained on the same steps. "Max weight diff" is against that reference.

| Implementation   |         Scenario | Initial loss | Final loss | Reference final loss | Max weight diff |
| ---------------- | ---------------: | -----------: | ---------: | -------------------: | --------------: |
| neuro-learn      |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.4e-6 |
| neuro-learn      |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          9.3e-7 |
| neuro-learn      | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          2.8e-8 |
| neuro-learn      | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          2.0e-8 |
| tfjs-cpu-layers  |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.2e-6 |
| tfjs-cpu-layers  |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          9.0e-7 |
| tfjs-cpu-layers  | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          4.5e-8 |
| tfjs-cpu-layers  | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.5e-8 |
| tfjs-cpu-ops     |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.2e-6 |
| tfjs-cpu-ops     |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          9.0e-7 |
| tfjs-cpu-ops     | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          4.5e-8 |
| tfjs-cpu-ops     | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.5e-8 |
| tfjs-wasm-layers |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.1e-6 |
| tfjs-wasm-layers |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          9.6e-7 |
| tfjs-wasm-layers | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          3.1e-8 |
| tfjs-wasm-layers | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.9e-8 |
| tfjs-wasm-ops    |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          9.9e-7 |
| tfjs-wasm-ops    |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          9.3e-7 |
| tfjs-wasm-ops    | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          3.1e-8 |
| tfjs-wasm-ops    | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.9e-8 |
| tfjs-node-layers |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.2e-6 |
| tfjs-node-layers |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          8.4e-7 |
| tfjs-node-layers | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          2.8e-8 |
| tfjs-node-layers | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.4e-8 |
| tfjs-node-ops    |  mlp-train-small |       1.0931 |   0.005993 |             0.005993 |          1.2e-6 |
| tfjs-node-ops    |  mlp-train-large |       1.5865 |   0.025963 |             0.025963 |          8.4e-7 |
| tfjs-node-ops    | delta-rule-small |       0.1115 |   0.000000 |             0.000000 |          2.8e-8 |
| tfjs-node-ops    | delta-rule-large |       0.0482 |   0.005325 |             0.005325 |          1.4e-8 |
