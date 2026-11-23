# neuro-learn vs TensorFlow.js benchmark

## Environment

- date: 2026-11-23T15:46:37.294Z
- node: v24.11.1
- cpu: 12th Gen Intel(R) Core(TM) i5-12500H
- logicalCpus: 16
- memoryGb: 15.283103942871094
- platform: linux 7.0.0-43-generic
- pinnedCpu: 2
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
| **neuro-learn**  |                 532 ns |   1,881,370 |       baseline |       0.3% |           1.00 |
| tfjs-cpu-layers  |               106.0 µs |       9,432 |    199× slower |      57.7% |           1.00 |
| tfjs-cpu-ops     |                8.75 µs |     114,238 |     16× slower |      43.7% |           1.00 |
| tfjs-wasm-layers |                33.6 µs |      29,722 |     63× slower |      27.2% |           1.00 |
| tfjs-wasm-ops    |                12.3 µs |      81,588 |     23× slower |      22.5% |           1.00 |
| tfjs-node-layers |               292.4 µs |       3,421 |    550× slower |      46.0% |           1.00 |
| tfjs-node-ops    |                81.8 µs |      12,230 |    154× slower |      70.4% |           1.00 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            532 ns |    71 MB |     1 MB / 18 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       105498 B | 0.4% (2.5 per 1k calls) |          155.7 µs |   209 MB |   78 MB / 439 ms |            0.1 MB |           0 |
| tfjs-cpu-ops     |        49544 B | 1.6% (0.8 per 1k calls) |           11.4 µs |   199 MB |   70 MB / 449 ms |            0.9 MB |           0 |
| tfjs-wasm-layers |        81866 B | 2.6% (1.0 per 1k calls) |           35.7 µs |   277 MB |   90 MB / 545 ms |            0.0 MB |           0 |
| tfjs-wasm-ops    |        28059 B | 4.0% (0.4 per 1k calls) |           14.4 µs |   270 MB |   88 MB / 516 ms |            0.0 MB |           0 |
| tfjs-node-layers |       104925 B | 0.0% (0.0 per 1k calls) |          314.6 µs |   288 MB |  160 MB / 529 ms |            0.3 MB |           0 |
| tfjs-node-ops    |        24253 B | 0.0% (0.0 per 1k calls) |          120.3 µs |   273 MB |  148 MB / 521 ms |            0.0 MB |           0 |

### MLP forward [64,128,64,8], 1 sample per call (`mlp-forward-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                18.6 µs |      53,905 |       baseline |       1.7% |           1.00 |
| tfjs-cpu-layers  |               183.4 µs |       5,454 |    9.9× slower |      62.9% |           1.00 |
| tfjs-cpu-ops     |                46.9 µs |      21,325 |    2.5× slower |      55.0% |           1.00 |
| tfjs-wasm-layers |               168.3 µs |       5,943 |    9.1× slower |      59.1% |           1.00 |
| tfjs-wasm-ops    |                38.5 µs |      25,980 |    2.1× slower |      55.4% |           1.00 |
| tfjs-node-layers |               362.7 µs |       2,757 |     20× slower |      46.7% |           1.00 |
| tfjs-node-ops    |                97.5 µs |      10,260 |    5.3× slower |      67.6% |           1.00 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |           18.7 µs |    71 MB |     2 MB / 22 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       207943 B | 0.5% (2.9 per 1k calls) |          258.2 µs |   205 MB |   74 MB / 470 ms |            0.1 MB |           0 |
| tfjs-cpu-ops     |       129961 B | 1.3% (1.3 per 1k calls) |           64.9 µs |   201 MB |   75 MB / 440 ms |            0.1 MB |           0 |
| tfjs-wasm-layers |       114734 B | 0.8% (3.5 per 1k calls) |          166.2 µs |   212 MB |   88 MB / 516 ms |            0.0 MB |           0 |
| tfjs-wasm-ops    |        42480 B | 0.0% (0.0 per 1k calls) |           58.9 µs |   272 MB |   86 MB / 511 ms |            0.0 MB |           0 |
| tfjs-node-layers |       146194 B | 0.0% (0.0 per 1k calls) |          454.8 µs |   279 MB |  153 MB / 548 ms |            0.3 MB |           0 |
| tfjs-node-ops    |        37145 B | 0.0% (0.0 per 1k calls) |          149.2 µs |   279 MB |  153 MB / 564 ms |            0.1 MB |           0 |

### MLP forward [64,128,64,8], 256 samples per call (`mlp-forward-batch`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                19.5 µs |      51,409 |       baseline |       1.3% |           1.00 |
| tfjs-cpu-layers  |                33.0 µs |      30,267 |    1.7× slower |      22.3% |           1.00 |
| tfjs-cpu-ops     |                32.2 µs |      31,020 |    1.7× slower |      24.1% |           1.00 |
| tfjs-wasm-layers |                4.34 µs |     230,422 |    4.5× faster |      12.8% |           1.00 |
| tfjs-wasm-ops    |                3.82 µs |     261,841 |    5.1× faster |       4.3% |           1.00 |
| tfjs-node-layers |                2.43 µs |     411,438 |    8.0× faster |      42.3% |           1.00 |
| tfjs-node-ops    |                1.10 µs |     911,971 |     18× faster |      46.5% |           1.00 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |  0.0% (0.0 per 1k calls) |           19.6 µs |    74 MB |     3 MB / 28 ms |            0.1 MB |         n/a |
| tfjs-cpu-layers  |        30607 B | 1.5% (37.0 per 1k calls) |           37.0 µs |   238 MB |   78 MB / 458 ms |            8.5 MB |           0 |
| tfjs-cpu-ops     |        59659 B | 2.0% (44.4 per 1k calls) |           37.1 µs |   225 MB |   71 MB / 428 ms |            0.6 MB |           0 |
| tfjs-wasm-layers |       116590 B | 3.4% (10.1 per 1k calls) |           4.59 µs |   253 MB |   90 MB / 520 ms |           16.1 MB |           0 |
| tfjs-wasm-ops    |        42939 B |  0.0% (0.0 per 1k calls) |           3.90 µs |   186 MB |   91 MB / 514 ms |            8.3 MB |           0 |
| tfjs-node-layers |       146534 B |  0.0% (0.0 per 1k calls) |           2.52 µs |   285 MB |  156 MB / 553 ms |            0.1 MB |           0 |
| tfjs-node-ops    |        37189 B |  0.0% (0.0 per 1k calls) |           1.36 µs |   297 MB |  161 MB / 551 ms |            0.0 MB |           0 |

### Population: 500 creatures, each with its own MLP [16,16,4] (one tick) (`population-forward`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 657 ns |   1,521,553 |       baseline |       0.3% |           1.00 |
| tfjs-cpu-layers  |                36.6 µs |      27,324 |     56× slower |      32.9% |           1.00 |
| tfjs-cpu-ops     |                1.07 µs |     931,810 |    1.6× slower |      37.7% |           1.00 |
| tfjs-wasm-layers |                44.0 µs |      22,712 |     67× slower |      43.5% |           1.00 |
| tfjs-wasm-ops    |                 576 ns |   1,736,531 |    1.1× faster |      29.4% |           1.00 |
| tfjs-node-layers |               163.4 µs |       6,121 |    249× slower |      19.9% |           1.00 |
| tfjs-node-ops    |                 535 ns |   1,869,196 |    1.2× faster |      23.2% |           1.00 |

| Implementation   | JS heap / call |          GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ------------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |   0.0% (0.0 per 1k calls) |            658 ns |    77 MB |     3 MB / 28 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |      1390563 B | 4.5% (611.1 per 1k calls) |           44.6 µs |   356 MB |  119 MB / 523 ms |            0.2 MB |           0 |
| tfjs-cpu-ops     |       101177 B | 10.8% (13.9 per 1k calls) |           1.35 µs |   229 MB |   72 MB / 435 ms |           17.6 MB |           0 |
| tfjs-wasm-layers |       972545 B | 4.5% (611.1 per 1k calls) |           54.7 µs |   374 MB |  123 MB / 597 ms |            0.1 MB |           0 |
| tfjs-wasm-ops    |        28960 B |   0.0% (0.0 per 1k calls) |            656 ns |   230 MB |   94 MB / 517 ms |            3.5 MB |           0 |
| tfjs-node-layers |      1249565 B | 2.4% (777.8 per 1k calls) |          181.0 µs |   459 MB |  197 MB / 634 ms |            2.4 MB |           0 |
| tfjs-node-ops    |        24625 B |   0.0% (0.0 per 1k calls) |            560 ns |   285 MB |  156 MB / 582 ms |            0.1 MB |           0 |

### MLP online SGD step [16,16,4] (`mlp-train-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                1.98 µs |     505,398 |       baseline |       0.2% |           1.00 |
| tfjs-cpu-layers  |               400.0 µs |       2,500 |    202× slower |      31.0% |           1.00 |
| tfjs-cpu-ops     |               368.6 µs |       2,713 |    186× slower |      34.1% |           1.00 |
| tfjs-wasm-layers |               419.0 µs |       2,386 |    212× slower |      32.1% |           1.00 |
| tfjs-wasm-ops    |               269.0 µs |       3,717 |    136× slower |      38.9% |           1.00 |
| tfjs-node-layers |               718.9 µs |       1,391 |    363× slower |      34.9% |           1.00 |
| tfjs-node-ops    |               483.2 µs |       2,070 |    244× slower |      37.4% |           1.00 |

| Implementation   | JS heap / call |          GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ------------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |           16 B |   0.0% (0.0 per 1k calls) |           1.98 µs |    76 MB |     2 MB / 18 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       379230 B |   3.2% (5.8 per 1k calls) |          462.5 µs |   287 MB |   75 MB / 437 ms |            0.9 MB |           0 |
| tfjs-cpu-ops     |       321355 B |   0.7% (3.1 per 1k calls) |          358.1 µs |   278 MB |   68 MB / 430 ms |            0.1 MB |           0 |
| tfjs-wasm-layers |       337678 B |   3.1% (3.5 per 1k calls) |          446.2 µs |   296 MB |   96 MB / 508 ms |           -0.9 MB |           0 |
| tfjs-wasm-ops    |       244706 B |   2.2% (3.0 per 1k calls) |          320.7 µs |   279 MB |   90 MB / 497 ms |           -0.1 MB |           0 |
| tfjs-node-layers |       347295 B | 23.6% (33.3 per 1k calls) |           1.03 ms |   301 MB |  157 MB / 558 ms |            1.3 MB |           0 |
| tfjs-node-ops    |       233729 B |   0.0% (0.0 per 1k calls) |          689.1 µs |   291 MB |  152 MB / 559 ms |            0.6 MB |           0 |

### MLP online SGD step [64,128,64,8] (`mlp-train-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                94.4 µs |      10,588 |       baseline |       1.8% |           1.00 |
| tfjs-cpu-layers  |                1.19 ms |         843 |     13× slower |      40.3% |           1.00 |
| tfjs-cpu-ops     |               749.6 µs |       1,334 |    7.9× slower |      56.4% |           1.00 |
| tfjs-wasm-layers |               804.9 µs |       1,242 |    8.5× slower |      25.7% |           1.00 |
| tfjs-wasm-ops    |               493.6 µs |       2,026 |    5.2× slower |      42.4% |           1.00 |
| tfjs-node-layers |                1.27 ms |         789 |     13× slower |      33.7% |           1.00 |
| tfjs-node-ops    |               562.2 µs |       1,779 |    6.0× slower |      29.1% |           1.00 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |           16 B |  0.0% (0.0 per 1k calls) |           95.7 µs |    76 MB |     2 MB / 22 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       243681 B | 3.0% (32.7 per 1k calls) |           1.47 ms |   286 MB |   78 MB / 450 ms |            2.0 MB |           0 |
| tfjs-cpu-ops     |       107344 B | 3.1% (31.7 per 1k calls) |           1.05 ms |   287 MB |   86 MB / 445 ms |            0.9 MB |           0 |
| tfjs-wasm-layers |       484131 B |  1.8% (6.2 per 1k calls) |          841.5 µs |   275 MB |   89 MB / 526 ms |           24.9 MB |           0 |
| tfjs-wasm-ops    |       349292 B |  0.0% (0.0 per 1k calls) |          664.3 µs |   279 MB |   89 MB / 507 ms |            0.0 MB |           0 |
| tfjs-node-layers |       494206 B |  0.0% (0.0 per 1k calls) |           1.48 ms |   326 MB |  156 MB / 573 ms |            6.6 MB |           0 |
| tfjs-node-ops    |       324439 B | 1.5% (11.1 per 1k calls) |          667.6 µs |   284 MB |  146 MB / 533 ms |           -0.3 MB |           0 |

### Delta rule (Rescorla-Wagner) 16 to 4 (`delta-rule-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 243 ns |   4,109,750 |       baseline |       0.4% |           1.00 |
| tfjs-cpu-layers  |               139.5 µs |       7,166 |    573× slower |      23.4% |           1.00 |
| tfjs-cpu-ops     |               203.6 µs |       4,910 |    837× slower |     144.3% |           1.46 |
| tfjs-wasm-layers |               155.9 µs |       6,414 |    641× slower |      22.1% |           1.00 |
| tfjs-wasm-ops    |               156.5 µs |       6,390 |    643× slower |      60.7% |           1.00 |
| tfjs-node-layers |               393.9 µs |       2,539 |   1619× slower |      48.1% |           1.00 |
| tfjs-node-ops    |               378.7 µs |       2,641 |   1556× slower |      49.2% |           1.00 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            243 ns |    74 MB |     1 MB / 18 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       177160 B | 3.3% (2.6 per 1k calls) |          165.8 µs |   286 MB |   74 MB / 447 ms |           -1.1 MB |           0 |
| tfjs-cpu-ops     |       124375 B | 0.0% (0.0 per 1k calls) |          918.0 µs |   204 MB |   77 MB / 625 ms |            0.5 MB |           0 |
| tfjs-wasm-layers |       164760 B | 0.0% (0.0 per 1k calls) |          163.4 µs |   292 MB |   94 MB / 618 ms |           -0.3 MB |           0 |
| tfjs-wasm-ops    |       113025 B | 0.0% (0.0 per 1k calls) |          177.9 µs |   286 MB |   89 MB / 552 ms |           -0.3 MB |           0 |
| tfjs-node-layers |       166420 B | 0.0% (0.0 per 1k calls) |          540.4 µs |   332 MB |  156 MB / 543 ms |           -0.7 MB |           0 |
| tfjs-node-ops    |       101791 B | 0.0% (0.0 per 1k calls) |          412.5 µs |   297 MB |  150 MB / 551 ms |            0.1 MB |           0 |

### Delta rule (Rescorla-Wagner) 256 to 64 (`delta-rule-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                56.4 µs |      17,726 |       baseline |       2.2% |           1.00 |
| tfjs-cpu-layers  |               655.5 µs |       1,525 |     12× slower |      40.5% |           1.00 |
| tfjs-cpu-ops     |               684.6 µs |       1,461 |     12× slower |      41.5% |           1.00 |
| tfjs-wasm-layers |               296.4 µs |       3,374 |    5.3× slower |      24.8% |           1.00 |
| tfjs-wasm-ops    |               276.3 µs |       3,620 |    4.9× slower |      41.1% |           1.00 |
| tfjs-node-layers |               495.6 µs |       2,018 |    8.8× slower |      30.1% |           1.00 |
| tfjs-node-ops    |               544.5 µs |       1,836 |    9.7× slower |      57.8% |           1.00 |

| Implementation   | JS heap / call |         GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | -----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B |  0.0% (0.0 per 1k calls) |           56.7 µs |    77 MB |     1 MB / 17 ms |            0.0 MB |         n/a |
| tfjs-cpu-layers  |       500769 B | 2.4% (26.1 per 1k calls) |          841.3 µs |   278 MB |   73 MB / 504 ms |            0.0 MB |           0 |
| tfjs-cpu-ops     |       451013 B | 5.9% (52.6 per 1k calls) |          794.5 µs |   280 MB |   80 MB / 437 ms |            0.0 MB |           0 |
| tfjs-wasm-layers |       164748 B |  0.0% (0.0 per 1k calls) |          309.0 µs |   284 MB |   88 MB / 533 ms |           -0.8 MB |           0 |
| tfjs-wasm-ops    |       113462 B |  0.0% (0.0 per 1k calls) |          314.3 µs |   285 MB |   89 MB / 514 ms |           -0.6 MB |           0 |
| tfjs-node-layers |       166585 B |  0.0% (0.0 per 1k calls) |          500.7 µs |   300 MB |  159 MB / 555 ms |           -0.2 MB |           0 |
| tfjs-node-ops    |       101555 B |  0.0% (0.0 per 1k calls) |          554.6 µs |   295 MB |  154 MB / 583 ms |           -0.3 MB |           0 |

### Elman RNN step 16-16-4 (`rnn-forward-small`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                 875 ns |   1,142,412 |       baseline |       2.1% |           1.00 |
| tfjs-cpu-layers  |                67.5 µs |      14,817 |     77× slower |      44.2% |           1.00 |
| tfjs-cpu-ops     |                14.1 µs |      70,844 |     16× slower |      28.9% |           1.00 |
| tfjs-wasm-layers |                75.1 µs |      13,308 |     86× slower |      25.5% |           1.00 |
| tfjs-wasm-ops    |                17.9 µs |      55,762 |     20× slower |      76.0% |           1.00 |
| tfjs-node-layers |               231.4 µs |       4,322 |    264× slower |      11.6% |           1.00 |
| tfjs-node-ops    |                61.6 µs |      16,236 |     70× slower |       5.5% |           1.00 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |            878 ns |    75 MB |     1 MB / 19 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       173394 B | 3.2% (4.0 per 1k calls) |           81.3 µs |   208 MB |   90 MB / 492 ms |            0.5 MB |           0 |
| tfjs-cpu-ops     |        71817 B | 1.9% (1.1 per 1k calls) |           16.0 µs |   208 MB |   79 MB / 427 ms |            1.5 MB |           0 |
| tfjs-wasm-layers |       126707 B | 2.5% (1.8 per 1k calls) |           83.8 µs |   275 MB |   91 MB / 584 ms |            2.3 MB |           0 |
| tfjs-wasm-ops    |        36431 B | 0.0% (0.0 per 1k calls) |           25.6 µs |   274 MB |   91 MB / 519 ms |            0.0 MB |           0 |
| tfjs-node-layers |       158117 B | 0.9% (4.4 per 1k calls) |          247.9 µs |   285 MB |  170 MB / 643 ms |            0.6 MB |           0 |
| tfjs-node-ops    |        28367 B | 0.0% (0.0 per 1k calls) |           64.9 µs |   279 MB |  151 MB / 656 ms |            0.1 MB |           0 |

### Elman RNN step 64-128-8 (`rnn-forward-large`)

| Implementation   | Time / sample (median) | Samples / s | vs neuro-learn | Noise (CV) | CPU cores used |
| ---------------- | ---------------------: | ----------: | -------------: | ---------: | -------------: |
| **neuro-learn**  |                29.5 µs |      33,841 |       baseline |       2.7% |           1.00 |
| tfjs-cpu-layers  |               150.3 µs |       6,652 |    5.1× slower |      70.3% |           1.00 |
| tfjs-cpu-ops     |                62.1 µs |      16,098 |    2.1× slower |      64.5% |           1.00 |
| tfjs-wasm-layers |                79.0 µs |      12,659 |    2.7× slower |      17.8% |           1.00 |
| tfjs-wasm-ops    |                21.8 µs |      45,963 |    1.4× faster |      72.6% |           1.00 |
| tfjs-node-layers |               273.5 µs |       3,657 |    9.3× slower |      29.7% |           1.00 |
| tfjs-node-ops    |                61.2 µs |      16,349 |    2.1× slower |       5.4% |           1.00 |

| Implementation   | JS heap / call |        GC share of time | CPU time / sample | Peak RSS | Load: RSS / time | RSS growth in run | Tensor leak |
| ---------------- | -------------: | ----------------------: | ----------------: | -------: | ---------------: | ----------------: | ----------: |
| **neuro-learn**  |            0 B | 0.0% (0.0 per 1k calls) |           30.0 µs |    73 MB |     2 MB / 22 ms |            0.3 MB |         n/a |
| tfjs-cpu-layers  |       248146 B | 0.9% (3.0 per 1k calls) |          229.1 µs |   282 MB | 198 MB / 1521 ms |            1.4 MB |           0 |
| tfjs-cpu-ops     |       129096 B | 0.8% (1.3 per 1k calls) |           86.0 µs |   200 MB |   68 MB / 436 ms |            2.0 MB |           0 |
| tfjs-wasm-layers |       129038 B | 0.0% (0.0 per 1k calls) |           77.4 µs |   296 MB |  140 MB / 656 ms |           -0.2 MB |           0 |
| tfjs-wasm-ops    |        36447 B | 0.0% (0.0 per 1k calls) |           30.6 µs |   271 MB |   89 MB / 536 ms |            0.0 MB |           0 |
| tfjs-node-layers |       161105 B | 0.7% (3.7 per 1k calls) |          324.1 µs |   296 MB |  213 MB / 715 ms |            0.1 MB |           0 |
| tfjs-node-ops    |        28367 B | 0.0% (0.0 per 1k calls) |           64.7 µs |   279 MB |  150 MB / 575 ms |            0.1 MB |           0 |

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
