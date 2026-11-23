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

## Allocation

`npm run check:allocation` (run with `--expose-gc`) forces a collection, runs 4000-call windows and reads the heap growth; the median over 50 windows must be <= 2 bytes/call. A deliberately allocating control must read >= 50 bytes/call, which proves the detector works. Every `predict`, `learn`, `decay`, `observe` and `forward` is checked with an integer bound (`1`) and a fractional bound (`0.9`), because V8 takes different paths for them.

Lessons (see [ADR 0007](adr/0007-hot-path-numerics-and-failure-handling.md)):

- A helper that returns a fractional number per element makes V8 box a heap number each call; work on whole buffers instead.
- Merging an integer-valued bound and a computed value into one variable (`next = limit`) boxes a number per weight; store inside each branch.
- Reading `heapUsed` inside the measured loop is unreliable (it jumps by whole memory pages); measure per short window.

## Denormals

Weights that decay into the Float32 denormal range (below about 1.2e-38) cause no measurable slowdown, because values are widened to float64 when read (`predict` ratio 1.05 for rescorla-wagner and 0.79 for the MLP, i.e. within noise). No flush-to-zero option is needed.
