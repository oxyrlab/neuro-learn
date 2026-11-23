---
id: nn/recurrent-network
kind: network
status: experimental
files: [src/nn/recurrent-network.ts]
tests: [src/nn/recurrent-network.test.ts]
---

# RecurrentNetwork

**Purpose:** Elman network whose hidden state persists between calls; weights are meant to be evolved (no gradient training).

## Parameters

| Name                                | Type         | Default  | Range | Description           |
| ----------------------------------- | ------------ | -------- | ----- | --------------------- |
| inputSize / hiddenSize / outputSize | int          | —        | >= 1  | Sizes                 |
| rng                                 | Rng          | —        | —     | Weight initialization |
| hiddenActivation                    | ActivationId | `tanh`   | —     | Hidden state          |
| outputActivation                    | ActivationId | `linear` | —     | Output                |

## State

`weights` (live view): `Wxh`, `Whh`, `bh`, `Why`, `by`. Serialized state = weights followed by the hidden state (`stateSize = weights + hiddenSize`).

## Behavior

One `forward` = one time step: `h' = f(Wxh·x + Whh·h + bh)`, `out = g(Why·h' + by)`. `reset()` zeroes `h`. With `tanh`, `|h| <= 1` always.

## Gotchas

Call `reset()` between independent episodes. `restore` also restores the hidden state, so a sequence continues exactly.
