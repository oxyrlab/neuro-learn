---
id: nn/feedforward-network
kind: network
status: experimental
files: [src/nn/network.ts, src/nn/feedforward-network.ts]
tests: [src/nn/feedforward-network.test.ts]
---

# FeedforwardNetwork

**Purpose:** fully connected network (forward pass only) with mutable weights; the `nn` brain building block.

## Parameters

| Name             | Type         | Default  | Range                       | Description                                              |
| ---------------- | ------------ | -------- | --------------------------- | -------------------------------------------------------- |
| layerSizes       | number[]     | —        | >= 2 entries, each int >= 1 | Input, hidden..., output sizes                           |
| rng              | Rng          | —        | —                           | Weight initialization                                    |
| hiddenActivation | ActivationId | `tanh`   | —                           | Hidden layers                                            |
| outputActivation | ActivationId | `linear` | —                           | Output layer                                             |
| zeroOutputLayer  | boolean      | false    | —                           | Start with a zero output layer (zero initial prediction) |

## State

`weights`: for each layer the row-major `W[out][in]` then bias `b[out]`. `weights` is a **live view**: write to it for mutation or crossover. Total length: `Σ out·(in + 1)`.

## Behavior

Xavier-uniform init (`limit = sqrt(6/(in+out))`), zero biases. `forward` writes into the caller's `out`; it leaves per-layer values in `layerOutputs` for [backprop](backprop.md).

## Example

```ts
const network = new FeedforwardNetwork({ layerSizes: [2, 3, 1], rng: createSeededRng(1) });
const out = new Float32Array(1);
network.forward(Float32Array.of(0.5, -0.5), out);
```
