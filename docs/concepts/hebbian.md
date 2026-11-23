---
id: learner/hebbian
kind: learner
status: experimental
files: [src/learning/hebbian.ts]
tests: [src/learning/hebbian.test.ts]
---

# Hebbian

**Purpose:** co-activation association with bounded weights.

## Parameters

| Name                | Type   | Default | Range  | Description                  |
| ------------------- | ------ | ------- | ------ | ---------------------------- |
| inputSize           | int    | —       | >= 1   | Pre-synaptic inputs          |
| outputSize          | int    | —       | >= 1   | Post-synaptic outputs        |
| maxWeight           | number | 1       | > 0    | Clamp on `\|W\|`             |
| weightDecay         | number | 0.1     | [0, 1] | Per-update decay `d`         |
| forgetRatePerSecond | number | 0       | >= 0   | Exponential forgetting (1/s) |

## State

`W`, row-major `outputSize × inputSize` floats.

## Behavior

- `predict`: `out = W · features`.
- `learn`: `W[j][i] += rate · (target[j]·x[i] − weightDecay · W[j][i])`, clamped. `target` is the post-synaptic activity.
- Fixed point (unclamped): `target·x / weightDecay`.
- `decay(dt)`: `W *= exp(−forgetRatePerSecond · dt)`.

## Example

```ts
const learner = new HebbianLearner({ inputSize: 2, outputSize: 1 });
learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.2);
```

## Gotchas

Negative `target` gives anti-Hebbian (weakening) updates. With `weightDecay = 0` only the clamp bounds the weights.
