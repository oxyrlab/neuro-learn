---
id: learner/mlp-sgd
kind: learner
status: experimental
files: [src/learning/mlp-sgd.ts]
tests: [src/learning/mlp-sgd.test.ts]
---

# MLP with online SGD

**Purpose:** small multilayer perceptron learner for cases where the linear learners are not expressive enough (e.g. XOR-like relations).

## Parameters

| Name                | Type         | Default  | Range                       | Description                                            |
| ------------------- | ------------ | -------- | --------------------------- | ------------------------------------------------------ |
| layerSizes          | number[]     | —        | >= 2 entries, each int >= 1 | Input, hidden..., output                               |
| rng                 | Rng          | —        | —                           | Hidden-layer initialization                            |
| hiddenActivation    | ActivationId | `tanh`   | —                           |                                                        |
| outputActivation    | ActivationId | `linear` | —                           |                                                        |
| maxWeight           | number       | 5        | > 0                         | Clamp on `\|w\|` (also applied to the initial weights) |
| forgetRatePerSecond | number       | 0        | >= 0                        | Exponential forgetting (1/s)                           |

## State

All weights of an internal [FeedforwardNetwork](feedforward-network.md).

## Behavior

- Initial state: random hidden layers, **zero output layer** → predicts 0 for `linear`/`tanh`/`relu` outputs (0.5 for `sigmoid`). Contract option `initialState: 'zero-prediction'`.
- `learn`: forward, [backprop](backprop.md), `w ← clamp(w − rate · ∂loss/∂w)`. If the forward pass overflowed Float32 (huge inputs with unbounded activations such as `relu`/`linear`), the loss or gradient is non-finite and the update is **skipped** so the weights are never corrupted.
- `decay(dt)`: `w *= exp(−forgetRatePerSecond · dt)`.

## Example

```ts
const learner = new MlpSgdLearner({
  layerSizes: [2, 6, 1],
  rng: createSeededRng(1),
  maxWeight: 10,
});
learner.learn(Float32Array.of(1, 0), Float32Array.of(1), 0.1);
```

## Gotchas

`restore` rejects weights beyond `maxWeight`. `restore` overwrites everything, so an instance created with a different seed behaves identically after it.
