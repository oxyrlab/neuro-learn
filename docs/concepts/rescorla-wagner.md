---
id: learner/rescorla-wagner
kind: learner
status: experimental
files: [src/learning/rescorla-wagner.ts]
tests: [src/learning/rescorla-wagner.test.ts]
---

# Rescorla–Wagner

**Purpose:** classical conditioning: associate cues with an outcome strength; supports extinction and blocking.

## Parameters

| Name                | Type   | Default | Range | Description                  |
| ------------------- | ------ | ------- | ----- | ---------------------------- |
| inputSize           | int    | —       | >= 1  | Number of cues               |
| outputSize          | int    | —       | >= 1  | Number of outcomes           |
| maxStrength         | number | 1       | > 0   | Clamp on `\|V\|`             |
| forgetRatePerSecond | number | 0       | >= 0  | Exponential forgetting (1/s) |

## State

`V`, row-major `outputSize × inputSize` floats.

## Behavior

- `predict`: `out = V · features`.
- `learn`: for each outcome `j`: `error = target[j] − Σ V[j][i]·x[i]`, `V[j][i] += rate · x[i] · error`, clamped.
- `decay(dt)`: `V *= exp(−forgetRatePerSecond · dt)`.

## Errors

See [learner](learner.md).

## Example

```ts
const learner = new RescorlaWagnerLearner({ inputSize: 4, outputSize: 1 });
learner.learn(Float32Array.of(1, 0, 0, 0), Float32Array.of(1), 0.2);
```

## Gotchas

Cues presented together share one prediction error, which is why a pre-trained cue blocks learning about a new one.
