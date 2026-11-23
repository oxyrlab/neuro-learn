---
id: learner/eligibility-trace
kind: learner
status: experimental
files: [src/learning/eligibility-trace.ts]
tests: [src/learning/eligibility-trace.test.ts]
---

# Eligibility trace

**Purpose:** credit assignment for delayed outcomes: cues seen shortly before a reward or pain get credit.

## Parameters

| Name                     | Type   | Default | Range | Description                        |
| ------------------------ | ------ | ------- | ----- | ---------------------------------- |
| inputSize                | int    | —       | >= 1  | Number of cues                     |
| outputSize               | int    | —       | >= 1  | Number of outcomes                 |
| traceTimeConstantSeconds | number | —       | > 0   | Trace falls to 1/e after this time |
| maxStrength              | number | 1       | > 0   | Clamp on `\|V\|`                   |
| forgetRatePerSecond      | number | 0       | >= 0  | Forgetting of strengths (1/s)      |

## State

Strengths `V` (`outputSize × inputSize`) followed by the trace (`inputSize`).

## Behavior

- `observe(features)`: `trace[i] = max(trace[i], features[i])` (replacing trace). Extra method, not part of `Learner`.
- `decay(dt)`: `trace *= exp(−dt/τ)`; `V *= exp(−forgetRatePerSecond · dt)`. Time passes only here.
- `predict`: `V · features` (current cues); never changes state.
- `learn`: calls `observe(features)`, then the delta rule using the **trace** as cue strength.

Typical tick: `decay(dt)`, `observe(features)`, `predict(...)`; call `learn(...)` on outcome.

## Example

```ts
learner.observe(Float32Array.of(1, 0, 0, 0));
learner.decay(1);
learner.learn(new Float32Array(4), Float32Array.of(1), 0.5); // credit flows to cue 0
```

## Gotchas

- Call `decay` exactly once per elapsed interval; `observe` does not advance time.
- Cue strengths must be **>= 0** in `observe` **and** `learn` (a replacing trace cannot represent negative cues); negative values throw `InvalidParameterError` and leave the state untouched.
- `restore` rejects a negative trace and strengths beyond `maxStrength`. The trace itself is not bounded by `maxStrength`: it mirrors the cues you pass.
