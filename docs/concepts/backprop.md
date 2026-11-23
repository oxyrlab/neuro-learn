---
id: nn/backprop
kind: network
status: experimental
files: [src/nn/backprop.ts]
tests: [src/nn/backprop.test.ts]
---

# Backpropagator

**Purpose:** gradient of `½ Σ (output − target)²` with respect to every weight of a `FeedforwardNetwork`.

## Usage

Call `network.forward(...)` first, then `computeGradient(target, gradient)`; it returns the loss and fully overwrites `gradient` (same layout as `network.weights`). Buffers are allocated once in the constructor.

## Verification

Tests compare analytic gradients with central finite differences for every hidden/output activation pair (`EPSILON = 5e-3`, relative `TOLERANCE = 2e-3`). The step is large because weights are `Float32`.
