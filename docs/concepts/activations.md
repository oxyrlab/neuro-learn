---
id: nn/activations
kind: network
status: experimental
files: [src/nn/activations.ts]
tests: [src/nn/activations.test.ts]
---

# Activations

**Purpose:** built-in activation functions with derivatives expressed through the function's own output (all backpropagation needs).

| id        | `f(x)`       | Output range | `f'` from output `y` |
| --------- | ------------ | ------------ | -------------------- |
| `tanh`    | `tanh(x)`    | (-1, 1)      | `1 − y²`             |
| `sigmoid` | `1/(1+e^−x)` | (0, 1)       | `y(1 − y)`           |
| `relu`    | `max(0, x)`  | [0, ∞)       | `y > 0 ? 1 : 0`      |
| `linear`  | `x`          | (-∞, ∞)      | `1`                  |

`getActivation(id, parameter?)` returns the entry or throws `InvalidParameterError` listing the valid ids.

## Gotchas

`relu` has a kink at 0; numerical gradient checks near 0 are unreliable (the derivative at 0 is defined as 0).
