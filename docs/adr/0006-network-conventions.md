# ADR 0006: Network conventions and "empty memory" for MLPs

- **Status:** accepted
- **Date:** 2026-11-23

## Context

A network with all-zero weights cannot learn (symmetry is never broken), but the learner contract says a new instance starts with empty memory. EcoSim also needs networks whose weights are evolved, not trained.

## Decision

1. Networks (`FeedforwardNetwork`, `RecurrentNetwork`) implement `Network`, not `Learner`: forward pass, a live `weights` view for mutation and crossover, and serialize/restore. They are initialized from an injected `Rng` (Xavier uniform, zero biases).
2. `MlpSgdLearner` starts with random hidden layers and a **zero output layer**, so its initial prediction is zero (for linear, tanh and relu outputs). `runLearnerContract` gains `initialState: 'all-zero' | 'zero-prediction'` (default `'all-zero'`).
3. `RecurrentNetwork` is Elman-style with no gradient training (no BPTT); its hidden state is part of the serialized state.
4. Backpropagation is verified against numerical gradients; the finite-difference step and tolerance are documented in the tests because weights are `Float32`.
5. No hard-coded size limits: layer counts and sizes are only validated as integers >= 1; hosts apply their own limits (EcoSim Rules).

## Rejected options

- Initializing every layer randomly in the MLP learner: predictions would not start at zero, breaking "empty memory".
- All-zero MLP: cannot learn.
- BPTT now: large surface area, not required by EcoSim's evolved `nn` modules.

## Consequences

Hidden-layer initial weights are state a host must not inherit as "learned memory"; offspring simply construct a new instance with their own `Rng`.
