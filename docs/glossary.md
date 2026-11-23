# Glossary

| Term               | Definition                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------ |
| Learner            | Object with fixed-size `Float32Array` state that can `predict`, `learn`, `decay`, `serialize` and `restore`. |
| Feature vector     | Input `Float32Array` describing a stimulus; values are normalized by the caller.                             |
| Target             | Desired output (or outcome strength λ) a learner moves its prediction toward.                                |
| Learning rate (α)  | Step size of an update; always a parameter, validated against documented bounds.                             |
| Modulator          | External scalar signal (reward, pain) that gates or scales learning; supplied by the caller.                 |
| Eligibility trace  | Decaying memory of recent activity that lets delayed outcomes assign credit.                                 |
| Extinction         | Weakening of an association when the cue appears without the outcome.                                        |
| Decay (forgetting) | Movement of state toward zero over time, `decay(dt)`.                                                        |
| Rng                | Injected source of randomness (`next()` in [0, 1)); never `Math.random`.                                     |
| Hot loop           | Code called every tick; must not allocate.                                                                   |
