# Ordered-query shortcut audit

This is a CPU audit of the proposed target, not training of TF/LeWM. Separate existing train/validation episode sets; each episode contributes one 8-frame window and 8 randomly permuted image-prototype queries. Prototype frame indices/timestamps are not classifier inputs. The readouts use identical query assignments in every control.

Features: raw 35-step logged actions; three fixed 8×8 RGB query prototypes; optionally the initial fixed 8×8 RGB image. Thus “state-blind” means no initial observation, not no visual information: query prototypes themselves contain future appearances. Each whole/half target has its own readout, trained simultaneously as separate output columns.

Fixed classifiers: linear ridge and Gaussian-kernel ridge; ridge coefficient 1 throughout. Feature normalization uses training rows only. Gaussian bandwidth is the median positive squared training-pair distance, without tuning. Adding initial features changes the input dimension and kernel geometry; this is not a capacity-matched functional proof or an exhaustive nonlinear information test.

| Task / classifier / inputs | Train combined | Val whole | Val first half | Val second half | Val combined | Val macro recall (mean3) |
|---|---:|---:|---:|---:|---:|---:|
| tworoom / training-majority constant | 39.81% | 66.99% | 25.78% | 26.86% | 39.88% | 30.00% |
| tworoom / linear/actions | 50.13% | 58.40% | 25.20% | 25.68% | 36.43% | 31.11% |
| tworoom / linear/queries | 71.61% | 58.69% | 25.68% | 26.86% | 37.08% | 33.04% |
| tworoom / linear/queries_actions | 76.99% | 53.91% | 23.24% | 27.34% | 34.83% | 31.76% |
| tworoom / linear/queries_actions_initial | 80.83% | 56.93% | 24.90% | 23.14% | 34.99% | 31.73% |
| tworoom / rbf/queries_actions | 82.36% | 70.51% | 31.45% | 33.59% | 45.18% | 36.79% |
| tworoom / rbf/queries_actions_initial | 82.78% | 71.39% | 34.77% | 36.33% | 47.49% | 38.90% |
| cube / training-majority constant | 44.73% | 74.80% | 28.61% | 28.22% | 43.88% | 30.00% |
| cube / linear/actions | 57.29% | 63.09% | 24.22% | 27.15% | 38.15% | 29.94% |
| cube / linear/queries | 88.31% | 63.67% | 27.44% | 27.05% | 39.39% | 32.93% |
| cube / linear/queries_actions | 93.55% | 59.18% | 27.15% | 23.54% | 36.62% | 32.00% |
| cube / linear/queries_actions_initial | 93.85% | 57.62% | 25.98% | 23.63% | 35.74% | 33.14% |
| cube / rbf/queries_actions | 85.58% | 74.32% | 36.43% | 34.08% | 48.27% | 36.44% |
| cube / rbf/queries_actions_initial | 84.57% | 74.90% | 38.77% | 35.45% | 49.71% | 37.57% |

| Task / classifier | Initial image gain (combined) | Episode bootstrap 95% CI | Whole / first / second gain |
|---|---:|---:|---:|
| tworoom / linear | +0.16pp | [-1.99, +2.34]pp | +3.03/+1.66/-4.20pp |
| tworoom / rbf | +2.31pp | [+0.98, +3.68]pp | +0.88/+3.32/+2.73pp |
| cube / linear | -0.88pp | [-3.06, +1.27]pp | -1.56/-1.17/+0.10pp |
| cube / rbf | +1.43pp | [+0.33, +2.57]pp | +0.59/+2.34/+1.37pp |

Bootstrap resamples entire validation episodes (5,000 fixed-seed draws), retaining all queries and segments within each episode. Training/validation identities and per-class recalls remain in the JSON. The intervals quantify this fixed-fit paired readout comparison; they do not include retraining-seed uncertainty.

A positive initial-image increment supports state information being useful to this readout, not that a new world-model loss must encode or use it. A missing increment leaves the state-forcing claim unsupported; it does not prove the initial image contains no information. Actions and future query prototypes can already leak substantial task progress in expert trajectories. Native-world-model interventions and actual planning remain required.
