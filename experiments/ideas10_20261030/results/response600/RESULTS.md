# First response-query gate: no task winner

Frozen source `80fb8bd`, eight fresh600-update runs, two tasks ×two seeds ×direct/action-annihilator. All40 training/health/information/query/planning endpoints exit0; all8 full-state audits pass. Four real two-step runs plus four explicit gradient audits passed first. All550 original initial tensors equal the respective TF controls, and each pair's12,480 extra head parameters are identical. The auxiliary path adds seven clean predictor calls during training; the native inference/planner is unchanged. All runs and idle dispatchers are held/closed.

| Same50-case strict validation | TF600 reference, seeds1/2 | Direct query, seeds1/2 | Action-annihilated query, seeds1/2 |
|---|---:|---:|---:|
| TwoRoom | 6 / 8% | 4 / 10% | 2 / 10% |
| Cube | 14 / 10% | 2 / 4% | 6 / 6% |

Direct is the equal-capacity/compute mechanism control; TF is the historical600-budget reference. These are small, immature gates during the1249-step warmup, not mature benchmark estimates. There is no consistent advantage to promote, and no justification for a long continuation. Raw successes and exact manifest hashes are retained under `planning-*`.

## What was actually learned

For the action-annihilator, query error is nearly the energy of its fixed projected target. A temporally constant predicted query is annihilated by P and has exactly that error. On64 fixed held-out windows:

| Task/seed | Learned projected MSE | Zero projected prediction MSE | Effect of replacing initial state |
|---|---:|---:|---:|
| TwoRoom1 | .000880638 | .000880527 | −2.71e−10 |
| TwoRoom2 | .000879262 | .000880527 | +9.09e−12 |
| Cube1 | .003404723 | .003409041 | +6.16e−8 |
| Cube2 | .003408624 | .003409041 | +2.29e−8 |

Action permutations also barely change the projected prediction. Thus the new target's algebraic property did not translate into a functionally useful learned response at this gate. A lower residual loss than direct supervision is not evidence of better dynamics: the projected target itself has much less energy.

TwoRoom clean position R² is roughly.18–.23 on x and.44 on y across both arms; at posterior sigma.1 those position R² values are negative. Cube retains more clean positional information, but small query-state effects and poor actual planning remain. These diagnostic labels never enter training. This does not imply that the native clean latent planner ignores all state; it is specifically the auxiliary readout being tested.

## One justified next question

The shared initial-image reconstruction error is about.149/.214 on TwoRoom, while projected outcome energy is about.00088. The original unit-scale auxiliary sums these very different magnitudes. This motivates one controlled question, not an automatic rescue claim: does relative target-energy normalization make the new channel learn, and does alignment to the executed actions outperform a generic temporal contrast of the same rank?

The next separate gate compares identity, action-nullspace, and fixed random projectors, with the same relative loss and initial grounding. The random projector removes constants/time and matches the action projector's rank. The denominator is detached per-clip target energy plus the square of one normalized8bit image level. Report low-energy/floor-dominated clips; relative weighting can overemphasize unpredictable changes and is not known to improve the zero-state gradient at exact collapse. No extra idea is counted for this normalization, and no existing checkpoint is extended beyond600.
