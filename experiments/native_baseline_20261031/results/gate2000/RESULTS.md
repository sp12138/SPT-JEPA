# Native LeWM2000: learning progress and fixed train-only BN control

The 2000-step model has learned substantially more useful control than at600. Fixed train-only BN calibration reveals TwoRoom82/76% and Cube14/16%; raw deployment remains8/0% and12/6%. The normalization mismatch hides the TwoRoom learning signal. It is not evidence of a broken native loss, and the posthoc calibrated variant must remain separate from literal native inference.

| Task | Seed | Raw600 | BN600 | Raw2000 | BN2000 | Fidentity2000 | Ffree2000 | Fbounded2000 | TF2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tworoom | 1 | 10% | 0% | 8% | 82% | 72% | 74% | 76% | 2% |
| tworoom | 2 | 8% | 8% | 0% | 76% | 70% | 82% | 80% | 6% |
| cube | 1 | 8% | 8% | 12% | 14% | 8% | 10% | 8% | 12% |
| cube | 2 | 0% | 8% | 6% | 16% | 16% | 10% | 14% | 6% |

All native600/2000/raw/BN and F/TF2000 comparisons use exactly the same ordered50cases, case hash, solver, planner horizon/budget, strict success semantics and training-only action normalization. Equal2000 updates mean256,000 clips with8RGB frames/7conditioned action blocks each. Native~18.04M parameters and192-state coordinates differ from TF/F~51.1M and64-state coordinates; F also freezes its encoder. This is matched data/update/evaluator exposure, not equal capacity or compute. PreciseBN adds fixed128training clips (1024 RGB frames) without gradients.

| Task | Contrast | Mean difference | Conditional same-case95% interval |
|---|---|---:|---|
| tworoom | BN2000-minus-raw2000 | +75.0 pp | [64.0, 85.0] pp |
| tworoom | raw2000-minus-raw600 | -5.0 pp | [-11.0, 1.0] pp |
| tworoom | BN2000-minus-BN600 | +75.0 pp | [64.0, 85.0] pp |
| tworoom | BN2000-minus-TF | +75.0 pp | [64.0, 85.0] pp |
| tworoom | BN2000-minus-F-identity | +8.0 pp | [-4.0, 20.0] pp |
| tworoom | BN2000-minus-F-unconstrained | +1.0 pp | [-12.0, 14.0] pp |
| tworoom | BN2000-minus-F-bounded | +1.0 pp | [-12.0, 14.0] pp |
| cube | BN2000-minus-raw2000 | +6.0 pp | [0.0, 12.0] pp |
| cube | raw2000-minus-raw600 | +5.0 pp | [-1.0, 12.0] pp |
| cube | BN2000-minus-BN600 | +7.0 pp | [-2.0, 16.0] pp |
| cube | BN2000-minus-TF | +6.0 pp | [-2.0, 14.0] pp |
| cube | BN2000-minus-F-identity | +3.0 pp | [-5.0, 11.0] pp |
| cube | BN2000-minus-F-unconstrained | +5.0 pp | [-3.0, 13.0] pp |
| cube | BN2000-minus-F-bounded | +4.0 pp | [-4.0, 12.0] pp |

Intervals resample the50cases jointly across the two fixed training seeds. They do not establish stability over a population of training seeds. The calibrated TwoRoom result is competitive with F2000 and far above this TF2000 checkpoint; the small Cube differences do not establish superiority over F or TF. The fixed-weight raw/BN contrast isolates the deployment-statistics intervention. LeWM-versus-TF/F changes architecture, objective and capacity jointly, so it cannot attribute the gap specifically to SIGReg or a single learned mechanism.

| Task | Seed | Raw clean/copy600→2000 | Raw teacher/copy2000 | Raw clean/shifted-observed-copy2000 | Train preciseBN one-step MSE600→2000 |
|---|---:|---|---:|---:|---|
| tworoom | 1 | 333.212→8.183 | 7.222 | 15.878 | 0.6343→0.0378 |
| tworoom | 2 | 346.946→5.478 | 3.506 | 10.448 | 0.5815→0.0430 |
| cube | 1 | 1.025→0.387 | 0.203 | 0.985 | 0.1765→0.0342 |
| cube | 2 | 5.277→0.517 | 0.273 | 1.300 | 0.1562→0.0355 |

Cube raw clean forecasts now beat retaining the initial state for both seeds. Wrong initial states and wrong actions substantially increase error, supporting learned action-conditioned dynamics rather than a copy shortcut. TwoRoom raw forecast errors remain dominated by deployment inconsistency; its wrong-action error effects are mixed, so raw sensitivity alone cannot claim useful prediction. Teacher forcing and shifted-observed copy receive extra real observations and are diagnostic controls. Only raw2000 has the64window audit; calibrated planning success and training replay must not be relabeled as calibrated heldout rollout MSE. MSE is in each learned coordinate system, not a common physical error unit.

The calibration rule was fixed before2000 outcomes: identical training128clip manifests as600 and shared across seeds, observation BN1024vectors first, then prediction BN896vectors, global actual-count mean/unbiased variance, FP32 evaluation with dropout off. Direct CPU tensor comparison verifies exactly four running_mean/running_var buffers changed, all other state bitexact, and num_batches_tracked remains2000. Fixed vs batch-BN prediction discrepancies are below5e-7; single-clip vs full-batch replay errors are below3e-12. No validation examples or physical/optimal-action labels fit calibration.

Integrity: all28 endpoints exit0; four2000 checkpoint archives match last.ckpt hashes, portable tensors match full-state and state audits, native AdamW has one group/297 parameter tensors with exact optimizer/scheduler roundtrip, scheduler step2000 and sample cursor256000. Config semantics match600; original fullstate600 was resumed, never the calibrated export. Native training source remains7a1ab4f; helper source0934d3c is independently hash-checked. Native MSE+.09SIGReg, target gradients, BN, Cube smoothed_dim10, history7/window8 and per-update scheduler are preserved. All four gradient audits pass.

Recommendation: a bounded continuation of all four original native2000 full states to6000 is justified as a fair baseline budget, not a new method promotion. Actual Cube state/action learning and TwoRoom deployment performance improve strongly beyond the1249-step warmup. Keep the native objective/trainer/scheduler unchanged, preserve raw and the same predeclared train-only BN variant at the endpoint, and stop for review at6000. No launch follows from this report. Do not claim final baseline maturity, broad two-task superiority, or an unmodified official-recipe reproduction.
