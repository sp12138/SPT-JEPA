# Independent Carrier review before the600 gate

Reviewed frozen source `dd2aeaa40e8a8050439ef036b402f289d006a257`, four completed calibrations and all12 prepared Carrier configurations. No model, optimizer, configuration or frozen snapshot was changed. The earlier eight CPU contracts were not rerun. One specifically authorized read-only precision measurement used dev2 GPU4, PID5705, for48.4 seconds, then exited successfully; GPU4 was confirmed at0MiB/0% afterward.

## Provenance and integration

All four calibrations contain128 distinct training episodes and zero validation overlap. Their source/helper/config/dataset-card/training-ID hashes match the authoritative frozen files. All three modes within each task/seed reuse exactly the same mean/scale arrays and calibration hash. No encoder warm start or legacy state calibration is enabled. All four standard-deviation floors have zero active coordinates.

| Task/seed | Minimum raw coordinate std | Mean raw coordinate std | Maximum |
|---|---:|---:|---:|
| TwoRoom1 | .114454 | .194195 | .329676 |
| TwoRoom2 | .094680 | .179205 | .274230 |
| Cube1 | .449731 | .791104 | 1.637074 |
| Cube2 | .473149 | .784068 | 1.530339 |

Same-seed encoder hashes match across tasks, as expected for the same fresh ViT initialization. The projection hash is identical everywhere. Actual exported seed1 preflight initial tensors match the calibration encoder hash, projection hash and normalization arrays for both tasks. Seed2 calibration/config provenance was checked; its later initial-model export still needs the campaign's normal comparison.

The factory dispatches Carrier in training, full-state health and information audits. Trainable parameters alone enter the existing optimizer. The health expectations correctly allow frozen encoder/pool and identity chart, while requiring active predictor/renderer/learned chart. The information assay explicitly differentiates frozen state **values**, without pretending those are encoder-parameter gradients. The benchmark planner loads the serialized Carrier target dynamically and keeps its original cost/controller. Legacy PushT-only planner/evaluation entry points do not support this target, but they are outside the current TwoRoom/Cube plan.

One minor reporting issue: the inherited `encoder_frozen` scalar measures the old scheduled-freeze window, not permanent `requires_grad=False`. It can be0 for Carrier. Use declared gradient expectations and unchanged frozen tensors as the authoritative checks.

Calibration is based on one frame per selected training episode/window, not a guarantee of exact unit variance on every training or validation distribution. This sampling choice is shared across all three arms and recorded; it neither supplies a semantic label nor guarantees representative coverage of rare contacts.

## Measured numerical train/evaluation distinction

Training enables TF32 (`float32_matmul_precision="high"`); calibration defaults to highest precision, and strict benchmark evaluation explicitly selects highest. Disabling autocast inside the frozen carrier does not disable TF32. The authorized assay compared these settings on the same first16 existing validation-manifest windows per task, using both initial and step2 bounded checkpoints from the frozen preflight. It also separated chart arithmetic from the encoder/projection by passing an identical normalized carrier into the chart.

| Task | Encoded-state RMS difference | RMS / observed coordinate std | RMS / sigma=.1 | Maximum absolute difference / sigma |
|---|---:|---:|---:|---:|
| TwoRoom | .0107501 | 1.1760% | .10750 | .59177 |
| Cube | .00312834 | .29467% | .03128 | .14503 |

Initial and step2 results agree to the reported precision. Raw-carrier RMS differences are .00183960 and .00224628, respectively. With identical chart inputs, step2 RMS differences are only8.5e-10 and2.0e-9; the measured discrepancy is almost entirely frozen encoder/projection arithmetic at this stage. All model tensors were verified unchanged.

This shared numerical difference does not show one arm has an advantage, nor establish a planning failure. It does mean that the carrier is not bitwise the same function across training and evaluation modes. The TwoRoom worst-coordinate shift is appreciable relative to posterior sigma and must not be described as exact alignment. After reviewing the measurement, the campaign owner retained the original high-training/highest-evaluation convention unchanged across all six arms. The three Carrier arms share this convention; no post-hoc precision variant is added to this gate. [Full measurement and provenance](precision-review.json); the EFS original is `/efs/prescottsun/SPT-JEPA/reports/carrier-precision-review-20261031/precision-review.json`.

## Endpoint interpretation criteria

1. **Verify the intervention actually occurred.** Frozen encoder, projection and normalization must remain unchanged. Identity chart parameters must remain unchanged. Learned charts must receive finite updates. Report raw/effective matrix norms, normalization activation, residual Lipschitz upper bound, and actual chart displacement/pair-distance change relative to calibrated u. Total parameter equality does not erase the identity arm's8,320 fewer trainable parameters.

2. **Separate factor normalization from preservation of information.** Raw W1 norm1.119 makes its normalization active from initialization. However the relevant global residual bound is `.5*||W1||2*||W2||2`. If W2 stays small, even the unconstrained chart can remain safely co-Lipschitz. Active factor normalization alone does not demonstrate an avoided destructive map. An upper bound exceeding.5 is also not proof that the true residual Lipschitz constant exceeds.5; the product bound can be loose. Actual loss of observed distinctions would be stronger evidence.

3. **Do not count standardized variance as learning.** Near-unit initial carrier variance is largely built into calibration. Clean physical probes, the same probes with the actual sigma noise, and generator/forecast state interventions answer different questions. The chart guarantees preservation relative to u in exact arithmetic, not preservation of every original-image factor, invariant noisy mutual information, or invariant linear-probe accuracy. A nonlinear invertible chart may change a linear probe without creating or deleting clean information.

4. **Require a learned transition to beat the appropriate copy baseline.** Compare teacher-forced prefix prediction against shifted observed carry, since both see the latest real state. Compare causal rollout against initial carry, since both begin from only frame0. Use per-horizon values and raw denominator energies alongside each arm's normalized errors. A smaller raw latent MSE across different charts may only reflect geometry. State/action intervention sensitivity is useful only together with better factual prediction than carry; a model that merely copies the initial state is state-sensitive too. Wrong-action comparisons against the original outcome are sensitivity checks, not ground-truth counterfactual dynamics scores.

5. **Keep planning separate.** Native P0 uses the same50 fixed cases and unchanged CEM/goal cost. Compare both seeds, report paired cases and uncertainty, and do not drop a losing task/seed. A P0 change without stronger outcome prediction may be a changed goal geometry. Conversely, a better learned transition without P0 gain leaves the Cube readout/goal-semantics issue unresolved.600 steps remain an early gate.

| Observed pattern | Supported interpretation |
|---|---|
| Identity improves over old TF; learned charts match it | Fixed measurement/calibration/frozen training is the useful change; no learned-chart gain established. |
| Both learned charts beat identity on dynamics and planning | Coordinate adaptation is useful under this budget; the bound's necessity remains unproven. |
| Bounded beats free, with free losing useful distinctions and worse dynamics/planning | Evidence supports the proposed preservation mechanism, subject to seeds/task support and actual chart activity. |
| Bounded beats free while both remain near identity and safely co-Lipschitz | An optimization/parameterization difference is plausible; do not attribute it to prevention of information deletion. |
| Probes improve, forecasts remain at carry | Observation information was retained/exposed but not transferred into causal dynamics. |
| Dynamics improve, P0 does not | World-model/readout separation remains; no successful end-to-end claim. |
| Only one seed/task wins or all arms remain near copy baselines | No robust promotion or automatic longer continuation is justified. |

The review found no source/config/statistical-leakage blocker. Numerical precision remains an explicitly measured limitation under the retained protocol. No mature superiority, academic novelty, or completed ten-idea result follows from this preflight review.
