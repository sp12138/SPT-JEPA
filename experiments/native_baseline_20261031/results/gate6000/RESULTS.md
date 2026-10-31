# Native LeWM6000: retain raw and calibrated variants on both tasks

The raw native world model improves substantially by6000. Its Cube advantage over calibration on primary cases does not replicate on fresh cases: raw24/22% versus calibrated16/18% becomes raw12/14% versus calibrated20/16%. Train-only BN recalibration remains a separate uniform variant. Choosing calibrated TwoRoom and raw Cube after seeing primary outcomes would create a task-specific selected system, not a single evaluated method.

| Task | Seed | Raw2000 primary | Cal2000 primary | Raw6000 primary | Cal6000 primary | Raw6000 fresh | Cal6000 fresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| tworoom | 1 | 8% | 82% | 80% | 82% | 76% | 84% |
| tworoom | 2 | 0% | 76% | 72% | 78% | 70% | 88% |
| cube | 1 | 12% | 14% | 24% | 16% | 12% | 20% |
| cube | 2 | 6% | 16% | 22% | 18% | 14% | 16% |

| Task | Reference6000 | Primary seed1/2 | Fresh seed1/2 |
|---|---|---|---|
| tworoom | F-bounded | 92% / 88% | 80% / 94% |
| tworoom | F-identity | 80% / 86% | 74% / 80% |
| tworoom | F-unconstrained | 90% / 94% | 82% / 86% |
| tworoom | TF | 2% / 4% | 4% / 2% |
| cube | F-identity | 10% / 12% | 12% / 10% |
| cube | TF | 24% / 14% | 14% / 18% |

All comparisons verify identical ordered50cases, case hashes, planner, budget, strict success and normalization within each case set. Primary and fresh remain separate; fresh means new pairs from the existing validation episode pool, not an independent test set. Cube Ffree/Fbounded were not continued to6000, so no missing result is filled with its2000 score. Native~18.04M parameters versus old F/TF~51.1M; architectures, latent dimensions and frozen/trainable capacity differ. This is matched6000 gradient-update exposure, not equal capacity or compute. BN calibration adds the same fixed128 training clips without gradients.

| Task | Case set | Cal − raw mean | Same-case conditional95% interval |
|---|---|---:|---|
| tworoom | primary | +4.0pp | [-5.0, 13.0]pp |
| tworoom | fresh | +13.0pp | [3.0, 23.0]pp |
| cube | primary | -6.0pp | [-15.0, 2.0]pp |
| cube | fresh | +5.0pp | [-3.0, 14.0]pp |

Intervals resample cases jointly across the two fixed training seeds; they are not uncertainty over the training-seed population. Full per-seed and F/TF paired results are in results.json.

| Task | Seed | Raw clean / initial copy2000→6000 | Raw clean / shifted-observed copy6000 | Wrong-order MSE / clean MSE6000 |
|---|---:|---|---:|---:|
| tworoom | 1 | 8.183→0.508 | 0.797 | 2.126 |
| tworoom | 2 | 5.478→0.603 | 0.872 | 1.892 |
| cube | 1 | 0.387→0.177 | 0.369 | 4.692 |
| cube | 2 | 0.517→0.298 | 0.633 | 2.981 |

All four raw models now predict better than retaining the initial state and better than the shifted-observed copy control; wrong initial states or wrong action order substantially increase error. This is useful action-conditioned latent learning. Teacher/shifted-observed controls receive extra real observations. The same-case planning gains supply separate task evidence; latent MSE is not physical accuracy or a common scale across architectures.

The large TwoRoom raw deployment deficit at2000 largely resolves during continued native training. Cube calibration effects change with the evaluation case set; primary-only raw superiority is therefore not a stable conclusion. Precise training moments are not a universally beneficial controller correction. This experiment does not isolate whether finite calibration support, training-versus-planning context distributions or another mechanism explains the reversal. No tuning or new assay is added to select among them.

Integrity:32 original endpoints and4 independent raw-fresh endpoints completed0. Full6000 archives match last.ckpt/portable hashes; all four full-state/gradient checks pass, native optimizer order and scheduler roundtrip pass, cursor768000 and cross-epoch sampler checks pass. Original2000 full states were resumed;4000/6000 exports and source7a remain preserved. Direct CPU comparison confirms calibration changes exactly four BN mean/variance buffers, not parameters or tracked counts; fixed training128 manifests and1024/896 vector counts are unchanged. Calibrated replay is training-only; there is no calibrated heldout state audit.

Both idle owned dispatchers are closed and their roots held. No further training is launched. Retain raw native as its original deployment result and preciseBN as a fully reported separate variant across both tasks; do not publish a posthoc task-wise maximum.
