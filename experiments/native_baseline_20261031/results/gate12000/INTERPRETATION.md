Native LeWM's Cube maturity check is complete: all 26 bounded endpoints passed, both seeds stopped at 12,000 updates, and the exact idle dispatcher was closed. The monitor had already exited. Full-state archives at 8k, 10k and 12k retain model, native optimizer/scheduler, RNG and sampler cursor; all source, configuration, archive, parameter, case and planner checks passed. Staged resumes preserve the original training recipe but are not claimed bitwise identical to uninterrupted training.

| Cases | Seed | Raw 6k | Cal 6k | Raw 12k | Cal 12k | TF 12k |
|---|---:|---:|---:|---:|---:|---:|
| Primary | 1 | 24% | 16% | 26% | 30% | 18% |
| Primary | 2 | 22% | 18% | 26% | 28% | 20% |
| Fresh | 1 | 12% | 20% | 28% | 30% | 14% |
| Fresh | 2 | 14% | 16% | 26% | 26% | 16% |

The original raw baseline improves most clearly on fresh cases: +14 percentage points averaged over the two fixed seeds, with a conditional paired-case 95% bootstrap interval of [2, 26]. Primary improvement is +3 points [−5, 11]. Against TF at the same 12k update/window exposure, raw native is +7 points [−3, 17] on primary and +12 [3, 22] on fresh. This is stronger evidence of a competitive native baseline than the immature 600/2000-update observations. It does not establish universal superiority: there are only two fixed seeds, 50 cases per set, known environment repeat variation, and unequal model architecture/capacity/compute.

The independently specified train-only precise-BN variant remains separate. It changes exactly four running mean/variance buffers, no parameters or counters, using the same fixed 128 training clips and 1,024/896 vectors. At 12k it differs from raw by +3 points [−2, 9] on primary and +1 [−8, 11] on fresh. These intervals do not establish a planning benefit from calibration. Its mean advantages over TF are +10 [0, 20] and +13 [6, 21], respectively; neither raw/cal maxima nor task-specific variant selection define the baseline.

The raw model's held-out causal forecast also improves in its learned representation. The table uses the same 64 fixed windows. Initial-copy and local-copy denominators are reported because latent coordinates can change during training; the ratios do not isolate physical prediction accuracy.

| Diagnostic | Seed 1: 6k → 12k | Seed 2: 6k → 12k |
|---|---:|---:|
| Clean rollout MSE / initial-copy MSE | .1774 → .0505 | .2976 → .0543 |
| Teacher-forced one-step MSE / shifted-observed-copy MSE | .2397 → .0398 | .3447 → .0393 |
| Initial-copy MSE denominator | 1.2460 → 1.3502 | 1.3037 → 1.2609 |
| Shifted-observed-copy MSE denominator | .5992 → .7125 | .6128 → .6647 |
| Observed code standard deviation | .9983 → 1.0048 | 1.0207 → .9551 |

At 12k, raw clean rollout MSE is .0682/.0684; replacing the initial observation raises it to 1.9457/1.8581, reversing action order to 1.0311/.9818, and using another episode's actions to 1.6813/1.6147. Thus the low error is accompanied by substantial state/action dependence rather than an initial-copy shortcut. Teacher forcing receives additional real observations and is not a causal rollout result. The calibrated model's training replay agrees closely with the batch-BN diagnostic (prediction disagreement ≈4.85/4.91×10⁻⁷); single-clip versus batch fixed-BN inference disagreement is <6.3×10⁻¹³. This confirms the declared calibration relation, not a causal attribution of task improvement to BN.

Native retains its native batch-coupled objective, 192-dimensional state and approximately 18.0M trainable parameters. TF uses a different 64-dimensional state and approximately 51.1M total parameters; raw latent MSE should not be compared across these representations. Both use the matched RGB/action data, split, training-seed convention, batch/window exposure and original-command evaluation protocol. Native remains an adapted native-objective baseline, not a claim of exact official end-to-end recipe replication. All reported intervals condition on the two realized training seeds and environment outcomes; they exclude seed-population and environment-repeat uncertainty. No training beyond 12k or further evaluation was launched.

The full audited result is `/efs/prescottsun/SPT-JEPA/reports/native-cube12000-20261031/results.json` (SHA-256 `af1152dc6d04db20bdf13972e0e29e5692201c7dd100c7cbfbcf004a9c4708c0`). The adjacent `summary.json` retains all per-seed outcome vectors, paired intervals, archive hashes, and source provenance. Original model source: `7a1ab4f`; staged archive helper: `a34257e`; unchanged precise-BN helper: `a8b7792`.
