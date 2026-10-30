# Read-only calibration diagnosis

Do not approve the original diagonal-normalized TwoRoom local600 configuration unchanged. The anomaly is principally finite-sample, coordinate-specific scale amplification, with a smaller precision contribution. A single global RMS scale resolves the measured amplification across both tasks and both fixed projections; it warrants a new immutable calibration/configuration and real preflight, not an immediate performance claim.

Frozen model source: `96aca13f99b25dcf880346a1e699318150139ae5`. Four read-only jobs used dev15 GPUs0–3 (PIDs1020727–1020730), completed in10.7/10.8/16.9/16.8 measured seconds after imports, and released all four GPUs. All original configs, checkpoints, exits and holds remain unchanged. EFS evidence: `/efs/prescottsun/SPT-JEPA/reports/local-calibration-diagnosis-20261031`; its `source.py`, `launch.json`, four `report.json`/`windows.json`/`measurements.npz` and `isotropic-counterfactual.json` preserve provenance. No validation or physical labels were read.

Each job measured both projections from the same frozen encoder patches: the original128 calibration windows and the exact native sampler's first128 training windows, all8 frames. It independently switched encoder and projection precision between high/TF32 and highest/no-TF32. No renderer, optimizer, or training update was run.

**Evidence of the cause.** Original calibration-frame coordinate standard deviations reproduce1.000000. On the native training batch, TwoRoom local coordinate-mean std becomes3.905/3.486 even at highest precision, versus3.916/3.498 under high. Same-frame precision error has normalized RMS.370/.366, mainly projection error, and cannot explain overall normalized RMS11.07/9.77. Under highest precision, the original calibration windows' future frames already give std1.53/1.60; calibrating on all8 frames would still give native-batch std3.75/3.01. Merely matching high-precision calibration to training leaves3.60/3.17. The original sampling uses one random start in each of128 random training episodes, not episode-start-only frames; its start quantiles resemble the native batch. Thus no systematic episode-start selection error was established. The evidence supports under-sampled local variation: originally low-variance spatial coordinates become highly variable on another genuine training sample. All1e-6 numerical floors were untriggered.

The alternative below preserves original per-coordinate means and uses **one scalar**

`scale² = mean_j population_variance_128frames(feature_j)`

for every coordinate, task and arm. It needs no new calibration images, tuned floor, validation parameter, labels, or model change. This is a CPU counterfactual on saved measurements only.

| Task/arm | Native high RMS, diagonal→isotropic, seeds1/2 | Analytic initial carry-rate, diagonal→isotropic, seeds1/2 | Top16 coordinate energy, diagonal→isotropic |
|---|---|---|---|
| TwoRoom local | 11.074/9.766 → .979/.994 | 442.89/438.80 → 3.453/3.650 | 66.0/67.3% → 21.8/25.7% |
| TwoRoom dense | 1.005/1.009 → 1.000/1.004 | 5.536/5.632 → 5.438/5.550 | 2.32/2.22% → 2.65/2.55% |
| Cube local | 1.088/1.038 → .971/.961 | 4.234/4.268 → 3.560/3.750 | 9.18/5.54% → 16.44/17.25% |
| Cube dense | .975/.973 → .964/.964 | 4.695/4.824 → 4.485/4.634 | 2.32/2.25% → 2.85/2.69% |

The carry-rate is the analytic expectation of the original rate under the **initial zero-output predictor**, including its rho=.8 correlated posterior noise and KL constants; it is not a replay of real step2. At initialization, predicted state is the previous noisy state, so the dominant signal contribution is `sum_j E[(z[t+1,j]−z[t,j])²]/(2 sigma² FRAME_ELEMS)`. TwoRoom original rates remain442.39/438.44 at highest precision, whereas their temporal-signal terms under high are441.86/435.49. This isolates the dominant effect from DiT fitting and floating-point precision.

**Why isotropic scaling is principled.** Empirical coordinate whitening selects the diagonal Mahalanobis metric `diag(estimated variance)^−1`; one underestimated coordinate can create an arbitrarily large task distance and temporal prediction loss. Global scaling instead uses `trace(estimated covariance)/1024`. Under any orthogonal change of measurement basis, the trace is unchanged and centered codes rotate by the same matrix, preserving pairwise Euclidean distances up to one common scalar. The observed per-coordinate-mean std falls to.627/.618 on TwoRoom local while RMS stays.979/.994; these are compatible with heterogeneous coordinate variances, not collapse.

This does not prove optimal control geometry or pose recovery. The global scale itself can still be estimated poorly under a severe whole-distribution shift. Local versus dense changes spatial averaging and the projection subspace, not merely coordinate orientation; the two measurements are not related by an established invertible rotation. Their frozen information, learned dynamics and native planning must still be compared after a reviewed real gate. No new algorithm or novelty claim follows from this normalization repair.
