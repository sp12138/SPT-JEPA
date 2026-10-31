# Cube action trajectory support: fixed train128 CPU diagnosis

The original planner samples and updates a diagonal Gaussian in 125 coordinates (25 raw steps × 5 actions). Training trajectories have strong temporal correlations, including across the five-step block boundaries. The old fixed pool shows a large joint-distribution mismatch, but proximity to the training distribution is neither sufficient nor monotonic for task success. This is evidence for a bounded common-controller control, not proof of a planning cause or a new learning algorithm.

## Inputs and estimation

Only the first 25 raw actions of each of the original 128 training clips were additionally read: 64,000 float32 payload bytes, 128 distinct training episodes, no validation episode overlap. Exact manifest/card/metadata/row/episode checks and per-clip action hashes are saved in the EFS full results. No RGB, encoder, renderer, simulator, training, or new planning was called. Existing fixed-pool trajectories and their physical outcomes were read only after training statistics were frozen.

The dataset's existing training-action normalization is used. Flattened trajectories have 125 coordinates. Ledoit–Wolf covariance uses only train128, with automatic shrinkage 0.15193379; no outcome selects an estimator or coefficient. Support is squared Mahalanobis distance divided by 125. Each of 128 training reference scores holds out its entire episode and refits on the other 127; candidate scores use the full128 fit. The LOO 95th-percentile reference is 2.241633. This is a descriptive score, not a calibrated OOD probability. In-sample mean .463 versus LOO .903 exposes substantial in-sample optimism.

## Temporal structure

Empirical covariance has participation rank 14.04, entropy rank23.41, and35 components explain95% variance; numerical rank is still125. Shrinkage preserves full rank and raises participation to18.70, with86 components for95%. This does not justify a hard low-rank truncation. Channel-wise lag1 correlations are .802/.838/.947/.840/.906; block-internal correlations .806/.853/.948/.868/.908 and cross-block correlations .785/.763/.942/.710/.900. Structure is not confined to individual action blocks.

All125 empirical coordinate variances are nonzero. Only2 of640 channel/clip sequences are exactly constant, both action channel3; this is not a constant-action artifact. Channel4's observed raw range is[-.462,.700], versus[-1,1] in the other four; whole-cohort normalized standard deviations are .994/.969/1.017/.976/1.017. Long-lag correlations can be negative (channel2 lag20 −.755), so the full covariance is more than generic low-pass smoothing and may capture clip phase as well as useful dynamics.

DCT-II power uses25 raw steps with orthonormal scaling. High-frequency fraction is power at indices7–24 divided by all power; DC is retained in the denominator. Mean high-frequency fraction is9.91% in training,71.83% for random Gaussian candidates,47.40–58.84% for old CEM selected trajectories,7.46% for recorded actions and12.62% for their low-noise neighborhood. These old CEM trajectories are final plans from historical models, not all intermediate populations or current models' own proposals.

## All fixed-pool groups

Hold is the fraction of individual candidates satisfying the stored physical hold test, **not** per-case planning success or candidate coverage. Each row retains50 cases; multi-candidate rows contain dependent samples. Outlier confidence intervals in JSON resample cases, not candidates. Recorded-repeat is a duplicate control, not an independent demonstration.

| Group | Candidates/case | Mean support | Above LOO95 | High-frequency power | Outside Box coordinates | Candidate hold |
|---|---:|---:|---:|---:|---:|---:|
| cem_lewm | 1 | 2.601 | 40.0% | 47.4% | 3.66% | 22.0% |
| cem_lower-beta-s1 | 1 | 2.390 | 24.0% | 56.7% | 2.48% | 10.0% |
| cem_lower-beta-s2 | 1 | 2.289 | 22.0% | 57.8% | 2.16% | 16.0% |
| cem_original-s1 | 1 | 2.655 | 18.0% | 57.0% | 2.82% | 8.0% |
| cem_original-s2 | 1 | 2.124 | 22.0% | 58.8% | 1.79% | 10.0% |
| cem_temporal-forecast-s1 | 1 | 1.821 | 16.0% | 55.4% | 1.60% | 20.0% |
| cem_temporal-forecast-s2 | 1 | 1.854 | 16.0% | 55.6% | 1.73% | 10.0% |
| gaussian | 14 | 4.045 | 100.0% | 71.8% | 2.84% | 6.0% |
| recorded_oracle | 1 | 0.652 | 0.0% | 7.5% | 0.00% | 56.0% |
| recorded_perturb_0.25 | 8 | 0.903 | 0.0% | 12.6% | 4.87% | 52.0% |
| recorded_perturb_1.0 | 7 | 4.773 | 100.0% | 43.2% | 9.51% | 13.4% |
| recorded_repeat | 1 | 0.652 | 0.0% | 7.5% | 0.00% | 56.0% |
| zero | 1 | 0.038 | 0.0% | 0.0% | 0.00% | 2.0% |
| nonoracle | 22 | 3.291 | 70.8% | 63.4% | 2.55% | 8.3% |
| oracle_neighborhood | 17 | 2.467 | 41.2% | 24.6% | 6.21% | 36.6% |
| all | 39 | 2.932 | 57.9% | 46.5% | 4.14% | 20.6% |

## Counterevidence and interpretation limits

- Zero actions are the most central point (support .0375) but hold only2%; smoothness/support cannot specify the goal or supply effective contact sequences.
- Both TF CEM groups have *higher* mean support distance among successful than failed candidates (1.920 vs1.796;1.935 vs1.845). Recorded and both recorded perturbation groups show the same direction. A pooled negative association is confounded by candidate family; it does not establish a within-family causal relationship.
- Released LeWM CEM has40% beyond the reference threshold and22% candidate hold, whereas TF CEM has16% beyond and20%/10% hold. Better support is not a cross-model ranking criterion.
- All low-noise recorded neighbors are below the support threshold, but only52% hold. All high-noise neighbors exceed it, yet13.4% hold. Thus neither threshold separates success from failure.
- The full17-candidate recorded neighborhood is not uniformly well supported: its7 high-noise candidates are all outliers. Preserve this group rather than reporting only low-noise wins.
- Low-noise recorded perturbations cross the action Box more often than independent Gaussian (4.87% vs2.84% coordinates) yet are much closer in joint temporal statistics. Previous uniform bounded-controller results were mixed across seeds/tasks and did not establish a strong winner. Coordinate bounds and joint temporal support are distinct hypotheses. Cube's environment does not originally clip input coordinates to its nominal Box; this report does not reinterpret old executions.
- The fixed nonoracle bank has40% per-case any-hold coverage; recorded neighborhoods make the full bank82%. Those existing library figures are separate from candidate-prevalence rows above. The bank is one old physical realization, includes privileged recorded neighborhoods only diagnostically, lacks warm-planner history, and has known reset/replay variability. No claim about current closed-loop CEM or a new correlated bank follows without further authorized experiments.

## Prospective common-controller control (preparation only)

Fit the same automatic train-only covariance, set R=D^(-1/2) C D^(-1/2), and use the predeclared positive-diagonal Cholesky L with LLᵀ=R. Original CEM searches white coordinates y using its unchanged independent draws and diagonal elite mean/std update; evaluate and execute x=Ly in the original normalized-action coordinate system. The dataset scaler then maps x to raw environment actions. Training mean is used to estimate covariance, never injected as a goal/action proposal. No clipping, trajectory truncation, optimal-action labels, model changes or learned conditional policy are added.

At initial y~N(0,I), x has mean zero and unit marginal variance, matching original coordinate scales while changing temporal/cross-channel covariance. Invertible L keeps the unbounded feasible set unchanged. Later covariance is L diag(s_y²)Lᵀ, so this is an explicit finite-budget search geometry intervention, not an algebraically equivalent planner. Cholesky is ordering-dependent; raw-step-major/channel-minor order is frozen. No factorization search is allowed. Any zero empirical coordinate variance must fail preparation rather than silently importing a positive exploration prior through shrinkage.

Every compared model receives the same per-task factor, fixed planner300 samples×30 iterations/top30, horizon5, history3,25 raw execution steps/replan and50-step evaluation budget, original task cases/seeds. L=I must execute the exact native numerical path. Native warmstart, callbacks and return/action-history contracts require CPU checks before use; scalar variance stopping is absent in the pinned solver. No covariance has been deployed and no planning is authorized by this report.

The novelty claim is explicitly absent: temporally correlated action search is established, notably [iCEM](https://proceedings.mlr.press/v155/pinneri21a.html), which uses colored-noise sampling and other search modifications. This narrower control isolates a fixed training-covariance coordinate parameterization; it does not reproduce or claim superiority over the full iCEM algorithm.

A prospective falsification is straightforward: freeze the estimator/factor before new task outcomes; if a matched controller changes temporal support but fails to improve coverage or closed-loop task performance across seeds, the hypothesis is not supported. Do not select a new covariance, clipping rule, coefficient, or model after seeing outcomes. Marginal trajectory statistics are state/goal-independent and can encode dataset phase/multimodality, so failure is plausible even with an accurate covariance.

## Artifacts

- Full CPU arrays and39×50 candidate scores: `/efs/prescottsun/SPT-JEPA/reports/action-covariance-20261031/`.
- Small summary: `results/summary.json`; executed analysis: `analyze.py`.
- Original CEM: `sources/stable-worldmodel/stable_worldmodel/planning/solver/cem.py`, initialization97–120, sampling196–215, elite updates244–253, return276–283. Variable `var` holds standard deviations in the implementation; its initial value1 makes this naming harmless here.
- `summary.json`'s original scope sentence says “no planning reads”; precisely, no **new** planning is performed, while existing pool outcomes and planner source are read as stated above.
