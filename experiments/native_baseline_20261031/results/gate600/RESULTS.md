# Native LeWM600 and one train-only BN calibration control

| Task | Seed | Native | Calibrated BN | Fidentity | Ffree | Fbounded | TF600 context |
|---|---:|---:|---:|---:|---:|---:|---:|
| tworoom | 1 | 10% | 0% | 44% | 34% | 46% | 6% |
| tworoom | 2 | 8% | 8% | 30% | 30% | 42% | 8% |
| cube | 1 | 8% | 8% | 4% | 4% | 10% | 14% |
| cube | 2 | 0% | 8% | 6% | 8% | 6% | 10% |

Train-only BN calibration removes the large one-step prediction bias but shows no consistent planning improvement across tasks or seeds. The raw baseline cannot support a healthy-baseline advantage claim; the calibrated result also does not establish mature LeWM inferiority. All four checkpoints are still inside warmup.

Native/calibrated/F comparisons have identical50 cases, manifest hash, solver, horizon, budget, strict semantics and action normalization. TF is scalar context only. Capacity is18.04M for native LeWM versus approximately51.1M for TF/F; these are not equal-compute/capacity comparisons.

| Task | Seed | Clean rollout / initial copy | Teacher / initial copy | Observed RMS | Observed std |
|---|---:|---:|---:|---:|---:|
| tworoom | 1 | 333.212 | 331.943 | 16.271 | 0.660 |
| tworoom | 2 | 346.946 | 342.241 | 18.105 | 0.741 |
| cube | 1 | 1.025 | 0.788 | 1.378 | 0.931 |
| cube | 2 | 5.277 | 3.842 | 2.787 | 0.925 |

Teacher-forced and shifted-copy controls receive additional real observations. Their errors must not be called clean planning performance. On Cube, wrong states/actions increase error, especially seed1; this establishes input use, not sufficiently accurate control. TwoRoom original rollouts nearly predict zero relative to the huge shared target offset.

The fixed-weight experiment reproduces the discrepancy on the exact last training batch, using the actual bf16-mixed/high training precision. Thus validation marginal shift and FP32-only evaluation are not its main cause. Replacing only observation-projector BN running statistics by current batch statistics largely removes the error; replacing prediction BN alone does not. Original BN counters equal200/400/600 and default momentum=.1: no missing-stat-update defect was found. The observation BN variance contracts sharply, especially on TwoRoom.

| Task | Seed | Last-train FP32 running MSE | Last-train BF16 running MSE | FP32 projector-batch MSE | Mean-bias share of FP32 error |
|---|---:|---:|---:|---:|---:|
| tworoom | 1 | 262.3862 | 261.4438 | 0.6213 | 99.82% |
| tworoom | 2 | 320.5125 | 321.6971 | 0.5663 | 99.82% |
| cube | 1 | 0.7718 | 0.7861 | 0.1773 | 61.83% |
| cube | 2 | 4.0716 | 4.0825 | 0.1499 | 86.60% |

Mechanism: training BN is invariant to a shared translation of its pre-BN features. A loss applied after BN therefore need not anchor that feature origin. Evaluation instead uses running means: the current-to-running mean offset is divided by sqrt(running variance + epsilon). Current TwoRoom projector inputs differ by roughly20–24 running standard deviations, and almost all prediction error is a shared offset. This supports inconsistent running statistics under changing features, amplified by shrinking variance; it does not separately establish how much comes from EMA lag versus the final parameter updates. Native-function loss/gradient equivalence was already verified; the diagnosed failure is real deployment behavior, not evidence that the native objective was replaced.

The calibrated variant uses one predeclared128-clip training manifest per task, shared across seeds. Fixed FP32 weights first estimate observation-projector moments over1024 vectors, then prediction-projector moments over896 vectors after applying the first correction. Float64 actual-count pooled moments include between-chunk variance; both use unbiased global variance. Only four running_mean/running_var buffers change. No validation labels/images fit these statistics. Fixed single-clip replay and finite-sample batch-BN corrections are verified before unchanged primary50 planning. This is a posthoc baseline adaptation, not a new objective or the literal native result.

The extra calibration exposure is 128 training clips / 1024 RGB frames per model, with no gradient updates. Native optimizer state and original exports remain unchanged. The calibration rule is identical for both tasks and seeds; its settings were not selected on planning outcomes.

| Task | Seed | Fixed train MSE | Batch-BN train MSE | Fixed/batch prediction discrepancy | Single-clip/batch discrepancy |
|---|---:|---:|---:|---:|---:|
| tworoom | 1 | 0.634264 | 0.634813 | 7.37e-08 | 1.62e-12 |
| tworoom | 2 | 0.581488 | 0.582008 | 1.10e-07 | 2.10e-12 |
| cube | 1 | 0.176487 | 0.176666 | 3.35e-07 | 4.91e-13 |
| cube | 2 | 0.156177 | 0.156338 | 3.49e-07 | 5.56e-13 |

| Task | Mean calibrated − native | Conditional same-case95% interval |
|---|---:|---|
| tworoom | -5.0 pp | [-11.0, 1.0] pp |
| cube | +4.0 pp | [-2.0, 11.0] pp |

Native learning objective and modules were retained; history7/split/exposure/budget adaptation remains explicit. PosthocBN control uses train-only128clips and changes4 buffers, not weights. Models differ in capacity/compute from TF/F.600 remains inside1249warmup; no mature superiority or automatic2000 continuation.

Native adaptation: official modules, BN, target gradients, prediction MSE + .09 SIGReg, Cube smoothed_dim10 and per-update warmup scheduler are retained. History7/window8, episode-level train split, training-only action normalization and project sampler/seed match the current experiment exposure. The literal official history3/window4/random-clip split recipe is distinct. Planner history3 is unchanged. Native BN/SIGReg are batch coupled; candidate objectives remain sample-level.

The 16 main endpoints, four attribution jobs, four train-only calibrations and four calibrated evaluations have exit0. The first planning supervisor attempt failed before opening logs or starting any GPU evaluator because nested shell quoting removed Python string quotes; its source and launch records are preserved. The corrected copied Python supervisor uses argument lists and records actual evaluator PIDs. This infrastructure failure changed no model/data or evaluation setting.

All four full-state gradient audits passed, but this does not establish a healthy original deployment baseline. No2000 run is launched by this report. Preserve native and calibrated results separately; judge a next budget using deployed planning and frozen-state evidence together.
