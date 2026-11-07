# Current evidence

The 5,000-update screen has **not met the performance goal**. All 32 original models completed 32 evaluation episodes each (1,024 episodes), including the recovered run. Their six-manipulation evaluations contain just one Push success; the other 31 models have zero manipulation successes. No configuration passes the two-seed long-training gate, so no 20,000/41,200 continuation or full-50-task performance evaluation has started.

Representative complete two-seed configurations:

| Configuration | Reach | ReachWall | Six manipulation | Eight-task mean |
| --- | ---: | ---: | ---: | ---: |
| observed-s16-w10 | 87.5% | 87.5% | 0% | 21.875% |
| observed-s4-w10 | 50% | 100% | 0% | 18.75% |
| predictive-s4-w100 | 25% | 12.5% | 2.083% (1/48) | 6.25% |

The last row's only manipulation success is Push in seed1; seed2 has 0/24 manipulation successes. Each task uses four shared physical development cases crossed with two independently trained seeds, not eight independent physical cases. These are small internal eight-task/500-native-any-success results, not a public SOTA comparison. The previous admissible original-corpus best remains 25% across six manipulation tasks, using QueryRecurrentH15-rollout at a different training budget. This screen does not establish convergence or a causal effect of adding inverse. Withdrawn goal-to-action policy results remain inadmissible.

The first 992 traces passed read-only verification of physical-case identities, SHA256, bounded finite actions, actual success arrays, frozen model/export identities and common initial CEM banks. The [final review](results/screen5000-final/summary.json) inherits those checks and separately verifies the recovered32 traces. One of the 32 shared expert goal constructions failed and remains in the main denominator. See also the [initial review audit](results/screen5000-review31/audit.json). The 768 manipulation episodes repeat the same physical cases across32 different models; they are not768 independent contexts or one algorithm's success estimate.

The last original model, predictive-s4-w10-seed1, initially failed at step4,000 while deleting an already-absent obsolete portable export. Its full4,000-step checkpoint actually restored model/optimizer/scheduler with the exact512,000 consumed-sample cursor. The cleanup fix preserves non-benign errors. The first repair then failed at4001 because the original stage-runtime source record already existed; that one unsaved update was replayed. Per-attempt metadata filenames now preserve all attempts without collision. Both errors are retained. The bounded recovery completed exactly4,000->5,000 and the original32-episode evaluation: all32 episodes failed, including all24 manipulation cases. There was no new method/seed/baseline or repetition of checkpointed updates. The original two-seed gate remains failed. Recovery completed at2026-11-07T15:03:38UTC; effective execution errors are0, with one original recipe failure and one failed repair attempt archived.

Frozen gradient decomposition also completed for31 models, with zero weight changes, optimizer updates, simulator runs or baseline evaluations. It reused only four unique held-out windows across the two training seeds; this is a targeted objective diagnosis, not a large independent dataset.

| Family | Median weighted inverse/main encoder gradient-norm ratio | Predictor ratio | Predictor cosine |
| --- | ---: | ---: | ---: |
| observed (8 models) | 1106.35 | 0 | N/A |
| highnoise (8 models) | 1159.62 | 0 | N/A |
| generative (8 models) | 1387.08 | 0 | N/A |
| predictive (7 models) | 390.26 | 49.09 | -0.256 |

These are raw gradients before clipping/AdamW preconditioning, not actual update ratios or a causal proof of the performance failure. They reveal a concrete scale/compatibility issue in the tested10/100 weights: the auxiliary dominates shared-encoder gradients, while the predictive variant directly supplies a much larger, partly conflicting dynamics gradient. An auxiliary weight's number cannot be transferred between objectives with different normalizations: our world rate is normalized by image elements, whereas inverse MSE is averaged over20 standardized action coordinates. The screen rejects the tested configurations at5,000 updates, not the general inverse principle. See the [frozen diagnostic summary](results/gradient-diagnostics-v1/summary.json).

The 32 new candidates passed native two-update CPU model/optimizer/scheduler restoration, checkpoint RNG/cursor checks, held-out FP32 backward/required gradient checks, padded-action poisoning, inverse sample-independence and exact inference after inverse-head removal. All passed actual 20-native-step MPC entry runs. These establish execution correctness only; the audit does not prove bitwise equivalence to uninterrupted training.

All 32 candidates also passed the native 1,000-update full-state audit. Its FP32 held-out backward checks give mean action-encoder gradient norms 0.00449 / 0.00314 / 0.00425 for observed / high-noise / generative auxiliaries, versus 7.52 for the predictive auxiliary. Mean predictor gradients are 0.0310 versus 10.75 for observed versus predictive. Families aggregate two state sizes, two weights and two seeds, with only two held-out windows per model. This is an early training diagnostic, not a causal estimate of downstream performance. The predictive auxiliary directly supplies this gradient by construction. It could still teach an action watermark rather than correct physical outcomes. Closed-loop success must distinguish those explanations.

Read-only statistics on 170,040 usable TRAIN action words show:

| Quantity | Value |
| --- | ---: |
| Standardized action target mean square | 0.9957 |
| Within-word variation after subtracting the five-control mean, standardized mean square | 0.0790 |
| Standardized 20D covariance participation rank | 4.61 |
| Dimensions explaining 95% / 99% covariance | 6 / 12 |
| Adjacent native-command correlations, XYZ/gripper | 0.954 / 0.949 / 0.927 / 0.987 |
| Words with any gripper command change | 3.47% |
| Input components exactly at a Box boundary | 32.94% |

The 20 coordinates therefore contain strong temporal redundancy. This weakens the explanation that the auxiliary primarily fails because it must reconstruct 20 independent unknown controls. It does **not** establish invertibility from two RGB observations: endpoint aliasing, actuator/contact ambiguity and hidden velocities remain possible. The observed within-word residual is descriptive, not an irreducible-error bound and not a proposed replacement target. No training field, input or target was changed by this analysis.

The new full-50 cohort is constructed: 1,000 physical cases, exact state/physics replay and strict RGB gates passed, all expert-goal failures retained. The cohort is disjoint from the original 160 cases' task vectors. No candidate outcome on this cohort has yet been measured.

The screen and long-training gates, fresh replica seeds and full-50-task cohort are specified in [PROTOCOL.md](PROTOCOL.md). Published scores and their setting differences are recorded in [public_baselines.json](public_baselines.json). Neither gradient growth nor a small development-screen gain establishes a public-baseline performance gap or SOTA.
