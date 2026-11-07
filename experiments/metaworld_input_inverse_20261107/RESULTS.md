# Current evidence

No formal candidate success-rate result is available yet. The previous admissible original-corpus best remains 25% across the six manipulation tasks, using QueryRecurrentH15-rollout. Withdrawn goal-to-action policy results are not valid achievements under the supervision constraint.

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
