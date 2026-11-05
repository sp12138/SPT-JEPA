# DROID mechanism round, 2026-11-05

The user authorizes a new research round on dev13/dev15 to find a useful method.
Earlier closed campaigns, goals and recurring LLM monitors stay closed. The one
remaining H-JEPA reproduction and its 48 evaluations keep their original budget.

Question: does the current generative compression objective discard information
needed to distinguish executed actions? Frozen diagnostics compare within-scene
motion, factual/zero/reversed action forecasts and prediction versus copying.
These are retrospective diagnostics, not proof of physical counterfactuals.

Seven arms, seeds 1/1000, all random initialization:

| Arm | Change |
| --- | --- |
| tf | Existing temporal-forecast, plus an inactive identical IDM head |
| tf-idm | tf plus sample-level regression of factual executed actions, coefficient 1 |
| f8 | Lingjing forecast-scored, lambda=1, one-step sampled prior, rho=.8 |
| f8-idm | f8 plus the same action regression |
| variance-fixed | One-step sampled prior, rho=0, doubled mean/variance heads, fixed sigma=.1 |
| variance-prior | Same, learns prior variance only |
| variance-joint | Same, learns both posterior and prior variance |

All use the existing 51M candidate, RGB and recorded actions, reward-free,
sample-level objectives, no pretrained backbone/teacher, no optimal-action
labels. IDM is an established diagnostic control, not a novelty claim. Its head
has no batch statistics and is present in every arm. The variance trio has
identical augmented heads; fixed outputs replace the disabled variances.
Mean-producing initial tensors are paired across arms within each seed. No
calibration, loss-weight search or task-specific knobs are added.

Use the already verified H-JEPA DROID release cc05cb8d and public code 97054ec.
The official native Lightning/SPT trainer, resume/checkpoint callbacks, data
transforms, 2 GPUs/local128/global256/BF16, optimizer groups, 29200-step learning
rate horizon and public offline planner/scorer remain the starting point.
Within this new round, seed the otherwise entropy-seeded video-window RNG per
training seed/rank/worker. This improves paired comparisons; the old native
baselines did not use this change. Worker RNG is not stored by native resume;
do not claim exact sample replay after continuation.

First disposable 3-update GPU/restore/planner gates, then 600 updates and a real
full-state restore/backward health gate, then at most 5840 total updates per arm.
Evaluate all 16 existing public planning clips with planner seeds 1/2/3 at the
20-epoch update budget. These reused clips are exploratory selection data, not
untouched confirmation. Frechet fidelity and xyz ATE are offline trajectory
metrics, not robot success. Use case-cluster comparisons, not 96 independent
cases per arm. Do not reject a method solely for poor planning at 600 updates.

At most two non-control arms may continue to 29200 total updates, together with
both matched tf controls, only if each selected arm gains at least 10 percentage
points over tf separately in BOTH training seeds, has positive mean fidelity,
has case-cluster conditional 95% lower gain bound above zero, and does not worsen
mean xyz ATE by more than 5%. Rank qualifying arms by mean paired gain. No
qualified arm means HOLD after the screen, not an automatic next search.
Baseline reference is the already completed scratch native LeWM+IDM reproduction
(33.93% at 100 epochs), with architecture/optimizer/data-window differences
disclosed. It is not an equal-FLOPs, contemporary paired-architecture control.

Finite deadline: 48 hours after formal approval. No old HOLD/STOP is cleared.
Use immutable committed helper/model snapshots, preserve every failed attempt,
and require actual advancing steps and gradients before claiming GPU work.
