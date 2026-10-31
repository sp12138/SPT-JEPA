# One common-bank counterfactual check — design, not a new method

The question is whether orthogonal-basis2000 ranks planner-distributed candidate actions worse than identity-local2000, although it fits logged trajectories better. Compare the same seed's two models against the **same physically executed endpoints**. This separates factual fit from candidate ranking; it does not identify the architecture component responsible or establish a general planning mechanism.

The proposed bounded audit uses the original primary manifest's first8 cases in its existing order: episodes94/115/241/544/578/669/1006/1159, starts36/19/19/71/42/42/44/72. No selection from model results. Two trained seeds; identity checkpoints are `runs/local_isotropic2000_20261031/tworoom-spatial-local-seed{1,2}/model-step0002000`, basis checkpoints are `runs/local_basis2000_20261031/tworoom-basis-orthogonal-seed{1,2}/model-step0002000`. Paths are under the authoritative EFS base. Both load through original basis-capable source `da26748`; the pilot checks all210 common frozen tensors exactly, strict checkpoint loading, Q orthogonality and unchanged parameters/buffers.

Use indices0–63 of the **original CEM first iteration with300 candidates**, batch1,30 iterations,seed42. Candidate0 is the mean. Do not instantiate a64-sample replacement, select good candidates, or substitute a CPU RNG bank for CUDA. For eventual8-case execution, preserve the original first-planning-call RNG consumption: each preceding case consumes30×300 draws before the next case's first iteration. Shapes/counts are independent of model costs. The present `pilot.py` intentionally supports only the first case; it interrupts the real solver at its first cost call and cannot silently run a larger campaign.

For each candidate, call the **original `World.evaluate` dataset reset** and execute its normalized action sequence through the original `WorldModelPolicy` queue. Its5 blocks contain25 distinct2-D micro-actions: action_block5 is grouping, not repetition of one action. Native train-episode action inverse normalization and environment `[-1,1]` clipping remain unchanged. Use the original world max_episode_steps100 and CLEAR swept collision/hold/route semantics. Stop this diagnostic at25 steps, freeze an early successful terminal exactly as native wait mode does, and record early termination. This is a horizon-end diagnostic, not50-step P0. Physical positions/route success are diagnostic records only.

Predictive cost is unchanged `ShootingCostEvaluator + GoalMSE(sum)`. Oracle cost is that same goal metric applied to encoded real terminal observations. Within each trained seed, identity/basis oracle costs must agree under Q to a predeclared relative numerical tolerance (`5e-6 × max(1,max|cost|)`); failure stops interpretation. Report all64 predicted/true costs, Spearman ranking, selected candidate's true-cost regret and range-normalized regret. A fixed bank's oracle is only a within-bank reference. Never train on rollouts or replace the planning loss with position/success labels. A ranking gap would implicate learned candidate predictions; similar rankings would leave later CEM iterations, replanning and other mechanisms unresolved.

**Current authorized stage:** code and CPU contracts only. `--cpu-contracts` hides CUDA and loads0 models; it checks the native CPU300→64 sampler against an independent oracle, actual dataset start/goal positions, repeatability of a whole25-step trajectory, native inverse-normalization/unblocking, saturated-action clip equivalence, and the native GoalMSE algebra. CPU bank results are not the future CUDA bank. `--pilot` additionally checks repeated branches0/1, common observations, true-cost Q invariance and unchanged checkpoint state. Root must freeze/review source before the single-case GPU pilot; no full8-case evaluation or automatic launch is implemented.

Proposed pilot command, **not executed** (replace `<frozen pilot path>` and `<new output>` after root review):

```sh
python <frozen pilot path>/pilot.py \
  --source-record /efs/prescottsun/SPT-JEPA/runs/local_basis2000_20261031/source-record.json \
  --pilot --device cuda \
  --identity /efs/prescottsun/SPT-JEPA/runs/local_isotropic2000_20261031/tworoom-spatial-local-seed1/model-step0002000 \
  --basis /efs/prescottsun/SPT-JEPA/runs/local_basis2000_20261031/tworoom-basis-orthogonal-seed1/model-step0002000 \
  --output <new output>
```
