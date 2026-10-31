# Goal semantics and one fixed local block-cost screen

**Decision: do not promote this block-max control or launch training.** The goal-history alternative is unavailable under the actual information conditions. The single prescribed block-max comparison does not repair the true encoded goal geometry on the existing nonoracle Cube library. No alternative norm, scale, temperature, threshold or matrix was tried.

## What the actual evaluator supplies

`World._evaluate_from_dataset` extracts a single start and a single goal row. At `world/world.py:581–606`, it stacks that single goal image per case and broadcasts it along the time dimension. Lines608–624 save and restore the same goal snapshot each environment step. `planning/objective.py:53–80` uses the last goal embedding and the last predicted endpoint. The policy's `history_len=3` supplies real past/current observations, not three different target images. Consequently a target-history covariance/tangent fitted to the legal goal images is exactly zero; an affine goal-tube cost reduces to the original point-goal distance. Taking nearby demonstration frames would supply additional information and was not done.

Read-only pinned dependency hashes:

- `sources/stable-worldmodel/stable_worldmodel/world/world.py`: `3392bd31159daab859f2d73e5a52e59ac898259f829c9d1363e64eef151b91b3`.
- `sources/stable-worldmodel/stable_worldmodel/planning/objective.py`: `b959f83c94a7543df113e474b3fec77882a7ef259c6cc42fe6a16a124877fdcf`.
- `sources/stable-worldmodel/stable_worldmodel/policy.py`: `a32f763026a57d5bf8ef5a3298d003c9a4d85c218865463cc4ca43c03b9a1f59`.

The earlier [planner-context audit](../../native_baseline_20261031/PLANNER-CONTEXT-AUDIT.md) records the shared frozen evaluator hash and all relevant family/protocol checks. This read does not change those sources.

## The one tested cost

The complete local6000 seed1/2 caches contain true and predicted1024-dimensional endpoints plus the exact goal. Their original carrier flattens an8×8 spatial grid with16 channels per cell. Set `e_j=||z_j−g_j||²`; compare `mean(e)` with the only alternative `max(e)`, separately on true encoded and predicted endpoints. The mean differs from native GoalMSE only by the positive constant64. Both seeds, all50 cases, and all22 nonoracle candidates were fixed before reading results. The39-candidate library is a secondary check containing17 recorded-action neighbors. Physical labels were used only to evaluate selected actions.

The block-max rule enforces a conjunction in the precise sense that `max(e)≤τ` iff every block error is≤τ; a mean bound allows one block up to64τ. This does **not** make the blocks correspond to objects or task factors. ViT mixing precedes the local projection. A useful physical guarantee would additionally require block errors to be calibrated to task residuals, task-relevant factors to remain distinguishable, and nuisance blocks either to be satisfied or known to be irrelevant. Reward-free observations and one goal image do not establish these conditions.

An executed two-block witness has squared errors A=(4,0), B=(2.5,2.5), with the second block representing the desired object. Mean chooses object-correct A; max chooses object-incorrect B. Moreover, identical interaction laws and one identical goal image admit both hand-only and object-only success definitions with opposite preferences. Thus evaluator-specific nuisance equivalence is not universally identifiable from these inputs. This is not an impossibility claim about reaching the exact goal or learning useful biases. The witness also confirms the repeated-goal tube has rank0.

## Complete primary result:22 nonoracle candidates

Arrows show original mean→max. Hold denotes cached sustained success during the25-step first segment, not a new50-step planning score. Pose cost is `max(position/.03, cube-symmetry angle/15°)`.

|Endpoint|Seed|Pose cost|Position cm|Angle degrees|Joint endpoint %|Hold %|
|---|---:|---:|---:|---:|---:|---:|
|True encoded|1|2.832→2.967|7.97→8.31|10.67→11.53|26→24|18→18|
|True encoded|2|2.540→2.766|7.00→7.63|11.38→11.38|32→22|24→20|
|Predicted|1|3.607→3.494|10.15→10.12|13.68→11.31|20→18|20→20|
|Predicted|2|3.798→3.493|10.81→10.02|13.20→14.39|16→14|14→16|

The two-seed case-average max-minus-mean **true-encoded pose cost worsens by+.181,95% paired interval[+.039,+.366]**; position worsens+.487cm[+.067,+1.049]. True joint pass changes−6pp[−13,0], and hold−2pp[−8,+3]. Predicted pose changes−.209[−.654,+.187] and hold+1pp[−3,+5], which does not establish a benefit. The per-seed true physical-cost rank also decreases .3080→.2905 and .3085→.2988; predicted ranks decrease .2512→.2443 and .2718→.2377.

For all39 candidates, true-encoded selections are unchanged in all100 seed/cases: joint100% and hold56%. Predicted joint falls88→66% and90→74%; hold52→42% and50→46%. Mean predicted pose cost increases by+.652[+.219,+1.130]. The full-pool change therefore supplies adverse evidence, not a reason to select a favorable candidate subgroup. Max can expose a single large local prediction error rather than averaging it, but the present outcomes do not uniquely identify that as their cause.

## What is established and what is not

The earlier [goal-factor audit](../../mechanism20261030/results/cube-audit/MECHANISM.md) already found TF true goal cost strongly associated with effector distance: partial ranks .928/.935 controlling cube position and angle, versus cube-position partial ranks .034/.022 controlling the hand and angle. That supports the hand-weighting concern, not a background-specific causal claim. Mature TF12k's cached nonoracle encoded selectors achieved14/18% hold despite40% available coverage; the current test uses local6k, not those TF weights. Their raw latent errors are not compared.

The new result narrows the remedy: replacing mean aggregation with the worst existing spatial block does not solve local6k's actual goal geometry. It does not show that all localized/multiscale representations fail, nor that task information is absent. It introduces no new sample-level learning mechanism and does not qualify as an original algorithm. D4 actuator quotient, D9 synchronous forgetting, the rejected DC-coupled measurement, and NGC were not rerun or relabeled.

## Integrity and artifacts

Zero model/encoder/renderer calls, zero simulation, zero fitting, zero GPU work, no goal-neighbor reads. Both complete source reports assert unchanged model state/RNG; all100 endpoint-cache file hashes were rechecked. All400 mean-selector indices across seed×case×endpoint×library match saved native-cost selections exactly. Float32 reduction order produces a maximum absolute cost discrepancy .00048828125 (within the prespecified2e−6 relative tolerance); no bitwise cost claim is made. The protocol, script and cost-array hashes are recorded.

Intervals resample50 cases with both training seeds averaged within case, conditional on those two models and this fixed pool. They exclude environment repetition, fresh-case generalization, current-model CEM proposal coverage and50-step closed-loop control. The existing first-render/dataset difference and maximum3.07mm recorded replay discrepancy remain unchanged. No extra experiment is recommended from this screen.

Small artifacts here: `PROTOCOL.md`, `blockmax.py`, `results.json`, `witness.py`, `witness-results.json`. Full candidate cost arrays remain at `/efs/prescottsun/SPT-JEPA/reports/next-mechanism-blockmax-20261101/costs.npz`; source cache `/efs/prescottsun/SPT-JEPA/runs/cube_pool_reuse2000_20261031/local6000-seed{1,2}`. All source/config/checkpoint/cache paths and hashes are retained in results provenance. No shared instructions or Git index were changed.
