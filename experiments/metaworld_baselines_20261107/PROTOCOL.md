# MetaWorld scratch baseline comparison

The current finite study runs baseline training, evaluation, and mechanism analysis
only on dev13, dev15, and dev6. No dev6-2c/dev2 connections or scheduling, old goal, or recurring LLM monitor are
authorized. The latest user additionally authorizes evidence-led candidate research
after baseline mechanisms are understood, and long runs for promising methods.
Current baseline jobs/finite caps remain unchanged; no new candidate training has
started. The separate mechanism evidence is indexed in
`../metaworld_baseline_insights_20261108/RECOVERY.json`.
The canonical root is `runs/metaworld_baselines_native_20261108`, with immutable
`runtime-native-v7`. Its collector and launch records are indexed in RECOVERY.json.
The original deadline remains 2026-11-12 16:02:48 UTC; it has not been extended.

All learners start from random weights. Inputs are RGB and actual executed forward
controls. No rewards, state, task ID, language, external teachers, or desired-goal
to unknown-next-action supervision enter learners or planners. Inverse reconstructs
only the same factual forward inputs from their observed completed transition.
Public baseline batch regularizers remain intact.

The independent public LeRobot MetaWorld MT50 release b46a1c18 supplies 2500
trajectories, split 45/5 trajectories per task. All five temporal phases stay in
their trajectory partition. RGB is224px; one20D macro input contains five native4D
controls. This is not the gated JEPA-WMs corpus or canonical MT50 policy benchmark.

The six public-source transfers are LeWM, LeWM+IDM (H-JEPA one-level recipe),
H-JEPA, SMWM, AC-MTM, and LpWM, each with three independent seeds. They are strong
representatives within these constraints; no matching published full50-task SOTA
ranking has been verified. Source commits and native resolved settings are pinned.

Each method now uses every valid start for its native window, including eligible
terminal windows and short trajectories. Global batch256/local128 uses two GPUs.
There are twelve paired slots on the three hosts and eighteen learners total.

| Methods | Frames | Train windows | Updates/epoch | Updates at20/50/100epochs |
|---|---:|---:|---:|---:|
| SMWM | 2 | 172608 | 674 | 13480 /33700 /67400 |
| LeWM, AC-MTM, LpWM | 4 | 150109 | 586 | 11720 /29300 /58600 |
| LeWM+IDM, H-JEPA | 8 | 105486 | 412 | 8240 /20600 /41200 |

An epoch drops the final incomplete global batch after a fresh seeded permutation.
Full common corpus coverage does not imply equal images, updates, FLOPs, or capacity.

**Protocol correction:** the initial harness used window8-valid anchors and took
shorter prefixes. That omitted terminal observations and short trajectories:
2frame methods reached116245/183858 eligible training frames, and4frame methods
137763/183790. This is a data-coverage confound, not an algorithm failure. All twelve
short-window recipes are replaced by fresh native-window runs. Eight active pilot
parents and descendants were stopped by validated ownership; four other pilot
recipes were unstarted or only saved3update diagnostics. Pilot results/errors remain
archived. Six valid window8 LeWM+IDM/H-JEPA learners continue unchanged; no completed
checkpointed SGD is repeated for those parents. The original five-day cap remains.

The corrected data contract passed60 actual terminal-window RGB/action checks,
short-trajectory inclusion, causal action padding, sampler step3 resume and epoch
boundary checks, with Python/NumPy/Torch RNG unchanged. Each fresh learner must
pass3 actual DDP updates, full CPU model/AdamW/scheduler restore, held-out FP32
backward, padding poison invariance, and20native-step MPC before continuation.
All four fresh-method representative gates passed; active fresh learners also
passed100update restores. Later queued seeds undergo the same gates.

Public model and loss source is unchanged across the correction. Tiny methods use
ViT-Tiny192/patch14; H-JEPA and its one-level control transfer the native DROID
small384/patch16 model at224 instead of256px and20D instead of7D controls. The
one-level IDM control is not an equal-capacity LeWM ablation. Losses execute verbatim
pinned public train.py function ASTs, avoiding unrelated Hydra/WandB side effects.
LeWM retains SIGReg.09; SMWM inverse1/no SIGReg; AC-MTM ActionNCE.30, temperature.1,
no SIGReg; LpWM RDMReg10/native Cube temporal-Jaccard.1; H-JEPA fine SIGReg.08/IDM100
and coarse SIGReg.16/IDM50/action-SIGReg.005. These are global transfer settings,
with no MetaWorld task-specific tuning. Native AdamW settings are retained:
5e-5 for LeWM/AC-MTM/LpWM,1e-4 SMWM,5e-4 H-JEPA family.

One disclosed harness adaptation uses update-based warmup/cosine over100 native
epochs: public entries mix epoch intervals with update-count scheduler lengths.
H-JEPA warmup fraction.1 and other methods'.01 remain global. Native local-batch
losses/BatchNorm remain under DDP; native H-JEPA ECF reductions are preserved.
No cross-rank ActionNCE negatives are added. This is a controlled transfer, not an
exact reproduction of published training budgets or reported scores.

Common evaluation uses RGB-goal CEM300x15/top10/H6/execute3/context2/f5, warm starts,
physical action projection, and500native-step any-success. All methods use the same
physical case bank, action-candidate seeds, and RGB/trace checks. At20/50epochs each
model gets100cases (2/task);100epochs gets1000 (20/task). All expert-goal construction
failures remain in the primary denominator. Frozen400 common held-out window8
forecasts accompany stages: persistence, factual, physical-zero, and reversed-action
rollouts. Each model's latent errors have its own units; these diagnostics are not
realized physical counterfactuals. Full checkpoints retain optimizer/scheduler,
rank RNG states, sample cursor, source provenance, and required-gradient health.

The separate native-controller track applies the public DROID GD recipe to the
same six mature H-JEPA/one-level parents and1000cases each. Flat36macro/32samples
or hierarchical12/36macro with4/16samples and12subgoals uses90iterations, minimum30,
patience5 and lr.01 at both levels. Cold180native-step blocks replan to500native
steps. H-JEPA retains its native4000-entry learned train-action-code queue. These
feedback/compute/prior differences are explicit; common CEM is not labeled the
full native hierarchy. Two H-JEPA and one flat parent passed actual20step technical
checks. Frozen native evaluation uses released pairs only after all primary recipes
have claims, protecting unfinished owners and live CUDA processes.

Historical20epoch NativeLeWM reference evaluation (two parents,2000episodes) and
three goalblind controls (3000episodes) remain valid and separate. All5000 physical
traces were checked. New100epoch baseline ranking is pending. Operational errors,
including packaging/resolver, custom H-JEPA CPU queue placement, collector ESTALE,
and native-queue launch environment failures, are retained separately from measured
method performance. Bounded metadata retries propagate persistent errors.
