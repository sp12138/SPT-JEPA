# Active causal followup: harder-task anchor information, 2000 -> 2600

The complete2000 screen has60 exits0 and is held. Strict50-case validation: TwoRoom original8/2%, temporal-forecast2/6%, temporal-forecast-highnoise14/4%; Cube original6/12%, temporal-forecast12/6%, temporal-forecast-highnoise12/6%. Released LeWM scores72%/30% on the same cases with different budget/size/data exposure. These are early results and do not establish a method win. TwoRoom remains almost state-independent; Cube has learned state/action dependence without a planning advantage.

Active16 same-state adaptations: `/efs/prescottsun/SPT-JEPA/runs/hardtasks_anchor_20261030`, source-v1 commit `a5e3857683786531502c9d03c1f45561e0f2fe75`. Four branches per task/seed: observed versus constant-zero renderer anchor, crossed with0 versus0.8 additional training flow-logit shift. All start from the identical ordinary-noise temporal-forecast2000 full checkpoint within task/seed. All parameters and original optimizer/RNG/cursor/scheduler are preserved; constant-anchor generation also ignores raw anchor images. Eight contract tests pass, including nonzero-anchor-weight generation invariance. Each branch stops at2600 (600 extra updates), then enhanced state-use diagnostics and50 fixed strict-validation cases;48 endpoints total. This is an adaptation diagnosis, not a fresh-training method claim. No automatic longer stage.

Slots: TwoRoom dev2 GPUs2–7 +dev13 GPUs0–1; Cube dev13 GPUs2–7 +dev15 GPUs0–1. Other GPUs may remain idle rather than run unsupported grids. All previous600/2000 and PushT snapshots stay immutable. All retained old PushT controls finished56/56 and120/120 exits0; old and completed harder-task dispatchers were verified idle and terminated. No old STOP marker was cleared.

Published evidence: experiments/hardtasks20261030/results/learn2000 and legacy-closure. Next: inspect the complete48 adaptation endpoints before any further training decision. Test episodes stay held out. A useful mechanism must improve actual task control; noise scheduling/anchor removal alone is not a novelty claim.

---

# Active followup: harder tasks, reviewed 600 -> 2000 updates

All20 Cube/TwoRoom gates stop exactly600 and all20 full-state/gradient audits pass;550 initial tensors match within each task/seed. Numerical health is not task effectiveness. TwoRoom state/action use is still weak. Cube shows small action sensitivity, with no established high-noise planning gain. Explicit decision: preserve all5 arms and both seeds and continue only to2000, beyond1249 warmup, then run enhanced different-video-state checks and50 fixed strict-validation cases. No automatic longer run.

Active root: `/efs/prescottsun/SPT-JEPA/runs/hardtasks_learn2000_20261030`; source-v1 commit `bf184904a0a3b426ea39ee83862f088d72cfb56e`;20 exact full-checkpoint copies and SHA256 proofs in resume-audit.json. Original optimizer, scheduler, RNG, sample cursor, data splits and model settings are preserved. Current data reader is byte-identical and faster. Ten TwoRoom continuations use dev2 GPUs2–7/dev13 GPUs0–3. Ten Cube continuations use dev2 GPUs0–1/dev13 GPUs4–7/dev15 GPUs0–3 after old closure; four random/released references use dev15 GPUs4–7. Existing workflow checks actual GPU occupancy.

Closure watcher: harder-task gate root/close-legacy-v1.py, immutable source8caa1be, PID664315 ondev13. It checks every56forecast-confirm and120new-seed-epoch2 endpoint, exact training caps, and empty worker queues before writing PUSHT-CLOSURE-REVIEWED.json and old HOLD_DISPATCH markers. It polls locally every5minutes without API calls. It makes no research-performance decision. Failures remain recorded and prevent slot release. The two older3epoch forecast-only controls and twelve2epoch new-seed replications finish once; no old grid is resumed.

Next review: inspect all60 learning-stage endpoints (20train+20health+20planning), paired validation successes and state/action interventions. This may justify longer selected learning or a simpler forecast-only component; neither is automatically launched. Test episodes remain held out. Compare against the queued released reference with its distinct size/budget/data-exposure caveats. Full reports: experiments/hardtasks20261030/results. No novelty or substantial harder-task-gain claim is established.

---

# Harder-task campaign recovery — 2026-10-30

Authority: `/efs/prescottsun/SPT-JEPA`; branch `research/ideas-20261028`. User redirect prioritizes harder tasks, effectiveness, self-contained simplicity, genuine conceptual novelty and understanding, with low API-token use. New method objectives remain sample-level. Long runs are allowed after evidence review; do not claim that a noise schedule alone is novel.

## Active work

`runs/hardtasks_gate600_20261030/plan.json` is authoritative. Training source is frozen at9c8abbded858ce495fd2954f3bbb97a0be052afd. Twenty600-update runs: Cube and TwoRoom, original/.0625-beta/temporal-forecast/original-highnoise/temporal-forecast-highnoise, training seeds1/2. All start from fresh paired initial weights per task/seed. Batch128,51.1M-scale model, raw pixel/action inputs, first-observed-frame anchor, fixed LR horizon124910. Extra training-only logit shift.8; inference sampler/parameter count unchanged.7 focused contracts and both2-update GPU execution preflights passed; neither is a functional gate result.

TwoRoom uses dev2 GPUs2–7 +dev13 GPUs0–3. Each Cube run will reuse its corresponding slot after TwoRoom health completes, and only after `CUBE-EVALUATOR-PREFLIGHT-PASSED.json` appears. Exact operational routing amendment is in `slot-reuse-review.json`; the old queue is retained as `plan-before-slot-reuse.json`. No trainer/checkpoint reimplementation. Original source/configs/STOPs remain unchanged.

The two-case Cube evaluator preflight is itself a frozen scheduled job on dev2 GPU2 after original-seed1 health. It uses project-local EGL dispatch libraries in `environments/egl-lib`; system services/shared Python were not changed. The first Cube evaluator attempt failed only for missing EGL dispatch libraries; its logs remain under `runs/hardtasks_preflight_20261030`. GPU training preflight passed. TwoRoom random and released-LeWM evaluator2-case preflights passed, no performance inference.

Four strict-validation reference evaluations (random/official LeWM x2tasks) remain on dev15 GPUs4–7 behind `reports/harder-task-redirect-20261030/PUSHT-CLOSURE-REVIEWED.json`. Create that marker only when all56forecast_confirm and120newseed_epoch2 jobs have exited0 and their live processes are gone. The original PushT parents must finish existing endpoint chains without additional training. The queued references do not block Cube training.

## Stopped/retained PushT work

Forecast-only fresh seeds5–8 firstepoch stopped gracefully at6777/6808/6800/6778. All four full optimizer/scheduler/RNG/cursor and portable/full-equality audits passed. Their parent is held; old STOP markers are permanent. Idle dev2/dev13 firstepoch dispatchers and completed seed34 dispatcher were closed after checking actual command lines and empty running sets.

Retained only14already-near-complete trainings: two forecast-only third-epoch to37473, twelve original/lower/combined newseed second-epoch to24982, with their planned endpoints. No further PushT training. Seed3/4 two-epoch all58exit0, all4fullstates audited; all numerical outcomes and reviewed figure are in `experiments/promisingseed34epoch2_20261029/results` (92f4a86 plus per-case artifact followup).

## Data/evaluation

Official Cube and TwoRoom datasets and released weights are acquired and hash/revision recorded in `data/hardtasks-v1`. Each has9000train/500validation/500test episodes. Test episodes are not used for model selection. Action normalization uses train episodes for our models; released LeWM retains its full-data normalization and may have trained on evaluation episodes. Report different model size/training exposure separately.

Pinned CLEAR scoring14acb4924f6cb9a658b62c9be63efb78e59f1987 supplies strict/moderate task semantics and swept-disk navigation. Our adapter uses the project's existing SWM CEM300x30,topk30,batch1,history3,horizon5,budget50. These are custom SPT manifests, not canonical CLEAR leaderboard results. TwoRoom test100queries have60/61episode clusters; do not assume100independent episodes. Original raw navigation training data may include trajectories invalid under corrected physics; this mismatch is explicit and shared across matched arms. Cube initial-success candidates (~38%) are excluded before evaluation.

## Next review

When gates finish, inspect all exits,20full-state health reports, encoder/predictor/renderer gradient norms and fixed-noise posterior/forecast/zero/carry/wrong-action curves. Check initial-model weights match across five arms within each task/seed.600updates are not mature planning evidence. Prepare a fresh bounded learning stage only after this review, preserving all seeds and controls; evaluate validation manifolds at learning endpoints, then reserve test manifests for confirmation. Preserve failed preflights and do not edit frozen snapshots.

Use long waits once no independent work remains (30–60minutes; shorter near gates/failures). Record useful findings with full numbers and reviewed figures, then commit/push via EFS->local ff-only sync. Read actual processes/checkpoints, not stale PIDs. Do not spawn subagents.

Data-read optimization review: unstarted Cube train/health jobs now use immutable source-v2 at04c8ea5f5ab16b34ff49bfe2beeee5eab6f227c3. Only the HDF5 read path changes;19 real windows per task (including boundaries and encoded epoch) match every tensor/action/anchor exactly. Existing active TwoRoom retains source-v1. See data-read-source-review.json and optimized-data-read-audit.json. New all-gate collection tool: experiments/hardtasks20261030/review_gates.py; it never approves longer training automatically.
