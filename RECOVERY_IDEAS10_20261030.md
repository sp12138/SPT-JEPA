# Idea admission and response-query short gates

Authoritative work remains `/efs/prescottsun/SPT-JEPA`; current user asks for ten mechanism-distinct ideas with serious evidence, mostly independently designed. This file is a recovery record, not a claim that ten have passed. See `experiments/ideas10_20261030/README.md` and candidate-specific rejection records.

One concrete gate is prepared: `runs/ideas10_response600_20261030` and separate `runs/ideas10_response_preflight_20261030`. `prepare_response_controls.py` archives a committed source and writes configs; it does not launch. Existing trainer/export/checkpoint machinery is unchanged. Four batch2/two-step runs plus full-state and explicit gradients must pass before the eight batch128/600-update runs start.

Paired modes are `direct` and `annihilator`, TwoRoom/Cube, seeds1/2. Both add the same12480-parameter query head, shared initial-image grounding, and seven clean predictor calls. They differ only in the temporal residual operator: identity versus orthogonal complement of `[1,time,cumulative standardized executed actions]`. Input targets are fixed8x8RGB observations; no privileged coordinates or optimal-action labels train the model. Read `spatial/IMPLEMENTATION.md` for exact normalization, RNG isolation and Cube's usually one-dimensional residual limitation.

Training: dev2 GPUs0–3 for TwoRoom; dev13 GPUs0–3 for Cube. Real preflight: dev15 GPUs0–3. After preflight closes, full-state health, frozen information/query audits and unchanged strict50-case evaluations use dev15 GPUs0–7. Verify actual GPU processes before scheduling; do not modify running snapshots. Free slots need not be filled with unjustified experiments.

The old TF600 controls are references. Direct is the equal-capacity/compute control for the proposed temporal operator. Released LeWM shares the evaluator, not an established identical training budget. The auxiliary head is unused by the unchanged clean latent GoalMSE planner. Renderer state-use and planner state-use remain separate questions.

All eight runs stop600 for review. There is no automatic longer run. Preserve all prior holds, checkpoints, logs, manifests and source snapshots. Full finiteness alone is not a functional noncollapse result. Record common550-tensor equality, query-state/action sensitivity, physical readout diagnostics, and paired actual task outcomes before any new decision.

The independent design audit has already rejected several tempting mechanisms due to exact objective equivalence, direct published predecessors, or failed real-data assumptions. These are useful negative results, not entries in the ten-idea quota. Academic priority and substantial performance gains remain unproven.

## Latest review and focused correction, 2026-10-31

First response gate is complete:40/40 exits0, all8full-state audits and original550-tensor identity pass. P0 action-null2/10%TwoRoom,6/6%Cube; direct4/10,2/4. HistoricalTF6006/8,14/10. Allthree idle workers closed, first gate held. No task winner or functional response-query success. Read results/response600/RESULTS.md.

A separate source/config snapshot prepares `runs/ideas10_relative600_20261031` with24fresh600runs: identity/action-null/fixed-random-null x forecast-only/half-observed-half-forecast Q x2tasks x2seeds. Everyarm uses the same newhead, same seven clean forecast calls, same relative per-clip outcome loss and unchanged initial grounding. Exact carry-gradient cancellation motivates the source-route factor; low target energy motivates the relative scaling factor. These are diagnostic controls within one candidate, not extra ideas. Observed future states never enter the forecast.

Preflight `runs/ideas10_relative_preflight_20261031`:12two-step runs plus12full-gradient audits, on dev2/dev15 GPUs0-5. Only after all24pass may main jobs use all24GPUs across dev2/dev13/dev15. Each slot trains to600then runs health,information,query-use and strict50-case planning. Noautomatic longer run. The fixed oneRGBLSB-squared floor is not claimed to equal true observation-noise variance; audit low-energy clips and weighted-target bias.

Eight frozen posterior-transport audits also complete0 and closed. They measure scalar noise/dynamics mismatch but do not prove planning causality. SRNN/VCDT already cover the broad residual-posterior idea; do not claim that cancellation as new. Independent design dossiers retain rejected ideas and conditional candidates; the user-requested ten strictly qualified mechanisms remain an open objective.
