# Idea admission and response-query short gate

Authoritative work remains `/efs/prescottsun/SPT-JEPA`; current user asks for ten mechanism-distinct ideas with serious evidence, mostly independently designed. This file is a recovery record, not a claim that ten have passed. See `experiments/ideas10_20261030/README.md` and candidate-specific rejection records.

One concrete gate is prepared: `runs/ideas10_response600_20261030` and separate `runs/ideas10_response_preflight_20261030`. `prepare_response_controls.py` archives a committed source and writes configs; it does not launch. Existing trainer/export/checkpoint machinery is unchanged. Four batch2/two-step runs plus full-state and explicit gradients must pass before the eight batch128/600-update runs start.

Paired modes are `direct` and `annihilator`, TwoRoom/Cube, seeds1/2. Both add the same12480-parameter query head, shared initial-image grounding, and seven clean predictor calls. They differ only in the temporal residual operator: identity versus orthogonal complement of `[1,time,cumulative standardized executed actions]`. Input targets are fixed8x8RGB observations; no privileged coordinates or optimal-action labels train the model. Read `spatial/IMPLEMENTATION.md` for exact normalization, RNG isolation and Cube's usually one-dimensional residual limitation.

Training: dev2 GPUs0–3 for TwoRoom; dev13 GPUs0–3 for Cube. Real preflight: dev15 GPUs0–3. After preflight closes, full-state health, frozen information/query audits and unchanged strict50-case evaluations use dev15 GPUs0–7. Verify actual GPU processes before scheduling; do not modify running snapshots. Free slots need not be filled with unjustified experiments.

The old TF600 controls are references. Direct is the equal-capacity/compute control for the proposed temporal operator. Released LeWM shares the evaluator, not an established identical training budget. The auxiliary head is unused by the unchanged clean latent GoalMSE planner. Renderer state-use and planner state-use remain separate questions.

All eight runs stop600 for review. There is no automatic longer run. Preserve all prior holds, checkpoints, logs, manifests and source snapshots. Full finiteness alone is not a functional noncollapse result. Record common550-tensor equality, query-state/action sensitivity, physical readout diagnostics, and paired actual task outcomes before any new decision.

The independent design audit has already rejected several tempting mechanisms due to exact objective equivalence, direct published predecessors, or failed real-data assumptions. These are useful negative results, not entries in the ten-idea quota. Academic priority and substantial performance gains remain unproven.
