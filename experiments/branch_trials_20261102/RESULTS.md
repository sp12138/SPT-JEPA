# Three-machine branch trials, 2026-11-02

The 28 fresh runs have passed native two-update training, full CPU model/AdamW/scheduler restoration, GPU objective backward, and two-case planning integration. Sixteen actual paired-initialization comparisons pass. The bounded 2000-update stage was dispatched on all three hosts at about 01:55 Beijing. These are technical preflight results; no 2000-update task performance is available yet.

| Machine | Mechanism | Runs |
| --- | --- | --- |
| dev2 | Future-query score versus fixed direct forecast, PushT/TwoRoom/Cube, two seeds | 12 queued across 8 GPUs |
| dev13 | Uploaded decoder-free, per-window sample-level estimator on all three tasks, plus two explicitly batch-coupled TwoRoom controls | 8 |
| dev15 | Uploaded forecast-scored model, TwoRoom initial scale × anchor distance, two seeds | 8 |

The query model learns `F(history, actions) · B(goal)` from actual future observations and independently sampled reference observations. It reads no future images when constructing `F`, never feeds a predicted physical state back into itself, and receives no optimal-action, physical-state or simulator-success labels. Its loss is separable across examples. It uses additional reference images, so equal update count does not imply equal training image exposure or FLOPs. This is an experiment grounded in known conditional-density-ratio and future-feature ideas; no novelty or performance superiority is established.

The branch checks use `lingjing/decoder-free-as-jepa` at `fbb3efa6` and `lingjing/anchored-state-jepa` at `43962c52`. Selected reference code and tests were copied byte-for-byte, with provenance in [upstream/provenance.json](upstream/provenance.json). All 38 vendor CPU tests and the three full-model query contracts pass. The trials isolate mechanisms rather than reproduce every grid in the uploaded specifications.

The round2 experiment separates one-time train-only pool-head calibration to standard deviation 0.1 from near/past anchors. Past gaps use the largest fixed candidate with at most 20% train fallback: TwoRoom 2 and Cube 4. No task score or physical label selected those gaps. Calibration also consumes initialization-loader RNG, so the scale contrast concerns the complete initialization treatment rather than an identical full training RNG trajectory.

All runs use batch 128, the unchanged 124910-update scheduler, and a 2000-update cap. Native CEM remains 300 candidates × 30 iterations, top30, horizon5, history3, action block5 and budget50. Both seeds and primary/fresh 50-case evaluations are retained, giving 56 final task endpoints. Query uses its learned goal score; other models use their native latent MSE. Different learned-coordinate MSEs and training losses are not directly comparable across families. A 2000-step screen is not a maturity comparison against the much longer PushT controls.

The first preflight had 26 planning successes and two PushT CLI failures. The original failures remain intact. They were caused by appending Hydra's objective override after optional flags. Snapshot v2 changes only that launcher ordering among existing production files; model, loss, trainer and dataset bytes remain identical. Both failed integrations passed from the original two-update checkpoints after the repair. The complete binding is in [results/gate-approval.json](results/gate-approval.json).

Recovery and exact process/source pointers: [RECOVERY.json](RECOVERY.json). EFS root: `/efs/prescottsun/SPT-JEPA/runs/branch_trials_20261102`. Frozen model/training source: `branch_trials_source_20261102/source-v2`, commit `53763a01`. A separate immutable completion helper at `62aa1241` waits for all finite dispatchers, checks complete endpoints, writes `results/summary.json` and seals the stage. It never restarts or extends training. All historical HOLDs and the paused goal remain unchanged.
