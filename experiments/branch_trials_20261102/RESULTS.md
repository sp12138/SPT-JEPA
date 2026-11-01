# Three-machine branch trials, 2026-11-02

All 28 runs finished at exactly 2000 updates, and all 56 full primary/fresh task endpoints passed. Model/full-state/backward checks, input cases and both seeds are retained in [results/final2000-summary.json](results/final2000-summary.json). No unified three-task improvement was established.

Values below are mean success percentages over the two fixed training seeds, primary / fresh (50 cases each).

| Method | PushT | TwoRoom | Cube |
| --- | --- | --- | --- |
| Future-query score | 7 / 8 | 66 / 68 | 4 / 3 |
| Fixed direct forecast | 10 / 13 | 61 / 65 | 15 / 4 |
| Decoder-free, sample-level window | 1 / 2 | 9 / 12 | 2 / 3 |
| Decoder-free, batch-pooled control | — | 6 / 5 | — |

The query-minus-forecast conditional primary case intervals are PushT −3pp [−10,+4], TwoRoom +5 [−8,+18], Cube −11 [−20,−3]. Fresh deltas are −5 [−13,+3], +3 [−11,+17], −1 [−6,+4]. These condition on two trained models and exclude simulator repeats, seed-population uncertainty and multiplicity. TwoRoom is inconclusive; Cube primary is worse. The 2k screen is not convergence or a maturity comparison against the much longer PushT TF controls.

TwoRoom forecast-scored original-near/original-past/calibrated-near/calibrated-past means are 3/4, 3/3, 3/6 and 8/5 percent (primary/fresh). Neither calibration nor anchor placement established a substantial planning improvement.

A subsequent matched-input audit isolates pooling precision as a numerical defect in the uploaded-model trials. On the same eight TwoRoom windows, decoder-window seed1 has information surrogate 5.756 nats under full BF16 versus 0.0238 under FP32; keeping the BF16 encoder and changing only the pool to FP32 gives 0.0352, whereas FP32 encoder plus BF16 pool gives 5.881. All 16 frozen models passed exact manual/native encoding and FP32 train/eval equivalence checks with unchanged state hashes. This proves precision sensitivity of the reward, not that it caused every failure or that a training repair improves task performance. See [repair evidence](../branch_repairs_20261102/results/mechanism-evidence-v1.json).

The frozen query classifiers also separate global references much more easily than references within the same episode. PushT factual/reference pair accuracy falls from 97.5–99.1% to 78.1–78.8%; Cube from 97.8–99.7% to 90.9–91.3%. The factual query/cache is unchanged. This supports a more demanding reference-distribution test; it does not prove background reliance or task improvement.

The new bounded repair/continuation protocol is tracked separately in [branch_repairs_20261102](../branch_repairs_20261102/RESULTS.md). All original outputs, failures and HOLD remain intact.

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

At 02:02 Beijing a metadata-only follow-up found all 24 launched models at least 200 updates, with finite logged losses and the required positive active/zero frozen gradients. The four query-family PushT training arms reached exactly 2000 and proceeded to final audits/planning; the other 20 logged 200–1800, with four dev2 jobs still queued. No failures or task success scores were read. The finite completion helper was alive. Evidence: [results/health100.json](results/health100.json).
