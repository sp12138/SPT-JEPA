# Final repair results, 2026-11-02

All 34 finite chains and all 68 primary/fresh planning endpoints finished successfully. Final collection completed at 2026-11-01 20:31:04 UTC / 2026-11-02 04:31:04 Beijing, with no collection error. The stage is sealed with HOLD; there was no automatic extension. This study did not establish a high-performance candidate or a unified three-task improvement. The complete evidence is [final-summary.json](results/final-summary.json), with [completion proof](results/completion.json).

Values are success percentages averaged over the two fixed training seeds, primary / fresh, 50 cases per set. All rows in this first table are at 4000 updates.

| Method | PushT | TwoRoom | Cube |
| --- | --- | --- | --- |
| Fixed direct forecast control | 19 / 15 | 66 / 69 | 18 / 14 |
| Query, unchanged global references | 10 / 8 | 67 / 63 | 7 / 7 |
| Query, episode-conditioned references | 5 / 7 | 48 / 54 | 8 / 3 |

Global-query minus forecast primary deltas are PushT −9pp, conditional95%CI[−16,−2]; TwoRoom +1[−10,+12]; Cube −11[−20,−3]. Fresh deltas are −7[−15,+1], −6[−19,+7], −7[−15,0]. Episode minus global on TwoRoom is −19pp[−29,−9] primary and −9[−19,+1] fresh. The harder reference distribution did not improve task performance in this continuation experiment and substantially hurt the TwoRoom primary cases. Query versus forecast has additional reference-image/encoder exposure; matching updates is not matching FLOPs. Both query arms share the initial 2000 global-reference updates.

The uniform precision repair was tested at 2000 updates against the original 2000-update runs:

| Decoder-free window | Original | FP32 pool |
| --- | --- | --- |
| PushT | 1 / 2 | 5 / 5 |
| TwoRoom | 9 / 12 | 5 / 6 |
| Cube | 2 / 3 | 4 / 4 |

The batch-pooled TwoRoom control is 5/4 after repair versus6/5 originally. It is not a sample-level candidate. Forecast-scored TwoRoom original-near/original-past/calibrated-near/calibrated-past becomes8/11,5/3,8/9,11/9, versus3/4,3/3,3/6,8/5 originally. Some cells improve modestly, but all remain weak; no broad gain is established.

The matched-input precision defect remains real, but correcting pooling precision alone did not solve the task failures. Repaired decoder-free TwoRoom FP32 observer standard deviations remain .00338/.00359; its horizon5 prediction/copy error ratios are209/112. Uncalibrated forecast-scored states remain around .00014–.00017. These observations support persistent representation/dynamics problems at this budget rather than a numerical-only explanation. Ratios refer to each model's own coordinate system and are not shared physical-state errors.

The query model also has a nonzero factual-versus-swapped-action score gap while planning remains weak. Sensitivity to actions and an easy source-classification task do not establish useful optimization over CEM candidates. This experiment does not isolate whether the remaining cause is action-support extrapolation, score geometry, cold-start context, or another mechanism. Making references harder was therefore an unsuccessful intervention here, not an established general remedy.

All seeds, checkpoints and case sets are retained. Case bootstrap intervals condition on two trained models and omit simulator repeats, seed-population uncertainty and multiplicity. Cases were inspected for research selection. These are bounded2k/4k results, not convergence or canonical CLEAR/public-checkpoint superiority. No longer run was selected from these results. Historical HOLDs and the paused goal remain unchanged.

## Protocol and pre-run evidence (historical)

The original 2000-update study is complete: all 28 full-state chains and all 56 planning endpoints pass. It did not establish a method that improves all three tasks. Full scores and both case sets are in [the original report](../branch_trials_20261102/RESULTS.md).

Two interventions are being tested separately. They do not change the neural architecture, add physical/optimal-action/simulator-success training labels, or change the native evaluation budget.

| Machine | Intervention and controls | Runs / cap |
| --- | --- | --- |
| dev2 | Same 2k query full state forked into unchanged global-reference and episode-reference continuation, plus matched direct-forecast continuation; three tasks × two seeds | 18 queued / 4k |
| dev13 | Uniform existing `fp32_pool=True` on every decoder-free original cell; all three tasks, both seeds, both window and explicit pooled TwoRoom controls | 8 / 2k |
| dev15 | The same precision repair on the complete TwoRoom scale × anchor factorial, both seeds | 8 / 2k |

All 18 copied model/AdamW/scheduler states passed actual native CPU restores before updates. The new sampler passed six real-data train/validation contracts, preserves original global indices exactly, stays within each episode/split, preserves factual frames/actions and does not consume global RNG. All 34 native preflight chains passed, and all 34 initial states match their original parents tensor-for-tensor. The finite training stage started at 03:12 Beijing on all three hosts. Actual startup verifies 24 native training children, their commands/cwd/PYTHONPATH/CUDA, source hashes and restored full states; ten query jobs are queued. At 03:17 all 24 had finite logged losses and all expected active/frozen gradient groups passed. Query logged2300–3600, precision100–1300; no new task success scores were read. The finite completion helper is alive. Evidence: [health100.json](results/health100.json).

## Why repair pooling precision

All 16 frozen uploaded-model checkpoints passed matched-input precision decompositions, with unchanged network states and no simulation. For decoder-free TwoRoom seed1, on the same eight windows:

| Encoder / pool precision | Observation standard deviation | Information surrogate, nats |
| --- | --- | --- |
| FP32 / FP32 | .00298 | .0238 |
| BF16 / BF16 | .09053 | 5.7556 |
| BF16 / FP32 | .00367 | .0352 |
| FP32 / BF16 | .09400 | 5.8807 |

The other seed and all other cells are retained in [mechanism-evidence-v1.json](results/mechanism-evidence-v1.json). FP32 train/eval encodings are exactly equal on the tested inputs; the mode switch is not the explanation. Pooling precision dominates the artificial variation in these examples. The information reward can therefore credit numerically unstable variation rather than robust visual differences. This is a demonstrated evaluation-time numerical effect, and a hypothesis about failed training; it does not establish that the repair will recover task performance.

The intervention uses the existing precision flag for every original cell. It also keeps the resulting state and posterior-noise arithmetic in FP32 and can change noise rounding/random draws. The complete numerical treatment is compared; an identical full training RNG trajectory is not claimed. The precise mathematical loss is unchanged. The pooled decoder estimator remains a batch-coupled diagnostic, outside the sample-level candidates.

## Why test episode-conditioned references

The six frozen query models were tested on the same cached factual query outputs, changing only reference observations. PushT factual-versus-reference pair accuracy drops from 97.5–99.1% with global references to 78.1–78.8% within the same episode; Cube drops from 97.8–99.7% to 90.9–91.3%. This exposes a much easier source-classification task than fine discrimination within a trajectory. It does not prove background reliance or action mediation.

The new rule samples a uniform independent window in the same training episode, using no distance threshold, physical labels, batch statistics or extra images relative to global references. Collisions with factual windows are allowed because a source label is not a nonreachability label.

At the ideal source-classification optimum,

`score*(h,a,g) = log p(g | h,a) - log q(g | episode)`.

For a fixed context and goal, changing the reference distribution adds an action-independent term. It therefore preserves ideal action ranking while changing finite-capacity learning pressure. This is an optimization/sampling control, not a different ideal planning objective or a novelty claim. Both reference arms inherit the same first 2000 global-reference updates; the result will be a continuation intervention. The 4k direct-forecast control prevents comparisons against an undertrained 2k baseline.

All runs retain both seeds, primary/fresh cases, train action scalers and native CEM300×30/top30/H5/history3/block5/budget50. Query uses its learned score. The cases were already inspected for research selection and are not untouched confirmatory cases. Additional query reference images mean query versus forecast is not equal training image exposure/FLOPs; global versus episode is matched in image count. No claim of convergence, a three-task improvement or new public-method superiority is made before complete results.

Recovery: [RECOVERY.json](RECOVERY.json). NN/data source `58c2bd6`, immutable EFS `runs/branch_repair_source_20261102/source-v1`; training helper `5819ad3`, completion helper `4ec188b`. NN model files are byte-identical to the original source. Final output will be `/efs/prescottsun/SPT-JEPA/runs/branch_repairs_20261102/results/summary.json` after all 34 chains and 68 task endpoints finish. Historical HOLDs and the paused goal remain unchanged.
