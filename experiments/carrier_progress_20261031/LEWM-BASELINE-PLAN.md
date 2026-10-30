# LeWM baseline: bounded preparation, no launch

Recommendation: add **four native-LeWM-objective, exposure-matched runs**, TwoRoom/Cube × seeds 1/2, through the existing project Lightning Trainer/checkpoint/sampler. First stop at 600 updates. Label them as an explicitly adapted recipe, not a reproduction of released weights or the unmodified official recipe. Do not add an eight-arm history/scheduler grid.

## What already exists

The clean pinned `sources/stable-worldmodel` checkout is commit `21446f1ede6d5284e981bd7b47f432b994e6d812`. Its official entry is `scripts/train/lewm.py`, defaults `scripts/train/config/lewm.yaml`, task configs `data/tworoom.yaml` and `data/ogb.yaml`. Its native `LeWM`, `Predictor`, `Embedder`, BN projectors and `SIGReg` are already installed and used by this project.

The project already has `as_jepa_lewm.train` and `ASJepaLeWM(mode="L0")`. Completed PushT L0 evidence includes `runs/insights12h_20261027/L0-standard-cursorfix-seed0`, `L0-standard-seed1`, and `runs/completion_20261028/L0-25k-seed2`, all at 25,000 updates. Their recorded model size is 18,034,478 parameters. `completion_20261028/full-L0-smoke` completed 200 updates; the full-L0 10-epoch directories are not evidence of completed training. A read-only scan of `runs/*/config.json` found no L0 config targeting the hardtasks/TwoRoom/Cube data.

Historical L0 is **not automatically the official recipe**: it uses the project data, sampler, schedule and loss wrapper; Cube's constructor would set action `smoothed_dim=25`, whereas official `Embedder` defaults to 10. The native official model must be instantiated from its own resolved model config to avoid inheriting this change. Current released-model results remain evaluation references with different/unknown training exposure; their model-only configs do not establish their actual training budget.

## Recipe distinction

| Item | Pinned official entry/default | Recommended matched adaptation |
|---|---|---|
| Model | Random ViT-tiny, width192, depth12, patch14, RGB224; CLS → BN MLP; native predictor and BN prediction projector | Same modules, no renderer, learned pooling, residual carry, pretrained weights or calibration |
| Action encoder | Input10 TwoRoom /25 Cube; **smoothed_dim10**; emb192, MLP scale4 | Preserve exactly, including Cube smoothed_dim10 |
| Temporal input | history3, prediction offset1, four observations, frameskip5 | **history7/window8**, offset1, frameskip5, matching current TF/F RGB and action exposure; explicit configuration adaptation |
| Objective | One-step MSE + .09 SIGReg; knots17, projections1024; BN and SIGReg batch coupling | Exact native forward/loss, no substituted sample-level regularizer or extra target |
| Split | Random 90/10 split of overlapping clip indices | Existing episode-disjoint 9,000/500/500 split, seed20261030, hashes unchanged |
| Action statistics | Fit all rows before train/val split | Existing training-episode-only dataset-card mean/std |
| Loader/order | Native SWM clips, shuffled DataLoader | Existing `BenchmarkWindows` and `WindowSampler`; same batches/rows/seed as TF/F |
| Last frame/action | Loader requires `window*frameskip` raw rows, including an unused final action block | Existing loader includes all valid observation windows and zeros the unused final action block; first seven action blocks unchanged |
| Optimizer | AdamW lr5e-5, weight_decay1e-3, default betas(.9,.999), gradient clip1 | Same native single-group settings; no renderer group |
| Schedule horizon | 100 epochs by YAML default; total updates inferred from loader length | Fix total124,910, warmup1,249, matching existing horizon; cap externally at600/2000 without changing horizon |
| Scheduler execution | See verified implementation below | Native warmup/cosine class stepped once per optimizer update; explicitly preserve its initial-zero convention |
| Seeds | Script's `cfg.seed` seeds the split/loader generator; no global model seeding call in that entry | Existing `pl.seed_everything(1/2)` and deterministic project sampler, full saved RNG/cursor |
| Trainer/state | Lightning + stable_pretraining.Module/Manager; native model exports | Reuse project Lightning Trainer, PersistentProgress, resumable full checkpoint and `train_stage` cap |

Why history7 is the smallest clear matched design: using the official four-frame input with the same batch and updates halves RGB exposure and reduces transition/action exposure from seven to three blocks. Doubling updates would give six blocks versus seven, and change optimizer-step count. The proposed history7 adaptation preserves every RGB/action row and the native objective algebra; it adds only 768 positional parameters. It must still be reported as a history adaptation. The literal history3 recipe can remain a separately specified reference, rather than silently claim all axes match or launch another grid now.

Official loss remains `mean((predict(z[:,:H], a[:,:H]) - z[:,1:])²) + .09*SIGReg(z.transpose(0,1))` with H=7 in the matched recipe. No stop-gradient, target-network or optimal-action target is introduced. Baseline BN/SIGReg coupling is retained, as required; only proposed methods must use sample-level objectives.

## Scheduler: actual calls, not the field name

Official `lewm.py` sets the outer scheduler `interval="epoch"`. However the current installed `stable_pretraining/module.py` sets `automatic_optimization=False` and directly calls `schedulers[idx].step()` after each `opt.step()` in `training_step` (lines 585–592). Therefore actual execution advances the schedule **per optimizer update**. `_build_scheduler_config` retains the interval field, but it does not control that manual call.

Its `LinearWarmupCosineAnnealingLR` starts at lr0 and reaches the base rate after the prescribed warmup. Project `TrainModule.configure_optimizers` currently uses `LambdaLR(max(1,step)/warmup)`, giving a nonzero first step; do not call these traces exactly identical. The minimal native adapter should use the existing native scheduler class with the matched horizon and per-step interval. This needs only an optimizer/scheduler configuration hook, not a new Trainer. Before launch, compare the first few rates and rates at 600/1249/2000 from actual optimizer state. A first no-op weight update at lr0 is expected, not collapse. The official default100 epochs does not prove that released checkpoints used that exact budget.

## Capacity and budget

CPU-only Hydra instantiation of the released model configs (no forward or training) gives:

| Task | Official H3 parameters | Matched H7 parameters |
|---|---:|---:|
| TwoRoom | 18,034,478 | 18,035,246 |
| Cube | 18,034,628 | 18,035,396 |

All are trainable. Each has encoder5,501,376, projector792,768 and pred_proj792,768. H7 predictor10,792,128; action embedder156,206/156,356. This is substantially smaller than the current approximately51.1M TF/F model. Cube Fidentity has total51,109,834, trainable45,447,674, frozen5,662,160; it also has a34,595,376-parameter renderer. Do not label the comparison equal capacity or equal compute. Record wall time, GPU memory, throughput and total/trainable counts; do not enlarge LeWM solely to hit a parameter total in this first control.

At batch128/window8, 600 updates expose76,800 clips,614,400 RGB frames and2,688,000 executed raw action vectors; 2,000 expose256,000 clips,2,048,000 frames and8,960,000 actions. These counts include repeated frames/overlapping windows, not unique experience. The existing train-only window counts are513,726 TwoRoom (4,013 full batches/epoch) and1,494,000 Cube (11,671 batches/epoch). Thus 2,000 updates remain about0.50/0.17 window epochs; a poor600 gate cannot establish mature LeWM performance. Native runtime must be measured at the short gate; no GPU-hour estimate is justified from parameter count alone.

## Minimal implementation and gates, for a later explicit launch

1. Add one repository-owned native-model/loss adapter using official resolved modules and official `lejepa_forward` semantics, including SIGReg precision behavior. Keep official source/environment read-only. Reuse existing `BenchmarkWindows`, `WindowSampler`, Lightning `Trainer`, progress/checkpoint code and stage wrapper. Factory routing and a scheduler hook are the only needed integration points; no trainer rewrite or collection of legacy-L0 variants. Model-only exports must load under the existing evaluator with matching training normalization and fixed planner history3.
2. Before any short gate: native-vs-adapter same tensors/RNG MSE, SIGReg, total loss and gradient equivalence; exact sampled row/RGB/action checks against the matched project loader; held-out split exclusion; parameter count and no renderer/carry; optimizer/scheduler/RNG/data-cursor save/resume; two actual update steps plus gradient review (interpret initial lr0 correctly). Relevant active groups are encoder, predictor, action_encoder, projector and pred_proj, not nonexistent renderer/pool. Keep loss autocast behavior native; historical L0 explicitly casts embeddings to float for SIGReg, another detail requiring equivalence checks.
3. Four fresh runs stop600 with full recoverable state, export and existing state/forecast/gradient diagnostics; compare clean rollout to initial carry, wrong actions and other initial states on the same held-out windows. Use BN eval mode for inference. Run unchanged50-case strict evaluator with same seeds/CEM/cases and train-only normalization. Distinguish extra-real-observation teacher-forced scores from clean predictions. Dataset physical labels are diagnostics only.
4. Review actual health and useful learned behavior before root authorizes continuation to2000 or a later evidence-supported budget. Save zero-step and600 checkpoints, original native configuration, resolved matched configuration, source/environment hashes and departures above. Do not select the better seed or silently alter planner/cases/loss after results.

No GPU was launched for this review. Four model constructions for parameter counts were CPU-only; no encoder forward, optimizer update, shared-source edit or baseline implementation occurred.
