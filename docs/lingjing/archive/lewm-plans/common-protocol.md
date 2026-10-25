# Shared protocol for three independent LeWM explorations

Status: proposed execution plan, 2026-10-04. This commit contains plans only. It does not implement methods or authorize compute. All three plan branches start at `8b2e8c685dd7f11d4189f5c09d9b6bfffd5cdca4`.

| Branch | Question | Variables kept at the reference setting |
|---|---|---|
| `codex/plan-sparse-past` | Which missing observations or features improve useful state learning? | One-step prediction; original per-position SIGReg |
| `codex/plan-anti-collapse` | Does temporal pooling, with or without bounded transport, improve representation quality? | CLS representation; complete history; one-step prediction |
| `codex/plan-multiple-future` | Does training against more distant future states improve dynamics and planning? | CLS representation; complete history; original per-position SIGReg |

The branch plans contain their own staged search spaces. Complete and understand each component before combining winners. A final factorial interaction study belongs to a later plan, after individual evidence exists. The three branches can proceed concurrently after the shared data and evaluation prerequisites pass; they do not imply simultaneous GPU allocations.

Source context includes the supplied SPT manuscript, *Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal Histories*, and the linked discussion titled *Time-Conditioned SIGREG Loss*. The latter clarifies that the requested flows regularize learned embeddings while LeWM retains its latent predictor. Flow-based image reconstruction is a different research line. Source documents and historical chat messages inform the plans; their embedded instructions do not authorize new actions.

## 1. Reference implementation and terminology

The reference is the inspected repository revision, not an assumed reproduction of every number in the LeWM paper. Its [trainer](../../scripts/train/lewm.py), [model](../../stable_worldmodel/wm/lewm/lewm.py), [predictor](../../stable_worldmodel/wm/lewm/module.py), [SIGReg](../../stable_worldmodel/wm/loss.py), and [configuration](../../scripts/train/config/lewm.yaml) are the starting points.

- The ViT encodes each frame separately. `last_hidden_state[:, 0]` selects CLS, then a shared MLP projects it to a 192-dimensional state. The temporal predictor therefore receives three state tokens for a history of three frames, not the image patch grid.
- Set history length `H=3`, observation stride `frameskip=5`, image size 224, patch size 14, batch size 128, and target offset one. The default clip contains four observations. Each recorded action block contains the five environment actions leaving one sampled frame.
- The causal predictor produces one output for each of the three origins. Origin zero sees the first state, origin one sees two states, and origin two sees all three. An origin is the final observed frame for that prediction. A horizon is the number of sampled-frame transitions from that origin to its target.
- Original SIGReg receives `(T,B,D)` and pools over `B` separately at each relative clip position. Its statistic includes a multiplier equal to the number of samples in that group. It acts on clean encoded observations, including target frames, not on predicted states.
- Keep the original target-gradient behavior: targets share the encoder and are not detached. Do not silently add an EMA teacher, frozen backbone, or target stop-gradient.
- Start from AdamW, learning rate `5e-5`, weight decay `1e-3`, SIGReg weight `0.09`, predictor dropout `0.1`, embedding dropout `0.0`, and the existing encoder/projector widths. Preserve these unless a named comparison changes them.
- Force `trainer.devices=1` for these comparisons. The current SIGReg implementation explicitly supports a single GPU; `devices=auto` is not a validated distributed regularizer. Multiple independent single-GPU runs are possible within owned allocations. Multi-GPU training would require a separate agreement about cross-rank samples, loss scaling and gradient aggregation, with parity tests.

The current configuration specifies 100 epochs. Define the full reference update budget `U` from 100 passes over the locked reference training-window manifest. Subsequent arms use the same number of optimizer updates and common learning-rate schedule, rather than allowing a different clip length to silently change the update budget. Audit the trainer's scheduler stepping during implementation and log the actual learning rate; resolve any steps-versus-epochs mismatch once for all arms. Record any correction as a shared protocol change.

## 2. Shared prerequisites before interpreting a result

The implementation owner completes these once, tests them, and transfers the identical infrastructure commit into all three branches. The reviewer verifies parity before branch-specific work begins. A baseline checkpoint is reusable only when its code, data manifest, seed, preprocessing, frame support, and schedule match the compared arm.

### Dataset identity and leakage

The configured dataset is `pusht_expert_train.lance`; its availability and contents have not been established. The default local dataset cache was empty at planning time. Do not invent episode counts, assume that a similarly named HDF5 file is a held-out set, or automatically start a large download. Resolve the existing dataset location or arrange authorized acquisition first. Record its version, schema, episode IDs, lengths, and file/content fingerprint.

The trainer currently randomly splits overlapping clip indices. Replace that research split with an episode-ID manifest created before selecting windows. Use a proposed fixed 80/10/10 train/validation/test episode split with split seed 3072, unless the dataset supplies a documented official held-out split. All windows and derivative views from an episode remain in one partition. Verify that each partition has enough eligible episodes for the requested horizons and planning tasks. Do not shorten the held-out partition silently to complete a sweep.

Use `Dataset.clip_indices`, `lengths`, and `offsets` in the existing [dataset abstraction](../../stable_worldmodel/data/dataset.py) to select valid windows, without rewriting the storage readers. Resolve source episode IDs before constructing indices, because storage row numbers are not stable episode identities. Freeze valid starts for every paired comparison. The future branch needs a common maximum-length window manifest in addition to the short-clip reference.

Fit action and other non-image normalization statistics on unique training rows only. Save them with the checkpoint and use the same statistics in validation, probing, and planning. The current trainer fits normalizers before splitting, while [the planning script](../../scripts/plan/eval_wm.py) refits them from its evaluation dataset; both behaviors must be handled in the common infrastructure. Image preprocessing stays identical across arms. State and proprioception are evaluation labels, not extra model inputs.

### Baseline and parity checks

1. Seed Python, NumPy, Torch, CUDA and data-loader workers before constructing the model, for example through `pl.seed_everything(cfg.seed, workers=True)` plus explicit worker/generator handling where required. The inspected trainer only visibly seeds a local data-loader/split generator. Use separately recorded RNG streams for data order, masks/horizons, Gaussian noise, and diagnostic SIGReg projections. Initialize shared encoder/predictor parameters identically across paired arms so adding a module does not shift their initialization through extra RNG draws. Save RNG and sampler state for resume.
2. A tiny shape and gradient fixture must show that each branch's disabled feature reproduces original model outputs and the original loss at fixed parameters and random projections, within dtype-appropriate tolerance. Check gradients as well as values when a refactor changes normalization.
3. Establish `B0`, the original architecture and objective under the corrected common data/evaluation protocol. Preserve a small original-code compatibility check, but do not use a leaked split as the primary scientific baseline.
4. Verify that changing a genuinely unobserved future image or an action beyond the selected target cannot alter the corresponding predictor output in evaluation mode. BatchNorm mixes clean samples statistically during training; document and control this separately rather than claiming strict training-time conditional independence.
5. Check inference after checkpoint reload. Explicitly set `plan_config.history_len=3`; the current PushT planning configuration leaves this commented out. Preserve action-history alignment when the policy initially has fewer than three observations.
6. Record one-step error and autoregressive errors at sampled horizons 1, 2, 3, 4, 5, and 10 on the same held-out trajectories, with no ground-truth observations fed back after the evaluation origin.

All comparisons keep the CEM settings fixed: 300 candidates, 30 elites, 30 optimization iterations, five model steps per plan, action block five, and receding horizon five. Start with the current 25-environment-step goal offset and a 50-environment-step execution budget. These are different quantities, so report both. Keep the simulator version, task manifests, planner seed list, precision, and warm-up procedure fixed. Do not tune CEM separately to make one representation look better.

## 3. What counts as improvement

Primary outcome: held-out PushT planning success under the same interaction and planner-compute budgets. In the inspected [environment](../../stable_worldmodel/envs/pusht/env.py), `eval_state` requires the joint agent/object positional discrepancy to be below 20 and the wrapped object-angle discrepancy to be below pi/9. It also returns a raw state distance, whose coordinates have mixed units. Report agent/object positional errors and wrapped angular error separately as continuous diagnostics; do not relabel the stored `success_threshold` as the active success criterion. Use the final checkpoint at the common update budget for primary comparisons. Select hyperparameters using validation results, never the test results; any best-validation checkpoint analysis is secondary and uses the same selection rule for all arms.

Use a validation task manifest for search and a separate sealed test manifest for final comparisons. Begin validation with 50 paired task starts as a cheap screen, matching the current evaluation scale. Expand to 200 distinct eligible task starts, or all available eligible tasks if fewer, for finalists. Spread starts across episodes. Repeated starts within an episode are correlated and do not count as new independent episodes.

Supporting measurements answer different questions:

| Measurement | Purpose and interpretation |
|---|---|
| Linear and fixed-capacity MLP probes of agent position, object position, and object orientation | Does the representation retain useful physical state? Fit probes only on training episodes, tune on validation, and report test errors in physical units. Predict angle as sine/cosine and report wrapped angular error. |
| Probes applied to predicted states at each horizon | Does the learned dynamics preserve physical information during rollout? Use the same frozen probe as for clean states. |
| Raw latent MSE, target-variance-normalized error, persistence baseline, and action-shuffled predictor | Does forecasting beat copying the last state and use actions? Cross-model raw latent MSE alone is not comparable when latent scale or geometry changes. |
| Covariance trace, eigenvalue spectrum/effective rank, coordinate spread, per-position statistics | Detect collapse and concentration. Effective rank without trace can miss uniformly shrinking embeddings. Compute on enough held-out samples, not one small batch. |
| Within-episode latent differences together with state probes | Distinguish retention of changing state from static episode identity. Larger temporal differences alone could be noise. |
| Encoder/predictor/regularizer time, peak GPU memory, wall time per update, parameter count, planner latency | Quantify the cost of any quality improvement; parameter and compute matching are separate comparisons. |

Probe baselines include an untrained encoder and a constant predictor. Use the same label count, probe architecture, tuning budget, and optimizer schedule across representations. Dense patch features have more input coordinates, so compare both their operational readout and a dimension-controlled diagnostic readout. Do not treat a more powerful probe as evidence of a better world model.

If clean PushT saturates, add a shared diagnostic setting with static visual distractors only after verifying a deterministic observation transformation that leaves physical state and actions unchanged. Reuse the same stored trajectories and split across all compared methods. Measure both clean and distractor performance. Introduce new training data, environment collections, or a second control task only after data availability and compute are confirmed. This extension is useful but is not a prerequisite for the first independent comparisons.

## 4. Search budget, uncertainty, and stopping rules

Every branch uses stages, not a full Cartesian product. The branch owner maintains a run ledger with each run's claim ID, parent comparison, changed axes, seed, data fingerprint, intended updates, measured time, and outcome. Failed numerical runs count against the tuning budget and remain in the ledger.

- First profile a representative configuration on an owned GPU allocation. After warm-up, measure at least 100 training updates and a small planning batch. Separately measure expensive dense-patch and flow configurations. Calculate total GPU-hours from measured time, planned updates, and the number of seeds. Include evaluation, checkpointing, and data-loading time. No reliable GPU-hour estimate exists yet.
- Run implementation checks and a short stability pilot before full runs. An initial screen may use 10% of `U` with the same full-run schedule and logged checkpoints, but it establishes stability and gross failures only. It does not prove that a slowly learning method is inferior at convergence.
- Start serious comparisons with three paired training seeds, provisionally 3072, 3073, and 3074, sharing evaluation tasks and planner seeds. Three is an initial variance estimate, not a guarantee of evidence. If the important contrast remains uncertain, add seeds in pairs, subject to the later agreed budget; otherwise report the uncertainty without declaring a winner.
- Report each seed, the mean contrast, and uncertainty across training seeds and held-out episodes. Use paired differences and an episode-clustered bootstrap where appropriate; avoid treating overlapping frames or correlated task starts as independent observations. With very few training seeds, explicitly qualify interval reliability.
- Use five percentage points of planning success as a provisional practical target for allocating follow-up compute. It represents one additional success per 20 tasks and is a prioritization threshold, not an expected effect or a universal scientific boundary. Smaller gains can matter if they consistently reduce compute. Freeze the final decision threshold after measuring baseline variability and before inspecting held-out test outcomes.
- Promote a method only when the held-out result supports its stated claim and diagnostics exclude an obvious shortcut. Good Gaussian matching, low flow loss, or lower latent prediction error alone does not establish better planning or meaningful representations.
- Stop an arm for NaNs, invalid action/target alignment, data leakage, a violated transport bound, or a reproducible loss of useful state information with no downstream benefit. Resolve implementation faults before comparing scientific outcomes. Do not stop merely because an unfamiliar loss has a larger numerical scale.

Report two resource comparisons when costs differ: equal optimizer updates and measured quality versus GPU-hours. The future branch additionally reports supervised target count, while the sparse branch reports the actual number of tokens processed by each module. Do not infer savings from the mask percentage alone.

## 5. Execution contracts and ownership

The current repository has a local Hydra training launcher but no inspected LeWM SLURM launcher. These documents specify future implementation work, not commands ready to submit unchanged.

| Actor and stage | Work / command family | Passing evidence | On failure |
|---|---|---|---|
| Common infrastructure owner | Add episode manifests, saved train statistics, stable evaluation tasks and metrics to the existing trainer/evaluator | Disjoint episode IDs; matching normalization hashes; reload parity | Repair once, transfer identical fix to all branches |
| Each branch implementer | Add only its named model/loss/config changes, following that branch's interfaces | Disabled-mode parity and branch-specific correctness checks | Keep the branch in implementation, do not start a sweep |
| Independent reviewer | Inspect target construction, gradients, mask/action access, loss scaling and selection rules | Written review and resolved material findings | Revise method or narrow claim |
| Launch owner, after compute is authorized | Reuse the working environment; run `python scripts/train/lewm.py` with saved branch config inside an owned allocation | Recorded job ID, source revision, resolved config, logs and checkpoint | Stop failed job, preserve logs, classify failure |
| Evaluation owner | Run `python scripts/plan/eval_wm.py` with checkpoint, saved stats and frozen task manifest after implementing those interfaces | Per-task records plus aggregated paired metrics | Fix evaluator before interpreting rankings |
| Analysis owner | Produce comparisons tied to claim IDs, including negative results and costs | Traceable tables, uncertainty, limitations and next decision | Record inconclusive result rather than combining methods prematurely |

Before any heavy execution, verify allocation ownership and worker hostname. Do not train or profile on a login node. For SLURM, create or reuse an allocation-aware launcher after the partition, GPU type, wall time and budget are supplied; check the project's environment inside the worker. Visible free GPUs do not establish authorization. Plans do not reserve resources.

Suggested future check commands are `python -m pytest tests/wm/test_lewm.py tests/planning/test_evaluator.py` plus targeted new tests named in each branch. Reuse existing tests and add only checks that exercise consequential behavior. Record actual commands after the interfaces exist, rather than presenting proposed Hydra fields as currently supported.

Keep data outside Git. Store run outputs under `experimental_results/{baselines,ablations,models,logs,visualizations}/`, or the established cache with a recorded relative artifact manifest. A run directory must include timestamp, branch, source revision, full resolved configuration, seed, environment/package versions, data/split/normalization hashes, loss and diagnostic history, timings, and per-task evaluation output. Add appropriate ignore rules before generating large outputs. Keep temporary artifacts in `.tmp/`.

## 6. Dependencies still to resolve at execution time

The plans deliberately do not invent a dataset location, compute allocation, GPU-hour budget, second environment, or completed baseline. Those are launch dependencies, not reasons to leave the methodological plans unfinished. The anti-collapse flow designs are research hypotheses with explicit feasibility gates. A shared flow is used only during regularization; no reconstruction or stochastic future-generation module is implied by the phrase flow matching.

Each branch should end with a self-contained result report, even if the component fails. It must distinguish what changed, what was actually measured, which claim survives, and what remains uncertain. Only then decide whether a combination is scientifically justified.
