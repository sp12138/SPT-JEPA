# Dense sparse JEPA: implementation and execution plan

Date: 2026-10-06. Status: implementation specification, not a submitted campaign.

The primary experiment is dense next-frame patch prediction with sparse predictor inputs. Establish the dense architecture first, measure random masking second, and introduce changes to stabilization third. Complete this core before exploring mask geometry or future horizons. Keep scheduling changes independent of model changes.

The core contains **51 new dense training runs**, arranged as stages of **6, 18, and 27 runs**. Every run has one model, one physical batch of 128, one visible GPU, and its own optimizer and checkpoints. At most **six distinct GPU nodes and 48 physical GPUs** may be reserved across all project GPU work. Two processes may share an owned GPU only after the particular workload passes a packing benchmark. Sharing does not increase the physical-GPU allowance.

## 1. Starting point and decisions carried forward

Use `codex/campaign-short-budget` at `4d478c401ee136ea52051f070777de34a20b2692` as the implementation base. Create a new implementation branch, `codex/dense-sparse-v2`, when implementation begins. The base already includes the dense model from `codex/plan-sparse-past` (`fed7f74f2dfea6864296bd0cea53bc938d9e219a`) and the short-budget coordinator. Preserve existing worktrees and campaign outputs. Locate worktrees with `git worktree list`; do not encode machine-specific checkout paths in new files.

The October 6 read-only inspection found no jobs in the user's scheduler queue. The old coordinator reports all 231 tasks complete and records 42 evaluated models. Its earlier 38/42 progress report is stale. Reconcile each reused checkpoint with its completed evaluation and source hash; an empty queue alone is not proof of successful training.

Preserve the shortened campaign's scientific budget:

| Item | Fixed value |
|---|---|
| Dataset | Existing audited PushT Lance snapshot, 18,685 episodes and 2,336,736 frames |
| Episode partitions | 14,948 train / 1,868 validation / 1,869 test |
| Dataset fingerprint | `a56e3733f778c66d311c3fca275f0c9aad6e4fd61193fbecd34438c83b238265` |
| Training windows | Existing `data_protocol/reference_windows.json`, 1,586,258 training windows |
| Normalization | Existing training-only statistics; never refit on validation/test |
| Physical batch | 128, one independent model per process |
| Images and state | 224 pixels, patch size 14, 256 patches, existing 192-dimensional projected features |
| Temporal support | Four encoded observations; three causal prediction origins; next-frame targets |
| Observation stride | Five environment steps; preserve the corresponding outgoing action blocks |
| Optimizer | Existing AdamW, learning rate `5e-5`, weight decay `1e-3`, clipping and numerical settings |
| Update budget | 123,920 optimizer updates per scientific run |
| Learning-rate schedule | Existing 1,239,200-update schedule, including its 12,392-update warmup |
| Paired seeds | 3072, 3073, 3074 |
| Inference | Complete observed grids and complete predicted grids; masking disabled |
| Planning | Existing CEM and task manifest; 300 candidates, 30 elites, 30 iterations, five model steps |
| Primary evaluation | Same 200 validation tasks per model, with saved planner seeds |

This remains a ten-reference-epoch truncation of the existing schedule. Do not silently substitute a ten-epoch cosine schedule, continue past 123,920 updates, or describe the result as convergence evidence. Longer training is a separate campaign decision. The current user request authorizes writing this plan, not launching it.

Retain CLS controls at SIGReg weights 0.09 and 0.0225. Their three-seed validation means were 84.50% and 93.67%, respectively. Reuse them only after configuration and compatibility checks. If a required common implementation change breaks compatibility, rerun both controls, adding at most six runs to the budget. Do not add more CLS-coordinate or patch-mean masking sweeps. A claim specifically comparing density against pooling would additionally need a tuned patch-mean control; that is outside the primary dense-versus-CLS question.

## 2. Reuse the current interfaces

Paths in this table are relative to the integration base, where these implementations already exist.

| File | Existing capability | Required work |
|---|---|---|
| `stable_worldmodel/wm/lewm/lewm.py` | Grid encoding, masked predictor inputs, complete-grid rollout | Preserve the core interface and target-gradient paths |
| `stable_worldmodel/wm/lewm/dense.py` | Actual context-row packing, spatial indices, causal actions, full-grid queries, checkpointed SIGReg | Add explicit stabilization/grouping choices to the existing dense loss |
| `stable_worldmodel/wm/lewm/masking.py` | Exact-count independent/tube/tile masks and named RNG | Reuse; log actual retained counts and rates |
| `stable_worldmodel/wm/lewm/training.py` | Fixed-update training, independent RNG streams, atomic full-state checkpoints, faithful resume | Add resource telemetry; teacher update hook only in the later teacher stage |
| `scripts/train/config/lewm_dense.yaml` | Dense configuration inheriting shared defaults | Add the few new objective fields with backward-compatible defaults |
| `scripts/experiments/run_sparse_past_arm.py` | Prepared-arm verification and launch gates | Replace hardcoded dense weight/grouping acceptance with descriptor-based checks |
| `scripts/experiments/short_campaign.py` | One-GPU jobs, paired planning processes, task receipts, scheduler polling | Generalize task preparation and workload-specific packing; preserve old plan compatibility |
| `stable_worldmodel/wm/lewm/execution.py` | Allocation ownership, shared admission lock, six-node union including pending reservations | Enforce the 48-GPU limit globally under the same lock |
| `scripts/experiments/evaluate_lewm.py` | Fixed planning tasks, probes, forecasts, fixed-position dense-collapse measurements | Preserve evaluation; add candidate chunking only if needed for feasibility |

Do not merge the entire anti-collapse branch to obtain pooling. Its compact-state reference, `stable_worldmodel/wm/flow/objective.py::endpoint_sigreg`, and `tests/wm/test_sigreg_grouping.py` provide the mathematics and useful checks. Transport maps and flow objectives are outside this campaign.

### Dense loss contract

Keep `dense_loss(model, batch, cfg, regularizer, rng)` as the training entry point. Add these proposed fields; they are not implemented by this plan:

```yaml
representation: patch_grid
context_mask:
  enabled: true
  domain: patch
  location: post_encoder
  geometry: individual
  temporal: independent
  fraction: 0.0
  recency_alpha: 0.0
loss:
  stabilization: sigreg       # sigreg | none
  sigreg:
    grouping: time           # time | time_pool
    weight: 0.09
    precision: float32
    kwargs:
      knots: 17
      num_proj: 1024
  dense_sigreg_group_chunk: 4
```

For clean encoder output `Z` with shape `[B,T,P,D]`, core training uses:

```python
prediction = model.predict(
    Z[:, :3], action_embeddings[:, :3], context_visible=visible
)
target = Z[:, 1:]                         # attached encoder targets
prediction_loss = (prediction - target).square().mean()

if stabilization == "none":
    loss = prediction_loss               # do not call SIGReg
else:
    loss = prediction_loss + weight * dense_sigreg(...)
```

The `none` arm must skip the regularizer, rather than compute a potentially invalid value and multiply it by zero. Preserve attached targets, student BatchNorm, dropout, optimizer, augmentation, prediction loss reduction and action conditioning. Omit the unused statistic from the numeric training-loss dictionary; record `sigreg_loss: null` and the explicit mode in scientific reports. If needed, log a separately named disabled indicator rather than a fictitious measured zero.

For `dense_sigreg`, add keyword-only grouping and optionally supplied directions for verification. Keep one projection matrix shared across groups and activation-checkpoint recomputation:

```python
B, T, P, D = Z.shape
if grouping == "time":
    groups = Z.permute(1, 2, 0, 3).reshape(T * P, B, D)
elif grouping == "time_pool":
    groups = Z.permute(2, 0, 1, 3).reshape(P, B * T, D)
# Compute the characteristic-function discrepancy over the sample axis.
# Both modes multiply by B, then average over groups and projections.
```

Thus pooled SIGReg equals ordinary sample-count-scaled SIGReg on `[P,B*T,D]` divided by **T=4**, not history length three. Do not average states over time or pool spatial positions into the sample axis. Keep all clean observations, including targets, in the regularization domain.

The dense interface is `Z_t -> [256,192]`, with entire token rows removed only from temporal-predictor context. Preserve original spatial coordinates and outgoing actions. Targets always contain all 256 rows. At nominal 90% masking, 26 rows remain, giving 89.84375% realized masking; report both quantities.

### Normalization and memory boundaries

The existing clean projector uses BatchNorm over `B*T*P` rows, and the prediction projector uses `B*3*P` rows. These operations couple training examples and time positions. Preserve them identically across the core. Attention and inference are causal, but complete training-time independence from future rows is not established. Do not change BatchNorm to LayerNorm only in a no-regularizer arm.

Physical batch 128 is part of the objective. Smaller microbatches with gradient accumulation alter BatchNorm and SIGReg and are not an equivalent memory fix. First try exact activation recomputation and bounded SIGReg groups with output/gradient/RNG parity. If batch 128 still cannot fit, declare one common smaller physical batch, regenerate the entire comparison matrix and controls, and treat it as a new protocol. Do not mix batch sizes in the primary table.

The current dense verifier tested the full model at batch two and isolated SIGReg at batch 128. It did not establish production-batch training or real CEM feasibility. Full-grid decoding remains expensive even when context is sparse. A `predict_last` rollout optimization or candidate chunking may be implemented only with parity checks, then applied uniformly to all dense arms before the campaign source is frozen.

If 300-candidate CEM does not fit, slice evaluation of the already sampled candidates into ordered chunks, initially 25 candidates, and concatenate costs before the unchanged elite selection. Keep all 300 candidates, iterations, task seeds and horizons. Verify scores and decisions against unchunked execution on a fitting fixture and fixed real tasks. Reevaluate reused controls under the frozen evaluator if necessary. Candidate chunking is not permission to reduce the planner budget.

## 3. Implementation packets and ownership

| Packet | Owner | Deliverable and acceptance |
|---|---|---|
| A. Objective | Model owner | Extend dense SIGReg and none mode, configuration and descriptor validation; prove unchanged default outputs/gradients and correct attached targets |
| B. Placement | Execution owner | Global GPU admission, explicit workload profiles, paired training worker and per-child receipts; prove cap, ownership and failure-isolation behavior |
| C. Measurement | Evaluation owner | Full-batch and CEM profiling recipe, CPU/RAM/data-wait telemetry, completed-task coverage and fixed-position diagnostics |
| D. Assembly | Integration owner | Integrate A-C into one frozen source revision, generate recipes and resolve the core matrix, run allocated acceptance, publish Gate 0 report |
| E. Execution/analysis | Campaign owner | Run only the open stage, reconcile SLURM/accounting and outputs, publish paired comparisons and the next gate decision |

A, B, and evaluation preparation can proceed concurrently in isolated worktrees. The model owner owns edits to `training.py`; the execution/evaluation owners provide reviewed telemetry or hook changes for that file. Merge source changes sequentially, then freeze one revision for scientific comparisons. A behavior-changing fix during a stage invalidates affected comparisons and requires matched reruns. Report-only changes can retain results when the computation fingerprint is unchanged.

Reuse the campaign entry point `python scripts/train/lewm.py --config-path <recipe-directory> --config-name <recipe-stem>` inside the existing guarded worker. Recipes must set `campaign.enabled=true` and carry the locked data/protocol, explicit output directory and stop update. The existing coordinator command is `python scripts/experiments/short_campaign.py controller --plan <plan.json>`; the revised stage-aware plan schema and profile operation are implementation deliverables, not commands that work unchanged today. Reuse the project environment named in the prior campaign manifest; do not install a replacement environment.

## 4. Gate 0: correctness and actual resource measurements

Run CPU fixtures on the login node only when lightweight. All CUDA fitting, profiling, planning and training belongs in verified owned allocations. Initially use at most one node; independently profile these workload classes:

1. Dense unmasked and nominal 90%-masked training with original SIGReg at batch 128.
2. Dense unmasked time-pooled SIGReg and prediction-only training.
3. Dense CEM planning at the actual five-model-step horizon and 300 candidates.
4. Full dense diagnostics, including spatial probes and forecast horizons 1, 2, 3, 4, 5 and 10.

For every class, measure process startup separately, then at least 50 kernel/data-loader warmup updates and 500 measured updates from a saved representative checkpoint. These profiling warmup updates are not the optimizer's 12,392-update learning-rate warmup. Include an actual atomic checkpoint write. Repeat the solo and packed measurements three times in alternating order; use at least five fixed planning tasks for planning calibration and validate any projection to 50 tasks with one full shard before committing evaluation reservations.

Record peak whole-GPU memory, per-process allocated/reserved memory, parent-plus-loader RSS, CPU time, data-wait time, step time distribution, throughput, checkpoint duration and GPU model. Sample whole-GPU memory/utilization at one-second intervals during the bounded benchmark and store the trace. Only the owned benchmark allocation may be instrumented.

Required correctness evidence:

- Default dense SIGReg output and gradients match the existing implementation with fixed projection directions; time-pool matches the direct `[P,B*T,D] / T` reference.
- `rho=0` matches masks-disabled outputs, gradients and BatchNorm updates; masked/target-clean paths preserve original spatial indices.
- Separate fixtures verify encoder gradients through context and target paths; `none` never invokes the regularizer.
- Attention and inference reject future observation/action leakage; training BatchNorm coupling is recorded as above.
- Complete dense rollout and goal costs agree after checkpoint reload; any memory optimization preserves the declared evaluator contract.
- Interrupted/resumed training reproduces the same sampler, optimizer schedule, RNG streams and completed-update state for every objective mode.
- Serial versus packed deterministic fixtures agree, and bounded real-data continuations stay within dtype-specific tolerances fixed from solo-repeat variability before inspecting packed outcomes.

A healthy unmasked dense model does not have to outperform CLS to open the masking stage. If the architecture is finite, learns from actual observations and actions, and fits the agreed resource protocol, masking remains a meaningful test. If it cannot learn at all, repair the implementation before sweeping masks.

Gate 0 publishes `acceptance.json`, `resource_profiles.json`, a source/configuration fingerprint, and the total estimated physical GPU-hours. Missing dense feasibility or a failed ownership/continuation check keeps scientific jobs closed. A failed packing benchmark merely sets that workload to one process per GPU.

## 5. Core stages and exact run counts

The accompanying [core matrix](dense-sparse-core-matrix.csv) contains all 51 proposed training runs. It is a specification, not a launch manifest. Each row must become a resolved immutable recipe with source/data/evaluation hashes before submission.

### Stage 1: dense architecture and regularization-strength controls

Train unmasked dense prediction at `lambda in {0.0225,0.09}`, each with seeds 3072-3074: **6 new runs**. Compare against the corresponding existing CLS controls at the same coefficient and budget. The architecture change includes token count, decoder and grid regularization, so describe this as a state-interface comparison rather than a pure pooling ablation.

Hold everything else at the common protocol. Publish model parameter count, per-component step time, peak memory, planning latency and spatial state probes. Keep both coefficients for the next stage; do not select one and then attribute all subsequent changes to masking.

### Stage 2: masking at fixed objective

For both coefficients, train independent post-encoder random row masks at nominal `rho in {0.50,0.75,0.90}`, each with three seeds: **18 new runs**. Reuse the six unmasked controls. This creates a complete `4 mask rates x 2 coefficients x 3 seeds` comparison under ordinary SIGReg.

The only model-facing change within each matched coefficient contrast is retained context rows. Keep all clean encoded frames, dense future queries and all regularized patches. Report actual token counts separately for the image encoder, context transformer and decoder. Freeze the deployment protocol as sparse training followed by unmasked rollout. Do not give selected masks a dense-history fine-tuning stage.

### Stage 3: stabilization and its interaction with masking

At each `rho in {0,0.50,0.90}`, add these three recipes, each with three seeds:

- Time-pooled SIGReg, lambda 0.0225.
- Time-pooled SIGReg, lambda 0.09.
- Prediction alone, attached targets, regularizer bypassed.

This adds **27 runs**. With the Stage 1-2 results it yields the complete core factorial:

| Factor | Values |
|---|---|
| Nominal masking | 0, 0.50, 0.90 |
| Stabilization | Per-time/patch SIGReg 0.0225; per-time/patch SIGReg 0.09; time-pooled/per-patch SIGReg 0.0225; time-pooled/per-patch SIGReg 0.09; none |
| Seeds | 3072, 3073, 3074 |
| Total | 45 runs, plus six 75%-masking rate-sensitivity runs from Stage 2 = 51 |

For grouping claims, compare the same coefficient and mask rate. For mask claims, compare the same objective and coefficient. Report both coefficient-specific effects and best-of-two validation choices with the same tuning opportunity. The extra 75% arms characterize ordinary SIGReg only; they do not establish an optimum for pooled or prediction-only training.

Report the masking-by-stabilization interaction as:

`(masked alternative - unmasked alternative) - (masked ordinary SIGReg - unmasked ordinary SIGReg)`.

Use the same coefficient for the ordinary-versus-pooled interaction; for removal, report both ordinary reference coefficients. Preserve every failed/collapsed seed. If prediction-only training looks viable, add seeds 3075 and 3076 to the four cells `{unmasked, selected mask} x {selected ordinary SIGReg, none}` before making a stability claim: **8 conditional runs**. Failure at tested settings supports a scoped negative result, not a claim that all possible masks fail.

Stages 1, 2 and 3 are scientific gates, not independent campaigns competing for hardware. Stage preparation and CPU checks may overlap, and completed-model evaluation should begin as soon as a checkpoint is final. Do not train the next stage until the preceding stage's implementation, outcome coverage and relevant diagnostics are accepted.

## 6. Conditional follow-ups, in order

These runs are outside the 51-run core. Open only the branch needed to resolve the recorded result; do not create a Cartesian product of every option. Each changed target architecture, history length or prediction workload requires renewed resource acceptance. Existing packing eligibility never transfers automatically.

### 6.1 Target asymmetry if removing SIGReg is still important

If the shared-encoder prediction-only recipe fails or is materially worse, test shared-encoder target detachment at `{rho=0,rho*}`, three seeds: **6 runs**. Here `rho*` is selected from `{0.50,0.90}`, so the exact paired attached-target controls already exist. Change only `target = target.detach()`. Selecting 75% instead requires three additional attached-target, prediction-only controls before interpreting detachment.

Then, if a teacher recipe is needed, compare a hard-copy target encoder (`momentum=0`) with an EMA target encoder (fixed `momentum=0.996`) at those same two rates, three seeds: **12 runs**. Both teacher variants use identical target architecture, evaluation mode, clean full-grid inputs, no target dropout, and the same target BatchNorm-buffer policy. Copy online buffers after each successful optimizer update in both variants; vary only parameter averaging. The hard-copy control isolates averaging from the extra target forward pass and target evaluation mode. Do not attribute the shared-stopgrad versus hard-copy difference solely to momentum.

Register teacher encoder/projector modules in an explicit EMA-capable model configuration. Exclude teacher parameters from the optimizer. Update once after each successful optimizer update through a small trainer hook, before saving checkpoints. Preserve teacher weights, buffers and update count in the model state. Verify no teacher gradients and faithful resume. Keep online clean-state planning as the declared readout for all variants; teacher-readout planning would be a separate comparison.

This stage tests replacement by optimization asymmetry. It is not the same claim as attached-target prediction alone. Inverse-dynamics anchoring, reconstruction/detail branches, transports and stochastic positional embeddings remain separate subsequent proposals.

### 6.2 Spatial geometry and temporal persistence

After selecting one stabilization recipe, use exactly 75% hidden patches, which is representable by all existing tile samplers. Cross `{individual,2x2 tiles,4x4 tiles}` with `{independent per frame,persistent tube}`, three seeds: **18 total runs**, of which three may already exist. Keep batch, coefficient, target policy and token counts fixed. These fixed-grid tile masks are not an exact implementation of overlapping V-JEPA rectangles.

First analyze the geometry effect with temporal mode fixed, then persistence with geometry fixed, then their interaction. If a new geometry is selected under a replacement stabilization recipe, add its missing matched ordinary-SIGReg cells before claiming the improvement is independent of stabilization. This costs at most six additional runs for the selected two-geometry comparison. Pre-encoder masking, recency schedules and longer context are excluded here because they change the available information or prediction interface.

### 6.3 Multi-future prediction last

Keep the selected architecture, mask, stabilization and physical batch fixed. Use the existing seven-frame `future_windows.json` support for **all** horizon arms, including a new matched K=1 control. Every arm encodes and regularizes the same seven clean frames and uses the same three origins. This isolates horizon selection from episode eligibility and BatchNorm/SIGReg sample support. Old four-frame or compact T7 results cannot replace these dense controls.

Implement a proposed `dense_future_loss` callable using recursive applications of the existing one-step predictor, with no new predictor parameters. Sample one horizon per clip, shared across its three origins, uniformly from `1..K`; compare `K in {1,2,3}`, three seeds: **9 runs**. Each clip supervises three full target grids regardless of K, and loss remains a mean. For K=1, explicitly gather `clean[:,1:4]`; unchanged `dense_loss` would incorrectly take all six later frames. For horizon h, gather target indices `origin + h`. Regularization always uses all seven frames.

Hold prediction-projector BatchNorm call populations and update counts constant too. In this first horizon experiment, **every arm computes predictions through maximum horizon three for every clip**, including the K=1 arm. The initial call predicts all three one-step origins together, exactly as in the core. At unroll rounds two and three, process origins in the same fixed order, with the full batch of 128 for each origin. Do not split batches by sampled horizon or skip longer forward passes in K=1/K=2. Only the choice of supervised horizon changes. This deliberately costs more than early-exit execution, but avoids changing prediction-projector normalization and forward compute with K.

For each origin, initialize only from its observed causal prefix, preserve those observed-row masks, and unroll using the real action blocks up to that forecast's horizon. Predicted grids remain complete and are fed back without ground-truth future observations. Preserve gradients through the unroll, and route targets according to the selected target policy. Each forecast remains independent of actions beyond its own horizon, although longer forecasts are also computed. Save a separate horizon RNG stream. Record all forward calls, supervised horizon counts and the distinction between computed and supervised grids.

The new seven-frame K=1 bridge includes the same fixed three-round forward schedule. Verify its selected predictions and gradients against an independent one-step reference with seven-frame encoding and target slice `1:4`; verify its BatchNorm buffers against the explicitly specified complete forward-call schedule. Test that changing unconsumed future images or actions cannot change the corresponding prediction, with the previously documented encoder BatchNorm qualification. The initial one-step forecast must not be recomputed in a different per-origin batch.

Extend the existing compact `future.py` target gathering and tests to grids rather than importing its compact predictor unchanged. Direct horizon adapters, supervising every horizon at once, and skipping unsupervised forward rounds are deferred because they introduce additional architecture, target-count or normalization changes.

## 7. Six-node scheduling and process sharing

### Global accounting and placement

Keep one admission authority at the existing campaign root. Store this revision's results beneath `experimental_results/lewm_campaign/dense-v2-20261006/`, but continue using the root's admission lock and job ledger so legacy, profiling, evaluation and new training jobs count together.

Under the same locked read/check/submit transaction, enforce:

```text
reserved distinct GPU nodes <= 6
sum(active or pending allocation requested_nodes * gpus_per_node) <= 48
```

Use assigned/pinned node union plus conservative unassigned pending reservations, as the existing code does. Include unresolved submission receipts conservatively until reconciled with SLURM. Count allocation IDs once, never count their child processes as extra GPUs. Include unrelated-to-this-revision project jobs already in the root ledger. Other users' allocations constrain placement but are not resources this campaign owns.

The existing coordinator counts its own one-GPU jobs, and the common admission function only enforces node count. Both must be fixed before allowing revised submissions. Tests must cover simultaneous submitters, a request for a seventh node, a request exceeding 48 GPUs, pinned and unassigned pending jobs, mixed allocation sizes, and lost receipts after an uncertain submission.

Prefer independent one-GPU allocations using available CPU/RAM/GPU capacity on at most six eligible nodes. Do not reserve six entire nodes merely because the ceiling allows them. Do not permanently dedicate a node to a method, since hardware placement would become correlated with treatment. Rotate paired seeds/treatments across nodes and record placement.

When a large backlog exists, target up to 40 training GPUs and eight evaluation/diagnostic GPUs. These are borrowable queue shares, not reservations: training can use all granted GPUs when there is no evaluation backlog, and evaluation can use all of them when a scientific gate is waiting on results. Calibration and solo speed measurements receive temporary exclusive use of their allocated GPU. The coordinator is a CPU-only SLURM job.

### Packing policy

Default to one trainer per GPU when enough physical GPUs are granted for the ready stage. Stages of 6, 18 and 27 runs already fit in one wave under the 48-GPU ceiling. Packing fewer GPUs can increase individual completion time and need not shorten these stages.

When capacity is tighter or a backlog benefits, test **two trainers inside one one-GPU allocation**, initially requesting **16 CPUs and 128 GiB host RAM**. Each child inherits the allocation's same verified GPU and owns eight CPU threads, its model, data sampler, RNG streams, optimizer and output directory. Partition its CPU affinity within the allocated CPU set. This is process sharing, not DDP, a larger batch or shared model state.

Current hardware inventory reports eight GPUs and 128 CPUs per node. Existing compact training jobs used about seven CPU cores on average and roughly 24 GiB maximum resident host memory, despite only about 14.5 GiB GPU-memory usage. Eight two-trainer groups would use the node's 128 allocated CPUs and reserve 1 TiB RAM. Consequently, **96 concurrent models is only a conditional two-process ceiling**, and four or more trainers per GPU are not in the initial plan.

Keep the current six-loader-worker/prefetch recipe for the first packing comparison so the numerical/data contract stays fixed. If CPU or I/O saturation prevents a gain, profile a reduced loader/thread recipe as an execution change, verify data-order and resume parity, and apply it consistently to compared workers before measuring again. Do not enable sharing just because model allocations fit in VRAM.

Approve a workload-pair profile only when:

1. Its median aggregate completed-update throughput is at least 1.10 times sequential execution of the same work. This uses the existing 10% margin to reject negligible noisy gains; retain all three trial results.
2. The co-running per-task latency is at most twice solo latency, so one task is not starved while aggregate throughput looks good.
3. Whole-GPU peak memory leaves at least `max(10 GiB, 15% of physical VRAM)` free in the measured worst phase, and the full process-tree RSS stays below 80% of the requested host memory. Check this against both the sampled whole-device trace and the conservative sum of child allocator high-water marks plus measured CUDA-context/non-PyTorch overhead; one-second polling can miss transient peaks. Capture phase-level peaks around backward, optimizer initialization and checkpointing. These are operational headroom margins, not learned scientific thresholds.
4. Checkpoint, deterministic-fixture and real-continuation parity checks pass without a model, batch, precision or optimizer change.
5. A node-level ramp from one to four to eight allocated GPUs retains useful aggregate throughput without CPU, data-loader or shared-filesystem saturation. Use paired and solo groups at each occupancy; if the gain disappears, retain solo training at that occupancy.

Profile keys include GPU type, source computation hash, physical batch, grid/history/decoder sizes, nominal and realized mask settings and retained-token counts, stabilization and target mode, precision, loader/thread recipe, and the two workload classes sharing a GPU. Intermediate 50%/75% masks need their own measured packing eligibility; the initial 0%/90% profiles do not establish a throughput gain for them. Compact or planning evidence cannot authorize dense training packing. Prefer pairs with comparable expected completion times; do not launch speculative new configurations to fill a slot.

Use the existing subprocess worker inside the owned GPU allocation. Do not bypass Slurm visibility to claim another GPU or create two independent jobs that both assume ownership of the same device. The initial implementation needs no MPS or MIG changes. Slurm's GRES allocation and device-visibility behavior are documented in the [official GRES guide](https://slurm.schedmd.com/gres.html).

### Failures and continuation

Publish a receipt for each child as it exits, including exit code, requested/completed update, checkpoint path/hash, source/configuration fingerprint and failure category. The current group-only receipt incorrectly makes one failed child invalidate its sibling. A zero exit code is insufficient if the requested update was not reached.

Forward scheduler warning/termination signals to all live children and allow each to commit its own completed-update checkpoint. Requeue only incomplete children, from their last verified checkpoint. Never save a partially failed optimizer step. Preserve completed siblings' results and release their run locks. Hardware/allocation failures may still interrupt every process sharing that GPU.

On OOM caused by sharing, mark that profile ineligible and resume affected children solo without changing batch size or scientific settings. A numerical model failure is recorded as a scientific failure after ruling out an implementation fault, not silently retried with a different seed. Limit unexpected infrastructure-failure retries to two; repeated identical failures block that task/profile for repair while unrelated valid tasks may continue. Routine wall-time-warning exits with a valid checkpoint are `incomplete_resumable`, not failures, and do not consume this retry allowance. Continue as many successful segments as needed to reach the fixed update cap.

Poll `squeue` and progress/checkpoint records every 60 seconds while active. When a job leaves the queue, inspect `sacct` and reconcile per-child receipts before declaring completion. Resume preserves model, optimizer, scheduler, regularizer, named/global RNG, data-sampler position and any teacher state.

## 8. Cost envelope and queue order

The old compact C0 profile is approximately 0.267 seconds/update, giving about 9.2 GPU-hours for 123,920 updates before evaluation. The future compact profile is approximately 0.484 seconds/update, about 16.7 GPU-hours. Neither is a dense training estimate.

For a measured solo step time `s_i` in seconds, the training estimate for run i is:

`123920 * s_i / 3600 = 34.4222 * s_i physical GPU-hours`.

For the core, sum this over all 51 runs. Add measured startup/checkpoint costs not already included, 51 complete diagnostics jobs and 204 planning shards of 50 tasks. Do not double-count checkpoint costs already included in the measured step interval. Estimate packed cost from actual group elapsed time times physical GPUs, not from summing both children as GPU allocations.

Illustrative sensitivity only, assuming every dense run has the same solo speed:

| Assumed seconds/update | Training GPU-hours per run | Core training GPU-hours | Three-stage one-wave lower bound |
|---|---:|---:|---:|
| 0.5 | 17.2 | 877.8 | 51.6 hours |
| 1.0 | 34.4 | 1,755.5 | 103.3 hours |
| 2.0 | 68.8 | 3,511.1 | 206.5 hours |

The last column is the sum of three sequential stage durations when sufficient GPUs are granted. It excludes queue delays, profiling, evaluation and repairs. It explains why `total GPU-hours / 48` is not an honest end-to-end deadline for a gated experiment. Replace these scenarios with measured per-class estimates at Gate 0.

Use the existing 24-hour job segment as a starting reservation, bounded by verified partition limits. If a run exceeds it, checkpoint and continue under the same run ID until the fixed 123,920-update cap; do not shorten the model's training to fit a wall-time request. Estimate reservation headroom from measured variance and checkpoint cost.

Queue priority is: repair-blocking acceptance checks; evaluations that close the current stage; incomplete paired comparisons; remaining runs within the open stage; conditional follow-ups whose evidence gate has opened. There is no automatic expansion to old flow searches or additional future variants. A 48-GPU ceiling does not itself promise that 48 GPUs will be available.

## 9. Evaluation, selection and interpretation

Each core model receives the same 200 validation tasks and full diagnostics at update 123,920. Use the existing four disjoint 50-task shards; merge only with exact task coverage, no duplicates, one checkpoint hash and matching evaluation settings. Training and evaluation may occupy different GPUs, but evaluation uses an immutable finished checkpoint.

Preserve 2,048 probe windows, 256 diagnostic windows, the fixed spatial readout, 1,000 MLP probe steps, train-only probe fitting, and the same held-out physical labels. Report planning success and physical position/orientation errors, forecast/persistence/action-shuffle comparisons, covariance trace and effective rank across examples at each fixed patch position, temporal change statistics, and isolated planning latency. Turn off dropout and training noise for representation diagnostics.

Show all three seed results, their mean and paired differences. Use episode-clustered uncertainty for repeated tasks from the same episode, with training-seed variation reported separately. Three seeds give an initial variance estimate, not proof of stability. Keep the existing five-percentage-point planning gain as a practical follow-up priority, not a significance threshold. Smaller gains can be useful if they reliably reduce measured cost.

Low prediction loss, low effective rank, or a temporary warmup shrinkage is not an automatic collapse diagnosis. Numerical failure stops the task. Apparent collapse is reviewed using sustained loss of across-observation variation together with physical probes, action dependence and planning. If a confirmed collapsed run is stopped early, retain its seed and outcome in the comparison and record its actual update count; do not average it away.

Select the strongest ordinary-SIGReg control and at most two scientifically distinct alternatives using validation, while retaining every control needed for their factorial contrast. If a contrast remains inconclusive, report it as such or add a predeclared paired seed extension to the complete contrast. No hyperparameter is chosen from the test set.

After the intended follow-ups finish, freeze the selected recipes, source hashes, seed list, evaluator, task IDs and claimed comparisons. Run the independent test comparison on every required matched cell. A mask-by-stabilization claim requires all four cells on test, not just the winning masked method and one baseline. Test evaluation does not reopen tuning.

## 10. Outputs and completion checklist

The new revision uses the shared admission root, with immutable revision-local files:

```text
experimental_results/lewm_campaign/dense-v2-20261006/
  plan.json
  recipes/
  acceptance.json
  resource_profiles.json
  profiles/
  runs/<arm>/seed<seed>/
  evaluations/<arm>/seed<seed>/
  stage_decisions/
  final_selection.json
  reports/
```

Every run records source revision and computation hash, resolved recipe, data/window/normalizer/task hashes, seed, target-gradient policy, mask counts, regularizer grouping and multiplier, physical batch, requested/completed updates, hardware, allocation and process-group IDs, packing-profile identity, timings and checkpoint hashes. Persist runtime paths in local resolved manifests; keep shared recipes root-relative or configured through the campaign manifest.

Focused verification extends existing `test_lewm_dense.py`, `test_lewm_dense_evaluation.py`, `test_lewm_masking.py`, `test_lewm_training.py`, `test_lewm_execution.py`, `test_short_campaign.py`, and `test_sparse_past_arm.py`. Add meaningful regression cases for grouping/target gradients, admission races, per-child completion and resume. Reuse passing results unless a relevant implementation change invalidates them. GPU acceptance is a bounded allocated check, not a substitute for learned comparisons.

The core is complete when all 51 rows have a valid finished result or an explicitly classified failure, all matched evaluations are reconciled, and the report distinguishes density, masking, regularization strength, grouping and their interactions. Conditional teacher, geometry and future stages each produce their own comparison table and gate decision. Completion does not require every optional branch to be run.

Local evidence consulted: the completed short-budget coordinator and recipes; `validated-data-handoff.json`; `bootstrap/C0_training_profile.json`; `dense/correctness_gpu.json`; the planning-packing result; current scheduler/configuration snapshots; and read-only accounting for compact training jobs 2594075 and 2594076. Existing planning packing measured 1.709x aggregate throughput with matching task-success outcomes, explicitly limited to the tested compact planning workloads.
