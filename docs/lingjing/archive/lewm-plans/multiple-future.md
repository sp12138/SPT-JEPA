# Multiple future targets in LeWM

Status: implementation-ready exploration plan; no implementation or training is authorized by this document alone. Planned branch: `codex/plan-multiple-future`. Base revision: `8b2e8c685dd7f11d4189f5c09d9b6bfffd5cdca4`.

Use the [common protocol](common-protocol.md) for episode splits, saved training normalizers, evaluation manifests, seeds, resource accounting, and decision thresholds. This branch changes future supervision while keeping CLS representations, dense observed histories, and the original per-time SIGReg calculation. It does not import sparse inputs, new anti-collapse losses, or SPT's flow decoder.

## 1. Question and primary decision

**Does supervising different future distances improve the information learned by LeWM and its existing one-step rollout planner?** The primary experiment samples one future distance per training clip and directly predicts the corresponding state at each causal origin. The initial planner continues to call the one-step predictor. This separates improved training from a new planning algorithm.

An **origin** is the last observed frame available to a particular prediction. A **horizon** is the number of action blocks from that origin to its target. With observed frames `z0, z1, z2`, origin 2 and horizon 3 means predicting `z5` using actions `a2, a3, a4`. Neither `z3` nor `z4` is a predictor input. At the current frameskip of 5, this is a prediction 15 raw environment steps ahead.

The existing causal transformer has three output rows. They correspond to origins 0, 1, and 2, with available histories of lengths 1, 2, and 3. Those are three supervised origins, not a three-step recursive forecast from origin 2. Recursive prediction would first produce `z3`, feed that prediction back to produce `z4`, and then produce `z5`.

Treating horizon choice as **target selection** is useful: from an origin-by-horizon table, select one complete latent target in each row. It does not mask latent coordinates, remove future actions, or ensure collapse prevention. The selected target remains a full `D`-dimensional embedding.

## 2. What transfers from SPT, and what does not

The supplied SPT manuscript, *Towards End-to-End Pretraining of Streaming Video Models from Sparse Causal Histories*, Sections 3.4 and 3.5, samples `k` uniformly from `1..K` independently at each origin. It predicts that complete target directly, without intermediate future observations or recursive predictions. Its decoder receives horizon conditioning. Section 4.3 varies `K=1,2,3,4`; continuation FVD is respectively 56.28, 55.41, 53.13, and 53.77 at 90% history masking. Thus K=3 is a useful candidate, not an established LeWM optimum. These studies use a causal ViT plus DiT, and all arms receive subsequent dense-history, next-chunk-only fine-tuning. We will first compare LeWM checkpoints without an adaptation stage.

Two transfer differences matter. SPT has fixed pixel or codec targets and a conditional flow-matching decoder, whereas LeWM predicts jointly learned latents and needs its explicit anti-collapse objective. Also, SPT uses no action labels in its video training. In an action-conditioned world model, a target at distance `k` must receive all `k` ordered action blocks. Simply increasing `wm.num_preds` does not implement this experiment.

## 3. Claims, competing explanations, and evidence

| ID | Claim under test | Evidence and competing explanation |
| --- | --- | --- |
| F1 | Supervising several future distances improves the learned state representation. | Better held-out physical-state probes and planning at the same one-step rollout interface. Compare against an identical long-clip, K=1 control so extra encoded frames, SIGReg samples, and BatchNorm statistics cannot explain the difference. |
| F2 | A range of distances is useful beyond a single distant target. | Uniform `1..K` versus fixed `K`, with matched prediction counts and encoded frames. A positive result confined to one distance supports temporal reach, not target diversity. |
| F3 | Sampling targets provides useful supervision at lower predictor cost than predicting every horizon. | Sampled versus all-horizon losses at both equal optimizer steps and measured equal training cost. Fewer FLOPs alone do not imply equivalent learned representations. |
| F4 | Direct future supervision and training on recursive predictions have different effects. | A later controlled K=2 comparison with the same origins, clean targets, evaluation, and explicit gradient policy. This is secondary, not part of the first horizon sweep. |

The branch makes no claim that longer horizons identify causes, eliminate static information, or force a minimal representation. Static geometry can determine future behavior; omitting it could harm planning. Distant futures may also be intrinsically ambiguous under the recorded state and action history.

## 4. Exact primary implementation

### Fixed clips, origins, and targets

Use `H=3` and a common horizon cap `Kcap=4`. Every matched horizon arm loads and encodes `T=H+Kcap=7` clean frames. The same eligible clip starts and episode manifest serve every K. With frameskip 5, these frames are at raw offsets 0, 5, 10, 15, 20, 25, and 30. Require all six intervening action blocks to be valid. The current dataset reader additionally requires a 35-row span and returns the outgoing block of the last frame; preserve this common eligibility rule, but never feed that unused final action block into a prediction. Do not substitute padded boundary actions for a required transition.

Use exactly the first H origins in every matched arm, irrespective of K. The current SPT convention of `T-K` origins would otherwise change the number of targets as K changes. For each clip draw one `k ~ Uniform{1,...,K}` and apply it to its H origins. Targets are `z[i+k]`, for `i=0,1,2`.

The clean encoder, its projector, and the target gradient path remain trainable as in current LeWM. Do not add target stop-gradient or an EMA encoder. Compute SIGReg on the same seven clean encoded states for every matched arm, using the existing per-time batch calculation and the same weight. Keep all frames out of the predictor except `z[:H]`.

Retain an original-length reference with `T=4`, K=1, and the original four-frame regularizer, using the same long-window-eligible clip starts as the matched sweep. Label it separately from the seven-frame K=1 control. Any difference between these references measures the consequences of the longer encoding and regularization window before attributing gains to horizon selection. The shared B0 checkpoint is reusable for this contrast only if its training-window manifest also matches; otherwise show it as an additional descriptive reference rather than attributing a difference solely to clip length.

### Ordered action conditioning with an exact K=1 path

Add a horizon-aware prediction method while preserving `predict(emb, act_emb)` and the planner's existing one-step interface. Its inputs are `emb: [B,H,D]`, ordered normalized action blocks `actions: [B,H+Kcap-1,A]`, and a per-clip integer `k: [B]`.

For origin i, construct:

```text
base_i = existing_action_encoder(a_i)
tail_i = concat(a_(i+1), ..., a_(i+Kcap-1), validity_bits)
tail_i positions after a_(i+k-1) are zero and marked invalid
c_i = base_i + 1[k>1] * (tail_mlp(tail_i) + horizon_embedding[k])
prediction = existing_predictor(z[:H], c)
```

The fixed tail order supplies the relative action positions. Validity bits distinguish padding from a valid zero action. Use the already saved action normalizer. A small tail MLP, `Linear((Kcap-1)*(A+1), D)`, GELU, `Linear(D,D)`, supplies a definite initial implementation. Initialize its final layer and the horizon embedding to zero. All K arms instantiate the same Kcap=4 adapter. At K=1 bypass the adapter completely, so loading the same base weights yields the original output and gradient exactly, subject to normal numerical tolerance and identical stochastic masks.

Do not concatenate all future actions without a validity mask, reorder them, pool away their order, or pass any action leaving the target frame. A target at `i+k` uses outgoing actions `i..i+k-1` only.

**Why one k per clip?** Actions influence transformer tokens before causal attention. With independent row horizons, an earlier token may contain actions beyond a later row's target. For example, origin 0 at k=4 can expose `a3` to origin 1 at k=1, whose allowed path ends at `a1`. With the same k across rows, every visible row j satisfies `j+k-1 <= i+k-1`. The shared draw gives each origin uniform marginal target coverage. Its intended per-clip objective averages shared-k forward tasks, where all rows visible to one query use that same k. It is not the objective of independently conditioning earlier rows on other horizons. This is a minimal, explicit adaptation of SPT rather than an exact reproduction of its separate decoder queries.

The existing encoder-projector and prediction-projector BatchNorm layers couple examples and time rows through training statistics. Keep their populations matched across the primary sweep. The action-path argument concerns explicit attention inputs; it does not claim that BatchNorm removes all training-time statistical coupling. Leakage tests should also run in evaluation mode, where BatchNorm uses stored statistics.

### Objective and interfaces

The primary prediction loss is the mean squared error over B clips, H origins, and D coordinates. Average, do not sum, so K does not change the loss scale. Uniform sampling gives each target distance equal marginal supervision. SIGReg is evaluated once on clean embeddings, not once per target or predictor call.

An exact sampled-versus-all-target estimator comparison requires query-independent normalization. Current prediction-projector BatchNorm mixes the sampled horizon assignments, whereas separate all-horizon calls create different normalization populations and update running statistics more often. If F3 tests estimator equivalence, run both policies as continuations of the same reference checkpoint with both projectors' BatchNorm statistics frozen; keep affine parameters trainable and apply the same freeze to both arms. Alternatively report the trainable-BatchNorm comparison as a combined target-selection and normalization-population change. Do not claim that it isolates Monte Carlo sampling alone.

Implementation locations:

| File | Planned change |
| --- | --- |
| [`scripts/train/lewm.py`](../../scripts/train/lewm.py) | Preserve original forward as the reference mode. Add the explicit sampled-horizon loss, fixed origin count, target gathering, per-horizon metrics, and one SIGReg evaluation. |
| [`scripts/train/config/lewm.yaml`](../../scripts/train/config/lewm.yaml) | Add a future-supervision configuration group with cap, sampled maximum, target policy, and adapter settings; leave existing one-step defaults usable. |
| [`scripts/train/config/data/pusht.yaml`](../../scripts/train/config/data/pusht.yaml) | Decouple window length from the existing `num_preds` target shift for these experiments. Fixed seven-frame windows need six valid outgoing action blocks. |
| [`stable_worldmodel/wm/lewm/lewm.py`](../../stable_worldmodel/wm/lewm/lewm.py) | Add `predict_horizon` while preserving `predict`, checkpoint compatibility, and normal planner rollout. |
| [`stable_worldmodel/wm/lewm/module.py`](../../stable_worldmodel/wm/lewm/module.py) | Add the small ordered-tail adapter and horizon embedding. Reuse the current causal transformer. |
| Proposed `tests/wm/test_lewm_future.py` | Verify indexing, K=1 equivalence, future-input exclusion, expected sampled objective, and gradient behavior. |

These are proposed changes, not existing flags or runnable commands. The implementing agent must expose the configuration and tests before any experiment launcher references them. The common protocol owns split and normalization repairs shared with the other branches.

## 5. Staged search

Do not run a Cartesian product of all options. Each stage has a fixed question and follows the evidence from the previous one.

| Stage | Systems and settings | Work, passing evidence, and next action |
| --- | --- | --- |
| F0: implementation audit | Original T=4 K=1; new adapter with k fixed to 1. | Implementer runs deterministic CPU indexing/gradient tests and an allocated-device smoke test. Outputs and shared-parameter gradients agree with identical weights; no future observation or beyond-target action changes a prediction in evaluation mode. At fixed parameters, normalization statistics, and disabled dropout, enumerate all k and check that averaging the per-k loss and gradient matches the explicitly averaged shared-k objective. Stop and repair failures. |
| F1: references | Original T=4 K=1 and matched T=7 K=1. | Establish stable reference metrics under the common data/evaluation protocol. If longer clips alone materially change results, keep both controls in reporting; do not silently replace the original. |
| F2: essential horizon sweep | Matched T=7, uniform horizons with K=1,2,3,4. | Same H=3 origins, examples, steps, optimizer, SIGReg weight, adapter cap, and evaluation. Report each target horizon separately and the common one-step planner. Advance a promising or diagnostically informative K; a K=1 winner is a valid outcome. |
| F3: selected mechanism tests | At a selected K: uniform `1..K`, fixed K, and all `1..K`. | Distinguish distance, diversity, and target count. The all-target arm averages over K and calls the horizon predictor separately for each k, avoiding cross-target attention. Compare equal steps and equal measured training cost; retain the same SIGReg once per batch. |
| F4: inference mismatch diagnosis | The selected direct model and its matched K=1 control; both unchanged and after the same optional next-state-only continuation. | Only if long-horizon probes improve while one-step planning falls, test a symmetric K=1 continuation of both arms for a preregistered fraction of the total step budget. Report before and after. This follows SPT's adaptation idea without hiding its extra optimization. |
| F5: recursive comparison | Direct sampled K=2; recursive sampled K=2; K=1 control. | Run only after the primary study. Use the explicit gradient and target policy below. If the effect depends on recursion or detachment, report that interaction rather than calling all cases multi-future prediction. |

F2 is four matched configurations plus the original-length reference, with K=1 reused from F1. First run a one-seed stability pilot to 10% of the common update budget U, using its unchanged full-run schedule. A complete initial comparison across the common protocol's three paired seeds is 15 full runs before reuse of genuinely matching references. Short pilots do not establish superiority. The budget is provisional until throughput is measured and compute is allocated. Add seeds only according to observed paired uncertainty and the common decision thresholds. F3 adds at most two primary-policy configurations at the selected K, fixed K and all targets, plus any explicitly chosen normalization-controlled continuations. Fixed K is especially useful when K>1 helps but its mechanism is unclear. F4 and F5 are optional, separately budgeted stages.

If the primary adapter fails to make meaningful use of action order despite correct indexing, the next architecture test is an action-prefix encoder with causal attention and separate horizon queries. Compare it to the current ordered-tail adapter at the same K and target policy. This is a declared second architecture, not an unreported repair to the primary method. Match or report parameter counts and measured cost.

## 6. Optional recursive and independent-query controls

For recursive training, implement a training helper that starts at each origin with its observed prefix, predicts one step, appends that prediction, and slides a context of at most H states. Preserve the known action sequence. Reuse the one-step `predict` operation, but do not call the inference-oriented `rollout` unchanged: it detaches initial encoder embeddings.

The first recursive control samples the same clip horizon k as the direct arm and applies loss only at that final k-step target, so there is one target per origin in either arm. It necessarily makes more predictor calls; report and match measured cost in a secondary comparison. For K=2, use full gradient through both predictor calls and through the initial encoded history, with the original target-gradient policy. Variable prefix lengths and repeated prediction-head calls also change BatchNorm populations and running-stat update counts. Use paired continuations with the same frozen normalization statistics for a comparison intended to isolate recursion, or explicitly report this confound in the trainable-BatchNorm version. A separate detach-between-steps arm answers a different optimization question. Do not combine these policies under one result. Later, an average loss at all intermediate predictions may be tested as a distinct all-horizon rollout objective.

Exact independent horizon sampling per origin is optional. It requires isolated queries or independent prefix copies so one target's longer action prefix cannot enter another target's context. Prefix replication gives a simple correctness reference but adds predictor work and changes BatchNorm grouping. A decoupled causal history encoder with separate action-conditioned queries is cleaner at scale but is a larger architecture change. Neither should be smuggled into the first SPT-inspired transfer.

## 7. Evaluation, diagnostics, and failure interpretation

Primary selection uses the common held-out planning endpoint with the unchanged one-step planner and `history_len=3`. Compare the final checkpoint at U updates for every full run. Select hyperparameters on the validation endpoint, then evaluate the sealed test tasks once the comparison is fixed; do not select the best test checkpoint. Include held-out linear state probes and, where useful, a fixed-capacity nonlinear probe. Probe labels must come from verified dataset fields, and probe fits must use training episodes only. Inspect object/agent position, orientation, and available motion quantities separately rather than relying only on one aggregate probe score.

Record one-step prediction, direct horizon errors for k=1..4, and recursive one-step rollout errors at common lengths, including beyond the maximum trained direct horizon when valid trajectories exist. Compare learned-latent MSE alongside latent variance/effective rank and physical-state metrics: different learned coordinate systems make raw MSE alone an unreliable cross-model score. Measure predicted displacement relative to encoded displacement to detect a persistence shortcut.

Log target frequencies, origins per batch, valid action counts, per-horizon prediction loss, regularization loss, encoder and predictor gradient norms, clean-latent variance, and training memory/throughput. Check action dependence with ordered-action versus shuffled-action evaluation using existing recorded paths; do not interpret off-distribution shuffles as causal ground truth. A synthetic deterministic action-accumulation fixture can verify that the adapter has enough information to distinguish action sequences and targets before expensive runs.

Interpretation rules:

- Better probes and unchanged-interface planning support improved representations or dynamics under this protocol. They do not by themselves establish causal representation learning.
- Better direct-horizon MSE but worse planning may indicate weaker one-step training, latent rescaling, or an inference mismatch. Inspect per-horizon frequencies and state probes before changing the planner.
- Improvement only in the seven-frame K=1 control points to extra encoding/regularization context, not multi-future supervision.
- All-horizon gains only at higher cost show a supervision-budget effect unless an equal-cost advantage remains.
- Worse results at large K can reflect uncertainty or finite capacity; do not infer that the target is meaningless solely because prediction is hard.
- If K=1 dominates across decision-relevant endpoints with adequate uncertainty resolution, keep the simpler baseline and report the negative transfer. Diagnose indexing or action-path failures before interpreting an algorithmic failure.

## 8. Prior work and scope of novelty

[Fast LeWorldModel, v2](https://arxiv.org/html/2606.26217v2) already directly predicts multiple future latents using ordered action prefixes. Its Table 4 separates architecture from dense supervision and reports that a naive longer-action LeWM and dense autoregressive LeWM lose performance under its protocol. This makes the prefix interface an important comparison and rules out claiming direct multi-horizon latent prediction itself as new. Our initial question is the narrower transfer of sampled horizon supervision into the present LeWM implementation with controlled origins, regularization support, and inference.

[V-JEPA 2](https://arxiv.org/html/2506.09985v1), Section 3.1, combines teacher forcing with a two-step recursive loss and differentiates through one recurrent step while keeping the visual encoder frozen. It establishes a precedent for rollout loss, but it does not settle the outcome for jointly trained LeWM representations.

[What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?, v4](https://arxiv.org/html/2512.24497v4), Table 1 and Section 5.2, compares rollout depths 1, 2, 3, and 6 and finds different favorable depths in simulated navigation and real manipulation. It also makes the gradient treatment consequential. This motivates a limited recursive control and domain-specific conclusions, not an assumption that more future steps must help.

## 9. Resources, artifacts, and completion

No valid GPU-hour estimate is available yet. On an owned allocation, profile the matched K=1 and K=4 sampled arms first, then the selected all-horizon or recursive arm only if needed. Record peak memory and synchronized step time after warmup, plus encoder and predictor time separately. Seven clean frames increase encoder work relative to the original four; changing K within the matched sweep should not change encoded frame count. All-horizon targets increase predictor work; they do not require repeated clean-frame encoding.

Follow the existing project launcher and the common protocol for SLURM submission. Do not execute training on a login node. The plan author owns the specified comparisons; the implementing agent owns code, tests, resolved configs, and launch commands; the experiment operator records allocation, job IDs, revisions, outputs, and interrupted-run status. A missing dataset, allocation, or measured budget is a launch dependency, not a reason to invent an estimate.

Write run records and resolved configurations under `experimental_results/ablations/multiple_future/<timestamp>_<run_id>/`, with metrics, checkpoints or checkpoint references, split hashes, source revision, and measured cost. Save the final comparison table and per-horizon plots under `experimental_results/visualizations/multiple_future/`. Preserve all run outcomes, including failures and the screening rule used to select follow-ups.

Completion means the primary horizon sweep and its matched controls have an interpretable result, the implementation passes the information-flow and baseline-equivalence checks, and the report distinguishes representation gains, direct prediction gains, planner gains, and training cost. Optional architecture, adaptation, and recursive studies remain separately identified if they are not justified or funded.
