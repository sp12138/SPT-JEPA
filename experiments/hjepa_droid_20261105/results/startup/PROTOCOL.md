# H-JEPA processed DROID comparison, 2026-11-05

The user authorizes testing our algorithms on H-JEPA's released datasets and using dev13/dev15 fully. This finite round prioritizes the processed DROID release. It does not reopen the old MetaWorld or three-environment campaigns, resume a goal, or install an LLM monitor.

## Immutable sources and data

- Public code: `kevinghst/H-JEPA`, commit `97054ecaff83e13839195ec4b92d74c08a0127c5`.
- Independent, ungated public dataset: `jepa-world-models/h-jepa`, revision `cc05cb8d2aaf9aec0d5364d073c6369ff760b60b`, `droid/*`: 66 files, 91,160,672,201 bytes. Actual anonymous downloads, size checks and published LFS SHA256 checks precede extraction. Each host independently hashes the archive bytes it extracts to local SSD. Archives and error records remain on EFS.
- The full released training CSV has 74,896 entries; it is neither a 16-clip training subset nor the entire original unprocessed DROID distribution. The released training files contain two external camera views. Missing paths/files remain in the native sampler's population, with its bounded retry policy; we do not silently filter the split.
- Validation has 64 entries, 62 overlapping training, and is monitoring only. The fixed 16 planning trajectories are excluded from the training CSV by canonical trajectory identity. The public `full_fps5` normalization statistics describe the whole 74,912-episode source, not a freshly fitted train-only normalizer; their scope is shared by every arm.
- Frozen public model files are independently downloaded with their resolved model-repository revision and SHA256s. They are reference comparators, never candidate initializations or teachers.
- Our unchanged neural/loss source is frozen from SPT-JEPA commit `256e7b5`; the four historical `stable_worldmodel.wm` compatibility files have individual source hashes. `bridge.py` exposes that namespace and imports model primitives without the unrelated Python-3.12 training facade. Public data and planner modules remain from the pinned release.

## Learner information

All candidates start from random weights. No external DINO, VLM, segmentation, depth, flow, or pretrained teacher is used. Learners receive images and factual executed 7D pose/gripper deltas computed by the released reader. Robot states are read to compute these action deltas; the models have no proprioceptive encoder, reward, success, language, task ID, or optimal-next-action policy label.

Native LeWM+IDM and two-level H-JEPA retain their published batch SIGReg and executed-action inverse-dynamics losses. These baselines are not described as sample-level methods. Our temporal-forecast and innovation losses remain sample-level; recorded actions condition prediction, not a behavior-cloning target.

## Fixed training comparison

Four arms, each with training seeds 1, 1000, and 10000:

| Arm | Neural/loss definition | Initialization |
| --- | --- | --- |
| LeWM+IDM | Native `droid_lewm` | Random |
| H-JEPA | Native `droid_hjepa_l2` | Random |
| temporal-forecast | Existing `RobustLeWM`, rho .8, forecast fraction .5 | Random |
| innovation | Existing reversible anchor-subtraction target | Random |

Every formal run uses 2 H200 GPUs, local batch 128/global batch 256, BF16 mixed precision, 100 epochs × 292 updates = 29,200 updates. All methods use native 5fps clips and native normalization. The public two-level dataset exposes the same 8-frame level-1 span. Our required interface adaptations are action dimension 7, encoder input 256, and the first observed frame as the sole renderer anchor; no older extra frame is provided. Renderer targets retain the existing area-downsampled 64×64 RGB convention. The last padded action is not a candidate prediction target.

Native optimizer/loss/encoder/predictor settings remain unchanged. Our original two-group optimizer and 1% warmup/cosine schedule are retained with the new full-budget horizon. Native 10% warmup/cosine behavior remains the public implementation. SPT 0.1.7 uses manual optimization and calls scheduler.step at every update despite the public configuration's `interval: epoch`; this was checked in source, so no scheduler correction is applied.

The architectures, optimizers, latent widths and FLOPs differ. This aligns data, observations, training updates and the public evaluation task; it is not an equal-FLOPs or identical-backbone claim. Actual parameter counts, gradient norms, update counters, learning rates and throughput are recorded. The native sampler uses unseeded `default_rng()` per read: same CSV and update budget do not imply identical random camera/windows, bitwise resumability, or paired stochastic training examples. Explicit seeds are installed before model construction. The two candidate arms' initial learned tensors are checked for equality per seed.

Technical preflights are separate disposable three-update runs. Formal runs do not inherit them. Preflight errors remain preserved as execution/interface failures, not low method scores. Required gates include full native model/optimizer checkpoint restoration on CPU, real-data GPU backward, differentiable 36-action planning, all 16 released clips decoding, exact split/normalizer identities, and at least one complete public-planner call for every relevant interface. Training health checks require finite loss and active required gradient groups.

## Evaluation

Use the pinned release's 16 curated waypoint clips, 37 observations/36 actions at 5fps, fixed camera/indices, start and terminal goal images only. Public center-square crop → 256 resize (`antialias=False`) → ImageNet normalization is common. Ground-truth action paths and grasp positions are used only after planning for the released offline scorer.

Flat candidates and LeWM use `droid_flat`: 32 action candidates, at most 90 AdamW iterations, lr .01, public early stopping/clipping, horizon 36, terminal latent goal MSE, one-state planning history. Our adapter inherits the public JEPA rollout/get_cost/criterion implementation. H-JEPA and frozen HWM retain their released hierarchical planner: level-1/level-2 candidates 16/4, at most 90 iterations each, and their published learning rates (.01/.01 for H-JEPA; .01/.3 for HWM). These are public algorithm-specific planner configurations, not matched planner FLOPs. Actual solver records and runtime are retained.

Each model is evaluated with planner seeds 1, 2, 3 on the same clips. Frozen public LeWM/HWM/H-JEPA references contribute 432 episode outputs. Newly trained models are evaluated at epochs 10 and 100: 1,152 outputs total. Repetitions of the 16 clips and planner seeds are not additional independent robot trials. Main metric is public mean Fréchet fidelity/skill-over-zero-motion-floor, with xyz ATE alongside it. This is offline trajectory fidelity, not binary task success or a real-robot execution result. No test-driven parameter search is authorized by this fixed plan.

## Finite execution and recovery

Only dev13/dev15. Eight two-GPU workers initially train one arm each on each host. Four third-seed jobs use an atomic shared queue: the first free pair takes the next pending seed, giving the slower candidate arms priority over already-completed training. This replaces the original preferred static host/pair assignments in the immutable plan without changing any recipe, batch, source, optimizer, or budget. The operational parent transfer preserves the original live torchrun children and every executed update; `queue-adoption-*.json` records both parent generations. The neural trainer remains `helper-v8/train.py`; only the finite queue uses helper-v10. Released epoch objects become eligible for evaluation only after a writer-completion marker, and loaded state hashes must match that marker. Pairs with no pending training switch to the finite shared reference/epoch evaluation queue, one worker per GPU. Allocation is not a claim of continuous utilization; actual GPU snapshots and advancing update counters are the evidence.

Maximum 72 hours after formal approval. Workers retain full public checkpoint files and all failures. The native resume loader restores model/optimizer/scheduler/loop state, but random video-window streams are not exactly replayed. No fabricated zero-replay guarantee is made. STOP or deadline terminates owned process groups; the collector writes summary and HOLD, then exits. No automatic extra seeds, hyperparameter sweep, another dataset, or new research round follows.
