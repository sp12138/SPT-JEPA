# Proposed only: Cube TF6000 information versus prediction

One question: does learned TF6000 provide more task-relevant observed state than the fixed carrier, or does its planning advantage mainly appear when that state is predicted forward? No extraction, fitting, GPU scheduling or model update is authorized by this document alone.

Use exactly the existing identity-carrier seed1/2 cohorts: 128 training clips ×8 frames, 64 episode-disjoint validation clips ×8 frames, evaluating the same448 future frames. Preserve manifest/data hashes, row order, camera/input224×224, action blocks, training-only quaternion codebook and pinned cube24 angular metric. Two frozen TF6000 checkpoints only; both have64 output coordinates. No case or label-based selection.

For each TF checkpoint, extract observed training/validation codes once using its original clean encoder in eval mode. On the same64 validation clips, obtain causal forecasts from frame0 plus recorded actions only, initial carry and teacher-forced prefix. Future observed encodings are diagnostic targets, never inputs to causal rollout. No posterior noise, altered planner, BatchNorm calibration or checkpoint update.

Fit the original diagnostic rules on observed training codes only: train-only coordinate standardization; linear ridge λ=.001 in mean-loss units; centered RBF ridge with the same λ and bandwidth equal to median pairwise training distance. Apply the same fitted readout to observed future, causal forecast, initial carry and teacher-prefix codes. Predict cube24-invariant orientation tensor81 and block xyz; decode angles through the unchanged training-only codebook. Report xyz mean error/≤3cm, angular error/≤15°, and joint framewise thresholds. These are diagnostic labels and are never used to train a world model; joint thresholds do not establish three-step sustained task success.

Reuse the existing F600 training/observed embeddings and completed readout results, because its frozen encoder/projection/calibration are unchanged at6000. For a matched6000 prediction comparison, reuse saved identity-carrier6000 `transition-*/state-audit.npz` causal/teacher/carry states if present. Do not re-encode the original F observations. If those saved causal arrays are unavailable, report the observed-state comparison plus TF's within-model observation→forecast gap, and explicitly omit an F6000 dynamics claim.

- TF observed readout improves over F: evidence that the learned representation makes relevant information more accessible under these fixed probes; not proof that F contains none.
- Similar observed readout but a smaller TF observation→forecast degradation: supports a dynamics contribution, conditional on probe sensitivity.
- Neither improves despite better planning: these diagnostics fail to explain the planning gap; metric/search interaction or unmeasured information remain possibilities.

Physical-label availability, frame exposure and readout rules are identical; world-model training budgets remain separately identified. The1024 correlated training frames, differing representation geometry and fixed kernel limit negative conclusions. Do not tune seeds, dimensions, kernel bandwidth or validation regularization.

Possible execution after root approval: two extraction jobs on dev2 GPUs4/5, then CPU-only fitting. This proposal does not launch them or alter the active training allocation.
