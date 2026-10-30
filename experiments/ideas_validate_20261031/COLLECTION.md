# Completed-gate collection

`collect.py` and `plot.py` are analysis-only. They do not load a model, allocate a GPU, update a run, or modify the frozen `dd2aeaa` source. Run them after the parent confirms all 120 endpoints have completed. No partial grid is silently accepted.

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 \
  /efs/prescottsun/project_jepa/envs/lewm/bin/python -B \
  experiments/ideas_validate_20261031/collect.py \
  --root /efs/prescottsun/SPT-JEPA/runs/ideas_validate600_20261031 \
  --output /efs/prescottsun/SPT-JEPA/reports/ideas-validate600-20261031

CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 \
  /efs/prescottsun/project_jepa/envs/lewm/bin/python -B \
  experiments/ideas_validate_20261031/plot.py \
  --results /efs/prescottsun/SPT-JEPA/reports/ideas-validate600-20261031/results.json
```

Requirements: `initial-tensor-identity.json` has `passed: true`, all 24 matching run IDs, 550 original tensors per run, and a passed within-family identity check; the protocol's preflight directory contains `review.json` with `passed: true`. The collector requires the exact 24 runs and 120 successful endpoint IDs, common source revision, matching config hashes, exact step 600, passed health checks, matching parameter accounting, 50 binary planning outcomes and the same planning cases/budget/normalization, and identical 64-window state audit and information/health windows within each task. Probe train and validation episodes must be disjoint.

Outputs: `results.json`, `RESULTS.md`, five figures in PNG and SVG, and `figures.json` with figure and input hashes. Copy only these small artifacts locally. Weight files and `state-audit.npz` stay on EFS. `results.json` retains both seeds, per-case strict outcomes, paired gains/losses, full health/information summaries, raw temporal-variation denominators, all intervention scores, trained-query per-window CE, class counts, carrier norm certificates, training traces and total/trainable/frozen parameter counts.

CPU validation: four synthetic collector contracts passed in 0.098 seconds. All five figure functions rendered 10 PNG/SVG artifacts, and all five PNG layouts were visually inspected. The synthetic figures are explicitly marked and are not experimental results. No additional training or GPU diagnostic was run for this tooling.

# Comparison boundaries

1. Within-family comparisons isolate the declared intervention most closely. Carrier identity intentionally has fewer trainable parameters than the other carrier arms, while all three have equal total parameters and identical complete initial states. Cross-family results and the historical TF600/untrained values are descriptive, not matched-capacity causal comparisons. The carrier's 128 training-frame calibration is additional disclosed observation access.
2. The learned chart changes the geometry of the target as well as prediction. A lower normalized latent error does not by itself prove better physical prediction: inspect raw denominators, retained information, correct-versus-wrong input errors and actual planning. An invertible chart can preserve information while making a Euclidean planning metric worse; a certificate alone cannot establish useful geometry. A bound that does not constrain learned weights cannot explain a gain through restriction of those weights.
3. Clean rollout uses frame zero and recorded actions. Teacher-forced prefix prediction and shifted-observed carry receive new real observations; their advantage cannot be credited to autonomous dynamics. Other-initial and other-action interventions test use of correct inputs under an observational change, not the true physical outcome of a newly executed intervention. Off-support sensitivity is weaker evidence still.
4. S3 prototypes come from the same real clip and are held fixed across diagnostic conditions. They enter the auxiliary head, not the planner or causal predictor. A low prototype/mask-only score exposes a shortcut. CE/log(K) normalizes the uniform-class reference, not label entropy, irreducible uncertainty, gradient magnitude or effective active output capacity. Compare conditions within each trained arm; do not rank the three mechanisms by cross-arm CE. The independent path-query summary additionally reports segment class support.
5. Both seeds share the same 50 planning cases. Seed-average intervals resample case indices jointly and condition on these two trained models. They do not count 100 independent cases, estimate training-seed uncertainty, correct exploratory multiple comparisons, or undo repeated development use of the same validation cases. Two tasks and two seeds remain visible, with no pooled headline that hides sign changes.
6. These 600 updates are inside the original 1,249-step warmup. A finite checkpoint is not learned control; a weak early gate is not proof of permanent architectural failure. Successful short gates do not automatically authorize longer runs or establish novelty.
