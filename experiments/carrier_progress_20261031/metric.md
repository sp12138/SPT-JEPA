# Frozen Cube metric diagnostic

Four existing checkpoints: carrier identity/bounded, seeds 1/2, exactly update 600. Each is evaluated with native Euclidean, full observation-covariance whitening, and full one-model-step-increment-covariance whitening. These are explicit controller variants and a causal diagnostic of metric insufficiency, not a new training algorithm or novelty claim.

## Definition and data

One frozen manifest selects 128 distinct training episodes, one length-8 clip per episode, by NumPy seed 1042. The authoritative `offsets.npy`, `lengths.npy`, and `episode_ids.npy` define the rows. Train/validation memberships and metadata hashes are verified. Every checkpoint and both covariance estimates use exactly these clips; no validation images or target physical states enter calibration. Each model step is the dataset's fixed frameskip (5 physical steps for this Cube dataset).

Let `z[i,t]` be the frozen checkpoint's **actual native state embedding**, including its learned chart. The observation samples are all 1,024 `z[i,t]`; the increment samples are the 896 within-clip differences `z[i,t+1]−z[i,t]`. Never form a difference across clip boundaries. Both estimates center their respective samples. With empirical covariance `S`, the fixed Ledoit–Wolf formula chooses lambda from those training samples:

`C = (1−lambda) S + lambda trace(S)/d I`, `W = C^(−1/2)`.

The symmetric eigendecomposition uses float64; its sole floor is `eps_float64 × d × largest_eigenvalue`. The floor count, all eigenvalues, lambda, covariance, mean, and whitening matrix are retained. Constant representations are rejected. The runtime matrix is float32, matching the native state path. Frame correlation means this is a deterministic regularizer, not a claimed iid-optimal risk guarantee. No validation-tuned shrinkage, rank, exponent or coefficient is introduced. [scikit-learn's primary API description](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html) documents the implemented estimator; the artifact also records the installed package version.

For the native terminal predicted state `p` and native encoded goal `g`, the changed cost is `sum(((p−g) W)^2)`. The training mean cancels in the difference. No extra normalization or global cost scaling is applied. If native GoalMSE uses mean reduction, that exact reduction is retained. The declared native objective's keys and reduction are recorded. Native time pooling remains last predicted step against last goal frame.

Observation whitening emphasizes directions relative to their observed variation. Increment whitening emphasizes directions relative to their one-step variation. Neither statistic identifies goal semantics or controllability: nearly static nuisance can receive high weight, and observational motion variance is not an intervention-response Gramian. Failure is informative; do not assume either change should improve Cube.

## Boundary of the intervention

`metric_evaluate.py` calls the existing `as_jepa_tools.benchmark_evaluate.main` and temporarily replaces only the objective passed into `ShootingCostEvaluator`. Native image transforms, action normalization, goal encoding, raw-state autoregressive predictions, history length, CEM samples/iterations/elites/seeds, 50 strict cases, physical evaluator and closed-loop budget remain unchanged. The identity branch returns the original GoalMSE tensor directly, with no casting or matrix multiplication. No model wrapper changes what the predictor consumes.

Every actual CEM bank is scored under all three metrics from the same already-produced terminal embeddings. No extra model or physical rollouts are made. `metric-costs.npz` retains all costs and stable diagnostic orders; `metric-candidate-banks.json` retains native call order, candidate-action hashes, goal and endpoint-difference hashes, observation/history hashes, and selected indices. All three evaluation arms pay the same diagnostic scoring overhead. Native CEM itself continues to use its original `torch.topk` tie handling. The diagnostic stable sort does not alter the solver.

After a metric changes an elite set, future CEM populations and later physical trajectories may diverge. Therefore same-bank rank differences and final closed-loop task gains are different outcomes. Compare each identity evaluation against its original native600 same-case result; exact identity is established by CPU native-solver contracts, with actual GPU/environment repeat checked separately by the parent. Raw arrays stay on EFS.

## Preparation and execution

After the parent freezes the new source and source record:

```bash
python -B experiments/carrier_progress_20261031/metric_prepare.py \
  --root /efs/prescottsun/SPT-JEPA/runs/carrier_metric_20261031 \
  --calibration-root /efs/prescottsun/SPT-JEPA/runs/carrier_metric_calibration_20261031 \
  --dispatch-root /efs/prescottsun/SPT-JEPA/runs/carrier_diagnostics_20261031 \
  --source-record /efs/prescottsun/SPT-JEPA/runs/carrier_progress_20261031/source-record.json
```

This only writes manifests/plans. Calibration's `plan.json`, workers and exits have their own root; four jobs occupy dev13 GPUs 4–7 when the parent launches them. Metric evaluation's `plan.json` supplies 12 records for the parent's **single combined dispatcher**, with actual exits under `carrier_diagnostics_20261031`, avoiding racing workers on the same GPUs. Its planned slots are dev13 GPUs 4–7 and dev15 GPUs 0–7. The parent must verify all four calibrations before dispatching evaluations. The prepared plan is not an automatic continuation or launch authorization.

Calibration refuses to overwrite existing matrices/reports and records weights/config/source/data/manifest hashes. Evaluation checks them before invoking the native evaluator. Completed metrics include the explicit metric name, original objective configuration and candidate audit. Seven CPU contracts passed: deterministic/increment-only calibration, rank-deficient regularization, native identity/read-only/RNG equality, explicit quadratic form and ordering, unchanged rollout inputs/call count, exact native CEM actions/costs, and the shared-bank archive.
