# Frozen global decoder-response metric: technical pilot only

No GPU job, complete calibration, planner run, or checkpoint update is authorized by these files. Root must review, commit and freeze the helper source before a GPU pilot. The candidate pilot reads one already selected training clip, runs one directional JVP and eight VJPs, and exits. Its rank-at-most-eight matrix must not be used for planning.

## Definition and fixed sampling rule

For a frozen renderer velocity `v(x_t, t, z; a)`, let `J = ∂v/∂z`, holding noisy image `x_t`, time `t`, and anchor `a` constant. With P=3×64×64 output values, define `G = E[JᵀJ/P]`. This is a PSD response seminorm, not an inverse covariance, likelihood Hessian, full-sampling decoder Jacobian or task value function. Only the 64-dimensional renderer condition is differentiated. The encoder and predictor provide fixed values; their Jacobians are excluded.

Use the existing 128 distinct training-episode clips, in their frozen manifest order. For each clip, draw one uniform horizon from 1…7, one time, one image Gaussian noise, and the posterior/forecast conditioning randomness. All streams use the fixed CPU generator rule in `core.py`, seed 20261031, shared across models and seeds; the draws do not depend on task outcomes. Averaging these 128 sampled contexts is a Monte Carlo estimate over the cohort's seven targets, not evaluation of every frame or a claim of converged population geometry.

Actual held F64/TF64 configurations use shifted logit-normal time, **not uniform/Beta**: `u~U(0,1)`, `s=sigmoid(.8+.8 Φ⁻¹(u))`, followed by the native `shift_timestep(s, model.training_shift)`. Both currently have flow_logit_shift=0; `t=1` is pure noise. `x_t=(1-t)y+t ε`, and the native supervised velocity is `y-ε`. The 64×64 target `y` comes from the selected future frame; the anchor is the same clip's first RGB frame. It is not a goal image, current replanning frame, or extra observation acquired during planning.

Condition sampling follows the trained renderer's distributional structure: correlated posterior noise with the checkpoint's rho/sigma and a Bernoulli forecast_fraction choice. Existing F64/TF64 use rho=.8, sigma=.1 and forecast_fraction=.5. Forecasts start at the noisy first state and consume only the recorded action prefix. All modules are in eval mode, hence predictor dropout and renderer condition dropout are disabled. This is an explicit deterministic-inference response diagnostic with matched conditioning/noise laws, **not exact replay of the stochastic training loss or historical RNG**. Recorded actions carry no optimality label.

For each context use exactly eight independent output Rademacher vectors `r` (entries ±1/√P). Compute `u=Jᵀr` by VJP, then average `uuᵀ`. Conditional on J, this is unbiased for `JᵀJ/P`. Sum the 128 context estimates in FP64, giving 1,024 rank-one contributions and a possible full-rank 64×64 matrix. There is no identity mixture, inverse, eigenvalue floor, grid search or success-dependent probe count. Tiny negative eigenvalues at FP64 roundoff may only be reported/clipped for a later square-root factor; substantial negatives fail integrity. Rank and participation rank must be reported. Eight probes need not estimate every direction accurately.

## Cost and falsification gates

The VJP path reuses one forward graph for eight backwards per clip; it avoids 64 forward-direction passes or 12,288 scalar output backwards. The pilot separately records forward-AD/finite-difference time, eight-VJP time and peak memory. Its 128× VJP timing extrapolation excludes future IO/encoding and is only an estimate. Full calibration, if subsequently approved, reports actual offline wall time. Planning would use one fixed matrix and the existing CEM rollout budget, with no DiT call or anchor recomputation during planning.

Both paths force FP32/highest matmul precision and `sdpa_math`, including explicit math SDPA dispatch. Fused attention forward-AD support is not assumed. The real tiny AnchoredWorldDiT CPU contract must pass first; full trained-model GPU support remains a pilot question. The direct renderer API is used because render_states/sample have no_grad decorators. One fixed random input direction is checked by central differences at `h=eps32^(1/3) max(1,||z||)` and h/2. Both relative errors must be below .03 and the response nonzero. This threshold is a technical numerical gate, never selected by planning results. Eight-probe energy versus exact JVP energy is reported without a pass threshold; noisy agreement must not select a model or estimator.

After a successful technical pilot, the only proposed controller arms are native Euclidean, G, and QᵀGQ using one fixed Haar-QR random orthogonal Q (seed 20261031). The last control preserves the entire spectrum/rank and scrambles orientation. All three costs should be recorded on each identical candidate bank via the existing `carrier_progress_20261031/metric_core.py` infrastructure; identity must exactly replay native cost. Later CEM banks may diverge and cannot be compared as a common bank. Raw prediction inputs, goal encoding, rollout, case manifests, strict success, CEM parameters and time pooling stay unchanged. No evaluator is implemented here.

Failure is informative: a nearly zero/rank-deficient G may expose anchor or noisy-image shortcuts; a strong response can still emphasize texture/effector motion. Global averaging can erase state-dependent contact geometry. Better local velocity response does not imply better endpoint pose or planning, and Euclidean pixel response does not identify desired goal semantics. This is a controller diagnostic, not a new sample-level learning method or novelty claim.

## Inputs and candidate command

Cube cohort: `/efs/prescottsun/SPT-JEPA/runs/carrier_metric_20261031/train-clips128.json`.

TwoRoom cohort: `/efs/prescottsun/SPT-JEPA/reports/native-lewm600-20261031/source-bncal-v1/train128-tworoom.json`. This already fixed 128-clip training cohort has the same verified manifest schema. No new cohort is selected. The code reads validation episode identifiers only to verify split disjointness; it reads no validation images, outcomes or physical labels.

Candidate first checkpoint: `/efs/prescottsun/SPT-JEPA/runs/carrier_learn6000_20261031/cube-carrier-identity-seed1/model-step0006000`. This is only a technical pilot, not model selection. The loader checks the export target/kwargs exactly against the parent training config and strictly loads original weights.

After root freezes a new committed snapshot and supplies its source-record:

```text
PYTHONPATH=<frozen helper source plus recorded read-only runtime/dependencies> \
<existing lewm python> -B <frozen source>/experiments/decoder_geometry_20261031/pilot.py \
  --checkpoint <the checkpoint above> --manifest <Cube cohort above> \
  --source-record <new frozen source-record.json> \
  --output <new independent EFS pilot output> --device cuda
```

CPU contracts: `CUDA_VISIBLE_DEVICES= ... python -B experiments/decoder_geometry_20261031/test_core.py`. The GPU pilot verifies its own frozen file hashes, records actual imported source hashes, checks the entire model state before/after, rechecks checkpoint bytes, and confirms all parameters frozen with no parameter gradients. No optimizer is constructed. JSON and a small response tensor artifact are written only to a new independent output directory; originals remain read-only.
