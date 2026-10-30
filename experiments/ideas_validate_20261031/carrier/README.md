# Fixed spatial carrier and learned chart: bounded F gate

This is a conditional test of [candidate F](../../ideas10_20261030/theory/INFORMATION-CARRIER-CHART.md), not a claim to have invented invertible networks or to have improved either task. This directory supplies model/calibration contracts. The parent campaign owns source freezing, calibration jobs, preflight, scheduling and evaluation. No GPU was launched while implementing these files.

## Model/config integration

Class: `as_jepa_lewm.carrier.CarrierLeWM`. The shared model factory selects it with top-level `model_class="carrier"`. It inherits `RobustLeWM.loss`, `prepare`, `predict` and the native action/renderer interfaces. Existing TF/noise/rate settings stay unchanged. There is no new objective, policy supervision, action data, or planner cost.

Add these model kwargs to each established task/seed configuration:

```json
{
  "carrier_chart": "bounded",
  "carrier_bound": 0.5,
  "carrier_hidden": 64,
  "carrier_mean": ["64 calibration floats"],
  "carrier_scale": ["64 strictly positive calibration floats"]
}
```

The arrays above are schematic, not executable string values. Copy the actual arrays from the calibration JSON's `model_kwargs` before any initial-weight export. The three arms use `identity`, `unconstrained`, or `bounded` with identical other settings and calibration. Encoding refuses absent calibration. Portable checkpoints persist the frozen encoder, fixed projection, mean/scale and chart; loading uses their serialized model config and needs no external calibration file.

The original ViT initializes normally and is then frozen and kept in eval mode. Its patch tokens are pooled to a position-major 4×4 grid, flattened to 3072, and multiplied by the exact RP64 matrix used in the earlier spatial audit: NumPy `SeedSequence([20261030,3072])`, standard-normal entries divided by sqrt(64). CLS is excluded. The unused original QueryPool remains instantiated/frozen to preserve every original tensor and initialization draw. No pretrained weights, privileged coordinates, segmentation or batch statistics are used.

## Three matched chart arms

All arms instantiate the same 64→64→64 residual MLP with 8,320 parameters:

    T(u) = u + c A2 tanh(A1 u+b1) + b2, c=0.5.

`identity` freezes its exact identity initialization. The two learned modes train the same tensors. `unconstrained` uses raw matrices; `bounded` divides each matrix by max(1, its operator norm), with a small conservative FP32 rounding margin. Full double-precision SVD computes that norm; no power-iteration estimate or unverified norm assumption is used. The chart and frozen carrier execute in FP32 even under autocast.

The real-arithmetic residual Lipschitz bound is c and the resulting distance bounds are 0.5||u-v||≤||T(u)-T(v)||≤1.5||u-v||. A contraction iteration gives the inverse. This guarantee concerns the calibrated carrier, not every piece of information in the original image, noisy-information invariance, a linear probe, or correct Cube goal semantics. Finite-precision executions are tested numerically; the proof does not assert exact continuous-function behavior for quantized floating-point arithmetic.

Use ordinary Kaiming-uniform first-layer weights/bias from the private fixed seed 20261031; second-layer weight/bias are zero. Thus all initial codes, base losses and outer RNG states are exactly identical. First-layer raw norm is **1.11903959**, bounded effective norm **0.99999909**; the constraint is active immediately. The learned chart gradients are deliberately allowed to differ. No seed search or task-specific norm threshold was used.

## Training-only calibration

Run once per task/encoder seed, after committing the source and before preparing final training configurations:

```bash
python experiments/ideas_validate_20261031/carrier/calibrate.py \
  --config ORIGINAL_TASK_SEED_CONFIG \
  --output NEW_CALIBRATION_JSON \
  --device cuda --source-commit FROZEN_COMMIT
```

CPU is the default; the campaign owner chooses and launches CUDA explicitly. The helper uses `sdpa` on CUDA and `sdpa_math` on CPU and records the choice. Defaults are 128 distinct training episodes, one randomly selected valid window per episode using fixed selection seed1042, and the initial image of each window. The native dataset defines the exact image resizing. Only the image is measured; no physical labels or validation dataset are opened. All three arms share this additional training-observation calibration exposure.

Per-coordinate population mean/std is computed in FP64 and stored in FP32. The fixed standard-deviation numerical floor is 1e-6; its activation count is recorded. The helper refuses existing output files and encoder warm starts. It records source/config/data/projection/encoder hashes, exact selected episode IDs, starts and HDF5 rows. The encoder hash is checked unchanged after calibration. This is a fixed training-derived affine transform, not adaptive batch normalization.

## Gradients and parameter accounting

`model.audit_expected_gradients()` returns:

| Prefix | Identity | Learned arms |
|---|---|---|
| encoder | frozen | frozen |
| pool | frozen | frozen |
| carrier_chart_module | frozen | active |
| predictor | active | active |
| renderer | active | active |
| action_encoder | optional | optional |

The action encoder can have zero gradient at exact zero-output predictor initialization. Chart input-layer gradients can also be zero on the first step because its output matrix is zero; the chart group has an active output gradient. Native full-model health must not require frozen encoder gradients.

CPU construction using the exact existing TwoRoom/Cube seed1 configurations verifies **all550 original tensors** unchanged and the outer RNG identical for every mode:

| Task | Total parameters, all arms | Identity trainable | Learned trainable | Identity frozen | Learned frozen |
|---|---:|---:|---:|---:|---:|
| TwoRoom | 51,105,454 | 45,443,294 | 45,451,614 | 5,662,160 | 5,653,840 |
| Cube | 51,109,834 | 45,447,674 | 45,455,994 | 5,662,160 | 5,653,840 |

The identity arm has exactly 8,320 fewer trainable parameters, by design. The three arms have identical total parameter counts. Relative to original TF, they freeze the encoder and original pool and add the chart; this is not a claim of equal trainable capacity or compute to a fully trainable TF/LeWM baseline. Learned bounded versus unconstrained is the direct equal-parameter constraint control. Runtime measurements should also expose FP32 carrier/SVD overhead.

## Verification and endpoint interpretation

Eight CPU contracts pass on dev13 with CUDA hidden, including the exact audited feature construction, pairwise-distance/inverse checks with large weights, a concrete unconstrained bound violation, native initial loss/metric/RNG equality, frozen parameter gradients, finite learned gradients and model/optimizer state roundtrip. The calibration-statistics helper also passes a deterministic constant-coordinate case. [Full CPU report](cpu-contracts.json). CPU tests use a reduced renderer/encoder for loss/backward; the separate initialization/count check uses actual full task configs. Real GPU execution/calibration remains the parent's preflight responsibility.

At600, report `model.carrier_chart_module.norm_certificate()` including raw/effective norms and which constraints activate. Also report native forecast/readout/planning and whether the chart differs meaningfully from identity. A preserved probe or an invertibility certificate alone is not a successful world model. If unconstrained and bounded trajectories remain nearly identical, that limits evidence about the bound even if the fixed carrier itself improves a task. Use the same held-out cases and original planner; no continuation follows automatically.
