# Frozen Cube spatial readout diagnostic

Prepared only; root owns source freezing and GPU scheduling. No change to a model, planner, learning loss, existing run or data. Physical labels fit diagnostic readouts only.

Two identity-carrier checkpoints at step 600 (seeds 1/2) each encode the same existing 128 training clips and 64 held-out clips, eight frames per clip. All rows, episode membership, dataset metadata, checkpoints and original reference embeddings are checked. Train and validation episodes are disjoint. The dataset loader reads RGB and recorded actions, but only RGB enters the frozen encoder.

For each encoder, retain position-addressed 4×4 pooled tokens and the original 16×16 patch grid (224-pixel images, patch size 14, token width 192). These contain 3,072 and 49,152 coordinates. Each spatial representation is compared as RP64, nested RP192, and the full vector. The original RP64 matrix is reproduced exactly from `SeedSequence([20261030, input_dim])`; additional 64-column blocks use `[20261030, input_dim, block]` for blocks 1/2. Concatenation divided by √3 gives RP192. This preserves the original projection directions in its first 64 columns, with the necessary global rescaling. No labels determine pooling, projection or normalization.

The 4×4 RP64 codes must match saved training and validation codes at fixed `rtol=atol=2e-5`, after applying original carrier calibration. Raw-code comparison uses the raw reference recovered from those saved normalized codes; no independent raw reference was previously saved. Projection matrices match exactly. Extraction fails on a mismatch.

All six feature representations use the existing train-only coordinate standardization and ridge coefficient .001 in mean-loss units. With standardized features X and centered labels Y, the dual solution predicts `K_test (XXᵀ + nλI)⁻¹Y + mean(Y)`. This equals the previous primal ridge for any feature dimension, avoiding a 49,152-square inverse. Two predeclared 4×4 controls, RP64 and full3,072, also use the previous centered RBF ridge: bandwidth equals the median off-diagonal training distance, λ=.001. No validation tuning or other kernel is added. Thus there are six linear and two RBF rows per seed.

The diagnostic fits the existing sign- and cube-symmetry-invariant orientation tensor (81 entries) and block xyz together. Targets are independent output coordinates, so concatenation does not reweight or couple fits. Angle decoding uses only the training quaternion codebook and the pinned 24-symmetry task metric. Validation scores cover the same 448 future frames (64×7) as previous orientation probes; fitting uses all 1,024 training frames. Report orientation R²/angles/≤15°, xyz R²/Euclidean error in cm/≤3 cm, and their joint threshold. Joint readout accuracy is not planning success or the required three-physical-step hold. Retain constant-mean and true-label codebook-quantization references, training scores, per-frame errors, and differences against both original RP64 probes.

The 1,024 training frames come from only 128 correlated clips. Feature dimensions and effective statistical capacity differ. A high-dimensional failure can reflect sample size, regularization or inductive bias; it does not establish that information is absent. A positive full-feature comparison would support an accessible-information loss at a particular compression stage, under these readouts, rather than establish the sole cause of planning failures.

Expected resources: two independent extractions of 1,536 frames, no backward pass or optimizer; approximately 0.32 GB raw arrays per seed plus small projections. CPU analysis reuses each train Gram/cross-Gram for its linear and selected RBF fit, with approximately 1–2 GB temporary memory. Large arrays and predictions stay on EFS. Small JSON/Markdown can be copied locally afterward.

## Commands after root freezes the code and allocates slots

Use the normal frozen-source, runtime-overlay and clear-lewm `PYTHONPATH`, and the existing LeWM Python environment. Run once for each seed with separate new output directories:

```sh
python -B -m experiments.carrier_progress_20261031.carrier_spatial_probe extract --seed 1 --device cuda --output /efs/prescottsun/SPT-JEPA/reports/carrier-spatial600-20261031/seed1-features
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -B -m experiments.carrier_progress_20261031.carrier_spatial_probe analyze --features /efs/prescottsun/SPT-JEPA/reports/carrier-spatial600-20261031/seed1-features --output /efs/prescottsun/SPT-JEPA/reports/carrier-spatial600-20261031/seed1-readouts
```

CPU contracts passed before any extraction: dual/primal maximum errors `3.77e-13` and `1.69e-12` for under- and overparameterized synthetic problems; RBF equals the existing fixed formula; original RP64 and nested first block are exact. These contracts make no model or encoder call. Run entry point is `...carrier_spatial_probe contracts`.
