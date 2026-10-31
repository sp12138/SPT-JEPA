# Native versus TF Cube: fixed readability and forecast diagnostics

All six frozen jobs passed and are held; exact owned dispatcher179907/monitor179908 were closed after GPUs0–5 became idle. Complete parameters and buffers stayed bit-exact. Exactly7,168 new RGB frames,224 predictor calls and zero renderer/training/planning calls were used. Native6k/12k validation states were reused; only TF12k required new validation states.

Native12k has more linearly readable xyz and a more consistent linearly decoded forecast than native6k; it also beats TF12k on these linear xyz measures in both seeds. This does not establish uniformly better dynamics: with the same predeclared RBF, TF12k has lower mean xyz forecast error in both seeds. Pose remains weak under both fixed probe classes.

All position values below are mean Euclidean error in cm, on the same448 future frames. Teacher receives actual past observations; shifted carry copies the last actual observation. Neither is a causal rollout.

| Model | Probe | Observed | Causal forecast | Teacher one-step | Initial copy | Shifted observed copy |
|---|---|---:|---:|---:|---:|---:|
| native6000-seed1 | linear | 5.11 | 9.83 | 9.15 | 10.81 | 6.61 |
| native6000-seed1 | rbf | 5.86 | 7.81 | 7.39 | 11.10 | 7.25 |
| native6000-seed2 | linear | 5.07 | 11.56 | 10.97 | 11.02 | 6.69 |
| native6000-seed2 | rbf | 5.92 | 9.11 | 8.15 | 11.19 | 7.27 |
| native12000-seed1 | linear | 3.90 | 4.93 | 4.62 | 10.99 | 5.92 |
| native12000-seed1 | rbf | 5.41 | 6.00 | 5.82 | 11.56 | 7.01 |
| native12000-seed2 | linear | 3.65 | 4.17 | 4.06 | 10.90 | 5.81 |
| native12000-seed2 | rbf | 5.11 | 5.12 | 5.09 | 11.36 | 6.82 |
| tf12000-seed1 | linear | 5.06 | 6.70 | 4.76 | 10.71 | 6.57 |
| tf12000-seed1 | rbf | 5.53 | 4.98 | 5.34 | 10.55 | 6.77 |
| tf12000-seed2 | linear | 4.69 | 4.48 | 4.45 | 11.34 | 6.51 |
| tf12000-seed2 | rbf | 5.12 | 4.77 | 4.95 | 10.69 | 6.54 |

Mean cube-symmetry pose angle in degrees, with identical conditions:

| Model | Probe | Observed | Causal forecast | Teacher one-step | Initial copy | Shifted observed copy |
|---|---|---:|---:|---:|---:|---:|
| native6000-seed1 | linear | 20.64 | 22.32 | 22.17 | 19.66 | 20.64 |
| native6000-seed1 | rbf | 21.69 | 22.13 | 21.88 | 20.51 | 21.55 |
| native6000-seed2 | linear | 21.04 | 21.26 | 21.26 | 21.19 | 21.05 |
| native6000-seed2 | rbf | 21.50 | 22.15 | 22.44 | 20.77 | 21.25 |
| native12000-seed1 | linear | 22.00 | 22.25 | 21.74 | 21.13 | 22.01 |
| native12000-seed1 | rbf | 22.21 | 21.78 | 21.78 | 21.59 | 22.18 |
| native12000-seed2 | linear | 21.45 | 23.52 | 23.22 | 21.56 | 21.47 |
| native12000-seed2 | rbf | 22.45 | 22.88 | 22.66 | 21.45 | 22.40 |
| tf12000-seed1 | linear | 21.71 | 24.11 | 23.12 | 20.50 | 21.34 |
| tf12000-seed1 | rbf | 22.27 | 22.20 | 22.14 | 21.83 | 21.93 |
| tf12000-seed2 | linear | 21.05 | 22.99 | 21.77 | 20.31 | 20.77 |
| tf12000-seed2 | rbf | 22.14 | 22.52 | 22.43 | 22.08 | 21.97 |

Threshold readout accuracy; entries are observed → causal forecast. These are not planning-success or hold rates.

| Model | Probe | xyz≤3cm | pose≤15° | Joint |
|---|---|---:|---:|---:|
| native6000-seed1 | linear | 40.0% → 2.5% | 41.3% → 33.7% | 15.0% → 1.3% |
| native6000-seed1 | rbf | 33.0% → 7.4% | 34.2% → 36.2% | 11.2% → 2.2% |
| native6000-seed2 | linear | 39.3% → 1.8% | 38.2% → 35.7% | 14.3% → 0.4% |
| native6000-seed2 | rbf | 32.1% → 8.3% | 36.2% → 31.0% | 12.1% → 3.6% |
| native12000-seed1 | linear | 54.5% → 34.6% | 35.5% → 32.6% | 21.2% → 12.7% |
| native12000-seed1 | rbf | 28.1% → 18.1% | 35.5% → 35.0% | 12.5% → 6.0% |
| native12000-seed2 | linear | 60.3% → 48.0% | 36.2% → 28.3% | 21.7% → 15.0% |
| native12000-seed2 | rbf | 34.2% → 31.2% | 32.8% → 32.1% | 12.1% → 10.9% |
| tf12000-seed1 | linear | 43.8% → 25.4% | 35.5% → 27.9% | 15.6% → 6.7% |
| tf12000-seed1 | rbf | 40.8% → 45.8% | 33.9% → 35.0% | 12.7% → 15.8% |
| tf12000-seed2 | linear | 45.3% → 43.5% | 38.2% → 34.2% | 17.6% → 14.3% |
| tf12000-seed2 | rbf | 41.1% → 47.1% | 33.9% → 32.6% | 13.4% → 15.0% |

Exact xyz MSE decomposition, in cm² per coordinate. The cross term is retained, not interpreted as dynamics error.

| Model | Probe | Observation MSE | Forecast–observation discrepancy | Cross term | Forecast MSE |
|---|---|---:|---:|---:|---:|
| native6000-seed1 | linear | 15.22 | 26.58 | -2.78 | 39.03 |
| native6000-seed1 | rbf | 19.21 | 6.68 | 1.26 | 27.15 |
| native6000-seed2 | linear | 14.54 | 40.28 | -1.65 | 53.17 |
| native6000-seed2 | rbf | 19.61 | 12.52 | 3.15 | 35.28 |
| native12000-seed1 | linear | 9.48 | 4.46 | -0.98 | 12.97 |
| native12000-seed1 | rbf | 14.69 | 2.01 | 0.06 | 16.75 |
| native12000-seed2 | linear | 8.63 | 3.44 | -2.03 | 10.03 |
| native12000-seed2 | rbf | 13.37 | 1.45 | -1.35 | 13.46 |
| tf12000-seed1 | linear | 15.93 | 26.80 | -21.10 | 21.63 |
| tf12000-seed1 | rbf | 18.79 | 0.97 | -3.43 | 16.33 |
| tf12000-seed2 | linear | 13.24 | 7.26 | -8.97 | 11.53 |
| tf12000-seed2 | rbf | 16.32 | 0.90 | -2.73 | 14.48 |

Native linear decoded forecast–observation discrepancy falls from26.58/40.28 to4.46/3.44cm² per coordinate between6k and12k. Its observed xyz error improves5.11/5.07→3.90/3.65cm; forecast error improves9.83/11.56→4.93/4.17cm. RBF also improves within native, but TF12k RBF forecast4.98/4.77cm beats native12k5.997/5.118cm. The linear comparison alone cannot establish an intrinsic dynamics ordering.
TF has large negative xyz cross terms: linear−21.10/−8.97cm² per coordinate and RBF−3.43/−2.73. Its decoded forecast can therefore be closer to the physical label than the decoded actual observation. That does not make the model more accurate than reality: observed-state probe error is correlated with the forecast displacement, and a fixed imperfect readout need not commute with prediction. Observation readout error is not a lower bound on forecast readout error.
All held-out observed tensor81 variance-weighted R² values are negative. Native linear training R² is.375–.409 but validation is−.194 to−.328; TF linear training is.048/.080 and validation−.016/−.003. RBF validation remains negative for all six. This documents weak generalization of these fixed pose probes, not absent orientation information or an impossibility of learning it.
Constant training-mean readout: xyz18.62cm, pose23.13°,32.8125% within15°, joint0%. True physical target decoded through the same training-only quaternion codebook: mean angle.552°,100% within15°. Thus codebook quantization alone is not the main restriction in these probe scores. The32.8% constant rate is a distribution-specific diagnostic reference, never a theoretical task-success ceiling.
The pose metric and≤15° inequality exactly match the pinned strict task: tasks.py:93–103 and runner.py:274–285 use the same24-symmetry geodesic function; the actual protocol also requires≤3cm and three consecutive physical steps. Probe and task have different operands (readout versus actual future state, compared with actual state versus desired goal), and probe frames are five steps apart. No hold claim is possible.
Controls/limits: exactly the old128 distinct training clips×8 frames and64 validation clips×7 future frames; all probe fits use observed training encodings, train-only standardization, fixed ridge.001 and fixed training-median RBF bandwidth. Tensor81/codebook/xyz targets are diagnostic privileged labels only. Native192-state/18.0M parameters differs from TF64-state/51.1M; probe dimension and model capacity are not matched. Native6k→12k has extra exposure. Native12k versus TF12k matches update/window exposure but not architecture/compute. No raw latent-MSE comparison or causal mediation claim is made.128 clips are128 episodes despite1,024 correlated frames.
Complete per-mode training/validation metrics, feature R², pose/xyz decompositions, hashes and original source paths remain in results.json; mechanism-summary.json is the small publication artifact. No extra fits, cohorts, counterfactual rollouts, dimension controls or new planning runs were added.
