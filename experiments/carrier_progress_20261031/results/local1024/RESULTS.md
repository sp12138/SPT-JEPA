# Fixed local-vs-global1024-D measurement diagnostic

| Seed | Measurement | Readout | Train pose R² | Val pose R² | Train xyz cm | Val xyz cm | Val angle | ≤3cm | ≤15° | Joint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | patch16_shared4 | linear | 0.8967 | -8.6296 | 1.69 | 17.20 | 21.89° | 2.90% | 38.17% | 1.12% |
| 1 | patch16_shared4 | rbf | 0.4240 | -0.0336 | 5.49 | 7.63 | 20.82° | 14.51% | 40.62% | 6.03% |
| 1 | cell8_shared16 | linear | 0.9036 | -4.8628 | 1.51 | 12.11 | 22.32° | 7.59% | 35.04% | 2.68% |
| 1 | cell8_shared16 | rbf | 0.3830 | -0.0749 | 4.83 | 6.65 | 22.15° | 21.21% | 35.04% | 7.37% |
| 1 | dense1024 | linear | 0.9788 | -5.6858 | 1.04 | 17.72 | 23.51° | 0.67% | 31.25% | 0.22% |
| 1 | dense1024 | rbf | 0.3645 | -0.0648 | 6.10 | 7.88 | 22.53° | 12.72% | 36.16% | 4.69% |
| 2 | patch16_shared4 | linear | 0.9094 | -10.9725 | 1.80 | 17.51 | 23.62° | 3.35% | 28.57% | 0.45% |
| 2 | patch16_shared4 | rbf | 0.4257 | -0.0517 | 5.46 | 7.60 | 21.90° | 14.51% | 36.83% | 5.13% |
| 2 | cell8_shared16 | linear | 0.9186 | -5.7296 | 1.39 | 13.36 | 22.62° | 3.57% | 33.26% | 1.12% |
| 2 | cell8_shared16 | rbf | 0.3766 | -0.0607 | 4.97 | 6.74 | 21.59° | 22.54% | 39.29% | 8.04% |
| 2 | dense1024 | linear | 0.9778 | -6.3105 | 1.04 | 18.78 | 22.52° | 0.89% | 32.59% | 0.22% |
| 2 | dense1024 | rbf | 0.3498 | -0.0625 | 6.22 | 8.05 | 22.34° | 8.93% | 34.60% | 3.12% |

Exactly three fixed1024-D measurements, two predeclared readouts and both seeds. Physical labels train only diagnostic readouts.1024 correlated training frames are128 episodes. High-dimensional ridge/RBF failure does not prove absent information; readout scores are not planning success. Local-vs-dense differences mix retained information, covariance and finite-sample readout bias. A new world-model algorithm and broad novelty are not established.

CPU wall time 6.1s; process peakRSS 0.55GiB. Same original readout helpers, train-only standardization, mean-loss ridge.001, training-median RBF bandwidth, training-only quaternion codebook and cube24-invariant pose error. No extra parameter search.

All training and validation frame errors, prior RP64/full-feature controls, matrix/array hashes and prediction arrays are retained.
