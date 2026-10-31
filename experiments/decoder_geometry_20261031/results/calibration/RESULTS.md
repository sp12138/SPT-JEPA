# Frozen global decoder geometry: eight calibrations complete

All eight exit0 and pass CPU integrity review. Each uses exactly128 original training clips and1024 finite VJP vectors;69 clips select forecast conditioning and59 observed posterior conditioning. All response traces are positive. Source/imports match original8e, weights and complete model state are unchanged, and the sampling schedule and fixed rotation agree across all eight models.

| Task | Model | Seed | Raw trace | Participation rank | Top3 trace | Condition number |
|---|---|---:|---:|---:|---:|---:|
| tworoom | carrier-identity | 1 | 0.013629 | 9.82 | 48.2% | 488 |
| tworoom | carrier-identity | 2 | 0.015498 | 8.91 | 52.5% | 623 |
| tworoom | temporal-forecast | 1 | 0.069134 | 2.31 | 97.6% | 19360 |
| tworoom | temporal-forecast | 2 | 0.054001 | 2.26 | 98.0% | 17103 |
| cube | carrier-identity | 1 | 0.012602 | 18.77 | 31.3% | 84 |
| cube | carrier-identity | 2 | 0.011499 | 20.70 | 28.9% | 68 |
| cube | temporal-forecast | 1 | 0.041563 | 3.06 | 89.8% | 15759 |
| cube | temporal-forecast | 2 | 0.036154 | 3.17 | 89.4% | 12470 |

All matrices have numerical rank64; participation rank describes concentration, not absent dimensions. TF response is concentrated in fewer directions than F64 in these coordinates. This is a reproducible description of the fixed estimates, not an explanation of task performance. Raw trace and spectral concentration depend on latent coordinates/scales, so they cannot establish superior representation quality across models.

Recomputed vector outer products match raw G. Gnorm differs only by the positive scalar64/trace(G); QᵀGnormQ preserves its spectrum. QR factors reproduce both quadratic forms without a floor, inverse or identity mixture. Single-point probe variance remains possible; these checks do not establish Monte Carlo or population convergence. The held original cohort and eight-probe rule were not changed.

Per-model calibration takes35–41 seconds internally, with20–23 seconds in VJPs; peak device allocation is about432MB. Original snapshots, candidates and preactivation HOLD are preserved. The completed root is held; owned idle dispatcher91717 and monitor91718 are closed. No planning, command projection, action clipping, training or further calibration was launched.
