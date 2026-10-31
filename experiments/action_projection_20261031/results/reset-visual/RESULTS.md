# Cube current-frame reset audit (visual v2)

One predeclared audit completed: cases **3,28,35**, three fresh Worlds, one identical bounded command per case. All nine paired comparisons retain exactly equal poststep physical fields, all **286 captured visual arrays** (including actual skybox/grid texture bytes, camera, lights, material, geometry/mesh), expanded XML and pixel-level segmentation. Initial/goal dataset images are exact. The poststep image is a newly rendered **224×224 current frame with T=1**, not history or stale buffer.

| Case | MAE, 0–255 | RMSE, 0–255 | Max difference | Changed pixels | Robot share of squared error |
|---|---:|---:|---:|---:|---:|
| 3 | 0.0143–0.0192 | 0.299–0.370 | 32–32 | 1.23–1.89% | 95.8–97.6% |
| 28 | 0.0202–0.0216 | 0.462–0.510 | 39–66 | 0.83–1.30% | 84.6–89.5% |
| 35 | 0.0152–0.0195 | 0.318–0.365 | 24–40 | 0.86–1.60% | 86.3–94.2% |

Ranges cover all three repeat pairs per case; these pairs are dependent. The variation is sparse and low in average magnitude, with larger isolated differences: only **0.040–0.108%** of full-frame pixels differ by more than 8 levels. **84.6–97.6%** of squared error lies on robot geometry, versus **0.012–0.666%** on the cube. No measured texture/camera/material shift appears; the XML random-skybox hypothesis is unsupported here. The precise render mechanism and any contribution to P0 flips remain untested. No reset/controller/source correction follows from this audit.

The fixed pair (0,1) is shown for every case; there is no best/worst selection. Full RGB, segmentation, visual arrays and physical state remain on EFS. `summary.json` retains all nine numeric comparisons, artifact hashes and provenance.

Cost: exit0, **42.05 seconds** wall time (0.0117 GPU-hour reserved), 18 native reset calls, 36 native internal settling steps, 9 explicit bounded steps and 9 descriptive segmentation renders. No learned model or training. dev2 GPU5 returned to 0 MiB and both runner/child PIDs exited. Native cleanup emitted ignored EGL destructor warnings after successful artifact completion; this does not identify the source of image variation.

Frozen helper: `826840174f0638c8990a6c5346ee245e5fb58bfd`; original native source `7a1ab4f`, frozen v1 parent `d75eb24`, all source/plan hashes checked before launch. Full report: `/efs/prescottsun/SPT-JEPA/reports/cube-reset-state-20261031/visual-v2`.
