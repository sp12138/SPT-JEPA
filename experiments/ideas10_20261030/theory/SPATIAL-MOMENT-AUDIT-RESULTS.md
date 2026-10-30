# Candidate D fails the fixed natural-image test

2026-10-30. **Reject the current raw-measure implementation of D.** Its ideal sparse-measure algebra is correct, but none of the measured image families satisfies the prerequisite remotely well enough. No new segmentation model, additional rank grid or GPU training was used to repair that premise.

The predeclared [protocol](SPATIAL-MOMENT-AUDIT-PROTOCOL.md) tests 128 training and 128 episode-disjoint validation images per task, three generic image measures, and K=4/8. All 3,072 fits completed. Each of the 12 task/measure/order combinations has **0/256** images passing the declared permissive numerical screen. Physical-position probes were therefore not run. All pixel operations, model orders, perturbations, fit/test frequencies and numerical gates were fixed before examining these results.

| Task | Measure | K | Median Hankel condition | Median mixed-shift commutator | Median radial root error | Median held-out moment relative error | Median amplitude-induced address drift |
|---|---|---:|---:|---:|---:|---:|---:|
| TwoRoom | luminance | 4 | 204 | .302 | .958 | 9.012 | .248 |
| TwoRoom | luminance | 8 | 1,028 | .176 | .951 | 6.394 | .218 |
| TwoRoom | gradient | 4 | 30 | .355 | .954 | 2.383 | .065 |
| TwoRoom | gradient | 8 | 200 | .209 | .971 | 1.752 | .065 |
| TwoRoom | chromatic deviation | 4 | 59 | .246 | .980 | 3.209 | .140 |
| TwoRoom | chromatic deviation | 8 | 316 | .174 | .993 | 2.638 | .161 |
| Cube | luminance | 4 | 50 | .367 | .882 | 6.852 | .130 |
| Cube | luminance | 8 | 147 | .315 | .960 | 5.762 | .134 |
| Cube | gradient | 4 | 22 | .271 | .819 | 2.591 | .029 |
| Cube | gradient | 8 | 240 | .255 | .948 | 2.029 | .039 |
| Cube | chromatic deviation | 4 | 15 | .326 | .859 | 4.208 | .047 |
| Cube | chromatic deviation | 8 | 129 | .224 | .909 | 3.657 | .084 |

Amplitude-induced drift uses the fixed mixed-sine multiplier `1+.1*sin(2π(x+y))`, and is RMS matched displacement as a fraction of image width/height on the image torus. These are diagnostic pencil-derived addresses, not claims to have identified actual objects. The other fixed perturbation and all quantiles are in the summary JSON.

**The failure is principally model mismatch, not merely floating-point conditioning.** Median condition numbers range from about 15 to 1,028, whereas the recovered roots lie far from the unit circle and the x/y shift matrices do not approximately share eigenvectors. Projecting roots to physical positions and fitting nonnegative weights still gives large held-out errors. The best relative residual is 1.75, worse than predicting zero on these held-out nonzero frequencies. Increasing K from 4 to 8 improves residuals somewhat without making the model appropriate.

The exact K-atom positive controls use the same mixed 2D pencils and recover the paired locations and held-out moments within the predeclared tolerances. Thus the negative natural-image result is not explained by silently using two marginal point sets or a solver that also fails its idealized case.

This audit rejects the current natural-image parameterization. It does not disprove Prony/FRI theory, nor prove that no learned feature measure could ever become atomic. But supplying such a measure would be a new representation-learning problem; treating that unknown as already solved would remove the substance of the original proposal.

Artifacts:

- [Executable audit](audit_spatial_moments.py).
- [Small summary, source SHA, positive controls and split IDs](ideas10-natural-moments-summary-20261030.json).
- Full per-image results preserved at `/efs/prescottsun/SPT-JEPA/reports/ideas10-natural-moments-20261030.json`.

No world-model training, benchmark success measurement or novel-method claim follows from this result.
