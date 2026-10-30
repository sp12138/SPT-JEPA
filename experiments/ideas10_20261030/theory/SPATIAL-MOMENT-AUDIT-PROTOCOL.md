# Fixed natural-image falsification of candidate D

Declared before reading the new audit results, 2026-10-30. No GPU or learned segmentation is used.

- Each task: 128 training and 128 disjoint validation episodes, one image at the same deterministic sampled window start as `BenchmarkWindows` (seeds 20261030/20261031).
- Three fixed nonnegative measures: luminance; luminance finite-difference gradient magnitude; RGB chromaticity deviation from the image's mean chromaticity. No object mask, threshold or coordinate-dependent target is fitted.
- Model orders K=4 and K=8. The fixed mixed 2D monomial bases are declared in `audit_spatial_moments.py`. Moment matrices include both x and y shifts. Marginal roots are not treated as paired 2D positions.
- Report Hankel condition, x/y shift commutator, simultaneous-diagonalization residual, deviation of roots from the unit circle, calibration fit and held-out Fourier moment fit. Fit positive weights by NNLS after projecting nodes to the unit circle; the projection error is reported, not hidden.
- Fit moments are exactly those used by the three pencils. Test moments are every remaining pair of frequencies in `{0,...,8}²`. No test-frequency selection.
- Two amplitude perturbations multiply the measure by `1+.1*cos(2πx)` and `1+.1*sin(2π(x+y))`. Support is unchanged; exact finite atomic measures would preserve all nodes. Report matched joint-root displacement on the image torus and change in an unweighted mixed-moment address code.
- The declared permissive numerical screen requires condition ≤1e8, commutator ≤.05, maximum radial error ≤.05, held-out relative residual ≤.2, and RMS root displacement ≤.02 image widths under both perturbations. Report continuous distributions regardless of this screen. These are diagnostic tolerances, not proposed training hyperparameters.
- A physical-coordinate ridge readout is allowed only if ≥90% pass in each split. Labels never influence the measures, order, fit, matching or numerical screen; failing families do not get a selective physical probe. Report both coverage and accuracy if a probe is run.
- Exact K-atom synthetic positive controls must first recover paired 2D nodes and held-out moments. The audit records source SHA256.

A failed audit rejects the raw-measure realization of D. It does not disprove the classical sparse-measure identity, nor establish that a future learned atomizer will work. A successful audit would still establish neither algorithmic novelty nor benchmark planning improvement.
