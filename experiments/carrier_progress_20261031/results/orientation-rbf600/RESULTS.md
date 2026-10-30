# One fixed RBF orientation probe

| Seed | Readout | Train invariant R² | Validation invariant R² | Validation angle | Within15° |
|---:|---|---:|---:|---:|---:|
| 1 | linear | 0.09844 | -0.07076 | 21.723° | 33.71% |
| 1 | rbf | 0.17959 | -0.01934 | 21.516° | 37.72% |
| 2 | linear | 0.09028 | -0.08089 | 22.200° | 36.16% |
| 2 | rbf | 0.17280 | -0.02592 | 21.747° | 36.38% |

Fixed nonlinear accessibility test, not total-information proof. Within-clip frames are correlated;1024 frames are128 episodes. Physical labels fit only probes; no proposed training objective changes. Failed RBF does not rule out other nonlinear readouts or finite-sample limitations.

Train-only coordinate standardization and centered RBF kernel; bandwidth is the median off-diagonal training distance. Fixed ridge .001 in mean-loss units, equivalent to n*.001 in the unnormalized Gram system. Same cube24 invariant labels, training-only quaternion codebook and pinned angular metric as the completed linear diagnostic. The linear probe reproduces its original results before this comparison.

No bandwidth/regularization/seed search. No active result was read. No encoder, GPU, world-model or planner update occurred.
