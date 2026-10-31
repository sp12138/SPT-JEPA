# Five bare Cube replay checks

All 5 endpoints exit 0; 2/5 bare repeats differ from historical case vectors.

| Check | Old / wrapped / bare success % | Bare vs old differing indices | Bare vs wrapped differing indices |
|---|---|---|---|
| cube-f64-primary-seed1 | 10 / 14 / 12 | 29 | 3 |
| cube-tf-primary-seed1 | 24 / 24 / 24 | none | 29, 42 |
| cube-tf-fresh-seed2 | 18 / 16 / 18 | none | 43 |
| cube-nativecal-primary-seed1 | 16 / 18 / 18 | 28 | none |
| cube-local-primary-seed1 | 16 / 18 / 16 | none | 35 |

A bare-repeat difference demonstrates that a wrapper is not necessary for that historical discrepancy. It does not establish zero wrapper effect or identify a specific numerical/reset cause. These are targeted reproducibility checks, not five independent benchmark seeds.

Cases, weights, CEM budget/seed, normalization, protocol and original model/evaluator source are verified. Actual device placement and EGL identifiers are retained alongside the original and wrapped dispatch records. No reset, simulator seed, action, planner or model change was introduced.

All original, wrapped and bare case vectors remain available; none replaces historical results. The new root is held and its owned idle dispatcher closed.
