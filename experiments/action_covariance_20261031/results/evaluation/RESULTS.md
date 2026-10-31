# Fixed action-correlation CEM: complete48-endpoint comparison

All48 endpoints exited0; original/Cholesky arms retain the same checkpoint, cases, native source, scaler, goal metric and300×30 CEM budget. All first-plan costs/actions are finite and fixed-map checks pass. No training, clipping or optimal-action supervision was added.

| Task | Family | Seed | Primary original → correlated | Fresh original → correlated |
|---|---|---:|---:|---:|
| cube | TF64 | 1 | 20 → 36% | 14 → 32% |
| cube | TF64 | 2 | 20 → 30% | 16 → 30% |
| cube | local1024 | 1 | 16 → 24% | 24 → 24% |
| cube | local1024 | 2 | 22 → 20% | 20 → 34% |
| cube | native-raw | 1 | 24 → 36% | 28 → 30% |
| cube | native-raw | 2 | 26 → 36% | 26 → 40% |
| tworoom | F64free | 1 | 90 → 90% | 82 → 84% |
| tworoom | F64free | 2 | 94 → 92% | 86 → 90% |
| tworoom | TF64 | 1 | 2 → 6% | 4 → 6% |
| tworoom | TF64 | 2 | 4 → 4% | 2 → 2% |
| tworoom | native-raw | 1 | 80 → 74% | 76 → 80% |
| tworoom | native-raw | 2 | 72 → 62% | 70 → 68% |

| Task | Family | Cases | Two-seed mean original → correlated | Paired change,95% case CI (pp) |
|---|---|---|---:|---:|
| cube | TF64 | fresh | 15 → 31% | +16.0 [+5.0,+27.0] |
| cube | TF64 | primary | 20 → 33% | +13.0 [+3.0,+23.0] |
| cube | local1024 | fresh | 22 → 29% | +7.0 [-5.0,+18.0] |
| cube | local1024 | primary | 19 → 22% | +3.0 [-7.0,+13.0] |
| cube | native-raw | fresh | 27 → 35% | +8.0 [-2.0,+19.0] |
| cube | native-raw | primary | 25 → 36% | +11.0 [-2.0,+24.0] |
| tworoom | F64free | fresh | 84 → 87% | +3.0 [-3.0,+9.0] |
| tworoom | F64free | primary | 92 → 91% | -1.0 [-6.0,+4.0] |
| tworoom | TF64 | fresh | 3 → 4% | +1.0 [+0.0,+3.0] |
| tworoom | TF64 | primary | 3 → 5% | +2.0 [+0.0,+5.0] |
| tworoom | native-raw | fresh | 73 → 74% | +1.0 [-7.0,+9.0] |
| tworoom | native-raw | primary | 76 → 68% | -8.0 [-15.0,+0.0] |

2/24 new original-arm success vectors differ from historical references. These are retained separately; the new original arm is the paired comparator. Exact changed-case indices and all per-seed confidence intervals are in results.json.

Paired comparisons use the newly recomputed original arm, never best-of historical/replay. Bootstrap unit is case; averaged-seed intervals resample the same case across both seeds and condition on the two trained models. No environment-repeat variance or training-seed population uncertainty included. Primary/fresh stay separate. Later CEM populations differ; only the first random bank is common within task/case set.

Common controller intervention. No changed world-model learning objective, metric, weights, labels, action bounds or sample budget. Candidate differs by task: TwoRoom F64free vs Cube local1024. Exposure/capacity mismatches remain.

This control tests search parameterization, not a new learning algorithm. TwoRoom and Cube use different candidate families; native/candidate capacity differs and Cube local6000 has less exposure than native/TF12000. Existing colored-noise CEM precedent is [iCEM](https://proceedings.mlr.press/v155/pinneri21a.html). Matrices were frozen solely from pre-existing training128 cohorts. No task results select covariance, factorization or model. No further runs are triggered.
