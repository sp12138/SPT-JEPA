# Decoder geometry at6000: original-command evaluation

All48 endpoints exit0; 13/16 identity evaluations reproduce every old6000 case outcome. Same cases, solver, budget, normalization and strict protocol are verified. Model/metric/source hashes and recorded candidate-bank archives pass integrity checks.

| Task | Model | Seed | Primary I/G/R | Fresh I/G/R |
|---|---|---:|---|---|
| tworoom | carrier-identity | 1 | 80 / 74 / 78 | 74 / 78 / 74 |
| tworoom | carrier-identity | 2 | 86 / 80 / 60 | 80 / 78 / 58 |
| tworoom | temporal-forecast | 1 | 2 / 14 / 16 | 4 / 16 / 18 |
| tworoom | temporal-forecast | 2 | 4 / 6 / 2 | 2 / 4 / 4 |
| cube | carrier-identity | 1 | 14 / 8 / 12 | 12 / 12 / 8 |
| cube | carrier-identity | 2 | 12 / 24 / 8 | 10 / 8 / 10 |
| cube | temporal-forecast | 1 | 24 / 8 / 14 | 14 / 24 / 12 |
| cube | temporal-forecast | 2 | 14 / 18 / 18 | 16 / 12 / 10 |

I=identity, G=decoder Gram, R=same-spectrum random rotation; values are strict success percentages.

| Task | Model | Cases | Contrast | Difference pp | Conditional95% interval |
|---|---|---|---|---:|---|
| tworoom | carrier-identity | primary | decoder-minus-identity | -6.0 | [-16.0, 4.0] |
| tworoom | carrier-identity | primary | decoder-minus-rotated | +8.0 | [-2.0, 18.0] |
| tworoom | carrier-identity | fresh | decoder-minus-identity | +1.0 | [-8.0, 11.0] |
| tworoom | carrier-identity | fresh | decoder-minus-rotated | +12.0 | [3.0, 22.0] |
| tworoom | temporal-forecast | primary | decoder-minus-identity | +7.0 | [2.0, 13.0] |
| tworoom | temporal-forecast | primary | decoder-minus-rotated | +1.0 | [-6.0, 8.0] |
| tworoom | temporal-forecast | fresh | decoder-minus-identity | +7.0 | [2.0, 13.0] |
| tworoom | temporal-forecast | fresh | decoder-minus-rotated | -1.0 | [-8.0, 5.0] |
| cube | carrier-identity | primary | decoder-minus-identity | +3.0 | [-5.0, 10.0] |
| cube | carrier-identity | primary | decoder-minus-rotated | +6.0 | [-3.0, 15.0] |
| cube | carrier-identity | fresh | decoder-minus-identity | -1.0 | [-5.0, 3.0] |
| cube | carrier-identity | fresh | decoder-minus-rotated | +1.0 | [-4.0, 6.0] |
| cube | temporal-forecast | primary | decoder-minus-identity | -6.0 | [-14.0, 2.0] |
| cube | temporal-forecast | primary | decoder-minus-rotated | -3.0 | [-10.0, 4.0] |
| cube | temporal-forecast | fresh | decoder-minus-identity | +3.0 | [-2.0, 8.0] |
| cube | temporal-forecast | fresh | decoder-minus-rotated | +7.0 | [0.0, 15.0] |

Intervals jointly resample50 cases across the two fixed training seeds. They do not quantify training-seed population uncertainty; multiple exploratory contrasts are fully retained. Fresh pairs come from the existing validation pool, not an independent test dataset.

The fixed decoder metric does not support promotion across both tasks. TwoRoom carrier changes from83% to77% on primary and77% to78% on fresh cases. Its advantage over rotation on fresh cases is evidence that the learned axes can matter, but it does not establish a gain over native Euclidean cost. TwoRoom temporal-forecast rises from3% to10% on both case sets, while rotation reaches9%/11%; that gain does not identify decoder-specific geometry as the cause. Cube effects change sign across seeds or case sets, and no family has a consistent positive effect.

Identity replay differences (zero-based case indices):
- primary-identity-cube-carrier-identity-seed1: old10% -> actual14%; 3:0->1, 29:0->1.
- primary-identity-cube-temporal-forecast-seed1: old24% -> actual24%; 29:1->0, 42:0->1.
- fresh-identity-cube-temporal-forecast-seed2: old18% -> actual16%; 43:1->0.

All original model/runtime byte checks, checkpoint hashes and evaluation signatures agree; every instrumented native cost has zero arithmetic error. These facts do not establish the cause of the five changed historical case outcomes. The table and paired comparisons use actual contemporaneous identity evaluations, and exact historical trajectory reproducibility is not claimed.

All16 first-bank action/goal/endpoint hash comparisons agree across I/G/R: True.

Every metric's three costs are recorded on identical banks within its run. First-bank hashes across treatments are reported separately; later CEM populations may diverge and are not treated as common banks. Raw trace/participation rank are coordinate-dependent decoder diagnostics, not task value or missing-information proofs.

This intervention changes only the fixed planning metric. Original command actions remain unchanged, including any existing command/executed-action mismatch. The independent action-projection control is not combined here. All completed roots are held and owned idle dispatchers/monitors are closed; no extension is launched.
