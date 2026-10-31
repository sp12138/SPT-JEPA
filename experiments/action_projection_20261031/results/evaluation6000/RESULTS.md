# Common bounded-controller evaluation at 6000

All 32 bounded endpoints exit 0: 24 original-root + 8 recovery-root. The two failed Cube replay controls remain exit 1; two recovered TwoRoom replay controls exit 0.

This common bounded controller does not yield a cross-task winner. TwoRoom TF improves to10% primary/7% fresh, mainly through seed1, while native-cal reaches76%/84% and F64 reaches80%/82%. In Cube, F64 falls on all four seed/case-set evaluations to8%/7% mean; local1024 reaches21%/22% against native-cal17%/19% and TF14%/20%, but these small Cube advantages do not establish a substantial gain and local1024 remains weaker on TwoRoom. The controller intervention does not resolve the contrasting task weaknesses.

| Task | Model | Seed | Primary historical → bounded % | Fresh historical → bounded % |
|---|---|---:|---|---|
| tworoom | native-cal | 1 | 82 → 74 | 84 → 80 |
| tworoom | native-cal | 2 | 78 → 78 | 88 → 88 |
| tworoom | TF64 | 1 | 2 → 16 | 4 → 12 |
| tworoom | TF64 | 2 | 4 → 4 | 2 → 2 |
| tworoom | F64 | 1 | 80 → 72 | 74 → 76 |
| tworoom | F64 | 2 | 86 → 88 | 80 → 88 |
| tworoom | local1024 | 1 | 60 → 66 | 72 → 68 |
| tworoom | local1024 | 2 | 60 → 64 | 60 → 64 |
| cube | native-cal | 1 | 16 → 14 | 20 → 20 |
| cube | native-cal | 2 | 18 → 20 | 16 → 18 |
| cube | TF64 | 1 | 24 → 20 | 14 → 20 |
| cube | TF64 | 2 | 14 → 8 | 18 → 20 |
| cube | F64 | 1 | 10 → 6 | 12 → 6 |
| cube | F64 | 2 | 12 → 10 | 10 → 8 |
| cube | local1024 | 1 | 16 → 20 | 24 → 18 |
| cube | local1024 | 2 | 22 → 22 | 20 → 26 |

| Task | Model | Cases | Contrast | Difference pp | Conditional 95% interval |
|---|---|---|---|---:|---|
| tworoom | native-cal | primary | bounded-minus-historical | -4.0 | [-12.0, 3.0] |
| tworoom | native-cal | primary | bounded-minus-bounded-TF64 | +66.0 | [54.0, 77.0] |
| tworoom | TF64 | primary | bounded-minus-historical | +7.0 | [3.0, 12.0] |
| tworoom | TF64 | primary | bounded-minus-bounded-native-cal | -66.0 | [-77.0, -54.0] |
| tworoom | F64 | primary | bounded-minus-historical | -3.0 | [-11.0, 4.0] |
| tworoom | F64 | primary | bounded-minus-bounded-TF64 | +70.0 | [59.0, 81.0] |
| tworoom | F64 | primary | bounded-minus-bounded-native-cal | +4.0 | [-9.0, 18.0] |
| tworoom | local1024 | primary | bounded-minus-historical | +5.0 | [-2.0, 13.0] |
| tworoom | local1024 | primary | bounded-minus-bounded-TF64 | +55.0 | [42.0, 68.0] |
| tworoom | local1024 | primary | bounded-minus-bounded-native-cal | -11.0 | [-22.0, 0.0] |
| tworoom | native-cal | fresh | bounded-minus-historical | -2.0 | [-9.0, 4.0] |
| tworoom | native-cal | fresh | bounded-minus-bounded-TF64 | +77.0 | [67.0, 87.0] |
| tworoom | TF64 | fresh | bounded-minus-historical | +4.0 | [1.0, 8.0] |
| tworoom | TF64 | fresh | bounded-minus-bounded-native-cal | -77.0 | [-87.0, -67.0] |
| tworoom | F64 | fresh | bounded-minus-historical | +5.0 | [-5.0, 15.0] |
| tworoom | F64 | fresh | bounded-minus-bounded-TF64 | +75.0 | [64.0, 85.0] |
| tworoom | F64 | fresh | bounded-minus-bounded-native-cal | -2.0 | [-12.0, 8.0] |
| tworoom | local1024 | fresh | bounded-minus-historical | +0.0 | [-6.0, 5.0] |
| tworoom | local1024 | fresh | bounded-minus-bounded-TF64 | +59.0 | [46.0, 71.0] |
| tworoom | local1024 | fresh | bounded-minus-bounded-native-cal | -18.0 | [-28.0, -8.0] |
| cube | native-cal | primary | bounded-minus-historical | +0.0 | [-5.0, 6.0] |
| cube | native-cal | primary | bounded-minus-bounded-TF64 | +3.0 | [-6.0, 11.0] |
| cube | TF64 | primary | bounded-minus-historical | -5.0 | [-13.0, 3.0] |
| cube | TF64 | primary | bounded-minus-bounded-native-cal | -3.0 | [-11.0, 6.0] |
| cube | F64 | primary | bounded-minus-historical | -3.0 | [-8.0, 2.0] |
| cube | F64 | primary | bounded-minus-bounded-TF64 | -6.0 | [-15.0, 2.0] |
| cube | F64 | primary | bounded-minus-bounded-native-cal | -9.0 | [-18.0, 0.0] |
| cube | local1024 | primary | bounded-minus-historical | +2.0 | [-5.0, 9.0] |
| cube | local1024 | primary | bounded-minus-bounded-TF64 | +7.0 | [0.0, 15.0] |
| cube | local1024 | primary | bounded-minus-bounded-native-cal | +4.0 | [-5.0, 14.0] |
| cube | native-cal | fresh | bounded-minus-historical | +1.0 | [-7.0, 9.0] |
| cube | native-cal | fresh | bounded-minus-bounded-TF64 | -1.0 | [-11.0, 9.0] |
| cube | TF64 | fresh | bounded-minus-historical | +4.0 | [-4.0, 12.0] |
| cube | TF64 | fresh | bounded-minus-bounded-native-cal | +1.0 | [-9.0, 11.0] |
| cube | F64 | fresh | bounded-minus-historical | -4.0 | [-10.0, 2.0] |
| cube | F64 | fresh | bounded-minus-bounded-TF64 | -13.0 | [-21.0, -6.0] |
| cube | F64 | fresh | bounded-minus-bounded-native-cal | -12.0 | [-21.0, -3.0] |
| cube | local1024 | fresh | bounded-minus-historical | +0.0 | [-8.0, 7.0] |
| cube | local1024 | fresh | bounded-minus-bounded-TF64 | +2.0 | [-6.0, 10.0] |
| cube | local1024 | fresh | bounded-minus-bounded-native-cal | +3.0 | [-5.0, 12.0] |

Historical reference is fixed for all comparisons, never chosen from old/wrapped/bare outcomes. Case bootstrap conditions on two fixed training seeds and observed simulator realizations; it excludes training-seed population and environment repeat variance. Native LeWM and the proposed models differ in capacity/objective/recipe. No matched-capacity or new learning-method claim.

Projection is applied uniformly before candidate model rollout and before returning raw actions. TwoRoom also changes recorded action history; Cube lacks equivalent native clipping, so this changes its actual controller. It does not change learned weights or sample-level losses. All within-task comparisons use the same bounded treatment and preserve CEM proposal/update rules and evaluation budget.

All four task/case-set first candidate-bank input hashes agree across all models and seeds; later banks are not assumed identical. All32 evaluations report zero returned-command bound violations. Across the four model families,21.6–29.8% of scored TwoRoom action coordinates and2.4–3.3% of Cube coordinates were clipped. These are descriptive fractions along each treated run, not common-bank causal comparisons. Box feasibility alone does not establish joint training-distribution support or prediction accuracy.

The four pass-through controls and five bare repeats are retained separately in results.json. Two bare runs also differ from their historical vectors; wrapper zero effect and a specific reset cause are not established. No historical result is replaced. Fresh cases come from the existing validation pool.
