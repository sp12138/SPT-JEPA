# Native Cube maturity reference at12000

All26 bounded endpoints exit0; both raw and train-only preciseBN variants are retained. All8000/10000/12000 full archives, finite portable states, optimizer/scheduler/RNG/cursor audits and source/config/case checks pass.

| Cases | Seed | Raw6000 | Cal6000 | Raw12000 | Cal12000 | TF12000 |
|---|---:|---:|---:|---:|---:|---:|
| primary | 1 | 24% | 16% | 26% | 30% | 18% |
| primary | 2 | 22% | 18% | 26% | 28% | 20% |
| fresh | 1 | 12% | 20% | 28% | 30% | 14% |
| fresh | 2 | 14% | 16% | 26% | 26% | 16% |

| Cases | Contrast | Mean change pp | Conditional95% interval |
|---|---|---:|---|
| primary | raw12000-minus-raw6000 | +3.0 | [-5.0, 11.0] |
| primary | cal12000-minus-cal6000 | +12.0 | [3.0, 22.0] |
| primary | raw12000-minus-TF12000 | +7.0 | [-3.0, 17.0] |
| primary | cal12000-minus-TF12000 | +10.0 | [0.0, 20.0] |
| primary | cal-minus-raw | +3.0 | [-2.0, 9.0] |
| fresh | raw12000-minus-raw6000 | +14.0 | [2.0, 26.0] |
| fresh | cal12000-minus-cal6000 | +10.0 | [0.0, 20.0] |
| fresh | raw12000-minus-TF12000 | +12.0 | [3.0, 22.0] |
| fresh | cal12000-minus-TF12000 | +13.0 | [6.0, 21.0] |
| fresh | cal-minus-raw | +1.0 | [-8.0, 11.0] |

Two native Cube seeds only. Original raw fullstate6000 resumed; staged8000/10000/12000 full archives gate continuation. Raw and fixed train128 preciseBN are separate variants. Original-command50-case protocols are unchanged; no projection treatment. Equal12000 updates/window exposure is not equal architecture, capacity or compute. Bootstrap conditions on two fixed seeds and realized environment outcomes, excluding seed-population and environment repeat variance. No automatic continuation or posthoc variant maximum.

Native has18,035,396 trainable parameters versus approximately51.1M total for TF. Native192-state and TF64-state errors are not comparable physical units. BN calibration changes exactly four running mean/variance buffers and no parameters, using the same training128 clips and1024/896 actual vectors. Historical and current outcomes are both preserved; neither variant is chosen after viewing task results.
