# Hybrid128 technical2step preflight — complete, held

All16 real endpoints exited0. All8 runs stopped at2 native updates and passed strict fullstate/gradient audits plus CPU native module/AdamW/scheduler restore, cursor256 and CPU RNG roundtrip. CUDA RNG records were verified without applying them or making further GPU calls. All555 initial state tensors match per pair. Fixed Q keeps9 exact initial tensors, no gradients and zero AdamW states; adaptive Q has9 finite gradients/states and a positive group norm.

Model source `f2ae93d4f0da2fb32021bfade7426b11ea06602b`; strict helper `3986d87dbbe8811ec2319f9706d7c40e1aabf594`. The unstarted8e9 candidate remains held. All8 owned slots were verified empty; dispatchers182433/1102486/1294856 were closed07:15:10–20UTC and the new root is HOLD. No600 training has been authorized or launched.

| Task | Seed | F centered RMS | Q mean / centered / total RMS | First total grad fixed / adaptive | Adaptive Q grad before clip |
|---|---:|---:|---|---|---:|
| tworoom | 1 | 0.9475 | 0.0815 / 1.0550 / 1.0582 | 1.0771 / 1.6737 | 1.2811 |
| tworoom | 2 | 0.9757 | 0.1480 / 1.0920 / 1.1020 | 1.0571 / 1.6076 | 1.2111 |
| cube | 1 | 0.9816 | 0.1165 / 1.0097 / 1.0164 | 0.7375 / 1.0567 | 0.7568 |
| cube | 2 | 0.9546 | 0.1480 / 1.0436 / 1.0540 | 0.7530 / 1.1143 | 0.8214 |

First-batch F/Q statistics match **exactly** within every real pair. The measured Q variation is not a constant offset. An isolated initial-Q copy on cached tokens gave exactly zero raw/normalized/σ-relative difference across8 versus1024 repeated inputs and grad/no-grad in all8 audits. This is only a Q-path statement on those measured inputs; it does not certify all encoder precision, data or batch regimes. No new encoder call, labels or fit were used for that assay.

Native clipping stayed unchanged at1 and was called once per update. Adaptive Q adds enough gradient to lower TwoRoom shared predictor/renderer gradients after the first clip by about34–36% relative to fixedQ; Cube fixedQ remains below the threshold while adaptiveQ crosses it. This is an observed optimization coupling, not proof that AdamW effective steps shrink by the same fraction. No LR/clip/floor remedy was introduced. Future effects cannot be assigned uniquely to representation information without considering this coupling.

Peak allocation was106.05GiB for fixedQ and106.64GiB for adaptiveQ. Two-step elapsed/seconds-per-step include startup, detached diagnostic reductions and checkpoint IO, and should not estimate600-step throughput. Numerical/integrity checks passed; two updates establish neither useful causal learning nor planning performance.

Full artifacts: `/efs/prescottsun/SPT-JEPA/runs/hybrid_carrier_preflight_strict_20261031`; compact reviewed data and provenance: `/efs/prescottsun/SPT-JEPA/reports/hybrid-carrier-preflight-20261031/summary.json`. The original calibration and all earlier holds remain unchanged.
