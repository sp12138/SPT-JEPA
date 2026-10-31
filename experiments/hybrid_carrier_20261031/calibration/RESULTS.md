# Joint initial calibration — complete, held

All4 real calibration jobs exited0 on dev13 GPUs0–3. Frozen source `f2ae93d4f0da2fb32021bfade7426b11ea06602b`;512 original training initial frames,64 ViT calls, zero optimizer updates. Every model/encoder/projection hash stayed unchanged. Dispatcher1102228 was closed at06:49:03UTC after verifying the owned GPUs empty; the calibration root is HOLD. No two-step or600 training has been launched.

| Task | Seed | F64 global RMS | Q64 global RMS | Q reciprocal scale | Q top16 energy |
|---|---:|---:|---:|---:|---:|
| TwoRoom |1| .201041 | .000287237 |3481.45|38.26%|
| TwoRoom |2| .184011 | .000306046 |3267.47|38.09%|
| Cube |1| .824433 | .00513135 |194.88|37.10%|
| Cube |2| .806565 | .00462140 |216.38|36.51%|

Every calibrated branch has centered RMS≈1 and uses a positive scalar with no floor. Scalar centering/rescaling adds no information and, in exact arithmetic, discards none beyond the existing64-D measurement. This does not establish task sufficiency, learning stability, numerical equivalence across modes/batch sizes, or useful additional information in Q.

F uses the original raw projection and exact same-seed ViT, but replaces historical diagonal scaling with one RMS. Original versus production-high calibration means differ by up to .00341–.00511; the metric and noisy-channel weighting are not historical F64's exact function. TwoRoom Q is amplified by3267–3481, making the upcoming already-cached-token mode/batch audit and native gradient/clipping observations necessary. Do not infer healthy optimization from calibration RMS alone.

Full raw measurements, exact128-frame manifests, hashes, backend/precision and branch statistics remain under `/efs/prescottsun/SPT-JEPA/runs/hybrid_carrier_calibration_20261031`. Small source/launch/closure and per-seed summaries accompany this report. Original requires-grad-v1 CPU failure remains preserved; the accepted stop-gradient version passed4 CPU contracts.
