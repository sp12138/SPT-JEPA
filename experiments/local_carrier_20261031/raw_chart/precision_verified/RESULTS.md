# Precision repair: completed CPU and captured-GPU gates

Commit `1060dde0e36ba4fb32a54e4a141c6e1d2cbd7a98` passed six raw-interface CPU contracts, four legacy numeric contracts, and strict reload of the actual100-step checkpoint with all557 tensors unchanged. Raw precision reconstructs from config; the dynamic branch stays disabled. Legacy defaults and identity loss/original gradients/RNG remain exact. Nonlinear raw arithmetic is intentionally changed from0c and is not claimed bitwise equivalent.

The repaired production adjoint was tested on the exact captured `[128,4,64,16]` batch/strides, with the unchanged64-iteration cap, atol0 and rtol1e-6. It passed all public-FP32 residual certificates while the process stayed `high`/TF32-enabled. Maximum solution error divided by RHS norm was4.751e-8; the corresponding certified residual bound was9.502e-8. An unattainable1e-12 public-rounding tolerance correctly raised the explicit certificate error.

| Captured-shape solver | One warmed call | Incremental allocated peak |
|---|---:|---:|
| IEEE FP32 diagnostic comparator | .706 ms | 18.125 MiB |
| Local FP64 production solver | .783 ms | 62.016 MiB |

These are single-call solver-only measurements, excluding the full world model and parameter-gradient reductions; they establish neither stable throughput nor end-to-end memory cost. No model training occurred in this verification. GPU process1042689 exited0; GPU0 was verified empty and the verification root held.

Source/proof: `B/runs/raw_chart_precision_verify_20261031/source-v1` and `B/reports/raw-chart-precision-verified-20261031`. Root subsequently authorized one **fresh** TwoRoom seed1 0→200 numerical gate with this fixed source, original batch128/schedule and native25-step checkpoint cadence. This supersedes the earlier proposed100→200 mixed-precision resume. No planning, four-run restart or automatic>200 continuation is authorized.
