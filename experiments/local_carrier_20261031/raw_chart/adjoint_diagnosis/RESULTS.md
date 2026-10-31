# Actual adjoint failure: precision replay

One diagnostic fresh TwoRoom seed1 run reproduced the original failure at112, including identical loss/fm/rate and learning rates. All557 initial tensors matched. The only operational change was native checkpoint_every25: the preserved100-step fullstate passed model/optimizer/scheduler/RNG/cursor checks, and all210 frozen tensors remained identical. No training followed the captured failure.

Captured target shape/stride were `[128,4,64,16]` / `[4096,1024,16,1]`:32,768 slot vectors,7,680 nonzero. Hidden/derivative arrays retained the full `[128,4,64,64]` shape. Exact tensors remain on EFS; capture SHA256 `a204b9fbacb6a70adc8526f8e7e52c9ccb50ef81afeabafbaf9edf3188f774cd`.

| Replay arithmetic | Unchanged residual test | Iterations | Max solution error / RHS norm |
|---|---|---:|---:|
| Captured high / TF32 allowed | Fails:1 slot | 64 | 4.709e-7 |
| IEEE FP32, TF32 disabled | Passes all | 4 | 4.751e-8 |
| FP64 | Passes all | 4 | 7.030e-16 |

All runs used the original64-iteration cap, atol0, rtol1e-6 and full tensor shapes/strides. The independent reference solved every16×16 adjoint system in FP64. Error denominators above are the FP64 per-slot norms of the captured RHS, not global batch norms. No norm-underflow rows occurred. Under captured arithmetic the same one residual violation remained from iteration4 through64.

The effective-weight bound was only `.0037104111` (inverse lower bound `.9962895889`), so this is not evidence of mathematical noninvertibility or an ill-conditioned chart. Original-high solutions remained close to the oracle; the actual precision policy caused the residual certification failure. This evidence supports a local arithmetic repair without relaxing tolerances, not a performance claim or proof about all future inverses.

Native source stayed `0c9ae74840cf28247ae74341d2656e2d9ddd457c`; instrumentation helper was committed/frozen as `a5f98d16df9a8caa366aa0f858efb81f01b8312e`. Capture PID1041089 exited at the expected failure; operand-only replay PID1041172 exited0. dev13 GPU0 was verified empty and the diagnostic root held. No production source/tolerance changed and no automatic restart occurred.

Full EFS proof: `B/reports/raw-chart-adjoint-diagnosis-20261031`; heavy operands and valid25/50/75/100 checkpoints remain under `B/runs/raw_chart_adjoint_capture_20261031`.
