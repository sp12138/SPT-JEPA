# Fresh raw-chart numerical recovery gate: 200 updates

The single TwoRoom seed1 run completed 200 native batch128 updates and the held-out full-state/gradient endpoint; both exits are zero. This crosses the original step112 adjoint failure with the approved local FP64 chart/inverse/adjoint implementation, unchanged tolerances and global training precision. This establishes this bounded numerical recovery, not reliable longer training or task performance.

| Actual native training record | Update100 | Update200 |
|---|---:|---:|
| Loss / flow / rate | 2.57447 / 1.70860 / 3.46346 | 2.25916 / 1.34629 / 3.65147 |
| Chart gradient norm | .00263669 | .000999711 |
| Predictor / action gradient norm | .0576867 / 4.73497e-7 | .0802504 / .000107689 |
| Recent seconds/update | 1.23589 | 1.13698 |
| Peak allocated GiB | 107.5713 | 107.5713 |

All557 initial tensors match the failed original seed1 initialization exactly; all210 frozen tensors remain unchanged. Final model and portable weights agree; optimizer339/339 parameter tensors, scheduler, RNG and sample cursor25600 pass. Held-out gradients are finite and positive for predictor(.216617), renderer(.179660), actions(.000134279) and chart(.00154281); frozen encoder/pool gradients are zero. The learned chart's residual Lipschitz upper bound is .00621350, giving exact-real bounds [.99378650,1.00621350]; input normalization is active and output normalization is inactive. This certificate supplements the actual finite-precision residual checks; it is not a task score or a global floating-point proof.

The native training timer reports250.34s including checkpoint/finalization; the finished-status `seconds_per_step=5.87661` is finalization duration, not steady training latency. The train process occupied290.70s including initialization/imports. Portable checkpoints25,50,…,200 exist; native `last.ckpt` is the final200 complete state, not eight separately archived optimizer states. This checkpoint cadence is an operational difference from the original600 plan and was authorized before execution.

Provenance: model1060dde0e36ba4fb32a54e4a141c6e1d2cbd7a98; helper6e6d69e3bb14bd9e977a48a39edf3e2926b8faae. Run `/efs/prescottsun/SPT-JEPA/runs/raw_chart_precision_gate200_20261031`; full small proofs `/efs/prescottsun/SPT-JEPA/reports/raw-chart-precision-gate200-20261031`. Dispatcher1042886 exited after both endpoints completed and GPU0 was empty; HOLD is set. Every old HOLD hash/mtime and the original diagnostic100 complete-state SHA remain unchanged. No planning, second seed, Cube or automatic continuation was executed by this gate.
