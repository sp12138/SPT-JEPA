# Raw-interface chart: production preflight

All eight native endpoints completed with exit0; all four actual fullstate reviews passed. Commit `0c9ae74840cf28247ae74341d2656e2d9ddd457c`, immutable source `B/runs/local_raw_chart_source_20261031/source-v1`; run root `B/runs/local_raw_chart_preflight_20261031`. Native batch128, original data/calibration/LR schedule, two updates only. No planning/learning-performance conclusion follows.

| Task / seed | Peak GiB | Native seconds/step | Final held-out chart gradient norm | Residual Lipschitz upper bound |
|---|---:|---:|---:|---:|
| TwoRoom / 1 | 107.4276 | 10.889 | 6.007e-6 | 2.253e-7 |
| TwoRoom / 2 | 107.4276 | 11.360 | 7.247e-6 | 2.539e-7 |
| Cube / 1 | 107.4288 | 15.447 | 3.602e-6 | 2.217e-7 |
| Cube / 2 | 107.4288 | 15.180 | 2.766e-6 | 2.041e-7 |

Every run matched all553 original initial tensors, added exactly four shared chart tensors/2,128 trainable parameters, preserved all210 frozen tensors, and matched339 unique saved optimizer parameter IDs to339 trainable tensors. Native/portable model, optimizer, scheduler, RNG and sample cursor checks passed; all declared gradient groups passed. The native learning rates were8.006405e-8 and1.601281e-7. These two-step timings are not steady-state throughput comparisons.

Both actual GPU endpoints executed the unchanged fail-closed inverse/adjoint residual checks. The additional CPU check of each actual final chart had zero measured roundtrip error/residual and finite gradients. The charts remain almost identity after two scheduled updates; these observations do not validate mature nonlinear GPU inverses or useful control. The independent nonlinear CPU oracle tests remain the nontrivial derivative evidence. A later inverse failure must stop, not relax tolerance.

dev13 GPUs0–3 were verified empty after completion; owned dispatcher1040280 was closed and this root held. Old hold hashes/timestamps stayed unchanged. Exact training PIDs were1040283/1040287/1040293/1040297; fullstate/backward PIDs1040485/1040489/1040497/1040501. No monitor was needed for this approximately two-minute preflight.

`B/runs/local_raw_chart600_20261031/candidate-plan.json` contains the four-run20-endpoint600 gate; no `plan.json` was created and no600 job launched. Root review is required before activation. Cached states already use raw coordinates and must not be inverse-transformed again.
