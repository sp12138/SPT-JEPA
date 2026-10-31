# Exact574 public-rounding failure: stopping-margin diagnosis

The source1060 TwoRoom seed1 failure was reproduced from its genuine400 complete state at the same574 step, with exactly equal loss, flow, rate, entry, surprise and feature-standard-deviation values. The actual training process exited1 and its dispatcher was closed after GPU0 became empty. The capture retains every vector in shape[128,4,64,16] on EFS; only compact metadata/results are local.

| Frozen-operand check | Maximum error / unchanged external tolerance | Violating vectors |
|---|---:|---:|
| Original FP64 iterate after4 updates | .9966365 |0|
| Actual FP32 cast of that iterate |1.0157375|1|
| One more unchanged update, then FP32 |.1012936|0|
| Next original check point:8 updates, FP32 |.0966679|0|

The single violating vector is[22,3,47], with RHS norm3.22885e-7 and tolerance3.22885e-13. Its smallest nonzero RHS component is5.22859e-9; neither RHS nor returned value has subnormal components. An independent16×16 direct solve rounds to a valid FP32 result at .0423164 tolerance. CPU replay reproduces the exact captured unrounded iterate, and its recomputed public residual differs from the actual GPU residual by at most1.67e-22 relative to RHS. For this failure, the internal stopping point was too close to the public tolerance; FP32 representability was not the obstruction. This does not prove universal representability.

The approved repair, source6e1693345786f2c190d0191685e4b485d24b4ab8, checks internal and actual public values at the same every4-update points and continues if either fails, retaining cap64 and the exact original tolerances. It changes only the raw forward/adjoint termination; the unrounded backward solution and all legacy paths remain. Three focused and six native CPU contracts pass, including a truly unattainable subnormal case that still fails and exact equality of previously passing nonzero-chart outputs/input/parameter gradients/RNG. The actual full-shape GPU replay also passes at8 with error .0966679 tolerance and unchanged high/TF32 settings. No broader numerical reliability or task-performance claim follows.

Capture helper08be83c, original model1060; capture SHA `dbbb5d4814869e760a7b409cd10aed261c99cc63c63c2beb492c6b6637edb91c`. Full capture `/efs/prescottsun/SPT-JEPA/runs/raw_chart_public_round_capture_20261031/capture/capture.pt`. Small diagnosis `/efs/prescottsun/SPT-JEPA/reports/raw-chart-public-round-diagnosis-20261031`; repaired GPU result `/efs/prescottsun/SPT-JEPA/runs/raw_chart_public_termination_verify_20261031/replay.json`. Original failed gates and400 states remain held and unchanged. Only the separately approved one-seed400→600/fullstate/health gate may proceed; other seeds and planning remain unlaunched.
