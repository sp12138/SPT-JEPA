# Raw-public chart1060: four600 gates failed numerically

All four training processes exited1 before600. No downstream fullstate/health/transition/planning endpoint ran; all16 remain blocked, and there is no600 task result. The failures occur after the FP64 internal solve, when the returned public FP32 value is checked against the unchanged original residual tolerance. Actual captured operands are required to distinguish stopping-margin and representability causes; this record does not assume either.

| Task / seed | Failed native step | Failed public certificate | Last complete checkpoint |
|---|---:|---|---:|
| TwoRoom /1 |574|Adjoint|400|
| TwoRoom /2 |553|Adjoint|400|
| Cube /1 |452|Forward inverse|400|
| Cube /2 |456|Adjoint|400|

Every last.ckpt400 passes CPU strict native model, optimizer and scheduler restoration; model/portable equality, saved RNG, cursor51200, deterministic next sampler sequence and all210 frozen tensors pass. Portable200/400 exports exist but are model-only: this does not assert two independently archived complete states. All557 initial tensors and identity553 equality were already verified at startup. Finite failed-status loss/flow/rate values do not establish a successful backward; the failed statuses correctly lack gradient metrics.

After confirming all owned processes idle and GPUs empty, dispatcher1043066 and monitor1043067 were closed and HOLD set. Four actual failed exits and16 blocked endpoints remain unchanged, as do all historical HOLD hashes/mtimes. Model1060, preparation814793e and control-plane wrapper22e02 are preserved. A separately authorized single TwoRoom seed1 diagnostic resumes its exact400 complete state with unchanged mathematics to capture the original574 public-rounding failure; it is not a continuation of these held gates or a task-performance evaluation.
