# Candidate technical preflight — no activation

Eight fresh runs, each **two** native updates at batch128, unchanged124910-step scheduler and original native AdamW/LRs/weight decay/gradient norm clipping at1. Model/training source stays immutable `f2ae93d4f0da2fb32021bfade7426b11ea06602b`; separate audit/observation helpers will be committed/frozen before use. The candidate preparation creates no active plan. No600 update is authorized by this protocol.

| Order | Task/seed/arm | Slot |
|---:|---|---|
|1|TwoRoom1 fixedQ|dev2:0|
|2|TwoRoom1 adaptiveQ|dev2:1|
|3|TwoRoom2 fixedQ|dev2:2|
|4|TwoRoom2 adaptiveQ|dev13:0|
|5|Cube1 fixedQ|dev13:1|
|6|Cube1 adaptiveQ|dev13:2|
|7|Cube2 fixedQ|dev15:0|
|8|Cube2 adaptiveQ|dev15:1|

Each train2 endpoint is followed by native fullstate/gradient verification:16 real endpoints total. Each pair's fullstate endpoints depend on **both** training endpoints, permitting all555 initial tensors to be compared directly across arms. Only output path and `hybrid_query` differ within each paired config. Original128 calibration frames/statistics are reused exactly; no refit/initial-weight loading or future/physical target is added. `freeze_encoder_steps` and `calibrate_state_std` callbacks must remain disabled.

The compatibility helper calls the unchanged original preflight for full checkpoint/portable equality, finite state, optimizer/scheduler, all RNG fields, sample cursor, all shared/frozen groups and actual objective backward on the existing2-case validation selection. All five hybrid projection/mean/scalar buffers remain exact. For Q alone, it replaces the obsolete “frozen implies requires_gradFalse” criterion with a stricter declared stop-gradient contract: all9 nominally differentiable Q tensors must match initial weights exactly, all9 gradients are None, and all9 saved native AdamW parameter IDs have **no** optimizer state. IDs are mapped from the actual unchanged native optimizer group order. Adaptive Q requires all9 actual finite gradients and all9 finite correctly shaped AdamW states at update2. The pool is never made optional or excluded. Negative CPU contracts reject changed fixed tensors, even zero-valued fixed gradients, unexpected fixed state, missing adaptive state/gradient and nonfinite optimizer state.

The native trainer only logs module gradient norms every100 updates, so it cannot supply these observations at two updates. A separate two-step-only wrapper observes detached F/Q64 coordinate means, mean RMS, centered RMS and total RMS on each existing training forward. It also wraps Lightning's original `configure_gradient_clipping`, calling it exactly once and recording module/global norms before and after, with its original threshold/algorithm and return value. No optimizer/clipping implementation is copied; no LR, floor or clip value is changed. CPU roundtrip checks match all clipped gradients, parameters, return object and RNG. Extra reductions/JSON writes add preflight overhead; its throughput is not a600 training throughput estimate. No per-step trace extension to600 is authorized.

The fullstate backward already computes frozen ViT tokens. A read-only hook retains those tokens only for that audit and evaluates an **isolated copy of initial Q** at8 inputs versus1024 exact repetitions, with grad/no-grad, using production `high` FP32 matmul policy. Four Q-only calls per run, zero extra encoder calls, no new cohort/labels/fit/backward. It reports raw error, normalized error divided by the frozen Q RMS, and error/σ=.1; there is no posthoc floor or implicit acceptance tolerance. Live weights/gradients/RNG and global precision flags are verified unchanged. This audits Q mode/shape sensitivity, not complete encoder train/eval equivalence. Root reviews the measured differences and actual finite native training/clipping/fullstate evidence before any600 budget.
