# Fixed local versus global carrier, 600-update gate

The completed CPU readout control motivates one bounded native training test. On identical Cube frames, 8×8 addressed cells with a shared RP16 gave RBF position errors **6.65/6.74 cm**, versus dense RP1024 **7.88/8.05 cm**. Pose recovery was not established. Labels were diagnostic only; none enter this model or its calibration.

`SpatialCarrierLeWM` inherits the unchanged Robust temporal-forecast objective and predictor. It replaces only observation encoding with a frozen, randomly initialized ViT measurement. Both arms use 64 slots × 16 channels and exactly the same learned tensors at initialization:

- `local`: average each spatial 2×2 patch block, retain its 8×8 address, apply one shared fixed 192×16 Gaussian projection.
- `dense`: flatten all 16×16×192 patch values and apply one fixed 49152×1024 Gaussian projection.

The projections exactly reproduce the prior CPU probe, including private NumPy seeds and float32 construction. No chart, auxiliary head, privileged state label or action imitation target is added. Encoder and unused QueryPool parameters are frozen; predictor, action encoder and renderer retain native training. The inherited rate remains beta=.25 and sigma=.1. Its dimension, learned capacity, computation and planner state differ from the old64-D baseline; **local versus dense is the matched learned-capacity comparison**. Fixed projection storage differs: 12 KiB versus 192 MiB, plus shared1024-coordinate mean/scale buffers.

Four joint calibration jobs reuse the exact original128 initial training frames per task/seed, checking old dataset/config/encoder/episode/row hashes. Each job runs one shared encoder pass and produces both arms' per-coordinate population means and scales, with the declared1e-6 numerical floor. No validation frame or diagnostic label is read. Calibration runs FP32/highest; training retains the native high/TF32 protocol, and diagnostics retain their native evaluation precision. Thus mathematical fixedness does not imply bitwise train/eval carrier equality.

Run only the frozen preparation helper with `--source-record <descriptor>`:

1. `prepare.py calibration`: creates four joint jobs on dev15 GPUs0–3.
2. `prepare.py preflight`: after all calibrations exit0, creates eight real native2-update runs at **batch128**, followed by eight fullstate/gradient/paired-initial audits. Slots enumerate task, seed, local/dense on dev15 GPUs0–7. Inspect actual peak memory. Do not silently reduce batch size.
3. Write `local_carrier_preflight_20261031/review.json` with `passed: true` only after review. `prepare.py training` then creates eight separate fresh600 runs, checkpoints200/400/600, native health/information/causal-transition diagnostics and the original strict50-case planner evaluation. No automatic extension follows.

The native Trainer, optimizer/scheduler, data sampler, full-state checkpoint and planner remain in charge. Preflight compares all initial encoder/pool/predictor/action-encoder/renderer tensors across the matched pair; frozen projection/statistics buffers must survive checkpointing unchanged. Expected gradients are frozen encoder/pool, active predictor/renderer, and action encoder required active at the two-update preflight.

Four focused CPU contracts passed in3.423s with CUDA disabled: paired parameters/global RNG, exact projection/addressing, checkpoint/normalization, and native objective backward/frozen parameters. `cpu-contracts.log` records the transcript. Real GPU batch128 memory, throughput, training and planning remain unverified until their gates complete. The approximately1249-step warmup also limits what a600-step result can establish.
