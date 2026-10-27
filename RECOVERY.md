# Remote recovery

Persistent root: `/efs/prescottsun/SPT-JEPA`. Active campaign plan and host worker state are under `runs/insights12h_20261027/`. Authoritative development source is `code/`, branch `insights/12h-20261027`. Every training job uses an immutable source archive and records its commit, command, host, GPU and deadline in `execution.json`.

After losing a host, read the plan, worker state, per-run `status.json` and `execution.json`, and inspect actual GPUs/processes. Do not infer liveness from an old status file. On an authorized replacement host, launch the same committed host-worker module with the persistent plan after adapting the host/GPU assignments. Resume a world-model run by appending `--resume` to its command: Lightning restores the model, AdamW state, LR scheduler, global step and saved RNG state from `last.ckpt`; the campaign also records the window sampler offset. A checkpoint/resume contract is checked before long runs.

All shared inputs and the original LeWM interpreter are read-only. Extra packages are in `environments/overlay`; the exact interpreter, PYTHONPATH and package versions are in the plan/environment record. The worker writes STOP at the hard deadline so training can commit a checkpoint before exiting. A new deadline requires a new user instruction.
