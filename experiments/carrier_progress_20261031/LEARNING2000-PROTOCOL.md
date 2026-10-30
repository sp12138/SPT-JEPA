# Carrier beyond-warmup continuation

This new stage continues all12 reviewed Carrier runs from600 to2000: identity/free/bounded × TwoRoom/Cube × seeds1/2. It tests whether factual dynamics and planning progress beyond the original1249-update warmup. The600 results did not establish a common two-task winner or a necessary role for the chart bound. Existing runs, held dispatchers, sources, calibration and STOP markers remain untouched.

`prepare_learning.py --source-record SHARED_SOURCE_RECORD [--dry-run]` consumes the parent's new immutable source descriptor. Before writing it verifies all old model/training/data/evaluator Python files against frozen dd2aeaa, the shared runtime descriptor, previous successful endpoints, calibration hashes, and all12 full-state600 checkpoints. `--dry-run` performs these checks without creating the new root. Preparation never launches processes. The authoritative new root is `/efs/prescottsun/SPT-JEPA/runs/carrier_learn2000_20261031`.

Each run receives a byte-identical copy of its native600 checkpoint. Model and loss configuration, all calibration coordinates, data, batch128, seeds, optimizer, scheduler horizon124910 and1249 warmup are unchanged. The original Lightning Trainer, checkpoint callbacks and sampler own resumption. The native checkpoint restores model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and sample cursor; the CPU audit also compares the next32 indices against the uninterrupted stateless sampler. DataLoader worker RNG is not saved, but this dataset deterministically constructs windows without stochastic augmentation. This is not a claim of bitwise identical resumed and uninterrupted GPU training.

Each slot runs the same dependency chain:

1. Native `train_stage --resume --stop-step1200`.
2. CPU full-state, frozen-tensor, RNG-presence, scheduler and sampler audit; archive `checkpoint-step0001200.ckpt` byte-identically. Failure blocks the next segment.
3. Native `train_stage --resume --stop-step2000`.
4. Repeat the CPU audit and archive `checkpoint-step0002000.ckpt`.
5. Existing native health, information audit, same64-window causal-transition audit, and strict native50-case planning.

This yields12 continuations and96 executable endpoints. Portable1200/2000 models are retained; both complete native states are archived. The operational checkpoint interval changes200→1200, with a common extra resume boundary at1200. No task-selection gate occurs at1200 and no automatic continuation follows2000. The new snapshot adds preparation/analysis helpers while verified model/training behavior remains unchanged. Native high-precision-mode training and highest-precision evaluation remain as previously measured; no precision variant is introduced.

Allocation is task first, then seed, then identity/free/bounded: slots0–7 map to dev2 GPUs0–7; slots8–11 map to dev13 GPUs0–3. Thus dev2 contains all six TwoRoom runs plus Cube seed1 identity/free; dev13 contains Cube seed1 bounded plus all three Cube seed2 arms. No dev15 jobs are prepared here.

Review all seeds, common-coordinate factual rollout versus initial copy and one-step versus shifted carry, wrong-state/action effects, frozen-channel information, actual chart norm certificates, and paired native planning. A smaller loss in a changed chart is not alone a dynamics gain. More informative codes do not alone establish useful goal costs; the unconstrained chart was already invertible at600. Do not select a chart-bound explanation or long continuation from TwoRoom alone.
