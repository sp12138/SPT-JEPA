# Predeclared fresh Cube validation

One CPU generation pass completed in 27.9 seconds. The independent
`fresh_cube_prepare.py` preserves the TwoRoom helper and its artifacts unchanged.
Sampling and native policy seed are both fixed at **2026103101**, before any
fresh Cube evaluation results. There was no outcome-based filtering or resampling.

Authoritative root:
`/efs/prescottsun/SPT-JEPA/runs/carrier_cube_freshcases_20261031`.
Its `fresh-manifest.json` SHA256 is
`9794cd32020fde94a452f1fe83c387eb86ea622b421be593c10bb9dd64c8e78f`.
Local copies use `fresh-cube-*.json` to avoid replacing TwoRoom files.

- Original native eligible candidate rows: 56,478 across 500 episodes.
- Excluded exactly 50 primary pairs before the unchanged episode-balanced sampler.
- Remaining candidates: 56,428 across 500 episodes.
- Selected: 50 unique pairs in 50 validation episodes.
- Exact pair overlap with primary: **0**; episode overlap: **1**.
- No train episodes or campaign test cases selected. The test split file was
  not read. Native dataset-wide metadata/valid-pair calculations remain intact.
- Strict protocol dictionary, task scoring, success exclusions and sampling
  mode equal the original primary manifest. No extra task-specific filter was added.

The candidate file preserves every admissible start row, the fixed goal-row
offset 25, excluded rows, episode IDs and hashes. Frozen dataset metadata maps
each row to its episode. The selection file records exact chosen triplets,
overlap IDs, all source/input hashes and passed CPU checks.

Three prepared baseline jobs use the unchanged native evaluator:

| Model | Checkpoint relative to `/efs/prescottsun/SPT-JEPA` |
|---|---|
| TF seed1 | `runs/mechanism_rate600_20261030/cube-fresh-rate-on-seed1/model-step0000600` |
| TF seed2 | `runs/mechanism_rate600_20261030/cube-fresh-rate-on-seed2/model-step0000600` |
| Released LeWM | `weights/hardtasks/cube` |

Checkpoint paths, export classes, sizes and full weight/config hashes are verified
in `fresh-jobs.json`. Parent preparation adds twelve old-600-checkpoint carrier
metric evaluations to this identical manifest: identity/bounded × two training
seeds × three predeclared metrics. The resulting fifteen evaluations are queued
in the existing single `carrier_diagnostics_20261031` dispatcher. This helper
creates neither workers nor an additional dispatch plan and launches no GPU job.

Shared source record is
`/efs/prescottsun/SPT-JEPA/runs/carrier_progress_20261031/source-record.json`.
The parent supplies frozen source, host/GPU and inherited evaluator environment.

This is a confirmation sample from the existing validation pool, not an
independent test set or new training-seed robustness evidence. Although only one
episode overlaps the primary fifty, the validation split and task distribution
are shared. Released LeWM retains its different exposure/capacity and native
normalization. Planner random draws change with the new seed, identically across
all fifteen comparisons; planner budget and mechanics remain fixed. Old manifests,
checkpoints and results are preserved. Do not regenerate to seek favorable cases.
