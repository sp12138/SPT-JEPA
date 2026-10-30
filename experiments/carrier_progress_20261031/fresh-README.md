# TwoRoom fresh validation confirmation

Prepared once on CPU using the pinned CLEAR generator and seed **2026103101**.
No evaluator or GPU task was launched by this preparation. Parent scheduling
supplies the new frozen source, hosts, GPU slots and inherited EGL environment.
The seven fresh evaluations are merged with the twelve metric diagnostics in
one `/efs/prescottsun/SPT-JEPA/runs/carrier_diagnostics_20261031` dispatch plan;
its exit records are authoritative. Fresh evaluation outputs remain under the
separate fresh root below. No competing dispatcher is created here.

Authoritative root: `/efs/prescottsun/SPT-JEPA/runs/carrier_freshcases_20261031`.
Manifest: `fresh-manifest.json`.
SHA256: `a709b2a94021d600d4c1f187882bcbbbfca7bf25004a50691bff3c8a65f7a75a`.

The generator has no exclude-pairs argument. The helper wraps its sampling
boundary within one process: after every original strict geometry/source-window
filter, remove the primary 50 start rows, then call the original episode-balanced
`sample_rows` unchanged. Fixed goal offset 25 makes this exactly exclusion of the
old `(episode, start, goal)` pairs. The temporary wrappers are restored afterward;
no pinned library or shared input file is edited.

- Eligible candidates: 498 before exclusion, 448 after.
- Eligible episodes: 69 before exclusion, 59 after.
- Selected: 50 distinct pairs in 50 distinct validation episodes.
- Exact old-pair overlap: 0. Episode overlap with primary 50: **34**.
- No training episodes selected. The campaign test split file was not read and
  no test cases were selected/evaluated. The unchanged pinned generator still
  computes validity from dataset-wide metadata and source transitions.
- All native strict cross-room, clear endpoints, clean source-window and
  non-initial-success checks pass. The protocol dictionary equals the original
  primary strict-validation protocol exactly.

This is new-case confirmation within the same validation episode pool, **not an
independent test set**, new training-seed replication, or a novelty claim.
No case was selected using model performance. Policy seed follows the native
generator convention and is also 2026103101, shared by all seven evaluations;
comparisons with the original primary sample therefore change both cases and
planner random draws. Planner architecture, budget, scoring and dataset
normalization conventions remain the existing native evaluator's settings.

`fresh-candidates.json` records every admissible episode/start-row/goal-row,
removed row, counts and canonical candidate-array hashes. `fresh-selection.json`
records exact selected pairs, overlap IDs, input/helper hashes and CPU checks.
Native manifest statistics before the sampling boundary remain intact, with
the post-exclusion counts explicitly recorded under `fresh_selection`.

`fresh-jobs.json` contains these seven fully specified native evaluator commands
and verified checkpoint/config hashes, but intentionally no machine assignment:

| Model | Checkpoint relative to `/efs/prescottsun/SPT-JEPA` |
|---|---|
| F identity seed1 | `runs/ideas_validate600_20261031/tworoom-carrier-identity-seed1/model-step0000600` |
| F identity seed2 | `runs/ideas_validate600_20261031/tworoom-carrier-identity-seed2/model-step0000600` |
| F bounded seed1 | `runs/ideas_validate600_20261031/tworoom-carrier-bounded-seed1/model-step0000600` |
| F bounded seed2 | `runs/ideas_validate600_20261031/tworoom-carrier-bounded-seed2/model-step0000600` |
| TF seed1 | `runs/mechanism_rate600_20261030/tworoom-fresh-rate-on-seed1/model-step0000600` |
| TF seed2 | `runs/mechanism_rate600_20261030/tworoom-fresh-rate-on-seed2/model-step0000600` |
| Released LeWM | `weights/hardtasks/tworoom` |

The six internal checkpoint exports and stages are verified at 600 updates.
Released LeWM remains a reference with different training exposure/capacity and
its existing normalization, not a matched-budget baseline. Prepared commands
all use `as_jepa_tools.benchmark_evaluate` and the same single fresh manifest.
Expected shared source record:
`/efs/prescottsun/SPT-JEPA/runs/carrier_progress_20261031/source-record.json`.

Do not rerun the generator to seek favorable cases. The helper refuses to replace
existing preparation artifacts. Subsequent checks can inspect the saved files
and hashes without repeating the dataset-wide generation pass.
