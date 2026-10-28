# SPT-JEPA current execution policy

User instructions, 2026-10-29 (Beijing):
- Prioritize key insights. Preserve and early-stop persistent collapse. Defer exhaustive grids and low-value extra seeds; complete decisive controls, existing checkpoint quality/planning and promising I08 replications first. Deferred original experiments remain unfinished.

User instructions, 2026-10-28:
- Continue unfinished original experiments and validate the ten candidate methods. dev2/dev13/dev15 have no campaign wall-clock limit; the previous12-hour deadline is lifted for new work. Publish useful additional numerical results and insightful figures. Preserve all historical run snapshots.

User instructions, 2026-10-27:
- Authoritative code, environments, data, logs, checkpoints, reports and recovery instructions live under /efs/prescottsun/SPT-JEPA. This overrides inherited SPT and parent workspace locations.
- Authorized compute: dev2, dev13 and dev15, 8 H200 each (dev15 added by the user at 2026-10-27 12:04 UTC). Other compute workloads on these hosts may be cleared per user instruction; preserve system services.
- This campaign is insight-first within 12 hours. Use single seeds, common held-out cases, early collapse gates and focused ablations; do not require all three-seed grids. Record every departure from the original spec and distinguish exploratory evidence from final benchmark claims.
- Preserve immutable committed source/config snapshots for running experiments and periodic resumable checkpoints. Never modify other projects or shared inputs.
- Local storage is limited to small Git-managed code and presentation artifacts.

## Inherited baseline constraints

# SPT project entry

SPT starts from the frozen RGB16 pretraining implementation in
/efs/stormyxie/reports/rgb16_scale_budget_20261010. Read README.md and HANDOFF.md
before operating the project. The guide in runtime/AGENTS.md governs framework code.
All project runtime, inference and FVD source is bundled in this repository.
Upstream paths in provenance/HANDOFF are historical evidence, not code dependencies.
Use explicit asset paths and the repository-owned preparation/worker entry points.

Code is managed at /threed-code/prescottsun/SPT. Git metadata is separately stored
at /efs/prescottsun/SPT/git/SPT.git. Put manifests, outputs, logs, caches and
checkpoints under /efs/prescottsun/SPT. Shared upstream data and the supplied
Python environment are read-only inputs; never change another user's experiment.

Use the existing Native model/task/training/checkpoint implementation. Do not
reimplement the Trainer or checkpoint manager. Freeze a committed revision into
each EFS run before launching. Running source/config snapshots are immutable.
Check actual GPUs/processes before scheduling work and use independent output
directories. Never stop unrelated jobs or restart historical jobs from handoff text.

The initial RGB16 training was requested by the user. This does not authorize
capacity ablations or the upstream five-branch tuning/FVD campaign. Report training
loss, throughput and sample/FVD quality separately. An early smoke result cannot
establish generation quality. Keep datasets, weights, logs and media outside Git.

Compute allocation (user instruction, 2026-10-13): use only dev3 and dev8. Do not
use dev0. Keep the current main training on dev3; prefer dev8 for demos and FVD.
The user explicitly authorized FVD on dev8 using the original xie protocol.
