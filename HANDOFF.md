# Major conclusions complete — 2026-10-28T18:57:13.226920+00:00

User scope: finish the major conclusions and necessary direct controls, notify the user, leave other work deferred.
The completed report is experiments/completion20261028/results/MAIN-CONCLUSIONS.md.
Checked delivery: 46 completed endpoints, 30 paired comparisons, eight independently confirmed collapsed checkpoints, three independent I08 pretraining/finetuning seeds and same-weight feedback controls.
Authoritative completion evidence: /efs/prescottsun/SPT-JEPA/runs/completion_20261028/MAIN-CONCLUSIONS-COMPLETE.json
Frozen reporter source: /efs/prescottsun/SPT-JEPA/runs/completion_20261028/main-report-source-v2.json
All new dispatch is held after completion. Do not resume other original grids, capacity/20-epoch extensions, extra ideas or unrelated repairs without a new user instruction.
The original full-budget protocol remains incomplete. Historical long-continuation errors and deferred jobs are preserved; do not relaunch from old handoff statements.

---

## 最新优先级，2026-10-29（北京时间）

用户要求关键 insights 优先：持续坍塌保留证据和可恢复检查点后早停；未启动长网格、额外种子及条件扩展暂缓。优先补候选共同基线、三个 epoch checkpoint 生成/规划、I08 匹配复现。完整原方案仍未完成，暂停项不是成功或失败。实际调度见 EFS 的 early-stops/、scheduling/insight-priority-20261028.json 及 plan.json。数值及图见 experiments/completion20261028/results/EARLY-STOP-TRIAGE.md。

# Active completion campaign — 2026-10-28

Latest authorization: complete outstanding original experiments and validate the ten candidates; dev2/dev13/dev15 have no time limit. Read [current recovery](RECOVERY_RESEARCH_20261028.md), then the authoritative EFS plan and actual processes. Historical completed/stopped statements below describe the prior12-hour campaign only.

The new campaign is `/efs/prescottsun/SPT-JEPA/runs/completion_20261028`, branch `research/ideas-20261028`. Per-host detached dispatchers survived a transient local SSH proxy outage. New seed replications, matched method controls and full-data runs are active. Sources/configs already launched remain immutable. Do not relaunch old workers or interpret old statuses as live processes.

Validated additions: full Lance/JPEG cohort exported with all five raw-start phases; original-size L2/L0 GPU training smokes exited0;19 CPU contracts passed at source35d069d; exactNative25000DCP weights forS0/S1 exported through torchDCP. The semantic carry GPU retry also exited0. Completion is still pending, including final seeds, formal gallery review, final full-budget selection and candidate evidence.

Additional completed measurements: tinyL2seed0/50k FVD278.21 versus copy1628.62 under the explicit256case16-frame protocol; sampledFVD286.18/282.16/278.33/289.32. I10 geometric planner tests on eval seeds142/143/144 strongly underperformGoalMSE; preserve this negative result. Neither observation replaces final matched comparisons or the three-training-seed gate.

The following entries are historical.

---

# Completed insight campaign — 2026-10-27T20:56:14.635676+00:00

Current research preparation is on research/ideas-20261028: read [research recovery](RECOVERY_RESEARCH_20261028.md). Ten candidates and CPU mathematical checks are prepared; scope/budget replies are pending and no new method training has started.

Original full-spec completion was audited separately: [remaining tasks](experiments/insights12h/TASK_AUDIT_20261028.md). The original full spec is incomplete; the 28 selected endpoint finalizers are complete.

The selected 28 main finalizers and all late diagnostics are complete. Read
[the final report](experiments/insights12h/results/INSIGHTS.md) and
[the numeric/figure index](experiments/insights12h/results/README.md).
Final verification: complete=true, 28/28 models, no pending entries or errors.

Key result: tiny L2 beta=.01, sigma=.1, one-time init std=.1, 50k, seed0:
noisy agent/block R2=.868/.971; h7 agent/moving-block error over copy=.234/.234;
four sampled latent rollouts=.265/.261; P0 seeds42/43/44=74/76/60%, mean70%.
One training seed; original three-seed and manual32-case gallery gate are incomplete.
Same 50k-run 24k->50k improves agent .860->.234 and P0 24->70.
All numerical observations and key scientific PNG/SVG figures are packaged in Git.

Authoritative assets: /efs/prescottsun/SPT-JEPA.
Final review: runs/insights12h_20261027/review/final_2100.
Report source: committed 0533159, source-report-v33; main/diagnostic/verifier bytes
are identical to v32. Training/config snapshots were not changed.
All 903 inherited runtime files remain unchanged; nine CPU contracts passed.

Campaign-owned idle workers/evaluator on dev2/dev13/dev15 were terminated after
completion. Final live GPU-process queries were empty on all three hosts.
Lost-host historical statuses are preserved and may still say Running; they are
not evidence of live work. Do not restart held or historical jobs.
This does not authorize new training or inherited SPT/FVD work.

Publication destination is https://github.com/sp12138/SPT-JEPA.git,
branch insights/12h-20261027. Remote push verification is recorded outside Git
in /efs/prescottsun/SPT-JEPA/reports/publication.json.
The following entries are historical and superseded by this completed report.

---

# SPT-JEPA insight campaign, 2026-10-27

Active recovery/status audit: 2026-10-27T16:43:52.903636+00:00. Read RECOVERY_ACTIVE_20261028.md before acting; it supersedes earlier provisional results below.

Latest checked status: 2026-10-27T13:33:10.378322+00:00. Read experiments/insights12h/INTERIM_20261027.md and EVALUATION_PROTOCOL.md for current results and limitations. Local review clone tracks insights/12h-20261027; authoritative runtime remains on EFS.

Current authoritative project root is `/efs/prescottsun/SPT-JEPA`, including code and Git metadata. The user authorized dev2, dev13 and dev15 (8 H200 each) for a 12-hour insight-first campaign starting 2026-10-27 10:24:58 UTC, ending 22:24:58 UTC. Single-seed screens and early stopping are authorized. dev15 was added at 12:04 UTC; the original deadline is unchanged. No other hosts are authorized by this campaign.

Read `runs/insights12h_20261027/plan.json`, the per-host worker JSONs, `execution.json`, `source_manifest*.json`, and actual processes before operating. Every running experiment has an immutable source archive and explicit config. Source development branch: `insights/12h-20261027`, based on Kong's `9e81418`. All 903 inherited runtime hashes remain unchanged.

Data is the existing read-only PushT expert HDF5, exported under this project's `data/pusht`. Export validation checks 1000 random original/resized frames and disjoint train/test episodes. Insight runs use one frameskip-aligned phase, rather than all five raw-frame phases, and a small renderer (192 wide, 6 blocks). Main AS-JEPA LeWM model: 21,580,398 parameters; Native S2: 10,589,072; small SPT baselines: about 9.12M. These are exploratory settings, not a complete reproduction of the original experiment budget.

At 11:28 UTC dev13 stopped updating training/worker heartbeats and became unreachable by SSH. Treat its process state as unknown, not Running. At 12:04 UTC the user supplied a replacement dev13 (prescottsun-dev13-2d-1027-worker-0) and dev15 (prescottsun-dev15-2d-1027-worker-0), both verified to have eight idle H200s. dev2 remains operational. All entries for the retired node are held; recovery outputs use new directories. The streaming evaluation service v3 is on dev15 GPU7. It uses pose v1; final quality batches explicitly use the separately validated pose-v2. Saved checkpoints and all data survive on EFS.

`cross_host_recovery_PASS.json` demonstrates exact FP32 CPU recovery: 5 steps on dev13 plus 15 on dev2 matches a 20-step uninterrupted reference, including all episode/start IDs and every parameter (max difference 0). The current training entry explicitly restores trusted full Lightning checkpoints; older source snapshots need that restore fix. The GPU SDPA trajectory is nondeterministic and is not claimed bitwise identical.

Initial 2000-step renderings do not beat copying the last context frame. Original state std is about 0.0007 vs injected sigma0=0.1, so a channel-capacity diagnostic and one-time affine head initialization ablation were added. L1's documented BatchNorm fallback is separate from AS-JEPA. The L0 baseline finished 25,000 updates and its first official P0 planning run (50 episodes, seed 42) reports 39/50 success (78%). AS-JEPA generation/planning success is not yet established; this L0 result is a control, not an AS-JEPA result.

Run/evaluation logs and generated files are experiment state outside Git. The following inherited SPT handoff is historical baseline evidence only and does not authorize restarting its jobs.

---

# SPT handoff

Checked 2026-10-12T15:32:51.608851+00:00 on dev3 (prescottsun-dev3-2d-worker-0).

## Current task

The user requested SPT initialization from the RGB16 main pretraining algorithm
in /efs/stormyxie/reports/rgb16_scale_budget_20261010 and a real training run.
The original campaign and all other users' jobs were left untouched.
The upstream five-branch tuning/FVD and capacity experiments were not launched.

The algorithm source revision is a9d4f877104633de820860f0199df13590e30af1.
Runtime source files are byte-identical to the frozen upstream snapshot.
Only project launchers, documentation and persistent paths were adapted.
The exact upstream configuration and source hashes are in provenance/.

## Active 50k training

- Run root: /efs/prescottsun/SPT/runs/rgb16_init_20261012
- Worker PID at launch: 213460; torchrun PID: 213532
- Run source: source/runtime; source/jobs; source_commit is in plan.json
- Launch: SPT_RUN_ROOT=<run-root> /efs/stormyxie/venvs/vforge/bin/python
  -u <run-root>/source/jobs/worker.py pretrain
- Status/logs: workers/pretrain/status.json, pretrain.log, launcher.log
- Per-step records: output/train_steps.jsonl
- Full checkpoints: output/checkpoints, every 2500 updates and final 50000
- Starts from random initialization, not from the validation checkpoint.

At the last initial check, 34 updates had completed.
First loss 1.28127529; latest loss 1.29353487;
last-20 mean 1.27846552; hot median step
1.653s; peak allocated
88.615 GiB/card. All checked losses were finite.
These early steps do not establish convergence or generation quality.
This is a timestamped snapshot; inspect live status before operating.

## Validation evidence

76 targeted tests passed in 72.70s: pixel codec, flow, train, full resume and EMA.
Log: /efs/prescottsun/SPT/validation/cpu_tests.log.

Full 8-H200 real-model/batch validation:
- rgb16_smoke_20261012: updates 1-3, full 00003.dcp, EMA updates 3.
- rgb16_resume_smoke_20261012: strict full resume of that checkpoint, update 4,
  full 00004.dcp, EMA updates 4; restored weights/optimizer/trainer state.
- Evidence: validation/gpu_training_resume_PASS.json.

EMA sampling produced 8 held-out videos, each 144 frames (16 history +128
generated), 256x256, CFG 1, 50 Euler steps/state, no context mask.
Outputs: runs/rgb16_resume_smoke_20261012/samples/step00004.
Evidence: validation/sample_smoke_PASS.json.
The generated frames are still visual noise after only four updates.
This validates inference execution, not a useful model. No SPT FVD was measured.

## Persistence and Git

Code: /threed-code/prescottsun/SPT.
Git metadata: /efs/prescottsun/SPT/git/SPT.git.
Manifest copies: /efs/prescottsun/SPT/data.
All new runs, caches, media and checkpoints are under /efs/prescottsun/SPT.
The supplied Python environment and source SolarWM videos remain shared read-only
inputs under /efs/stormyxie. environment.json records package versions per run.

Git origin is git@github.com:sp12138/SPT.git, main. Initial source is committed in
the EFS Git repository. The initial GitHub push was blocked at that time.
This historical status is superseded by the publication entry below.

To resume after a lost development node, connect another 8-H200 machine, inspect
GPU/process state, and re-launch the same frozen worker with its original
SPT_RUN_ROOT. It automatically uses the latest completed full checkpoint.
Before the first 2500-step checkpoint, a lost node would restart from scratch.
Do not edit a running run's source/config/hash manifests or reuse its output
directory for a different experiment.


## Latest demo review (2026-10-13 Asia/Shanghai)

The first full checkpoint at 2500 updates was sampled with EMA weights on dev0.
The dev3 training job was not stopped or modified, and continued after the save.
Eight cases use the exact same inputs and per-case seeds as the step-4 smoke demo.
Protocol: 16 history frames + 128 generated frames, 256x256, 16fps, CFG 1,
50 Euler steps per state, no context mask. Sidecars confirm EMA updates = 2500.

All eight cases still show colored random noise in the generated region; no
recognizable scene, subject, or useful motion emerges. The generated pixels have
changed from step 4, but no useful visual generation improvement is established.
Loss is lower, which does not establish video quality. The demo does not identify
whether more training or a training/sampling diagnosis is required. No FVD was run.

Demo directory:
/efs/prescottsun/SPT/runs/rgb16_init_20261012/samples/step02500_ema_demo_20261013
Review media: review/gt_vs_prediction_8cases.mp4 and temporal_cases_1_4.jpg,
temporal_cases_5_8.jpg; protocol/case checks in review/review.json.
Durable review record: /efs/prescottsun/SPT/validation/latest_demo_2500_review.json.
Finite helpers: jobs/checkpoint_demo.py and jobs/render_demo.py.
The demo's frozen source copies and all output media remain in EFS.


## Original launch alignment audit (2026-10-13 Asia/Shanghai)

Audited the actual effective config and worker path of
/efs/stormyxie/reports/rgb16_scale_budget_20261010/jobs/worker.py pretrain.
All 903 runtime hash-manifest entries match upstream and the immutable SPT run.
The training entry is byte-identical; train/checkpoint/run/hardware/idle worker
functions are AST-identical. Effective YAML differs only in four output/data
paths. Both manifest contents have identical SHA256 hashes.
Across 3122 paired updates, sample identities and learning rates have zero
mismatches; mean absolute loss difference is 0.00006292. This demonstrates
closely matching execution, not bitwise-deterministic training.

Read-only positive control: sampled original pretrain 2500, pretrain 10000,
and pretrain 10000 + tuning 2000 EMA checkpoints on dev0 using the same native
sampler, eight raw-video cases and per-case seeds as the SPT 2500 demo.
Original 2500 is also visual noise. Original 10000 produces recognizable
people, roads and woods in the first generated state; longer rollout becomes
blurred and drifts. The tuned endpoint also produces recognizable first-state
images with imperfect longer rollout. No migration-specific training/sampling
failure was found in these checks; early noisy output alone is not evidence of
a failed transfer. This does not guarantee future SPT quality or rule out every
possible bug shared by upstream and SPT.

The upstream main evaluation compares 10k/20k/... pretraining plus 2k tuning,
not 2500 pretraining alone. SPT is currently only the requested pretrain mainline.
The full tuning campaign and FVD have not been reproduced. Its original FVD
uses a fixed pixel bank, different case identities/seeds and batch size 8;
this eight-case raw-video diagnostic uses singleton sampling and is not an
exact reproduction of that 2048-case scoring protocol.

Evidence: /efs/prescottsun/SPT/validation/alignment_20261013/alignment_audit.json.
Side-by-side media: validation/alignment_20261013/review/
checkpoint_comparison_8cases.mp4 and comparison_f31/f143_cases1_4/5_8.jpg.
Original jobs and checkpoints were read-only. The dev3 SPT training was left
running, its frozen source/config unchanged; all dev0 diagnostic GPU processes
completed and exited. Reusable check: jobs/audit_alignment.py.


## Compute and FVD instruction (2026-10-13 Asia/Shanghai)

User allows dev3 and dev8 only; dev0 must not be used. Main training remains on
dev3. The user has explicitly requested FVD on dev8 following the reference
configuration under /efs/stormyxie/reports/rgb16_scale_budget_20261010.
Use the latest completed SPT checkpoint and the original frozen FVD protocol.
This authorizes evaluation; it does not request a new tuning/training branch.


## Active FVD evaluation (snapshot 1791877148.1745105)

The user explicitly requested evaluation on dev8 with the original xie settings.
Report: /efs/prescottsun/SPT/evaluations/rgb16_init_step10000_fvd2048_20261013
Frozen helper commit: dc149e98a1bf8ec2758425868376aa5e05b32ddc
Checkpoint: /efs/prescottsun/SPT/runs/rgb16_init_20261012/output/checkpoints/10000.dcp
Phase: pretrain_only, EMA 10000; no tuning branch was launched.
Worker PID: 3519; torchrun PID: 3643 on dev8.
Status at inspection: running, 320/2048 features.
Protocol SHA256 matches upstream: 649a3e06e6028bae012db80c53f27594e5c64f3ee6966c336f9362e2aff471dd.
All 2048 reference features and input metadata were checked before launch.
The current dev3 training continues unchanged. Inspect live status/processes
before any action; this historical Running snapshot does not authorize a restart.
The final score will be at /efs/prescottsun/SPT/evaluations/rgb16_init_step10000_fvd2048_20261013/fvd/results/spt_rgb16_pretrain__ema/10000/score.json.


## Completed FVD-2048 (2026-10-13T07:51:16.544045+00:00)

The preceding Active FVD entry is finished; no evaluation remains running.
SPT pretrain step 10000 EMA FVD = 1233.57348569.
Exactly 2048 real and 2048 fake 400-dimensional I3D features were included.
The original frozen evaluator validates each case's seed, checkpoint, config,
batch size, detector and protocol before score commitment. Protocol SHA256
matches xie: 649a3e06e6028bae012db80c53f27594e5c64f3ee6966c336f9362e2aff471dd.
Evaluation ran on dev8 for 986.9 seconds and exited.
This score is pretrain-only, not an extra 2k tuning endpoint. The main dev3
training and its immutable source/config were not modified.
Score: /efs/prescottsun/SPT/evaluations/rgb16_init_step10000_fvd2048_20261013/fvd/results/spt_rgb16_pretrain__ema/10000/score.json
Validation: /efs/prescottsun/SPT/evaluations/rgb16_init_step10000_fvd2048_20261013/result_validation.json
The report contains the frozen jobs/config, plan, complete features and logs.


## Original pretrain-only FVD comparison (2026-10-13T08:17:15.716819+00:00)

The user requested xie's performance before the additional 2k tuning stage.
Read-only checkpoint: /efs/stormyxie/outputs/vforge/rgb16_scale_budget_20261010/pretrain_50k/checkpoints/10000.dcp
Frozen report: /efs/prescottsun/SPT/evaluations/xie_rgb16_pretrain_step10000_fvd2048_20261013
Original xie pretrain 10000 EMA FVD-2048 = 1111.34561644.
SPT pretrain 10000 EMA FVD-2048 = 1233.57348569.
Existing xie pretrain 10000 + tuning 2000 EMA FVD-2048 = 586.21288827.
SPT minus original pretrain = 122.22786925
(10.99819% relative to original).
Protocol, pixel bank, 2048 cases, seeds, sampling batch and detector are identical.
Protocol SHA256: 649a3e06e6028bae012db80c53f27594e5c64f3ee6966c336f9362e2aff471dd
The original pretrain-only evaluation ran on dev8 for 712.8
seconds and exited; the GPUs were empty at final verification. No training or
tuning job was started, and the existing dev3 main training was not modified.
Results: /efs/prescottsun/SPT/evaluations/xie_rgb16_pretrain_step10000_fvd2048_20261013/comparison.json
Validation: /efs/prescottsun/SPT/evaluations/xie_rgb16_pretrain_step10000_fvd2048_20261013/result_validation.json


## Training/inference reproducibility audit (2026-10-13T08:28:54.222856+00:00)

The user requested a renewed comparison after the 11% pretrain-only FVD gap.
Evidence: /efs/prescottsun/SPT/validation/reproducibility_20261013/summary.json, reproducibility_audit.json,
checkpoint_weight_comparison.json and training_alignment.json.
All 903 runtime manifest entries and the complete 840-file Python tree match.
Training/evaluation configs differ only in paths; declared worker environment
changes are cache directories. Training uses seed 42, BF16 and FA4 with
runtime.deterministic=false in both experiments. GPU models/topology match.
Across the complete first 10000 updates, rank-0 sample IDs, learning rates,
EMA counters/decay and target horizon fractions have zero mismatches. First loss
is identical. Mean absolute loss difference = 8.588274159e-05.
Last-100 mean losses: original 0.02970593543,
SPT 0.02970184403.
At all four 2500/5000/7500/10000 boundaries and all eight ranks, saved Python,
NumPy, CPU/CUDA RNGs match; loader states match after manifest path normalization,
and scheduler, semantic configuration and execution policy match.
All 2048 FVD pairs have identical inputs/seeds/parameters/detectors/protocol;
strict checkpoint source and EMA 10000 metadata are verified per case.
The 897 saved checkpoint tensors were read with the standard CPU DCP loader.
EMA metadata/parameter names match; learned online and EMA weights differ
(relative L2 0.565387 and 0.624269).
No implementation or configuration discrepancy was found. Same RNG does not
ensure bitwise-identical non-deterministic FA4 backward/reductions; numerical
trajectory divergence is a plausible explanation, not proven causation for the
full 11% FVD gap. Historical upstream inherited environment/driver/package
snapshots and initial full model tensors are not completely archived. No new
training or tuning was started; the existing dev3 training remains unchanged.
Audit helpers are committed under jobs/; frozen copies are at
/efs/prescottsun/SPT/validation/reproducibility_20261013_source.


## Self-contained repository publication (2026-10-13)

The user explicitly requested that the whole repository be self-contained.
Code commit b25909af12e72e24dafa5407288993e36abcb111 was pushed successfully
to git@github.com:sp12138/SPT.git main from the Mac. It includes 1083 files.
Runtime source is unchanged. Training, inference, FVD detector/evaluator,
preparation, demo/rendering and audit helpers all reside in the repository.
Active jobs no longer import source from /efs/stormyxie or a separate checkout.
Original FVD sources remain under provenance/fvd_reference as historical evidence.
Asset paths are arguments; data, weights and artifacts remain outside Git on EFS.
Environment requirements and recorded package versions are included. An identical
fresh package environment was not rebuilt; the validated EFS interpreter was used.

Validation ran on dev8 from a clean clone. Six portability checks passed, covering
all 903 runtime hashes, model/training configuration semantics, repo-owned
train/eval imports, detector math, absence of upstream job-code paths, and freezing
a training run with explicit asset paths. FVD preparation checked all 2048 real
bank features. A 16-case GPU smoke loaded the 10k checkpoint and EMA strictly and
generated all features, then exited with GPUs empty. It used sampling batch 2,
so it validates execution and does not establish a new FVD score. Portable
evaluator hashes change when source paths change; new reports record their actual
hash and the historical reference hash separately.

Evidence: /efs/prescottsun/SPT/validation/self_contained_20261013/verification.json.
The development code and EFS Git repository were synchronized to this commit.
The existing dev3 main run's source/config/plan and training were not modified.
Local Mac workspace is now a Git checkout; artifacts remain ignored.
