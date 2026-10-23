## First CA-before-SA training verified — 2026-09-21
Three replacement first-layer-only tasks Running on8H200 each, fresh sequential
optimizer steps from0, finite losses, local8/global64, fixed FA4/no fallback.
Active jobs: /threed-code/stormyxie/vforge_jobs/scaling_first_cross_pre_20260921T081808Z
Active report: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_before_self
Added model field cross_attention_position default after_self; these3 override before_self.
Only block0 CA→SA→FFN; subsequent SA→FFN. Training and eval snapshots share exact model files.
163 CPU tests passed plus2 full masked flow/gradient-checkpoint/inference cases;
4 optional-asset skips. Strict changed lint/264vendor hashes/import passed.
Old order tasks stopped; no midrun weight/optimizer continuation.
Stable201–300 efficiency report auto-fills when new3 reach300. Original10 unchanged.

## B300 FA4 / SDPA efficiency — completed 2026-09-20 18:40 CST

User accepted prior100-step short-term loss differences and explicitly requested B300
for short performance tests; compatible long ablations remainH200.
Production Native trainer, full221240896 latent/no-zprev/cross baseline2500weights,
8B300 DDP local4/global32clip slots, freshMuon, BF16 autocast/FP32parameters,TF32off.
Two isolated jobs: deterministic true and false; per mode same node, SDPA/FA4/FA4/SDPA,
each10warmup+30timed steps. CUDA-sync wall intervals include online codec and optimizer,
exclude startup/JIT/warmup/filevideo decode/checkpoint/demo. No source/default changes.
Parsedconfigs differonlyattention backend/output; 64rank records/1920timed intervals,
allloggedlossfinite andFA4modelSDPAfallback0. Each backend only2run repeats/mode.

Regular nondeterministic: SDPA1.478647s,FA41.001918s; throughput+47.5816%,step-time-32.2409%.
Allocated56.909679 ->57.541840GiB/GPU (+0.632160GiB,1.1108%).
Clip slots/s21.6414 ->31.9387.
Deterministic: SDPA2.789563s,FA41.117970s; throughput2.495203x (+149.5203%).
Do NOT presentdeterministic2.5x asregulartraininggain.
Source immutable ae2ca4a67edb02f2999d5822d6c4d9c8e6b23b63.
Tasks stormyxie-fa4-perf-09201024 andstormyxie-fa4-perfnd-09201032 bothsuccess/cleaned;
independent tasknotfound+zeroPods verifiedforboth. No productionbackend switch/commit/push.
Reports /efs/stormyxie/reports/fa4_sdpa_perf_20260920T101954Z and
/efs/stormyxie/reports/fa4_sdpa_perf_nondet_20260920T103202Z/comparison (HTML/MD/PNG/SVG/JSON).
Jobs /threed-code/stormyxie/vforge_jobs/fa4_perf_20260920T101954Z andfa4_perf_nondet_20260920T103202Z.
Local outputs/fa4-sdpa-performance/fa4-vs-sdpa.html.
H200 latest18:39CST: B11000/C19800/D1200/E1000/G800 allRunningfinite noattemptErrors.
D/E/GfreshH200, Bresume10k (legacy notexacttrajectory), Cuntouchedlocal; Aintentionallystopped65600.

## FA4 / SDPA 100-step trajectory — completed 2026-09-20

User requested extending numerical comparison to 100 steps each + curves/statistics.
Immutable source ae2ca4a67edb02f2999d5822d6c4d9c8e6b23b63; full221240896 model,
same2500legacy weights, fresh independent Muon states,1B300,BF16,TF32off,deterministic ops.
Fixed SAME clip/noise/timestep/dropmask as prior3step gate; all input hashes and
first3losses exactly reproduce prior deterministic gate. This is fixed-batch numerical
diagnostic/overfitting, NOT dataset convergence. Three held-out-noise probes, SAME video,
evaluated every10steps; candidate also evaluated viaSDPA to isolate weight-path effects.

All100each finished, no FA4 SDPA fallback. SDPA/FA4 final preupdate loss .010122216/.009970005
(-1.50374%); meanabsdiff .000215528, meanabsrelative .55215%, P95 2.63316%, max4.79575%.
Update relativeL2 grows17.68->76.24% (mean35.75%); finalgradient119.16%, velocity6.2185%.
Finalparameterdistance3.68398 / referencenorm417.90097=.88154%; versus cumulative
movement25.33054 =14.54362%. Updatecosine .98437->.71357.
MSE gap exactly decomposes: final linear-.00839106 + quadratic.00823885=-.000152211.
Final3probe meanloss .47359867/.47012818 (-.73279%); unifiedSDPA candidate .47018131,
probeweight-only outputrel7.89068%, sameweights kernel-only1.23450%.
Loss closeness does NOT demonstrate numerical trajectory equivalence; default remainsSDPA.
One paired trajectory, temporally correlated steps: descriptive stats only, no equivalence p-test.

Jobs /threed-code/stormyxie/vforge_jobs/fa4_100_20260920T094849Z
Report /efs/stormyxie/reports/repworld_fa4_100_20260920T094849Z (CSV/JSON/PNG/SVG/HTML;
raw perstep/probe metrics; final_sdpa.pt and final_fa4.pt weights+optimizer retained remote).
Local outputs/fa4-100steps/fa4-100steps.html and .png.
Task stormyxie-fa4-100-09200948 succeeded and cleaned; independently tasknotfound/0Pods.
No production changes, no experimental default switch, no commit/push.
User additionally asked six-ablation health: see current Rep HANDOFF and report six-case-health.json.

## RepWorld FA4 — completed as opt-in; strict equivalence FAILED (2026-09-20)

User requested B300 FA4 with strict before/after numerics. Native RepWorld now has
explicit FA4 ViT/DiT via model-owned SegmentedSparsity; default remains SDPA because
independent optimizer trajectories failed the fixed thresholds. Original Rep
experiments/frozen snapshots unchanged. Frozen codecs retain published SDPA.
Use attention.backend=flash_attention_4 runtime.deterministic=true to opt in.
Same-backend deterministic reproducibility is NOT cross-backend trajectory equivalence.

xj_dev HEAD5273178 and original index unchanged; no branch commit/push.
13 source/doc/test files modified/untracked, plus this untracked HANDOFF.
Final source snapshot ae2ca4a67edb02f2999d5822d6c4d9c8e6b23b63 in final_bundle;
GPU snapshot021f774cdb2b1e157cce394cb063cb47877d2de4. Difference only cachedUlysses
local/global length naming fix and2CPUregressions; DDP exercised path unchanged.
Repaired productionbase530a58c9f38a2b37f303d525ac6e34a02c6af8d8.
Common ops/_attention/segmented.py fixes shortQ forward128/backward256 metadata,
reclassifying padded tail. No optimizer, CP, MCore, or engine modifications.

Validation:
-189CPU (135RepWorld/metadata+54attention/consumers),4optionalassets skipped.
-11file strictlint/264MCorechecks/import/docs19page build passed; last2file changes
ruff format/check +21visibility tests pass. Source hashes/diffcheck/index verified.
-8B300 attention cases FP64oracle passed; maxFA4relL2 forward .002324, gradient .002430;
exact future perturbation invariance and zero futureK/V gradients; zero SDPAfallback.
-All8small variants completed3independentMuonsteps (RGB2/4,latent,mask80,zprev,pureTF,AC);
updates fail5% threshold. No gradient copying/weight resync/threshold relaxation.
-Real221M12x768+12x768 baseline2500weights/fixed10stateRGB; freshMuon (NOT momentum
resume). Deterministic cross-backend3steps: grad1.2417/2.8658/5.3215%; delta-weight
17.6816/18.0554/17.4007%; velocity1.0553/1.1140/1.1648%. Loss differences small.
-FA4 deterministic full221M repeat3steps: output/ALLgradient/updates BITWISE equal.
-Defaultnon-detFA4 smallrepeat: tinygradientperturbation later amplified. Captured
Muon inputs replay confirms Newton–Schulz amplification even with CPUFP32 replay.
-Correct pureTF KVappend: samebackend full vs incremental BITWISE equal underboth
SDPA/FA4; crossbackend .3421%. Prior1.8% failure was TEST HARNESS droppingoldhistory;
production sampler alreadyappended correctly. CorrectCPUFP32 controlrel3.3e-7.
-Testfixturealso fixed pureTFnoise[B,C,S-1,P,H,W]. No model semanticschanged.
-NativeDDP2 local4/global8clips72targets3steps PASS; actual SM100FA4CUDAprofile confirmed.
No throughputclaim. FA4multi-node/FSDP/USP/othercodecfullmodels/longquality unverified.

Evidence /efs/stormyxie/reports/repworld_fa4_20260920T083928Z/REPORT.md andsummary.json.
Subdirs repaired,kv_final carry authoritativefinaldata; diagnostic hasMuonorhtoreplay
andnon-detcontrols. source_verification.json explainslastCPU-onlyUlysses delta.
All5ownKAP testtasks deleted, notfound andzeroPods verified incleanup_checks.json.
Last taskstormyxie-fa4-kv-09200926 platformSucceeded,cleanuptrue; individualcrossbackend
numericgates stillFAILED. Never treat diagnostic orchestrationSucceeded asnumericalPASS.
Persistent CuTe/FA4JITcache /efs/stormyxie/cache/repworld_fa4_b300_torch291.
No taskleftactive fromthiswork. Existingablationtasksuntouched.

Decision: FA4 can be a separately tracked backend experiment, not a transparent
same-trajectory replacement for SDPA+Muon. Do not enabledefault based only on
functionalDDP success. Long-runquality/performance acceptance requires separate work.

## Backend clarification — 2026-09-20

Current xj_dev5273178: RepWorld USP is Torch-native Ulysses, NOT Megatron SP.
MegatronEngine explicitly rejects usp_size>1. H3 default recipe engine/dp_mode=fsdp2;
documented normal H3 path overrides USP4, and H3 uses same distributed.ulysses module.
RepWorld engine gate is migration support boundary. Removing it is insufficient:
RepWorld and H3 TrainComponents supply optimizer_factory but no OptimizerSpec;
MegatronEngine requires OptimizerSpec and current MCore builder constructs Adam,
not the RepWorld Muon. MCore DDP/optimizer/checkpoint adaptation needs separate validation.
No production changes or H3 training launched. CP remains untouched.
Detailed evidence: /efs/stormyxie/reports/vforge_backend_audit_20260920/REPORT.md
Distributed-training capability table's MCore optimizer-save limitation is stale vs current
MCore checkpoint implementation; do not use that old table entry as current truth.

## Rep vs vforge DDP throughput — completed 2026-09-20

User requested same-config speed comparison, then restricted scope to DP (DDP only).
dev1 was NOT idle: C spatial-add-only on all8 H200; launcher3443431 untouched, latest log12400.
Used one isolated8B300 node, task stormyxie-rep-vf-perf-09200807: platform Succeeded, runner cleanup true;
task-not-found and zero remaining Pods verified. No training/source-code changes or pushes.

ABBA two runs each; latent/no-prev/cross-only baseline,10states160RGB256px,local4/global32
clip slots(288relatedtargets), same2500weights and fixedRGB SHA f9e550a873f5398ea638793e80dbcffa99ff5fa560b1a9197ff28d79cfbad739.
FP32params/BF16AMP,SDPA,TF32off,Muon1e-4/NS5/momentum.95/auxbetas(.9,.95)/eps1e-8,clip1,
noEMA/AC/checkpoints/demos. Original optionaldirectFAshortcut disabled only in benchmark wrapper.
10warmup+30timedsteps/run, postoptimizer CUDA-sync wall intervals,slowest-rankmeans.
Includesonlinecodec,H2D,CPUcollation,training/optimizer/logging; EXCLUDES filevideo decode/startup.
Repeatedfixedinput is throughput probe, not independent dataset/quality/optimization-path comparison.

Rep1.388243/1.423500sec;mean1.405872,22.7617clip/s,peakallocated59.9877GiB/reserved62.
Native1.422110/1.391436;mean1.406773,22.7471clip/s,peakallocated56.8486GiB/reserved57.5859.
Native throughput -0.064%: effectively equal within2–2.5%runvariation;noDDPspeedbenefit demonstrated.
Native allocated peak3.139GiB(5.2%)lower. Do not generalize to H200/otherbatch/I/O/parallelisms.
ParamsRep221833024vsNative221240896: exactly592128unusedViTimage/modalityembeddingparams
removed inNative. Strictactiveweightimport andparameter_mapping_audit passed;notashrunkenactivebackbone.
All32rankrecords completed40steps,traininglossfinite. Native optimizer CUDAevents~88.97ms vsRep82.76ms.

Firststart failed before anyoptimizerstep due emptyfixed-demo path normalized bylegacyCLI;
corrected benchmarkparsed.result_fixed_video=None,rerun valid. No productionfix.
User removedFSDP/USP whilecoordinatorrunning: pendingworkers explicitlyexit77BEFOREtorch/CUDA.
Rawoptionalstageexit1 entriesareuser-skips,NOTcapabilityfailures; fourDDPrunsallpassed.
Evidence /efs/stormyxie/reports/rep_vforge_throughput_20260920T080101Z/REPORT.md,summary.json,validation.json,parameter_mapping_audit.json,rankrecords.
Scripts+frozenRep snapshot /threed-code/stormyxie/vforge_jobs/throughput_20260920T080101Z;Native immutablebundlecommit 527317858e8ab5b975fab1859a0317f53503ebdd.
Pointer /efs/stormyxie/work/rep_vforge_throughput_active.json. Six-case readonlymonitor remains active.

## Mask80 32-GPU task deleted at user request — 2026-09-20

User explicitly requested deleting the mask80 32-GPU run.
Task stormyxie-vf-mask80-r2-09200606 (4x8 B300 DDP, context_mask_ratio=.8, latent/no-zprev/local8)
was Queuing at deletion. Exact task delete succeeded, task get returns not found,
current Pods totalSize=0. Verified local controller PID2287714 was the mask80
submit_mask80_resume.py process; stopped with SIGINT and confirmed exited.
Do NOT resume/resubmit this experiment without a new explicit user request.
Both mask80 and shared preempted_resume pointers now record deleted_by_user/auto_resume=false;
shared spatial task fields are untouched. No model/source/config or other task changes.
All output/checkpoints retained, including2500/5000/7500/10000/12500/15000 fullDCP and latest.
Output: /efs/stormyxie/outputs/vforge/mask80_ddp4_20260919T155758Z
Audit: /efs/stormyxie/reports/stop_mask80_20260920T072117Z
Verified UTC: 2026-09-20T07:22:42.883016+00:00

## 2026-09-20 restart submitted / vforge xj_dev published

User explicitly authorized restarting preempted runs and pushing vforge to its new branch (not master).
Restart assets: /threed-code/stormyxie/rep_miniworld_jobs/preempted_resume_20260920T060408Z; audit: /efs/stormyxie/reports/preempted_resume_20260920T060408Z.
Old spatial resubmitted as stormyxie-lvit-pos-r2-09200604, 1x8 B300, unchanged frozen config/code.
Loaded 5000 model and encoder with all keys matched; preempted again14:15:01, now Queuing.
No newly completed optimizer interval observed. Keep queued task; do not submit duplicate.
Old Failed task has zero workers; residual is still Running8200, do not duplicate it.
New pure spatial-only dev1 run healthy6000, do not interrupt.
Mask80 resubmitted as stormyxie-vf-mask80-r2-09200606, same4x8B300/batch8/bundle_v2/output.
New external mask80_resume_worker.py uses native latest-complete resolver on every boot,
requires full state>=15000, reuses original successful gate, uses explicit selected checkpoint,
restores weights/optimizer/training_state; creates unique per-node attempt logs. 8 launcher tests pass.
Composed config differs ONLY resume.path from original; initial4nodes selected15000.
Platform preempted it AGAIN at14:08:42 before first step; now Queuing, controller2287714 waits.
Do not resubmit duplicate while this task exists; no train/core/model code changed for restart.
Rep legacy resume does not restore full data cursor/RNG; don't claim bitwise continuation.

Vforge publication COMPLETE at2026-09-20 14:20 CST.
User supplied author stormyxie <stormyxie@tencent.com>, configured ONLY repo-local.
Committed61files as 527317858e8ab5b975fab1859a0317f53503ebdd; pushed new origin/xj_dev, upstream set.
Remote ls-remote confirms exact SHA; master remains fe8a51affcd0def8b63d256f649d7a4ee726a254.
Branch is based on1791f31, intentionally publishes the previously validated implementation.
Remote master has9 newer commits not integrated; no MR or merge created.
109 CPU passed/4assetskips/27GPUdeselected; strict lint,264mcore hashes,import smoke,19page docs allpass.
Staged/worktree61-file hashes rechecked against validated manifest before commit.
No distributed/training/mcore changes; HANDOFF and experiment launchers/artifacts not published.
Only untracked HANDOFF.md remains in original vforge worktree; index clean.
Tunnel uses user local git-tunnel.sh dev1 with id_ed25519, detached persistent invocation;
remote /tmp/git-ssh-config and /tmp/git-agent.sock, explicit GIT_SSH_COMMAND.
Evidence: audit/push_verified.json, push_result.txt, committed.json and push_* checks.
No pending commit/push work remains.

## 2026-09-20 stop/resume and CFG audit — complete

Authoritative current state supersedes older running-status notes below.
RGB patch4 task stormyxie-rgbt4-eupe-09181015 and stormyxie-rgbt4-vdrae-09181015 deleted;
exact task-not-found and zero Pods verified, all outputs retained.
DuET local launcher1816263 and watcher process group1816264 stopped; all8 GPU workers gone.
Graceful final latest+archive checkpoint175000 both readable, optimizer present.

Two latent ablations resumed from latest durable2500, not unsaved log3200:
stormyxie-lvit-res-resume-09200246 / stormyxie-lvit-pos-resume-09200246, each1x8B300.
Frozen original runtime byte copy; only allow_existing/resume enabled, auto-resume future latest.
Models+encoder all keys match, optimizer restored; both reached2900, cleanres new3000demo saved.
Old code does not save complete data cursor/RNG; NOT bitwise data continuation.
Original configs and old3000demos backed up before resume. Same fixed input content hash verified.
Mask80 stormyxie-vf-mask80-09191612 stays running4x8B300; step15724 at11:03CST.
Three active trainings48B300. No dev1 GPU process after audit sampling finished.

CFG audit report: /efs/stormyxie/reports/cfg_paths_20260920T025208Z/REPORT.md
Actions: /efs/stormyxie/reports/latent_ablations_resume_20260920T024240Z/REPORT.md
Original source vs native RGBpatch2/latent real2500checkpoints, CFG2/50Euler/10states160RGB,
fixed seed1234 and SDPA: full latent+RGB BITWISE equal in both tests (2 passed).
All three use noise1 -> data0; Rep clean-noise vs DuET noise-clean with matching update sign.
Rep/native BF16 subtraction before FP32 CFG promotion differs from DuET float-before-diff.
112 real branch calls: mean relative arithmetic error .110%; inactive scale1 roundtrip .0786%.
Isolated15k mask80 2cases CFG1/originalCFG2/FP32CFG2; all finite,18 null-independence exact
checks,900 paired calls. Same isolated RNG seeds/shapes/counts; actual noise hashes NOT recorded.
State10 FP32-vs-original latent relative RMS11.51/21.92%, but latent total RMS nearly equal
and contacts retain similar late geometric drift. Do NOT claim precision fixes major drift.
No production sampler/model/training code changed; only temporary diagnostic monkeypatch.
DuET native generate has no CFG; earlier CFG2 used explicit hook, zero prefix before normalization,
float-before-diff,20steps all guided. Existing early CFG2 also showed artifacts;175k CFG2 not sampled.
Keep CP/core/mcore unchanged. Do not restart stopped tasks without new authorization.

## RepWorld latent mask80 / 4-node DDP — RUNNING (2026-09-20 CST)

Latest task: stormyxie-vf-mask80-09191612, platform Running at 2026-09-20T00:18:51+08:00.
Formal training completed52steps, loss1.899909, finite gradients,
rank0 peak82.89GiB. Four distinct physical nodes/32GPUUUIDs; actual NCCL EFA-direct.
Gate3steps+fullDCP passed, all4nodesexit0; formal fresh training then passed20step
startup check. Continue this job; do not submit duplicates or clean it early.

User configuration:4x8B300 pureDDP, VideoRAE3D latentViT, no_zprev, mask80,
localbatch8/global256 independentclips/2304correlatedtargets,10states160RGB256px.
Original12x768ViT+12x768World, Muon1e-4, FP32params/BF16autocast, noAC,
seed42, accum1, fresh start. Checkpoint every2500, keep history; max100000steps,
KAP deadline7days. ControllerPID1912033 ondev1; normal completion cleans KAP.
Original8DuETprocesses1816301..1816308 preserved and checked running.

Jobs:/threed-code/stormyxie/vforge_jobs/mask80_ddp4_20260919T155758Z
Outputs:/efs/stormyxie/outputs/vforge/mask80_ddp4_20260919T155758Z
Read REPORT.md, submitted.json, snapshot.json, startup_validation.json,
final_status.json, train/resolved_config.yaml and logs/train-node0.log.
Model/task23files match immutable bundle_v2 revision cedd2a2473605997293e13a3d918c5610d59c046.
Isolated snapshot only; originalHEAD/index unchanged, no push.

Implemented conditioner.context_mask_ratio default0: DuET independent spatial
sampling perclip/state/plane, floor256*.2=51visible/plane,204/state beforeViT,
original3DRoPE/statecausality; DiT reads visiblefeatures. Eval dense.
Selected optional-question default was mask-only; no reply requested full DuET
prefix completion. No densecompletion/shortcut/spatialinjection added.
No new parameters; maskedtraining rejectsUSP>1 atpreflight. Existing unmaskedUSP
retained; CP/distributed/engine/mcore untouched.

Validation:108CPU passed/4assetskips/27GPUdeselected aftermask;8new tests include
independentdensekeymask forward/gradient/RoPE oracle, causality/noleak,DuETsampling,
evalbitwise/RNG. Datafix26tests thenstrict44filelint/264mcorecheckspassed;
strict19pagedocs passed. FullH200local8 two steps passed82.36GiB.
No long-run convergence or maskedUSP/FSDP comparison claimed.

Firsttask stormyxie-vf-mask80-09191601 had NO optimizer step, explicitly stopped,
deleted and verified0Pods. Diagnosed all32ranks lstat-ing everyvideo during dataset
construction: Path.resolve on244472entries/rank. Fixed tasks/rep_world/data.py to
keep absolute/manifest-relative paths, defervideoaccess to __getitem__, preserving
file and symlink semantics. Newtest prohibits pervideo lstat atindex construction;
wholemanifest index2.84seconds. Oldfailureevidence inattempt1; do not restart it.

## RepWorld multi-node validation — completed (2026-09-19)

User requested real remote multi-node validation. CP remains unchanged.
Task `stormyxie-vf-mn-0919150228`: 2×8 NVIDIA B300 SXM6 AC,
torch2.9.1+cu130, actual NCCL Libfabric/efa-direct. Two physical nodes and 16 unique GPU UUIDs checked.
KAP final status **Succeeded**, both workers exited0, task deleted and logs retained.

Report/evidence: `/efs/stormyxie/outputs/vforge/multinode_audit_20260919/REPORT.md`.
Main evidence: task_result.json, validated.json, ddp16_validation.json, pods_initial.json,
run1/result.json, run1/resume_comparison.json, run1/eval_validation.json.
Scripts/profile/manifest: `/threed-code/stormyxie/vforge_jobs/multinode_audit_20260919`.
Use the bundle path in snapshot.json: the authoritative revision bundle is on EFS.
The partial jobs/bundle from slow object-mount extraction is unused.

Immutable snapshot `fbb4677bdb05bc84b3ef0b62b73766d4f18b4c19` lives in an isolated checkout
at /efs/stormyxie/work/vforge_multinode_20260919. Original xj_dev HEAD/index unchanged, no push.
This follow-up changed tests/template/docs only; RepWorld implementation, CP/core/engines unchanged.

Verified:
- DDP16, FSDP16, genuine cross-node USP16, DP8×USP2, HSDP(2 replica×4 shard)×USP2.
- 3 topologies ×20 numerical cases =60: RGBpatch2/4, latent, z-prev, pureTF;
  FP32/BF16, recompute, complete gradients, Muon update, shared RNG and KV cache.
  Max relative gradient error FP32 4.19e-7, BF16 1.145%; original tolerances unchanged.
  Muon bitwise update check uses the same independently validated gradient; no independent-trajectory claim.
- Complete12×768 models: VideoRAE RGB10states across all main topologies;
  all four codec RGB paths with DP8×USP2 (VideoRAE1D10states, VRAE EUPE/VJEPA20states), each3steps.
- Complete VideoRAE latent, HSDP+USP2, AC+accum2, deterministic: continuous4 vs step3→4
  full DCP weights(454 tensors/221372000 elements), optimizer(733/221542784), all16 trainer/RNG/data
  states and loss bitwise equal. Loss both1.9860607385635376.
- Native eval from HSDP DCP, DP8×USP2,8cases,CFG2,5Euler,3states: all8 MP4 decode48×256×256.
- Supplemental DDP16 ran on the same two nodes/code bundle; both nodes exit0,3steps/globalbatch16.
  Initial node0 pod-exec HTTP500 was retried successfully; ddp16-node*.json are the training results.
- 43-file strict changed lint, 264 mcore checksums and import smoke passed before submit.
  Final strict43file lint,264 mcore checksums,import smoke,19page docs build and git diff --check passed.
  Cleanup API recheck: no remaining Pods; original8 DuET GPU processes still present.

Reusable gate: tests/tasks/rep_world/multinode_worker.py plus scripts/ci/kap/rep_world.task.yaml.
Numerical worker now accepts --min-hosts and records actual rank/host/USP groups.
Main gate has13 distributed stages; supplemental DDP is separately recorded.
Some stages overlap with supplemental DDP: do not interpret timing as isolated throughput.
No long training started. Original DuET job was not modified.

Still unverified: Ring/CP, TP/EP/PP, >2 nodes, other hardware combinations, cross-topology bitwise resume,
full-size multi-node z-prev/pureTF training, long-run convergence and isolated performance.


## RepWorld Ulysses support and parallel audit — completed (2026-09-19)

Latest user steering: **do not modify CP**. Implemented model-owned Ulysses only.
In this repository `usp_size` means Ulysses all-to-all, not the broader Ulysses+Ring USP.
No CP/distributed/engine/mcore implementation changed; preflight hashes prove this.
No commit/push, no changes to the original DuET job (8 processes still running at completion).

Durable report: `/efs/stormyxie/outputs/vforge/parallel_audit_20260919/REPORT.md`.
Exact commands, source snapshots and helpers:
`/threed-code/stormyxie/vforge_jobs/parallel_audit_20260919`.
User guide: `docs/models/rep-world.md`. Capability remains internal validation.

Implemented in RepWorld model/task only:
- Native token slice/gather + Ulysses self-attention, global RoPE/causal masks, local Q/full condition KV.
- Correct gradient scaling on replicated/sharded boundaries; same FM noise/time/dropout per USP group.
- Noncontiguous RNG tensors use contiguous broadcast transport then copy back, preserving tensor layout.
- z-prev prefix handling and pure-TF global KV cache layout; cache remains replicated.
- Native train/eval accepts USP with FSDP2/HSDP. DDP+USP still rejected; TP/CP remain unsupported.
- RepWorld default `parallel.mixed_precision.enabled=false` preserves FP32 parameter/BF16 autocast semantics.
  For older DCP produced with true, strict resume requires the original resolved precision configuration.

Verified single-node 8×H200, torch2.9.1+cu130, seed42:
- Full 12×768 ViT + 12×768 World: DDP2, FSDP2, HSDP4, USP2/4/8, DP2×USP2,
  HSDP4×USP2, activation checkpointing and gradient accumulation. All short native train jobs 3 steps.
- Four codecs USP2; VideoRAE RGB/latent, z-prev/pure-TF (latter 3 states).
- 5 topologies ×20 numerical oracle cases =100: RGB patch2/4, latent, z-prev, pure-TF,
  FP32/BF16, recompute on/off, complete gradients, two Muon updates, RNG and KV cache.
  Validate independent gradients first, then bitwise optimizer update with identical validated gradients.
  This is not a claim of bitwise independent BF16 training trajectories.
- Full USP2 latent AC+accum2 DCP: deterministic continuous4 vs checkpoint3→4 weights, optimizer,
  loss, RNG/data/trainer state all bitwise equal. Non-deterministic run had tiny differences, retained.
- Native eval from USP DCP: single/DP2/USP2 ×4 cases, CFG2,5Euler,3states. All12 videos decode.
  DP2 equals single decoded pixels; USP2 mean absolute pixel difference 2.00–2.18/255 (not bitwise).
- Actual CLI negative gates TP, EP, PP, Megatron, DDP+USP rejected as expected.
- CPU98 passed,4 asset skips,27 GPU deselected; strict changed42file lint passed;
  264 mcore checksums/import smoke passed; strict19page docs build passed.

Effective oracle files: usp2_oracle_v3.json, usp4_oracle_no_tf32.json,
dp2_usp2_oracle.json, usp8_oracle.json, hsdp_usp2_oracle.json.
Initial USP4 oracle in oracles.json failed at FP32 relative gradient3.22e-5 due TF32-enabled baseline;
explicit TF32 off rerun passed WITHOUT loosening tolerance. Current worker disables TF32.
Initial four USP train failures were noncontiguous broadcast, fixed and rerun under *_v2.
Initial oracle's independent Muon updates amplified tiny gradient differences; revised test isolates
gradient correctness then optimizer correctness. No production optimizer workaround.

All audit controllers/GPU workers finished; no audit process left on GPU. Do not restart tests merely
because old failure logs remain. Final report, final_checks.json, final_cpu_v2.xml,
deterministic_resume_comparison.json, eval_validation.json and final_source.json are authoritative.
Historical single-node limits: multi-node and B300 are superseded by the completed audit above.
Still unverified: Ring/CP, TP/EP/PP, long-run convergence and isolated throughput.


# RepWorld Native migration — completed main path

Updated 2026-09-18. Branch `xj_dev`, base `1791f319840ceb5b6407d65002c482e907abb8cc`.
No commit/push. The clean main-path migration and independent correctness review are complete.
Model maturity remains **internal validation**, not production/KAP acceptance.

## Read first

Durable audit report and evidence:
`/efs/stormyxie/outputs/vforge/repworld_migration_validation_20260918/REPORT.md`.
That directory includes final logs/XML, exact legacy cases, fixed-input SHA256, source fingerprints,
10-case zero-error results, wheel, and small Native config/video/checkpoint-manifest artifacts.
User-facing contracts and runnable commands: `docs/models/rep-world.md`.
All LIVE handles in the archived old handoff are stale; none are pending migration work.

## Delivered scope and ownership

- `models/rep_world`: World-DiT, causal RGB/latent ViT, original FM objective, no_zprev/zprev/pure_tf,
  strict-causal inference and four frozen codecs (VideoRAE3D/1D; VRAE EUPE/V-JEPA2.1).
- `tasks/rep_world`: Native train/eval, online video data, codec provider, Muon, strict initial weights.
  Four experiment presets expose public overrides; no duplicated Trainer/checkpoint/attention backend.
- Existing infra changes are only registry entries and license package-data. mcore is unchanged.
- Runtime never imports the old project or upstream codec checkout. Published weights include teacher
  parameters. Preserve codec licenses/sources; EUPE has research/noncommercial terms.
- Original online train/eval is the migrated main path. Offline cache/prepare/mixed sources are
  explicitly unsupported, not a pending migration blocker. No encode builder is registered.
- Historical diagnostics, launchers, RCG/NextLat/joint-pixel experiments, EMA and legacy optimizer
  resume are out of scope. Source repo remains the reference/history; do not move its experiments.

## Semantics that must not drift

Packed latent `[B,P*C,S,16,16]`, plane-major then channel; explicit `[B,S,C,P,16,16]`.
3D:16 RGB/state,C64/P4. 1D:16 RGB/state,C32,512 learned slots arranged P2 only for indexing;
this is NOT physical temporal compression. EUPE:4 RGB,C768/P1; V-JEPA2.1:4 RGB,C1024/P1.
RGB ViT temporal patch2/4 is distinct from codec compression. RGB rollout consumes decoded generated
RGB; latent rollout consumes generated latents directly, without re-encoding/extra normalization.
EUPE decoder streams chunk-causal KV; V-JEPA decoder recomputes full-attention prefixes;
VideoRAE decodes independent states. Pure-TF must decode one state at a time too.

- CFG/Euler use dimensional broadcast tensors to preserve FP32 state/BF16 velocity promotion.
- Pure-TF first KV prefill dynamically computes RoPE under autocast even at offset0; never reuse the
  Native single-state FP32 table. Source trained-window cache always misses for valid rollout prefixes.
- Legacy pure-TF full-window cos/sin tables are shape-validated and converted to the Native cache;
  all learned weights/base frequencies remain strict. Example old0.22B is16×960/15heads/MLP4.3125.
- Missing legacy previous-context=false; missing/None/empty feature layers=last layer;
  missing semantic_input_size=224. Geometry/config/active state tensors must match.
- Legacy init is weights-only, not optimizer/step/data/RNG resume. Full resume is Native DCP.
- Muon must gather complete DTensor matrices before orthogonalizing, then restore placements.
- Preserve BF16 codec details: contiguous decoder input, teacher-specific RoPE dtypes/frequency order,
  VideoRAE complex128 lazy tables, and teacher-internal SDPA backend selection.

## Verified

Local H200, Python3.12, torch2.9.1+cu130, FP32 trainable parameters + BF16 autocast.
Dedicated `/efs/stormyxie/venvs/vforge`; original training environment unchanged.

- Final targeted CPU:122 passed,4 optional asset skips,22 GPU deselections. Includes causal/gradient,
  activation checkpoint, Muon/DTensor and real Native DCP bitwise split-vs-continuous contracts.
- Published codec GPU parity:4/4 bitwise encode/decode/stream.
- Native CLI12 full-size codec×RGB2/RGB4/latent cases +8 codec×zprev/pure-TF cases passed:
  2 train steps, full DCP, short eval; all20 MP4s decode. Seed42, single H200; mode cases3states.
- Real trained full50Euler rollouts:10/10 BITWISE latent+RGB (atol=rtol=0), seed1234, fixed inputs,
  80/160frames, semantic CFG2/pureTF CFG1. VideoRAE3D+EUPE RGB2/RGB4/latent; VideoRAE3D/1D
  zprev; VideoRAE16×960 pureTF; V-JEPA2.1 RGB. Both sides SDPA. Optional source FA2 is disabled
  only in test, not production. No across-backend bitwise claim.
- PureTF first-state CUDA RoPE regression passes and fails when old cache behavior is injected.
- Prior agent's single-card full-size EUPE DCP resume, small two-H200 DDP/FSDP2 activation-checkpoint
  and FSDP2 resume artifacts were inspected and retained. Not rerun in this review, no topology parity claim.
- Final strict changed lint,264 mcore checksums/import smoke, strict19-page docs build passed.
- Wheel rebuilt with final source; all RepWorld modules/configs/licenses checked against source,
  imported from isolated extraction with source-project imports forbidden. See durable wheel evidence.

The20 CLI smokes preceded final numerical fixes; final CPU/GPU regression and full legacy rollouts
cover changed behavior. Do not confuse those smoke videos with the final trained-weight parity proof.

## Whole-repository limitations

Full diagnostic CPU:1733pass/34skip/14fail/14collection-errors. Not a green whole-repo suite.
7fail+6collection-errors: absent private Gen Engine submodule;1fail+1collection-error: absent H3 in
current diffusers.6 Wan failures reproduced on clean baseline archive (6fail/46pass).
7 additional helper collection errors came from diagnostic importlib mode; normal collection confirms
only7 external dependency errors remain. GenEngine init failed because localhost2222 SSH tunnel is down.
No unrelated code/env change to conceal failures. Exact logs and baseline reproduction are in REPORT.md.

Historical migration-only limits (2026-09-18; USP superseded by audit above):
KAP/B300, multi-node/full-size distributed throughput, TP/CP, production quality,
all long-rollout combinations. Do not infer them from local smokes. User DuET training was untouched;
no Koala/KAP tasks were launched. Original source model/train/sample/config were untouched by this
review; only its AGENTS/HANDOFF were updated. Earlier old-project latent diagnostics remain separate.


## H200 scaling sweep — active, updated 2026-09-20 23:04 UTC

Goal remains ACTIVE until all 10 runs train 10,000 steps and all 200 formal FVDs
complete. User allows extra sampling on spare H200 nodes and prompt cleanup.
Canonical handoff: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/HANDOFF.md
Pointer: /efs/stormyxie/work/scaling_fa4_active.json
Report: http://127.0.0.1:8884/ ; preserve 8882/8883 too.

Six 125M/355M/775M arms COMPLETED 10,000 and terminal audits PASS. Each has
10,000 contiguous finite-loss step records, 20 full Native checkpoints, and all
required components, fixed recipe, fingerprints and final rank states verified.
All 20 per arm are tracked for FVD. Do NOT restart these six because they are
absent from Running-only task listings.

Latest Native/direct KAP inventory at 23:01:32 UTC:
remaining four trainers Running: 1.56B entry8607/cross7335 (latest8500/7000);
20B entry1426/cross1247 (latest1000/1000).
155 complete checkpoints,146 rawqueued,4 active,8 scored,untracked[].
Rawqueue overlaps external active jobs; counts are not disjoint.

Four actual evaluation tasks / 32 H200, all Running with advancing outputs:
primary0920154306:20Bcross500,880 cases;
extra0920224122:775Mentry500,592;
extra0920224445:125Mentry1000,648;
extra0920224846:125Mcross1000,440.
Extras cap6; inspect live controller for later allocations. Quota is not actual
free-node capacity. Six small trainers remain complete; four trainers remain.

Eight independently audited FVD2048/CFG1 scores at step500:
125M entry5132.92898 / cross3543.09943;
355M entry1908.56935 / cross2009.59295;
775M cross1945.28992, entryrunning;
1.56B entry1775.48702 / cross1889.53491;
20B entry1643.00389, crossrunning.
20Bentry vs1.56Bentry uses identical cases/seeds/GT/settings, point difference
-7.4618%. These are early single-training-seed results, not confidence intervals
or a scaling law. DiT FFN widths differ across modes to match total parameters.

NEW CPU extras controller training-release wakeup is DEPLOYED/VERIFIED:
PID2079785 replaced1710760 at22:57:50. Source8e3273f8...,21 CPU tests PASS.
Known successful trainers must have KAPSucceeded and all worker PodsSucceeded,
assigned nodes, exit0 and finishedAt. Actual exit must postdate the capacity
failure before only that cluster's cooldown is cleared. Per-task dedup prevents
old completions cancelling newer backoff. Six old completions were observed
without clearing backoff; all three existing samplers kept their taskIDs and
advanced across the CPU controller replacement.

DISTINCT primary GPU pool v2 is still STAGED ONLY, source62a8412b...,25 tests.
Parent lease, held-head skip and1.56B-exclusive dispatch have NOT been deployed.
Do not interrupt useful20Bcross inference to activate this staged GPU scheduler.

Recipe unchanged: five sizes × entryconcatonly/crossattentiononly, ViT:DiT~2:1,
latent/no-zprev,mask80%,8H200 each,physical local8/global64,FA4 trainable towers
without fallback,NativeFSDP2,fullDCP500/keepall. Independent2048case/CFG1/50Euler/
10state evaluation outside training jobs. Published frozen VideoRAE retains SDPA.
No B300 or numerical/batching/resharding/4GPU migration. No subagents/commit/push.

Full proofs and guarded next actions are in the canonical handoff.

## 2026-09-21 — First-block CA-before-SA, EMA 0.99 restart; old condition report

Newest user request: maintain EMA decay 0.99 for the three newly added first-layer
CA experiments, evaluate BOTH online and EMA at every checkpoint, restart.
The previous xpre trainers and their monitor/eval controllers were stopped;
all three former KAP tasks independently return task not found. Their outputs
and logs remain intact. Original 10 scaling experiments and evaluations continue.

Active replacement:
- Jobs: /threed-code/stormyxie/vforge_jobs/scaling_first_cross_ema_20260921T084210Z
- Report: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema; preview http://127.0.0.1:8884/first-cross/
- 125M / 355M / 775M baseline families; actual 117661696 / 333760224 /
  728713152 trainable parameters. First block CA -> SA -> FFN, no later CA.
- Each 8 H200, FA4, FSDP2, local8/global64, latent/no-zprev, mask80%, step0
  fresh outputs, max10000, checkpoint500. Only EMA and output/checkpoint paths
  differ from the immediately previous CA-before-SA configs (config_changes.json).
- Tasks: stormyxie-sc-125m-xema-0921085012,
  stormyxie-sc-355m-xema-0921085016, stormyxie-sc-775m-xema-0921085020;
  tenc-awsusw2 / gwm. Actual training events verified finite from step0, with
  ema_updates == step+1 in every observed event.
- Canonical task code: vforge/tasks/rep_world/ema.py; shared/train/eval assembly
  plus declared experiment extras. Reuses ShardedEmaState, post_optimizer_step,
  Native DCP and component loader; no copied trainer/backend or core edits.
- Joint ViT+DiT FP32 EMA shadows preserve FSDP layout, copied from initial weights.
  Every optimizer step: EMA=.99*EMA+.01*online (including first step); num_updates,
  parameter-name order and decay persist in DCP and are validated. Frozen codec excluded.
- Eval aliases are in eval_runs.json: three __online and three __ema entries,
  same checkpoint paths with independent output directories/leases/identities.
  runs.json has THREE trainers only. Each checkpoint enqueues both aliases.
  expected_scores=120. Metrics/protocol/cases/seeds remain held-out2048, CFG1,
  16 observed+144 generated, 50 Euler/state, I3D400FP32/ddof1. Batch256 for125,
  batch128 for355/775. Local dev1 only when all8GPUs idle; up to3 remote H200
  extras; original parent extras remain capped17. Never use B300.
- Services: monitor317417, extras317418, local317419. Exact live cmdline must
  be checked before signaling. Monitoring and dual queue are running.
- Full CPU formula + joint Muon/DCP exact-resume + strict online/EMA load tests
  passed. Targeted initial53tests; final13EMA/resume tests; changed lint56files,
  264 vendored mcore hashes unchanged; canonical source branch not pushed.
- Actual dev1 8H200 FSDP2/FA4 smoke: train2, full-resume3, then BOTH weight paths
  sample16 cases at batch2/rank with samecase/seed/GT. All16 generated hashes
  differ online vs EMA; correct saved EMA update count3. Complete evidence:
  validation/gpu_ema_smoke_PASS.json.
- DCP2->3 recurrence checked over all338 parameter tensors /117661696 elements;
  maximum CPU recomputation difference 7.450580596923828e-09, EMA tensors FP32,
  DCP metadata proves8way sharding. validation/gpu_checkpoint_ema_recurrence.json.
- Dual queue contract validates3checkpoints ->6distinctjobs/directories; waiting
  for first production500step, so no formal new-run FVD result yet.
- Source digests train 7ab8825d853433c33cfc2ae1514fecd450af3c628ace5f7ecb5f162cccc4030d, eval 681d02e8bdfa33f4ebf63ae35f66721c3f581810bfa760ad0eae8e6a7a51c298.
  Immutable snapshots must not be edited after launch.

Preceding request also completed: report for yesterday's A/B/C/D/E/G/H condition
injection ablations (old RepMiniWorld, distinct from scaling):
- /efs/stormyxie/reports/condition_ablation_report_20260921T083044Z
- jobs /threed-code/stormyxie/rep_miniworld_jobs/condition_report_20260921T083044Z
- http://127.0.0.1:8884/condition-ablation/
- All7 at common37500, fixed16cases CFG1 batch16. Persistedinput/history/prefix/
  all9actualnoisetensors/Eulerschedule/GT/outputhashes checked. Original models,
  checkpoints and legacy attention unchanged. All old training remains stopped.
- 16percase videos GT+A/B/C/D/E/G/H (1024x608), plus native7columns x16rows
  matrix1888x4224/160frames/16FPS, plus existingearly2500/5000 six-arm videos.
- Diagnostic FVD16 only,144generatedRGBframes, rawNPY/I3D400FP32/ddof1:
  A785.0771 B929.5789 C894.9839 D800.8839 E794.4768 G787.9001 H755.7581.
  NOT statistically stable ranking, NOT directly comparable to formalFVD2048.
- Stable loggedsteps/s last5clean200stepwindows:
  A .76 B300; B .82 C .90 D .91 E .93 G .82 H .83 H200.
  rank0 peakallocatedGiB60.11/60.18/50.23/50.40/50.40/60.33/60.33.
  Report excludesdemo/checkpoint/startup/epochboundaries, explains B resume/hardware
  and unequalparameter confounds. Training curves go to37500.
- 8884 server retains original chart routes and now supports Range206 media
  with a fixed report/media whitelist. Main tunnel verified200; MP4 byte-range
  verified206. Media decoded/inspected in composition; posters/charts visually checked.

Original long goal still active; do not mark complete or discard20B remaining
training/FVD. Total original200 + supplemental120 formal FVD commitments.
