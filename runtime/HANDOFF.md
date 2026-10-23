## 2026-09-24 — Wan2.2 VAE RGB4 latent experiment (active)

User requested another ~355M case replacing VideoRAE with Wan2.2 TI2V-5B VAE, DiT predicting Wan latents, with 4 real RGB frames/state. Native code: `vforge/models/rep_world/wan22_codec.py` and official Apache-2.0 VAE source in `vendor/wan22/` (unmodified source + license + SHA). `config.py` registers `wan22/ti2v5b` geometry `(rgb_frames=4, channels=48, planes=1, grid=16)`; `tasks/rep_world/shared.py` loads the frozen asset through CodecProvider. The RGB ViT remains E:C=3:18 blocks; WorldDiT remains 7 coarse blocks, first-block global CA+3D RoPE, no PixelDiT branch. Total trainable params354,974,240. The official VAE independently encodes its first frame and then groups four frames. For 40 actual frames, the codec prepends a duplicate first frame to make 41 VAE inputs, obtains11 latents, keeps the final10 as World states, and stores initial latent only as decode context. Decode prepends anchor and drops first output frame, preserving exactly4 RGB frames/state. The VAE uses its official 48-channel normalization; no VideoRAE stats or decoder. Stream decode currently recomputes full latent prefix, no VAE KV/cache optimization.

Frozen job/config/runtime and logs `/threed-code/stormyxie/vforge_jobs/wan22_rgb4_355m_20260924/`, `/efs/stormyxie/reports/wan22_rgb4_355m_20260924/`. Job `stormyxie-wan4-35-0924153051-c2d62` submitted to H200-2D with8 GPUs; first real FA4/FSDP2 steps completed. At step1 loss1.945, e2e7.71s/step, peak allocated7.42GB; online VAE dominates time. Local batch8 remains unchanged. Plan inherited from RGB-input baseline: mask90/K2 10k pretrain, every2500 checkpoint, then mask0/next-state full-state2k tuning every500, EMA0.99. Durable controller running on dev1, retries preemptions. No FVD yet.

Checks: official VAE checkpoint196/196 keys match architecture and all shapes; fake-VAE RGB4 boundary/stream tests passed, targeted rep-world CPU tests27 passed. Only real training first steps validated, not a completed10k run or FVD. Wan job and prior pure-RGB PixelDiT job are separate concurrent cases; do not stop PixelDiT for this one.

## 2026-09-24 — 355M direct RGB4 PixelDiT ablation (active)

User requested a DuET `duet_pixeldit`-style VAE-free model, four RGB frames per state, keeping 10 states and comparing against the preceding RGB-input/VideoRAE-target case. Native implementation is in `/efs/stormyxie/vforge/vforge/models/rep_world/pixel_codec.py` (lossless RGB4 patch16 packing), `pixel_dit.py` (global patch pathway is existing WorldDiT first-block CA+3D RoPE, followed by one fine PiT block with pixel-to-patch compression/cross-patch attention/expansion and direct 12-channel RGB velocity), plus `config.py`, `world.py`, `tasks/rep_world/shared.py`. No RAE checkpoint or statistics are loaded. This follows the two-level *method*, scaled to ~355M; it is not a weight-identical copy of DuET. ViT-E42.1M/ViT-C234.0M/DiT77.5M; total353,584,192. 10 states =40 actual frames; an eval rollout of9 states is36 frames, not comparable in time length to the old VideoRAE 144-frame rollout.

Frozen configs/runtime/controller: `/threed-code/stormyxie/vforge_jobs/pixeldit_rgb4_355m_20260924/` and `/efs/stormyxie/reports/pixeldit_rgb4_355m_20260924/`. One H200-2D 8-GPU job `stormyxie-pdit-35-0924151146-6fd0a` submitted and Running. Local batch8, FA4 strict, FSDP2, mask90/K2, 10k pretrain (save every2500), then full-state 2k mask0/next tuning (save every500), EMA. Durable controller PID observed on dev1, retries only preempted attempts. First 18 training steps passed; steady steps about1.05–1.15s except data outliers, peak GPU allocation ~7.19GB, loss ~1.25–1.33 early. Old RGB-input/VideoRAE-target task `stormyxie-rgb4-35-0924134347-3daff` was deleted only after new task reached Running; old controller STOP_CONTROLLER marker set. Old task stopped at ~970 steps without a completed checkpoint (first save scheduled2500), so its in-flight progress is not recoverable from DCP.

Validation: pixel codec exact roundtrip and CPU WorldDiT+CA+PiT forward/backward tests passed; targeted rep-world train tests 24 passed. Real 8-H200 FA4/FSDP2 has completed multiple train steps. No FVD is available yet. Do not reinterpret a 36-frame FVD as an equal-length comparison to VideoRAE latent runs.

## 2026-09-22T09:40:35.101998+00:00 — User correction: K counts ZERO-BASED choices; supersedes earlier offset/SA coordinate descriptions

K=3 means offsets0/1/2; fixed quadrant K=4 means permutation0/1/2/3.
Offset0 is ordinary NEXT-state prediction, NOT reconstructing the history origin.
Canonical helper/API target_offsets and quadrant_offsets now use zero-based values.
Both source gather paths use origin+1+offset, so physical targets remain future states1..K.
Random draw probabilities/RNG and source targets are unchanged by renumbering.

Mosaic SA Q/K now use offset*P+plane; P4 ->0..3/4..7/8..11/12..15.
All-zero offset map matches ordinary prediction exactly in CPU tests (nonzero output head,
entry/globalCA3D/sameCA3D/noCArope). CA keeps its original condition-centered reference:
Q=(offset+1)*P+plane, K=plane. This is equivalent to SA-centered Q with K=-P+plane.
Do not subtract one state from CA Q alone: that would reduce the real condition-target gap.
Gap embedding is sincos(offset+1)-sincos(1); offset0 remains exactly zero.
Whole-state distant-target SA stays state-local (a common temporal shift cancels in ideal
relative logits). ViT history positioning is unchanged. Logged horizon1..4 still means
physical future-state distance = offset+1, deliberately distinct from zero-based offset.

Validation after this correction:63 targeted CPU passed, including source gather,
independent SA/CA rotation oracles, new zero-map/ordinary exact parity, RNG and full DCP
continuous/resume with EMA;12 H200 BF16 FA4 model gates passed, no SDPA fallback;
changed-file lint and docs build passed. No new multi-node/USP test or distributed launch.
Previous two-H200 FSDP2 smoke belongs to preceding revision; no claim of re-running it here.
Evidence/preimages/final hashes: /efs/stormyxie/reports/quadrant_future_zero_based_20260922.
Docs updated in docs/models/rep-world.md. Short GPU checks finished; no long run started.
Existing355Mmask0 training/evaluation and monitor goal remain USER PAUSED.

## 2026-09-22T09:27:01.772732+00:00 — K=4 quadrant future-target implementation; campaign remains USER PAUSED

Canonical repo /efs/stormyxie/vforge now supports model.extras.flow.target_layout=quadrant_mosaic.
Default single_state and old distant_target_prob behavior preserved. Set distant_target_prob=0
for mosaic; no_zprev only; spatially ordered codec and even DiT patch grid required.
Each clip/origin draws an independent permutation of horizons1/2/3/4 in TL/TR/BL/BR order,
gathers the SAME spatial quadrant from origin+gap, preserving C and P. All four temporal
planes of VideoRAE share each quadrant's gap. Units are codec states: VideoRAE one state
is16 sampled RGB frames/P4; this is not four consecutive RGB frames.
Uses S-4 origins (10states ->6 targets), condition never sees future states.
Single FM timestep per composite target; normal pixelwise noise; existing objective preserved.
Logs target_layout, prediction_positions, prediction_quadrants, horizon fractions (all.25)
and per-horizon loss contributions summing to loss_fm.

DiT ALL self-attention Q AND K use tokenwise time gap*P+plane, unchanged h/w.
Example TL/TR/BL/BR [3,1,4,2], P4 ->times12..15/4..7/16..19/8..11.
SA remains full across quadrants. Common time shift cancels in ideal RoPE relative logits;
different quadrant shifts do not, so CA Q alone would be insufficient.
CA3D Q uses same token positions; K stays origin-local0..P-1 with original visible indices.
ViT causal history RoPE unchanged. Existing sincos(gap)-sincos(1) becomes tokenwise after
entry fusion/before first block. Diffusion timestep/AdaLN unchanged. No new trainable params.
Low-level velocity accepts int64 target_offsets[B,N] in1..4; normal rollout/FVD remains next-state.
Train test hook quadrant_offsets[B,S-4,4] enforces permutation; target_uniform incompatible.
Global/no-CA-RoPE and entry/same/CA3D paths work within their preexisting restrictions.
Position-aware CA still USP1; new mosaic multi-node/USP path NOT verified.

Validation:286 CPU passed4 skipped;12 H200 BF16 FA4 GPU cases passed (6old scalar+6mosaic),
covering entry/global3D/same3D and mask0/.8, no SDPA fallback.
Independent source-coordinate and complex-rotation oracles passed; every SA layer and CA
coordinates checked. Checkpoint-recompute gradients exact on CPU; strict full DCP continuous/
resume checks include permutation RNG, all model/optimizer/EMA state and loss.
Two H200 FSDP2 real video+VideoRAE small-tower train2/save/full-resume3 passed for entry and
globalCA3D, Muon+EMA. CP1/2/3 committed, EMA counts1/2/3, fractions.25 and metadata verified.
These are smoke validations, not a new convergence experiment or GPU bitwise trajectory claim.
Changed-file lint (including264 vendored hashes/imports) and docs build passed.
Docs:docs/models/rep-world.md; tests:test_quadrant_future.py, extended distant_target_gpu,
config and real-video DCP resume tests.
Evidence/preimages/hashes:/efs/stormyxie/reports/quadrant_future_k4_20260922.
External smoke scripts/configs:/threed-code/stormyxie/vforge_jobs/quadrant_future_k4_20260922.
Short smoke processes finished. No long training launched. Existing355Mmask0 campaign and
monitor goal remain USER PAUSED; do not revive from historical Running entries.
Frozen runtime/config/protocol snapshots unchanged.

## 2026-09-22T08:52:22.475229+00:00 — DuET position clarification

The historical VideoRAE DuET demo/checkpoints use additive sincos3d, NOT Q/K RoPE:
condition positions stay at the historical origin; fused target tokens get actual future
absolute 3D position plus sincos(gap)-sincos(1). DuET has no DiT history CA.
Current DuET default Wan recipe instead uses relative_time: ViT temporal RoPE, no additive
absolute time, still explicit gap embedding. Do not conflate the two recipes.
RepWorld feature preserves its own RoPE and adds the DuET sampler/gap embedding; gap-dependent
CA Q time translation is specific to RepWorld, not copied from DuET's concat architecture.
Detailed code lines/evidence: /efs/stormyxie/reports/duet_distant_target_20260922T083000Z/duet_position_path_audit.txt.
Campaign remains user-paused; no extra experiment launched for this read-only audit.

## 2026-09-22T08:47:44.975358+00:00 — DuET distant-target implementation complete; mask0 campaign remains USER PAUSED

Canonical repository: /efs/stormyxie/vforge. Implemented model.extras.flow.distant_target_prob
(default 0.0; example 0.25 -> next/second/third state 75%/12.5%/12.5%).
Supported no_zprev, RGB/latent conditioner, entry concat and global/same-spatial CA.
p>0 uses S-3 causal origins; gathers target origin+gap; never feeds skipped future states to ViT.
Adds parameter-free DuET sincos(gap)-sincos(1) to DiT hidden before first block, after entry fusion.
Explicit gap leaves diffusion timestep/AdaLN unchanged. Existing p=0 RNG/loss/gradient path preserved.
CA3D K coordinates remain [0,P); Q becomes [gap*P,(gap+1)*P). ViT position encoding is unchanged;
DiT SA retains state-local coordinates. VideoRAE P=4: Q4..7/8..11/12..15, K0..3.
With CA RoPE off or entry concat, gap information still arrives through hidden embedding.
Default rollout/FVD remains next-state only (gap1 -> exactly zero embedding).
No parameters/buffers added. New sampler/gap code: vforge/models/rep_world/horizon.py.
Task adjusts prediction_positions/local_tokens and logs horizon fractions/loss contributions.
User docs: docs/models/rep-world.md. Public Hydra field is declared, no + override needed.

Validation: 241 CPU passed, 4 skipped; 6 H200 BF16 FA4 model gates passed (0/80% mask x three
conditioning modes; no SDPA fallback). Two-H200 FSDP2 actual-video/VideoRAE Native smoke passed
for entry concat and global CA3D, train2 + full resume3, Muon+EMA, saved CP1/2/3, EMA updates1/2/3.
CPU resume with one thread is bitwise for all parameters, Muon state, EMA, losses and horizon
metrics. A 2-thread CPU run showed tiny reduction roundoff in parameters; strict test pins one
thread, does not weaken exact comparisons. No claim of GPU independent-trajectory bitwise parity.
Sampler and gap embedding match actual inspected DuET function bodies bitwise for tested
p=.25/.5/1 and FP32/BF16 widths24/25/192/768.
Offset audit checks unchanged ViT/K, changed Q time phase, unchanged spatial rotation, and
equivalence to moving K into negative relative time (max FP32 QK-logit error 1.431e-6).
Lint including vendored checks/imports and docs build passed.
Evidence, preimages and final source hashes: /efs/stormyxie/reports/duet_distant_target_20260922T083000Z.
External smoke/audit scripts: /threed-code/stormyxie/vforge_jobs/duet_distant_target_20260922T083000Z.
All smoke processes finished; GPU compute-process list empty after completion.
No new long experiment has been launched; user has not specified the final selected conditioning arm.

PAUSE remains in force: all four 355M mask0 training tasks and owned FVD tasks deleted,
controllers stopped and controls disabled. The monitor goal is PAUSED. Latest full CPs:
global CA3D5000, same CA3D4500, entry4500, original CA4500. All task Pods gone on final check.
70 completed FVD scores retained; preview8884 remains healthy and report shows Paused.
Do not revive campaign from historical Running entries below. Frozen snapshots were not changed.

## 2026-09-22 08:30 UTC — USER PAUSED mask0 campaign; overrides earlier monitoring instructions

User explicitly requested pause after selecting the final conditioning method. The monitor goal is PAUSED.
Do not restart training, FVD workers, or submission controllers without a new user request.
All 4 owned training tasks and 4 active remote FVD tasks were deleted; task get confirms NotFound.
dev1 monitor / fvd_extras / fvd_local controllers and local 8-rank FVD child processes stopped;
dev1 GPU compute-process list was empty after cleanup. Same-patch training container terminated at
08:16:49 UTC; its Pod object was still in deletion grace during the 08:18 inspection.
Latest full checkpoints retained: global CA 3D RoPE 5000; same-patch CA 3D RoPE 4500;
entry concat 4500; global CA original/no CA RoPE 4500. Refreshed report retains 70 valid FVD
scores and 4 incomplete pending evaluations; pending work is paused, not automatically queued.
Preview server and local 8884 tunnel remain up. Mutable controls are disabled, report marks Paused.
Evidence/preimages: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/user_pause_20260922T081510Z.
Current new work: implement DuET distant_target_prob random horizon in canonical vforge, default off;
no new long training has been requested. Do not edit frozen runtime/config/protocol snapshots.

# vforge / RepWorld 当前交接

## mask0 Online新增历史入口concat mask80曲线；mask计算量核查（2026-09-22T08:10:04.116110+00:00）

- 用户追加要求：Online同时展示历史355M入口concat mask80%结果，并比较0/80%单步计算量。已加入原scaling_fa4_20260920T134440Z/355m_entry_concat的20个Online分数（500..10000）；蓝绿色#12879c短虚线，原mask80首层CA粉色虚线保留。Online共6条；EMA仍4条。
- 原始entry名称与当前mask0同名，展示层用355m_entry_concat_mask80并保留source_name和score SHA，避免混线。20个score身份、step/checkpoint、协议/source、eval manifest SHA逐一核验；两轮protocol.json逐字段相同、相同held-out2048/CFG1/144生成帧/50Euler/batch128，参数都355083936。旧entry未维护EMA，页面已明确只用其Online；两条历史对照均不进入本轮160计数/队列/训练loss。
- 数据继续嵌入mask80-reference-data（由单对象变为两组数组），无需新增HTTP路由。来源文件mask80-entry-online-reference.json；20点4k632.066298、10k422.975501。修改/原始index/HANDOFF备份及验证证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/entry_mask80_reference_20260922T080223Z。真实HTTP200与NodeVM/DOM stub验证Online linear/loglog各73点（当前33+历史40）、6条线，EMA33点4条线；所有raw FVD与坐标精确匹配，自动刷新/URL/任务表/计数/loss保持，不声称浏览器截图。
- 链接 http://127.0.0.1:8884/rope-mask0/?weight=online&scale=linear&compare=mask80&refresh=entry80#fvd-results ；需刷新HTML获得新增曲线。
- mask在19层ViT之前gather，4*floor(256*.2)=204tokens/state（mask0为1024）；9个conditioning states每clip9216→1836。ViT在state-causal45对state上attention。DiT仍72targets/rank×1024queries；entry先将204个condition投影/scatter到完整1024再concat，全部DiT层仍dense；全patchCA的KV1024→204、Q不变。
- 分析脚本 /threed-code/stormyxie/vforge_jobs/condition_rope_mask0_355m_20260922T035234Z/analyze_mask_compute.py，证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/mask_compute_comparison_20260922T080800Z/comparison.json。local8/rank的主要矩阵乘前向：entry83.878→27.344TF（-67.40%），首层CA84.094→27.234TF（-67.61%）。按3×前向+1×checkpointed blocks重算粗估训练334.50→108.46TF及335.54→108.19TF；排除frozen VideoRAE、optimizer/EMA/通信/非线性/FA tilepadding，非硬件计数器测量，nonreentrant早停重算实际可能更少。
- 同2001–4000completed steps、每500phase51..450、checkpoint_s=0，各1600条日志：entry mask0/80中位2.534/1.939秒（0.395/0.516steps/s，1.307x）；CA2.636/2.042秒（0.379/0.490steps/s，1.291x）。统计为历史不同节点，非受控同机benchmark；旧entry80无EMA维护，其余有。旧mask80日志长尾明显，全部选中记录的aggregate steps/s entry .355 vs mask0 .337、CA .338 vs .338，不能把中位1.3x误称整个训练wall-clock加速。
- 只改报告并加入只读分析脚本/证据；冻结runtime/config、现有训练评测服务未变。持续监控goal仍active至4×10k和160FVD。


## mask0 四组4000 Online/EMA完整，进入4500评测（2026-09-22T07:58:59.109570+00:00）

- 4000 FVD Online / EMA：全patch+3D RoPE310.619181 /106.789336；同patch+3D RoPE282.044854 /106.516733；入口concat391.920737 /89.561830；全patch原CA RoPE341.217497 /122.015093。入口EMA本档最低，但Online仍波动，尚非最终收敛排序。
- 固定2048case的入口EMA4000独立FVD89.561829569616 vs reported89.561829569936，差3.197e-10；输入hash/seed/detector/features通过。四组4000 CP完整性/EMA/scalars/8rank恢复状态此前已通过，均为CPU只读审计。
- 到4000共64分数身份、checkpoint、manifest/source/protocol/H200/2048/CFG1/144frames通过元数据核验；此前已审核63score SHA未变，最后原CA Online4000单独核验。不声称独立重算全部64特征统计。
- 07:58UTC四组日志4678/4372/4347/4305，4个训练实际task此前07:56确认Running；服务monitor/extras/local实际进程仍在。当前完整CP应有66项，64score+2个global4500评测覆盖，无缺项/重复；两个新采样Pod实际Running。
- 全部4000远端采样Pod已释放，最后dev1 worker812845已退出且score完整；global4500评测节点与训练节点分离。已完成旧任务不重启。
- 证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/complete4000_review_20260922T075814Z/review.json与all64_score_metadata.json，包含之前清理/CP/独立重算证据指针。last_notified_common_checkpoint=4000，last_notified_ema_checkpoint=4000；后续不重复该档结果。
- goal仍active至4×10k/160FVD及最终清理；保持冻结配置，不恢复已停止20B/旧消融。


## mask0 同3500step完整比较；四组4000 checkpoint审计通过（2026-09-22T07:50:05.818109+00:00）

- 3500 FVD Online / EMA：全patch+3D RoPE274.716368 /109.925354；同patch+3D RoPE269.977214 /107.174844；入口concat139.128719 /108.893850；全patch原CA RoPE423.424720 /138.786867。此档前三组EMA接近，不把早期点估计当最终排序。
- 新8score核验权重身份、step/checkpoint、2048cases、manifest SHA、source/protocol、H200/CFG1/144帧/50Euler/batch128；此前48score字节SHA未变。入口Online3500此前独立特征/FVD重算误差4.289e-9，已引用；没有声称独立重算全部56分数。
- 07:48UTC实际四个训练task及Pod均Running，日志4490/4199/4173/4127，所有已写loss有限。四个远端评测Pod与训练节点无重合；当前完成CP应有64项，score+queue+active覆盖，无漏项/重复。3500及全patchRoPE4000共7个已完成远端采样Pod清单为空；dev1同patchEMA4000 worker811273已退出且score已提交。
- 四组4000完整CP只读CPU审计通过：EMA updates4000、decay.99、FP32 shadows/name/shape/参数数目、所有DCP storage bounds、8rank scheduler/RNG/data cursor [1,1818]；非全部权重数值扫描或GPUrestore。审计进程已正常结束。
- 证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/complete3500_review_20260922T074825Z/review.json、score_metadata.json、followup.json；4000审计 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/production_step4000_ema_audit.json。监控goal继续active至四组10k和160FVD；后续通知从3500之后的新共同档位开始。不得恢复旧20B或旧消融。


## Mac 8884预览隧道卡住已修复（20260922T074549Z）

- 用户反馈当前报告打不开。远端共享ThreadingHTTPServer PID750580一直正常，dev1直接HTTP约0.004–0.009秒；故障在本机SSH转发连接仍存活但HTTP无数据。
- 旧Mac仅检测SSH退出的监督进程90448已退役，替换为本机 /Users/stormy/Documents/Codex/2026-09-19/zhih/work/scaling-preview-supervisor.py。当前detached supervisor PID61276（PPID1），SSH子进程62704；子PID会随重连变化，不应硬编码。
- 新监督器fcntl锁防重复；每20秒HTTP探测 /rope-mask0/，10秒超时，连续3次失败才重建自己拥有的SSH连接；保留SSH keepalive。已实际发生自动重连并恢复，最后连续3轮页面HTTP200约0.65–1.61秒，根页面与progress.json亦200。
- 本机状态 outputs/scaling-preview-health.json、日志 outputs/scaling-preview-health.log；检查状态时间戳和stage=healthy。不要再启动旧90448监督器或额外重复8884转发。
- 报告路径及原Online/mask80对照均保留；无训练、评测、远端服务器重启或实验配置修改。现有355M mask0监控goal继续active。
- 取证 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/preview_tunnel_repair_20260922T074549Z/local_qa.json；QA来自真实HTTP与进程状态，不声称浏览器截图。07:41UTC进度58项正式FVD，此处只验证预览，不把58项作为全部结果独立审计结论。



## mask0报告Online新增mask80%首层全patch CA历史对照（2026-09-22T07:24:48.122380+00:00）

- 用户明确要求在当前Online图加入“355M首层全patch CA · 原RoPE · mask80%”。来源是已完成补参数首层CA实验 first_layer_cross_ema_matched_20260921T093440Z 的355m_cross_attention__online，非原始每层CA、非同patchCA。
- 已验证参数354,975,424、CA仅block0且CA→SA→FFN；冻结训练attention.py确认CA Q/K无RoPE、全context attention。规范化显式global/false默认和输出路径后，与本轮原CA组配置差异为训练mask .8/.0。
- 20个真实Online分数（500..10000，每500）全部核对arm/step/checkpoint/protocol/source/2048cases与eval_manifest SHA。两报告fvd/protocol.json逐字段一致，评测均不mask、CFG1/144帧/50Euler/batch128。
- 页面Online新增第五条粉色虚线/空心点，EMA保持本轮四条。历史reference不进入本轮progress.scores/160计数、任务表、loss或评测队列；没有恢复旧训练/服务。
- 历史20点嵌入index.html的application/json节点mask80-reference-data，避免既有预览路由对额外JSON返回404；原始带provenance记录另存报告mask80-online-reference.json，不依赖该文件HTTP路由。
- 链接 http://127.0.0.1:8884/rope-mask0/?weight=online&scale=loglog&compare=mask80#fvd-results 。须重新打开/刷新HTML获得新脚本，原有30秒数据刷新不会替换旧HTML脚本。
- 证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/mask80_online_reference_20260922T071935Z：原始20个score路径/SHA、完整index/HANDOFF preimage、修改SHA、embed_fix与live_check。liveHTTP200+隔离NodeVM/DOM stub验证Online双log/linear均45点（本轮25+历史20），EMA25点、无历史，原loss/任务表/计数未变；不是浏览器截图。
- 监控goal仍active，仅训练本轮四组mask0到10k并评测160分数。历史对照是用户额外要求的可视化，不扩大训练/评测范围。



## mask0 同3000step FVD完整（2026-09-22T07:06:28.548731+00:00）

- 3000step Online / EMA：全patch+RoPE423.942433 /121.457952；同patch+RoPE274.061101 /128.548267；entry concat416.925257 /126.285239；全patch原CA RoPE375.756814 /197.397476。
- 前三组EMA目前约121–129，不能凭这一档的小差距宣称最终优劣；Online存在波动。已通知用户同3000step完整比较，不重复通知此档。
- 全部48score元数据、checkpoint对应、协议/source、manifest SHA、H200/2048cases/CFG1/144frames均核对；此审计不宣称独立重算全部48的feature统计。此前CP3000四组EMA与8rank恢复状态审计已通过。
- 当前完整CP对应50项评测，已有48项、其余队列覆盖，无漏项或重复。3500新评测自动推进。
- 最后3000 dev1 worker803770已退出；平台Pod清理与当前新评测证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/complete3000_review_20260922T070628Z/review.json。训练仍未10k；goal active，继续到四组10k和160FVD并最终收尾。



## mask0 四组3000step checkpoint/EMA恢复状态核验（2026-09-22T06:52:57.290366+00:00）

- 四组3000 fullCP均通过现有只读CPU审计：EMA updates=3000、decay=.99、shadow FP32、names/shapes/参数数量一致，weights/optimizer所有DCP storage bounds与8rank trainer scheduler/RNG状态通过。
- 四组8rank数据游标一致为epoch1/batch818，与2182 optimizer steps/epoch相符；没有声称执行完整权重数值扫描或GPU恢复。
- 审计已正常退出，无遗留watcher。证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/production_step3000_ema_audit.json。
- 当前已提交42项正式FVD；6项待测，完整CP应有48项均被score/queue覆盖，未发现queue或active重复。
- 全patch+RoPE3000 Online423.942433 / EMA121.457952；其余三组3000评测仍在推进。两项已完成global3000采样Pod此前确认已释放。
- 实际平台任务/本地worker与当前训练进度证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/checkpoint3000_review_20260922T065257Z/review.json。目标仍是4×10000step及160FVD，goal active；不得恢复旧20B/旧消融。



## mask0 同2500step对比完整（2026-09-22T06:41:26.834815+00:00）

- 四组仍Running且loss有限；06:40平台核验全patch+RoPE的3000 Online/EMA新任务Running（stormyxie-sc-r0-fvd-0922063828 / stormyxie-sc-r0-fvd-0922063832）。前一档4个远端采样Pod已全部释放，最后dev1 noRoPE Online2500 worker788635亦已退出。
- 2500 FVD Online / EMA：全patch+RoPE287.311869 /144.716717；同patch+RoPE264.423810 /152.490894；入口concat519.312829 /131.580959；全patch原CA RoPE312.990589 /268.903270。早期同step结果，不当作最终排序。
- 前40score的arm/step/权重、checkpoint路径、source/protocol、manifest SHA、H200、2048 real/fake、CFG1/144帧/50Euler/batch128均核对通过。
- entry EMA2500完整2048case/hash/seed/features独立重算FVD131.580959079101，reported131.580959082857，误差3.7553e-9。
- 证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/checkpoint2500_review_20260922T064126Z/review.json。仍需四组10k +160FVD +最终收尾；goal active。双log报告/rope-mask0/继续自动刷新。



## 355M mask0 FVD 双 log 显示已上线（2026-09-22T06:37:21.819060+00:00）

- 用户要求当前 /rope-mask0/ 的 FVD 也用双 log；现默认 Training step / FVD 都采用 log10 坐标，刻度显示原始值，保留“线性”按钮。
- Online / EMA 两种权重均支持双 log，URL ?weight=ema&scale=loglog#fvd-results 可直接打开；切换坐标和权重保留在 URL，30秒自动刷新沿用当前选择。
- 训练 loss 图保持原有线性坐标与50步滑动平均。没有改训练、采样、FVD数值、冻结配置或进程。
- 只编辑报告 index.html。证据 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/loglog_chart_20260922T063559Z：完整index/HANDOFF preimage、SHA、diff、live_check.json。
- live HTTP200 + isolated NodeVM/DOM stub验证Online/EMA × linear/loglog全部真实分数坐标、标签、点数、URL及刷新通过，loss不受影响；不声称真实浏览器截图。
- 持续监控goal保持active：4组训练尚未10k，未完成160FVD；此前被停止的20B/旧实验仍不恢复。



## mask0监控：四组CP2000/EMA与Online-EMA差异核验通过（2026-09-22T06:25:46.708095+00:00）

- 实际平台确认四个训练task Running，日志推进且loss有限；globalRoPE2500 Online远端任务stormyxie-sc-r0-fvd-0922061444与EMA dev1 worker784530在06:20检查时实际存活、各1024/2048，非卡死。下一次观察应查其当前状态，不将旧PID/任务完成误判为故障。
- 四组2000 full checkpoint已核验：EMAupdates=2000、decay0.99、FP32shadow参数名/shape一致，8rank trainer RNG/scheduler/cursor(0,2000)，DCPstorage bounds通过。证据production_step2000_ema_audit.json；未进行全部权重数值扫描/GPUrestore。
- 500/1000/1500/2000共32score的arm、weights、source/protocol、2048 real/fake等元数据核对通过。globalCA+RoPE@2000 Online790.912389 / EMA197.115452，分别完整独立重算，差2.256e-9 /1.631e-9；各2048case/hash/seed正确，未发现评测混用。
- 2k其余FVD Online/EMA：samepatch+RoPE240.306807/210.875655；entry448.269922/212.673369；globalCA无CA-RoPE702.732257/433.395656。只作为早期同step结果，不宣称收敛排序。
- 本轮review：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/monitor_review_20260922T062546Z/review.json。continuous_monitor_state.json记录已通知2000档EMA差异，之后不重复汇报同一结果或无变化状态。goal保持active，目标4×10000step和160FVD。


## 新活动goal：持续监控355M mask0四组（用户明确要求，2026-09-22T06:19:20.249989+00:00）

- 原mask80 scaling goal已完成；当前新goal仅覆盖355m_global_ca_rope / 355m_same_patch_ca_rope / 355m_entry_concat / 355m_global_ca_no_rope，不重新开启20B和旧消融。
- 目标：保持当前冻结配置，四组各到10000step，每500 fullCP，Online+EMA0.99各固定2048case/CFG1/144帧FVD，共160分数。独立H200和空闲dev1评测；去重排队，完成采样及时释放资源。
- 当前直接平台API和Pod检查四组均Running；全训练loss记录有限。06:17:36UTC步数2575/2396/2374/2328。截至2000step的四组Online+EMA已齐，共32项，2500档已开始采样。
- 远端monitor777265、extras750057、local757461均已核验/proc活着；现有60秒报告监控与约30秒额外评测调度继续使用，无额外重复控制器。
- 只在实质进展、异常、需要决策或完成时通知。不得将观察超时/日志批间静默当作任务失败；先核查实际任务和进程，不盲目重启/重复提交。
- 若平台失败/抢占，先读实际终态、错误日志和完整checkpoint，再决定恢复方式；保留现有冻结代码/配置/EMA/RNG/loader状态，不自行从头复跑或换配置。
- 报告 http://127.0.0.1:8884/rope-mask0/，包含FVD Online/EMA和独立Online训练loss曲线。
- 基线与监控约束：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/continuous_monitor_20260922T061920Z/baseline.json；/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/continuous_monitoring_policy.json。policy用于goal交接说明，未声称它是新增的可执行调度器。


## 原10组mask80 scaling已全部收尾；新mask0四组继续（2026-09-22T06:14:19.278967+00:00）

- 原goal的用户最终范围已完成：8组125M/355M/775M/1.56B各到10000step；两组20B按用户明确要求停止，最后完整CP分别concat6000/cross5500。**不得为凑原计划200分数而恢复20B训练**。
- 全部183个保留CP均完成固定2048cases / CFG1 / 144generated frames FVD，原计划200另标。最终20B cross5500 FVD=113.565387549280，独立公式重算差1.986e-09。
- 最后任务stormyxie-sc-fvdx-0922034755和Pod已释放；两处H200集群对原277个task身份当前完整清单均无残留。旧monitor755921、extras348487已退出；原extras control disabled、cap0。watch780011完成CPU核验后已结束。原monitor --once刷新最终report后退出。
- 报告http://127.0.0.1:8884/：183/183，默认同5500step10/10对比；标准/LOWESS/OLS图都更新且核对。CSV/所有拟合输入一致，OLS独立重算通过。8个早期score汇总缺hardware字段的case级16384条记录已全部确认H200，score文件未被改写。
- 完整requirement-by-requirement完成审计：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/final_original_closure_20260922T060805Z/completion_audit.json；报告completion.json引用该审计。
- 保留新355M mask0四组服务：monitor777265、extras750057、local757461，均核验/proc在线；共享预览server750580及Mac8884隧道继续服务。此处完成结论不包含尚在跑的新mask0四组。
- 证据界限：没有宣称重新哈希全部TB级权重载荷或新GPU全量restore；报告验证为liveHTTP + NodeVM DOM stub，无真实浏览器截图。


## 最后FVD的独立CPU核验已接上自动等待（2026-09-22T05:58:21.773418+00:00）

- 守候PID 780011，源码 /threed-code/stormyxie/vforge_jobs/scaling_fa4_20260920T134440Z/wait_for_final_original_fvd.py，状态/日志 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/final_fvd_watch_20260922T055743Z。
- 目标仅20b_cross_attention@5500 / stormyxie-sc-fvdx-0922034755；直接平台确认Running，当前1792/2048。
- 该守候用flock防重复，每约60秒检查已存在任务，score.json完整落盘后调用既有audit_fvd_score.py 20b_cross_attention 5500做CPU独立2048case核验。没有新GPU任务、没有修改采样、不会重启worker。
- 后续先检查 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/final_original_fvd_watch.json 和对应/proc、status.json；不要重复起watcher/auditor。
- independent_audit_finished 后仍需确认最后任务/pods删除、刷新原始183/183报告、终止旧extras/monitor并保留共享server与新mask0服务，最后完成goal审计。
- timeout/observation_deadline只是观察期限，不代表评测失败；必须再次查同一个实际任务，不能据此重启。



## 原 scaling 逐项完成核对已建立，仍等待最后256cases（2026-09-22T05:54:41.752472+00:00）

- requirement audit：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/goal_requirement_audit_20260922T055436Z/audit.json，complete=false。核对10组当前配置SHA、原冻结模型/CP参数数量、2:1分配、latent/no-zprev/mask0.8、FA4无fallback、8GPU local8、每500fullCP、8组真实10k完整记录、两组20B用户停止、固定源视频隔离2048case和报告曲线，均已检查对应证据。
- 初始H200 FA4 sparse probe的guard失败未算作通过；引用后续h200_fa4_sparse_probe.json的实际前后向通过证据。
- 唯一分数缺项仍为20b_cross_attention@5500。平台Pod直接确认 stormyxie-sc-fvdx-0922034755-worker-0 在hyperpod-i-0c4861ab06cfa7bc2 Running，8H200；目前1792/2048。
- 最近6次256case提交间隔稳定约1017秒，最后1792case批于05:48:39UTC写入，预计最后一批约06:05:37UTC（14:05:37上海），此为估计。禁止因批间日志无新增而重启。
- 原始source/配置/训练和FVD任务未修改；监控继续。最后score完整落盘后必须独立audit_fvd_score、确认task/pod清理、刷新183/183最终报告、收束旧原始控制器，再考虑goal complete。保留新mask0实验和共享预览服务。



## 原 scaling 报告说明与曲线一致性核对；最后FVD已1792/2048（2026-09-22T05:50:34.462324+00:00）

- 本goal turn有实际进展：修正原始报告index.html和README.md中过期的“20B训练继续”“首个production分数待完成”“常驻评测节点”等陈述，并删除HTTP404的旧batch1历史链接（历史文件未删）。没有修改训练/评测协议、runtime、配置或提交新任务。
- 原始10组状态仍为8组10k完成、2组20B用户主动停止。评测范围183个已保留CP，原计划200另列；不得重启20B训练。
- 20b_cross_attention@5500 / stormyxie-sc-fvdx-0922034755 在2026-09-22T05:49:57.111664+00:00通过平台API确认Running，独占8H200，已写1792/2048 case；score尚未写。需要等待最后一批，不能因日志间隔长而重启。
- 当前182个正式score与CSV逐项一致（2048real/fake、400维、ddof1、固定CFG1/144帧/50Euler/protocol）；LOWESS与OLS全部20条输入序列与真实分数一致；OLS系数独立重算通过；拟合不外推。6份SVG XML有效，5份页面PNG HTTP200。
- 当前十组同step完整对比上限5000；页面默认5000，手选10000正确显示8/10且刷新保留选择。live HTTP + fresh Node VM/DOM stub验证，不声称真实浏览器截图。
- 验证/文件preimage/hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/report_scope_closure_20260922T054756Z。
- 原始monitor PID755921、extras PID348487继续运行；完成最后score后独立运行audit_fvd_score.py 20b_cross_attention 5500，确认任务/Pods释放，收束原始183/183并停用旧extras。保留共享8884 server PID750580和新mask0实验服务（monitor777265、extras750057、local757461）。



## 训练 loss 可视化已上线（2026-09-22T05:45:36.959438+00:00）

用户要求在当前 /rope-mask0/ 报告加训练 loss；已保留原 FVD 图，新增独立 Online training-loss panel：
http://127.0.0.1:8884/rope-mask0/?weight=ema#training-loss
- 四组同色，浅色每步全局 loss，实线 trailing 50-step mean；开头不足50步取已有步平均。
- 全部step / 最近500step可切换；FVD Online/EMA按钮只切FVD，EMA没有单独训练loss。
- /threed-code/stormyxie/vforge_jobs/condition_rope_mask0_355m_20260922T035234Z/monitor.py 从各 train_steps.jsonl 全量读取 event=step 的 loss_global（缺失时回退loss），将零基step+1转换为完成optimizer步数。忽略写入中的末行；resume回退时替换该步及以后的旧轨迹。
- history写入现有 progress.json.training_loss；页面30秒拉取，monitor60秒刷新。仅CPU读日志，本次全量读取约0.195秒。
- monitor旧PID757460已退出，新PID777265，services.json已更新；extras/local/server/训练和评测配置均未改。
- 证据：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/validation/loss_chart_20260922T054235Z。全量原始日志逐点核对、连续步号/有限数检查、全部50步平均独立核算、全量/最近范围回切、FVD Online/EMA点数和loss独立性检查通过。
- 验证方式：live HTTP 200 + fresh Node VM/DOM stub；未声称真实浏览器截图验证。
- 初次source mount atomic rename不支持（ENOSYS），未改动原文件；保留完整preimage并compare-before-write后正常写入。
- 同期原始scaling retention审计完成：183 retained checkpoints / 182 valid FVD，唯一缺项20b_cross_attention@5500；
  /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/retention_and_coverage_20260922T053805Z/audit.json。所有10个原始训练任务和pods确认已释放；仍不得重启用户停止的20B训练。


## 最新生效：355M mask0 四组 CA / channel 消融（2026-09-22T04:17:16.942510+00:00）

用户要求三组355M mask0，后追加“首层全patchCA、CA RoPE保持原样”第四组；四组均 fresh step0。
入口 channel 采用已有 entry concat；不新增 PrefixExpander / 逐 block channel concat。
用户明确停止两个20B训练，该决定覆盖旧goal中20B继续到10k的要求，**不得自动恢复这两个训练**。

| Arm | task（tenc-awsusw2/gwm，各8H200） | 平台状态 | 已完成step |
| --- | --- | --- | ---: |
| 355m_global_ca_rope | stormyxie-r0-355-gr-0922041056 | Running | 131 |
| 355m_same_patch_ca_rope | stormyxie-r0-355-sr-0922041257 | Running | 82 |
| 355m_entry_concat | stormyxie-r0-355-cat-0922041413 | Running | 60 |
| 355m_global_ca_no_rope | stormyxie-r0-355-gn-0922041623 | Running | 20 |

共同配置：VideoRAE3D latent / no-zprev / mask ratio0 / local8 global64clips / 10states，
NativeFSDP2 / gradient checkpointing / trainable ViT+DiT FA4且禁止fallback / EMA0.99。
Frozen codec维持原SDPA。max10000，每500 fullDCP且全部保留；独立H200节点逐checkpoint
评测 online+EMA（目标160分数），固定heldout2048 / CFG1 / 16观察+144生成帧 / 50Euler；
沿用原bank/seed/GT/detector协议。355M FVD batch128/卡。不得在正式训练节点评测。

- 全patch+新RoPE / 同patch+新RoPE / 全patch+旧RoPE：354,975,424参数；ViT239,433,728+DiT115,541,696，
  DiT768×10，FFNhidden3008，只有block0 CA→SA→FFN。
- entry concat：355,083,936参数，DiTFFNhidden3064，无CA；与CA差约0.031%，其余主宽度/深度相同。
- 新公共字段 model.extras.world.cross_attention_3d_rope，默认false；只对本轮两组设true。
  CA Q/K在qk_norm后施加3DRoPE，V不旋转；沿用现有feat_rope，未新增参数/buffer。
  condition时间坐标0..3，下一state target4..7；h,w用原始latent网格。
  同patchCA固定(h,w)，可关注全部4个condition时间平面；全patchCA覆盖全部1024。
  第四组保持CA Q/K无RoPE；所有组的ViT/DiT self-attention原有RoPE仍保留。
- Entry concat张量：x[B,64,4,16,16]→[B,1024,64]；ViT[B,1024,1024]经LN+Linear→[B,1024,768]；
  按各state内部同(t,h,w)拼接→[B,1024,832]→Linear832→768→SiLU→Linear768→768。
  训练B=8×9=72个next-state pairs；这是目标视频latent时间维，不是diffusion timestep。
  mask0无mask产生的零槽，原condition_dropout0.1独立保留。
- canonical source仍 /efs/stormyxie/vforge，xj_dev dirty，未commit/push；老runtime/config/protocol未改。
  本轮冻结runtime /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z/runtime
  source digest a5c67a5a9beaa9493dd5e3d18cc9294f3a59db79c4ca343d64d7458c8b8b4783，1011files。完整meta参数实数核对通过。
- 验证：CPU定向46项（旧29+新增17）通过；6个改动PY文件ruff通过。
  H200 FA4独立complex-RoPE+FP32手工attention参考，dense/sparse×global/same四项前后向通过；
  SDPA被禁止调用，FA4实际调用7次；输出相对L2 0.00145–0.00385，梯度0.00251–0.00466。
  四组实际8H200/local8/mask0各train2steps fullDCP、resume到step3全部通过，
  loss有限，EMA更新数1/2/3正确；这不是声称不同attention结构的训练轨迹相等。
- 额外dev1推理短测：global+RoPE online和same+RoPE EMA，各16cases；
  只使用smoke ckpt3，输出validation/eval_smoke，不计正式FVD。
  两组各16cases均通过，online/EMA严格加载、CFG1/144生成帧、I3D有限400D特征核对完成；
  证据 validation/eval_smoke_PASS.json。dev1验证进程正常退出，GPU已全部空闲。
- 20B concat stormyxie-sc-20b-cat-0920143127、CA stormyxie-sc-20b-xa-0920143629已删除，
  最新直接pod查询均0；最后完整CP分别6000/5500，保留全部历史checkpoint。
  原monitor已识别USER_STOPPED.json；保留现有完整CP的FVD，不再排新20B训练。
  未保存的日志步6279/5524不能当作可恢复checkpoint。
- jobs /threed-code/stormyxie/vforge_jobs/condition_rope_mask0_355m_20260922T035234Z
  report /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/rope_mask0_355m_20260922T035234Z
  output /efs/stormyxie/outputs/vforge/condition_rope_mask0_355m_20260922T035234Z
  pointer /efs/stormyxie/work/rope_mask0_355m_active.json。
  证据：validation/launch_verified.json、parameter_counts.json、fa4_rope.json、
  gpu_smoke_PASS.json、native_audit_*.json、control_updates/*。
- 全四组训练日志最终实查 2026-09-22T04:18:49.488104+00:00：完成步 131/82/60/20，全部loss有限且EMAcount等于完成步。
  证据 validation/all_four_training_verified.json；初次launch_verified.json较早，第四组当时尚在初始化。
- 服务：本轮monitor PID757460（60秒扫描），FVDextras PID750057（最多4个独立8H200节点）；
  单checkpoint/权重独立lease防重，完成/失败任务及时释放。旧matched/same-spatial已完成的controller不恢复。
  原scaling monitor PID755921；8884 server PID750580；本地8884自动重连supervisor PID90448保留。
  页面 http://127.0.0.1:8884/rope-mask0/，旧三个报告路由保留。实时状态以任务和日志为准。










### 1000-step全部8项FVD已齐、globalRoPE首个CP1500（2026-09-22T05:30:49.473590+00:00）

本turn为progress：新增entry online844.5203497243037、原RoPE global EMA1328.0361438629243、
原RoPE global online1252.6147042072107三项完整2048case审计通过；step1000共8/8核验完成。
汇总 validation/complete1000_watch_20260922T052435Z/result.json：

| 条件方式 | online@1000 | EMA@1000 |
| --- | ---: | ---: |
| global CA + 3D RoPE | 648.128338952188 | 773.176868422528 |
| same-patch CA + 3D RoPE | 510.133931959078 | 544.588294242603 |
| entry concat | 844.520349724304 | 681.764084252623 |
| global CA原实现（CA Q/K无RoPE） | 1252.614704207211 | 1328.036143862924 |

所有数值固定355M/mask0/CFG1/2048heldout/144frames；不要与旧mask80实验混算。
原实现的ViT/DiT self-attention仍有原RoPE。仅描述当前训练点，不声称最终收敛排序。
连同step500，报告16/160正式分数全部完成独立重算。

- globalRoPE的CP1500已完成单组metadata audit：EMAcount1500/decay0.99、
  完整权重/optimizer/trainer存储边界、FP32shadow names/shapes、8rank cursor(0,1500)通过；
  证据同proof目录checkpoint1500_global_first.json。该检查没有覆盖其余三组未到达的CP1500。
- globalRoPE@1500已被原controller自动分配：remote EMA stormyxie-sc-r0-fvd-0922052615；
  dev1 online stormyxie-r0-local-0922052626，worker772140，/proc命令确认live。
  完成1000-step的两项原RoPE remote task0922051106/0922051725已删、直接pod查询均0。
- 前台watch session79431、单组CP审计45624已正常exit0，不能再当作live handle。
  05:29:23UTC观察：四训练报告step1576/1455/1431/1387、仍Running；只有global组已有完整1500。
  original20B CA@5500 stormyxie-sc-fvdx-0922034755直接task+pod仍Running，实际1280/2048cases；
  此模型batch间隔长，不据此误判hang或重复启动。20B训练继续保持用户停止。
  完整现场证据同proof目录cleanup_and_live.json；goal active。

### 1000-step追加4项FVD审计（2026-09-22T05:22:38.213524+00:00）

本turn为progress，新增4项完整2048case独立重算全部通过：
globalRoPE online648.1283389521877（误差1.57e-9）；
sameRoPE online510.1339319590775（1.50e-9）；
sameRoPE EMA544.5882942426028（5.28e-10）；
entry concat EMA681.7640842526225（1.21e-9）。
加此前globalRoPE EMA773.1768684225281，step1000已核验5/8；step500的8/8仍全部保留。
证据 validation/new1000_results_20260922T051835Z/audits.json 和各fvd_score_<alias>_01000.json。
CPU批量审计session73451已正常exit0，不要重复poll；没有修改训练或采样参数/冻结source。

05:21:18UTC报告/进程核验：
- 四训练继续Running，step1408/1298/1271/1229，完整CP500与1000。
- global原RoPE EMA@1000 stormyxie-sc-r0-fvd-0922051106，报告1024cases；
  global原RoPE online@1000 stormyxie-sc-r0-fvd-0922051725，正在首batch；
  dev1 entry online@1000 stormyxie-r0-local-0922051640，worker769492实际/proc命令确认live，1024cases。
  这三项尚未完成，不要把pending当缺失/失败而重复提交。
- 原scaling CA@5500 stormyxie-sc-fvdx-0922034755直接Koala task API仍Running，
  实际1280/2048特征，20B训练继续停止，原182/183未到终点。
  证据同proof目录live_after_audit.json；goal active。

### 首轮8项FVD已全齐、四组CP1000通过、原10组scope复核（2026-09-22T05:16:19.994806+00:00）

本turn为progress：entry concat online@500与globalRoPE EMA@1000全2048case独立重算通过；
四组完整1000-step metadata/EMA audit均已完成；并复核原scaling全部10组真实参数metadata/配置/heldout隔离。

- step500四组×online/EMA共8项均已完成并审计。entry concat online=2374.425317025448，
  EMA=3661.6048556303176；其余6分数见上一段或
  validation/step500_matrix_and_1000_watch_20260922T050834Z/complete_first500_comparison.json。
  所有8项固定2048case/CFG1/144frames/mask0，共享reference bank；不是小样本demo。
- 首个step1000结果global CA+3DRoPE EMA=773.1768684225281，独立重算差2.85263e-9；
  相同EMA模型step500为3092.3229871182366。online@1000已另行提交score，
  本turn结束时尚未执行其独立CPU重算，下一turn优先审计，不要将其与EMA混用。
- 1000-step checkpoint audit complete=true，四组EMAcount1000/decay0.99、参数shape/FP32shadows、
  fullDCP组件与存储边界、8rank scheduler/RNG/cursor均通过。
  Auditor764089已完成、无有效cmdline；/proc终态记录在该proof目录checkpoint1000_auditor_final_proc.txt，
  不要因pid文件仍在而重复启动/当作live工作。前台observer session68542已正常exit0。
- entry online@500 task stormyxie-sc-r0-fvd-0922050126平台Succeeded后被controller自动删除；
  05:14:45UTC直接task查询NotFound、pods0，资源释放。该proof目录post_completion_live.json保存证据。
  四个训练仍推进，最后日志/报告step1262/1158/1133/1090，均有500与1000完整checkpoint。
  dev1当前entry EMA@1000 task stormyxie-r0-local-0922050946 worker767246。
  其余1000-step online/EMA继续由独立H200 remote controller调度，维持已有lease和cap4。
- 原scaling所有10组重新用冻结原runtime做meta构建，并逐参数对应各自实际CP500 DCP shape：
  125M/355M/775M/1.56B/20B的入口concat与全层CA参数目标误差<0.024%，ViT:DiT约1.9815–2.0713；
  latent/no-zprev/mask0.8/FA4禁止fallback/8卡local8/max10k/每500且全保留配置通过，
  实际production manifest/backend/world8/DPshard8与配置一致。全层CA覆盖全部block，concat无CA。
  这是元数据/结构审核，不是重新训练或全payload数值扫描。
  证据原R0/validation/original_scope_precompletion_20260922T051346Z/audit.json。
- 原heldout split文件hash保持冻结值：train139670 clips/3796源视频组，eval2048 clips/2048独立源视频组，
  源视频组与文件路径交集均0；复用了纯group函数，没有执行make_split.py或改写数据。
  原10组已有共同step5000分数身份/manifest hash/同FVD协议也重新核对，保存于上述audit。
- 原20B CA@5500 task stormyxie-sc-fvdx-0922034755直接task+pod仍Running，05:14:45已1280/2048cases；
  原scaling仍182/183，等待最后score。两个20B训练继续保持用户明确停止，不得恢复。
- preview /rope-mask0/已经自动推进到10/160（含新未独立重算的globalRoPEonline@1000），
  旧scaling/same-spatial/first-cross报告保留。未改冻结模型、训练/eval配置或协议。
  本段完整前像/hash在上述proof目录，goal保持active。

### 四组EMA@500已齐、7项正式FVD核验（2026-09-22T05:06:55.011242+00:00）

本turn为progress：新增5项2048case完整审计（此前2项已验证），目前step500的4个EMA与3个online齐全。
同协议：355M/mask0/CFG1/2048heldout/144生成帧，固定case噪声身份；EMA0.99。
各项独立矩阵平方根重算误差<=2.5e-9。只描述早期单checkpoint，不据此宣布结构优劣。

| 首层条件方式 | online FVD@500 | EMA FVD@500 |
| --- | ---: | ---: |
| 全patch CA + 3D RoPE | 2096.426129853844 | 3092.322987118237 |
| 同patch CA + 3D RoPE | 1279.993363357355 | 2811.259050646026 |
| 入口concat | 尚在评测 | 3661.604855630318 |
| 全patchCA原RoPE | 1970.983905256391 | 3005.873294967830 |

- 全部audit证据 validation/fvd_score_<alias>_00500.json；汇总
  validation/first500_results_watch_20260922T050037Z/result.json、comparison.json。
  本turn观察session66618已正常exit0，不要再次poll或当作运行进程。
- 已完成远端采样任务0922045136/0922045213/0922045250/0922045326均由既有controller自动删除；
  直接task API返回NotFound，各自pod查询0，已释放资源。前缀均stormyxie-sc-r0-fvd-；
  完整身份与观察在该proof目录completed_task_cleanup.json、same_patch_task_cleanup.json。
- entry online@500继续由stormyxie-sc-r0-fvd-0922050126采样。
  globalRoPE@1000已经自动分配两套权重：online task stormyxie-sc-r0-fvd-0922050240；
  dev1 EMA task stormyxie-r0-local-0922050202，worker764723，既有lease防重复。
  这是新checkpoint，不是重复采样@500。旧first500 local worker762368已结束。
- 新CPU checkpoint1000 auditor PID764089已启动、/proc身份核验live；
  命令audit_checkpoint_ema.py --step 1000 --watch，进程记录
  validation/production_step1000_ema_auditor_process.json，日志production_step1000_ema_audit.log。
  当前已完成['355m_global_ca_rope', '355m_same_patch_ca_rope']，complete=False；
  其余等待各自完整1000-step提交后自动检查。核验EMA count/FP32shadows/所有模型参数shape、
  fullDCP组件与存储边界/8rank trainer状态，非全weight payload数值扫描或GPU恢复试验。
- 直接训练日志快照仍前进，最后1098/998/975/933step，所有近71-72条loss有限、日志年龄<10s；
  证据该proof目录training_live_tail.json。此为时间快照，实时状态以当前日志/API为准。
- 报告http://127.0.0.1:8884/rope-mask0/自动更新到7/160，EMA4曲线点、online3点；
  实际页面脚本已在独立Node VM+DOM stub运行，权重切换/URL参数/点数量均通过，
  证据report_script_check.json。不是真实浏览器布局检查：browser-act及uv未安装，没有装新环境。
  初次复用测试上下文产生点数不符；新的隔离上下文通过，页面源码没有修改。
- 原20B训练保持用户停止；原20B CA@5500仍是原scaling最后未完成项。
  goal active，旧runtime/config/FVD协议均未修改，服务与已有monitor继续运行。

### 第二个正式FVD与远端评测入场（2026-09-22T04:59:04.641912+00:00）

- 本turn完成新的2048case完整审计，属于progress：355m_global_ca_no_rope__ema @500，
  FVD=3005.873294967830；独立矩阵平方根重算差2.433e-09。
  核对原reference bank、全部case sidecar/seed/GT/detector/checkpoint/config/协议，
  finite400D features，2048个独立case seed和generated video hash。
  证据 validation/fvd_score_355m_global_ca_no_rope__ema_00500.json。
- local task stormyxie-r0-local-0922044633 worker760555已正常exit0，2048case完成；
  总耗时484.34秒（包含load/generation/I3D/FVD，不含排队）。
  首个global+RoPE EMA@500=3092.3229871182366；只比较该早期checkpoint不足以判断RoPE优劣。
- 04:57:52UTC开始的独立Koala task+pod/API核验：四个训练均Running，
  已完成step934/845/820/777（globalRoPE/sameRoPE/entry/globalOriginal），日志年龄低于8秒。
  四个远端FVD已分配实际H2002D节点并运行：
  stormyxie-sc-r0-fvd-0922045136（globalOriginal online@500）、
  0922045213（globalRoPE online@500）、0922045250（sameRoPE EMA@500）、
  0922045326（sameRoPE online@500）。
  前两项已产生1024case，同patchEMA现场416case，后者当时尚在第1batch；
  节点集合与四个训练节点无交集。早先Unschedulable已由既有controller冷却后恢复，无手动重启。
- dev1当前local目标entry concat EMA@500，task stormyxie-r0-local-0922045438，
  worker762368的/proc cmdline确认live；现场552cases，按既有去重lease运行。
  所有活跃身份/物理节点/直接API证据 validation/live_node_separation_20260922T045752Z/live.json。
  这些计数是带时间快照，不替代当前API或日志。
- 原scaling唯一剩余20B CA@5500 task stormyxie-sc-fvdx-0922034755，
  直接API和pod均Running，生成已推进到1024/2048；不要把大batch间隔当作hang。
  两个20B训练停止决定不变，不得恢复。原182/183覆盖仍待最后score提交与独立审计。
- 04:58:04UTC本地8884 /rope-mask0/?weight=ema HTTP200，报告已自动收录这两个分数。
  monitor757460 / extras750057 / local757461以及共享8884服务保持运行；没有调整冻结runtime/config/protocol。
  本段完整前像与SHA记录 validation/second_formal_fvd_handoff_20260922T045904Z/，goal保持active。

### 四组首个500-step与首次正式FVD（2026-09-22T04:51:36.865732+00:00）

- 四组均已提交完整500-step DCP；validation/production_step500_ema_audit.json complete=true。
  核对EMAupdates500/decay0.99、全部trainable names/shapes与FP32shadows、
  8rank trainer/RNG/data cursor、所有DCP weights/optimizer存储边界、精确配置/FA4/mask0。
  对应CPU auditor PID757939已自然完成；不是全payload数值扫描或500-step GPU恢复试验。
- validation/config_and_input_parity_20260922T042803Z.json确认四组仅存在声明的
  condition/RoPE/scope/FFN补参数差异；共同前198步rank0样本身份一致、loss有限、EMA计数正确。
  该证据不代表全rank像素/噪声逐元素一致，也不声称各结构优化轨迹一致。
- 首个正式FVD：355m_global_ca_rope__ema @500，2048cases，
  FVD=3092.3229871182366；独立矩阵平方根重算差1.6943886294029653e-09，
  2048case/seed/GT/detector/checkpoint/有限400D特征审计通过。
  validation/fvd_score_355m_global_ca_rope__ema_00500.json；
  首个local任务stormyxie-r0-local-0922043827退出0，整轮484.16秒（含加载/采样/I3D/FVD，不含排队）。
  首分未形成四组对比，不能据此判断RoPE优劣。
- dev1独立采样controller fvd_local.py已启用，只在全部8H200空闲时领取未被lease占用的目标，
  最多同时1个8卡任务；与4个远端训练节点分开，模型/evaluator/protocol快照未改。
  PID757461；当前local目标355m_global_ca_no_rope__ema @ 500，
  worker PID760555，task stormyxie-r0-local-0922044633，直接/proc确认仍在运行。
  控制文件 RN/fvd/local/control.json；结果/receipts独立保存，不与远端重复采样。
- 本轮monitor现在PID757460，60秒扫描并把local评测纳入报告；
  remote extras PID750057，cap4独立8H200节点。
  远端首次online@500 task stormyxie-sc-r0-fvd-0922043835在H2002D明确Unschedulable
  （整机GPU/CPU/memory/EFA不足），120秒后按既有规则撤销，NotFound，无GPUworker启动。
  这是调度容量问题，不是模型失败；原目标仍在队列，冷却后重试，不改采样参数。
- 页面 http://127.0.0.1:8884/rope-mask0/?weight=ema 可直接看首个EMA分数。
  online/EMA选项现在保留在URL；local失败会在页面显示。旧标准/双log/OLS报告保留。
- first500前台观察session6183已正常结束；首次FVD观察+audit session76903也已正常结束；
  不要再poll它们或重复启动auditor。证据分别在 validation/first500_watch_20260922T0429Z/
  与 validation/first_score_watch_20260922T0446Z/。
- 原20B训练停止决定保持不变。原scaling仍182/183已保存checkpoint有正式分数，
  CA@5500 stormyxie-sc-fvdx-0922034755最后确认768/2048且仍在运行，等待完成再独立审计。
  本turn为实质进展；goal继续active。


### 最近 goal 核查：停止后的原 scaling 收尾（2026-09-22T04:24:29.270096+00:00）

上一轮为实质进展：四组355M mask0已启动并完成训练/续训/采样gate。本轮完成新的原scaling结果与checkpoint审计，未重启20B。
- 原10组已有182个正式FVD；全部已保存的完整checkpoint共183个，原规划仍为200。
  两个20B停止后，最后完整CP为concat6000 / CA5500；两者直接查询仍0Pod。
- 新增concat@6000 FVD=236.19934114251282，全部2048case/seed/GT/detector/checkpoint/config/
  有限400D特征核验通过，独立协方差矩阵平方根重算236.1993411429239，差4.11e-10。
  证据 R/validation/fvd_score_20b_entry_concat_06000.json。
- CA@5500完整DCP元数据、全部storage边界、8rank trainer/游标/调度状态、
  FA4/local8/mask80固定recipe和11份checkpoint保留检查通过；
  证据 R/validation/intermediate_checkpoint_20b_cross_attention_05500_20260922T042224Z.json。
- 唯一未完成原scaling评测：stormyxie-sc-fvdx-0922034755，tenc-awsusw2，8H200，
  直接task查询Running，采样从256推进至512/2048。下一步等此**同一任务**出score，
  核验2048case并独立重算，然后核对任务清理；不得以无新输出一段时间为由重启。
- 原monitor增加停止后的FVD覆盖计算：显式USER_STOPPED按保留checkpoint计入；
  未请求的Failed/Running仍保留20个规划目标，避免把失败误当完成。3项纯函数检查通过。
  原报告显示182/183并注明原计划200；原runtime/config/protocol均未改。
  原monitor现在PID755921，server仍750580；最终183全部完成且训练均结束/用户停止后，
  monitor可自然退出。FVDextras PID348487留待最后任务完成并释放资源后清理。
- 证据目录 R/validation/stopped_scope_report_20260922T042318Z：
  live_and_score_matrix.json核对全部182score的协议和manifest哈希，
  coverage_check.json与完整修改前image/hash。四组新mask0仍独立由自己的monitor/extras维护。
  本轮没有完成全goal声明；仍有上述正在运行的评测和新mask0持续训练/后续FVD。


最近实查与主状态表更新：2026-09-21 13:46:20 UTC。工作在dev1；先读AGENTS.md，以实际task、日志与完整checkpoint为准。
此次只更新交接记录；运行snapshot、配置、协议与任务未改，原文和校验记录已保留。

## 最近已解决：775M online@3500 FVD I/O 恢复（2026-09-21 13:22 UTC）

本节记录13:22 UTC已解决的I/O事件；最新总体状态见下方主表。长期goal仍active，Native数值snapshot、frozen evaluator body、模型配置和固定FVD协议未改。

- 旧评测stormyxie-sc-xmatch-fvd-0921123033在fvd_1510.npy.tmp→.npy的rename处FileNotFoundError，
  rank6在finally清理，其余rank等待all_reduce，随后NCCL 600秒timeout。
  旧task已Failed、删除且无Pod。诊断、SIGINT时机、原始异常及缓存证据保存在
  RM/validation/775m_online3500_wait；不可将这次恢复描述为一次API超时后的重启。
- dev1补采stormyxie-sc-xmatch-local-0921131351于13:13:51启动，13:20:18正常退出(returncode0)，
  2048cases正式出分。online FVD=313.44881984328686，同步EMA=114.90147473527486。
  wrapper PID407040已不存在，本地controller可接续下一任务。远端-r1重试只曾Unschedulable，
  已删除；不要恢复它或任何历史失败task。
- recovery_verified.json逐一核对2048case身份/seed/GT/detector/online权重/Native checkpoint/
  固定协议和有限FP32/400D特征；恢复前1515份.npy及sidecar SHA全部未变，补齐533case。
  从最终特征独立重算FVD=313.44881984328595，与正式score差约9.1e-13。
  recovery_completed.json确认成功receipt、PID退出和报告progress内分数一致。
  此验证没有宣称重新生成全部视频；全部数据仍在EFS。
- 版本化I/O helper：
  /threed-code/stormyxie/vforge_jobs/artifact_io_retry_20260921/artifact_io_v2.py，
  SHA256 976586152b0dbb19b515b06722ff636dff5721278c42067845457edf621f5f3a。
  捕获预期字节后，若rename抛FileNotFoundError，最多5秒轮询目标，逐字节完全相同才承认成功。
  7项CPU定向测试通过。J/JM的eval_entry.py导入此helper；已启动进程仍持原版本，
  不要声称所有现存worker已换v2。frozen evaluator body/protocol hash未变。
  Native异常finally清理可能等待NCCL超时的核心行为尚未修改。
- JM/fvd_extras.launch新增可选task_suffix以避免手工重试同秒名称碰撞，默认调用不变；
  常驻控制器没有重启。提交补采前仍必须核对活动receipt/local目标/lease及实际cluster。
  py-spy0.4.2仅安装在/efs/stormyxie/work/diagnostics/pyspy_20260921，没有改训练venv。
- 后续按原策略监控新完整checkpoint，独立节点评估online+EMA。不得重复评估该已恢复结果。
  /threed-code挂载不支持symlink/rename；外部小helper修改须备份并直接写入后校验，
  不得修改正在使用的不可变runtime/config/protocol。

## 当前任务与约束

- 原始10组scaling的8个小模型已完成10k/160次FVD，两组20B继续，不受这次重启影响。
- 新四档125M/355M/775M/1.56B，仅block0做CA→SA→FFN，其余SA→FFN。
  加宽每层DiT SwiGLU FFN补回CA参数，hidden取8倍数，总参数对原全层CA误差小于0.02%。
  ViT、DiT主宽度/深度/heads不变；这是等参数结构比较，FFN与计算量随之改变。
- fresh step0，无旧权重warm-start；新输出目录。EMA0.99包含ViT+DiT，FP32shadows，
  初始参数初始化，第一次optimizer update开始维护；shadow不计可训练参数。
- 每组8×H200 / NativeFSDP2 / FA4 / local8 global64 / mask80% / latent input / no-zprev。
  不用B300，不允许trainableViT/DiT SDPAfallback；frozenVideoRAE codec仍为原SDPA路径。
- runtime.max_steps=10000，每500step fullDCP；独立节点评估online与EMA，CFG1/2048held-out cases。
  dev1仅8卡空闲时采样；不在训练节点评测。原200分数+本轮160=**360个目标正式结果**。
  旧未补参数版分数归档，不替代本轮。长期goal保持active。
- 旧condition A/B/C/D/E/G/H、RGBpatch4、DuET、mask80的32卡任务保持停止。CP不改；
  当前USP是TorchUlysses，不能称MegatronCP/Ring。
- 当前源码/efs/stormyxie/vforge，xj_dev，HEAD527317858e8ab5b975fab1859a0317f53503ebdd；
  FA4/batch/首层CA/EMA仍有dirty修改，HEAD不代表完整运行代码。保留workingtree，不reset/clean。
  作者stormyxie@tencent.com，不直接合master。

## 四档参数与启动

2026-09-21 09:37UTC提交，09:38UTC直接KAP和Pod均Running，各8H200，tenc-awsusw2/gwm。
2026-09-21 09:43:05 UTC独立核验4组从step0连续训练，完成步分别72/56/83/61；全部loss/grad有限、EMAcount逐步一致。证据RM/validation/training_started_verified.json。

| 档位 | 可训练参数 | ViT / DiT | FFN hidden（原→补齐） | KAP task |
| --- | ---: | ---: | --- | --- |
| 125M | 125,044,096 | 83,457,088 / 41,587,008 | 1176 → 1776 | stormyxie-sc-125m-xmatch-0921093733 |
| 355M | 354,975,424 | 239,433,728 / 115,541,696 | 2088 → 3008 | stormyxie-sc-355m-xmatch-0921093730 |
| 775M | 775,044,480 | 520,742,208 / 254,302,272 | 3256 → 4512 | stormyxie-sc-775m-xmatch-0921093726 |
| 1560M | 1,560,089,984 | 1,041,291,776 / 518,798,208 | 3232 → 4848 | stormyxie-sc-1560m-xmatch-0921093722 |

总参数相对原全层CA：+23,552 / −54,176 / +136,256 / +233,920。
EMA维护会影响训练速度，历史全层CA/concat无EMA；online FVD按相同权重类型比较，EMA单列。

## 目录和事实来源

| 名称 | 路径 |
| --- | --- |
| J：原始10组jobs | /threed-code/stormyxie/vforge_jobs/scaling_fa4_20260920T134440Z |
| R：原始报告 | /efs/stormyxie/reports/scaling_fa4_20260920T134440Z |
| JM：当前等参数首层CA jobs | /threed-code/stormyxie/vforge_jobs/scaling_first_cross_ema_matched_20260921T093440Z |
| RM：当前报告 | /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z |
| OM：当前输出 | /efs/stormyxie/outputs/vforge/scaling_first_cross_ema_matched_20260921T093440Z |
| JE旧：已停未补参数jobs | /threed-code/stormyxie/vforge_jobs/scaling_first_cross_ema_20260921T084210Z |
| RE旧：已停未补参数报告 | /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema |

活动指针/efs/stormyxie/work/scaling_first_cross_active.json与first_cross_ema_active.json均指向JM；
first_cross_matched_prepared.json也记录本轮。原始scaling_fa4_active.json不变。
JM/runs.json=4训练run，eval_runs.json=8别名（__online/__ema）；不可加入原始J/runs.json。
JM/state.json是配置，RM/progress.json是约60秒状态；原始J/state.json存在，R/state.json不存在。
当前runtime/runtime_batch_candidate链接到RE旧不可变snapshot；source文件逐项hash相同。
仅配置/目录/外部监控报告helper改变；不要就地改任何旧/新实验共享的snapshot。

## 当前训练与结果

核查时间：2026-09-21 13:46:20 UTC。task状态来自本次直接查询；步数、loss与最新DCP来自带时间的监控快照，不代表此刻精确步数。

| 模型 | 已完成步数 | 最新完整 DCP | task 状态 | task |
| --- | ---: | ---: | --- | --- |
| 20b_entry_concat | 3897 | 3500 | Running | stormyxie-sc-20b-cat-0920143127 |
| 20b_cross_attention | 3424 | 3000 | Running | stormyxie-sc-20b-xa-0920143629 |
| 125m_cross_attention | 4816 | 4500 | Running | stormyxie-sc-125m-xmatch-0921093733 |
| 355m_cross_attention | 4808 | 4500 | Running | stormyxie-sc-355m-xmatch-0921093730 |
| 775m_cross_attention | 5102 | 5000 | Running | stormyxie-sc-775m-xmatch-0921093726 |
| 1560m_cross_attention | 3831 | 3500 | Running | stormyxie-sc-1560m-xmatch-0921093722 |

原始 campaign 173/200，本轮补参数首层 CA 66/160，合计 **239/360** 份正式结果。长期goal仍active；原两组20B及本轮四组继续到10000step，后续评测尚未完成。
原八个小模型的160份结果已完成，不重复计算。每500step读取已提交的完整checkpoint；online/EMA身份与输出目录分开。
证据：RM/validation/handoff_refresh_20260921T134620Z/verified_current_state.json；同目录保留原HANDOFF及SHA。

本轮最新已成对产出的FVD（生产正式分数，额外独立复算范围见下文；不同模型行不一定同step）：

| 模型 | step | online | EMA 0.99 |
| --- | ---: | ---: | ---: |
| 125m_cross_attention | 4500 | 365.697417 | 203.049022 |
| 355m_cross_attention | 4000 | 280.443845 | 127.189880 |
| 775m_cross_attention | 4500 | 184.185288 | 94.673597 |
| 1560m_cross_attention | 3000 | 361.255526 | 111.136876 |

新结果与资源回收证据在RM/validation的775m_step4500_pair_complete.json、125m_step4500_scores.json、125m_step4500_resources_released.json、355m_step4500_ema_score.json。
775M的5000完整DCP已入队；对应online/EMA实际Running证明见775m_step5000_eval_tasks.json及775m_step5000_online_running.json。历史Running不能作为当前自动恢复依据，继续工作仍须核对实际task/进程。

原始all-CA是SA→CA→FFN，新first-CA是CA→SA→FFN并扩大FFN；原始baseline没有EMA。在线权重与EMA分列，不能把EMA优势全部归因于结构，也不能称为等FLOPs对比。

## 当前控制器与调度

下列 PID 的完整 cmdline 已在上述核查时匹配；发送 signal 前仍须重新核验，不能仅凭历史 PID。

| Campaign | script | PID |
| --- | --- | ---: |
| 原始 | monitor.py | 348486 |
| 原始 | fvd_extras.py | 348487 |
| 本轮补参数 | monitor.py | 330373 |
| 本轮补参数 | fvd_extras.py | 339124 |
| 本轮补参数 | fvd_local.py | 330375 |

本轮 monitor 约60秒扫描完整 DCP 并排 online/EMA 队列；extras 约30秒协调实际 worker 与清理。
本轮远端 cap7 节点，原始 cap17，各任务8H200，cap不是实际用量；都须经过配额和可调度节点检查。
本轮 local 仅在 dev1 八张卡空闲时采样。当前有新评测，不得沿用旧快照称 dev1 空闲。
没有周期训练自动重启器；submit_first_cross.py只提交无receipt项，不能盲目重复。
train_entry.py 仅在本轮自己输出目录有完整DCP时自动full-resume。

原始常驻GPU池 stormyxie-sc-fvdbatch-0921074040 已主动删除并确认NotFound/无Pod，不恢复。
原始J/monitor.py按 fvd/pool_control.json primary_scheduler=false 由dev1 CPU发布完整DCP队列；
extras不再对无primary的队列预留前两个checkpoint，使用既有lease去重、按需启动独立8H200任务。
证据 R/validation/retire_idle_primary_20260921T1031Z：两项定向检查、171分数扫描、
CA3000与entry3500真实新checkpoint入队和worker启动，评测节点与6个训练节点独立。
原始20B CA3000与entry3500的FVD仍在采样，查看当前status，不能把dispatch完成当成FVD完成。

查询Task/Pod必须使用receipt.cluster；错误cluster的NotFound不能证明任务失效。
真实Pod Unschedulable才按控制器规则撤回和换集群；API/DNS观察超时本身不是模型失败。
775M、125M曾短暂observation_unavailable，复查同task/pod Running、restartCount0、日志推进，无须重启。
仅清理明确自有已完成或失败资源；不要恢复用户取消或已经被替代的旧实验。

已读脚本：extras中的35秒仅用于API/CLI查询；fvd_single.py对实际采样process.wait()不设运行时限，
其中wait(timeout=10)仅用于已发送终止信号后的清理；coordination的12小时仅用于等初始lease。
20B CA3000实际runPolicy未设置activeDeadlineSeconds。这不承诺平台不会抢占。
Native eval_manifest 在全部case结束后写入；采样期间缺少该文件或暂时0case不是失败。

20B CA3000 fullDCP约160.10GB，实测前台4.44秒、后台393.82秒（upload185.96、commit203.41）。
Native storage先逐payload SHA256再写manifest并rename完整目录；保存时训练继续。
文件写出或.metadata存在仍不等于完整checkpoint，不得手动提升incomplete目录。
证据 R/validation/retire_idle_primary_20260921T1031Z/checkpoint3000_timing.json。

[当前等参数报告](http://127.0.0.1:8884/first-cross/)；
[旧未补参数已停归档](http://127.0.0.1:8884/first-cross-unmatched/)；
[原始scaling](http://127.0.0.1:8884/)；
[历史七组condition](http://127.0.0.1:8884/condition-ablation/)。
本次四个路由均HTTP200。8884由原始J/serve.py提供，保留8882/8883与预览隧道；
大媒体留EFS，不因网页故障重启训练。

## 正式FVD与已有实现验证

固定、与训练视频分离的 **2048 cases**；同一 bank、case ID 与独立噪声流贯穿全部 checkpoint。
每段 10 states = 16 observed RGB + 144 generated RGB，256×256、16 FPS；
CFG1、每 state 50 Euler steps、seed 20260920。只对后 144 帧算指标。
I3D Kinetics400 pre-softmax 400D、FP32、resize224 bilinear
`align_corners=False, antialias=False`、covariance ddof=1、float64 Fréchet。
不得混入 smoke8/16、训练 demo、观察帧或另一个 detector 版本。

| 模型族 | 每 GPU sampling batch |
| --- | ---: |
| 125M | 256 |
| 355M / 775M | 128 |
| 1.56B | 64 |
| 20B | 32 |

同一个 case 的 autoregressive states 顺序生成，不将其当作独立 batch 样本。
批量路径保持按 case 的噪声身份；已验证的数值容差不等于任意 batch 的 bitwise parity。
正式 score 只有协议、checkpoint/manifest provenance、case 数量与有限特征通过验证才计入；
仅目录存在或某个 .json 存在不足以判断完成。

原始 campaign 的综合 protocol hash（包括 evaluator / detector / Native source 身份）：
`60d20796dd8e3a6d2b66facfde14c18ab899cd673c8a0b4494444a50bb42e225`。
当前等参数与上一轮 EMA campaign 的综合 protocol hash 为
`8fe120f902ac5a6ba5bbca90e31933d59935f03a3581da61094bbbe4bc2ead79`；
两者基础 fvd/protocol.json 按字节相同，文件 SHA256 为
`96aae9b370d92009adbbf8480f05cb3d41bb98d6ea5ece6ba81fe24574a4345c`。
综合 hash 不同来自 EMA evaluator / source 实现身份，不能把 file hash 与综合 hash 混用。
原始 batch eval body SHA256：
`9c0badd5345c380017fcc8fa0a50a438f69126b15e591aac39f3b0179aee559c`。
I3D 权重 SHA256：
`bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`。
当前 train/eval snapshot 摘要：

| Campaign | train digest | eval digest |
| --- | --- | --- |
| 原始 10 组 | `1918e80a87747155e2898543c666c8799220c28dbea0b73826cb7e3859882dc8` | `a22aa5c3abe00b41bf893bdfe2823e1f4955e6c52a1bcb0fc4f0e1963892f2ed` |
| 首层 CA + EMA | `7ab8825d853433c33cfc2ae1514fecd450af3c628ace5f7ecb5f162cccc4030d` | `681d02e8bdfa33f4ebf63ae35f66721c3f581810bfa760ad0eae8e6a7a51c298` |

各报告的 `runtime` / `runtime_batch_candidate` 是不可变快照；
canonical 仓库后续修改不会自动进入已运行任务，严禁就地改快照。


上述首层CA+EMA source digest亦为本轮复用的字节身份，runtime未修改。
本轮JM/verify_ema_scores.py支持4组8别名，可用--step 500 --recompute；有缺失就不能报全通过。
评测使用Native strict组件loader，EMA必须加载shadow及更新数，缺失时报错不能回退online。

## 验证索引与边界

下表路径除显式R/RE旧外，均相对 RM/validation；保留证据中具体时间和hash，不外推验证范围。

| 证据 | 已验证范围 |
| --- | --- |
| parameter_matching.json | 四个实际meta模型参数、CA块[0]、最近8倍数FFN候选和逐字段configdiff |
| preflight.json | 4train/8eval、before_self、fresh、online/EMA权重选择、源码及基础协议身份 |
| superseded_tasks_stopped.json | 旧未补参数3训练及剩余1评测删除NotFound；旧CPU服务和local退出 |
| launch_status.json / training_started_verified.json | 四组真实KAP/Pod启动、step0连续有限loss/grad和对应EMA更新数 |
| production_step500_ema_audit.json | 四个完整Native DCP的模型/EMA名称形状、参数数量、FP32shadow元数据、实际加载EMA更新数500与decay.99 |
| production_step1000_ema_audit.json | 同上，四组updates1000；CPU PID352468正常退出 |
| production_step2000_ema_audit.json | 同上，四组updates2000；CPU PID374107正常退出 |
| ema_scores_verified_00500.json | 8/8正式分数、每份2048cases身份/seed/GT/detector/manifest/权重类型与FP32有限特征；独立FVD复算最大误差2.274e-12 |
| ema_scores_verified_01000.json | 8/8、同上，最大误差3.638e-12；四对online/EMA同输入、各2048生成hash不同；CPU PID360134正常退出 |
| current_20b_scores_verified.json | 原始当前11份20B分数全部独立协方差复算，score hash与现文件一致 |
| 1560m_step2000_scores_committed.json | online456.201173、EMA182.145162的生产committed-score、Native manifest、身份、固定2048协议校验 |
| 125m_step2500_scores_committed.json / 355m_step2500_scores_committed.json / 775m_step2500_scores_committed.json | 三档2500 online/EMA生产校验、相应已完成任务清理 |
| common_step2500_conditioning_comparison.json | 三小档的entry/all-CA/first-CA-online/first-CA-EMA共12值与图表一致，base protocol字节相同 |
| production_step2000_dispatch_verified.json / next_eight_evaluations_dispatch_verified.json | 相应下一批实际worker启动证据；dispatch complete不等于FVD complete |
| training_efficiency_201_300_verified.json | 12行/1200原始事件独立复核、固定窗口/hash/EMA更新数 |
| resources_after_first30_scores.json | 历史11:40资源快照，非当前资源；勿沿用64GPU或dev1空闲结论 |

500/1000/2000 DCP审计是metadata和加载EMA标量检查，不能称为全部权重数值扫描。
1500及后续分数通过生产校验，但尚未做500/1000那种额外全case独立复算。
JM/verify_ema_scores.py已把required_scores从遗留6改为4组8别名；有缺失不能报全通过。
JM/audit_checkpoint_ema.py使用Native完整提交DCP；原PID334170的500检查已正常结束。
旧RE的三组审计不证明新FFN生产checkpoint，查上面本轮证据。

CA顺序163CPU+2maskedGPU回归；EMA原53CPU及最终13定向测试、56filelint、264mcorehash；
RE旧/validation保留真实8H200 train2→resume3→双权重16case及FP32 EMA递推核验。
这不是全仓库tests全绿声明。单H200及双机2×8B300的DDP/FSDP/HSDP/Ulysses有实证，
DDP+USP拒绝，CP/Ring/TP/EP/PP未验证。Muon会放大小梯度差异，FA4/SDPA短期loss接近不代表相同优化轨迹。
代码默认SDPA，当前实验显式FA4。

R/validation/atomic_artifact_recovery/recovery_audit_complete.json记录EFS原子写修复；
不覆盖原1.56B entry9000/9500已经审计的结果。
training_remaining_time_20260921T1145Z.json仅是11:45的最后200连续完成step估计，
包含慢步，未计未来抢占/排队/最终FVD；需要当前ETA时重读日志。

## 训练效率口径

固定201–300窗口，典型=1/median(step_time_s)，有效=N/sum(step_time_e2e_s)；
窗口没有checkpoint/demo，不删慢步。不同H200节点，原baseline无EMA，FFN补参改变计算量。

| 模型 | entry典型/有效steps/s | all-CA | first-CA补参+EMA |
| --- | --- | --- | --- |
| 125M | .528/.341 | .565/.345 | .540/.321 |
| 355M | .533/.342 | .455/.337 | .499/.325 |
| 775M | .403/.338 | .365/.342 | .391/.345 |
| 1.56B | .288/.287 | .244/.244 | .270/.268 |

本轮rank0逐步peak allocated/reserved GiB依次8.292/10.090、12.443/16.223、
17.243/20.543、22.538/28.178；不是整机nvidia-smi。
125M主报告.600/.593来自9970–9999最后30步，不能和201–300直接混用。
证据R/validation/conditioning_speed_windows_20260921.json及conditioning_speed_main_window_20260921.json。
新版first-CA不得复用旧未补参数版速度。

## Loss 与 FVD 的实测关系（2026-09-21 09:16 UTC）

只读分析原始 scaling 的训练日志与正式 FVD，未启动/停止训练或评测。
报告 `/efs/stormyxie/reports/scaling_loss_fvd_20260921T091640Z/REPORT.md`；
同目录含对齐数据、source hashes、独立统计核验和散点图；复现脚本
`/threed-code/stormyxie/vforge_jobs/scaling_loss_fvd_analysis_20260921/analyze.py`。
8个完整run×20档共160点：checkpoint前100步global训练FM loss与FVD的
Pearson=0.891、Spearman=0.703；仅5k–10k为0.453/0.446；10k横向8点为0.456/0.452。
125M@10k concat/cross的loss=0.498880/0.498179，FVD=578.91/221.30。
80对同模型族同step结构比较仅36对loss/FVD优劣方向一致，cross FVD较低73对。
这是移动**训练**loss的经验统计，不是held-out FM loss，也不是PPL=exp(NLL)式恒等关系；
不同checkpoint非独立重复。20B仅6条早期正式结果，独立列出；新首层CA+EMA未混入。
拟合生成质量scaling或选结构/checkpoint仍须正式FVD，不能用loss替代。

## 历史归档

整理前全文：[handoff/archive/20260921T122015Z/HANDOFF.md](handoff/archive/20260921T122015Z/HANDOFF.md)。

SHA256：1ba40d7d368a8d58686ae84665e900a0d6c981852696014b4e9657520293435e。索引 [handoff/archive/README.md](handoff/archive/README.md)。
此前20260921T104726Z及两仓库20260921T090641Z完整历史保留。历史Running/PID不构成恢复指令。
原参考仓库交接：/threed-code/stormyxie/rep_miniworld/HANDOFF.md。


## 原始 all-layer SA→CA→FFN 的 FVD spike 权重诊断（2026-09-21 12:53 UTC）

CPU只读检查原始scaling，不混入新首层CA+EMA。报告
`/efs/stormyxie/reports/scaling_spike_spectra_20260921T123934Z/REPORT.md`；
图为同目录`spike_spectra.png`，逐checkpoint JSON、完整SVD和feature复算证据均保留。
脚本：`/threed-code/stormyxie/vforge_jobs/scaling_spike_spectra_20260921/`。

- 共39份完整Native checkpoint：四档125/355/775/1560M all-CA及775/1560M concat的2.5k–5k每500步，额外355M all-CA 6.5k/7k/7.5k。全部weights payload/metadata hash与manifest一致，参数无NaN/Inf。
- 7734份矩阵谱统计；大矩阵8向量64轮子空间迭代，小矩阵精确SVD。775/1560M的3k/4k/5k对所有CA投影及异常候选做378次完整FP32 CPU SVD，校验的最大sigma1相对误差0.0498%。
- 775M在3k/4k/5k：FVD196.32/527.14/166.74，最大单矩阵sigma1=4.721/5.592/6.269；1.56B为209.02/410.95/156.37与4.997/5.269/5.504。四档all-CA共800矩阵中，4k的sigma1无一超过其3k/5k二者最大值1%以上；精确候选同样无此峰值。
- CA投影最大sigma1：775M 2.061/2.113/2.327；1.56B 2.077/2.159/2.254。Q/K norm gains与FP32 timestep/AdaLN probe均未发现4k独有峰值；frozen Native类在4k复核probe，RMS差0。不是实际视频激活或GPU采样验证。
- FFN谱确实渐进集中：block0 w12稳定秩3k/4k/5k，775M106.75/86.42/77.46、1.56B146.59/115.46/100.60；FVD恢复后仍继续下降，不能单独解释短暂spike。
- 775M的3500→4000这500步净参数位移，encoder12.20%、FFN18.01%、CA4.40%，相邻窗口相近；不是单步update比例。LR恒定1e-4，无全局梯度爆炸。3963的单步grad=1.495触发clip，但3500的FVD已经恶化；1.56B该窗口无clip。
- 对775/1560M的3k/4k/5k全部2048现存I3D features/sidecars核验固定协议、case、seed、GT、detector、checkpoint身份并独立复算，误差<1e-5。移除feature变化最大的1% cases后，775M仍194.58→523.04→167.29，1.56B210.01→407.25→156.71，非少数极端样本独自造成。
- 后期355M 6.5k/7k/7.5k同样：FVD179.77/358.29/159.44，max sigma1=6.413/6.578/6.724。

结论：当前证据不支持孤立checkpoint损坏或权重最大奇异值突然爆炸；存在持续参数漂移与谱集中，但未因果定位FVD spike。未排除奇异方向旋转、实际attention entropy/branch activation、conditioning表示漂移与autoregressive误差放大。全层CA为无独立timestep gate的加法分支，Q/K有RMSNorm，故原始投影权重谱不能直接代替logit或网络Jacobian。下一步应固定case/noise比较3k/4k/5k的attention entropy、branch/input RMS、逐state latent，配对真实历史与生成历史；新首层CA+EMA改变结构，不能直接用于归因旧all-CA。

未启动/停止训练或GPU评测，未改任何冻结runtime。上述CPU诊断全部完成，10个自有诊断PID已核对退出，见`process_completion_verified.json`；此处不声明其他任务当前状态。


## Same-spatial 首层 CA 三档消融（2026-09-21T15:12:58.739860+00:00）

用户确认：125M / 355M / 775M，沿用补参数 first-CA、CA→SA→FFN、
EMA 0.99；只限制 DiT CA 的空间查询范围。原 global CA 对照继续保留。
新训练从零开始，各8 H200，FA4（禁止SDPA fallback），FSDP2 DP8 /
USP1，local batch8 / global64，latent input、no-zprev、mask0.8、10 states，
max10k、完整DCP每500。以下只是启动验证时刻快照，后续读progress/log。

- Jobs: /threed-code/stormyxie/vforge_jobs/scaling_same_spatial_ca_20260921T145503Z
- Report: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z；预览 http://127.0.0.1:8884/same-spatial/
- Output: /efs/stormyxie/outputs/vforge/scaling_same_spatial_ca_20260921T145503Z
- Train frozen source: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/runtime_v2
  digest: a1266a17b672d6a3e95fa49a8e2c51656e68e5e96861496ad8196b270c8c0f36
- Eval frozen source: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/runtime_batch_v2
  digest: c9f4f189510bf62e185cf0eda5a8ec92db299e1f3c99d8593f3843a3290eb171
- Baseline: /threed-code/stormyxie/vforge_jobs/scaling_first_cross_ema_matched_20260921T093440Z
- Public option: model.extras.world.cross_attention_scope=same_spatial
  (default global). 不增加参数；三档与匹配首层global CA参数量完全一致，
  分别125044096 / 354975424 / 775044480。配置审计除scope、输出目录外逐项相等。

|档位|task (tenc-awsusw2 / gwm)|已完成step|loss|EMA updates|
|---|---|---:|---:|---:|
|125m|stormyxie-sc-125m-xlocal-0921150647|72|2.031418|72|
|355m|stormyxie-sc-355m-xlocal-0921150537|105|1.889641|105|
|775m|stormyxie-sc-775m-xlocal-0921150533|117|1.831894|117|

语义：同(h,w)的4个temporal planes可以互相查询；不是singleton (t,h,w)。
原始sparse condition_positions从ViT保留，不能拿压缩后的token序号对齐。
按空间位置和可见key数量分组调用Native FA4，相当于禁用位置logit=-inf /
softmax probability=0。空位置更新严格为0，包含projection bias；训练dropout
替换可见keys为learned null并保留visibility。80%独立per-plane mask下约41.1%
的位置无key。ViT与DiT SA仍有空间混合，只能检验DiT直接跨patch检索。

验证：37 CPU tests通过；H200 BF16 FA4显式dense reference对比，dense
output/gradient relL2=0.002443/0.003423，sparse=0.001014/0.002162；
SDPA调用0，dense sampling batch256通过。8 H200 Native真实训练step1/2、
保存DCP、恢复step3通过，EMA计数1→2→3。证据位于validation/：
config_parameter_match.json、fa4_same_spatial.json、gpu_smoke_PASS.json、
launch_verified.json。首次GPU短测发现FP32 normalized-Q accumulator与BF16
FA4输出的scatter dtype不一致；已修复、补CPU autocast regression并在新的
runtime_v2冻结，保留v1失败证据和旧snapshot。两个短测任务均已删除。
355M首次submit因平台DNS超时失败；确认原task不存在后用同一manifest/
task名重试成功，未重复提交另一任务。证据submission_dns_recovery.json。

FVD：同一固定2048 held-out bank、CFG1、16 observed+144 generated、
50 Euler、同case seed，online/EMA每500步分别评估，共120个新分数。
monitor_same_spatial.py每60秒发现完整DCP，fvd_extras.py在独立H200任务
评估，最多3个8卡eval jobs并自动清理完成/失败任务。
不在训练任务节点并行做eval；不启动dev1采样，以免碰另一任务的loss诊断。
监控PID见 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/services.json；队列见fvd/queue.json。
报告按同step展示global/local online和EMA。所有旧campaign/controller保留。
预览server仍使用原serve.py全部路由，新增/same-spatial/；
日志放EFS，避免/threed-code上长期打开日志句柄导致HTTP空响应。
当前server PID452224；本机8884隧道为foreground SSH工具session96817，
localhost HTTP已验证200。不要结束该SSH session。


## 固定测试输入 no-grad training loss 与 FVD 视频（2026-09-21）

核查时间：2026-09-21T15:18:54.771165+00:00。本轮全部完成。
报告 `/efs/stormyxie/reports/scaling_test_loss_20260921T134909Z/REPORT.md`；
代码 `/threed-code/stormyxie/vforge_jobs/test_loss_20260921T134909Z`；
[三个视频与全部曲线](http://127.0.0.1:8891/)。

- 清单冻结于13:51:33 UTC，208份完整DCP、243个权重评测点全部完成。
  原8小模型各500–10000全20档；20B entry截至3500、all-CA截至3000；
  补参数首层CA125M/355M截至4500、775M截至5000、1.56B截至3500，online/EMA分列。
  不含已取消的旧未补参数实验，也不代表后续新增checkpoint已经评测。
- 按用户要求train()+no_grad()，直接运行各自冻结Native的原training forward，
  不构建optimizer、不backward/update。Native strict组件loader恢复模型参数/buffer及真实EMA shadow，
  核对参数指纹、state名称/shape、EMA顺序/decay/updates=step；不声称恢复optimizer或data cursor。
  固定64个held-out case（35 Mira、29 Sekai，64个来源组，与训练组无交集），共576个相关next-state targets。
  RGB/帧/codec latent/noise/uniform抽样数/mask80索引/condition dropout持久化并核对hash，所有checkpoint共用。
  t由原训练logit-normal+shift变换生成，不是均匀t。小模型batch8、20B batch1，
  H200/FA4/FP32参数/BF16 autocast/TF32关闭；不同batch可能有小浮点差异。
  Native scalar与逐state MSE核对一致，验证无梯度、参数版本不变和固定首batch重复一致。
  协议SHA：`edc6d319ffd46335166441a43ec7bd890853e0818a9aeb3e9118e2e978438c53`。
- 775M all-CA 3k/4k/5k test loss=0.537437/0.514190/0.497798，FVD196.32/527.14/166.74；
  1.56B为0.516751/0.494434/0.480692，FVD209.02/410.95/156.37。
  两者3k→4k的64/64 case均改善，9个state位置和4个非空t分桶的平均loss也下降。
  原8小模型160点test loss/FVD Pearson=0.889、Spearman=0.707；≥5k为0.464/0.462。
  20B all-CA 1500→2000同样loss下降0.518137→0.492346、FVD上升258.85→415.86。
  真实历史上的训练目标改善不保证闭环FVD改善，不是PPL=exp(NLL)式关系；具体spike机制尚未证明。
  这是64-case估计，非完整2048均值；bootstrap按case。t五等宽区间样本数0/5/59/222/290，
  不能据此排除t<.2噪声端退化。详细分桶与局限见报告。
- 正式FVD只保存I3D特征/RGB hash，没有保存生成视频。本轮按固定编号fvd_0000/1024/2047，
  用原775M all-CA 3k/4k/5k、原case seed及冻结runtime_batch_candidate重放，
  四列GT/3k/4k/5k，每段160帧/16FPS/10秒，16帧观察+144帧生成，CFG1/每state50Euler。
  预览batch3不同于正式batch128，9份RGB hash均不同，不能声称原采样逐像素复现；
  标题FVD为整体2048分数。3份MP4全帧解码与HTTP Range检查通过，媒体留EFS，未下载本地。
- 14个loss run和全部预览采样进程结束，仅保留本轮loopback8891预览服务/SSH隧道。
  原coordinator因EFS临时文件rename中断，修复为唯一临时名+目标字节核验后从236点续跑。
  末段并行调度导致20B entry3000重复一次，两次576个逐state loss完全相同，副本保存在validation。
  最终coordinator complete，14份run COMPLETE均核验通过。未修改冻结runtime、未停止其他训练/FVD控制器，
  8882/8883/8884不变；这里不声明其他任务的当前状态。

证据：inventory.json、protocol.json、input_identity_audit.json、results/、summary.json、
aligned_results.json、fvd_preview/media_validation.json、process_completion_verified.json。
写入前HANDOFF原文与SHA保存在该报告validation目录。


### 20B新增分数独立审计与报告维护（2026-09-21T15:24:32.690461+00:00）

原始20B entry concat step3500、cross-attention step3000的全部2048个
fake features/sidecars、2048个GT features/metadata已核对。逐case检查稳定seed、
CFG1、50Euler、16 observed+144 generated、batch32、checkpoint identity、
resolved config、I3D detector identity和GT pixel identity；确认完整DCP与step。
使用独立float64 scipy.sqrtm covariance-product公式重算（生产使用factor SVD）：
concat3500=277.6083050576108，与报告差7.45e-10；
CA3000=281.2534789546389，与报告差3.99e-9。
此次未重新生成RGB或重新提取I3D；只审计已保存特征和元数据。
证据：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/fvd_new_20b_scores_20260921T1522Z 下两个20b_*.json。
脚本：/threed-code/stormyxie/vforge_jobs/scaling_fa4_20260920T134440Z/audit_new20b_20260921T1522Z.py。
不要重复做这两项同范围审计，除非相关产物改变。

同空间CA报告修正了metric是字典时应取metric.fvd的显示逻辑，已检查74个
现有global对照数值；仅页面展示变化，无训练/采样/分数修改。证据：
same_spatial_ca_20260921T145503Z/validation/report_metric_display_fix.json。
新消融monitor状态标签已由静态configured改为实际training等状态，当前PID454736，
FVD extras PID450930，实际PID仍以services.json和/proc为准。

原始20B两项当前FVD task仍Running且16张H200实测100% GPU利用率；
concat4000最新观察1024/2048、CA3500为768/2048，均尚无最终score。
1.56B首层CA step5000 online/EMA当时均1536/2048。
H200项目配额当时2B 697/704、2D 768/768；无完整8卡新slot。
不因评测批次间无产物或API timeout而重启；已有任务释放后controller继续去重补交。


### 同空间首层CA新增1.56B（2026-09-21T15:39:20.199864+00:00）

用户追加“有卡的话把1.5B也测一下”，本轮已为同空间patch首层CA消融追加1.56B。
训练/online/EMA配置由既有补参数global首层CA的1.56B逐项复制；
唯一语义差异cross_attention_scope=same_spatial，其余差异仅output/checkpoint路径。
Native meta模型实数1560089984，ViT1041291776/DiT518798208，CA仅block0、before_self；
FA4/noSDPA、8H200/FSDP2、local8/global64、mask80%、EMA.99、step0新训练、10k上限、DCP500。
FVD固定2048/CFG1/144generated/50Euler，online/EMA分别独立节点评测；1.56B采样batch64/rank。
注册表由3档扩展4档，预期同空间CA FVD由120增至160；既有global对照不变。
原runtime_v2/runtime_batch_v2与所有冻结model/evaluator/protocol文件未改，逐文件hash复核通过。
combined FVD protocol仍为189a016d1d58b5bb8a75d7cfb653fb30eff7ebe7656215145a70e465141b2dd1。
新task：stormyxie-sc-1560m-xlocal-0921153625，cluster=tenc-szr3212j，namespace=gwm，8H200。
提交时H200 2D项目配额744/768，但随后实际Pod仍Pending/Unschedulable、未分配node；
配额余量不代表完整物理节点空闲。此刻尚无1.56B真实训练step/显存/吞吐验证。
不要重复提交此task；下次先查询同一handle的真实Task/Pod/node/train_steps。
容量等待器submit_1560m_when_available.py已完成提交并退出，不是新常驻训练循环。
当前monitor PID461655，extras PID461656（核对services.json和/proc）；只重启了此campaign控制器以载入第四档。
monitor expected_scores已由硬编码120改为state中的160；/same-spatial/页面四档已上线，HTTP200。
扩展注册表时/threed-code的rename返回ENOSYS，已在控制器停止期间直接写入+读回核验后恢复服务；
未编辑既有三组配置，未停止任何训练或评测任务。
证据：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/add_1560m 下config_parameter_match.json、final_configs.json、launch_observation.json及metadata preimages。


### 1.56B同空间CA首次节点分配失败后重新候补（2026-09-21 15:45 UTC）

Task stormyxie-sc-1560m-xlocal-0921153625 在H200 2D分配到hyperpod-i-067cd8fe4602b742b后，Pod终态Failed/UnexpectedAdmissionError：申请8GPU而device allocation实际Available:0。
没有node.json、training progress或训练日志，模型尚未启动；不是loss/OOM/FA4错误。Task状态当时仍滞后Updating，以Pod终态为依据。
已保存失败证据并删除该任务，直接复核Task NotFound及Pod rows=[]后才移除当前submitted receipt；无重复训练。
新的候补进程PID 464851，脚本 /threed-code/stormyxie/vforge_jobs/scaling_same_spatial_ca_20260921T145503Z/submit_1560m_retry_admission.py；先给2D故障集群300秒冷却，期间仍可使用有完整8卡配额的2B，之后两集群均可候补。
候补状态/当前PID在same_spatial报告validation/add_1560m/capacity_status.json和capacity_waiter.json；旧任务身份仅在admission_failure_attempt1归档。
下一轮先检查新submitted.json和实际Task/Pod/训练步数，不能沿用15:39的旧pending身份。新配置、frozen runtime及FVD协议均未改变。
另已核对355M补参数global首层CA 6500 online/EMA：FVD180.6740346/103.8201875；两组2048 case ID集合完整，4个边界/中间sidecar的seed/reference/checkpoint/EMA核对一致。未独立重算FVD。证据为first_layer_cross_ema_matched_20260921T093440Z/validation/pair_355m6500_20260921T1544Z.json。


### 同空间CA第四档候补第二次提交（2026-09-21T15:51:24.103667+00:00）

1.56B当前身份更新为 stormyxie-sc-1560m-xlocal-0921155005 / tenc-szr3212j / gwm，8H200，15:50:05 UTC提交。
原15:36任务已确认删除；容量等待器完成此次提交后退出。新任务最近实查Pod Pending/Unschedulable、尚无node.json或训练step；不要因为等待器退出而重复提交。
继续跟踪该Task/Pod的实际状态。证据same_spatial_ca_20260921T145503Z/validation/add_1560m/attempt2_submission_verified.json与tasks/1560m_cross_attention/submitted.json。
同空间CA三小模型00500.dcp均经Native discover_checkpoints确认完整，保存的semantic_config为same_spatial、first CA before SA、mask80、local8/world8、EMA.99、FA4无fallback；6项online/EMA评测队列身份分别核对。
证据validation/first_production_ckpts_500_20260921T1548Z.json；本次只查manifest/完整DCP发现与队列，未重新载入production weights。


### 同空间首层CA第四档1.56B已真实开跑（2026-09-21T16:03:05.205876+00:00）

当前唯一任务仍为 stormyxie-sc-1560m-xlocal-0921155005 / tenc-szr3212j / gwm；Pod在hyperpod-i-024fd6d251ef0ec11启动，8张H200均143771MiB。
coordinator于16:00:39 UTC启动训练，从step0新训练；此刻已完成24步，loss=2.043032556772232，EMA更新数=24.0。
已核验全部已记录step的loss/grad有限、local8/global64/world8、EMA.99更新数=completed_steps，FAIL.json不存在。
rank0 peak allocated=22.542405366897583GiB，peak reserved=43.30859375GiB；仅启动期观察，不当作稳态benchmark。
源digest仍a1266a17b672d6a3e95fa49a8e2c51656e68e5e96861496ad8196b270c8c0f36，FA4/noFallback配置与原已核验config不变。
为避免采样队列先占用所有新释放卡，曾启动一次15分钟上限的AWS节点预留进程，短暂暂停matched/local两组extras的新提交；所有在跑训练/评测仍继续。
2D原候补任务先启动，因此未转移任务、未额外提交；预留进程正常以original_task_started退出，两个control.enabled均恢复true、reservation字段已删除。证据add_1560m/reserve_next_aws_node/controls_restored.json。
不要重启旧capacity候补/预留进程；当前1.56B已进入常规monitor。同空间CA现在4档均已训练，4x20x2=160个FVD目标。
完整证据：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/add_1560m/training_launch_verified.json。
补参数global首层CA对照1.56B step5500正式2048case FVD：online158.0412995582、EMA81.7520924586；EMA5000为88.2085534736。


### 125M全局首层CA online 7000 FVD跳升核验（2026-09-21T16:11:53.914355+00:00）

已核验同一2048case的online6500、online7000、EMA7000全部feature及sidecar：case身份、seed、reference RGB hash、detector配置、CFG1/50Euler/144generated、batch256、checkpoint完整性与identity、online/EMA选择和EMA更新数一致。
独立float64 scipy.sqrtm covariance-product重算，分别230.1528667921、454.1132291267、159.0440383447，与报告最大误差1.38e-9；没有重新生成RGB或重提I3D。
因此跳升存在于已保存的生成特征分布，不是FVD聚合公式算错或所检查元数据的混用；不据此声称完整排除模型推理实现问题，也不据此停训。
逐case到对应GT的I3D平方距离只作诊断代理（不是逐case FVD）：online7000相对6500上升比例=0.65380859375，平均差=196.2211121688287。EMA6500=158.8287947，EMA7000=159.0440383，基本稳定。
脚本/证据：scaling_first_cross_ema_matched_20260921T093440Z/audit_125m_online7000_jump_20260921T1607Z.py；报告validation/125m_online7000_jump_audit_20260921T1607Z.json。相关产物不变时不要重复这项全量审计。
同空间CA首个125M EMA500 sampler task=stormyxie-sc-xlocal-fvd-0921155619，虽然15:56提交，实际RUNNING.json为16:04:19 UTC；实测8H200 GPU100%、每卡约125589MiB（rank0约128661MiB），无FAIL/eval traceback。当时首批B256尚未落盘，不能仅因零case计数判断挂死。


### 同空间CA首个正式FVD完成并独立核验（2026-09-21T16:17:20.707509+00:00）

125M / step500 / EMA.99 / CFG1 / 固定2048case：同空间首层CA FVD10261.9622227335，对应补参数global首层CA7292.9811112292，早期高40.71%。
这是500步早期点，不足以判断长期优劣；global该档EMA1000已降至1398.2625。继续按固定协议测后续checkpoint。
新点全部2048feature/sidecar核验通过：同一case seed/reference RGB/detector、EMA updates500、cfg1/50Euler/144generated/batch256、checkpoint/evaluation scope均same_spatial。
独立float64 scipy.sqrtm重算10261.9622227328，误差6.63e-10；未重新生成RGB。
脚本：scaling_same_spatial_ca_20260921T145503Z/audit_first_fvd_125m_ema500_20260921T1613Z.py；证据报告validation/first_fvd_125m_ema500_20260921T1613Z.json。不要无新疑点地重复同范围审计。
报告/same-spatial/已按同step与global对照展示，monitor预期仍160点，所有四档训练已启动。


### 同空间首层CA首对online/EMA结果（2026-09-21T16:25:08.660330+00:00）

125M / step500 / 固定2048case / CFG1：global首层CA online5613.6911294、EMA7292.9811112；same_spatial online5107.2083122、EMA10261.9622227。
该早期点online同空间约改善9.02%，EMA同空间则差40.71%；不能用单个500-step EMA结果推断长期性能。继续同协议观察1k及以后。
online/EMA引用同一完整checkpoint identity；显式权重选择、same_spatial配置、两组2048个feature/sidecar ID集合以及4个成对metadata核验通过。
EMA点上轮已全量sidecar核验并独立sqrtm重算；此次online没有重复做独立重算。
证据：same_spatial_ca_20260921T145503Z/validation/first_pair_125m500_20260921T1625Z.json。


### 同空间1.56B首个正式500-step DCP核验（2026-09-21T16:45:20.922593+00:00）

当前唯一任务 stormyxie-sc-1560m-xlocal-0921155005 / tenc-szr3212j / gwm，
实际Task/Pod再次核对Running，node=hyperpod-i-024fd6d251ef0ec11；
最近monitor step659，历史pending/失败任务不可作为当前状态。
00500.dcp经Native完整checkpoint发现通过。CPU读取实际保存的EMA四个scalar/name
buffers，initialized=true、num_updates=500、decay=.99；完整metadata有660个
FP32 shadow tensor、共1560089984元素，与冻结meta模型ViT+DiT参数名顺序、
shape/数量完全一致，未包含frozen codec。online/EMA step500已分别入队。
此次不是production full-weight reload/resume测试，不把metadata检查代替恢复训练。
证据same_spatial_ca_20260921T145503Z/validation/add_1560m/first_checkpoint500_verified.json。

顺带核对global补参数首层CA 1.56B step6000：online FVD365.6457871211、
EMA74.7290396264；上个5500 online158.0412995582、EMA81.7520924586。
两份6000的2048 feature/sidecar ID集合及4个成对metadata核对通过，
checkpoint identity相同、显式online/EMA、case seed/reference一致。
本次未独立重算该对FVD，证据first_layer_cross_ema_matched_20260921T093440Z/
validation/1560m_pair6000_20260921T1637Z.json。
不凭一次online波动停训或归因checkpoint损坏。

对冻结global/local Native checkpoint实现做静态检查：manager同步stage weights
及optimizer完成后才提交background upload；DefaultStager的async/nonblocking
选项均false，两个冻结版本manager/storage文件hash完全一致。
这只证明所检查代码的阶段顺序，不替代新的数值resume或并发压力验证；
hash与检查范围记在first_checkpoint500_verified.json。


### 采样进展与调度重试核验（2026-09-21T17:05:18.358275+00:00）

原始scaling已175/200分数，八个小模型仍160/160，新增20B all-CA step3500
FVD273.2207722835（step3000为281.2534789506）。score/eval-manifest hash、
实际checkpoint manifest与complete组件、2048 feature/sidecar ID集合及4个对照
metadata通过；未独立重算FVD。证据原报告validation/20b_ca3500_20260921T164812Z.json。
该采样task stormyxie-sc-fvdx-0921142402已直接核验Task NotFound、Pod rows=[]。

同空间CA355M step500完整online/EMA已完成：global1727.0939373899/3252.1335283674，
same-spatial1775.6227369662/4009.4759746637。775M step500 EMA：
global1599.7838433718、same-spatial1353.8897101719；online仍待调度。
上述新点做score/eval-manifest、scope/weights/EMA500、全部ID集合及每结果4个
配对metadata检查，没有独立FVD重算。证据同空间报告validation/
first_pair_355m500_20260921T165213Z.json、first_775m_ema500_20260921T165533Z.json。
两个采样任务stormyxie-sc-xlocal-fvd-0921163259/0921164121均确认Task/Pod清空。
不要因这些500-step早期点就改变训练或断言长期排序。

H200 2D两项评测旧任务：xmatch-fvd-0921165025（125M EMA7500）与
xlocal-fvd-0921165118（775M online500），实际Pod Pending/Unschedulable，
CPU/内存/GPU/EFA无法凑齐完整节点；未启动模型，无推理FAIL。
extras按既有策略120秒确认无可调度节点后清理，并冷却10分钟。
两项旧Task NotFound且Pod rows=[]均直接核验；队列各保留一项，controller PID
339124/461656的真实cmdline核对存活、enabled=true、failed_targets=[]。
17:04 UTC新retry xmatch-fvd-0921170359已创建，Updating；继续查同一handle，
不要因排队重复提交。另一项由local extras自动重试，以最新receipt为准。
17:04:29核验global1.56B EMA6500远端仍Running、1024/2048，
dev1 online6500实际PID491363/fvd_single.py存活、1536/2048。

17:00–17:01直接两次读取原始20B train_steps，concat4459→4461、all-CA3920→3921，
持续推进。最近200步loss/grad有限；无checkpoint步骤e2e中位21.27/24.14秒，
至10k尚约33/41小时，不含额外保存/失败开销。原始monitor.py刻意sleep300秒，
不要把report几分钟旧视作挂死。此处训练步为日志step字段，不混作完整resume证明。


### 20B concat 4500完整提交与新FVD（2026-09-21T17:26:26.265194+00:00）

原始20B entry concat的04500.dcp已完整提交；训练节点与dev1直接读到同一完整
manifest，weights/optimizer/trainer均true，step4500；semantic config与04000完全
相同，FA4/no-fallback、DP8/local8保持不变。队列恰有一项，对应独立H200采样已提交：
stormyxie-sc-fvdx-0921172421 / tenc-szr3212j / gwm。此刻实际Task为Running，
不能把submitted/Updating等同于开始GPU采样。
证据原报告validation/20b_entry4500_commit_20260921T172517Z.json及最新admission JSON。

这次Native日志给出的async_checkpoint_committed：foreground9.789秒、
stage9.439秒、upload188.463秒、commit250.273秒、total448.525秒、wait0。
曾观察约160GB的weights/optimizer文件写出后.incomplete仍保留，最终正常提交；
训练期间仍持续推进到4507、4514等。不要单凭目录数分钟未提交或某次无open FD
就判定保存线程卡死；队列正确等待完整manifest。没有修改保存代码或重启训练。
原始FVD仍175/200，新增4500只是待测checkpoint。

global补参数首层CA 1.56B step6500 online125.5554139732 / EMA70.4428177958，
同一checkpoint identity，固定2048case。EMA5500/6000/6500为81.7521/74.7290/70.4428；
online5500/6000/6500为158.0413/365.6458/125.5554，online再次出现短期波动后回落。
证据matched报告validation/1560m_online6500_20260921T170844Z.json、
1560m_pair6500_20260921T171310Z.json。两组full feature/sidecar ID集合、
score/manifest/protocol/显式权重及4个成对metadata核对通过；没有独立FVD重算。
远端EMA任务stormyxie-sc-xmatch-fvd-0921164123已确认Task NotFound与Pod rows=[]，
dev1 online PID491363已退出。

global125M step7500 online259.1947586348 / EMA143.9844468007，配对同类检查通过；
证据matched报告validation/125m_pair7500_20260921T171626Z.json，未独立重算。
该点dev1 PID499075已接续采样后完成，随后local状态指向PID501292 / 1.56B EMA7000；
后续必须查真实PID和最新status，不用本段历史状态自动恢复。
CN旧retry stormyxie-sc-xmatch-fvd-0921170359已直接核对Task NotFound、Pod rows=[]，
之后才由dev1接手125M EMA7500，未发现重复执行。

新增same-spatial1.56B 01000.dcp已完整提交且online/EMA各一条队列身份；
检查components、scope/order与queue，无step1000全权重reload/resume或新EMA buffer load。
证据same_spatial报告validation/add_1560m/second_checkpoint1000_verified.json。


### 775M global首层CA训练完成；同空间1.56B继续推进（2026-09-21T17:50:11.697780+00:00）

补参数global首层CA任务 stormyxie-sc-775m-xmatch-0921093726 实际Succeeded，
完成时间17:39:38 UTC，PASS stages正确、0..9999共10000条训练记录loss/grad有限，
EMA更新数全程对应completed_steps；500..10000全部20个完整DCP已发现。
最终DCP实际CPU读取EMA initialized=true、updates10000、decay=.99，
metadata含494个FP32 shadows、775044480元素。本次不是全权重reload/resume。
证据matched报告validation/775m_train10000_verified_20260921T174355Z.json。
已在核验成功终态及最终checkpoint后删除训练Task，并直接确认Task NotFound、
Pod rows=[]；证据775m_task_cleanup_20260921T174457Z.json。
step10000 online/EMA各保留一条独立评测队列，不恢复这个已完成训练任务。

澄清控制器边界：fvd_extras.py的reconcile_training_releases仅观察成功训练退出、
记录training_releases.json并酌情清除较旧容量cooldown，不执行训练Task删除。
eval terminal cleanup会删除评测任务；不可把它等同于训练Task自动回收。
此次775M训练Task由单独核验后手动清理；没有为此修改/重启控制器或冻结runtime。
随后125M/355M完成时仍应核验实际终态、PASS、最终完整DCP后单独处理Task记录。

同空间1.56B唯一训练仍stormyxie-sc-1560m-xlocal-0921155005，直接Task核验Running；
17:49 UTC monitor已1665步，500/1000/1500完整checkpoint已列出。
step500 EMA task=stormyxie-sc-xlocal-fvd-0921173432，直接核验Running，
已1024/2048；online的新receipt为stormyxie-sc-xlocal-fvd-0921174859，
尚不能把submitted等同GPU运行。775M同空间online500旧排队task
stormyxie-sc-xlocal-fvd-0921171228已Running且1024/2048，
证明此前Updating/无Pod不能仅按等待时间判失败。不要重复提交这些handle。

原始20B all-CA的04000.dcp现已提交完整manifest，weights/optimizer/trainer均true，
queue恰有一条4000项；此次只做manifest和queue检查，未重载全权重。
原始20Bconcat4500 sampler仍Running，最近256/2048。
global1.56B EMA7000新FVD70.18571327399104，light audit已完成，证据
matched报告validation/1560m_ema7000_20260921T173347Z.json；
未独立重算该新点。原始scaling仍175/200，20B两组继续训练，goal保持active。
本轮观察证据：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/milestones_20260921T175011Z/observations.json。
预览/same-spatial/本机HTTP200；8884 tunnel未改动。


### 同空间1.56B首个500步EMA FVD完成；20B all-CA4000已开始采样（20260921T175652Z）

同空间首层CA 1.56B / step500 / EMA.99 / CFG1 / 固定2048case正式FVD为
1114.9851677441497，对应global补参数首层CA EMA500为1281.1846751187777，
早期低约12.97%。online500当前身份为stormyxie-sc-xlocal-fvd-0921175313 /
tenc-awsusw2；不要沿用旧2D失败/重试handle。不凭一个500步早期点判断长期优劣。
同轮global1.56B online7000=133.72114134083176，EMA7000=70.18571327399104。
以上新点均核对score/manifest hash、完整saved checkpoint manifest、所有2048
feature/sidecar文件ID集合及4个配对case的seed/reference/detector/显式权重/EMA步数，
没有重新生成RGB或独立重算FVD。证据：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/new_1560m_fvd_20260921T175609Z.json。

775M同空间step500现在online/EMA齐：1666.3848550129876 /1353.8897101718658；
global对应1838.1308301532076 /1599.7838433717504。
global775M step7500 online156.379679604192 /EMA72.86756662303151。
新online点同范围核验通过，证据validation/new_775m_online_fvd_20260921T175412Z.json。
此前新增355M online7500=173.99568267872564，EMA7500=90.04925451386725；
775M EMA7500从7000的80.5836369190215下降到72.86756662303151。
证据matched/validation/new_fvd_355online_775ema7500_20260921T175138Z.json。
不要无新疑点重复这些light audits或误称已独立重算。

原始20B all-CA4000独立H200 FVD任务 stormyxie-sc-fvdx-0921175305 /
tenc-awsusw2，17:54 UTC实际Task直接核对Running；concat4500的
stormyxie-sc-fvdx-0921172421继续运行。原始eight smaller已160点放行20B；
fvd/eligible_queue.json是已退役primary留下的旧快照，实际extras直接读queue.json
再调用eligible()，不可用旧eligible_queue的mtime/内容判断漏调度或重启primary。
实际queue对4000保持唯一项，immutable runtime/config/protocol均未修改。

最新独立Task/Pod观察及本次HANDOFF全量preimage归档于：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/first_1560_same_fvd_milestone_20260921T175652Z。
当前global125M/355M约9757/9753步，尚未完成；之后核实真实Succeeded、
PASS、最终完整DCP，再处理训练Task记录，不要提前删训练任务。
原始20B两组仍在训练、FVD未满200，goal继续active。


### Global补参数首层CA125M/355M完成10k并清理（20260921T181033Z）

125M task stormyxie-sc-125m-xmatch-0921093733、355M task
stormyxie-sc-355m-xmatch-0921093730 均从step0连续训练至10000，
Task实际Succeeded时间分别2026-09-21 18:07:31/18:07:38 UTC，
全部worker容器exitCode0，PASS stages正确且无FAIL。
验证完整0..9999日志序列、每步loss/grad有限及EMA更新数=step+1；
500..10000全部20个manifest完整，weights/optimizer .metadata存在。
实际CPU加载最终EMA initialized=true、updates10000、decay.99；
FP32 shadows元素分别125044096/354975424，CUDA未初始化。
覆盖只包括日志/manifest、EMA小buffer及shadow metadata，不是全权重reload/resume。
证据matched报告validation/
125m_cross_attention_train10000_verified_20260921T180858Z.json、
355m_cross_attention_train10000_verified_20260921T180858Z.json、
125_355_train_cleanup_preflight_20260921T180918Z.json。

上述核验后再次直接确认Task Succeeded，已依次删除两个完成Task；
两者均直接复核Task NotFound与Pod rows=[]。证据
validation/125_355_train_task_cleanup_20260921T1810Z.json。
末档step10000的125M online/EMA、355M online/EMA恰各一项在queue.json中。
不要恢复这两个已完成训练任务。现在global首层CA125M/355M/775M都训练完成，
1.56B仍继续；FVD队列继续独立消化后续checkpoint。

此次等待用的只读观察器已以BOTH_TRAINERS_VERIFIED_FINISHED正常退出。
没有修改冻结runtime/config/protocol，也没有重启任何训练或评测进程。
最新所有campaign快照及same-spatial1.56B online500直接Task观察：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/completed_125_355_20260921T181033Z/state_snapshot.json。
原始10组scaling的两个20B仍未到10k、FVD未满200，goal保持active。
本轮新落盘但尚未手工做light audit的点：global125M EMA8000
146.12254094003947、same-spatial125M EMA1000 1078.2343032359995；
controller常规valid_score不等于独立FVD重算，后续引用时注意验证范围。


### 新FVD、全部已提交checkpoint队列覆盖及资源空档核验（20260921T182227Z）

当前全部完整checkpoint的评测身份覆盖检查：
original177个目标=175个score+2个未完成队列；
matched152=123+29；same-spatial52=8+44。三者无漏项、无重复队列身份，
已有score的cases2048/smoke=false/step/protocol/eval-manifest hash均核对通过。
这只覆盖目前已提交checkpoint，不能把未来未产生的20B checkpoint当作完成：
原始目标仍200、matched与same均160，goal继续active。
证据validation/checkpoint_eval_coverage_20260921T182048Z.json；
未重算全部FVD、未重载完整checkpoint payload。

此前标注未做light audit的125M EMA8000(146.1225409400)及same-spatial125M
EMA1000(1078.2343032360)现已做四个配对sidecar、全部2048文件ID、manifest检查；
后者对应global EMA1000=1398.2624869901。另global1.56B EMA7500=70.7397234440，
EMA7000=70.1857132740。证据validation/new_ema_fvd_20260921T181158Z.json。
相关旧远端EMA任务已实查NotFound/Pod空，本机旧PID517178已退出；
证据validation/new_ema_fvd_cleanup_20260921T181341Z.json。

global1.56B online7500=139.3136399016、EMA7500=70.7397234440，
配对同checkpoint与固定case核验通过。
20B all-CA4000采样task stormyxie-sc-fvdx-0921175305 / tenc-awsusw2已产出首批256，
concat4500 task stormyxie-sc-fvdx-0921172421 / tenc-szr3212j至少768；
两者实际Task/Pod Running，18:12 GPU观察各8H200均100%、约119359–122785MiB。
不是仅凭GPU利用率断言无hang；all-CA随后真实feature落盘提供了推进证据。
第一批sidecar ID是0..255，不能假设按rank连续256分区已生成0/256/1024/1792。
初版部分结果审计因此选了尚未生成的ID，已改按实际落盘的0/85/170/255核对；
没有修改sampler、runtime、配置或协议。
证据validation/20b_sampler_gpu_20260921T181251Z.json、
fvd_pair1560_7500_and_20b4k_firstbatch_20260921T181707Z.json。

global1.56B08000与same-spatial1.56B02000完整manifest已确认，
online/EMA各唯一入队；仅manifest/queue检查，未新做全量load/resume：
validation/1560_new_checkpoint_queues_20260921T181734Z.json。

18:19实查H200 quota：2B697/704，2D744/768；有配额不等于完整节点资源。
matched旧355M online8000 task stormyxie-sc-xmatch-fvd-0921181506实际历史Pod
Pending/Unschedulable，CPU/内存/GPU/EFA不足；控制器按既有120秒策略清理，
2D cooldown至约18:28:15 UTC。已再次直接确认该旧Task NotFound与Pod rows=[]。
新工作由dev1本机PID530034接手：任务stormyxie-sc-xmatch-local-0921181833，
真实cmdline匹配fvd_single.py/355m_cross_attention__online/8000。
matched extras PID339124、local PID330375均实际存活、enabled=true；
remote active为空不是controller死亡，未重启/改写服务或冻结快照。
旧远端清理证据与本次全量HANDOFF preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/coverage_handoff_20260921T182227Z。
尚未人工sidecar audit的新增点：125M online8000=263.4627789259、
355M EMA8000=85.0345572828；常规score/hash覆盖已通过，不等于独立FVD重算。


### Same-spatial1.56B首对500-step FVD齐全（20260921T183045Z）

same-spatial1.56B step500 online=1135.1113659148214、EMA=1114.9851677441497；
global matched首层CA对应online=1863.2430339552466、EMA=1281.1846751187777。
online低约39.08%、EMA低约12.97%，均只是500步早期点，不据此判长期优劣。
new online的score/eval-manifest hash、完整checkpoint manifest、全部2048
feature/sidecar文件ID、4个配对metadata已通过，同一checkpoint identity
ca2fb6503d56109900ba5469b7b87ba531e950e3a7bea0f607d99dba42f9eff9，
固定case seed/reference/detector、显式online/EMA、EMA500一致。
没有独立重算FVD或重新生成RGB；证据原报告validation/light_fvd_20260921T182938Z.json。
sampler stormyxie-sc-xlocal-fvd-0921175313实际Succeeded后已由controller清理，
本轮直接确认Task NotFound与Pod rows=[]。只读等待观察器正常退出。

same-spatial125M step1000 online/EMA=1154.2183853516876/1078.2343032359995，
global对应1453.2115539230672/1398.2624869900833；新online配对核验通过。
之前待审计的global125M online8000=263.462778925893、
global355M EMA8000=85.0345572827882也已完成同类light audit。
证据原报告validation/light_fvd_20260921T182503Z.json；
125M同空间online1000旧任务与global online8000旧任务已直接确认Task/Pod清空，
证据validation/latest_online_cleanup_20260921T182636Z.json。
不要无新疑点重复这些检查；没有做新的独立FVD重算。

最新任务快照、1.56B sampler cleanup与本次HANDOFF preimage：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/1560_pair500_complete_20260921T183045Z。
frozen train/eval/config/protocol未修改，原始20B两组仍未到10k、总FVD未满200，
goal保持active。

当前会话工具store内有可复用只读light audit代码键fvdLightAuditPy：
输入specs为[campaign,name,step,ref_campaign,ref_name,ref_step,batch]列表，
campaign仅matched/same；生成一份validation/light_fvd_TIMESTAMP.json，
检查完整结果的score/hash/manifest、全部文件ID及4个metadata，不做独立FVD计算。
如工具store不可用，按上述证据范围恢复检查即可；它不是新增训练runtime文件。


### 355M/775M新FVD配对及同空间4k队列（20260921T184430Z）

global matched355M step8000 online/EMA=125.30323477283082/85.03455728278823；
online7500=173.99568267872564。新online配对light audit通过，
证据原报告validation/light_fvd_20260921T183258Z.json。
same-spatial355M EMA1000=702.811943214012，global对应1270.1813358263848；
固定case/detector/配置元数据对照通过，
证据validation/light_fvd_20260921T183509Z.json。
该same EMA task stormyxie-sc-xlocal-fvd-0921182329已Task NotFound/Pod空，
旧dev1 global355M online8000 PID530034已退出。
同空间125M/775M04000完整manifest已确认，online/EMA各唯一入队；
证据validation/new_checkpoints_cleanup_20260921T183618Z.json，
只查manifest/queue，无本轮full resume或新EMA scalar load。

global matched775M step8000 online=189.11410197139548、EMA=74.74108858639647；
step7500对应156.379679604192/72.86756662303151。两个分数均上升，
EMA幅度较小；不要把所有checkpoint描述为单调改善，也不凭单点停止训练。
新的online与前档、EMA与同档online已做score/manifest hash、完整checkpoint
manifest、全部2048文件ID及4个配对case metadata检查，
同档identity ad7c533bda42b5ad5e9ceffdeb2466170f9e51d19f728a54bf7579a867e28481。
证据validation/light_fvd_20260921T183901Z.json、
validation/light_fvd_20260921T184236Z.json。
这些light audits没有重新生成RGB、独立FVD重算或全权重resume。

EMA采样task stormyxie-sc-xmatch-fvd-0921182331 / tenc-awsusw2 已实际Succeeded，
本次再确认Task NotFound/Pod rows=[]；只读等待进程已正常退出。
新dev1 sampler的真实PID/命令匹配、各campaign最新步数/score总数、
旧PID状态及本次HANDOFF preimage见 /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/775m_pair8000_complete_20260921T184430Z/observations.json。

原始20B concat4500/all-CA4000采样在18:42仍持续运行，至少1280/512 features；
两个20B训练仍未达到10k，原始FVD仍未满200。goal继续active。
冻结runtime/config/protocol和controller配置均未修改。
18:42本机/first-cross/ HTTP200，原8884隧道持续在线。


### 原始八个已完成小模型的历史Task/Pod清理（20260921T185017Z）

原始125M/355M/775M/1.56B各entry-concat/all-CA，共8个训练Task仍保留Succeeded记录。
本轮先直接核对真实Task均Succeeded、原creationTimestamp、PASS对应单模型且无FAIL，
最终10000 manifest weights/optimizer/trainer完整、各20个FVD存在；
8个Pod均Succeeded且全部worker容器exitCode0。随后逐个再次核对Task终态与身份，
依次删除这8个已完成Task，并逐一直接确认Task NotFound/Pod rows=[]。
没有删除仍在训练或评测的Task，也未删除持久化产物。
这是历史已结束记录的清理，不把它计为新释放64张在用GPU。

证据原报告validation/
original_completed_trainers_observation_20260921T184617Z.json、
original_completed_trainers_cleanup_preflight_20260921T184731Z.json、
original_completed_trainers_cleanup_20260921T184812Z.json。
清理后重新确认这8组全部160个checkpoint manifest及160个score仍在；
这是文件存在性post-check，score/hash全量覆盖已在18:20做过，没有重算全部指标。
两个原始20B Task已再次直接观察仍Running，继续推进到10k目标，不受此清理影响。

保留8个训练的submitted/PASS及日志/退出证据以保留历史身份。
monitor.py第94–96行按PASS+completed_steps10000+完整10000 checkpoint判complete，
优先于Task查询ObservationUnavailable；后续看到已删除Task NotFound不能自动恢复训练。
training release observer已有历史seen记录，仅观察退出/清cooldown，不是trainer删除器。

global1.56B08500、same-spatial1.56B02500与same-spatial355M04500的最新提交/入队检查、
三个campaign快照及本次HANDOFF全量preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/original_cleanup_postcheck_20260921T185017Z/observations.json。
新增checkpoint检查范围仅manifest/component flags/唯一online+EMA队列，未新做full resume。
原始FVD仍未满200、20B仍未到10k，goal保持active；
冻结runtime/config/protocol和现有controller未修改或重启。


### Global1.56B step8000配对完成及same355M step1000新online（20260921T185936Z）

global首层CA1.56B step8000 online/EMA FVD=118.35701849194174/69.20949487919233；
step7500对应139.31363990155614/70.73972344401272。
新EMA对上一档、online对同档EMA的light audits通过：
原报告validation/light_fvd_20260921T185632Z.json、light_fvd_20260921T185748Z.json。
同档checkpoint identity为de100a7326b20f56f9d5b2c1f9051ef9d299cca39ba13bcff6de1b15a348d143。
覆盖全部2048文件ID集合、score/manifest hash、完整checkpoint manifest与4个配对
case metadata，不包含独立FVD重算、RGB再生成或full resume。

same-spatial355M step1000 online/EMA=872.3663296058144/702.811943214012；
global对应1172.0293473135025/1270.1813358263848。
新online light audit通过：validation/light_fvd_20260921T185314Z.json；
这是1000-step早期点，不据此断言长期排序。

两个已完成远端sampler xmatch-fvd-0921183319 / xlocal-fvd-0921184159
本轮再次直接确认Task NotFound与Pod rows=[]；旧dev1 online PID538543已退出。
最新本机接续sampler真实PID匹配、各campaign状态及本次HANDOFF preimage：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/1560_pair8000_complete_20260921T185936Z/observations.json。
只读配对等待观察器已正常退出，没有保留额外监控子进程。

18:52直接读取原始20B两组最近100条训练日志，loss/grad均有限并持续更新；
entry/all-CA completed_steps当时4772/4197，两组FVD任务实际Running。
原始scaling仍未到200/200、20B未到10k，goal保持active，冻结配置/协议未修改。


### Same-spatial775M step1000配对完成、1.56B持续训练（20260921T191118Z）

same-spatial775M step1000 online/EMA FVD=1028.4175716962477/534.6472879184685；
global matched首层CA对应1135.4645450894554/856.9306745674162。
EMA跨scope对照和online/EMA同checkpoint light audits通过，证据原报告
validation/light_fvd_20260921T190719Z.json、light_fvd_20260921T190958Z.json。
same checkpoint identity b5a0e0c24754fd8dab7db22ffe9f3b5122882d3ad02df3e9c200a015c4b191ec。
覆盖全部2048 feature/sidecar文件ID、score/eval-manifest hash、完整checkpoint
manifest及4个配对case元数据；未独立重算FVD、重生成RGB或full resume。
早期1k点不代表长期优劣，也不能解释成ViT/SA禁止了跨空间信息混合。

上述两项sampler xlocal-fvd-0921185210和xlocal-fvd-0921185327均Succeeded后
controller清理，19:10直接确认Task NotFound与Pod rows=[]。
只读observer PID550370/session54801已PAIR_COMPLETE正常退出。
Global125M online8500=251.03125541725603，light audit已通过：
原报告validation/light_fvd_20260921T190342Z.json。

用户要求追加的1.56B same-spatial任务stormyxie-sc-1560m-xlocal-0921155005
已于19:10再次直接确认实际Task/Pod Running，节点hyperpod-i-024fd6d251ef0ec11；
最新报告2915步/最新完整2500 checkpoint。首个500 online/EMA结果已见此前记录，
1k EMA/online独立采样分别为xlocal-fvd-0921185904（2D）与
xlocal-fvd-0921190715（2B），尚未完成，不提前计分。
19:10快照：原始175/200；matched130/160；same14/160；
原始20B训练4812/4231，global1.56B训练8853，same125/355/775训练4780/5062/4774。
这些是带时间快照，不是最终值；原始goal保持active。
任务清理与训练快照证据：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/775m_pair1000_status_20260921T191036Z.json。
19:10本机/same-spatial/ HTTP200；8884预览隧道保留。
冻结train/eval/config/protocol/controller均未修改。本次全量preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/775m_pair1000_complete_20260921T191118Z。


### 原始20B concat4500 FVD完成（20260921T191952Z）

20B entry-concat step4500 FVD=268.6468060552029，前档4000=292.7603421801573；
新结果2048cases齐全。score/eval-manifest hash、完整保存checkpoint manifest、
全部2048 feature/sidecar文件ID及4个跨checkpoint配对case元数据已通过light audit：
原报告validation/light_fvd_20260921T191547Z.json；
identity 4fc17e0fbfc66175e0c41264a5f08aa016c62fb6bb265cf76494145522001f17。
未独立重算FVD、重新生成RGB或full resume；原始模型无EMA字段，未误按EMA协议验收。

只读observer实际轮询x-fvd任务stormyxie-sc-fvdx-0921172421，
19:15看到Running+完整score，19:16看到Task NotFound；随后Pod查询rows=[]。
没有声称这段观察捕捉到了Succeeded过渡状态；score/完整产物与已清理状态分别验证。
证据validation/20b_concat4500_wait_20260921T191610Z.json、
validation/20b_concat4500_cleanup_20260921T191637Z.json。
临时observer PID553645/session29363已正常退出。
原始monitor PID348486实际cmdline匹配monitor.py，按既有300秒刷新；
19:19重新确认报告从175更新到176并包含268.6468060552029。
不要因单轮网页落后新落盘score重启服务。8884与图像HTTP证据见本次observations.json。

Global matched125M EMA8500=141.0619657148277，
同档online=251.03125541725603，light audit通过：
validation/light_fvd_20260921T191219Z.json，identity
31e2f4c7f068d02341e84de3e221553676f3a72a5eb777cdba4063c86a36fe9d。
same-spatial125M/775M04500、355M05000完整manifest/组件标记和唯一online+EMA队列检查通过：
validation/same_new_checkpoint_queues_20260921T191426Z.json。
初版临时检查错读queue项为checkpoint键而失败，已按实际path键修正验证；
没有更改queue、runtime或训练/评测配置，不是scheduler或训练故障。

20B all-CA step4000任务stormyxie-sc-fvdx-0921175305仍未完成；
原始两个20B训练未到10k，goal继续active，冻结runtime/config/protocol/controllers未改动。
本次最新报告快照与全量HANDOFF preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/20b_entry4500_complete_20260921T191952Z。


### Same-spatial1.56B step1000完整配对与9k/3k checkpoint（20260921T192850Z）

same-spatial1.56B step1000 online/EMA FVD=412.4827596994919/425.8857758268391；
global matched首层CA同档849.5912447532928/726.0126766916042。
两种权重在这个早期点均更低，不据此判长期排序；此档same online略低于EMA，
不能将EMA描述为每一个checkpoint都更好。
新EMA跨scope、online对同档EMA的light audits通过，
证据原报告validation/light_fvd_20260921T192229Z.json、
light_fvd_20260921T192809Z.json；同档identity
eaad4e8af9e6efebdf9777caa8df1b4f10a6c5d6b9b67ecd02453b0bee494760。
覆盖score/manifest hash、完整checkpoint manifest、2048文件ID集合及4个配对
元数据；没有独立FVD重算、RGB再生成或full resume。

两个1k采样task xlocal-fvd-0921185904（EMA、2D）与
xlocal-fvd-0921190715（online、2B）已直接确认Task NotFound/Pod rows=[]；
观察器分别PID557762/session74383、PID558589/session26449均正常结束。
等待轨迹：same报告validation/1560_ema1000_wait_20260921T192202Z.json、
1560_online1000_wait_20260921T192758Z.json；
EMA与之前355M online8500已清理、旧dev1 PID551934已退出的证据：
validation/1560_ema1000_cleanup_20260921T192234Z.json。
不要把观察中Running→NotFound表述成亲眼捕捉了Succeeded过渡。

global1.56B09000与same1.56B03000完整manifest/唯一online+EMA队列已确认：
原报告validation/1560_new_checkpoint_queues_20260921T192135Z.json。
same125M/775M05000也完成同类检查，见本次observations.json。
范围仅manifest与队列，不是新full state resume。

global355M8500 online/EMA=144.83547180695632/85.27800781873702；
相较8000对应125.30323477283082/85.03455728278823，两个分数均略有回升，
EMA变化较小。新online与前档、新EMA与同档online的light audit通过：
原报告validation/light_fvd_20260921T192113Z.json、light_fvd_20260921T192913Z.json。
该EMA任务xmatch-fvd-0921191806最新清理观察见本次matched355_ema8500_cleanup.json。
global775M EMA8500=70.0584147505765（8000=74.74108858639647），
也已在192113Z proof验证；online8500尚待结果/核验，19:22实际dev1 PID557134，
真实cmdline匹配该case/step，后续仍需用现状重新确认。

19:28报告已包含same1.56B完整1k配对；快照original176/200、
matched133/160、same16/160。global1.56B训练9138，same四档训练
5140/5440/5129/3181；原始20B仍未到10k，all-CA4000采样仍未完成，goal active。
same125M EMA1500任务xlocal-fvd-0921191809于19:26真实Updating/无Pod，
未视为失败、未重启；按原句柄后续轮询。原始all-CA4000仍推进中。
冻结runtime/config/protocol/controllers未改动。本次全部证据与HANDOFF preimage：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/1560_pair1000_complete_20260921T192850Z。


### Global775M 8500配对、20B all-CA采样批耗时（20260921T193825Z）

global matched775M step8500 online/EMA=160.25544179439476/70.0584147505765；
step8000对应189.11410197139548/74.74108858639647。
新online对同档EMA的light audit已通过，证据原报告
validation/light_fvd_20260921T193232Z.json，checkpoint identity
2445fd3c97cda191aea157c52582516d8eb8d9e28f45aacbf898f39d80e36bdc。
范围同既有light audit：2048文件ID集合、score/manifest hash、完整checkpoint
manifest和4个配对元数据；不含独立FVD重算、RGB再生成或full resume。
旧dev1 PID557134已退出，接续1560M online8500真实PID562399/
task stormyxie-sc-xmatch-local-0921193137，19:32实际cmdline匹配。
same355M05500完整manifest与唯一online+EMA入队也已确认。
上述证据validation/775_online8500_completion_20260921T193234Z.json。

原始20B all-CA4000任务stormyxie-sc-fvdx-0921175305实际持续Running，
19:37 features已推进到1536/2048，尚无score，不能提前算177/200。
原报告validation/20b_ca4000_batch_timing_20260921T193654Z.json按实际npy mtime
统计前5批（每批256），相邻完整批间隔1014.757/1018.892/1019.069/1017.838秒，
median1018.365秒，约17分钟。19:36距离上一批1007.7秒，随后19:37确有新批落盘；
这是正常批时长的直接证据，不是仅据Task Running或GPU利用率判没有卡住。
19:37完成1536后，余两批按近况估计约34分钟，非保证的deadline。
等待轨迹validation/20b_ca4000_nextbatch_20260921T193746Z.json。
两个只读observer PID562407/session89099、PID563046/session89201均正常退出。
不要因整批17分钟才集中落盘而重启当前健康任务。

19:30真实查询：两组20B训练仍Running，最新日志快照4878/4290，loss/grad有限；
same125M EMA1500 xlocal-fvd-0921191809已从Updating进入Running，
这说明此前无Pod是调度等待，未做人工重启。
后续125M online1500 xlocal-fvd-0921192822当时Updating，继续按原句柄观察，
不要因为临时无Pod认定失败。
原始目标仍未完成：两组20B未到10k且FVD未满200，goal active。
冻结runtime/config/protocol/controllers没有改动。
本次最新报告快照、775M8500已展示结果与HANDOFF全量preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/775_pair8500_complete_20260921T193825Z。


### Global1.56B 8500与same125M 1500配对完成（20260921T195208Z）

global matched1.56B step8500 online/EMA=107.66382073930026/68.64693300837462；
step8000对应118.35701849194174/69.20949487919233。
EMA对前档、online对同档EMA的light audits通过：
原报告validation/light_fvd_20260921T194912Z.json、
light_fvd_20260921T195036Z.json；同档identity
ba91fc53bac507f35d114f6c5da592f600ffb0fbb89ec507d1019ad06e35e844。
远端EMA任务xmatch-fvd-0921192817已Task NotFound/Pod空；
本机online旧PID562399退出，新接续实际cmdline见本次observations.json。

same-spatial125M step1500 online/EMA=708.0274818269381/733.7714600867403；
global对应1489.2867466108114/1163.4787964357688。这是早期点，
不能据此判长期效果；same此档online低于EMA，不称EMA恒优。
新EMA跨scope、新online对同档EMA的light audits通过：
validation/light_fvd_20260921T194023Z.json、light_fvd_20260921T194239Z.json；
同档identity ba0df616d3e7a0ac8781ebd1a46cfc8b39aaadd10c1dd870aca60a814c93a0a1。
125M两项任务xlocal-fvd-0921191809 / xlocal-fvd-0921192822直接确认
Task NotFound/Pod空，证据validation/same125_ema1500_cleanup_20260921T194026Z.json、
same125_pair1500_cleanup_and_pending_20260921T194447Z.json。
本节light audits范围为全2048文件ID、score/manifest hash、完整checkpoint manifest、
4个配对metadata；没有独立重算FVD、重新生成RGB或full resume。

19:39实际H200额度：gwm 2B697/704、2D768/768，没有整组8卡的新增配额。
2D新任务xmatch-fvd-0921193845（125M EMA9000）真实Pod Pending/
Unschedulable，CPU/内存/GPU/EFA不足；19:46已实际Task NotFound/Pod空。
same355M EMA1500旧2D尝试xlocal-fvd-0921193837也已清理。
两个目标仍各唯一在queue，failed_targets为空；matched extras PID339124、
same extras PID461656真实cmdline匹配且enabled=true。
2D cooldown当时分别至19:54:49与19:51:54 UTC，控制器可在其它有资源池接续。
证据validation/h200_admission_cooldown_20260921T194615Z.json；
不要把暂时active=[]解释成controller死亡，也不要手工重启旧已清理尝试。

same1.56B03500完整manifest/唯一online+EMA入队已确认；
global1.56B09500在19:50查询尚无完整manifest，等待实际训练/保存完成，
不是保存失败。证据validation/1560_pair8500_cleanup_new_checkpoints_20260921T195039Z.json。
19:48原始20B all-CA4000任务仍Running/1536features，未计score；
两组20B训练本轮再次直接查实际状态，见本次observations.json。
原始goal未完成（20B未10k、FVD未200），保持active；冻结runtime/config/protocol/controllers未改。

只读observer PID566393/session42260、PID567113/session88650、
PID569799/session11570均正常结束；等待证据
validation/fvd_active_wait_20260921T194158Z.json、
matched1560_8500_wait_20260921T194848Z.json、
matched1560_online8500_wait_20260921T195013Z.json。
本次最新报告快照与HANDOFF全量preimage：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/1560_pair8500_and_same125_1500_20260921T195208Z。


### 原始20B all-CA4000完成、9k配对与统一评分元数据检查（20260921T201846Z）

原始20B all-CA step4000 FVD=275.33816344335537，3500=273.2207722834596，
本档略回升，不把曲线描述为单调改善。固定2048cases的新点light audit通过，
证据原报告validation/light_fvd_20260921T201252Z.json，checkpoint identity
f59a2772f4379abecbda1f5ce4a3d5ffd9e37e917e6cd20807d16e5d9c0bdb65。
20:11实际观察task stormyxie-sc-fvdx-0921175305 Succeeded及完整score，
随后直接确认Task NotFound/Pod空，证据
validation/20b_ca4000_final_wait_20260921T201142Z.json、
20b_ca4000_completed_20260921T201255Z.json。
observer PID574426/session49668已正常退出。
原始monitor按既有300秒周期已自动把报告更新到177/200，无重启或手工刷新脚本。

global matched125M9000 online/EMA=205.49002351344598/128.59881904937538；
8500为251.03125541725603/141.0619657148277。配对light audit：
validation/light_fvd_20260921T195651Z.json，identity
75483552d8cf2948a3b325e0fd636b8190c40f259df80af4d918d7f4fb582033。
对应任务xmatch-fvd-0921194146 / xmatch-fvd-0921194824的receipts记录真实Succeeded，
本轮直接确认两者Task NotFound/Pod空，证据
validation/matched125_pair9000_cleanup_20260921T195927Z.json。
找历史sampler身份可查各报告fvd/extras/receipts/TASK.json，不要猜task时间戳。

global matched355M9000 online/EMA=167.0501718553437/80.93124517050092；
8500为144.83547180695632/85.27800781873702，本档online回升、EMA下降。
light audits：validation/light_fvd_20260921T200238Z.json、
light_fvd_20260921T200638Z.json；identity
d3d06fae560a73e9156100ef0c2c6d8102ff11eb668764e541f566046c31a512。
online任务xmatch-fvd-0921195453已直接Task NotFound/Pod空：
validation/matched355_pair9000_cleanup_20260921T201031Z.json；
旧本机EMA PID570094退出。后续实际本机任务见本次observations.json。

same-spatial355M EMA1500=389.57366270388593，global对应602.474634323718；
新EMA light audit也在200238Z proof。same EMA任务xlocal-fvd-0921194844
已直接Task NotFound/Pod空：validation/355_new_ema_cleanup_20260921T200559Z.json。
same355M online1500旧2D尝试xlocal-fvd-0921195156的receipt明确是
no_schedulable_H200_node清理，非模型运行失败；重试继续按controller的新任务身份查看。

完整checkpoint/唯一online+EMA入队的新检查：global1.56B09500、
same125M/775M05500、same355M06000，证据
validation/new_checkpoint_queues_20260921T195652Z.json。
20B concat已超过5000步，05000.dcp.incomplete仍写入时不能入评测队列。
本次只检查路径/manifest/queue，未干预保存；上次大20B提交用了约448.5秒，
不要因为几分钟没有最终manifest就重启。当前5000实际路径/队列见本次observations.json。

20:01新增一次所有已落盘334个score JSON的元数据复核：
original176/matched139/same19，全部通过，证据
validation/all_score_metadata_20260921T200108Z.json。
覆盖真实/生成各2048、feature_dim400、covariance_ddof1、有限FVD、非smoke、
CFG1/50步/16观察帧/144生成帧/16fps/seed20260920、各档固定batch、
campaign协议hash、权重tag和目录身份。不是重生成RGB、重载全权重或重算全部FVD。
会话工具store的fvdLightAuditPy与fvdLightAuditAllPy已加入上述metric计数和
score.protocol采样参数断言，未来新点沿用；没有改动冻结train/eval代码或协议。
本节light audits仍额外覆盖全2048文件ID、score/eval-manifest hash、完整checkpoint
manifest及4个配对case metadata，不将其夸大成所有RGB逐像素重算。

原始goal未完成：20B未到10000、FVD未200，保持active。
冻结runtime/config/protocol/controllers未修改；8884隧道继续保留。
本次最新报告/实际训练task/评测接续快照与HANDOFF全量preimage：
/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/20b_ca4000_and_9k_pairs_20260921T201846Z。


### Matched四档训练完成、same1.56B继续训练（20260921T204048Z）

global matched首层CA的1.56B任务stormyxie-sc-1560m-xmatch-0921093722
实际于20:24:44 UTC Succeeded，10000完整checkpoint和PASS随后落盘。
最终CPU核验：日志step0..9999完整，10000条loss/grad有限，EMA更新数逐step正确，
最终EMA initialized=true/updates=10000/decay=0.99；
FP32 shadow metadata覆盖1560089984参数，20个完整checkpoint/组件均在。
核验只加载三个EMA标量buffer、读取shadow metadata，CUDA未初始化；
不是full weight reload/resume或所有checkpoint payload hash。
证据matched报告validation/1560m_cross_attention_train10000_verified_20260921T202750Z.json。

在重复核对同一Task创建身份、Succeeded、Pod退出0后，仅删除该已完成Task，
输出/checkpoint/PASS保留；直接post-check Task NotFound/Pod空。
证据matched报告validation/1560_train_cleanup_preflight_20260921T202746Z.json、
1560_task_cleanup_20260921T203224Z.json、all_matched_training_complete_20260921T203428Z.json。
此时matched四档全部训练到10k、80个完整checkpoint保留；不要因已清理Task NotFound重启。
最终1.56B10000的online/EMA各唯一入队，20:40复核通过，
见matched报告validation/matched_complete_same1560_status_20260921T204011Z.json。
临时finish observer及CPU audit均正常退出，无额外残留监控。

新FVD light audits：
- global775M9000 online/EMA=118.35298111767588/69.14983613183722；
  same355M1500 online/EMA=847.1480931354035/389.57366270388593，
  新online已核验，原报告validation/light_fvd_20260921T202101Z.json。
- global1.56B9000 online/EMA=75.3650339866972/63.697079521273736；
  global125M9500=229.62228508437238/127.97908531987787；
  global355M EMA9500=81.13181262972012；
  same775M1500 online/EMA=353.95051542205374/283.7103919523579，
  原报告validation/light_fvd_20260921T203945Z.json。
范围为score/manifest hash、协议与实际计数、完整保存checkpoint manifest、
全部2048文件ID集合和4个配对case元数据；不包含独立FVD重算或RGB再生成。
早期点不判长期排序，曲线不宣称单调、EMA不宣称恒优。

原始20B concat5000完整checkpoint已提交，20:21初查尚未入队，
随后producer正常刷新到唯一队列并提交stormyxie-sc-fvdx-0921202511（AWS）。
20:34直接Task/Pod Running/request8GPUs；一次pod exec NVML只读查询失败，
不能把该次查询当实际GPU型号/利用率验证，更不能解释为Task运行失败。
证据原报告validation/20b_concat5000_commit_20260921T202102Z.json及
matched报告validation/all_matched_training_complete_20260921T203428Z.json。
20:40该采样仍active，完整批尚未落盘，不提前计score。

用户追加的1.56B same-spatial任务stormyxie-sc-1560m-xlocal-0921155005
20:40再次直接Task/Pod Running，节点hyperpod-i-024fd6d251ef0ec11；
最近日志4293步、loss有限。1000 online/EMA已完成412.4827596994919/425.8857758268391，
1500两项独立FVD在跑（xlocal-fvd-0921202503 / xlocal-fvd-0921203409）。
20:40快照original177/200、matched147/160、same22/160；
原始20B仍约5070/4459步，未达到10k，原始goal保持active。
控制器enabled、dev1仍在matched355M online9500采样；same没有local runner。
matched FVD全完成后若dev1空闲，可复用现有调度继续same backlog，先核实现状。
8884/same-spatial/本机HTTP200，永久SSH预览隧道保留。
冻结train/eval/config/protocol/controller未修改或重启。
本次full HANDOFF preimage及摘要hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/all_matched_complete_and_same1560_20260921T204048Z。


### Same1.56B EMA1500完成、20B5000首批核验（20260921T204726Z）

same-spatial1.56B step1500 EMA FVD=204.5209313546393；
global matched同档EMA=277.74665797002535，same前档1000EMA=425.8857758268391。
新2048case结果light audit通过：原报告validation/light_fvd_20260921T204635Z.json，
checkpoint identity eda0765cd214e6918fb12bbe10aed21264c67bea070cd3c87f8cfc2181f8e56c。
核验范围同前：计数/协议、score与manifest hash、全部文件ID、完整保存manifest、
4个配对元数据；未独立重算FVD/RGB，不据早期点判断最终排序。
online1500任务xlocal-fvd-0921203409仍在评测，尚未补上此档配对分数。

EMA任务stormyxie-sc-xlocal-fvd-0921202503的controller receipt记录
20:45:37 UTC实际Succeeded并成功删除；20:46直接Task NotFound/Pod空。
证据validation/same1560_ema1500_wait_20260921T204619Z.json、
same1560_ema1500_complete_20260921T204655Z.json。
只读等待observer/session4518已正常结束；ObservationUnavailable最后一轮
经复查是已清理Task，不是训练/评测失败。
global355M online9500新FVD=173.9087575085182（同档EMA81.13181262972012），
light audit：validation/light_fvd_20260921T204420Z.json。

20:41实际查询两组原始20B训练Task均Running，直接日志5080/4468步，
近期loss有限；20Bconcat5000独立评测Task stormyxie-sc-fvdx-0921202511也Running，
首批256features已落盘。首/中/尾3个已有case侧记录核对：
checkpoint5000与保存manifest一致，和4500相同case/seed/reference/detector，
CFG1/50Euler/10states/144frames/batch32/协议一致。
证据validation/20b_entry5000_firstbatch_20260921T204436Z.json，
identity 072c4d2473626d5a66f8bdce8853916c2bfe18242f6a97be90f09d9b0f2b1497。
这是partial batch metadata检查，不是完整FVD验收。
临时审计初版错把仅完成时写的eval_manifest当成途中必有文件，
又错按原文件bytes算checkpoint manifest_sha256；现按已保存sidecar核对，
该hash是sort_keys+compact separators的canonical JSON hash，实际一致。
上述是临时审计脚本假设修正，没有改评测/训练或checkpoint产物。

20:46报告快照original177/200、matched148/160、same23/160；
same四档6705/7093/6686/4391步，matched四档训练已10k；原始20B未10k，goal active。
冻结runtime/config/protocol/controllers未修改，永久8884隧道未动。
本节证据位于原报告validation；完整HANDOFF preimage及hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same1560_ema1500_verified_20260921T204726Z。


### 原始20B CA4500提交与same1.56B1500配对（20260921T210406Z）

原始20B all-layer CA任务stormyxie-sc-20b-xa-0920143629本轮按同一真实Task
持续查询Running，20:54达4500，21:01:52已完整提交04500.dcp，
manifest/weights.metadata/optimizer.metadata及组件flags齐全；21:03已训练4522步。
只读观察轨迹validation/20b_ca4500_commit_wait_20260921T210324Z.json，
session98635按900秒观察上限正常结束，训练任务未终止/重启。
20:57分片存在性/size快照约148.92GiB：validation/20b_ca4500_partial_20260921T205730Z.json。
21:03原始queue仍无CA4500：原监控PID348486实际cmdline匹配，
上一轮progress更新21:01:07，早于完整提交，等待原300秒间隔下一轮扫描。
这不是漏评故障或观察超时导致的重启依据；下轮优先确认CA4500唯一入队及独立评测。
concat5000在20:59已512/2048features，未计完整FVD分数。

20:49直接查当前两个20B训练Pod及concat5000评测Pod：
三者均Running/request8GPU，并位于三个不同节点，
证明当前该评测未与两个20B训练共节点。
证据validation/20b_train_eval_nodes_20260921T204924Z.json。
20:52原始已提交178档checkpoint覆盖检查通过：177有2048case score，
唯一未评分concat5000有唯一queue项；所有已提交manifest/组件/metadata齐全。
证据validation/original_checkpoint_queue_coverage_20260921T205219Z.json。
此检查早于新增CA4500完整提交，不能宣称覆盖179档。
临时检查初版错误要求已评分checkpoint仍在queue，后按实际pending-only
队列语义修正；已完成项自动移除，177项均有score，无丢失未评分项。
未改queue或controller逻辑。

same-spatial1.56B step1500 online/EMA FVD=409.7052422542811/204.5209313546393；
global matched同档=404.70487247310075/277.74665797002535。
因此此档same EMA更低，online略高，不宣称两种权重都胜出或长期排序已确定。
online同checkpoint/跨scope light audit已通过：
validation/light_fvd_20260921T205557Z.json，identity
eda0765cd214e6918fb12bbe10aed21264c67bea070cd3c87f8cfc2181f8e56c。
online任务xlocal-fvd-0921203409的receipt确认20:54:20 Succeeded并删除，
直接Task NotFound与Pod空；validation/same1560_pair1500_cleanup_20260921T205617Z.json。
global775M9500 online/EMA=112.89339110924084/68.56808279589131，
同一light audit也已通过，identity
59ae8e9f74cf22a65623a2d361c4ec041893e43391e5a5696dcbd1f85a4da7a1。
light audit仍为协议/计数/score-manifest hash/完整保存manifest/
全部2048文件ID及4个配对元数据，不含RGB重生成、独立FVD重算或full resume。

原始20B未到10k，原始FVD仍177/200，goal继续active；
matched四档训练10k已结束，FVD在20:59为150/160；
same1.56B继续训练，最新各campaign快照见本目录observations.json。
冻结runtime/config/protocol/controllers未更改，永久8884隧道保留。
所有本轮临时读取/等待进程均结束。完整HANDOFF preimage及hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca4500_complete_same1560_pair1500_20260921T210406Z。


### 原始20B CA4500评测已自动启动、新EMA结果（20260921T211045Z）

CA4500从21:01完整checkpoint→21:06:54唯一队列→21:09直接Task/Pod Running，
全程未人工改队列/重启监控。任务stormyxie-sc-fvdx-0921210822，
AWS H200/8卡，节点hyperpod-i-0095afef5754f65ea。
21:09:58同时直接查两组20B训练Pod均Running/request8GPU，
新采样与两个训练分别不同节点；评测协议hash/完整manifest与唯一queue path一致。
证据validation/20b_ca4500_evaluator_wait_20260921T210911Z.json、
20b_ca4500_started_20260921T210958Z.json。
只读observer/session49151观察到实际Running后正常退出。
21:08曾实际quota 2B697/704、2D768/768，原extras PID348487真实cmdline匹配、
enabled=true、failed_targets=[]、无cooldown；当时是8卡配额不足的短暂等待，
不是controller死亡或Task失败，随后自动接续。勿恢复旧Task或新造重复采样。

新增EMA light audits：
- global matched125M10000=118.24338296157521，9500=127.97908531987787；
  identity 78e97ab7e5179739b7e39bbb98e2f178f2d474e11118e15ef92a3d219fa11d7f。
- same125M2000=504.73412868976766，global同档796.7356851750951；
  identity 7d66ffb72ea6de61108510a3f38f7d2788e9ddcfa6cdc436dd59f7686c7802e8。
  证据validation/light_fvd_20260921T210503Z.json。
- global1.56B9500=62.863437460078785，9000=63.697079521273736；
  identity 11de246b9458c35e8270b5b75d01a7fec63866a5324f3774820c6f7fd085ecda。
- global355M10000=78.97768251520438，9500=81.13181262972012；
  identity c6d1a5395fd925cb680f8b66bde1a85ac9b6226b025b01e7e6839183825c5de3。
  证据validation/light_fvd_20260921T210939Z.json。
覆盖协议/实际2048计数、score/eval-manifest hash、完整保存manifest、
全部文件ID及4个配对元数据，未重算FVD/RGB或新full resume。
这些新EMA点不代表所有同档online也已经完成；按实际score判断。

原始两个20B训练仍未10k、原始FVD仍177/200；新增CA4500启动不提前算完成。
goal继续active；冻结runtime/config/protocol/controllers没有改动。
下一步继续观察两个20B独立评测与训练、新checkpoint；matched剩余FVD完成后，
先实际核对dev1空闲再考虑复用local评测给same backlog，不能与现有local sampler重叠抢卡。
本次最新各campaign快照、完整HANDOFF preimage与hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca4500_eval_started_new_ema_20260921T211045Z。


### Matched1.56B9500完整配对、本机继续10k评测（20260921T211454Z）

global matched1.56B step9500 online/EMA FVD=119.29702044194119/62.863437460078785；
9000对应75.3650339866972/63.697079521273736，本档online回升、EMA小幅下降，
不能写成训练越久FVD单调改善。
新online同checkpoint配对light audit通过：
validation/light_fvd_20260921T211348Z.json，identity
11de246b9458c35e8270b5b75d01a7fec63866a5324f3774820c6f7fd085ecda。
same125M2000 online/EMA=834.8361321662564/504.73412868976766，
新online light audit通过：validation/light_fvd_20260921T211158Z.json。
范围同前：协议/计数、score-manifest hash、全2048文件ID、保存manifest及4个
配对元数据；不是重新生成RGB、独立FVD重算或full resume。

本机原1560online9500 PID595905实际cmd为fvd_single.py加分开的name/step/task参数，
已核对确为目标；21:14进程已退出，新PID603841真实cmdline匹配
355m_cross_attention__online 10000 stormyxie-sc-xmatch-local-0921211256。
本机并非空闲，不能启动第二个占满8GPU的同类runner。
证据validation/matched1560_online9500_wait_20260921T211327Z.json、
matched1560_9500_local_completion_20260921T211409Z.json。
只读observer/session61416已正常结束；其stdout的cmdline_matches使用了错误的
路径子串(name/step)，所以打印false，原始保存cmdline与直接读取已确认参数匹配；
这不是采样目标错误或PID丢失，未干预实际进程。

21:11直接查询原始两组20B训练与两项采样（concat5000 x-fvd0921202511、
all-CA4500 x-fvd0921210822）四个真实Task均Running。
concat5000当时768/2048features，all-CA4500刚启动尚无首批，未提前计score。
原始报告训练当时5150/4530；两组未10k、原始FVD仍177/200，goal active。
冻结train/eval/config/protocol/controllers未更改，永久8884隧道保留。
本次最新快照与完整HANDOFF preimage/hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/matched1560_9500_pair_20260921T211454Z。


### Dev1同patch接续器已等待、125M/355M10k配对（20260921T212233Z）

新增外部orchestration脚本（不是Native train/eval改动）：
/threed-code/stormyxie/vforge_jobs/scaling_same_spatial_ca_20260921T145503Z/fvd_local_after_matched.py。
复用现有matched fvd_local调度实现，保留同一eligible/lease/fvd_single/Native入口、
8×H200空闲检查和score验证；仅更改local task前缀为xlocal-local并添加前置接续。
前置条件是matched固定8个权重alias×20档全部160个score有效，
并且原matched local controller.lock已经释放，随后原GPU空闲检查仍执行。
独立handoff.lock防止重复等待器，现有同patchcheckpoint租约防止与远端重复采样。

已启动PID606450，写入same报告services.json及fvd/local/controller_launch.json。
21:20直接检查实际cmdline、无子进程、nvidia-smi计算进程中不含该PID；
状态waiting_for_matched_completion、gpus=0，未启动同patch本机GPU评测。
等待状态的completed计数按30秒刷新，可能比新score文件晚一轮，正常。
后续matched160/160且旧local退出后应自动接续；启动后仍需验证首个真实same
本机sampler参数/进程、FA4冻结runtime与协议，不提前宣称本机新路径GPU已验收。
当前matched本机已有任务，不要再手工开启第二个占满8卡的sampler。

准备/CPU-only --check通过：same报告
validation/local_handoff_prepare_20260921T211831Z/validation.json，
脚本SHA 9a89fdf9fd47ba53333176fa1b49b50262d3e1485f28ea56f1a175d98688e6a2。
launch/status/真实进程证据：
validation/local_handoff_launch_20260921T211909Z.json、
local_handoff_waiting_verified_20260921T212039Z.json。
新脚本解析的J/R、runtime_batch_v2、source_digest c9f4f189...0eb171、
协议189a016d...2dd1、batch125=256/355=128/775=128/1560=64已核对。
现有state/eval_entry/eval_entry_batch_frozen/fvd_single/protocol的前后hash未变；
原matched controller源码未改，也未重启训练/评测/监控。
same index.html目前不展示local资源字段，实际local状态以fvd/local/status.json为准。

新结果核验：global matched125M10000 online/EMA=187.78629052035876/118.24338296157521；
global355M10000=111.89317997344642/78.97768251520438。
same355M EMA2000=253.70300002838667，global对应370.7779811772634。
新online/EMA light audit：原报告validation/light_fvd_20260921T212114Z.json；
范围仍为协议计数、score/manifest hash、完整保存manifest、2048文件ID及4个
配对metadata，不是FVD/RGB独立重算或full resume。

代码阅读发现一个未来清理注意点：same monitor_same_spatial.py当前优先把
TaskNotFound归为observation_unavailable，尚无original/matched的
PASS+10000步+完整final checkpoint优先判complete分支。
在same训练真正完成并清理Task之前应补上这一状态判定，避免已完成任务被误报；
当前same四组还在训练，没有因该问题重启任何任务，也不能按旧Running记录自动恢复。
本轮仅新增等待接续器及其services记录，没有改现有monitor。

原始两个20B及两项独立评测的最新实际Task状态见本目录observations.json。
原始仍未到10k/200分数，goal active；全部冻结runtime/config/protocol保持不变。
本次最新各campaign快照与全量HANDOFF preimage/hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/local_same_handoff_final_pairs_20260921T212233Z。


### Same只读监控/报告状态修复与20B首批验证（20260921T212927Z）

已解决上一节所记same monitor完成任务清理后误报问题：
monitor_same_spatial.py新增纯training_lifecycle函数；
明确失败仍优先；PASS或实际Succeeded结合10000日志与Native发现的完整10000 checkpoint
才判complete，缺日志/最终checkpoint判finished_early；
无PASS的ObservationUnavailable仍保留未确认状态，不触发任何恢复。
报告新增local_evaluation字段，same index.html显示本机采样状态/GPU数/前置评测计数；
训练行改用已核验r.status，避免仍显示底层scheduler的ObservationUnavailable。
没有改Native训练/评测、FVD协议或checkpoint格式。

8个CPU边界例子、Python syntax通过，原监控和HTML完整preimage：
same报告validation/monitor_lifecycle_local_20260921T212437Z/validation.json。
remote无node；随后在工具V8编译并执行新报告JS，用已完成但scheduler unavailable
样例确认显示complete，以及等待样例显示GPU0和156/160；
证据validation/monitor_ui_check_20260921T212635Z.json。
本机HTTP实际读取same页面与progress.json成功，新增字段及local等待状态可见；
这是语法/状态显示验证，不是新的图像布局截图检查。

仅终止经实际cmdline确认的旧只读monitor PID461655，并确认锁释放后启动
新monitor PID609245，services.json已更新。
新进程首次tick成功、四组训练仍training，协议hash仍189a016d...2dd1；
证据validation/monitor_lifecycle_deployed_20260921T212537Z.json。
原same extras PID461656和等待接续器PID606450未重启；Native训练/采样进程未动。
接续器仍等待matched160完成，当前以fvd/local/status.json及新报告字段为准。

20B all-CA4500首批256features已生成，真实Task x-fvd0921210822仍Running；
首/中/尾3个已落盘case核对checkpoint4500、canonical manifest hash、
相同case/seed/reference/detector、CFG1/50步/144frames/10states/batch32/latent协议通过。
原报告validation/20b_ca4500_firstbatch_20260921T212725Z.json。
这是partial batch metadata检查，不是完整FVD验收；concat5000同时已1024/2048。
原始完整FVD仍177/200，两个20B未到10k，goal active。

global matched775M10000 EMA FVD=59.23811418486639（9500=68.56808279589131），
light audit通过：原报告validation/light_fvd_20260921T212812Z.json，
identity 24083e059a124edbcb202edfc8f0cf55bbd5224b1cbf30e5d01e4f43eba176b3。
覆盖协议计数/hash/保存manifest/全2048文件ID与4个配对metadata，
没有重新生成RGB、独立FVD重算或full resume。
本次最新各campaign、same1.56B5000 checkpoint/队列快照见observations.json。
matched剩余评测结束后，等待器应自动接续；实际启动后还需验新本机sampler。
如H200资源释放，可依实际quota/节点状况增加same远端采样并发；
不要把max_extra_nodes上限当作实际占用或可用节点。
永久8884隧道保留。本次full HANDOFF preimage/hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/monitor_status_fix_and_20b_batch_20260921T212927Z。


### Matched四档10k EMA齐全、本机执行最后1.56B online（20260921T214126Z）

global matched1.56B10000 EMA FVD=59.39598667769269（9500=62.863437460078785）。
新点light audit通过：原报告validation/light_fvd_20260921T213931Z.json，
identity 0d53fb8ba7addc34f27a17a943534ad13564408e2efe05585a06144112c22944。
四档最终10k EMA：125M118.24338296157521 /355M78.97768251520438 /
775M59.23811418486639 /1.56B59.39598667769269。
这一档775M和1.56B数值相近；没有多seed/置信区间，不据此声称统计学饱和。
light audit范围仍是协议/计数、score-manifest hash、完整保存manifest、2048文件ID
及4个配对case元数据，不是FVD/RGB重算或full resume。

原本机EMA PID607250确认为指定fvd_single参数，持续推进1024→1536→2048；
21:39成功退出，receipt returncode=0、PASS存在，21:40 /proc已无旧PID。
新本机PID614436实际cmdline：
fvd_single.py 1560m_cross_attention__online 10000 stormyxie-sc-xmatch-local-0921213919。
证据validation/matched1560_ema10000_wait_20260921T213940Z.json、
matched1560_ema10000_local_completion_20260921T214018Z.json。
只读observer/session52425在完整score出现后正常退出。
same等待器PID606450继续按160完成+原local锁释放+GPU空闲门禁等待，不能另起占卡进程。

21:34直接检查matched eligible()：唯一可领取剩余目标是1.56B online10000；
775M online10000有真实远端任务xmatch-fvd-0921212650，1.56B EMA有本机租约。
reserved列表无卡住的旧目标，failed_targets=[]，实际extras PID339124 cmdline匹配。
当时真实H200 quota 2B697/704、2D768/768，未触及本地max_extra_nodes上限，
所以未盲目提高并发上限；最后1.56B online现已由本机自动领取。
2D旧775M online10000尝试xmatch-fvd-0921211921、same775M EMA2000尝试
xlocal-fvd-0921211926的receipt均是实际no_schedulable_H200_node清理，
不属于模型运行失败，不能恢复旧task。
已完成matched775M EMA10000 xmatch-fvd-0921211344、
same355M online2000 xlocal-fvd-0921211351均有Succeeded后deleted receipt。

same355M2000 online/EMA=1019.1623355926345/253.70300002838667，
新online配对light audit通过validation/light_fvd_20260921T213053Z.json，
identity c7ca9fc067cbc7eed7d02a67110ef4e9cb719ae02239e52baa0405be68aabde1。
21:39 original20B concat5000已1280features，CA4500当时256features；
完整原始FVD仍177/200，两个20B训练未10k，goal active。
所有冻结runtime/config/protocol及现有controllers本轮未改动。
最新各campaign、本机任务、最终EMA表与HANDOFF完整preimage/hash：/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/all_matched_final_ema_20260921T214126Z。

## 2026-09-21T22:06:52.221675+00:00 — matched first-CA complete; same-spatial local sampling activated

- Latest user request fulfilled: added parameter-matched 1.56B same-spatial first-CA on 8 H200 in gwm/tenc-szr3212j, Task stormyxie-sc-1560m-xlocal-0921155005, from zero, local batch8, FA4, EMA .99. Direct CA scope changes; SA/ViT remain spatially global. Four-run snapshot: [{"name": "125m_cross_attention", "status": "training", "completed_steps": 8288}, {"name": "355m_cross_attention", "status": "training", "completed_steps": 8756}, {"name": "775m_cross_attention", "status": "training", "completed_steps": 8257}, {"name": "1560m_cross_attention", "status": "training", "completed_steps": 5603}].
- Score counts: {"original": 177, "matched": 160, "same_spatial": 31}. Original goal remains ACTIVE: two original20B trainers have not reached10k and original scores are not200/200. Last verified healthy steps5282/4646, last30 losses/grad norms finite: validation/20b_training_health_20260921T215313Z.json. Do not restart due to slow checkpoint commit or observation expiry.
- Global matched first-CA now has four trainings complete at10k and 160/160 expected online/EMA FVD scores. validation/matched160_matrix_20260921T220150Z.json checks every score/protocol/count/output-manifest hash/full saved checkpoint manifest and all80 online/EMA checkpoint identity pairs.
- Final-band verifier completed: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema_matched_20260921T093440Z/validation/ema_scores_verified_10000.json; eight final scores, all2048 sidecars and finite FP32 400D feature arrays per score, all4 paired case/seed/GT/checkpoint identities. No independent FVD recomputation/RGB regeneration in these audits.

| Size | 10k online FVD | 10k EMA .99 FVD |
| --- | ---: | ---: |
| 125m | 187.78629052 | 118.24338296 |
| 355m | 111.89317997 | 78.97768252 |
| 775m | 85.95326519 | 59.23811418 |
| 1560m | 86.96089759 | 59.39598668 |

- Last matched local sampler614436 committed1560m online10k and exited. validation/matched_final_online_wait_20260921T215753Z.json. Shell observer97582 and CPU checker finished successfully; no need to poll/restart them.
- Matched monitor330373 and local330375 naturally exited. Idle extras339124 was SIGTERM'd after160 scores, empty active/failed_targets and last remoteTask Succeeded/deleted/noPods verified. No training/sampler signals. control.json enabled=false; services.json=[], services_completed.json records retired services. Full preimages/proof: validation/matched_completed_services_20260921T220422Z/validation.json. Do not restart completed services/trainers.
- Same gated local606450 automatically began stormyxie-sc-xlocal-local-0921215800 (775m online2000, launcher621738) on dev1 only after predecessor lock released and GPUs empty. Actual8 GPU worker commands/ancestry verified; same_spatial/before_self, FA4, perGPUbatch128. /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/local_handoff_active_20260921T220226Z.json. No inherited CPU-only CUDA_VISIBLE_DEVICES override.
- First local same batch1024 features/sidecars; four finite FP32 arrays and bank/detector/checkpoint canonical hash/CFG1/50steps/10states/144generatedframes/batch128 verified. Frozen runtime_batch_v2 source c9f4f189510bf62e185cf0eda5a8ec92db299e1f3c99d8593f3843a3290eb171, protocol189a016d1d58b5bb8a75d7cfb653fb30eff7ebe7656215145a70e465141b2dd1. /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/local_handoff_firstbatch_20260921T220528Z.json. Partial progress, not final FVD.
- Same-spatial1.56B step2000: online310.39918637 / EMA154.59420000. Global matched at2000:456.20117346 /182.14516165. Fixed bank/seed/protocol and full checkpoint identity2fe72febab54b9b0db4ca776d5096d3bf65695a011ac214b6bc78ff16cf2195f audited: validation/light_fvd_20260921T220400Z.json. Single early step does not prove universal advantage.
- New matched775m online10k and same775m EMA2k audited: validation/light_fvd_20260921T214338Z.json. Same775m online2k remained eligible; failed_targets=[]; old2D stormyxie-sc-xlocal-fvd-0921213949 deleted for actual no_schedulable_H200_node, not model failure. Dev1 sampler now owns target; do not duplicate.
- Same monitor609245/extras461656/local606450 and original348486/348487 remain; preview server452224 and persistent SSH tunnel session96817 untouched. http://127.0.0.1:8884/ , /first-cross/ , /same-spatial/ . Both child progress HTTP verified. No frozen train/eval/config/protocol changed.
- Next: continue actual terminal cleanup and separate H200 evaluation; same training/evals remain active; original20B still must reach10k and all200 scores. Goal is not complete because matched160 is complete.

## 2026-09-21T22:11:27.302767+00:00 — live20B verified wait; first same-local full score completed

Previous goal turn made progress (matched160/final-band completed, completed controllers retired, new same-local GPU handoff verified). This continuation verified actual current Task handles and advancing logs/features, and audited three new completed same-spatial scores. Original goal remains active/incomplete.

- Actual original Tasks stormyxie-sc-20b-cat-0920143127, stormyxie-sc-20b-xa-0920143629, stormyxie-sc-fvdx-0921202511 (concat5000), stormyxie-sc-fvdx-0921210822 (CA4500) all returned Running. Trainer completed steps5324/4683, last30 losses and both diag/grad_norm_post_reduce/post_clip finite. Eval feature counts1792/768 at observation. Proof validation/20b_live_wait_20260921T220815Z.json. That auditor initially looked up absent generic metrics.grad_norm; corrected the schema lookup and rechecked the exact same30 steps. Initial false was absent audit key, not a model NaN; preimage retained.
- Same first dev1 local task stormyxie-sc-xlocal-local-0921215800 completed775m online2000 FVD313.81836817 with2048 cases; PASS present, no FAIL, launcher621738 exited. Controller606450 automatically started 355m online2500 as stormyxie-sc-xlocal-local-0921220933, launcher625332, actual command verified. Proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/local_first_score_complete_20260921T221041Z.json. All4 same-spatial sizes now have both online/EMA scores at2000.
- Same125m step2500 online508.99236104, EMA398.93131440; global matched same-step819.85990289/535.58184705. All3 new scores audited for all2048 fileID sets and score/manifest/protocol/saved-checkpoint metadata, four paired case sidecars each; no independent FVD recomputation. validation/light_fvd_20260921T221019Z.json.
- Same125m remote evals stormyxie-sc-xlocal-fvd-0921220202 and ...0921220251 both Succeeded/deleted per cleanup receipts. Earlier same1560 step2000 evals ...0921214035/...0921214156 Succeeded/deleted and actual pod queries returned0.
- New2D eval admission stormyxie-sc-xlocal-fvd-0921220618 was actually Unschedulable (GPU/CPU/memory/EFA capacity), then deleted by controller after grace as no_schedulable_H200_node. Quota availability alone is not an available whole8-H200 node. Do not relabel this as model failure or restart an existing live training job.
- HTTP /same-spatial/progress.json showed34/160 scores. Original remains177/200; matched160/160 complete. All original and same active runtimes/protocols unchanged; preview/tunnel maintained. Continue the two original20B trainers/full checkpoints and their independent evaluations, normal dedup/terminal cleanup.

## 2026-09-21T22:16:25.435148+00:00 — original20B concat5000 FVD completed and verified

- This goal turn made progress: original20B entry-concat step5000 FVD2048/CFG1/144generatedframes completed, raising original campaign to178/200. Score323.06050136449585, versus4500 score268.6468060552029. Preserve this nonmonotonic measured result; do not smooth away or infer a universal trend.
- validation/light_fvd_20260921T221403Z.json checks full saved checkpoint manifest, all2048 file-ID sets, score/output-manifest hashes, original protocol60d20796dd8e3a6d2b66facfde14c18ab899cd673c8a0b4494444a50bb42e225, sourcea22aa5c3abe00b41bf893bdfe2823e1f4955e6c52a1bcb0fc4f0e1963892f2ed, perGPUbatch32 and four paired cases.
- Additional validation/20b_concat4500_5000_full_pair_20260921T221527Z.json checks ALL2048 paired sampleIDs/seeds/GT/detector/protocol/checkpoint identities and all saved fake/bank feature arrays (finite FP32,400D). Recomputed both FVDs from saved features using existing frechet_distance(ddof=1): exact recorded values, absolute difference0. No RGB regeneration/multiseed significance claim.
- Task stormyxie-sc-fvdx-0921202511 actually Succeeded2026-09-21T22:13:36Z then controller deleted it; actual Pod query returned0. Runtime roughly108m23s fromTask creation. Its checkpoint identity072c4d2473626d5a66f8bdce8853916c2bfe18242f6a97be90f09d9b0f2b1497.
- HTTP root /progress.json now includes concat step5000 (10 score entries); root report reflects178/200. Plot-file observations saved at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/20b_concat5000_completed_20260921T221625Z/validation.json.
- Original20B cross-attn4500 evaluation stormyxie-sc-fvdx-0921210822 remains Running (lastobserved768features); do not restart it. Both original20B trainers remain in progress; latest served concat step5342. Goal remains active until required training/checkpoints/evaluations complete, not achieved at178/200.
- No runtime/config/protocol changes, no new sampler duplicates, no training signals. Keep original/same controllers and persistent preview SSH tunnel96817 alive. CPU pair-check session6356 completed successfully; no pending foreground observer from this turn.

## 2026-09-21T22:19:06.619200+00:00 — live20B wait and same355m online2500

- Actual two20B trainers and CA4500 evaluator still Running; verified completed steps5350/4705, finite last30 loss and post-reduce/post-clip gradients, fresh logs. CA4500 features1024/2048; original178/200. validation/live_training_eval_20260921T221715Z.json. No failure, no restart.
- New same355m online2500 FVD379.39734236907725 versus matched global230.73286615295137 (worse). validation/light_fvd_20260921T221816Z.json; all2048 feature/sidecar file-ID sets, score/protocol/full checkpoint manifest and four paired metadata checked. Identitya44fb8d8c806b0c5e4333a13146f3e7b4c4133e0c58e1de8ef031087bf5eb3cb. This differs from the favorable1.56B early comparison; do not claim universal same-spatial improvement.
- Same campaign35/160 scores at observation; pending-only queue88; extras3 active H200 nodes, failed_targets=[]; original queue1 is currentCA4500. Same cap3 now filled; only expand if actual released H200 capacity supports it, keep max-pending safeguards and user-authorized same fixed protocol. No frozen code/config/protocol changed.
- Current dev1 sampler observation: [{"task": "stormyxie-sc-xlocal-local-0921221725", "name": "1560m_cross_attention__ema", "step": 2500, "pid": 627550, "actual_command": "/efs/stormyxie/venvs/vforge/bin/python /threed-code/stormyxie/vforge_jobs/scaling_same_spatial_ca_20260921T145503Z/fvd_single.py 1560m_cross_attention__ema 2500 stormyxie-sc-xlocal-local-0921221725 ", "features": 0}]. Durable controllers continue; goal remains active/incomplete. Previous turn completed/recomputed originalconcat5000 and cleaned its task; this turn yields a new audited score plus verified live wait.

## 2026-09-21T22:21:49.241039+00:00 — same355m EMA2500 and complete9000 checkpoint

- New same355m EMA2500 FVD210.7029622748737 versus global matched262.0295503958823, while online379.39734237 versus global230.73286615 had the opposite direction. Keep online/EMA separate; no universal method ranking from these early points. validation/light_fvd_20260921T222026Z.json checks same complete checkpoint identitya44fb8d8c806b0c5e4333a13146f3e7b4c4133e0c58e1de8ef031087bf5eb3cb, all2048 fileID sets, protocol and four paired cases.
- Same355m actual train Task stormyxie-sc-355m-xlocal-0921150537 still Running; step9048 at recent report observation. Native09000.dcp manifest completed_steps9000 and weights/optimizer/trainer all present, both metadata files present. Queue contains exactly two distinct online/EMA9000 targets pointing to that full checkpoint; all queue(name,step) identities unique. Proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same355_9000_and_ema2500_20260921T222149Z/validation.json.
- Same355m EMA2500 task stormyxie-sc-xlocal-fvd-0921220921 Succeeded/deleted; new1560m online2500 task stormyxie-sc-xlocal-fvd-0921221916 actual observation: {"observation_error": "Error: \u83b7\u53d6\u4efb\u52a1\u8be6\u60c5\u5931\u8d25: \u4e1a\u52a1\u9519\u8bef (code=1002): task not found: stormyxie-sc-xlocal-fvd-0921221916\n"}. Dev1 local1560m EMA2500 remains launcher627550 under controller606450, previous turn verified actual cmd.
- Same scorecount36 at observation, original178, matched160. Actual originalCA4500 Task Running with1024 features at start of turn. Original objective still incomplete; no code/protocol edits or training restarts. Continue monitoring and evaluate only committed full checkpoints.

Admission follow-up 2026-09-21T22:23:46.485958+00:00: old1560m online2500 Task ...0921221916 was actually in tenc-szr3212j (2D). Initial direct lookup assumed AWS2B; audit record now explicitly labels that observation-routing error. The old task had an actual Unschedulable Pod and was deleted by controller; correct2D Task query nowNotFound and Pods0. Target remains unique/failed_targets=[], and replacement stormyxie-sc-xlocal-fvd-0921222216 in tenc-awsusw2 was actually confirmed Running. Proof: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/1560_online2500_admission_20260921T222230Z.json. Do not restart either existing trainer or deleted admission task.

## 2026-09-21T22:28:35.071575+00:00 — same775m step2500 online/EMA pair

- New same775m step2500: online369.0607178198711 versus global478.95516431306123 (lower); EMA170.66891981867911 versus global154.23035071114418 (higher). Online and EMA rankings disagree, as for355m with opposite signs; do not claim universal same-spatial/global superiority from these early checkpoints.
- Both new same scores audited against fixed matched references: validation/light_fvd_20260921T222453Z.json and validation/light_fvd_20260921T222735Z.json. Same checkpoint identityfb0b65c0262c340654c2ed3342a4021f09d4a5b21be6fc82f70947b940e928c6, saved full manifest/protocol/all2048 fileID sets/four paired case metadata; no RGB regeneration or independent FVD recalculation here.
- EMA task stormyxie-sc-xlocal-fvd-0921221010 actually Succeeded/deleted and correct receipt-cluster Pod query0: /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/775ema2500_cleanup_20260921T222646Z.json. Online task cleanup snapshot at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same775_2500_pair_20260921T222835Z/validation.json; do not infer terminal from score alone.
- Actual original two20B trainers and CA4500 evaluator queried Running at22:24:30UTC; CA4500 had1024features then. Original178/200 remains incomplete. Dev1 actual launcher627550 continues1560m EMA2500 (512features at22:24); remote1560m online2500 actualRunning onAWS2B ...0921222216. Next same125m EMA3000 ...0921222336 Running onAWS2B. Keep protocol/runtime immutable and existing dedup/cleanup controllers active.
- This turn yielded two new verified results plus a specific live-task wait. No user-action blocker, no training restart, no goal completion claim.

## 2026-09-21T22:33:07.637443+00:00 — same125m EMA3000, verified live20B wait

- New same125m EMA3000 FVD342.0436701998947, global matched370.5929741844243. validation/light_fvd_20260921T223157Z.json checks2048 feature/sidecar ID sets, fixed protocol, full saved checkpoint manifest and four paired case metadata; checkpoint identitydb8efabf4f963813fa7e4e5c495d903f7ed363bb1376df5d3fca8d4f324d9705. Cleanup snapshot at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same125_ema3000_20260921T223307Z/validation.json.
- At22:29:39UTC both original20B trainers and CA4500 evaluator actually queried Running; completed train steps5384/4736 with last30 loss/gradient metrics finite. Original remains178/200. No failed state or model restart.
- At22:31:35 actual dev1 launcher627550 still matched expected1560m EMA2500 command,1536/2048features. Remote1560m online2500 ...0921222216 actually Running onAWS2B,512features. Keep waiting on these exact live handles, not duplicate submissions.
- Current remote same125m online3000 ...0921222732 is AWS2B. Same campaign now includes completed125EMA3000 in addition to38 previously audited points. Preserve online/EMA separate comparisons and frozen runtime/protocol. Goal remains active; this turn produced a new audited score plus verified live wait.

## 2026-09-21T22:38:14.817943+00:00 — same1.56B EMA2500 complete; common four-scale EMA comparison

- Same1.56B EMA2500 FVD133.53748623661568 versus global matched138.7484633410504. New validation/light_fvd_20260921T223711Z.json audits2048 fileID sets, full checkpoint/protocol/hash and four paired metadata cases. Checkpoint identity394a05a450e111e38bcfdf50c27bb5f02bc54b427275a8a7f5f58bd4e348a6db.
- All four sizes now have common step2500 EMA .99 / CFG1 / fixed2048 /144generated-frame comparison:

| Size | Same-spatial EMA FVD | Global EMA FVD |
| --- | ---: | ---: |
| 125m | 398.93131440 | 535.58184705 |
| 355m | 210.70296227 | 262.02955040 |
| 775m | 170.66891982 | 154.23035071 |
| 1560m | 133.53748624 | 138.74846334 |

- Early single-seed results; keep online/EMA distinct and retain measured points. Same1.56B online2500 is still an active remote evaluation; do not claim that pair complete before its score arrives.
- Local task stormyxie-sc-xlocal-local-0921221725 PASS verified, launcher627550 exited, no FAIL. Next dev1 actual sampler: [{"task": "stormyxie-sc-xlocal-local-0921223611", "name": "775m_cross_attention__ema", "step": 3000, "pid": 632712, "actual_command": "/efs/stormyxie/venvs/vforge/bin/python /threed-code/stormyxie/vforge_jobs/scaling_same_spatial_ca_20260921T145503Z/fvd_single.py 775m_cross_attention__ema 3000 stormyxie-sc-xlocal-local-0921223611 "}]. Full proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same_all_ema2500_20260921T223814Z/validation.json, including HTTP report scorecount41.
- Also new same125m online3000 FVD425.2423131368464 versus global651.1157175277863; validation/light_fvd_20260921T223535Z.json. Same125m EMA3000 was342.04367020 versus global370.59297418, same checkpoint identitydb8efabf4f963813fa7e4e5c495d903f7ed363bb1376df5d3fca8d4f324d9705.
- OriginalCA4500 Task actually Running at22:34:13UTC and1280features. Original scaling178/200 remains incomplete,20B trainings continue. No runtime/config/protocol changed and no original trainer restarted. This turn made concrete new-score/completion evidence progress while preserving full goal.

## 2026-09-21T22:44:48.620520+00:00 — same1.56B online2500 complete; all four common pairs available

- Same1.56B online2500 FVD306.21400542365654 versus global matched285.18285458333503 (higher); EMA133.53748624 versus global138.74846334 (lower). Same checkpoint identity394a05a450e111e38bcfdf50c27bb5f02bc54b427275a8a7f5f58bd4e348a6db. New audit validation/light_fvd_20260921T224324Z.json checks fixed2048file IDs/protocol/fullcheckpoint/output-manifest hash/four paired case metadata.
- Four sizes now have complete online/EMA pairs at2500; same online/EMA checkpoint identities matched for all4. Aggregate values and checked manifests at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same_all_pairs2500_20260921T224448Z/validation.json. Preserve online/EMA distinct comparisons.
- Native new checkpoint observations: [{"name": "1560m_cross_attention", "step": 6000, "complete": true, "queued": ["ema", "online"]}, {"name": "355m_cross_attention", "step": 9500, "complete": true, "queued": ["ema", "online"]}, {"name": "125m_cross_attention", "step": 9000, "complete": true, "queued": ["ema", "online"]}, {"name": "775m_cross_attention", "step": 9000, "complete": true, "queued": ["ema", "online"]}]. The bounded foreground observer confirmed355m9500 full commit before queue ingestion, then automatic online/EMA enqueue, plus125m9000 and1560m6000; no premature incomplete checkpoint evaluation. Proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/new_cp_online2500_wait_20260921T224305Z.json. Observer session16991 completed; no pending foreground wait.
- Actual remote1.56B online2500 Task ...0921222216 Succeeded2026-09-21T22:42:44Z. Current cleanup snapshot: deleted=True, actualcorrect-clusterPodCount=0; see aggregate proof. Do not restart completed task.
- HTTP same report scorecount44. Original objective still178/200, both20B trainings andCA4500 evaluator continue; lastactualstatuses Running at22:39 and1280evalfeature progress verified earlier. Keep originalgoalactive, frozenruntimes/protocols unchanged, no training signals.

## 2026-09-21T22:54:42.890215+00:00 — CPU final-EMA auditor installed and live; new3k results

- Reused the existing read-only matched audit_checkpoint_ema.py in same-spatial job directory, adding only assertion world.cross_attention_scope==same_spatial. Native train/eval/config/protocol snapshots unchanged; compile/source/preimage evidence at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/final_ema_auditor_prepare_20260921T224851Z/validation.json.
- Real355m9500 preflight PASSED: initialized, updates9500, decay.99, all436 named trainable tensors exactly matched metadata/model shapes, numel354975424, FP32 EMA shadows, largest tensor8shards, CA onlyblock0 beforeSA, FFNhidden3008. /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/production_step9500_ema_audit.json (complete=false means other models do not yet have9500, not audit failure).
- Final-checkpoint watcher **PID636918**, script same job audit_checkpoint_ema.py --step10000 --watch, running CPU-only CUDA_VISIBLE_DEVICES empty; actual cmd/fresh status/nvidia-smi PID absence verified. /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/final_ema_auditor_launch_20260921T225052Z/validation.json and final_ema_auditor_verified_20260921T225203Z.json. services.json now includes this fourth service. Log /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/audit_checkpoint_ema_final.log; status /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/production_step10000_ema_audit.json; exclusive lock checkpoint_ema_10000.lock. Do not start a duplicate watcher.
- Scope is metadata/model-on-meta names+shapes/FP32 shadow inspection and loaded small EMA buffers; no GPU restore, no all-weight numeric scan, no training cleanup. Watcher will audit each committed10000 checkpoint and exit only afterall4. Before retiring completed trainers still verify actualTaskSucceeded+PASS,exact10k logs,all20full checkpoints and finaloptimizer/trainer metadata/rankstates; the EMA check alone is not full training completion.
- CAUTION: matched audit_completed_training.py was inspected but NOT copied/run: it assumes unaliased FVD targets and fvd/status.json, incompatible with same-spatial online/EMA aliases without adaptation. Use correct actual task receipt cluster in observations.
- New same355m3000 online272.19478764504265 / EMA168.09867631621046; global333.19413828665756 /199.55694255615845. validation/light_fvd_20260921T224643Z.json, identity7c546178f0981264ca36398d8867cbee4f5b593c410b471770d88e6fe745d77d.
- New same775m EMA3000 FVD132.5695375491391 versus global130.7749269501992. validation/light_fvd_20260921T225303Z.json, identity2b226f0292c687c1c1c96c5f1c64f3d14b1ab616cfbd96c0a705c235bb925013. Both are fixed2048 audits with allfile IDs,fullcheckpoint/protocol/hashes and four paired cases, not RGB regeneration.
- Original two20Btrainers and CA4500 eval freshly queried thisturn: [{"task": "stormyxie-sc-20b-cat-0920143127", "cluster": "tenc-awsusw2", "status": {"creationTimestamp": "2026-09-20T14:31:29Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-20b-cat-0920143127 is running.", "status": "Running"}}, {"task": "stormyxie-sc-20b-xa-0920143629", "cluster": "tenc-awsusw2", "status": {"creationTimestamp": "2026-09-20T14:36:30Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-20b-xa-0920143629 is running.", "status": "Running"}}, {"task": "stormyxie-sc-fvdx-0921210822", "cluster": "tenc-awsusw2", "status": {"creationTimestamp": "2026-09-21T21:08:23Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-fvdx-0921210822 is running.", "status": "Running"}}]. Original178/200 stillincomplete, same45+scores, matched160complete. This turn added working CPU audit automation plus3 audited FVD points; no blocker/goalcompletion/trainingrestart.
- Same355m wasstep9591 at22:46, approaching10k. When actual completion is verified and its H200 node released, consider increasing same extra sampling capacity fromcurrent3 if actual available capacity supports it; user authorized more H200 sampling. Retain max_pending safeguards/dedup; do not treat namespace quota alone as a free8GPU node.

## 2026-09-21T23:20:02.260088+00:00 — same355 full10k complete/cleaned; cap4 orchestration repaired; 1.56B progressing

- Same355m reached actual TaskSucceeded at23:04:39UTC. PASS/monitor lag resolved naturally; no training restart. New external JL/audit_completed_training.py adapts matched auditor to online/EMA aliases, optional primary/extras/local status, all40 eval targets and finite post-reduce/post-clip gradient norms. The legacy JM script warning above remains valid; use this new JL script for remaining same-spatial trainers BEFORE deleting their Tasks.
- Full355 completion audit PASSED: exact10000 logged optimizer steps, finite loss/gradient norms,20 full checkpoints every500, Native DCP metadata/chunk bounds,8 CPU trainer rankstates/scheduler.last_epoch10000, all40 online/EMA eval targets completed or tracked, fixed same_spatial/before_self/FA4/mask80/local8/no-zprev recipe. Evidence RL/validation/training_complete_355m_cross_attention.json. New auditor prepare/alias preflight proofs: training_completion_auditor_prepare_20260921T225824Z/validation.json and training_completion_alias_preflight_20260921T225935Z.json.
- Existing CPU final EMA watcher PID636918 audited355 final: initialized, decay.99, updates10000, FP32 shadows/names/shapes,354975424params. RL/validation/production_step10000_ema_audit.json remains complete=false only because other3 runs are not10k yet. Watcher stays live; do not start a duplicate.
- Deleted ONLY the verified-complete355 trainer stormyxie-sc-355m-xlocal-0921150537 on AWS2B. ActualTaskNotFound and zeroPods, all20 checkpoint manifest hashes retained. RL/validation/355_training_cleanup_20260921T230729Z.json and reuse_completed355_h200_20260921T230931Z/validation.json. Report correctly calls it complete despite schedulerNotFound; DO NOT RESUME355. Bounded observer session37778 finished, no pending foreground completion observer.
- Cap3→4 control change at23:09 initially hit hardcoded MAX_NODES=3 and generated AssertionError before status publication. Existing GPU samplers continued; controller was alive but cap4 was NOT effective. Fixed only JL/fvd_extras.py constant to4 (SHA3a42ea52c49f234b6a63674f59ab19a9eb72ffa37ac2bb7c4d7635c45c075a5d), preserving source/control/services preimages. Replaced only CPU orchestration PID461656 with PID644285 under existing controller lock and receipts/leases. No native runtime/config/protocol or GPU trainer/evaluator changed. services.json updated.
- Repair evidence RL/validation/extra_cap4_fix_20260921T231738Z/validation.json; actual new controller publishes cap4, maxlaunch1/maxpending1, no new errors. RL/validation/extra_cap4_verified_20260921T231902Z.json. At23:19 extra fourth Task ...0921231749 (125EMA4000,H2002D) was Updating with Pending unassignedPod; do not claim four runningnodes yet. Existing AWS eval ...0921230524 (775online3500) Pod just Succeeded; ...0921230617 (1560EMA3500) and ...0921230738 (775EMA3500) Running. Controller handles normal terminal cleanup/120s provenUnschedulable/cooldown; observations alone are not failure.
- At23:18 actual trainers Running: same125/775/1560, original20Bconcat/allCA. Report steps same9721/10000/9683/6700; original5519/4855. Same52 scores, original178/200, matched160/160. Original20B long-running goals remain incomplete; keep goal active and monitors running.
- Audited new same1560 step3000 pair: online378.1100640028967 vs global361.2555259514661; EMA104.18870097674926 vs global111.13687556068408. Same checkpoint identity9772f8b5e0ff09a3af7de142fd6206df63ac7d6f37d8654dfda98c7b33dff111. Also125online3500=510.83294302769445,355online3500=226.7001059753229,355EMA3500=155.19648868907097. R0/validation/light_fvd_20260921T231824Z.json checks all2048file IDs, outputmanifest/fullcheckpoint/protocol and four paired case metadata per point; no FVDrecompute/RGBregeneration. Earlier light_fvd_20260921T231006Z.json covers775online3000 and125EMA3500.
- Active local same sampler at23:18:1560online3500, dev1 launcher641363,1024/2048, controller606450. Keepdev1 single sampler. Same monitor609245; finalEMAwatcher636918. Preview server452224 and persistent tunnel session96817 retained; local HTTP /same-spatial/ returned200 at23:18. No reason to restart trainers or completed matched campaign.

## 2026-09-21T23:28:30.522797+00:00 — original179/200; CA4500 independently recomputed; concat5500 complete/enqueued

- Previous goal turn made concrete progress repairing same extras cap4. This turn verified a new original full checkpoint and completed a new original FVD point, with independent saved-feature recomputation. Full original goal remains ACTIVE: original20B training not10k and21FVDpoints stilloutstanding. Do not mark complete or blocked.
- Original20B concat5500 full checkpoint audit passed: Native component metadata integrity;12,279weight and21,369optimizer DCPchunks withinfilebounds; all8 CPU trainer states scheduler.last_epoch5500, dataloader epoch2/batch1136; FA4/no fallback/local8/mask80/latent/no-zprev/entryconcat. All11 checkpoints500..5500 retained. R0/validation/20b_concat5500_checkpoint_20260921T232157Z.json. RawmanifestSHA152826d3eb8eb3d164d3f3ba30b0eabc37d97afcc73f256b192365b9ac8bef50; canonicalSHA bd40f495e91e81e2d474694e7e47d76bc090f4f2b5fec18640ca9d6a2c9f59b7. No full payload rehash/GPUrestore.
- Bounded live monitor348486/extras348487 observer session13849 finished (not running). Atitsdeadline concat5500 had1queueentry but0publishedactive; normal nexttick subsequently submitted Task stormyxie-sc-fvdx-0921232401 onH2002D. At23:25:31 actualUpdating/PendingunassignedPod, notfailure. Current published active: [{"task": "stormyxie-sc-fvdx-0921232715", "cluster": "tenc-awsusw2", "name": "20b_entry_concat", "step": 5500, "scheduler": {"creationTimestamp": "2026-09-21T23:27:16Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-fvdx-0921232715 is running.", "status": "Running"}}]. Do not restart on observation expiry; letexisting admission/cooldownhandleactualresource shortage.
- Original20B CA4500 finished TaskSucceeded23:22:29UTC; controllerdeletedtask; actualcorrectclusterAWS2BTaskNotFound andPods0. Fullresult2048cases retained, originalscorecount179. R0/validation/ca4500_complete_concat5500_handoff_20260921T232531Z.json (alsoactualsame4extraTasksRunning,rootHTTP200).
- CA FVD4000=275.33816344335537,4500=150.87807302610372. LightauditR0/validation/light_fvd_20260921T232504Z.json. Becauseoflargeimprovement, reusedexisting read-only audit_new20b_20260921T1522Z.py in-memory withtargets4000/4500 anduniqueoutputdir. No live script/native snapshot edits. All2048 features/caseidentities/seeds/GTpixel/detector/config/protocol checked forboth; independentfloat64 scipy.sqrtm covariance recomputation agrees: errors5.024958227295429e-10 and6.411937647499144e-11. Proofdir /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/fvd_ca4000_4500_20260921T232634Z; sameGTaggregate4ad523cb85686fee12060661058b4e791f5a4f506d23afd1bf21e826f1e2a8ae. Scope savedfeatures/sidecars only, noRGBregeneration/newdetector/multiseedCI. CPUaudit session98843 completed.
- Both20BtrainingTasks actualRunning23:20/continuedfinitelast30loss+gradientlogs23:21. Latestmonitor5536/4870 near23:27; remainlive. Originalmonitor348486/extras348487 unchanged, sameextras644285 cap4 healthy. Extra1.56B3500 and125/3554000 etcadvance; mostrecent published samecounts/steps 57, [{"name": "125m_cross_attention", "status": "training", "completed_steps": 9923}, {"name": "355m_cross_attention", "status": "complete", "completed_steps": 10000}, {"name": "775m_cross_attention", "status": "training", "completed_steps": 9885}, {"name": "1560m_cross_attention", "status": "training", "completed_steps": 6854}].
- New same7753500 pairauditR0/validation/light_fvd_20260921T232229Z.json: online260.5534213557771 vs global313.44881984328686;EMA122.22483116385774 vs global114.90147473527486. Samecheckpointidentityc60787f5be907a6686b51b6aa8cf05742f09cf222827a3da4a2ee2218e5de93e. This islightallfileID+4pairedmetadata audit, notfullrecompute.
- Upcoming same125/775near10k: use JL/audit_completed_training.py beforedeletion, pairwithCPUfinalEMAauditor636918 perrunproof, actualTaskSucceeded+PASS/full10000. Same355alreadyverifiedcomplete/deleted; DO NOT RESUME. No pendingforegroundtool sessions afterthisrecord; keep previewtunnel96817 andservers/controllers.

## 2026-09-21T23:38:28.067151+00:00 — same125/775 finished10k and cleaned; six extra H200 samplers verified

- This goal turn made concrete completion/cleanup/resource reuse progress. Original goal still incomplete (originalscorecount179, both20Btrainers continue); keep ACTIVE. Do not redefine success around completed small/auxiliary campaigns.
- Same125 TaskSucceeded23:31:47UTC and775 TaskSucceeded23:34:03UTC. Bounded observer shell28746/PID648521 finishedsuccess, audited both; no pendingforegroundobserver. RL/validation/125_775_completion_watch_20260921T233025Z.json complete=true. Brief file/queue visibility lag resolved naturally, no training restart.
- Both full completion gates passed viaexisting JL/audit_completed_training.py: exact10000 logged optimizersteps/allloss+postreduce/postclipgradfinite,20fullCP500..10000fixedrecipe/nativecomponentmetadata,finalDCPchunkbounds,8CPUtrainerrankstates/scheduler10000 anddataloadercontract,40online/EMAevaltargets completedortracked. RL/validation/training_complete_125m_cross_attention.json and training_complete_775m_cross_attention.json.
- FinalEMA watcher636918 verified both10k updates,decay.99,initialized,FP32shadownames/shapes:125044096 and775044480params. production_step10000_ema_audit.json nowcontains125/355/775; overallcompletefalse because1.56Bstilltraining. Watcher remainslive; do not duplicateorstopit.
- Fresh actualSucceeded beforedeletion; deleted ONLY completed125/775TasksonAWS2B. ActualTaskNotFound andPods0; all20manifesthasheseachstillidentical. Proofs RL/validation/125_training_cleanup_20260921T233405Z.json and775_training_cleanup_20260921T233525Z.json. Alongwithprevious355, allthreesame smalltrainerscomplete/deleted; DO NOT RESUME anyofthese.
- Reusedreleasedcapacity: JL/fvd_extras.py MAX_NODES4→6, controlmax_extra_nodes6, maxlaunch1/maxpending1unchanged. StoppedCPUcontroller644285onlywhileverifiedidle(hrtimer_nanosleep,nochildren),preservedreceipts/leasesandcheckedlockrelease; newPID650662. SourceSHA841ccef3535318c1cd55a793d0718a1c5ff559ff8da704beedc9d804a8280712. services.json updated. Source/control/servicespreimages at RL/validation/extra_cap6_reuse_20260921T233604Z/validation.json. No native training/eval/config/protocol changes or GPU task signals.
- Cap6 actuallyverified with6differentAWSH200remoteTask+PodRunning,no newcontrollererrors,unique(name,step): RL/validation/extra_cap6_verified_20260921T233738Z.json. Tasks ...0921232648(775EMA4000),...0921232736(775online4000),...0921232855(1560EMA4000),...0921233532(1560online4000),...0921233615(125online4500),...0921233700(355EMA4500). This isextraremote6; dev1 singlelocal sampler606450continues separately. Do notstartanotherdev1sampler.
- New fixed2048 lightaudits: R0/validation/light_fvd_20260921T233044Z.json coverssame1560online3500=222.59201766314982/global251.83162242207845 andEMA3500=109.83400971710716/global118.18427306306239;125online4000=476.14561301402864/global542.4884088638405 andEMA4000=256.127123166229/global243.09930704696808. New R0/validation/light_fvd_20260921T233712Z.json covers355online4000=230.60640113992986/global280.44384518666493 andEMA4000=138.10119964036858/global127.18987993263272. Scopeall2048fileIDs,protocol,fullCP/outputmanifesthash,fourpairedcases; nofullrecompute/RGBregeneration. Rankingsvarybyweight/scale/step; nosingle-seeduniversalityclaim.
- Current actualtasks and reportsnapshot saved /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/same_spatial_ca_20260921T145503Z/validation/small_final_cap6_handoff_20260921T233828Z/observations.json; same1560step7000checkpoint/queueobservation {"complete": true, "queued": [], "manifest_sha256": "0d0c4bbeb6585dad0d33c16c1b44576db90fc33ed816ea8227c04accd5c9388a"}. Original20Bconcat5500evaluator ...0921232715/AWS actualobservedalongwithtrainers; do notreviveobsolete2Dadmission ...0921232401(alreadydeleted/Pods0lastturn).
- Canonicalservices: originalmonitor348486/extras348487; samemonitor609245/extras650662/local606450/finalEMAwatcher636918; preview452224 + persistentSSHtunnel96817. LocalHTTP /same-spatial/ returned200at23:37. No foregroundexecwaits pendingafterthisrecord; preservealltunnelsanddurablemonitors.

## 2026-09-21T23:43:59.588609+00:00 — concat5500 first batch verified; same1560_7000 both aliases queued

- Previous goal turn completed/cleaned same125/775 and verified6extraH200samplers. This turn confirmed live originaltraining/evaluation, newtargetqueuehandoff and actualfirstbatchprogress plus2newFVDaudits. Goal remainsACTIVE/incomplete: original179/200, both20Btrainers stillrunning. No genuineblocker; no restart on oldlogmtime orzero initialfeatures.
- Same1.56B step7000 completecheckpoint (rawmanifestSHA0d0c4bbeb6585dad0d33c16c1b44576db90fc33ed816ea8227c04accd5c9388a lastturn) nowonline+EMA bothautomaticallyqueued, confirmed23:39 andagainnow. Latestreportsnapshot /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/firstbatch5500_and_same7000_20260921T234359Z/observations.json.
- ActualTaskRunning queries23:39: original20Bcat/...0920143127,allCA/...0920143629,originalconcat5500FVD/...0921232715, same1.56B/...0921155005 andall6sameextraevaluators. OriginalCPUcontrollers348486/348487,same609245/650662/606450/636918 retained. No runtime/config/protocol changes.
- Investigatedinitialzero featurefiles on20Bconcat5500withoutassumingstall: boundedEFSlogtail hadonlyexistingautocast warnings, noFAIL; read-only Koala podexec on stormyxie-sc-fvdx-0921232715-worker-0/containerworker/AWS2B showedall8H200GPUutil100pct,memory119341..122413MiB. Nativebatch32reachedfirst256features+256sidecarsat23:43:00UTC. Proof R0/validation/20b_concat5500_firstbatch_wait_20260921T234300Z.json checks4savedcases0000..0003 finiteFP32/400D,batch32/CFG1/50Euler/144metricframes/fullcheckpointstep5500. This provespartialprogress, NOT acompleteFVDscore/all256numeric scan. Foregroundobserver4453 completedsuccess; no pendingtoolwait.
- Lightaudits R0/validation/light_fvd_20260921T233954Z.json: same125EMA4500=218.5890038636926 vs global203.04902242466778, identityd3b0d2e0e586168b1b314f7c41877db3a1e69b0b393e6c58c12a2713df2a1d8b; same775EMA4000=108.00202885789531 vs global115.91794932516359, identityaa39c63201e668c490db0dd3e2ead1c265d0ba380d00f96f9204ccfc69b5a11a. Checksall2048fileIDs/protocol/fullCP/outputmanifest/fourpairedcases, noindependentFVDrecompute/RGBregeneration.
- Current samecompletedscorecount64. Same125/355/775trainingfinished10kandTasksdeleted, neverresume;1.56Bcontinues. Preserveallcheckpoints,backgroundFVDqueue/leases/controllers,previewserver452224andSSHtunnel96817.

## 2026-09-21T23:51:01.152312+00:00 — allfour same/global EMA4000 pairs available

- This turn verified newFVDresults/common-stepcomparison andlivejobs; originalgoalremainsACTIVE/incomplete179/200 withtwo20Btrainers+concat5500evaluator. ActualTaskRunning23:44:53: originalcat...0920143127/allCA...0920143629/eval...0921232715; originaleval256/2048then. No training/runtime/config/protocol changes or restarts.
- Liveboundedobserver shell42103 completed aftersame1.56BEMA4000 progressed1536→1544→2048→score. Proof RL/validation/1560_ema4000_wait_20260921T234549Z.json; finalTaskNotFound meansnormalpostsuccesscleanup, NOT failure. ReceiptactualSucceeded/deletedandcorrectclusterPods0verified in /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same_all_ema4000_20260921T235101Z/validation.json. No pendingforegroundobserver.
- New same1.56BEMA4000 FVD116.46019808720212 versusglobalfirstCAEMA93.9425407508943; checkpointidentity20e8be2abf191242b5c27ce0f9b95f80d267ef54449005db6d7a27681cf4fd45. R0/validation/light_fvd_20260921T234959Z.json checksall2048fileIDs/fullCP/outputmanifest/protocol/fourpairedcases; notfullFVDrecompute/RGBregeneration.
- Allfour4000EMA pairs: [{"size": "125m", "step": 4000, "same_spatial_ema_fvd": 256.127123166229, "global_first_ca_ema_fvd": 243.09930704696808, "difference": 13.027816119260933}, {"size": "355m", "step": 4000, "same_spatial_ema_fvd": 138.10119964036858, "global_first_ca_ema_fvd": 127.18987993263272, "difference": 10.911319707735856}, {"size": "775m", "step": 4000, "same_spatial_ema_fvd": 108.00202885789531, "global_first_ca_ema_fvd": 115.91794932516359, "difference": -7.915920467268279}, {"size": "1560m", "step": 4000, "same_spatial_ema_fvd": 116.46019808720212, "global_first_ca_ema_fvd": 93.9425407508943, "difference": 22.51765733630782}]. Exactscoreprotocol equality acrosssame/global per-sizeverified. Table comparesmatchedFFN-compensatedFIRST-layerCA variants, notoriginalall-layerCA. Globalisbetterhereat125/355/1560, sameisbetterat775; onecheckpoint/one trainingseeddoesnotestablishuniversalordering. Aggregateproof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same_all_ema4000_20260921T235101Z/validation.json; reportHTTP200.
- Othernewlightaudits: R0/validation/light_fvd_20260921T234549Z.json for775online4000=286.9068573473887/global215.53887469442316 and125online4500=390.16852140727315/global365.6974172383316. R0/validation/light_fvd_20260921T234959Z.json also355EMA4500=131.09584592998863/global119.89281782158378 and355online4500=144.45566371346968/global151.41796845271256, pairedcheckpointidentity26f83a7aa64492ab34f981ebe7c2c52eb436126e016a3511eafb1bba1991328e.
- RootreportHTTP/5plotpaths (training_loss,fvd_progress,loglog,loglog_ols,fvd_scaling) all200 thisturn; originalprogress containsCA4500=150.87807302610372. No plotstyle changes. Samecompletedscores67. Canonicalcontrollers/tunnelunchangedfromprevioushandoff, sameextras650662 cap6; keep previewSSH96817. Same125/355/775trainerscomplete/deleted: DO NOT RESUME.

## 2026-09-21T23:59:29.491328+00:00 — second20B5500 samplebatch and refreshed production timing report

- Goalturnclassification: progress (real secondbatch, repaired/refreshedtimingreport), withverifiedwaitonexactliveTask. Originalobjective stillincomplete/ACTIVE,179/200; both20Btrainers continuing. FreshactualTaskRunning23:51:45 forcat...0920143127,allCA...0920143629,eval...0921232715. Directtraininglogs5615/4940then, recent30loss+postreduce/postclipgradmetricsfinite. No new5000CAcheckpointyet atthatobservation; do not synthesize/enqueueincompletecheckpoints.
- Exactliveconcat5500evaluator progressed256→512 feature/sidecarpairs duringboundedobserver2033; observerfinishedsuccess. Proof R0/validation/20b_concat5500_secondbatch_wait_20260921T235614Z.json (actualRunning23:56:14,512features/512sidecars). No newnumericparityclaim, no completedFVDscoreyet. PreserveNativeeval, do not restartlongsamplerbecausefeaturebatchesarrivesparsely.
- Foundproduction_timings.json stalefrom06:34UTC (103observations). Existing J0/refresh_production_timings.py initialrefreshrun failedassert beforeanyreportmutation on2historicrecords whose score.finished islaterthanmatchingPASS.time:355m_cross_attention/07500 by81.5262599s and775m_cross_attention/10000 by77.4679363s. This does NOT establishtraining/evaluationfailure or invalidFVD; elapsedtimefromtheserecordsisambiguous.
- Fixedonlytimingaggregation toexclude non-finite/non-positive/non-monotonic elapsedrecordswithexplicittarget/start/score/PASStimestamps, updatingreportexclusionwording. AllFVDscoresretained. Sourcepreimage/ASTproof R0/validation/timing_timestamp_fix_20260921T235713Z/validation.json. New J0/refresh_production_timings.pySHA4a656ecf5feb53659fb4d4db4572ea98acb583f5045d34834b6d4d42f9b2fd00. Native train/eval/runtime/config/protocolsnapshotsandcontrollersunchanged.
- RealfullrefreshPASSED with162cleancompleted8-H200observations(includesdev1andremote). Explicit2timestampregressiontargetsexcluded, positive-cleanacceptedrecordlogic; all179existingFVDscorefilesSHAunchanged;HTTP200/10modelrows/exclusionnoteverified. Proof R0/validation/production_timing_refreshed_20260921T235751Z/validation.json; reportdata R0/fvd/production_timings.json androotindex details production-timing-details updatedwithpreimages. Scriptisstandalone reportrefresh; notnewGPUwork ornewmonitorservice.
- Timingdefinition:workerRUNNING→scorefinished, includingload/compile/sampling/I3D/FVD, excludingqueue; omitsCPUrecovery,failed/ambiguousretry/missinglaunch/badtimestamprecords. Medianminutes entryconcat/all-layerCA:125M5.899/6.876(n20/18),355M7.922/9.633(n20/18),775M10.969/13.606(n19/18),1.56B17.021/22.033(n16/16),20B108.667/137.063(n10/7). TheseareORIGINALconcat/all-layerCAjobs, notauxiliaryfirstCA/same-spatialtrainingbenchmarks orconfidenceintervals.
- Currentcounts/steps /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/timings_and_secondbatch_handoff_20260921T235929Z/observations.json; same72scores. Same125/355/775trainingcomplete/deleted;1.56Bstilllive. Originalcontrollers348486/348487,same609245/650662(cap6)/606450/636918 andpreview452224+SSH96817retained. All foregroundaudit/watchsessions inthisturnfinished; no pendingexecwait.

## 2026-09-22T00:03:13.333285+00:00 — all4000 online/EMA pairs verified; new1255000/7754500

- Previousgoalturnadvancedsecondbatchandtimingreport; thisturnverified5newFVDpoints/fullcommon4000pairs andlivejobs. OriginalgoalstillACTIVE/incomplete179/200; no failure/restart/blocker. ActualTaskRunning00:00:40UTC:20Bcat...0920143127,20BallCA...0920143629,same1.56...0921155005,originalconcat5500FVD...0921232715. Originaleval512features then, extrasfailed_targets emptybothcampaigns.
- New R0/validation/light_fvd_20260922T000124Z.json: same1255000 online296.6554734602546/global372.5584999172502,EMA186.42955252243746/global184.1012214740099; pairedidentitybba9b9d58d58cb70125b402b88919cc630825c7b66c2648a2faa5ca70889b979. Same7754500 online248.36183126710694/global184.18528822051348,EMA100.38226528625819/global94.67359686397685;identity544de71096c7a711aa7bcaf1e6de0cecb98e53ea3e2845b7688980c461f514eb. Same1560online4000=230.49854853865554/global175.65871595933822,pairedidentity20e8be2abf191242b5c27ce0f9b95f80d267ef54449005db6d7a27681cf4fd45. Scopeall2048fileIDs/fullCP/protocol/outputmanifesthash/4pairedmetadata; no independentrecompute/RGBregeneration.
- Allfour4000 online+EMA pairs nowavailableagainstglobalFIRST-layerCA, andonline/EMA checkpointidentities matchedwithinall8modelcampaignpairs. Aggregate /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same_all_pairs4000_20260922T000313Z/validation.json with16scores. Fixed2048/CFG1/144generatedframes; do notmixonlinewithEMA orcomparethislabeltooriginalall-layerCA.
- Same1560online4000Task ...0921233532 actualSucceeded receiptanddeleted, correctclusterPods0verified; nativeFVDresultretained. No cleanup/restartofotherjobs. Samecompletedscorecount72, original179; continuingstates original[{"name": "20b_entry_concat", "status": "training", "completed_steps": 5635}, {"name": "20b_cross_attention", "status": "training", "completed_steps": 4957}], same1.56[{"name": "1560m_cross_attention", "status": "training", "completed_steps": 7393}].
- No source/runtime/config/protocol/controller changes thisturn. Timingsreport162observations andtimestamp-exclusionfixfromlastturn retained; sameextras650662 cap6, local606450,monitor609245,CPUfinalEMA636918,original348486/348487. Same125/355/775complete/deleted; DO NOT RESUME. Preview452224/SSH96817preserved. All foregroundtool sessionsfinishedafterthisrecord.

## 2026-09-22T00:14:25.626456+00:00 — same1560_7500 full commit/EMA audit/online+EMA enqueue

- Thisgoalturnmadeconcreteprogress: observed1.56B7500fullcheckpointcommit, verifiedactualEMAbuffers+metadata forall4size7500points, verifiedbothnewaliasesqueued, audited8newFVDpoints. OriginalgoalstillACTIVE/incomplete179/200; continuingoriginal20Btraining. Actualtaskstatuses/stepsat /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same7500_ema_queue_20260922T001425Z/validation.json.
- Boundedlive1.56Btrainerobserver session72490 completedsuccess; RL/validation/1560_7500_checkpoint_wait_20260922T000524Z.json recordsfullcommit00:10:38 thenonline/EMAqueue00:11:20, actualTaskRunningthroughout. No prematureincompleteCPevaluation, no restart.
- Reusedexisting JL/audit_checkpoint_ema.py --step7500 --watch, CUDAhidden/CPU2threads; foregroundsession88491 completedsuccess. RL/validation/production_step7500_ema_audit.json complete=true,4runs. ThiswasONE-SHOTforegroundaudit, notanewpersistentservice. Existingfinal10000watcher636918unchanged/stillwaiting1.56B10k.
- Same1.56B7500 audit: initialized,decay.99,updates7500,660namedtensors/1560089984trainableparams,FP32shadownames+shapesexact,largesttensor8shards,CAblock0beforeSA,FFNhidden4848. RawmanifestSHA504b172704a18a8af9cf2b3fe0666a4bbc080521ac3ffb9489e0f8c35eb6d704; configSHA5a551454fb95502ee1b100f6aa52eda8a68fb27a759a94e0cde2044a07ebfafb. Scopecommittedmanifest/meta-model/loadedsmallEMAbuffers, notall-weightnumeric scan orGPUrestore.
- R0/validation/light_fvd_20260922T000543Z.json: same1560EMA4500=99.97860819729567 vs globalfirstCA84.3630125456234,identityf4707a9aef70bca17e79d68b2dfc50538e08b677282867968863287d8632d58a.
- R0/validation/light_fvd_20260922T001245Z.json new7points: same3555000 online236.68066121051788/global217.2605559851022 andEMA138.61815900785587/global116.53421004220854;7755000 online196.34726864077948/global260.1552361791778 andEMA97.2623209129979/global97.49814080601391;1255500 online251.0577355442897/global264.53068984130687 andEMA170.3659403430711/global174.29043729072373;1560online4500=170.74083967968681/global235.1321767192403. Lightcoverageall2048fileIDs/fullCP/protocol/outputmanifest/fourpairedcases; noindependentFVDrecompute/RGBregeneration.
- Samecompletedscorecount80; originalconcat5500evalfeaturecount768 (partialonly). Dev1localcontroller606450 running775EMA5500 sampler ...0922001026,launcher662960 lastchecked00:11; do notstartseconddev1sampler. Remoteextras650662 cap6; original348486/348487; samemonitor609245; CPUfinalEMA636918.
- Same125/355/775trainerscomplete/deleted, NEVERresume.1.56Bstilltraining,nocleanupyet. Original20B5000CA mayapproachnext; inspectactualstep+fullcommitbeforeenqueue. AllNative/runtime/config/protocol/samplingcontrollers unchanged. Previewserver452224/SSH96817preserved. Allforegroundwatches/audits inthisturncompleted,no pendingwait.

## 2026-09-22T00:31:48.305579+00:00 — 20B CA5000 full checkpoint, independent evaluation queued; requested1.56B progress

- User-requested same-spatial 1.56B already running on8H2002D, task stormyxie-sc-1560m-xlocal-0921155005; actualTask/PodRunning at00:31UTC, directcompleted7833, EMAupdates7833, local8/global64. ExistingCAblock0 beforeSA→FFN, FFNcompensation, mask.8, FA4 andEMA.99 unchanged. No duplicate launch/restart.
- Verifiedsame1560FVD5000 online172.2865027924372 versusmatchedglobalfirstCA185.34447402395517; EMA102.5864664140936 versus88.20855347360293. Online/EMAsamecheckpointidentity46d39717be55a69bfb08e91b9884104394cc404e74d9ed2063b3f1d19e27e961. R0/validation/light_fvd_20260922T003038Z.json checks2048fileIDs/manifest/protocol/fourpairedcases; notindependentfeature-recompute orRGBregeneration. Samecompleted89scores; original179/200. Preview /same-spatial/ localHTTP200.
- Addedexternalread-only J0/audit_intermediate_checkpoint.py SHA05587bc5da914dc81aa8a03f7130f8b1473ba3dd29802f23822d7d5c542e9606. Preparationproof R0/validation/intermediate_auditor_prepare_20260922T001910Z.json. ValidatesNativefullcommit/metadata/chunkbounds/8CPUtrainerstates/cursor+schedule/retained500stepcheckpoints/recipeidentity againstfirst500. CUDAhidden; nofullpayloadrehash orGPUrestore. NOTterminaltrainingcompletiongate; neverusealoneforTaskcleanuporobjectivecompletion.
- Knownconcat5500 preflight passedR0/validation/intermediate_checkpoint_20b_entry_concat_05500_20260922T001955Z.json. NewCA5000 passedR0/validation/intermediate_checkpoint_20b_cross_attention_05000_20260922T002510Z.json: weights1916entries/15287chunks,opt4445/26636,8trainerstates,10fullCPs500..5000; rawmanifestSHA4708d7de000f0aece6d537f2325ad64141b672ab7bdf808a8002cb3a8ecdbbcd. Weights+optimizer149.1GiB. Asyncsave~8minwithtrainingcontinuing5020at00:24, notfailure.
- Boundedobserver26763 finishedsuccess; full5000commit00:23:52thenqueued00:27:42. ProofR0/validation/20b_ca5000_commit_wait_20260922T001735Z.json. Execwait4373 consumed; no foregroundsessionsstillwaiting.
- Automaticoriginalextras submittedCA5000 task stormyxie-sc-fvdx-0922003028 H2002D00:30:28. Exact00:31TaskUpdating/PodPending Unschedulable(insufficientwhole-nodeGPU/CPU/memory/EFA); submittedbutNOTRunningatthisobservation. Existingcontroller348487 handlesprovenunschedulableadmissions120s/cooldown/fallback. Thisisnottrainfailure; do notrestart20Btrainer/duplicateeval. Correct-clusterTask/Poddetailsin /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca5000_enqueue_1560_progress_20260922T003118Z/observations.json.
- Originalconcat5500eval ...0921232715 remainsRunning AWS2B, now1024/2048featurepairs at00:30controllerobservation. CA5000evaladditionalindependenttask; original20Btrainersstillrunning, goalACTIVE/incomplete, no claimed200/200completion. Same125/355/775trainerscomplete/deleted; NEVERresume.
- No Native train/eval/config/protocol modifications; onlystandaloneCPUauditoradded. Originalmonitor348486/extras348487; samemonitor609245/extras650662 cap6/local606450/finalEMAwatcher636918 unchanged. Previewserver452224/SSH96817 retained. Keepterminalcompletionaudits separatefromintermediateaudit.

## 2026-09-22T00:35:14.714159+00:00 — original20BCA5000 evaluator admitted on8H200

- Previousgoalturnmadeprogress(fullCP+CPUmetadataaudit+enqueue); thisturnverifiedlive20Btraining andwatchedactualCA5000admission. Originalobjective stillACTIVE/incomplete; original179/200scores, same91/160. No newtrain/eval/runtime/config/protocol/controller edits.
- At00:32actualoriginaltrainers ...0920143127 and...0920143629 Running; directcompleted5729/5040, freshlogs andloss0.37429/0.40836, e2e21.31/24.11seconds. Existingconcat5500evaluator ...0921232715 actualRunning.
- Boundedobserver5537 finishedsuccess, proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca5000_admission_watch_20260922T003237Z.json. InitialCA5000H2002Dtask...0922003028 provenUnschedulable thenautomaticallydeleted; oldcorrectclusterPods0confirmed. Thiswasadmissioncapacity, notmodel/trainingfailure.
- Controllerautomaticallysubmittedstormyxie-sc-fvdx-0922003400 toAWS2B at00:34:00. ActualTaskRunningandPodstormyxie-sc-fvdx-0922003400-worker-0 Running00:34:43. ExactlyoneundeletedCA5000receipt;8H200protocolunchanged. Admissionproof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca5000_admitted_20260922T003512Z/validation.json. NewCA5000nowevaluatingindependentlyalongsideconcat5500; notcompletedscoreyet.
- All foregroundobserver/tool sessionsfinished. Persistentcontrollersandpreview/tunnelunchanged. Same125/355/775completedtrainersmustnotresume;1.56stilltraining. Continueobservingnewcommittedcheckpoints/completedFVD, donotmarkgoalcompleteuntilfulloriginalscopeverified.

## 2026-09-22T00:42:46.628361+00:00 — CA5000 live GPU startup, concat5500 fifth batch, same1560 new5500 FVD

- Previousgoalturnmadeprogress byverifyingCA5000actualTask+PodRunning. Thisturnverifiedlivewaitonexact20Bhandles andnewFVD5500pair; goalACTIVE/incomplete179/200. NoNative/runtime/config/protocol/controller changes orGPUjobrestarts.
- Originalconcat5500Taskstormyxie-sc-fvdx-0921232715 actualRunningand1280featurepairs00:36:25; boundedobserver43015 completed. R0/validation/concat5500_fifthbatch_20260922T003625Z.json. Partialfeaturesonly, nofinalFVDscoreyet.
- New20BCA5000Taskstormyxie-sc-fvdx-0922003400 actualRunning00:35thencontinuousGPUprobe, actualRunningrevalidated00:42:14 afterobserverexpiration.8H200 started12–15GiB/0percent then25–42GiB/mostly100percent; lastprobeall8GPU100percent,0features. NoFAILmarker. ThisprovesGPUactivity, notfirstsamplesorcompletedFVD.
- BoundedCPUobserver11069 finishednormallyafter300s, proofR0/validation/ca5000_gpu_startup_20260922T003652Z.json. Its complete=false meansARBITRARYprobecondition(all8GPUmemory>80000MiBandutil>=80)wasnotmet beforedeadline; NOTevaluationfailure/terminalstate. DoNOTrestartormarkfailedbecauseofthisobserverflag. ExactTaskRunningrecheckedat /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca5000_startup_and_same5500_20260922T004215Z/observations.json.
- NewlightFVDaudit R0/validation/light_fvd_20260922T004022Z.json: same1560online5500=180.27289938350577/globalmatchedfirstCA158.0412995581678;EMA98.501362006401/global81.75209245862925. Bothsamealiases checkpointidentity1c6439daeacd56b4103e9bd4bb3103887723c51e0ad6188e209894ae757daff2. Coverage2048fileIDs/manifest/protocol/fourpairedcases; notindependentsavedfeatureFVDrecompute. Onecheckpointcomparisononly, no universalorderingclaim.
- Same1560directlogcompleted7999/EMAupdates7999at00:42;08000.dcp.incompleteexists, fullmanifestabsent andnotyetqueued. Waitfornativefullcommit beforeevaluation. Samecompleted95scores; threeothertrainersalready10kcomplete/deleted, neverresume.
- Allforegroundsessions43015/11069/48610/8497 consumedandcompleted; nopendingexecwait. Existingcontrollers/tunnelremain; nofalsecompletion/blockingstatus. NextmeaningfuloriginaleventsCA5000firstbatch/concat5500completion; auxiliary15608000fullcommit/queue.

## 2026-09-22T00:54:52.990977+00:00 — 20B CA5000 first batch verified; same-spatial 1.56B 8000 committed and queued

- Previous goal turn was progress plus verified waiting. This turn verified the new 1.56B 8000 full checkpoint and both evaluation aliases, observed the original20B CA5000 first256 feature outputs, and checked their provenance/format. The original goal remains ACTIVE and incomplete: 179/200 original FVD scores; same-spatial has 102/160.
- Same-spatial 1.56B 8000 is full, no incomplete directory, all weights/optimizer/trainer components present, world_size8. Model fingerprint and execution policy match its first500 checkpoint. Both online and EMA aliases point to the exact08000.dcp. Raw manifest SHA2b0472f3c88fd086291bcb84efcd37b82b6dd72458854ac76f63a19f2f9df03c. Proof R0/validation/same1560_8000_committed_20260922T004729Z/validation.json. This is manifest/queue verification, not numerical EMA shadow restore.
- Bounded observer2958 expired normally before CA5000 first batch; its complete=false is an observation deadline only. Exact evaluator stormyxie-sc-fvdx-0922003400 was revalidated Running00:51:59, then bounded followup47546 observed256 features and actualRunning00:53:19. Both foreground observers are consumed/completed. Proof R0/validation/ca5000_firstbatch_followup_20260922T005319Z.json. Do not restart an evaluator because of a bounded observer timeout.
- New originalCA5000 first256 outputs verified at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/ca5000_first256_validation_20260922T005414Z/validation.json: all256 sidecars CFG1,50steps,batch32,10states,metricframes16:160, expected frozen protocol; GT pixel hashes, detector metadata and per-case seeds match priorCA4500. Embedded checkpoint manifests exactly match current5000 Native full manifest; all identities a70cc9b61fc6eb116a56474162621c27c9ab30e5c3069ca70febf917468925e7. Cases0000..0007 arrays checked finite FP32/400D. Scope first256 metadata/eight arrays only; no finalFVD or independent RGB regeneration.
- Originalconcat5500 evaluator ...0921232715 advanced to1536/2048 feature pairs during the same observer. OriginalCA5000 remains on8H200 AWS2B. PreviousCA4500 first feature was17.806min after its worker RUNNING marker; this was reference timing, not a deadline/failure gate.
- Fresh original trainer observations: [{"task_name": "stormyxie-sc-20b-cat-0920143127", "name": "20b_entry_concat", "scheduler": {"creationTimestamp": "2026-09-20T14:31:29Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-20b-cat-0920143127 is running.", "status": "Running"}, "completed_steps": 5792, "loss": 0.41465218737721443, "log_age_seconds": 12.947166204452515}, {"task_name": "stormyxie-sc-20b-xa-0920143629", "name": "20b_cross_attention", "scheduler": {"creationTimestamp": "2026-09-20T14:36:30Z", "description": "", "reason": "MPIJob gwm/stormyxie-sc-20b-xa-0920143629 is running.", "status": "Running"}, "completed_steps": 5096, "loss": 0.4434768185019493, "log_age_seconds": 11.835835933685303}]. No training restarts or source/runtime/config/protocol changes. All persistent controllers and preview452224/SSH96817 retained. Same125/355/775 trainers already complete/deleted; never resume. All foreground sessions from this turn finished.

## 2026-09-22T01:04:24.925732+00:00 — original concat5500 reaches1792/2048; verified wait on the final batch

- Previous goal turn made progress (CA5000 first256 provenance/feature checks, same-spatial1560 8000 commit/queue). This turn was a verified wait on actual live original concat5500 Task stormyxie-sc-fvdx-0921232715, which advanced1536→1792 features with actualTaskRunning. Bounded observer43526 completed and was consumed; proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/concat5500_seventhbatch_20260922T005600Z.json. No pending foreground session.
- No final concat5500 FVD score yet. Next meaningful event is final2048 features, score/provenance validation, then controller cleanup. Do not equate partial features with completed FVD. Original score count remains179/200; same-spatial count106/160. Goal ACTIVE/incomplete; no source/config/protocol/controller changes.
- At00:55:32 both original evaluators ...0921232715 and...0922003400 were actualTaskRunning; original extras status fresh,2active, failed_targets empty. Controller observation at this handoff is stored in /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/concat5500_final_batch_wait_20260922T010424Z/observations.json. CA5000 had already verified first256 outputs in previous turn.
- Informational ETA from recent100 steady training rows at~00:56UTC: originalconcat completed5797, median21.3286058sec/step, ~24.90traininghours remaining to10000; originalCA completed5100, median24.1685977sec/step, ~32.90hours. These are compute-walltime extrapolations excluding interruptions and final evaluation tail, not completion guarantees.
- All persistent monitors, evaluators, training jobs, finalEMA watcher, previewserver452224 andSSH96817 retained. Original two20B trainers remain required for full goal; same125/355/775 completed trainers must not resume.

## 2026-09-22T01:20:21.222784+00:00 — original180/200: concat5500 FVD260.66 verified, evaluator cleaned

- Previous goal turn was a verified wait that observed1792 features. This turn made concrete progress: originalconcat5500 reached2048 samples, published its FVD, passed an independent full saved-feature recomputation, appeared in the report, and its8-H200 evaluator was cleaned. Original goal remains ACTIVE/incomplete180/200; both20B trainers still need10000steps/all20checkpoint evaluations.
- Result: 20b_entry_concat/05500 FVD260.6555900799708 versus05000=323.06050136449585 (~19.316% lower),04500=268.6468060552029. Fixed2048 heldout/CFG1/144generatedframes,50Euler,batch32. Do not infer monotonic convergence from one new point.
- Full audit reused J0/audit_new20b_20260921T1522Z.py with ONLY in-memory target/output-dir replacement; original source unchanged. All2048 generated features/sidecars and pinned GT features/metadata checked, exact case set/seed/config/checkpoint/detector/GT pixel identities verified. Independent float64 scipy.sqrtm recompute260.65559008219043, absolute error2.2196218196768314e-09, imaginary_max0. No RGB regeneration/new detector extraction. Proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/concat5500_completion_20260922T010533Z/20b_entry_concat_5500.json andsource_adaptation.json. GT aggregate4ad523cb85686fee12060661058b4e791f5a4f506d23afd1bf21e826f1e2a8ae; generated aggregate1dfe3c9ae554ea1d633d0b2ad06d22ff33eb81a0106f7335a789d8aed877e2de.
- Exact evaluator stormyxie-sc-fvdx-0921232715 AWS2B receipt observedSucceeded(completion01:17:04UTC), PASS present, automatic delete succeeded, actual correct-cluster Pods0 andTaskNotFound. Checkpoints/results retained. /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/concat5500_completion_20260922T010533Z/cleanup_and_report.json. A missing cleaned evaluator is NOT a training/evaluation failure; never restart it.
- Originalreport progress.json nowcontains260.6555900799708 andHTTP200; rootHTTP200. Score count180. Refreshed existing production timing report to163 clean observations;20Bconcat n11median6521.9406304s(108.699min). Preimages/source hash/all180scoreSHA preservation proof R0/validation/timings_after180_20260922T011836Z/validation.json. Existing timestamp-anomaly exclusions unchanged; timing table is descriptive, not an independent benchmark.
- Actual01:10TaskRunning: originalcat...0920143127, originalCA...0920143629, same1560...0921155005, CA5000eval...0922003400. OriginalCA5000 sampled512/2048 then. Samecompleted112FVDscores at01:19, withfresh1560observation {"completed_steps": 8583, "ema_updates": 8583.0, "log_age_seconds": 4.311168670654297, "checkpoint8500_full": true, "checkpoint8500_incomplete": false, "checkpoint8500_queued_weights": ["ema", "online"]}.
- Foreground57754 completed successfully and was consumed; all other foreground tools consumed. Originalmonitor348486/extras348487, samemonitor609245/extras650662(cap6)/local606450/finalEMA636918 andpreview452224/SSH96817 retained. No Native training/evaluation/config/protocol/controller source edits or restarts this turn. Same125/355/775 completed trainers must not resume.

## 2026-09-22T01:27:20.122449+00:00 — original CA5000 third batch; new same1560 FVD comparisons

- Previous goal turn made concrete progress: originalconcat5500 final score, full saved-feature independent audit, automatic8-H200 cleanup, report updated to180/200. This turn verified five new same1560 comparison points and waited on exact live CA5000 Task; full original objective remains ACTIVE/incomplete180/200.
- Bounded observer53803 completed successfully: stormyxie-sc-fvdx-0922003400 AWS2B actualRunning,512→768/2048 features, no final score yet. Proof R0/validation/ca5000_thirdbatch_20260922T012236Z.json. All foreground sessions from this turn consumed; no pendingexecwait. Do not restart an actively progressing evaluator.
- New light audit R0/validation/light_fvd_20260922T012149Z.json: same1560 online6000=173.3576225054503/globalmatchedfirstCA365.64578712108914; EMA6000=91.2310455518741/global74.72903962635792, pairedcheckpointidentitycc1da083c9028517ecc1f72f7e6eb040d8c46b6d3d4e6b54c02bb830c591d4d0. Sameonline6500=161.22445769007436/global125.55541397322077;EMA6500=93.5459314616487/global70.44281779584571,identityc813a7bdcfde079d71c7f01df349ce15f5d9df29947b242eef74c76ca8c6b014. SameEMA7000=82.822438487196/global70.18571327399104,identity2adc082f7dec9f672ae54941db1c920318f68f79dd09668ac8645ef96312d60c.
- Audit scope all2048 fileIDs, committed checkpoint/output-manifest/protocol, four paired per-case metadata samples; NOT full independent FVD recomputation for these auxiliary points. Reference is matched-parameter FIRST-layer globalCA, not original all-layerCA. Single-seed checkpoint comparisons do not establish universal ordering.
- Same score count118/160; original180/200. Final10000 CPUEMA watcher check: {"pid": 636918, "alive": true, "expected_command": true, "state": "S"}. Existing original/same controllers and preview/tunnel retained; no source/runtime/config/protocol/controller changes. Same125/355/775 trainers already completed/deleted; never resume. Same1560 still training;8500 full checkpoint and both aliases were verified in previous turn.

## 2026-09-22T01:31:04.934317+00:00 — complete checkpoint-to-evaluation coverage inventory

- Previous turn made progress on five new auxiliary comparisons and verified a live CA5000 batch advance. This turn audited all current full checkpoints against score files and pending queues across original, matched-firstCA, and same-spatial campaigns. No full checkpoint target was missing from completed scores or the queue. This resolves queue coverage; it does NOT establish that future unsaved training steps are complete.
- Authoritative inventory proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/checkpoint_queue_coverage_20260922T013007Z/validation.json:
  * Original:181 full checkpoints,180/200 final target scores complete,1 unscored full target queued/running (20B CA5000),19 target checkpoints not yet saved. The remaining original work is primarily20B training. Full objective remains ACTIVE/incomplete.
  * Matched global firstCA:80 full checkpoints,160/160 online+EMA scores,0 pending. Its old queue timestamp is expected because its campaign/services completed; do not restart.
  * Same-spatial:77 full checkpoints→154 available online+EMA targets,119/160 scored,35 unscored full targets all queued,6 targets not yet saved (1.56B9k/9.5k/10k ×online/EMA).
- Inventory checked each manifest full state/completed_steps/expected model-and-step target mapping, all weight/optimizer/trainer presence flags, raw manifest SHA, exact score→checkpoint path, expected target name/step,2048 cases/non-smoke/finite score, and queue coverage. It is not a full tensor payload rehash/GPU replay or final goal completion audit.
- Actual01:28:41UTC TaskRunning verified for original20Bcat...0920143127/allCA...0920143629, same1560...0921155005, CA5000eval...0922003400. Originalextras1active, sameextras6active, failed_targets empty, controllers fresh. No task restart, new GPU launch, controller/source/runtime/config/protocol changes.
- All foreground sessions12121/86670 and final handoff write completed/consumed. Existing monitors/finalEMA watcher/preview tunnel remain. Same125/355/775 completed/deleted trainers must never resume. Continue the two20B trainers and their500-step evaluation pipeline through10000; no remaining currently-complete checkpoint was left untracked.

## 2026-09-22T01:50:37.685756+00:00 — same1560 9000 full checkpoint and both aliases queued; original CA5000 half sampled

- Previous turn established full checkpoint/score/queue coverage. This turn made progress by observing new same1560 9000 full commit and automatic online+EMA enqueue, while verifying a live originalCA5000 evaluator batch advance. Original goal remains ACTIVE/incomplete180/200; same-spatial has130/160 scores.
- Bounded observer73642 completed and was consumed; proof /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/same9000_and_ca5000_half_20260922T013332Z.json. It repeatedly queried actual Task handles: stormyxie-sc-fvdx-0922003400 AWS2B and same1560 trainerstormyxie-sc-1560m-xlocal-0921155005 H2002D. Both actualRunning at final01:49:42UTC. CA5000 advanced768→1024/2048 features (no final score).
- Same1560 log reached9007/EMA9007 with09000.dcp.incomplete, then full09000.dcp committed01:48:56, online+EMA aliases queued01:49:42. Lastobservedcompleted9025/EMA9025. Manifest full/completed_steps9000/world_size8/all3components checked; model fingerprint and execution policy equal first500. Raw manifest SHA27a8e659fd7b388713d6cd3001979d2ed823ec3a8db074cbb376b6cb5672f531. Scope manifest/recipe/queue, not numericEMA shadow restore or full tensor payload hashing.
- Recent30 numerical training-health check at01:41:34 passed for original20Bcat(completed5923,meanloss0.3959216112),original20BCA(5212,mean0.4093021192),same1560(8907,mean0.4219578414). All recorded loss/postreduce/postclip gradient norms finite, sameEMA count matched every examined step. Proof R0/validation/training_numeric_health_20260922T014134Z.json. This is not a scan of every model tensor.
- No Native/source/config/protocol/controller changes or job restarts. Originalmonitor348486/extras348487; samemonitor609245/extras650662(cap6)/local606450/final10000EMA watcher636918; preview452224/SSH96817 retained. Same125/355/775 trainers already complete/deleted, never resume. Same1560 still has9500/10000 ahead; do not clean its running trainer.
- All foreground sessions in this turn consumed. Next original events are continuingCA5000 sampling and future20Bconcat6000 full checkpoint; next auxiliary training points9500/10000, followed by required terminal-training/full-checkpoint/finalEMA audits before cleanup.

## 2026-09-22T02:30:24.707889+00:00 — preview tunnel recovered with reconnect supervisor; original concat6000 audited and evaluating

- Previous turn made progress on same1560 9000 full checkpoint/queue and originalCA5000 sampling. This turn recovered a real local preview forwarding failure, revalidated production tasks/controllers, and audited/admission-verified a newly saved originalconcat6000 checkpoint. Original goal remains ACTIVE/incomplete180/200; same-spatial scores153/160.
- SSH jump server closed the long-lived observation channel43783 and old preview tunnel session96817; both returnedexit255. Initial new short ssh dev1 hostname succeeded, so alias was NOT stale; no SSH config refresh or koala-refresh skill applied. Local8884 hadHTTP000/refused while remote8884 remainedHTTP200; remote preview452224 and all checked monitor/extras PIDs alive. Exact original20Bcat/allCA trainers and CA5000 evaluator were actualRunning after reconnection. Do NOT restart production jobs due to this SSH observation failure.
- IMPORTANT PREVIEW CHANGE: oldSSHsession96817 is dead/consumed. New persistent LOCAL Python SSH reconnect supervisor is tool session69295, supervisorPID90448, current childSSH90454 (child may change on reconnect). It runs ssh -N with BatchMode=yes,ConnectTimeout15,ExitOnForwardFailure=yes,ServerAliveInterval20,ServerAliveCountMax3, -L127.0.0.1:8884:127.0.0.1:8884 dev1; waits3seconds then restarts SSH after exit. LocalHTTP200 andlocalssh listener verified. KEEP69295 alive; do not create a second competing tunnel. Remote serverPID452224 unchanged. Supervisor was added for this actual disconnect; automatic retry code exists but a forced-disconnect test was not performed. Metadata /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/preview_reconnect_and_concat6000_20260922T023024Z/tunnel.json.
- Interrupted observer43783 left R0/validation/ca5000_fifthbatch_20260922T015302Z.json complete=false at lastupdate~01:59; this means its observation channel ended, NOT the evaluator failed. Requery showed actualCA5000Running and1536/2048 features at02:27. No foreground observer remains from43783.
- New original20Bconcat06000.dcp is full; existing CPU J0/audit_intermediate_checkpoint.py passed at /efs/stormyxie/reports/scaling_fa4_20260920T134440Z/validation/intermediate_checkpoint_20b_entry_concat_06000_20260922T022820Z.json:1540weightentries/12279chunks,3567optimizerentries/21369chunks,8trainer ranks,12retained full CPs500..6000, rawmanifestSHAb2515f1cb5eaf296ba7755815473d14e6b756339448861382c381f5f0c3fa632. Recipe matches500,FA4/no fallback/local8 unchanged. Scope metadata/chunkbounds/CPU trainer states, not full tensor payload rehash/GPU replay or terminal-training-completion gate.
- Automatic concat6000 evaluator stormyxie-sc-fvdx-0922021752 AWS2B submitted02:17:52; actualTask/PodRunning02:28,8GPU limit,nodehyperpod-i-01b2ca7a727a52940. It is an independent evaluation node; original training nodes remain separately allocated. Frozen code/config/protocol unchanged; no duplicate submission.
- Fresh02:27logs: originalcat6051/allCA5325, same15609605. Same09500.dcp full and both9500aliases alreadyevaluating; its final10000 is not yet saved. Sameextras active tasks include355online10000,775EMA/online10000,1560EMA/online9500; only terminate after real success. H2002D Updating admissions may be capacity waits, use exact handle/cluster.
- All foreground audit/read sessions in this turn completed. Only new persistentLOCAL supervisor69295 remains. Originalmonitor348486/extras348487; same609245/650662(cap6)/606450/finalEMA636918 unchanged. Same125/355/775 training tasks already completed/deleted, never resume. Next crucial events: CA5000 final score, same1560final10000+EMA/full-training audit and cleanup; originalconcat6000 sampling continues.


## 同 patch CA 可视化报告（2026-09-22 02:49 UTC）

- 用户要求补同 patch CA 可视化；仅更新报告 HTML，训练/eval runtime、配置、协议、监控与队列均未改。
- 当前入口 http://127.0.0.1:8884/same-spatial/；建议双 log 平滑入口：
  http://127.0.0.1:8884/same-spatial/?weight=ema&scale=loglog&fit=lowess
- 四档 small multiples 对比参数补齐的 global FIRST CA 与 same_spatial FIRST CA；
  支持 online/EMA、线性/双 log、实测折线/LOWESS/OLS、全区间/2500+切换。
  保留完整逐 ckpt 数值表，增加同 step 相对差距、最新共同 step 表、SVG/CSV导出；
  实测点透明度0.26，LOWESS在当前坐标空间拟合，span40%，不外推。60秒刷新。
- source: JL/report_same_spatial.html；served: RL/index.html。JL/RL均指same_spatial_ca_20260921T145503Z。
  index最终SHA256: af7a87f8d6cab8257bc13319a4156803e85f63a9973a3ae8a9a0aff39aa36af2。
- 原HTML完整备份与部署/渲染证据：RL/validation/visual_report_20260922T024456Z/。
  本机隔离Chrome headless实际渲染3种视图，每种5SVG/12curves，无NaN/Infinity坐标；
  已目视线性和loglog LOWESS截图。browser-act/uv不可用，未安装，改用已有Chrome。
- 02:46 UTC的实际progress为158/160；三小档10k已完成，1.56B在9906/9500完整ckpt，
  最终10000 PASS尚未生成，不得据此清理。EMA final CPU watcher636918仍等待1.56B。
- 原20B CA5000 foreground observer shell25806仍在运行；最新已直接Task Running，
  会在2048特征/正式score出现后自动做独立float64 FVD复算，别启动重复评测。
  8884预览继续使用自动重连shell69295；旧96817已退出，不要恢复第二条竞争隧道。


## 三方 condition 报告与最终 same-spatial 训练回收（2026-09-22 02:57 UTC）

- 用户要求在同 patch CA 报告加入入口 concat。已上线：
  http://127.0.0.1:8884/same-spatial/?weight=online&scale=loglog&fit=lowess
  三方分别入口concat、参数补齐全空间FIRST CA、same_spatial FIRST CA。
  默认online；入口concat没有EMA，不在EMA视图混入online曲线。
- 四档entry concat共80个正式checkpoint点来自原始R/progress.json；三campaign基础
  fvd/protocol.json SHA相同96aae9b370d92009adbbf8480f05cb3d41bb98d6ea5ece6ba81fe24574a4345c。
  参数是近似匹配，图上显示CA/concat各自参数；更新曲线、共同step表、完整表、CSV。
  在线图每档3曲线；EMA每档2曲线。保留线性/loglog/LOWESS/OLS及原始表格。
- RL/index.html与JL/report_same_spatial.html SHA ca350b0954596082932ecff89a6c9b7ab8361629b1dcadab89082f856fd521bf。
  备份和验证 RL/validation/visual_three_way_20260922T025411Z/；
  Chrome真实渲染online有5SVG/16curves/80entry散点，EMA有5SVG/12curves/0entry散点，
  无非法坐标，并目视online截图。未改训练、评测、协议或控制器。
- Same-spatial 1.56B Task stormyxie-sc-1560m-xlocal-0921155005 (tenc-szr3212j/gwm)
  02:53:53平台Succeeded。实际PASS、10000 full DCP、20个500间隔checkpoint、
  精确10000条有限loss/grad日志、8rank CPU trainer状态及40个FVD目标覆盖审计通过。
  FP32 EMA最终updates=10000，decay=.99；完整最终EMA审计现在4/4 complete。
  证据 RL/validation/training_complete_1560m_cross_attention.json、
  production_step10000_ema_audit.json、final1560_completion_20260922T025116Z/。
  02:56:38精确任务已删除，复查NotFound/0Pods；20份manifest SHA完全保留。
  同patch四训练均完成并清理，不得恢复；FVD仍158/160，最后1.56B在线/EMA@10000待完成。
- 原始20B all-layer CA @5000正式FVD=125.0583841819548。
  全2048GT/fake特征+metadata和独立float64 sqrtm复算通过，差3.0740920919924974e-10。
  证据R/validation/ca5000_completion_20260922T023315Z/。
  对应Task stormyxie-sc-fvdx-0922003400已Succeeded/PASS并由controller删除；
  本次直接复查NotFound/0Pods。原始正式分数181/200。02:55 root progress尚未纳入新点，
  属300秒监控发布延迟，不是评分失败；后续核验自动发布，不要重复采样。
- 原 foreground shell25806（CA5000 wait+audit）与97776（同patch1.56最终wait+audit）均正常完成，
  不再poll或重启它们。唯一需要继续保留的本机预览连接为自动重连shell69295。
  同patch控制器保留到最后两分数完成；原始两个20B训练、entry6000评测继续，goal仍active。


## 三种 condition 实现核对与当前等候（2026-09-22 03:06 UTC）

- 用户询问报告里entry concat / same-patch CA / all-patch CA的实际实现；已读三组
  immutable runtime的world/attention/conditioner/model/config及实际四档配置，
  源码SHA证据 RL/validation/implementation_audit_20260922T030554Z/source_audit.json。
- all-patch在此报告是GLOBAL FIRST CA，只在block0，CA→SA→FFN；不是原始每层CA。
  entry入口把raw noisy latent64维与LN+Linear投影到D的ViT condition做channel concat，
  Linear(64+D,D)→SiLU→Linear(D,D)，替代普通latent input projection；全block无CA。
  两个CA版本先latent64→D，再block0 RMSNorm→Q/K RMSNorm attention→proj→residual，
  CA分支没有tanh gate或timestep AdaLN；timestep调制SA/FFN。CA本身不做3D RoPE，
  3D RoPE在conditioner SA与DiT SA。DiT所有block的SA仍在单个目标state上全空间混合。
- 同patch按(h,w)分组，允许4个latent planes，不是单一(t,h,w)key。通过批处理独立组
  调FA4，异地key根本不进softmax；空组投影后强制零，屏蔽proj.bias。
  ViT 80% mask每plane保留floor(256*.2)=51，共204；推理1024全部可见。
  同patch每位置零key概率(205/256)^4=41.1202%，单key40.9196%，>=2keys17.9601%；
  这些是mask分布理论期望，不是实测数据频率，也未叠加10%整组CFG dropout。
  entry按原(t,h,w)scatter，缺失condition为0。ViT自身state-causal全空间SA仍传全局信息。
- 全局/同patchCA参数完全相同；entry近似匹配。FFN hidden entry/CA：
  125 1816/1776；355 3064/3008；775 4568/4512；1560 4896/4848。
  原entry没有EMA；三方online比较。仅CA两者是只改查询范围的架构配置比较。
- 原始CA5000=125.0583841819548已在root progress正式发布。timing刷新164条，
  全181份score SHA未变；证据 R/validation/timings_after181_20260922T025904Z/。
- 当前foreground shell67843、dev1 observer PID722522 正在等同patch最后online/EMA@10000，
  proof RL/validation/final_fvd_watch_20260922T030022Z/，窗口至约03:35 UTC。
  exact online Task stormyxie-sc-xlocal-fvd-0922025744 / tenc-awsusw2 Running；
  exact本机EMA wrapperPID720876 / stormyxie-sc-xlocal-local-0922025428 alive。
  03:03附近online406/2048、EMA512/2048。不要重启/重复提交；窗口到期不是失败。
  此observer等8个最终score齐全后，用JM/verify_ema_scores.py原文不变、
  JL/fvd_coordination上下文执行--step10000 --recompute，结果写到RL。
  这是全2048feature/metadata核验+用既有metric实现重算，不是独立算法复算。
  先poll shell67843；completed原shell25806/97776不再poll。预览仍shell69295。


## DuET mask 后的稠密补全：必须与本轮实现区分（2026-09-22 03:14 UTC）

用户指出DuET会在mask后通过cross-attn展开全空间，已直接核对，确实如此。
参考DuET configs/duet_videorae3d_300m.yaml: use_prefix_expander=true,
prefix_fill_masked_only=true,prefix_fill_mode=cross_attention,context_mask_ratio=.9。
每state1024 tokens，训练ViT只看100；保留100可见prefix，PrefixExpander生成924缺失prefix。
Q=shared learnable query+缺失位置3D sincos，K=normalized visible prefix+其位置编码，
V=normalized visible prefix的线性投影；每个missing query可读当前state全部visible prefix。
CA后有position-query residual与MLP residual。此CA在condition补全模块，
不是DiT内的noisy-x查询。完整prefix随后entryconcat，参考配置另有逐层deep add。
推理不mask，selected=None且fill_masked_only使expander直接bypass。
本轮RepWorld只复用了每plane独立随机丢token的采样规则，没有复刻PrefixExpander：
entry缺位0；global/same CA读取稀疏ViT features。此前“DuET mask策略”不能
外推成完整DuET mask→dense completion→injection等价复现，也不能把零key约41%的
现象归给DuET dense-prefix方案。用户此轮只询问实现，未启动新实验或修改运行模型。
静态代码/配置SHA证据：RL/validation/duet_mask_expander_audit_20260922T031554Z/evidence.json。


## 1万步 FVD 视频预览（2026-09-22）

核验时间：2026-09-22T03:17:39.214612+00:00。已完成3个10秒对照视频，页面 http://127.0.0.1:8892/ 。
每行固定fvd_0000/1024/2047之一，三列GT / 原775M all-layer SA→CA→FFN / 原1.56B同结构，均为online@10000。
对应正式FVD2048=146.495916/100.656934。原case/seed/checkpoint与冻结runtime_batch_candidate一致，
CFG1、每state50Euler、16观察+144生成帧、256方图、16FPS。预览batch3，正式batch128/64；
6份重放RGB hash均与原评测不同，不能声称原始RGB逐像素复现。原正式评测未保存视频。
用户本次明确允许显存够用时重叠跑；预览实际与现有正式FVD在GPU1/2重叠，
775M/1.56B的PyTorch allocator限制10/14GiB，两路exit0，无OOM，自己的GPU进程均已退出。
未停止其他任务或修改任何冻结snapshot。3份MP4全部160帧解码、SHA与HTTP Range播放通过。
报告与媒体：/efs/stormyxie/reports/fvd_preview_10k_20260922T030419Z；代码：/threed-code/stormyxie/vforge_jobs/fvd_preview_10k_20260922T030419Z。
证据preflight.json、media_validation.json、completion_verified.json、overlap_observations.jsonl。
媒体保留EFS；本轮HTTP8892继续运行，原8891隧道已恢复，两端口使用本地自动重连SSH；8884保持原状。


## 同 patch CA 全部完成；仅原始 20B 继续（2026-09-22 03:27 UTC）

- Same-spatial 四档125M/355M/775M/1.56B全部训练到10000，20份完整DCP/档保留，
  online/EMA各20份正式FVD/档，共160/160。最终1.56B online=106.79651467893109，
  EMA=68.05504574923316。报告 /same-spatial/ 已发布全部结果，8884隧道及server保持。
- 全160份评分的目标/step/协议/manifest/有限数值/精确500间隔核验通过：
  RL/validation/final_fvd_watch_20260922T030022Z/all160_score_matrix.json。
  最终8份评分每份2048条feature+metadata及FP32/有限性核验、同checkpoint/case/noise/GT
  配对和既有FVD实现重算全部通过；不是独立算法实现复算。
  RL/validation/ema_scores_verified_10000.json complete=true；
  final_fvd_watch_20260922T030022Z/complete.json 记录完成。
- 最后remote online任务stormyxie-sc-xlocal-fvd-0922025744实际Succeeded/PASS，
  controller已删除，复查TaskNotFound/0Pods。本机EMA任务
  stormyxie-sc-xlocal-local-0922025428 returncode0/PASS，worker已退出。
- 完成后停掉same-spatial monitor609245、extras650662；控制enabled=false。
  本机controller606450与finalEMA636918自然退出；不要恢复已完成的四个训练或这些控制器。
  80份DCP manifest和160份score SHA清理前后完全保留。
  RL/validation/campaign_retirement_20260922T032350Z/记录preimage、PID identity、
  精确SIGTERM与终态、最终特征审计、原始任务状态。services/progress标记retired/complete。
- final FVD foreground shell67843 已exit0、observer722522已完成，不再poll或重复执行。
  原始monitor348486/extras348487/reportserver452224保留；本机8884自动重连shell69295保留。
- 原始scaling正式分数181/200；最新实际日志concat 6219
  / all-layer CA 5474，完整checkpoint分别6000/5000。
  concat6000任务stormyxie-sc-fvdx-0922021752继续独立H200评测。原始goal保持active，
  不是全部10组完成。后续关注CA5500完整提交及concat6000正式分数；监控超时不得当失败。


## CA 的位置编码核验（2026-09-22 03:33 UTC）

- 用户要求检查CA 3D RoPE。已核对原始scaling runtime、matched FIRST CA runtime、
  same-spatial runtime_v2和RepMiniWorld当前源码/实际配置。三个vforge快照的CA都不调用RoPE；
  global CA只对投影后Q/K做RMSNorm。RoPE在ViT SA、DiT SA；condition LN+Linear不追加坐标。
- mask后ViT用原始selected索引gather temporal/row/column，未把稀疏序列重编号。
  原始/首层global CA不向DiT传condition_positions；entry用其scatter，同patch用其按(h,w)分组。
  same-spatial的分组提供显式空间对应，但不是Q/K RoPE，跨4planes没有额外时间RoPE。
- FIRST CA before_self处，query仅是noisy latent patch projection再RMSNorm，尚未经过DiT SA，
  没有加绝对(t,h,w)embedding。ViT特征受其内部位置相关混合影响，不可把它等同于显式坐标标签，
  或声称CA logits已有Q-K相对空间约束。重复query会在global CA产生相同输出。
- 原RepMiniWorld CrossAttention类虽然接受q_rope/k_rope，DiT raw调用miniworld.py:1203未传。
  semantic_query_use_3d_rope用于另一条learned-query resampler（:795/:816），
  不是当前semantic_raw_cross_attention_only路径；当前raw CA沿用无显式RoPE语义。
- 隔离CPU随机参数诊断：global CA打乱context输出maxdiff2.22e-16；相同query的输出差1.67e-16。
  same-patch打乱特征连同原坐标maxdiff0，仅打乱内容坐标固定差1.3197；
  ViT SA仅打乱内容坐标固定差0.0503，内容和坐标共同打乱再还原差1.94e-16。
  说明编码后token集合重排不变不能证明上游ViT RoPE失效。
  该诊断用冻结投影/norm/grouping类，CPU float64 matmul-softmax替代本进程attention kernel，
  不是FA4/GPU数值验证，不使用训练checkpoint，不修改运行快照或启动新训练。
- 证据 RL/validation/ca_position_audit_20260922T033210Z/；
  可复现脚本 J0/diagnostics/probe_ca_positions_20260922.py。
  若后续补CA RoPE，必须显式定义cond上一state与target下一state共同时间/空间坐标、
  gather原始mask索引；应作为新消融，不能就地改变现有checkpoint推理语义。


## 1万步 EMA 视频配对采样完成（2026-09-22）

核验时间：2026-09-22T03:36:53.717741+00:00。页面 http://127.0.0.1:8893/ ，3个10秒视频，五列GT / 775M online / 775M EMA / 1.56B online / 1.56B EMA。
使用补参数首层全空间CA→SA→FFN实验 first_layer_cross_ema_matched_20260921T093440Z；不是原all-layer CA（无EMA），也不是same-spatial。
两个尺寸均严格加载10000.dcp，EMA decay=.99、updates=10000，全部参数逐一确认等于Native载入的真实shadow，无online回退。
fvd_0000/1024/2047同GT、seed和各自checkpoint；跨4路逐case核对9段实际noise与初始encoded history一致，16观察+144生成帧，16FPS、CFG1、50Euler/state。
正式2048-case FVD：775M online85.953265→EMA59.238114；1.56B online86.960898→EMA59.395987。预览batch3，正式128/64，12份RGB hash均不同，不是原评测RGB逐像素重现。
4路H200并行均exit0，无OOM，自己的GPU进程已退出；3份MP4全部160帧解码及HTTP Range通过。HTTP8893和本地自动重连SSH保留，8891/8892/8884未变。
媒体/证据：/efs/stormyxie/reports/fvd_preview_ema_20260922T032656Z（preflight、media_validation、completion_verified、http_verified）；代码：/threed-code/stormyxie/vforge_jobs/fvd_preview_ema_20260922T032656Z。媒体留EFS，无本地下载，未改冻结snapshot。
另更正旧test-loss报告端点措辞：模型x_t=(1-t)x+tε，故t=1噪声、t=0数据；t<.2缺样本是生成末段/干净数据端覆盖不足，旧“噪声端”描述错误。数值和spike结论不变。
REPORT.md及eval_report.py已修正，旧字节与证据在 /efs/stormyxie/reports/scaling_test_loss_20260921T134909Z/timestep_endpoint_correction_20260922/ 。本节优先于旧handoff端点描述。


## DuET channel concat tensor形状核验（2026-09-22）

当前VideoRAE3D300M实际10k ckpt配置/权重与recipe核对一致；原实现meta forward训练+推理形状跟踪通过，CPU patchify坐标与逆变换逐元素相等。
训练160RGB=10states，random horizon选7origins，每state100可见prefix经Expander补924个位置后为[B,7,1024,640]；raw noisy[B,7,1024,64] concat→704→640→640，再reshape[B*7,1024,640]入13层DiT。B=4时有效DiT batch28。逐层另有sigmoid scalar gated add，入口concat不由prefix_concat_clean=false关闭。
推理K=1、dense1024tokens、跳过masked-only Expander。token索引s=(t*16+h)*16+w，4planes不并入channel。
证据与详细报告：/efs/stormyxie/reports/duet_concat_shapes_20260922/{REPORT.md,shape_trace.json,source_manifest.json}；诊断脚本在/threed-code/stormyxie/vforge_jobs/duet_concat_shapes_20260922/。只核验shape/路径，无GPU数值测试或生产源码改动，停止的DuET训练保持停止。


## DuET best_reference VAE/concat核对（2026-09-22）

已直接读best_reference YAML及111616实验原始diloco.yaml、immutable duet.py。Best使用Wan2.2 VAE48×15×20，每state1plane/300tokens；ViT1280→prefix1024；入口raw noisy48+LN(prefix)1024→1072→1024→1024，15层DiT另做sigmoid gated add。32latent窗口29origins，B4时[116,300,1024]入DiT。
原冻结源码dit_concat_clean=false只禁用第三路raw clean48维，基础noisy+prefix concat仍存在；与prefix_concat_clean字段不同。Best YAML mask.9留30补270；selected111616从106496起以mask0续训5120步，原配置已确认，续训/推理不走mask补全。两份model配置仅mask及省略的false实验字段不同，VAE相同。
详细证据：/efs/stormyxie/reports/duet_best_reference_audit_20260922/；本次仅静态源码/配置核验，未加载111616权重、未启动GPU工作。


## 原始20B vs 1.56B 六场景拼接视频完成（2026-09-22）

核验时间：2026-09-22T06:41:01.302283+00:00。用户明确允许本机无显存时submit job，并要求六case拼一起。最终单条20秒总览 http://127.0.0.1:8894/media/all_six_cases.mp4 ，页面 http://127.0.0.1:8894/ 。
六个不同场景0000/0409/0819/1024/1638/2047按2列×3行，每格GT / 1.56B / 20B。前10秒全层SA→CA→FFN（两尺寸5500步），后10秒entry concat（两尺寸6000步）；另有各10秒独立MP4。原始scaling online，未混用首层CA/EMA/DuET。
FVD2048（1.56B / 20B）：全层CA143.113754 / 113.565388；entry299.377138 / 236.199341。相同case、观察latent、全部9段实际noise逐case核验一致；CFG1、每state50Euler、16观察+144生成、256方图16FPS。每worker采样batch3，正式64/32；24份RGB中6份匹配原hash，预览不作原输出bitwise复现。
独立H200 general_job stormyxie-20b-preview-0922063140 的8个GPU采样workers均exit0；随后CPU拼接因云端字体路径缺失退出。已保存原FAIL/status/log，修复字体回退后在dev1 CPU复用已保存样本完成拼接，未重复GPU采样。云端task已精确删除并确认NotFound/0Pods。本机原等待launcher784884已因用户指示退出，未启动本机GPU采样，未停止无关任务。
总览1536×912/320帧、分组各160帧，全部解码/SHA/HTTP Range通过。HTTP8894和本地自动重连隧道保留；媒体留EFS。
结果与证据：/efs/stormyxie/reports/fvd_preview_20b_20260922T060933Z（preflight、media_validation、PASS、http_verified、remote_cleanup、completion_verified）；实际代码：/threed-code/stormyxie/vforge_jobs/fvd_preview_20b_20260922T060933Z/six_case_job。旧three-case launcher不再使用；未更改冻结runtime/config/checkpoint。


## VideoRAE3D batch128 提取与 Native cache 训练接入（2026-09-22T09:28:04.354780+00:00）

用户明确要求开始全量提取、保证当前训练格式对齐，并提高 batch/内部最大并行数量，跑起来后在代码中适配。
- 当前有效云任务 stormyxie-rae-b128-0922085323 / tenc-awsusw2 / gwm，8×H200；每卡视频B128、codec microbatch128、16个解码worker、B8解码chunk/prefetch2再组装pinned B128。旧B8任务stormyxie-rae-cache-0922084404已因用户调整要求精确删除并确认NotFound/0Pods；不恢复。
- 全量源139670条原train.jsonl，SHA a7c105fe725101d0705eb0ad53c5573e77154e475365b08c67de8484446ff6f5。逐条实际VideoReader审计全部160帧、无错误；fixed start0但保留random_clip一次torch.rand。没有加入eval或excluded数据。
- 缓存 /efs/stormyxie/cache/rep_world/videorae3d_train_20260922T082647Z_b128；FP32每样本[256,10,16,16]，channel=plane*64+channel，沿用16帧state、原checkpoint/statistics与BF16 codec，不重复归一化或量化。约340.99GiB/546shards，每256条一片，逐行SHA回读后原子提交。当前核查已提交69376/139670，仍在运行；只有COMPLETE.json + report/PASS.json表示全量完成。不要把部分分片或GPU gate PASS当完成。
- 代码/脚本 /threed-code/stormyxie/vforge_jobs/videorae3d_cache_20260922T082647Z_b128；状态/日志/验证 /efs/stormyxie/reports/videorae3d_cache_20260922T082647Z_b128；watcherPID961610继续观察并在任务终态保存日志、删除自己的云任务。关注scheduler_status/rank_*_status/FAIL/progress_estimate；watcher不发外部消息。
- B128微批128/64/256实测均与原B8 latent逐位相同；128速度11.84clip/s，64为11.77，256为11.83，因此选128。单卡峰值allocated60.74GiB/reserved75.42GiB；端到端近期约37clip/s总速，ETA随源视频解码/共享存储变化。完整当前Native step与实际落盘八条cache再核验逐位相等，保留CPU RNG消耗。
- 已在现有dirty /efs/stormyxie/vforge 上精确增量接入 Native：新增tasks/rep_world/latent_cache.py，修改shared.py/train.py；配置使用latent_cache.mode=offline + data.sources单cache + 原data.train_manifest。完整性、manifest/order、codec与stats SHA、dtype/shape/packing、固定窗口、逐行hash严格校验；原Native sampler/loader/cursor/EMA/DCP继续拥有生命周期。仅VideoRAE3D、latent conditioner或pure_tf；Native prepare/cache eval/其它codec离线未接入。当前在线训练未切换，不绕过跨online/offline semantic指纹检查。
- CPU42+27=69项通过；Ruff、docs build通过。真实H200、完整354975424参数模型B8 online/cache loss均1.999820351600647，CPU/CUDA RNG一致。真实cached B1三步训练与第2步DCP恢复至第3步loss均1.8605003356933594、sample ID匹配、EMA updates3；测试期间禁止加载codec。GPU验证已退出，缓存生产提取继续独立运行。
- 4份未来offline配置已准备在 /threed-code/stormyxie/vforge_jobs/videorae3d_cache_20260922T082647Z_b128/native_configs，正常CLI compose/preflight通过，输出/ckpt路径新建隔离，未启动新训练。完整cache完成前正式loader会拒绝。变更patch在 /threed-code/stormyxie/vforge_jobs/videorae3d_cache_20260922T082647Z_b128/native_adapter.patch；修改前备份/hashes在 /efs/stormyxie/reports/videorae3d_cache_20260922T082647Z_b128/native_adaptation_before，最终hash在native_source_hashes.json。保留其它dirty改动；无提交/推送。

最终缓存适配补充：增加RGB源路径身份校验，防止相对路径manifest搬迁后错误命中其它视频。最终43项定向+27项回归=70项CPU测试、Ruff和实际8条cached读取通过；Native GPU loss/DCP gate仍见native_gpu/PASS.json。公开配置使用data.sources=[{cache_dir:...}]，不传raw extras，不用data.cache_dir CLI shorthand或+。补丁与native_source_hashes/native_adapter_verified已刷新。


## VideoRAE3D 全量缓存完成与任务清理（2026-09-22T10:02:45.735796+00:00）

用户要求持续监控；已持续观察至生产提取、正式 Native reader 复核与 GPU 任务清理全部完成。此节取代前述 Running / ETA 快照。
- 全量139670/139670，546个FP32分片，NPY共366136594688字节（约340.99GiB）；单样本[256,10,16,16]，格式/原始manifest/codec/statistics不变。提取PASS时间2026-09-22T10:00:38Z（北京时间18:00:38）。
- 缓存 /efs/stormyxie/cache/rep_world/videorae3d_train_20260922T082647Z_b128 已发布COMPLETE.json与完整index.jsonl；报告 /efs/stormyxie/reports/videorae3d_cache_20260922T082647Z_b128/PASS.json。指纹 02db53d04bdc6441669a6f7644007323c01742eaf5b6c50ed23439835eed7cb8；所有分片生产阶段逐行hash回读已验证。
- 完成后在dev1 CPU通过当前vforge的正式CachedVideoLatents（完整性检查未绕过），核对全部139670条index和546份分片metadata/源路径身份；每片一条并覆盖跨片边界/最后尾批，共548条实际读取的SHA、finite、FP32 contiguous shape和worker RNG均通过。证据full_cache_native_reader_PASS.json；未初始化CUDA。诊断脚本在原job目录diagnostics/verify_full_native_cache.py。
- stormyxie-rae-b128-0922085323在tenc-awsusw2/gwm终态Succeeded、容器exit0；watcher随后精确删除此task并确认NotFound/0Pods，证据remote_cleanup.json。该提取任务已释放8张H200，不再恢复或监控；其它训练未动。
- 已准备的四份native_configs可使用此完整缓存；本次没有启动正式offline训练。既有70项CPU、H200在线/离线loss与DCP resume验证结论继续有效。稳定接入约束已在AGENTS，完成状态只追加HANDOFF。

## ViT-E / ViT-C Native 拆分完成（2026-09-22T10:19:49.883920+00:00）

已实现 split_ec：E 对每个 latent state 独立无 causal 编码，完整保留 tokens；C 对 E 编码序列做 state-causal prefill，仅在 C 前执行 mask，取最后 state 整张 token grid 作为 condition。E/C 端到端训练。默认 E 占原 ViT 总层数约1/4，也支持 encoder_depth；feature_layers 仍按 E+C 全局层号选择 C。
Separate encode_states/prefill_condition 走 Native engine；rollout 缓存 E 编码、只追加新生成 state，C 重做当前窗口 prefill，无 C KV cache。原单ViT路径保持默认。无USP支持（用户明确暂不考虑），不引入Megatron。
证据 /efs/stormyxie/reports/split_vit_ec_20260922/REPORT.md、validation.json、source_hashes.json：CPU326 passed/4 skipped/60 deselected；H200 BF16 FA4 12 passed；lint/docs通过；2xH200 FSDP2 entry/globalCA3DRoPE及DDP entry v2真实训练+full DCP resume至step3、EMA updates3、SDPA计数0。首次DDP拓扑配置在训练前被拒绝，原失败日志保留，纠正后的v2通过。未验证多机split，未提交长期训练。
355M split示例E5/C14/DiT10实际63,049,728 /176,349,184 /115,541,696，总354,940,608。未提交示例配置在 /threed-code/stormyxie/vforge_jobs/split_vit_ec_20260922，不代表新实验已运行。

## 最新五组实验设计更新：offline + K2 + 左右两半（2026-09-22T10:19:49.883920+00:00）

用户已将K4改K2，四宫格改左右两半，baseline也必须与K2使用相同历史范围；第5组固定总参数约355M。设计与精确meta参数候选 /efs/stormyxie/reports/future_k2_split_design_20260922/{DESIGN.md,design.json,model_budget_candidates.json}。
已确认完整缓存139670条/546片，/efs/stormyxie/cache/rep_world/videorae3d_train_20260922T082647Z_b128，fingerprint 02db53d04bdc6441669a6f7644007323c01742eaf5b6c50ed23439835eed7cb8。训练不在线codec提取，ViT/E/C仍可训练。
五组均355M、mask0、首层global CA→SA→FFN、CA3DRoPE、H200 FA4、no_zprev。1普通next；2整state均匀offset0/1；3左右各用0/1随机不重复；4split约.5:1.5:1；5split约.5:3:1固定总参数，next目标。所有组均10state源、8个origin0..7、condition history1..8state。1-3先10k，再统一next目标2k，保持8origins；4-5仅10k。第5组候选E3/C18/DiT7、FFN补齐后总355,015,704（E39,504,676/C236,618,456/DiT78,892,572），预算比约.50074:2.99925:1。
FVD拟统一2048heldout、CFG1、16观察+128生成帧、8次next-state rollout、50Euler/state，需新的real feature统计与协议身份，不能混用旧144生成帧分数。1-3在tuning0/500/1000/1500/2000，4-5在10k，共17ckpt点；online+EMA共34评测，独立评测节点。
状态明确：仅完成设计与预算/缓存核对，本轮五组尚未提交。当前canonical sampler仍K3、mosaic仍四宫格；K2/左右两半/统一8origins/两阶段调度及128帧FVD协议尚待实现与验证。旧355M mask0 campaign和监控继续保持暂停，不从历史Running恢复。

## 五组355M offline K2 / ViT split正式训练已启动（2026-09-22T10:52:49.837198+00:00）

此节取代前述“仅设计/未提交”状态。用户明确要求启动，已实现并验证K2、左右两半、baseline等8origins及10k→2k next-state tuning切换。5×8H200，共40训练卡；local8/global64；Native DDP/torch（非Megatron），FA4，FP32参数/BF16 autocast，mask0，首层global CA→SA→FFN +3DRoPE，Muон+EMA.99。
- 01_baseline: stormyxie-fk2-train-0922104219-a71ee / tenc-szr3212j / Running，已验证377步，目标12000，params=354975424。
- 02_random_k2: stormyxie-fk2-train-0922104224-72c10 / tenc-szr3212j / Running，已验证368步，目标12000，params=354975424。
- 03_left_right: stormyxie-fk2-train-0922104229-99ed3 / tenc-szr3212j / Running，已验证372步，目标12000，params=354975424。
- 04_split_balanced: stormyxie-fk2-train-0922104233-5afce / tenc-szr3212j / Running，已验证386步，目标10000，params=354940608。
- 05_split_context: stormyxie-fk2-train-0922104238-e6f93 / tenc-szr3212j / Running，已验证327步，目标10000，params=355015704。
全量offline cache仍为 /efs/stormyxie/cache/rep_world/videorae3d_train_20260922T082647Z_b128；训练wrapper禁止CodecProvider.get和SDPA，ViT/E/C继续更新。10state源统一origins0..7，condition仅states0..origin；p=.5随机offset0/1；左右两半各用0/1随机交换并逐token传递真实时间RoPE/gap。baseline及split也8origins。1–3总12k（10k主目标+2k next），4–5总10k。
代码 /efs/stormyxie/vforge；新公共flow字段 future_target_choices/target_origin_reserve/left_right_mosaic；task字段next_state_tuning_start_step按Native已完成步数切换，保持优化器/EMA/数据/RNG。旧K3/四宫格继续兼容。runtime snapshot不可变：/efs/stormyxie/reports/future_k2_split_20260922/runtime，digest ea0863fce84bc2705c31116412d61138b653102b5573bbcaec827295fe68d48f。
配置/控制脚本 /threed-code/stormyxie/vforge_jobs/future_k2_split_20260922；报告/证据 /efs/stormyxie/reports/future_k2_split_20260922；输出 /efs/stormyxie/outputs/vforge/future_k2_split_20260922。详见 REPORT.md、launch_verified.json、implementation.patch、source/launcher hashes、validation。
验证：357CPU passed/4skipped/66deselected；6H200FA4；5套实际355M B8双卡train2+full resume至3；跨tuning边界CPU严格参数/optimizer/RNG/EMA恢复通过；lint/docs PASS；正式8卡任务已100+步且finite、global64、每卡64targets、EMA updates正确。
FVD独立H200任务，2048heldout/CFG1/16观察+128生成/50Euler/state，online+EMA。1–3在10k/10.5k/11k/11.5k/12k，4–5仅10k，共17ckpt/34scores。128帧real bank全部完成；16cases baseline-online和split-context-EMA真实rollout/I3D已PASS。尚无本轮正式FVD分数，等待10k。每500步full DCP容灾。
新controller PID 976708，每60秒监控，最多3个独立8卡FVD任务，完成清理；progress.json/index.html持续更新。旧355M mask0训练/FVD/goal维持暂停，不能恢复旧campaign。
启动问题均已处理：第一版MagicMock attention计数保留tensor引用，前三个初始task OOM，已精确删除并归档输出（abandoned_instrumentation_attempts），正式任务从头重起，用train_entry_v2.py无引用计数；6步显存稳定/loss一致及正式100+步验证通过。原失败日志保留，不把这些attempt当当前实验。一次eval-smoke名称过长在调度前拒绝，短名重提交并通过；controller_errors.jsonl中的这条历史错误已有validation/eval_smoke_admission_fix.json及eval_smoke_PASS.json闭环。现行controller.py调用submit_v2.py，常规train/eval任务名长度合法；单次eval-smoke短名重试由submit_v3.py发起，已完成。

## 五组实验按用户要求从0重启 + 实时loss/速度报告（2026-09-22T11:15:36.350651+00:00）

此节替代前节运行任务与500步pretrain保存策略。用户最新明确要求改完从0开始；旧五个task均已精确删除并确认NotFound，旧训练输出保留在 /efs/stormyxie/outputs/vforge/future_k2_split_20260922，不再作为当前曲线。旧manifests/progress保存在 reports/future_k2_split_20260922/archive_before_restart0_2500。未恢复旧355M mask0或20B任务。
新输出 /efs/stormyxie/outputs/vforge/future_k2_split_20260922_restart0_2500；5×8H200，tenc-szr3212j，科学配置、seed、模型、优化器、EMA、offline输入、K2/左右/ViT拆分和训练目标保持不变。五组均确认first_logged_step=0、8个rank resume_path=none、实际多步finite loss/EMA updates。
- 01_baseline: stormyxie-fk2-train-0922111123-6a1a4，已验证96步。
- 02_random_k2: stormyxie-fk2-train-0922111128-0ec27，已验证104步。
- 03_left_right: stormyxie-fk2-train-0922111133-78a79，已验证100步。
- 04_split_balanced: stormyxie-fk2-train-0922111138-3e03b，已验证97步。
- 05_split_context: stormyxie-fk2-train-0922111143-502e7，已验证76步。

保存策略：pretrain在2500/5000/7500/10000保存；前3组10k后next-state tuning在10500/11000/11500/12000保存。split两组止于10k。FVD仍在既定17个ckpt点、online+EMA共34次，独立H200节点；real bank与协议不变。
J仍为 /threed-code/stormyxie/vforge_jobs/future_k2_split_20260922。原configs/evaluation_configs/runtime不可变；新configs_restart0_2500与evaluation_configs_restart0_2500，runs.json/eval_runs.json指向新输出。当前train_entry_v3.py + node_runner_v3.py + submit_v4.py + controller_v2.py；checkpoint_policy_v3.py仅按阶段替换Native CheckpointConfig的every_n_steps，所有保存/恢复由原Native manager执行。完整步数保存集合、disabled/final/边界及科学配置不变检查见 validation/checkpoint_policy_v3_PASS.json。新启动配置/launcher哈希 restart0_2500_hashes.json；实际新训练证据 restart0_2500_verified.json。runtime digest仍ea0863fce84bc2705c31116412d61138b653102b5573bbcaec827295fe68d48f。
controller PID 979448，60秒监控；STOP_CONTROLLER已移除。不要从历史PID 976708/旧任务名恢复。旧controller日志中的eval-smoke名称超长错误为已解决历史事件。
实时报告 http://127.0.0.1:8884/future-k2/ ，只展示训练loss/速度。training-live.html + training-live.json；collect_training.py只读当前五组日志，每30秒增量刷新，PID979431，记录collector错误到training-live-collector.log。主速度=最近1000个有效训练步的N/sum(perf.step_time_s)，不足1000用实际数量；排除启动/恢复前20步、保存步及后2个过渡步，不按速度筛选慢步。表格明确样本数/step范围；端到端单列为最近1000连续步N/sum(step_time_e2e_s)，包含保存。loss原始淡线+MA25；速度趋势100有效步N/sum，切换阶段/恢复重置窗口；日志回退删除失效尾部。对旧真实日志逐条复核统计已PASS，见validation/training_live_speed_PASS.json。
共享8884 server PID979430，只增加/future-k2/、index.html、training-live.json允许路由，旧首页/rope-mask0/same-spatial均HTTP200；原server预像与hash见preview_server_before_future_k2.py、training-live-deployment.json。本地8884 SSH supervisor仍在线。浏览器真实渲染已检查；不把新报告写到controller每分钟覆盖的index.html。
Mountpoint-S3下os.replace覆盖J现有manifest不支持，scp覆盖现有文件也可能Permission denied；本次在controller停止、留存原文后用Path.write_text更新可变manifest，原版本配置与runtime未修改。一次setup中断发生在提交前，修正后restart0_setup_v2.py通过；无训练失败或重复启动。


## Future K2 随机 token / EMA tuning / 持续监控（2026-09-22T12:02:23.363956+00:00）

本节取代前节旧03_left_right、统一2500保存、34个FVD及旧controller运行状态。用户要求case3改成随机一半tokens，而非硬左右分区；1/4/5每500保存评测；1/2/3的10k EMA作为tuning起点，重置optimizer。旧20B/355M mask0保持停止。
当前五组8xH200、DDP/local8、FA4无SDPA、offline VideoRAE3D cache、mask0、首层全patch CA→SA→FFN+3DRoPE。所有输入源10states，统一8origins。Case3随机1024 tokens中512取offset0、512取offset1，每clip/origin独立无重复；保持原P/H/W位置，逐token时间RoPE/gap对应真实offset。不是四宫格或左右两半。
Case3失败原因为GPU pytest选择器漏--gpu，28项全deselect后exit5，未进入正式训练；已修复node_runner_v5并由后续v6继承。旧stormyxie-fk2-train-0922112654-d1cde已删除；新113547-b44f9真实355M训练推进。token_mix/gpu_PASS.json为2项H200 FA4，CPU372 passed/4 skipped/68 deselected。
当前任务核查：
- 01_baseline: stormyxie-fk2-train-0922112649-d2d59 / Running / 1387 steps; EMA policy active=False.
- 02_random_k2: stormyxie-fk2-train-0922111128-0ec27 / Running / 2060 steps; EMA policy active=False.
- 03_token_mix: stormyxie-fk2-train-0922113547-b44f9 / Running / 992 steps; EMA policy active=False.
- 04_split_balanced: stormyxie-fk2-train-0922112659-dfedd / Running / 1424 steps; EMA policy active=None.
- 05_split_context: stormyxie-fk2-train-0922112704-9331b / Running / 1260 steps; EMA policy active=None.
保存/FVD：1/4/5从500起每500；2/3 pretrain每2500，正式FVD从10k起；1/2/3在10k之后做2k next-state tuning、每500保存评测。4/5止于10k。每个指定checkpoint均online与EMA .99分别严格加载、独立结果，共74checkpoint点/148score。固定2048 heldout、CFG1、50Euler/state、16观察+128生成、per-case noise seed20260920；与旧144生成帧FVD不混比。独立H200节点，max3个8卡eval；500步首批已自动提交。
EMA tuning代码已接入canonical tasks/rep_world：next_state_tuning_init=ema_reset_optimizer在Native step10000前精确复制EMA shadows到online、optimizer.state.clear，LR/scheduler/data cursor/RNG保持，不重置EMA shadow或全局更新计数。CPU379 passed/4 skipped/68 deselected；真实2xH200 BF16 FA4小模型Native train2 + full DCP resume到4，边界2两rank逐参数精确copy、optimizer清空、LR保留、EMA updates4、SDPA/online codec禁止，证据ema_tuning/gpu_PASS.json和gpu_native_v3。早期smoke的warmup0与num_workers0/prefetch配置错误已修正，仅测试启动失败；最终v3通过。Ruff scoped6文件与docs build通过；全changed lint仅既有same_spatial测试191行格式问题，不顺手修改。
runtime=/efs/stormyxie/reports/future_k2_split_20260922/runtime_token_mix_ema_tuning_v2; source_digest=3dd008e5bf4b9fb0aa8cfc337dc991b6a6f549754a4d188044adc87e85bc1511。旧runtime/config/protocol snapshots不改。J/runs.json引用configs_ema_tuning_v1的新1/2/3配置；为避免丢进度，controller在policy_upgrade_min_step对应的下一完整DCP后才精确停止旧worker并full-resume新策略；pretrain进度/optimizer/EMA/RNG保留。当前仍待升级的worker不应声称已在内存启用新策略；以receipt.policy_version和progress.ema_policy_active为准。
tuning step0的online与EMA权重均是10k EMA，因此复用该EMA FVD并写明来源到ema_tuning/initial_fvd_scores.json；10k pretrain原始online分数仍保留，不能把它冒充tuning step0。
控制器controller_v5.py / submit_v7.py / node_runner_v6.py / train_entry_v4.py，PID 986412。每60秒刷新调度、提交已完整checkpoint的FVD、按既有规则恢复被抢占训练、完成后清理自己的eval任务。普通代码失败不盲目重试；超过180s无日志或训练Failed/Terminated写progress.alerts、monitor_alerts.json和仅变更时的monitor_events.jsonl。heartbeat/controller_errors需一起看，历史旧错误不作为当前故障依据。
实时报告 http://127.0.0.1:8884/future-k2/ ，collect_training_fvd_v3.py每30秒增量更新training-live.json；保留loss/训练速度两图；用户授权后新增FVD online/EMA曲线、双log选择与逐checkpoint表。最近1000有效步N/sum耗时，剔除启动/恢复20步及保存步后2步；报告新增实际failure reason、monitor age与待EMA策略升级说明。真实Chrome渲染已检查，HTTP200。共享8884远端server和本地scaling-preview-supervisor持续在线；本地隧道有短暂断线后自动重连，不能把页面暂失联误判为训练失败。
J=/threed-code/stormyxie/vforge_jobs/future_k2_split_20260922；R=/efs/stormyxie/reports/future_k2_split_20260922。核心证据token_mix/、ema_tuning/、activation/manifest备份；新增fvd_coordination_v2.py仅显式兼容已审核的training-only EMA代码前协议hash，EMA已有eval函数AST相同，模型/rollout/配置/metric未改，见inference_protocol_compatibility.json；case3 batch map旧label已修正128。调度API偶发DNS超时，controller_v5即使提交失败也继续写本地训练进度和heartbeat，不把ObservationUnavailable当模型失败。、progress.json、receipts。当前任务进度动态变化，以新heartbeat和实际task为准。无commit/push。


## 最新：六组 / RGB 等网格 / 失败从0 / dev1-only FVD / 报告路径修复（2026-09-22T12:39:18.547383+00:00）

本节覆盖此前远端FVD节点、旧1/2任务、旧报告采集器与未经验证的策略迁移说明。用户最新要求：失败训练直接从0重训；只保留六个消融训练任务和dev1，所有后续eval都在dev1，不再提交远端eval。旧20B、355M mask0、condition实验保持停止。

### RGB case6 已验证并启动
Case6以case4为基线，split ViT-E/C/DiT深度5/14/10；RGB patch temporal4、spatial16，16×256×256 RGB经E输出4×16×16=1024 tokens/state，width1024，和原VideoRAE+ViT网格完全相等；C输入/输出condition网格同样1024。不是只对齐总token数而改变时空排列。参数358021824 vs case4 354940608，+3081216（0.87%），仅RGB patch projection/modality embedding差异。H200×8/local8/global64，mask0，首层全patch CA→SA→FFN+3D RoPE，FA4，EMA.99，10k，无tuning，每500保存+online/EMA FVD。
RGB训练目标仍读同一完整offline VideoRAE3D packed-FP32 cache；配对原始RGB，核对固定160帧窗口、只消耗一次clip RNG，不加载在线codec。训练E只读取对应8个历史origin的RGB，E逐state独立；rollout E读取观测RGB/后续解码生成RGB，不能读latent或未来GT。
Canonical实现涉及models/rep_world {conditioner,split_conditioner,model,inference,config}及tasks/rep_world {data,latent_cache,shared,train}。冻结parent runtime不改；RGB独立snapshot /efs/stormyxie/reports/future_k2_split_20260922/rgb_split/runtime_v1，digest 6b9fb8ee2f112a5416e3f273de0171bc0ed6cacaf981b40654aa976ecd23608b；jobs=/threed-code/stormyxie/vforge_jobs/future_k2_rgb_split_20260922。RGB source preimages/hashes在R/rgb_split/implementation。
验证：54个定向CPU通过；全模型/task回归388 passed/4 skipped/69 deselected，两个错误消息regex问题修复后定向2 passed（不是一次完整390-pass运行）。1×H200 FA4 RGB梯度+闭环rollout PASS；2×H200实际358M/local8 train2+full-DCP resume3 PASS，实际E输出[8,8,1024,1024]、RGB[8,3,128,256,256]，EMA updates3，无SDPA/online codec。证据R/rgb_split/validation/native_PASS.json及full_model。Ruff scoped11files、docs build PASS。
RGB online/EMA各16cases真实128生成帧FVD采样门禁PASS。首个eval-smoke失败为eval配置误继承offline模式；已用不可变evaluation_configs_v2修正为online eval、codec encode_bsz1、FSDP2，训练配置不改；新smoke通过，旧失败证据保留。R/rgb_split/validation/eval_smoke_PASS.json。正式RGB sampling_batch64，parent128；相同cases/noise/CFG/帧协议。

### 失败从0与健康任务的EMA阶段切换
旧1/2在策略升级full-resume时失败，原因是semantic_config新增next_state_tuning_init字段导致严格校验拒绝，非loss/OOM。用户要求后，1/2已用全新目录从0启动，8rank resume.path=none；seed/模型/EMA/optimizer重新初始化。旧checkpoint/FVD结果保留到R/restart_history，对当前对比不再显示；新500步结果会重新测。
- 01_baseline: Running, 353 steps, task=stormyxie-fk2-train-0922122608-58c8c, generation=restart0_20260922T122550Z_1.
- 02_random_k2: Running, 363 steps, task=stormyxie-fk2-train-0922122801-ad670, generation=restart0_20260922T122747Z_1.
- 03_token_mix: Running, 2564 steps, task=stormyxie-fk2-train-0922113547-b44f9, generation=random_token_mix.
- 04_split_balanced: Running, 3052 steps, task=stormyxie-fk2-train-0922112659-dfedd, generation=ckpt500.
- 05_split_context: Running, 2696 steps, task=stormyxie-fk2-train-0922112704-9331b, generation=ckpt500.
- 06_split_rgb: Running, 345 steps, task=stormyxie-rgb6-train-0922121807-bb4b3, generation=rgb_split_v1.
上面是核查快照；后续以runs.json、receipts和日志为准，不能自动恢复历史task。
训练失败策略由restart_zero_v3.py执行：确认当前terminal失败、保留证据、停止相关旧评测、创建独立新输出与配置、切换train/eval manifest，retire旧结果，再提交step0。连续三次新的失败后保留告警供检查，避免资源无限重试。调度API超时不是失败。已有运行中的健康3/4/5/6不从0重启。
健康case3不在2500中断：policy_upgrade_min_step=10000，在完整10k DCP时切换到已验证EMA初始化策略。EMA_tuning helper只允许full checkpoint在边界前/边界处，将唯一字段online→ema_reset_optimizer；其他semantic差异全部拒绝。在独立目录复制manifest、硬链接所有原始payload，原checkpoint、组件integrity、optimizer、EMA、scheduler、RNG、data cursor均不改；仍由Native严格full resume校验。不是关闭strict或伪造tensor恢复。此工具仅用于健康worker的明确未来阶段策略升级；失败worker走从0。
迁移验证R/ema_tuning/policy_migration_validation：CPU拒绝LR/mask/边界/已过边界修改；真实2H200旧策略train2→新策略严格resume4，边界2精确复制EMA，optimizer确实有恢复动量后清空，LR不变，两rank PASS。新node_runner_v7.py / train_entry_v5.py在迁移存在时显式传路径；其他情况下Native auto。1/2从0启动时没有迁移路径。首阶段10k后统一next-state2k；tuning0分数对应10k EMA，optimizer reset、LR/data/RNG连续；tuning500/1000/1500/2000 online+EMA分开测。

### 资源与持续调度
仅六个远端H200训练任务，共48卡；szr最新资源核查只有这六个Running，aws查询Stormyxie tasks为空。原远端eval/smoke全部清理，receipts无active非train项。资源证据resource_audit_latest.json、tenc-awsusw2_resource_audit.json。
Parent controller_train_only_v8.py PID992382，submit_v9.py明确拒绝kind!=train；RGB controller_rgb_train_only_v3.py PID992383，submit_rgb_v2.py同样拒绝远端eval。每60秒监控训练、失败重启、完整checkpoint；会写progress/heartbeat/alerts。不要重新运行旧带远端eval提交的controller_v6或controller_rgb.py。
本地共享队列J/dev1_eval_queue.py PID992067，只在dev18卡空闲时启动一个完整8-rank Native/FSDP2评测；两组study共用GPU pool，无Koala task submission。现有部分完成的feature按既有协议安全复用。队列15秒扫描，local_eval_status.json/dev1_eval_heartbeat.json/local_eval_receipts/tasks保存实际PID/进度；连续同一checkpoint三次失败后等待检查。训练重启时会先终止对应本地eval，避免不同generation混写。
当前共188个计划FVD分数：parent148+RGB40；固定2048heldout、CFG1、16观察+128生成、50Euler/state、per-case seeds，online+EMA.99。1/4/5/6每500保存评测；2/3 pretrain每2500保存，FVD从10k及后续tuning500步开始。所有FVD在dev1，与训练节点分离，不添加额外云GPU。

### 报告已修正
URL http://127.0.0.1:8884/future-k2/?refresh=restart-dev1 。训练loss/正常速度/FVD保留，online/EMA与双log可选，六组一起显示，增加从0重启轮次和dev1评测位置。速度仍最近1000有效正常步N/sum耗时，剔除启动/恢复20及保存步后2步；不筛慢步。
报告旧bug：manifest更新output后采集器保留原Log实例路径，导致新task Running但读旧曲线、触发错误stalled。collect_training_six_v5.py PID992347每轮重新核对每组output路径，变更就重建reader，清除旧曲线；启动后已核对所有六组log_path/step与真实文件一致、无告警，浏览器截图验收outputs/future-k2-restart-dev1.png。JSON30秒刷新、fetch no-store；新HTML需刷新一次。不能把旧日志停更、API ObservationUnavailable或case3待tuning策略提示当成当前训练失败。
所有新源码/配置/worker versions已写出，未commit/push。HEAD/HANDOFF已有其他历史更改不覆盖；本次preimage/hash见handoff_rgb_dev1_update.json。


## 继续监控：RGB正式FVD启动修复与报告评测告警（2026-09-22T13:31:41.954400+00:00）

六组训练仍在推进；1/4/5按500保存，case2首个2500完整DCP已核对，case3在10k切换EMA策略的安排不变。当前FVD已12份；4组1000步online907.824754/EMA692.58，1组新一轮1000步online864.63/EMA629.93，精确数值以score.json为准。
RGB500正式eval最初6次启动前失败（online/EMA各3），缺失R/rgb_split/fvd/legacy_unlocked_targets.json；未加载模型，eval.log不存在。核查无legacy未加锁eval进程、全部远端eval/smoke已删除后创建targets=[]清单，保留原FAIL.json/receipt备份并将对应receipt状态改Resolved（previous_state=Failed）重新开放本地重试预算。证据rgb_split/validation/legacy_targets_fix/PASS.json。正式64-batch RGB500EMA已在dev1运行且产出512份FVD特征，PID997629，还未完成2048样本分数，不宣称正式评测完成。
新增collect_training_six_v6.py，PID997835，从当前generation的local_eval_receipts检查未解决的eval失败，显示在报告告警；旧generation与Resolved历史不冒充当前错误。FVD字幕增加待评测数量。训练路径动态切换修复保留。当前无告警。详细核查monitor_20260922T1331.json。目标仍进行中，尚未到10k/tuning，也未完成全部188个FVD；不标记goal完成。


## RGB 首次完整FVD与汇总依赖修复（2026-09-22T13:42:09.488525+00:00）

RGB 500-step EMA完整2048cases、128生成帧、CFG1、batch64得到FVD=2001.018654634，有效score位于R/rgb_split/fvd/results/06_split_rgb__ema/00500/score.json。生成全程dev1，特征2048×400 finite；逐case identity、seed、checkpoint/config/protocol一致性已核查，见validation/rgb_500_ema_features_PASS.json。
首次全量汇总失败原因为RGB job目录遗漏fvd_pool.py，smoke没有执行commit_score因此未覆盖该依赖。已复制parent完全相同的只读feature validator（无训练/推理/指标协议变更），证据validation/fvd_pool_dependency_fix.json。保留失败日志和receipt preimages，将对应旧ModuleNotFoundError attempts标记Resolved，避免错误耗尽重试预算。
2048原样本全部复用，在输出Lease保护下调用现有commit_score进行CPU汇总，无重采样、无额外GPU；validation/rgb_500_ema_aggregate_PASS.json记录通过。RGB online500现已在dev1运行。后续commit_score依赖已安装并通过实际导入。原legacy清单缺失修复仍有效。
case3生产2500 checkpoint的唯一语义字段迁移预检也PASS：ema_tuning/case3_production_fingerprint_preflight.json，源manifest未修改，切换仍在10k，保留正常主训练。case1/2已从0的新轮次；六组训练继续，goal未完成。当前偶发调度接口ObservationUnavailable需结合实时日志，不能据此重启。


## 监控查询改为批量观察（2026-09-22T13:56:37.376159+00:00）

单task并发CLI查询频繁超时但训练日志继续推进；批量API两次实测返回全部目标Running（1.81s、11.82s）。已切换parent controller_train_only_v9.py PID1000887、RGB controller_rgb_train_only_v4.py PID1000888。只更换监控进程，六个train worker、dev1 FVD queue PID992067和collector PID997835未重启。每cluster/namespace一次task-list；只观察receipts中的精确task身份；列表缺项必须单task确认，超时/畸形/重复身份只记ObservationUnavailable，不推断失败或NotFound。7项观察contract通过，证据batch_status_validation.json、batch_status_deployment.json、batch_status_live_PASS.json。
submit_v9/submit_rgb_v2仍只允许远端train；失败从0和case3在10k切换EMA策略保持。HTML对ObservationUnavailable且日志新鲜时明确显示“查询暂不可用·日志持续更新”，保留原始status/error，Failed不被覆盖；report_scheduler_observation_labels.json。
RGB500完整online FVD2065.735574187、EMA2001.018654634均已进入报告，实际2048cases/128生成帧/batch64；online全流程包括自动commit_score成功，证明补齐的fvd_pool依赖可用于正式worker。证据rgb_split/validation/rgb_500_online_ema_complete.json。剩余训练/188份FVD和tuning仍进行中，goal不完成。


## 首个完整主训练完成：case4 10k（2026-09-22T15:21:03.528995+00:00）

Case4 split_balanced 已完整训练到10000：全部20个checkpoint（500..10000，每500）manifest completed_steps逐一核对；Native worker PASS、调度Succeeded（2026-09-22T15:19:05Z），controller已delete并确认NotFound，receipt deleted=True/final_status=Succeeded。最终10000.dcp含weights/optimizer/trainer，online和EMA FVD均已入dev1队列。证据R/case4_pretrain_complete_verified.json。Case4按用户设计无tuning，不得重启。其FVD尚未全完成。

当前只剩5个远端train receipts活跃，共40张H200；dev1继续8卡本地FVD，无新增远端eval。Parent/controller v9与RGB/controller v4未修改、未重启。报告已有25份FVD；最新baseline2000 EMA=205.6759366076899，RGB1500 online=1087.1227677992242/EMA=274.14854100651564，均2048cases。当前dev1在baseline2000 online，具体PID以local_eval_status和/proc实时为准。

Case1/2仍主训练约7k；case3约9.3k，下一关键事件是10000完整DCP后的EMA tuning策略切换，必须核对实际旧任务退出、新任务严格full-resume、EMA复制/optimizer reset指标和tuning checkpoint，不能仅凭配置宣称切换成功。Case5约8.8k、case6约3.6k仍训练。目标继续active：剩余主训练、前三组2k tuning及全部188份FVD未完成。


报告完成状态显示已修正（2026-09-22 15:27 UTC）：training-live.html 对 Succeeded 显示绿色“已完成”，顶部单列完成组数；已完成step>=target不会触发日志停更告警。真实Chrome DOM验收6行、case4绿色完成、顶部1组完成PASS。仅HTML改动，无训练/评测/collector/controller重启。证据R/report_completed_status_fix.json，HTML preimage=R/training-live.before-completed-status.html；最新HTML SHA256=74097287bb4b34c1830e580d158204abd3ccbdc2ebb4bde8a14f02477125eb01。本地work/future-k2-live/training-live.html同步。报告26份FVD，baseline2000 online473.97651301705673/EMA205.6759366076899；dev1已转case4 2000 EMA。case3约9.6k，仍需核实实际10k EMA tuning切换。


## Case3 生产任务 EMA tuning 切换已实证通过（2026-09-22T15:43:30.825429+00:00）

03_token_mix原任务stormyxie-fk2-train-0922113547-b44f9在完整10000.dcp后按计划delete并确认消失，superseded_reason=ema_tuning_policy_upgrade；新任务stormyxie-fk2-train-0922153750-380cf已Running，policy_version=ema_tuning_v1。Native从迁移副本的completed_steps10000严格full-resume。源manifest SHA=d4c58288eea76ff4269f578a27c8d229083ab76665d7501a2988e4c27dd571be仍一致，34个weights/optimizer/trainer payload逐个stat确认源/目标同device/inode/size，原tensor payload未复制修改。实际manifest差异见R/ema_tuning/case3_production_transition_PASS.json。

生产step10000日志确认tuning_init_ema=1、tuning_optimizer_reset=1、tuning_source_ema_updates=10000、EMA decay.99；target_phase=next_state_tuning，horizon1 fraction1、horizon2 fraction0。LR保持1e-4，rank0输入sample_ids与边界重放完全相同；后续已推进到step10158且loss有限。旧worker边界后临时推进的38步已由完整10k恢复丢弃，不能计入新EMA tuning优化路径；图表应使用回退后的新段。源原始10k online FVD仍独立，tuning0 online/EMA复用10k EMA分数。

Case3切换已完成，但2k tuning及10500/11000/11500/12000的online/EMA FVD仍未完成。Case1/2继续各自10k边界自动EMA初始化；case4已完成并释放；case5/6仍主训练。报告已有28份FVD，当前dev1在case5 2000 EMA。资源仍5组训练40张H200+dev1 8卡，无远端eval新增。目标未完成，保持active。


## 2026-09-22 15:52 UTC — case 5 completed; 32 training GPUs remain

- Case 5 (split context 0.5:3:1) completed all 10,000 pretraining steps, with all 20 committed Native checkpoints from 500 to 10,000. Native task PASS and scheduler Succeeded at 15:50:26 UTC are verified.
- Its task stormyxie-fk2-train-0922112704-9331b was deleted after success and its 8 H200s released. Do not restart or tune this completed case.
- Only cases 1/2/3/6 remain as four 8-H200 training tasks (32 GPUs); all FVD stays on dev1. Case 5 final 10k online and EMA evaluations are queued.
- Verification: /efs/stormyxie/reports/future_k2_split_20260922/case5_pretrain_complete_verified.json. Current report contains 29 completed scores; evaluations and cases 1/2/3/6 remain unfinished.


## 2026-09-22 16:12 UTC — explicit EMA-origin tuning FVD view

- Report now offers 全部 checkpoint and 前 3 组 · EMA 起点 tuning. Link: /future-k2/?fvd_view=tuning#fvd-results. Original total-step scores remain unchanged.
- Tuning view maps checkpoints 10500/11000/11500/12000 to updates 500/1000/1500/2000. Only a completed 10k EMA score creates the two identical online/EMA update-0 display points. Raw 10k online is never used as tuning initialization, and aliases do not inflate the 188 expected raw evaluations.
- Double-log omits zero from the graph and explicitly preserves update 0 in the table. Empty state remains until real applicable scores arrive; no synthetic score is published.
- Pure JS semantic checks (different raw-online vs EMA values, missing EMA, excluded cases, step mapping, unchanged full view), live Chrome DOM and visual checks PASS. No collector/trainer/controller restart. Verification: /efs/stormyxie/reports/future_k2_split_20260922/report_tuning_view_fix.json.
- Case 3 tuning checkpoints 10500 and 11000 have manifest component file counts/bytes verified, finite next-state-only loss and matching EMA update count; both weights queued. Proof: /efs/stormyxie/reports/future_k2_split_20260922/ema_tuning/case3_tuning1000_verified.json.


## 2026-09-22 16:29 UTC — case 3 finished; cases 1/2 entered EMA tuning

- Case 3 completed 10k pretraining plus all 2k next-state tuning. Native PASS and scheduler Succeeded at 16:27:02 UTC; task stormyxie-fk2-train-0922153750-380cf deleted after success. Do not restart this completed run.
- Tuning checkpoints 10500/11000/11500/12000 have complete Native weights/optimizer/trainer payloads with matching manifest file counts and bytes. Final EMA update count is 12000, phase next_state_tuning, finite loss. All five 10000/10500/11000/11500/12000 online+EMA pairs are queued on dev1. Tuning update 0 reuses the 10k EMA score, as documented; raw pretraining online remains separately visible.
- Cases 1 and 2 actually crossed 10k, each from its own EMA with optimizer states cleared. Boundary log records verify tuning_source_ema_updates=10000, EMA decay 0.99, next-state-only targets, and unchanged LR. Proofs: /efs/stormyxie/reports/future_k2_split_20260922/ema_tuning/01_baseline_restart0_20260922T122550Z_1_boundary_PASS.json, /efs/stormyxie/reports/future_k2_split_20260922/ema_tuning/02_random_k2_restart0_20260922T122747Z_1_boundary_PASS.json.
- Only cases 1/2/6 remain on the remote cluster: three 8-H200 tasks, 24 training GPUs. Cases 3/4/5 are training-complete. Dev1 FVD queue remains active; complete study still requires finishing 1/2 tuning, case6 pretraining, and all 188 evaluations.
- Completion evidence: /efs/stormyxie/reports/future_k2_split_20260922/case3_pretrain_tuning_complete_verified.json.


## 2026-09-22 17:16 UTC — all three future cases finished tuning; only RGB training remains

- Cases 1 and 2 completed 10k pretraining plus all 2k next-state tuning. Scheduler completion times: case 1 17:12:03 UTC; case 2 17:13:56 UTC. Native PASS, final 12k manifests, all tuning payload file counts/bytes, finite next-state-only final losses, EMA update count 12000, successful task cleanup, and full evaluation queue coverage are verified.
- Baseline retains all 24 checkpoints every 500 steps through 12k; random K=2 retains 2500/5000/7500/10000 plus 10500/11000/11500/12000. Initialization from each run’s own 10k EMA with optimizer reset and unchanged LR was verified at the actual boundary. Completion proof: /efs/stormyxie/reports/future_k2_split_20260922/case1_case2_pretrain_tuning_complete_verified.json.
- Training-complete cases: 1/2/3 at 12000, 4/5 at 10000. Do not restart or resubmit these successful tasks. Only case 6 raw RGB remains as one remote 8-H200 training task; its target remains 10000 with checkpoint/eval every 500. All FVD stays on dev1, with no remote eval submissions.
- The study remains incomplete: case6 training and remaining FVD evaluations must finish. Raw evaluation total remains 188; tuning update-0 online/EMA display aliases reuse the 10k EMA score and do not inflate that count. Keep goal active.
- Latest queue/protocol/generation inventory is /efs/stormyxie/reports/future_k2_split_20260922/eval_inventory_latest.json. New Native checkpoints continue to queue automatically, and training-live.html / training-live.json remain the live report.


## 2026-09-22 17:43 UTC — dev1 FVD queue v2; alternate future/tuning with regular checkpoints

- Active local queue is now /threed-code/stormyxie/vforge_jobs/future_k2_split_20260922/dev1_eval_queue_v2.py, PID 1017736. Do not restart the old v1 FIFO queue (old PID 992067 was deliberately terminated after its active evaluation finished).
- Policy: finish partial evaluations first; otherwise alternate one future/tuning evaluation (cases 1/2/3, steps 10000–12000) with one regular checkpoint evaluation. The future lane first evaluates all three 10k EMA origins, then orders remaining pairs by step/name/weight. This exposes the core future ablation earlier while continuing dense pretraining curves. All 188 evaluations and all frozen sampling configs remain in scope.
- Selection tests and real queue dry-run passed. Candidate discovery/validation is AST-identical to v1. Live first dispatch is baseline 10k EMA, with next_lane=regular persisted in /efs/stormyxie/reports/future_k2_split_20260922/dev1_eval_dispatch_state.json. Selection state survives queue restarts.
- Handoff did not interrupt the active evaluation: old coordinator alone was SIGSTOPed, its existing child completed all 2048 cases and became a zombie with observed exit_code=0; Native PASS and valid score were verified, receipt reconciled, then old coordinator terminated and v2 launched. Evidence: /efs/stormyxie/reports/future_k2_split_20260922/queue_v2_handoff_PASS.json and /efs/stormyxie/reports/future_k2_split_20260922/queue_v2_activation_verified.json.
- For a future graceful coordinator upgrade, v2 supports /efs/stormyxie/reports/future_k2_split_20260922/DRAIN_LOCAL_EVAL: when present it finishes the active evaluation and exits before dispatching another. Remove that flag before restarting. STOP_LOCAL_EVAL remains the immediate-stop control. Neither flag is currently set.
- Queue identity/log: /efs/stormyxie/reports/future_k2_split_20260922/dev1_eval_queue_v2_identity.json / /efs/stormyxie/reports/future_k2_split_20260922/dev1_eval_queue_v2.log. Parent/child training controllers are unchanged. Only RGB case6 still trains remotely; all evaluation remains on dev1.

## 2026-09-22T17:55:54.132018+00:00 — user-reported stale status display audited

- User saw cases 1/2/3/5 as problematic. Fresh served training-live.json and current scheduler terminal receipts confirm all four Succeeded and deleted after success: 1/2/3 at 12000, 5 at 10000. Case4 is also complete; only case6 continues (approximately 6.8k). Do not restart completed cases.
- Historical HTML colored Succeeded as warning; data polling updated JSON but not the already loaded JavaScript. Fresh Chrome render was correct. Live HTML now explicitly shows training-running/completed/failed counts, labels completed logs as last-step logs, and shows per-case FVD completion separately. A 60-second page-code comparison shows a reload notice when newer HTML script is available. Existing older tabs need one page reload via /future-k2/?refresh=status20260923.
- HTML SHA256: 3d023958023cf8d66c8c62052a88f3d93fe0d10249169cd2208f02f72d333ac2. Preimage and change proof: R/report_status_audit_fix.json. Current training evidence: R/report_status_audit_training_evidence.json. JavaScript parse, real Chrome DOM for normal/tuning views, and visual inspection PASS; proof R/report_status_audit_browser_PASS.json. No training, collector, or controller restarted.
- Actual first v2 future-lane evaluation finished: baseline 10000 EMA FVD 75.97003339837147 (2048 cases). Report raw count 44; tuning view correctly shows two identical update-0 online/EMA aliases without inflating raw count. Backend ema_tuning/initial_fvd_scores.json agrees.
- Actual v2 second dispatch is regular case4 online3000, with persistent next_lane=future_tuning, confirming real alternation. Current-generation/protocol inventory PASS: due174, raw completed44, active1, pending129, missing0. Cases2/3 each have all10 required raw evaluations queued. Resource policy remains only case6 remote training (8H200) plus dev1 evaluation (8H200).
- Goal remains active: finish case6 training and all188 raw evaluations. Empty FVD curves for cases2/3 mean queued evaluation, not failed training.

## 2026-09-22T18:20:14.937930+00:00 — all three 10k EMA / tuning-0 FVD origins complete

- Fresh-generation full2048 results under identical CFG1 / 50Euler / 16observed+128generated / seed20260920: baseline75.97003339837147, randomK2=59.022132849542686, random-token50/50=295.61510559985527. These are before tuning, not final 12k scores.
- Each succeeded receipt/RC0, current checkpoint path, real/fake count, protocol fields, report score and backend tuning-0 alias verified. Proof: R/all_three_tuning0_ema_fvd_verified.json. Report raw total48/188; each origin aliases online+EMA display at update0 without adding duplicate raw evaluations.
- Case3 poor tuning-0 FVD is a measured outcome, not an execution failure; do not restart or discard it. All three training/tuning runs already completed; future 10500/11000/11500/12000 results are still pending.
- Active dev1 queue v2 has advanced to case5 online3000 (PID1020049 at verification). Future/regular alternation continues. Case6 still trains near7.3k; committed7000.dcp component counts/bytes checked and both weight evaluations queued. No other remote training or remote eval jobs should be started.
- Goal remains active until case6 finishes10k and all188 raw FVD scores are verified. Live report: /future-k2/; tuning comparison: /future-k2/?fvd_view=tuning#fvd-results.

## 2026-09-23T02:37:12.516835+00:00 — all training and tuning evaluations complete; FVD queue continues

- Six training jobs are terminal Succeeded and deleted/released. Cases1/2/3 completed10k pretraining plus2k next-state EMA-origin tuning; cases4/5/6 completed10k. Case1/4/5/6 have every500-step committed checkpoints (24 for case1 through12k, 20 each for4/5/6 through10k); cases2/3 retain2500/5000/7500/10000 and each500-step tuning checkpoint. Final Native manifests include verified weights/optimizer/trainer payload counts and bytes, and worker PASS files exist. Proof: R/all_six_training_complete_verified.json. No remote training remains.
- All three future cases have full 0/500/1000/1500/2000 tuning FVD for both EMA and online under fixed2048-heldout/CFG1/50Euler/16+128/seed20260920. Tuning step0 online+EMA aliases each case's own10k EMA, while raw10k pretrain online remains separately recorded. EMA FVD at tuning0→2000: baseline75.970033→70.647581, randomK2 59.022133→54.566760, random-token50/50 295.615106→66.698575. Full exact per-step values and score/checkpoint/protocol validation: R/all_three_tuning_fvd_complete_verified.json. Served Chrome tuning table also checked for all15 rows and final values.
- All188 planned evaluation keys are due, with committed checkpoints; latest inventory at this milestone: completed114, active1, pending73, missing0, report sync gap0, protocol/generation checks PASS. Inventory file R/eval_inventory_latest.json is refreshed on later audits. Active local queue v2 remains PID1017736 using dev1 8H200 only; train controllers keep status/report current. No cluster eval jobs.
- Goal remains active until the remaining FVD scores finish. Do not resume or restart successful training. The live report at http://127.0.0.1:8884/future-k2/ is current; tuning view uses ?fvd_view=tuning#fvd-results.

## 2026-09-23 03:00 UTC — FVD moved to four H200 queue tasks; case 1/4/5 before RGB

- User superseded the earlier dev1-only evaluation restriction: run up to four independent 8×H200 FVD tasks through the remote task queue. Evaluate all remaining case 1/4/5 checkpoints first, then RGB case 6. Cases 2/3 and all six training jobs are already complete; do not restart training or tuning. No B300 jobs.
- Dev1 queue v2 PID 1017736 was gracefully drained via `R/DRAIN_LOCAL_EVAL` after its active score committed; leave the drain flag in place. Remote dispatcher is `J/remote_eval_queue_145_first_v1.py`, PID 1042476 at launch, identity `R/remote_eval_queue_145_first_v1_identity.json`, live state `R/remote_eval_queue_status.json`, log `R/remote_eval_queue_145_first_v1.log`. It permits at most four undeleted remote eval receipts, selects 1/4/5 before RGB, and automatically fills freed slots every 30 seconds. Parent and RGB train-only controllers (PIDs 1000887/1000888 at handoff) still reconcile terminal receipts and delete finished tasks. Use actual process/receipt state rather than these PIDs alone.
- A mistaken first RGB submission used the parent study's runner path because both submit modules imported a shared `common` module in one Python interpreter. That task failed and was deleted; incident evidence is `R/tasks/stormyxie-rgb6-eval-0923024831-80c48/submission_incident.json`. Correct RGB task `stormyxie-rgb6-eval-0923025028-80c48` was then explicitly stopped/deleted at the user's RGB-last request; its partial outputs remain reusable. The dispatcher runs each study's submit module in a fresh interpreter and asserts its `J` and `R` paths to prevent recurrence.
- Initial four production H200 evaluations were 01 EMA6000, 04 EMA6000, 05 EMA6000 and 01 online6000. First two new scores committed; dispatcher automatically backfilled with 04 online6000 and 05 online6000. At switchover 116/188 scores were complete; consult `R/remote_eval_queue_status.json` for current completion, active tasks, 1/4/5 priority and deferred RGB count. Fixed protocol remains 2048 held-out cases, CFG1, 50 Euler/state, 16 observed+128 generated frames, seed 20260920, online and EMA 0.99, H200 FA4.
- Live report `http://127.0.0.1:8884/future-k2/` now uses versioned `J/collect_training_six_v7.py` (PID 1042785 at launch) and updated `R/training-live.html`; its FVD panel shows H200 queue activity, pending and deferred RGB. Old v6 collector was stopped, and HTML preimage is `R/training-live.before-remote-queue-20260923.html`. Existing train/controller jobs were not restarted. Confirm `R/training-live.json` refreshes within 30 seconds and matches the queue after transient receipt/score reconciliation.
- Next: keep all four slots on 1/4/5; investigate `R/remote_eval_queue_errors.jsonl`, `R/tasks/<task>/FAIL.json`, and scheduler only if a task stops making progress. Once all 1/4/5 scores are valid, dispatcher will automatically resume RGB. Finish with an independent 188/188 score/protocol/checkpoint audit and report reconciliation before marking the active goal complete.

- 2026-09-23 03:06 UTC user asked to reduce manual polling: first four H200 evaluations took 541/548/575/594 seconds from submission to PASS (median 562 s). Manual check no more frequently than about 10 minutes; next-check schedule is R/manual_monitor_schedule.json. The independent 30-second queue dispatcher still fills slots promptly, and report collector v8 (R/collect_training_six_v8.pid) now surfaces blocked/stale-queue alerts. At this update 120/188 scored, four case1/4/5 eval tasks active, RGB deferred.

- 2026-09-23 03:40 UTC: User-requested visual demo (not added to FVD report) sampled on idle dev1 8×H200 from exact 6000-step EMA checkpoints for cases 1/4/5. Same held-out bank fvd_0000..0007, per-case seed, CFG1, 50 Euler/state, 16 observed+128 generated; three Native evaluations passed with 8 MP4 each. Assembled 8 rows × GT/1/4/5 columns into R/demo_ema6000_8cases_20260923/gt_1_4_5_ema6000_8cases.mp4 (144f@16fps, 1056×2096, SHA256 2c140f83775b843e52495321b24155ee05f01d571e58c094cd0824c60fd81447). Metadata/protocol and tile/frame QA passed; manifest in same directory. Dedicated dev1 preview port8885 + local SSH tunnel serve http://127.0.0.1:8885/gt_1_4_5_ema6000_8cases.mp4; preserve tunnel/server if user is viewing. Sampling scripts are versioned under J/demo_ema6000_*.py. FVD queue continued uninterrupted (135/188 at check, four active, RGB20 deferred).

- 2026-09-23 03:48 UTC: User allowed supplementary FVD on idle dev1. Migrated remote dispatcher from PID1042476 to J/remote_eval_queue_145_dev1_v2.py (PID1100586 at launch) without interrupting its four cluster tasks. Started J/dev1_extra_eval_145_first_v1.py on dev1 (PID1100627 at launch), using all 8 idle H200s for one additional FVD checkpoint. Both dispatchers serialize target claims through R/eval_dispatch_assignment.lock; remote v2 excludes the live local key from submissions, preserves max four remote jobs, and defers RGB until case1/4/5 complete. Local selects high steps to avoid remote low-step order; initial dev1 task is case4 EMA10000 and Native fvd_single.py is Running. R/local_eval_status.json and R/remote_eval_queue_status.json are live truth; errors R/dev1_extra_eval_errors.jsonl and R/remote_eval_queue_errors.jsonl. Collector v9 (PID1100962 at launch) and live HTML now display five active H200 evaluations/40 GPU maximum; at activation 137/188 scores, remote4+dev1 1, pending46, no errors. Old R/DRAIN_LOCAL_EVAL remains for retired local v2; new supplementary queue uses R/DRAIN_DEV1_EXTRA_EVAL. Do not restart old local v2 or remote v1.

- 2026-09-23 04:10 UTC: User-requested second fixed 8-case visual demo GT/1/2/3 at 12k EMA is complete on dev1 8×H200. Source checkpoints are cases1/2/3 12000.dcp after EMA-origin tuning; same fvd_0000..0007 bank, per-case seeds35972922..29, CFG1, 50 Euler/state, 16 observed+128 generated. Video R/demo_ema12000_8cases_20260923/gt_1_2_3_ema12000_8cases.mp4, 144f@16fps, 1056×2096, SHA256 73704ee6758c874d8f10181a5ddd9d0a4da11381193a0d490ead3cc8bd0d12ce. Manifest/sidecars, all 24 source videos and tile QA passed. Preview on dedicated dev1:8886/local tunnel at http://127.0.0.1:8886/gt_1_2_3_ema12000_8cases.mp4; do not add to FVD report. Dev1 supplementary FVD was gracefully drained after finishing case4 online10000, then resumed after montage with PID1156246 at launch; remote FVD queue continued. First 12k demo launch failed before sampling because a mechanical script substitution produced nonexistent 012000.dcp; no usable artifacts existed, failed startup outputs were removed, v2 scripts corrected to 12000.dcp.


## 2026-09-23 08:43 UTC — train-set visual comparisons

- The configured training manifest for these Future-K2 experiments is `/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/data/train.jsonl` (139,670 clips from `miradata-clean` and `sekai_walking-clean`). Original RGB clips are under `/efs/stormyxie/datasets/SolarWM-Data/annotation-extracted-parallel/SolarWM-Data-Annotation/<dataset>/clips/`.
- Fixed eight training cases were selected with seed 20260923 (four per dataset), distinct source groups and zero source-group overlap with the 2,048-case held-out bank. `R/demo_train8_20260923/selection.json`, `train_8.jsonl` and `bank/train_*.npy` preserve the exact sources and 256×256 RepWorld decoder pixels. Both montages use identical `train_0000` through `train_0007` rows, 16 observed + 128 generated frames, CFG1, 50 Euler steps, EMA weights and per-row matched seeds.
- 12k EMA GT/1/2/3: `R/demo_train8_20260923/train_gt_1_2_3_ema12000_8cases.mp4`; manifest `R/demo_train8_20260923/montage_manifest.json`; preview `http://127.0.0.1:8887/train_gt_1_2_3_ema12000_8cases.mp4`. 6k EMA GT/1/4/5: `R/demo_train8_20260923/ema6000/train_gt_1_4_5_ema6000_8cases.mp4`; manifest in same directory; preview `http://127.0.0.1:8888/train_gt_1_4_5_ema6000_8cases.mp4`. Both are 8×4, 144 frames at 16 fps; SHA256 agrees with manifests, and byte-range HTTP preview returned 206.
- Sampling ran on dev1 8×H200 after gracefully draining the supplementary local FVD dispatcher. Remote queue reported 188/188 FVD complete, active0, pending0, blocked0 at this update. `R/DRAIN_DEV1_EXTRA_EVAL` remains set; no reason to restart an evaluation dispatcher until a new evaluation scope is requested.

## 2026-09-23 14:52 UTC — SolarWM 全来源连续160帧 VideoRAE3D latent 已完成

- 新缓存：/efs/stormyxie/cache/rep_world/videorae3d_consecutive160_20260923/；每个来源各有独立子目录、contract.json、source_manifest.jsonl、shards/、index.jsonl 和 COMPLETE.json。目录说明及随机读取示例见缓存根目录 README.md、reader.py。
- 范围为 SolarWM-Data 当前已有、元数据帧数 >=160 的所有视频，不按 kept/high/xhigh 或原 train/test 身份排除。每条原视频从第0帧起按 [0,160)、[160,320)... 连续、不重叠取窗；不足160帧的尾段丢弃。每窗输出 frozen VideoRAE3D packed FP32 latent [256,10,16,16]，BF16 codec/autocast，256条/分片；原视频路径、source_sample_id、窗口序号及帧范围均可在 manifest/index 中追溯。
- 已完成 822391 个窗口：abot 185255、mind 3101、miradata-clean 135224、omniworld 3230、realcam_vid 14838、sekai_game 8836、sekai_walking-clean 109248、sekai_walking 137940、spatialvid-clean 224719。multicamvideo 当前仅有81帧素材，无可用160帧窗口，记录为 NO_ELIGIBLE.json。先前的2048条 held-out/test 视频均已纳入。
- 依据用户最新要求，已完成的 sekai_game、sekai_walking 缓存原样保留；spatialvid-clean 保留原有202752条/792个已提交分片，只用 H200 补齐21967条/86个缺失分片。其它已完成数据集未重做。续跑使用视频 batch128、codec microbatch192、每GPU 16个解码 worker；新分片元数据记录 NVIDIA H200。
- 2026-09-23 14:52 UTC 核查：/efs/stormyxie/reports/videorae3d_all_existing_ge160_20260923/FULL_VERIFIED_PASS.json 为 PASS，逐数据集检查 COMPLETE/manifest/index、记录顺序/帧范围与抽样 latent SHA，覆盖2048条旧 test；H200 补片任务 jobs/h200_spatial_resume/PASS.json 为 PASS；reader.py 对 spatialvid-clean 第0条实际回读 hash 匹配。生成器、任务日志与清单在同一报告目录。
- 这是一套新的 solarwm.videorae3d.consecutive160.packed_fp32.shards.v1 数据合同，不是旧的139670条训练缓存 /efs/stormyxie/cache/rep_world/videorae3d_train_20260922T082647Z_b128。当前 Native CachedVideoLatents 要求旧 rep_world.videorae3d.packed_fp32.shards.v1 schema、单一 cache 与精确 data.train_manifest 匹配；不得直接把新目录替换旧训练配置并宣称已支持多来源训练。本次只发布/验证数据，没有切换现有训练或修改旧缓存；后续如何选取来源、窗口与 train/test 划分由使用方决定。


## 2026-09-23 — 355M full-SolarWM mask-ratio ablation: review configs only

- User explicitly requested design/configuration before any launch. Six masks 0/30/60/80/85/90% now have 10k pretrain and 2k mask0 full-resume YAMLs (12 files) under `/threed-code/stormyxie/vforge_jobs/mask_ratio_fullsolarwm_355m_20260923/configs/`. `review_plan.json`, `README.md`, and `static_validation.json` are in the same job directory. Static CPU check PASS; no train task, checkpoint or output directory has been created.
- Reference is 354,975,424-parameter `355m_global_ca_rope`: one 19-layer causal ViT, 10-layer DiT, only first-block all-patch CA before SA with 3D RoPE, latent ViT input, no zprev, no distant/mosaic future target. FA4 strict, FSDP2, local batch8 on proposed 8×H200, EMA0.99. Pretrain full DCP every2500; tune continues from each 10000.dcp with `resume.weights/optimizer/training_state/strict=true`, retaining online+EMA/optimizer/scheduler/RNG/data cursor, mask0 and max_steps12000, full DCP every500. No EMA-origin copy or optimizer reset.
- All nine completed consecutive160 VideoRAE cache owners are included, totaling 822,391 windows; multicamvideo has no eligible 160f window. Immutable merged manifest: `/efs/stormyxie/reports/mask_ratio_fullsolarwm_355m_20260923/data/train_windows_822391.jsonl`, SHA256 `afa56257e8b401bca28104115f5e7e000fcc1892aa773f32073df8244e8346eb`; provenance in `data/manifest_contract.json`. The prior held-out2048 is in the training pool; old FVD cannot be used as an independent test set. With global batch64, 10k covers at most640k draws (77.8% of one pass), 12k 768k (93.4%).
- **Do not submit these YAMLs yet.** Current Native `CachedVideoLatents` only reads the old single-directory shard schema; the new per-owner consecutive160 cache needs a task-owned Native reader before the configs are runnable. User requested review first. After approval, implement adapter and validate random read + strict full-state boundary resume, then wait for a separate launch instruction.

## 2026-09-23 — 355M mask-ratio ablation: clip holdout and multi-source Native cache reader

- User approved a clip-level, rather than source-video-level, holdout and still requested review-only configs. Fixed seed 20260923 uniformly sampled 2,048 of 822,391 committed consecutive160 windows for test; the other 820,343 are training windows. Train/test sample IDs are disjoint, with no duplicates in train; 811 source videos occur on both sides by design. Nine cache owners are represented in both splits. Test includes 667 windows with a nonzero start frame; all 2,048 RGB paths exist. `multicamvideo` remains absent because it has no 160-frame windows.
- Split artifacts live at `/efs/stormyxie/reports/mask_ratio_fullsolarwm_355m_20260923/data/`: `train_windows_820343.jsonl` (SHA256 `074811e8c3c107bf6446f592efcce09b25ee712632ed650072f32a81108082e4`), `test_windows_2048.jsonl`, and `split_contract.json`. The original 822,391-window manifest remains as provenance and is not used for training. The new train rows carry cache owner, per-owner cache index, explicit frame window, and latent checksum.
- Native `vforge/tasks/rep_world/latent_cache.py` now has `CachedConsecutiveVideoLatents` for the per-owner `solarwm.videorae3d.consecutive160.packed_fp32.shards.v1` cache; `train.py` selects it when the cache root has no legacy single-directory contract. It validates split/manifests, owner fingerprints and complete indexes, codec assets/geometry, shard metadata and selected row identities, and each returned latent checksum. `data.py` honors explicit fixed-window starts for raw RGB evaluation. Legacy cache reading remains on the old path.
- Review-only job directory `/threed-code/stormyxie/vforge_jobs/mask_ratio_fullsolarwm_355m_20260923/` now has regenerated 12 YAMLs with the 820,343 train and 2,048 validation manifests, revised `review_plan.json`, `README.md`, `static_validation.json`, `make_clip_split.py`, `prepare_configs.py`, and `verify_configs.py`. Six mask cases and 10k+2k full-state tuning design are otherwise unchanged. `prepare_configs.py` no longer has the old manifest writer that could overwrite the split.
- CPU validation: 27 targeted pytest cases pass (24 legacy, 3 new), Ruff check/format pass, all 12 configs static-validate. Real cache adapter initialization succeeded across all nine owners (820,343 rows) and returned checksum-valid `[256,10,16,16]` latent rows from several owners; test RGB with `start_frame=800` decoded `[3,160,256,256]`. No six-case training task has been submitted; no H200 training/full-DCP resume smoke has been run. Approval/launch is still separate from this review-only preparation.

## 2026-09-23 — six mask-ratio H200 tasks launched with automatic full-state tuning

- User authorized launching all six 355M masks (0/30/60/80/85/90%) on 8×H200 each, and required tuning to start automatically after pretraining. Study paths remain `J=/threed-code/stormyxie/vforge_jobs/mask_ratio_fullsolarwm_355m_20260923`, `R=/efs/stormyxie/reports/mask_ratio_fullsolarwm_355m_20260923`, `O=/efs/stormyxie/outputs/vforge/mask_ratio_fullsolarwm_355m_20260923`. Frozen Native runtime is `R/runtime`; SHA audit in `MASK_ABLATION_SOURCE_HASHES.json`.
- A real 8×H200 FA4 gate in `R/validation/gpu_gate_PASS.json` trained mask80 to step1, saved full DCP, then resumed with mask0 to step2; model/optimizer/trainer components committed and EMA updates advanced 1→2. Initial strict tuning resume failed because `semantic_config` includes `context_mask_ratio`; tuned M30–M90 YAMLs now use `resume.strict=false` but retain `resume.weights/optimizer/training_state=true`. `J/node_runner.py` audits checkpoint fingerprint against the pretrain YAML and allows exactly that mask ratio change before tuning; M00 remains strict. No optimizer, EMA, scheduler, RNG or loader state is deliberately reset.
- `J/node_runner.py` launches two sequential Native 8-rank processes in one scheduled task: 10k pretrain, verify committed `10000.dcp` full components, audit stage boundary, then 2k mask0 tuning to `12000.dcp`. On a restarted task it chooses the latest committed full DCP and continues the interrupted phase. Pretrain saves every2500; tuning every500. `J/controller.py` on dev1 polls every120 seconds, submits pending tasks and retries explicit preemptions. Live receipts/status/logs are in `R/receipts/`, `R/dispatch_status.json`, `R/controller.log`, and `R/tasks/<task>/`.
- Initial submissions accepted: M00 `stormyxie-mask-00-0923154404-64fa4`, M30 `stormyxie-mask-30-0923154408-5c690`, M60 `stormyxie-mask-60-0923154412-eb935`, M80 `stormyxie-mask-80-0923154416-e590a` on `tenc-awsusw2`; M85 `stormyxie-mask-85-0923154420-47a38`, M90 `stormyxie-mask-90-0923154424-5253d` on `tenc-aws-yfxn-us-west`. All request `gpuType=H200`, 1 host×8 GPUs, FA4 strict, FSDP2. Four awsusw2 tasks were Running and writing pretrain progress at first check; two yfxn tasks were Updating, so inspect fresh status rather than assuming they have started. Dev1 controller Python PID was 1395269 at launch; verify current PID from `R/controller.pid` before acting.
- Old review-only handoff notes are superseded for launch status and the Native adapter. `J/review_plan.json` now marks `launched_h200`; regenerated `J/static_validation.json` still passes all 12 configs. No FVD jobs were submitted for this study.

## 2026-09-24 12:59 UTC — 0.5:3:1 split-K2 scaling submission

- User selected ViT-E:ViT-C:DiT=0.5:3:1 for the prepared 125M/355M/775M architecture comparison. The 355M checkpoint is reused from `future_mask90_355m_20260924/04_split_0p5_3_1`; no retraining of that case. New configs and launch artifacts are under `/threed-code/stormyxie/vforge_jobs/split_k2_scaling_20260924/`; report/status under `/efs/stormyxie/reports/split_k2_scaling_20260924/`. Exact parameter counts: 125M=124,999,446 and 775M=774,617,313. Local batch remains 8 on 8 H200, FA4 strict/FSDP2, mask90/K2 and offline latent cache.
- 125M task `stormyxie-sk2-12-0924125249-c8cf6` and 775M task `stormyxie-sk2-77-0924125237-d7645` were accepted on H200-2D. At last check, 125M was Running with an 8-GPU Pod and pretrain launcher active; 775M was still Updating with no Pod because 2D quota showed 0 free GPUs. Do not call the latter training until Pod/logs confirm it. `controller.py` is active to reconcile/retry preemption.
- 125M: 10k pretrain followed by 2k mask0/next-state full-state tuning in the same task. 775M: mask90/K2 pretrain max100k, full DCP every2500; `tune_watch.py` automatically forks 10k full checkpoint to separate 2k mask0/next-state tuning output without stopping pretrain. Dedicated FVD queue for 10k/10.5k/11k/11.5k/12k is armed; 775M EMA10k has first priority, but local dev1 evaluation is paused while another agent owns TF/DF baseline evaluation. Confirm smoke and task status before interpreting any future score.
- User explicitly assigned TF/DF EMA FVD evaluation entry/progress to another agent. This agent stopped its TF/DF FVD queue controller and set `fvd/STOP_QUEUE` under that report; one already-running DF125 EMA10k evaluation was allowed to finish. Do not resume that controller automatically or issue TF/DF evaluation conclusions here.

## 2026-09-24 — 355M RGB split ViT，四帧一个时间组

- 用户要求参考 DuET RGB 路径，并把本模型 RGB ViT 输入改成每四帧一个 temporal patch。研究配置在 `/threed-code/stormyxie/vforge_jobs/split_k2_rgb_355m_20260924/`；训练/状态在 `/efs/stormyxie/reports/split_k2_rgb_355m_20260924/`，任务 `stormyxie-rgb4-35-0924134347-3daff` 提交到 H200-2D（1×8 H200），13:45 UTC 首次核查为 Running。后续须看 `dispatch_status.json` 和当前日志，不把历史状态视为实时状态。
- 输入同窗 160 帧 RGB（10 state×16 帧），每 state 用 Conv3d patch `[4,16,16]` 变成 4×16×16=1024 token；DiT 目标仍用离线 VideoRAE latent。E3/C18/DiT7，参数 354,998,832；相对于 latent 355M 只变输入模态及 patch 投影相关参数，局部 batch 保持8。mask90 在完整 E 后、C 前；K2；10k pretrain + 全状态 2k mask0/next tune；EMA0.99、FA4 strict、FSDP2。
- 修复了 masked RGB 路径不传 visible token 位置给首层 3D RoPE CA 的缺口，改动 `/efs/stormyxie/vforge/vforge/models/rep_world/model.py` 和配置 guard；独立冻结运行快照含这两项修复，source hash 已封存。CPU：split conditioner 32 passed（13 GPU deselected），缓存/训练任务 26 passed；真实 cache 对齐读取 1 条得到 RGB `[3,160,256,256]`、latent `[256,10,16,16]`。H200-2D 单节点 8×H200 已跑通首批训练步：step7 loss=1.9716，EMA updates=8，显存峰值约12.05 GiB；完整 DCP 续训仍待 10k 边界。
- 本次不处理 TF/DF EMA FVD，因用户已交由另一 agent。RGB 的 FVD 暂未提交。

## 2026-09-26T03:27:36.700943+00:00 — 355M 0.5:3:1 EMA LPIPS 10k–12k launched

User explicitly authorized development-node plus H200-2B parallel evaluation.
Campaign: /efs/stormyxie/reports/lpips_split355_0p5_3_1_20260926.
Only 04_split_0p5_3_1 EMA checkpoints 10000/10500/11000/11500/12000.
Metric protocol inherited from lpips_tfdf_splitk2_20260925: fixed 2048 clips,
CFG1, 50 Euler steps/state, 16 observed + 128 generated RGB frames, VGG LPIPS
FP32/TF32 off, pinned weights. Frozen future_mask90 runtime and eval overlay
04_split_0p5_3_1_tune.yaml retained; no source/model/training modifications.
Controller queue.pid runs one 8-H200 dev1 evaluator plus at most four
8-H200 tasks on tenc-awsusw2 (2B); 180s polling, per-checkpoint leases and
strict EMA/provenance audit. Completed remote tasks automatically deleted;
aggregate.py joins existing FVD with newly completed LPIPS into campaign CSV.
Shared fvd_local_guard registers this campaign to protect local allocation.
Koala API DNS timed out on dev1; local DNS resolved 43.134.151.78, and verified
TLS curl succeeded. Added only that API hostname to /etc/hosts with marker
codex-lpips-c3-20260926-dns-workaround; controller removes its marked line
after all five evaluations complete and remote tasks are released.
Results pending; never substitute 03_split_0p5_2_1 for this requested model.

## 2026-09-26T03:46:36.250726+00:00 — compute-matched TF/DF vs split-K2 EMA comparison

User selected full 10k pretrain +2k tuning compute budget for all three Ours sizes; independent best tuning EMA FVD/LPIPS.
All Ours use 0.5:3:1; 355M is 04_split_0p5_3_1, never substitute 03 (0.5:2:1).
New 355M LPIPS campaign has committed all five strict full-set scores; 10k..12k means:
0.5209747115390257 / 0.5156996560767695 / 0.511970293289167 / 0.5108417912660954 / 0.5108840415321438.
Best 355M LPIPS and FVD both11500. Controller retains automatic task cleanup.
Compute audit uses meta instantiation of frozen runtime models to count every Linear/Conv weight and actual vector shapes, plus semantic attention QK/AV pairs.
Forward+backward; multiply-add=2; no activation recompute; excludes codec/optimizer/EMA/communication/elementwise/tile-padding. Not a GPU hardware-counter measurement.
Global64 independent clips. TF16 interleaved states/80 attention state pairs; DF9 states/45; Ours8 origin states, E1024tokens, C100 pretrain/1024tune, C36 state pairs.
Budgets EFLOP125=3.319908179902464,355=8.378085934104575,775=17.006312760016897.
Nearest available jointly FVD/LPIPS-scored baseline checkpoints: ~1x TF2500/DF5000; ~2x TF5000/DF7500 for all three sizes.
Always expose actual budget multiple; maximum deviation from target29.4%, no metric interpolation or exact equal-compute claim.
Evidence/full-precision CSV/matrix-shape ledgers/report: /efs/stormyxie/reports/compute_matched_tf_df_split_k2_20260926.
Mac artifact: /Users/stormy/Documents/Codex/2026-09-26/compute-comparison/compute_matched_report.md.

## 2026-09-26 — 约 2× 预算 EMA PSNR / SSIM 补测

用户要求 TF 三尺寸 EMA5000、DF 三尺寸 EMA7500；Ours125/355/775M 各 EMA10000/10500/11000/11500/12000，独立取最高 PSNR 和最高 SSIM。355M 必须是 04_split_0p5_3_1。共21个 checkpoint。
报告/冻结输入/脚本/队列：/efs/stormyxie/reports/psnr_ssim_compute_matched_20260926。复用已完成 LPIPS 的 lossless uint8 RGB，不重新采样或修改旧 runtime/config/output；2048固定clips、CFG1、50Euler/state、16观察＋128生成，仅评分[16:144]。PSNR 逐RGB帧均值（另列 pooled-MSE），SSIM 11×11 Gaussian sigma1.5 population covariance、valid边界、RGB均值。严格校验源score/eval manifest/EMA身份/每clip seed/GT与生成RGB SHA256。
验证 metrics_PASS.json：CUDA SSIM 对 scikit-image 最大误差2.3387e-6，PSNR与参考一致；chunk1/32相等。8clips短测1.27秒。controller每180秒轮询，最多dev1一台＋2台H200-2B，checkpoint独立EFS flock，自动释放完成远端任务；shared fvd_local_guard登记本队列。状态queue_status.json，逐checkpoint results/<key>/score.json，all_checkpoint_scores.csv、best_scores.csv、report.md自动聚合。不要把未完成部分称为全量最佳；最终最高分需要五个tuning点全部完成。

## 2026-09-26 — 21个 EMA PSNR/SSIM 全量完成

全量审计 validation/full_set_PASS.json=PASS，21 checkpoint ×2048 clips ×128生成帧。dev1＋1台H200-2B完成；第二台DNS提交失败，核实NotFound后撤销意图，没有占用第二台资源。远端任务已自动删除，local claim已释放，临时API DNS marker已移除。
R=/efs/stormyxie/reports/psnr_ssim_compute_matched_20260926；全部21点 all_checkpoint_scores.csv，最佳 best_scores.csv，完整预算/FVD/LPIPS/PSNR/SSIM comparison_all_metrics.csv/.md。
125M TF: PSNR=14.887679@5000, SSIM=0.34295034@5000.
125M DF: PSNR=14.674513@7500, SSIM=0.33385241@7500.
125M Ours: PSNR=14.980188@11500, SSIM=0.33799276@11500.
355M TF: PSNR=14.932375@5000, SSIM=0.34817616@5000.
355M DF: PSNR=14.743857@7500, SSIM=0.33799358@7500.
355M Ours: PSNR=15.042903@11500, SSIM=0.34462538@12000.
775M TF: PSNR=14.947636@5000, SSIM=0.35071071@5000.
775M DF: PSNR=14.856200@7500, SSIM=0.34428797@7500.
775M Ours: PSNR=14.988033@12000, SSIM=0.34608610@12000.
逐帧RGB PSNR先取dB再求均值；pooled-MSE PSNR只作secondary字段。SSIM采用11×11高斯sigma1.5/population covariance/valid边界。Ours四指标独立选最佳，355M PSNR最佳11500、SSIM最佳12000，不要合称一个checkpoint的联合最优。

## 2026-09-26 — NPY 转 MP4 工具

用户要求将保存的生成视频转成MP4，工具 /efs/stormyxie/tools/npy_to_mp4.py，说明 /efs/stormyxie/tools/npy_to_mp4_README_CN.md。复用现有vforge venv的NumPy/PyAV，不需安装FFmpeg CLI。支持单文件/批量/递归、并行workers、skip/overwrite、frame range；默认16FPS、H264/yuv420p CRF18 fast，保留原始NPY。临时MP4完成并核验帧数后atomic发布；附编码元数据。用户本轮只要求写脚本，未批量转换2048条。真实775M EMA12k一个样本已转并完整解码检查144帧256×256/16FPS，输出在 /efs/stormyxie/reports/psnr_ssim_compute_matched_20260926/mp4_preview_775m_12000/。

## 2026-09-26T10:44:16.565868+00:00 — 775M EMA11500 历史顺序单步消融完成

用户要求少量case，固定held-out2048清单等距选8clips。dev1 8×H200 Native FSDP2/FA4 strict、BF16；775M split E3/C18/DiT7、0.5:3:1、首层全patch CA→SA→FFN/3D RoPE；11500.dcp严格加载EMA0.99、updates11500。输入GT1–8，整state随机非identity排列与正序两臂，输入槽位重新赋时序；相同逐case噪声、CFG1、50 Euler，只生成一个latent，GT9仅评分。复用离线VideoRAE3D cache及Native sampler，不在线codec。8/8打乱MSE更高；normalized latent MSE正序均值0.4054664349，打乱1.2995901508，均值比3.20517（+220.517%）。本次8排列全部改变最后state，不能把退化单独归于时间顺序。报告/固定输入/种子/排列/source hash/Native provenance：/efs/stormyxie/reports/history_shuffle_775m_ema11500_20260926。原训练/runtime/config未修改，采样exit0耗时29.54s，local GPU claim已释放。Mac报告 /Users/stormy/Documents/Codex/2026-09-26/history-shuffle-775m/report.md。

## 2026-09-26T10:49:14.895028+00:00 — 775M EMA11500 训练集历史顺序单步消融完成

用户要求补测training。从模型配置的 train_windows_820343.jsonl 等距选8clips，Native CachedConsecutiveVideoLatents核验源manifest/cache/assets/checksum。与test协议一致：8GT history、单步GT9、同噪声、CFG1/50Euler/mask0、严格EMA0.99 updates11500、冻结runtime、dev1 8×H200 FSDP2/FA4；未提交集群。正序MSE均值0.5781089828，打乱1.1701639509，比2.02412（+102.412%），8/8退化。Case6排列保留GT8在最后，MSE0.3027856694→0.3282897885（+8.423%）；只是一个clip。7/8全打乱改变最后state；训练集成员身份并不保证optimizer实际已消费每个clip。与test为不同clips，不据均值推断泛化差距。报告/输入/排列/种子/Native provenance/artifact hashes：/efs/stormyxie/reports/history_shuffle_775m_ema11500_train_20260926。exit0、Native已提交8结果，采样29.38s，local claim释放。Mac报告 /Users/stormy/Documents/Codex/2026-09-26/history-shuffle-775m-train/report.md。

## 2026-09-26T10:58:33.898096+00:00 — mask ratio 355M 预训练 FLOPs（无重算）

用户只要pretrain、不看tuning。准确实验 mask_ratio_fullsolarwm_355m_20260923，单体causal ViT19/1024＋DiT10/768，参数354975424，10states→9next targets；每state1024tokens，CA首层before-self/global/3D RoPE，local8/global64。CPU/meta审计冻结runtime＋runtime_mask100，所有Linear/Conv真实权重/调用向量/input-gradient需求，19 ViT/1CA/10SA QKV shapes与45causal blocks，loss meta forward/backward PASS，checkpointing关闭以排除重算。仅统计矩阵主项，乘加=2，forward＋backward；不含codec/optimizer/EMA/通信/elementwise/FA tile padding或内部额外重算，不是GPU硬件测量。全局step TFLOPs：mask0=2018.114166、30=1399.511625、60=908.482069、80=653.485469、85=597.437316、90=545.025499；对应可见tokens1024/716/408/204/152/100。90mask比0节省72.9933%。100扩展完全短路ViT/CA，0tokens，450.124686TF/step。配置模型字段除mask均一致。报告CSV/逐模块shape账本/sourcehash验证在 /efs/stormyxie/reports/mask_ratio_flops_20260926；Mac副本 /Users/stormy/Documents/Codex/2026-09-26/mask-ratio-flops/。未占GPU/提交集群/修改训练runtime和配置。

## 2026-09-26T11:42:49.395784+00:00 — mask 消融预训练速度/FLOPs关系

用户要求看上述mask cases的训练速度与无重算FLOPs关系；只预训练。统一train_steps.jsonl step index500≤step<9900，排除checkpoint保存step2499/4999/7499/9999及随后50步，7组各9247稳态steps，不做其它latency outlier过滤，steps/s=N/sum(time)。预训练10000步唯一连续，7组node与effective config核验8×H200、FA4 strict、FSDP2、BF16、local8/global64。mask0/30/60/80/85/90/100 steps/s=0.571249/0.648011/0.798288/0.893374/0.905358/0.902744/1.175389；mean seconds=1.750549/1.543184/1.252681/1.119352/1.104535/1.107734/0.850782。原6组t=0.4655708*F(PFLOPs)+0.8379642，R²0.987639，RMSE0.027652s；过原点拟合R²-1.43077，表明带截距近似线性但不呈严格比例。0→90理论3.70279×算量减少，实测1.58030×加速；80..90平台。100组bypass ViT/CA单列，不参与fit。90maskoptimizer阶段0.290158s占26.19%，dataloader约0.298ms；communication/kernel具体占比未profile。注意实测速度含训练activation重算和通信，横轴依用户要求不含重算，不是控制变量同节点benchmark；拟合截距是描述性参数。图PNG/SVG、分阶段图、逐步selected_steps.csv、源logs/task/config/hash、窗口/保存margin敏感性在 /efs/stormyxie/reports/mask_ratio_speed_flops_20260926，Mac /Users/stormy/Documents/Codex/2026-09-26/mask-ratio-speed-flops/。只读历史日志，无GPU资源占用或新任务。
