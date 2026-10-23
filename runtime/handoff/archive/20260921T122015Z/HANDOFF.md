# vforge / RepWorld 当前交接

更新时间：2026-09-21 12:09:44 UTC。工作在dev1；先读[AGENTS.md](AGENTS.md)，核对实际task、日志、完整checkpoint。
本次用户要求删除CA的参数补回DiT FFN，**四档全部从头重启**。旧未补参数三组已停，不得恢复。

## 当前任务与约束

- 原始10组scaling的8个小模型已完成10k/160次FVD，两组20B继续，不受这次重启影响。
- 新四档125M/355M/775M/1.56B，仅block0做CA→SA→FFN，其余SA→FFN。
  加宽每层DiT SwiGLU FFN补回CA参数，hidden取8倍数，总参数对原全层CA误差小于0.02%。
  ViT、DiT主宽度/深度/heads不变；这是等参数结构比较，FFN与计算量随之改变。
- fresh step0，无旧权重warm-start；新输出目录。EMA0.99包含ViT+DiT，FP32shadows，
  初始参数初始化，第一次optimizer update开始维护；shadow不计可训练参数。
- 每组8×H200 / NativeFSDP2 / FA4 / local8 global64 / mask80% / latent input / no-zprev。
  不用B300，不允许trainableViT/DiT SDPAfallback；frozenVideoRAE codec仍为原SDPA路径。
- max10k，每500step fullDCP；独立节点评估online与EMA，CFG1/2048held-out cases。
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

## 本轮证据与监控

RM/validation：
- parameter_matching.json：实际meta模型参数、CA块[0]、FFN相邻8倍数候选、逐字段configdiff。
- preflight.json：4train/8eval，before_self，fresh无ckpt，online/EMA仅权重选择不同；
  runtime源码hash和基础FVD协议不变。
- superseded_tasks_stopped.json：旧3训练+1剩余远端eval均删除后NotFound；
  旧monitor/extras/local按完整cmdline核对后停止，活动local采样随退出清理。
- launch_status.json：4个KAP和实际Pod启动证据。training_started_verified.json记录实际训练核验，
  已经通过四组真实训练核验，具体步数与时间见该文件。
- preview_verified.json：当前网页/PNG和旧版归档HTTP200。

RM/services.json为本轮CPU服务：
- monitor.py PID 330373
- fvd_extras.py PID 339124
- fvd_local.py PID 330375
发送signal前必须重新核验/proc/cmdline。monitor每60秒扫描完整DCP→双权重队列；
extras每30秒协调KAP/Pod并清理，cap7节点（H200空余配额核查后由3提高，证据RM/validation/fvd_parallel_cap7.json）；local仅dev1空卡时采样，cap不是实际资源。
当前没有周期训练自动重启器；submit_first_cross.py只提交无receipt项，不可盲目重复。
train_entry.py仅在本轮自己目录有完整DCP时auto fullresume。

[当前等参数报告](http://127.0.0.1:8884/first-cross/)；
[旧未补参数已停归档](http://127.0.0.1:8884/first-cross-unmatched/)；
[原始scaling](http://127.0.0.1:8884/)；
[历史七组condition](http://127.0.0.1:8884/condition-ablation/)。
8884由原始J/serve.py服务，路由已切RM，新serverPID330812；隧道保留。
保留8882/8883；大型媒体在EFS，不因网页故障重启训练。

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

## 效率口径、边界与下一步

- 四组201–300固定窗口已自动生成，并独立复核12行/1200条原始事件、hash与EMA更新数。
  RM/validation/training_efficiency_201_300_verified.json；新版首层CA典型/有效steps/s依次
  0.540/0.321、0.499/0.325、0.391/0.345、0.270/0.268。新版不得复用未补参数速度。
  典型=1/median(step_time_s)，有效=N/sum(step_time_e2e_s)，无ckpt/demo，不删慢步。
  显存为rank0逐步allocated/reserved峰值。不同节点，EMA维护开销单独注明。
- 125M主报告0.600/0.593取9970–9999最近30步；旧first-cross的0.528/.341是201–300典型/有效。
  证据R/validation/conditioning_speed_windows_20260921.json和conditioning_speed_main_window_20260921.json；
  网页#speed-window-audit保留。不能从不同窗口forward波动判定CA结构更快。
- CA顺序163CPU+2maskedGPU回归；EMA原53CPU及最终13定向测试，56filelint、264mcorehash；
  RE旧/validation保存真实8H200 train2→resume3→双权重16case及FP32EMA递推核验。
  原3组500DCP核验在RE旧/validation/production_step500_ema_audit.json，不证明新FFN checkpoint已通过。
- FA4/SDPA短期loss接近不等于独立优化轨迹等价；Muon会放大小梯度差异。
  默认SDPA，当前实验显式FA4。并行单H200/双机2×8B300的DDP/FSDP/HSDP/Ulysses有实证，
  DDP+USP拒绝，CP/Ring/TP/EP/PP未验证；不是全仓库tests全绿声明。
- JM/audit_checkpoint_ema.py --step 500 --watch已在dev1以CPU完成并正常退出（原PID334170）；
  RM/validation/checkpoint_auditor_reference_test.json验证旧真实DCP读取路径；现新四组500DCP全部通过。
  输出production_step500_ema_audit.json，检查Native完整提交、实际FFN参数形状、FP32shadow及EMA标量，
  不把metadata验证称为全权重数值扫描。
- 四组从step0连续有限训练/EMAcount及四个500 fullDCP已核验；继续后续checkpoint的
  online/EMA独立采样与正式score；首批500的8份全case核验已完成，继续后续档；持续原20B至10k。
  仅清理明确自有失败/完成资源。
- 用户取消的未补参数三组、旧SA→CA版、旧无EMA版及更早condition训练不得恢复。
  原始8小模型已完成，不重复算；R/validation/atomic_artifact_recovery/recovery_audit_complete.json
  记录EFS原子写修复，不覆盖1.56Bentry9000/9500已审计结果。

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

## 首个生产checkpoint与采样

RM/validation/production_step500_ema_audit.json已complete：四档FP32 EMA、更新数500、
在线/影子张量形状与参数数量、CA仅block0、hidden1776/3008/4512/4848、最大8分片均核对。
初始CPU检查器正常退出，不要把旧PID不在误判故障。
775M首批16个共同case在online/EMA两路使用相同checkpoint、GT、seed和协议，
EMAcount500、16个生成hash全部不同、两路特征均finite；这是运行路径抽检，不是正式2048 FVD。
证据RM/validation/first775_online_ema_cases_verified.json。
355M online首次申请因真实Pod Unschedulable（GPU/CPU/内存/EFA不足）被控制器撤回，
删除后NotFound/无Pod，目标仍在队列；不是模型代码失败。配额空余不等于存在可调度8卡节点。

## 本轮正式分数核验

2026-09-21 11:22 UTC 核查：

原始campaign为171/200份正式FVD：8个小模型160份已完成，两组20B合计11份。
当前11份20B batch-v2均已独立协方差复算、proof hash匹配现有score文件；
见RM/validation/current_20b_scores_verified.json。两组20B继续训练，未达到10k，不得宣告goal完成。

当前等参数四档已产出至少36/160份正式FVD：500、1000、1500步各8份，2000步前三档各两份，2500步前三档各两份。
500步与1000步的**两批各8/8份全部复核通过**。
RM/validation/ema_scores_verified_00500.json：8×2048个case的身份、seed、GT、detector、
checkpoint manifest、online/EMA权重类型、有限FP32特征均核对，FVD复算最大误差2.274e-12。
四对online/EMA均同checkpoint/case/seed/GT，2048个生成RGB hash全部不同。
检查器required_scores原遗留6已修为8；该修复仅外部检查helper，不改变训练和推理。
第二轮检查器PID349177已成功退出，过程与日志见ema_scores_verify_second_process.json及同名.log。

| 模型 | 500 online | 500 EMA | 1000 online | 1000 EMA |
| --- | ---: | ---: | ---: | ---: |
| 125m | 5613.69113 | 7292.98111 | 1453.21155 | 1398.26249 |
| 355m | 1727.09394 | 3252.13353 | 1172.02935 | 1270.18134 |
| 775m | 1838.13083 | 1599.78384 | 1135.46455 | 856.93067 |
| 1560m | 1863.24303 | 1281.18468 | 849.59124 | 726.01268 |

1000步8份已独立完成全2048case复核和FVD复算，最大误差3.638e-12；
四对online/EMA同checkpoint、case、seed、GT，各2048个生成RGB hash不同。
RM/validation/ema_scores_verified_01000.json已complete=true，score文件hash再次核对未变；
ema_scores_verify_01000_process.json记录检查器PID360134正常完成/returncode0。
1500步现有6份正式输出尚未做这轮额外全case独立复核，勿将1000核验外推。
新报告图中的8个500点与score逐项一致，与原concat/all-CA共用同一reference bank且基础协议字节相同。
证据RM/validation/first500_baseline_comparison.json；first500_complete_verified.json保存首批完成核验。

500任务全部结束并及时释放。355M online@1000曾在szr因无可调度节点撤回，
自动改为AWS任务stormyxie-sc-xmatch-fvd-0921103743，已直接核对Task/Pod Running。
RM/validation/first500_completion_resources.json记录完成资源释放及上述重调度。

## 原始 scaling 评测改为按需节点（2026-09-21 10:35 UTC）

原常驻GPU评测 task stormyxie-sc-fvdbatch-0921074040 在确认所有已有checkpoint均出分、
active/pending均空后主动删除；任务NotFound且无Pod，释放8H200。不是模型失败，不要恢复常驻池。
J/monitor.py现按 fvd/pool_control.json 的 primary_scheduler=false 由dev1 CPU发布完整DCP队列；
J/fvd_extras.py对无primary的队列不再预留前两个checkpoint，仍用既有lease去重和独立8H200任务。
修改仅外部控制器，未改训练/采样runtime、配置或协议。2个定向检查及171份生产分数扫描通过。
新CPU服务在R/services.json：monitor PID348486、extras PID348487；发signal前仍须重新核对cmdline。
旧CPU PID249578/245631已经退出（可能保留Z状态的僵尸/proc项，不表示仍在运行）。
R/validation/retire_idle_primary_20260921T1031Z保存before/after、测试、删除和生产队列核验。
生产新队列OnDemand已完成真实新checkpoint验证：20B cross-attn@3000从未完成目录到
Native完整提交，再由CPU监控入队、独立任务stormyxie-sc-fvdx-0921105737启动，8H200。
直接KAP/Pod核对Running、同一目标仅一个活动receipt，评测物理节点与当前6个训练节点均不同。
证据R/validation/retire_idle_primary_20260921T1031Z/new20b_checkpoint_dispatch_verified.json；
dispatch_watcher.json显示CPU观察器正常完成。验证到worker启动，3k FVD本身仍在计算。
本次20B fullDCP总payload约160.10GB（weights约80.03GB、optimizer约80.07GB）。
实测前台4.44秒、后台总计393.82秒，upload185.96秒、commit203.41秒；
Native storage._commit_dcp_directory会先逐文件SHA256再写manifest/rename完整目录，
因此文件已写出/.metadata存在仍不等于可评测。保存期间训练继续，未发生模型错误或重启。
源码/日志证据checkpoint3000_timing.json；保留完整校验，不手动提升incomplete目录。

当前四档1000 fullDCP均完整提交，FP32 EMA形状/参数数量、更新数1000、decay0.99及8分片均核对。
RM/validation/production_step1000_ema_audit.json已complete=true，CPU检查器PID352468已正常结束。
RM/validation/production_step1000_dispatch_audit.json证明8个别名均在实际8H200 worker启动；
775M和1.56B的EMA先后在dev1，其他远端。该dispatch文件只证明调度与worker身份；1000完整FVD现已全部完成，独立核验见上节。
后续继续每500步两种权重独立评测；已完成远端任务自动删除，dev1任务退出后复用空卡。
原始20B CA@3000已实证CPU扫描→独立H200任务；继续跟随后续checkpoint，不恢复已退役常驻GPU池。

## 最新运行核查（2026-09-21 11:23:02 UTC）

RM/validation/progress_and_1000_verified_20260921T1122Z.json记录直接KAP核查：
6个训练任务均Running且无Failed/Succeeded条件；日志快照步数：
原20B concat3500、全层CA3074；当前等参数125/355/775/1560M为1967/1952/2079/1593。
11:26 UTC前三档均已有2000完整DCP，1.56B已有1500完整DCP；前三档2000的online/EMA共6个实际worker均直接确认Running，证据RM/validation/production_step2000_dispatch_verified.json。
原始171/200＋本轮至少36/160=207/360份当前正式结果；长期goal继续active。

本轮1000远端评测的6个成功任务已删除，另355M一次Unschedulable旧任务亦已删除，
receipt均记录NotFound；775M/1560M的1000 EMA为dev1本地已完成任务。
1500：125M online/EMA=1489.28675/1163.47880，
355M=1470.65333/602.47463，775M=719.47418/389.90424；
1560M online在dev1、EMA在独立远端8H200采样。
775M的2000 online/EMA均已自动提交远端独立8H200任务。

原20B CA3000评测stormyxie-sc-fvdx-0921105737已写出256/2048 cases。
20B entry3500在11:26 UTC完整提交，Native full manifest=step3500/world8，weights/optimizer的DCP metadata和payload、trainer八个rank文件均存在；原5分钟CPU扫描器已自动入队，首次szr申请因无完整8卡节点撤回；11:35 UTC自动改到AWS，实际8卡worker已Running。证据R/validation/retire_idle_primary_20260921T1031Z/entry3500_committed.json。
先前真实8H200均100% GPU、约99–102GiB显存，证据
R/validation/retire_idle_primary_20260921T1031Z/eval_step3000_gpu_activity.json。
Native eval_manifest只在全case loop结束后写入，运行期间没有manifest不是初始化失败。
5个当前CPU控制器cmdline均仍匹配，预览/first-cross/ HTTP200。
仅状态与文档更新，训练/采样snapshot、配置、FVD协议均未修改。

## 新分数与资源重排核查（2026-09-21 11:37:58 UTC）

当前等参数125M@2000 online/EMA为630.9794644082217/796.7356851750951；
775M@2000为328.9420814462792/233.50764673009462。
四份均通过当前production committed-score、2048cases、CFG1、144generatedframes、
weight identity和Native manifest/protocol校验；网页metadata中的四个新点逐项一致。
证据RM/validation/125m_step2000_scores_committed.json、
775m_step2000_scores_committed.json、first_step2000_plot_points_verified.json。
此轮只做production验证与图表对齐，不能称为额外全特征独立复算。
四个完成远端任务均已有deleted=true/NotFound，125M还直接核对无Pod。

原20B entry3500：
- 首次stormyxie-sc-fvdx-0921113212实际cluster是tenc-szr3212j，Pod Unschedulable，
  控制器已自动删除/NotFound，原因no_schedulable_H200_node；不是训练/模型失败。
- 替代stormyxie-sc-fvdx-0921113514在tenc-awsusw2，实际worker Running、GPU请求8，
  节点hyperpod-i-0d8225c1133797fd6，与当前六个训练节点均不同。
- 同一目标仅一个活动receipt，证据R/validation/retire_idle_primary_20260921T1031Z/entry3500_dispatch_verified.json。
- 查询必须使用receipt.cluster，不可假设新任务全在AWS；错误cluster的NotFound不表示原任务消失。

775M训练一次API DNS查询超时，页面暂时observation_unavailable；重查同Task/Pod为Running、
restartCount0，原始日志已继续到2406且loss/EMA有限，无需重启。
11:38 UTC，1.56B@1500 online/EMA=404.70487247310075/277.74665797002535，
355M@2000=681.2286499193619/370.7779811772634，四份production验证已通过，
证据RM/validation/first30_scores_latest_committed.json。本轮合计30份正式结果；
1500步8份均已产出，尚未做额外全case独立复算。
本轮没有修改任何训练或采样snapshot、配置、checkpoint或协议；继续两个20B及后续新checkpoint。

## 2000 fullDCP与下一批评测（2026-09-21 11:50:29 UTC）

四档2000 fullDCP的模型/EMA名称与形状、FP32shadow元数据、参数数量、
加载的EMA更新数2000和decay0.99均通过；manifest hash再次核对不变。
RM/validation/production_step2000_ema_audit.json complete=true，CPU检查器PID374107
正常退出/returncode0；这是元数据与EMA标量检查，不是全部权重数值扫描。

775M@2500已完整提交并自动启动：online远端stormyxie-sc-xmatch-fvd-0921114253、
EMA在dev1 stormyxie-sc-xmatch-local-0921114251（wrapper PID372529），各8H200。
实证见RM/validation/775m_step2500_dispatch_verified.json。
11:49 UTC125M/355M也已完整提交2500；1.56B已完整提交2000，
EMA任务stormyxie-sc-xmatch-fvd-0921114922在AWS提交，其他目标由既有队列继续处理。
不要为了更快观察手动重复submit或覆盖在线runtime/config。

11:56 UTC，775M@2500 online/EMA FVD=478.95516431306123/154.23035071114418，
两份均通过production提交、Native manifest、权重身份和2048case校验；
RM/validation/775m_step2500_scores_committed.json。保留online回升，不筛掉该点。
下一批8个目标已逐项核对实际启动，775M EMA随后完成；
next_eight_evaluations_dispatch_verified.json已complete=true（表示调度验证，不是8份FVD全完成）。
355M EMA@2500的szr任务stormyxie-sc-xmatch-fvd-0921115151因Unschedulable自动撤回，
替代AWS任务stormyxie-sc-xmatch-fvd-0921115448已实际Running，未修改模型配置。

11:59 UTC，125M@2500 online/EMA=819.859902888838/535.5818470501858，
两份production/protocol验证通过，图表值一致；证据125m_step2500_scores_committed.json。
stormyxie-sc-xmatch-fvd-0921115037与0921115114均已deleted/NotFound。
2,500步online存在回升，保持原始结果，不以EMA曲线替换online。

355M@2500 online/EMA=230.73286615295137/262.0295503958823，
production验证通过；355m_step2500_scores_committed.json。
online任务0921115155、EMA替代任务0921115448均已deleted/NotFound。
前三档同2500步的入口concat/全层CA/首层CA online/首层CA EMA共12个分数已整理，
与图表12/12逐项一致，基础FVD协议按字节相同；
见common_step2500_conditioning_comparison.json。只有首层版本维护EMA，不把EMA优势全归因于结构。
截至此轮四档补参数版至少36份正式结果；1.56B@2000还在采样。

RM/validation/training_remaining_time_20260921T1145Z.json使用最后200个连续完成step的
mean step_time_e2e_s（不删慢步）估算：20B entry/cross当前3571/3138步，
到10k剩约38.1/46.6小时；当前四档约6.2/6.2/5.7/8.4小时。
此为后续速度相近且无抢占时的训练时间，不含最终checkpoint额外FVD。
resources_after_first30_scores.json记录11:40附近训练48H200+评测16H200、匹配组29个
历史远端receipt均已deleted、dev1当时无GPU进程；其后新2500采样启动，不能继续称dev1空闲。

## 历史归档

本次前全文保存在[前一轮交接](handoff/archive/20260921T104726Z/HANDOFF.md)，hash见[索引](handoff/archive/README.md)。
更早完整611行/3351行分别在两仓库20260921T090641Z归档。稳定规则在AGENTS；
历史Running/PID不构成恢复指令。原参考仓库[交接](/threed-code/stormyxie/rep_miniworld/HANDOFF.md)。
