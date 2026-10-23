# vforge / RepWorld 当前交接

更新时间：2026-09-21 10:36:00 UTC。工作在dev1；先读[AGENTS.md](AGENTS.md)，核对实际task、日志、完整checkpoint。
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
  online/EMA独立采样与正式score，待首批8份齐全再复核完整双权重结果；持续原20B至10k。
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

2026-09-21 10:27:20 UTC 核查：

原始20B当前11份batch-v2正式结果全部独立协方差复算，proof SHA对应当前score文件；
包括entry500/1000/1500/2000/2500/3000与CA500/1000/1500/2000/2500。
新CA2000/2500为415.85880/362.74407，entry3000为281.65305。
证据RM/validation/current_20b_scores_verified.json及原R/validation/fvd_new_20b_scores_20260921T1025Z。
原始正式分数171/200；原8小模型已完成，不重新跑。

本轮四个500 checkpoint全部核对，8个评测别名均在实际8H200 worker启动，见
RM/validation/production_step500_dispatch_audit.json；其中775M EMA在dev1，其余远端。
截至核查已产出6/160份新正式FVD；1.56B online/EMA仍在采样。
当前离线完整case/有限特征/数值复算已核验其中5份；355M online在该检查器经过后才完成，
须下一轮连同1.56B重新核对，不能把本轮5份核验称作全部8份通过。
RM/validation/ema_scores_verified_00500.json记录确切范围。
检查器遗留required_scores=6已改为len(eval_runs)=8；证据ema_verifier_four_arm_fix.json。
这是外部结果检查helper修正，没有修改训练/采样runtime或实验协议。

| 模型 | 500-step online FVD | EMA FVD |
| --- | ---: | ---: |
| 125m | 5613.69113 | 7292.98111 |
| 355m | 1727.09394 | 3252.13353 |
| 775m | 1838.13083 | 1599.78384 |
| 1560m | 采样中 | 采样中 |

125M与775M两对全部2048 case/checkpoint/GT/seed一致，全部输出hash不同；
不要从500步单点声称EMA总是更好或已确定结构优劣。
完成的5个远端首批评测均Succeeded后自动删除NotFound；dev1首批local评测已完成退出。
355M首次Unschedulable后重提成功，并已产出online分数。

最近直接KAP查询四组训练均Running，completed_steps依次882/865/942/719；
EMA更新数与实际步数一致。monitor/extras/local三服务完整cmdline均核对。
下一步继续跟随每500步新DCP，独立双权重FVD，完成资源及时释放。

## 原始 scaling 评测改为按需节点（2026-09-21 10:35 UTC）

原常驻GPU评测 task stormyxie-sc-fvdbatch-0921074040 在确认所有已有checkpoint均出分、
active/pending均空后主动删除；任务NotFound且无Pod，释放8H200。不是模型失败，不要恢复常驻池。
J/monitor.py现按 fvd/pool_control.json 的 primary_scheduler=false 由dev1 CPU发布完整DCP队列；
J/fvd_extras.py对无primary的队列不再预留前两个checkpoint，仍用既有lease去重和独立8H200任务。
修改仅外部控制器，未改训练/采样runtime、配置或协议。2个定向检查及171份生产分数扫描通过。
新CPU服务在R/services.json：monitor PID348486、extras PID348487；发signal前仍须重新核对cmdline。
旧CPU PID249578/245631已经退出（可能保留Z状态的僵尸/proc项，不表示仍在运行）。
R/validation/retire_idle_primary_20260921T1031Z保存before/after、测试、删除和生产队列核验。
生产新队列已发布OnDemand状态，无重复评测；完整的新20B checkpoint→独立任务链路等待下一份DCP实证。

当前等参数首层CA：125M/355M/775M的1000 fullDCP与EMAcount=1000已核对，
RM/validation/production_step1000_ema_audit.json仍complete=false（1.56B未到1000）。
三个小档1000 online/EMA已自动排入独立任务，775M EMA在dev1。
1.56B 500-step EMA正式FVD已出1281.1846751187777；online尚在完成。
四档500离线全case复核第二轮正在CPU执行，PID349177，日志
RM/validation/ema_scores_verify_second.log、过程身份ema_scores_verify_second_process.json；
未读到最终complete=true前，不声明8份全部验证。此前5份验证范围保持成立。

## 历史归档

本次前全文保存在[前一轮交接](handoff/archive/20260921T094202Z/HANDOFF.md)，hash见[索引](handoff/archive/README.md)。
更早完整611行/3351行分别在两仓库20260921T090641Z归档。稳定规则在AGENTS；
历史Running/PID不构成恢复指令。原参考仓库[交接](/threed-code/stormyxie/rep_miniworld/HANDOFF.md)。
