# vforge / RepWorld 当前交接

更新时间：2026-09-21 09:23:13 UTC。本文件是接手入口；状态表是带时间的快照，不是实时心跳。
先读 [AGENTS.md](AGENTS.md)，再查实际配置、日志、完整 checkpoint 与 KAP 状态。
本次仅整理文档，未修改训练、评测、运行快照或任务。

## 当前任务与用户约束

当前工作在 dev1，主实现是本仓库 `xj_dev`，HEAD `527317858e8ab5b975fab1859a0317f53503ebdd`。
此 HEAD 之后的 FA4、batch 采样、首层 CA 顺序、EMA 代码及测试仍有未提交修改；
**HEAD 不能代表正在运行的全部代码**。保留已有 working tree，不 reset/clean，不直接合 master。
Git 作者邮箱为 `stormyxie@tencent.com`；SSH key 只证明推送身份，不替代 commit author。

- 原始 scaling：125M / 355M / 775M / 1.56B / 20B × 入口 concat only / 全层 CA only，共 10 组。
- 新增三组：前三个小模型仅 block 0 做 **CA → SA → FFN**，其余层 SA → FFN；
  从头训练并维护 **ViT + DiT EMA 0.99**，每档同时评估 online 与 EMA。
- 当前只用 **H200**。每组训练 8 卡、Native FSDP2、physical local batch 8 / global 64 clips，
  latent input、no-zprev、mask 80%、FA4，无 trainable ViT/DiT 的 SDPA fallback。
  Frozen VideoRAE codec 保留发布权重的 SDPA 数值路径。用户允许 FA3，但当前实跑为 FA4。
- 最多 10,000 optimizer steps，每 500 步保存完整 DCP，保留全部评测所需 checkpoint。
  FVD 在独立评测资源上跑；dev1 仅在所需 8 张卡空闲时参与采样。
- 全部采样 CFG1；batch level 已部署，不再使用早期 batch1 吞吐估计。
- 原始 200 个正式 FVD + 新三组双权重 120 个 = **320 个正式结果**。
  仍有 20B 训练与评测、新三组训练与评测未完成，长期目标保持 active。
- CP 仍按用户要求不改；不要把当前 Ulysses USP 称为 Megatron CP 或 Ring USP。

## 事实来源和目录

| 名称 | 完整路径 / 用途 |
| --- | --- |
| J：原始任务 | `/threed-code/stormyxie/vforge_jobs/scaling_fa4_20260920T134440Z` |
| R：原始报告 | `/efs/stormyxie/reports/scaling_fa4_20260920T134440Z` |
| 原始训练输出 | `/efs/stormyxie/outputs/vforge/scaling_fa4_20260920T134440Z` |
| JE：当前首层 CA + EMA 任务 | `/threed-code/stormyxie/vforge_jobs/scaling_first_cross_ema_20260921T084210Z` |
| RE：当前首层 CA + EMA 报告 | `/efs/stormyxie/reports/scaling_fa4_20260920T134440Z/first_layer_cross_ema` |
| 首层 CA + EMA 输出 | `/efs/stormyxie/outputs/vforge/scaling_first_cross_ema_20260921T084210Z` |
| 原项目参考 | `/threed-code/stormyxie/rep_miniworld`；[当前交接](/threed-code/stormyxie/rep_miniworld/HANDOFF.md) |
| Python / KAP CLI | `/efs/stormyxie/venvs/vforge/bin/python` / `/efs/stormyxie/koala/koala-cli` |

活动指针：
`/efs/stormyxie/work/scaling_fa4_active.json`、
`/efs/stormyxie/work/scaling_first_cross_active.json`、
`/efs/stormyxie/work/first_cross_ema_active.json`。

**J/state.json 存在，R/state.json 不存在。** state 是配置/指针来源，其旧
`status=design_and_preflight` 不代表当前健康状态。当前状态看 R/progress.json、
RE/progress.json、各 run 的原始训练事件、`tasks/<run>/submitted.json` 和实时 KAP。
JE/state.json 与 RE/state.json 是新 campaign 的记录，不要与原始路径混淆。
R/HANDOFF.md 留存调度过程与历史证据，旧段落不能覆盖本文件的新状态。

## 训练与正式评测快照

| Campaign / run | 已完成 steps | 最新完整 ckpt | 正式 FVD | 日志/调度状态 |
| --- | ---: | ---: | ---: | --- |
| 原始 125m_entry_concat | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 125m_cross_attention | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 355m_entry_concat | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 355m_cross_attention | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 775m_entry_concat | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 775m_cross_attention | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 1560m_entry_concat | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 1560m_cross_attention | 10000 | 10000 | 20 / 20 | complete / Succeeded |
| 原始 20b_entry_concat | 3115 | 3000 | 3 / 20 | training / Running |
| 原始 20b_cross_attention | 2737 | 2500 | 3 / 20 | training / Running |
| 首层CA+EMA 125m_cross_attention | 291 | 尚未到500 | 0 online + 0 EMA / 40 | training / Running |
| 首层CA+EMA 355m_cross_attention | 277 | 尚未到500 | 0 online + 0 EMA / 40 | training / Running |
| 首层CA+EMA 775m_cross_attention | 291 | 尚未到500 | 0 online + 0 EMA / 40 | training / Running |

原始报告快照：2026-09-21 09:02:59 UTC；新 EMA 报告快照：2026-09-21 09:06:34 UTC。

上表取自各报告的 progress.json（分别标注快照时间），不是重新执行全量训练审计。
2026-09-21 08:56:41 UTC 的独立 KAP/日志核查在
`R/validation/live_audit_20260921T085641Z.json`：两组 20B、三组新 EMA 训练、
当时的主/额外评测任务均 Running。后续以新日志和直接查询为准。
原始八个小模型均已完成 10k 和各 20 次正式评测；任务不再 Running 属于正常完成，
不得因此重启。

新三组完整任务名：

| 模型族 | KAP task（tenc-awsusw2 / gwm，各 8 H200） | 实际可训练参数 |
| --- | --- | ---: |
| 125M | `stormyxie-sc-125m-xema-0921085012` | 117,661,696 |
| 355M | `stormyxie-sc-355m-xema-0921085016` | 333,760,224 |
| 775M | `stormyxie-sc-775m-xema-0921085020` | 728,713,152 |

模型族沿用 baseline 标签；去掉后续 CA 后实际总参数减少，未重新扩 FFN，
**不能称为等参数消融**。原始入口 concat / 全层 CA 为匹配目标参数调整过 DiT FFN，
也要在比较时注明。新 EMA 与紧邻的 CA-before-SA 前版本只改 EMA 和输出/保存路径，
差异证据 `RE/validation/config_changes.json`。

## EMA、训练顺序与 checkpoint 语义

- `cross_attention_position` 默认 `after_self`，新三组显式 `before_self`；
  `cross_attention_every=DiT depth` 令仅 block 0 有 CA。默认模型顺序仍为 SA → CA → FFN。
- `model.extras.ema.enabled=true`、`decay=0.99`；从初始在线参数复制 FP32 EMA shadow，
  保留 FSDP shard 布局。从第一次 optimizer update 起每次执行
  `ema = 0.99 * ema + 0.01 * online`，不使用 bias correction / warmup decay。
- 覆盖联合训练的 ViT 与 DiT，排除 frozen codec。`num_updates`、参数名顺序、decay 与
  shadows 随 Native full DCP 保存恢复，更新次数必须与 completed_steps 一致。
- 代码在 [任务 EMA](vforge/tasks/rep_world/ema.py)，装配在
  [shared](vforge/tasks/rep_world/shared.py) / [train](vforge/tasks/rep_world/train.py) /
  [eval](vforge/tasks/rep_world/eval.py)。复用 ShardedEmaState、Native hook 与 DCP，
  不复制 Trainer，不让模型包 import training。
- 评测显式选择 `model.extras.ema.evaluation_weights=online|ema`。
  online 从同一 DCP 的 net + causal_semantic_encoder 组件严格加载；
  EMA 还严格加载 ema subtree 并检查更新数，然后 copy_to。缺 EMA 必须报错，不能退到 online。
- JE/runs.json 只含 **3 个训练 run**；JE/eval_runs.json 含 **6 个评测别名**，
  后缀 `__online` / `__ema`。每个已完整保存的 checkpoint 产生两个独立评测身份、
  lease 和结果目录：`RE/fvd/results/<alias>/<step:05d>`。
- 不把 6 个 alias 加进原始 J/runs.json；原始 small-model gate 仍按 8 组判断。
  旧权重只可 weights-only 初始化；不能假装恢复旧 optimizer/data/RNG。
  跨 wrapper 读取 FSDP checkpoint 使用 Native strict component loader，
  不通过关闭 strictness 绕过 whole-root fingerprint 不匹配。

## FVD 协议与采样资源

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
新 EMA campaign 的综合 protocol hash 为
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

## 监控与预览

以下 PID 于本次整理时通过 /proc/cmdline 核对，后续发送 signal 前必须重新核对身份：

| 服务 | PID | 脚本 / 行为 |
| --- | ---: | --- |
| 原始监控 | 249578 | J/monitor.py，约 300s 一轮 |
| 原始额外评测 | 245631 | J/fvd_extras.py，当前 cap 17 节点 |
| EMA 监控 | 317417 | JE/monitor.py，约 60s 一轮 |
| EMA 额外评测 | 317418 | JE/fvd_extras.py，cap 3 节点 |
| EMA 本地评测 | 317419 | JE/fvd_local.py，只在 dev1 所需卡空闲时分配 |
| 8884 预览 | 318092 | J/serve.py，remote loopback 127.0.0.1:8884 |

EMA 服务记录在 RE/services.json。cap 是控制器上限，不是已用资源；
统计资源必须去重直接查询 task / worker Pod / GPU 归属，不能把 queue、lease、active 重复计数。
原始 primary 评测是独立资源，不能算入 extras cap。
已完成/明确失败的自有评测任务及时清理，保留结果和证据；不触碰无关任务。

- [原始 scaling、标准 / 双 log LOWESS / OLS 图](http://127.0.0.1:8884/)
- [首层 CA + EMA](http://127.0.0.1:8884/first-cross/)
- [历史七组 condition 报告](http://127.0.0.1:8884/condition-ablation/)

视频保留在 EFS；8884 使用固定媒体白名单与 Range 206。
保持本地到 dev1 的 SSH 预览隧道；用户之前要求保留的 8882/8883 也不要无故关闭。
预览故障先分别查远端 loopback、SSH forward 与本地 HTTP，不要重启训练解决网页问题。
parent progress 可滞后 5 分钟，长批次的安静日志也不等于任务失败。
先看实际日志进度、产物 mtime、Pod exit/attempt 与 GPU 活动，再决定是否恢复。

## 已验证与尚未覆盖

- EMA：`RE/validation/gpu_ema_smoke_PASS.json`，真实 8 H200 / FSDP2 / FA4，
  train2 → full-resume3，online/EMA 各 16 cases、batch2/rank；
  相同 case/seed/GT、16 个生成 RGB hash 均区分两种权重、EMA count=3。
- `gpu_checkpoint_ema_recurrence.json`：全部 338 tensors / 117,661,696 elements，
  FP32、8-way sharding，step2→3 CPU 重算最大误差 7.450580596923828e-09。
- `dual_queue_contract.json`：3 checkpoints → 6 个独立队列项；
  `training_started_verified.json` / `restart_task_status.json`：
  新任务从 step0 连续有限值、更新数一致，旧任务已停止。
- EMA 初轮 53 个定向 CPU 测试、最终 13 个 EMA/resume 测试通过；
  56-file changed lint、264 个 vendored mcore hashes 与 import 检查通过。
  证据在 RE/validation；这不是全仓库 test suite 全绿声明。
- CA 顺序回归：163 个 CPU + 2 个 masked train/infer 用例通过，
  默认 after_self 的输出/KV/梯度与旧实现 bitwise 一致。
- 并行：单机 H200、双机 2×8 B300 的 DDP / FSDP / HSDP / Torch Ulysses 已做真实验证。
  双机证据 `/efs/stormyxie/outputs/vforge/multinode_audit_20260919/REPORT.md`；
  单机证据 `/efs/stormyxie/outputs/vforge/parallel_audit_20260919/REPORT.md`。
  DDP+USP 拒绝；CP/Ring、TP/EP/PP、>2 节点等仍未验证，不按框架有模块推断支持。
- FA4/SDPA 短期 loss 接近，**严格独立优化轨迹等价未通过**。100-step 固定 batch
  loss 诊断不是训练收敛评测；Muon 会放大小梯度差异。默认仍 SDPA，当前实验显式 opt-in FA4。
  `/efs/stormyxie/reports/repworld_fa4_100_20260920T094849Z` 保留原始度量。
- 历史全仓库 CPU 检查有 private Gen Engine / H3 依赖缺失与 baseline Wan failures；
  不能用定向测试通过抹去这些限制，详见归档。
- 新三组的 step201–300 效率已自动生成并独立核对全部900条原始事件、hash、
  无 checkpoint/demo、EMA更新次数与吞吐公式；网页明确区分历史基线及当前EMA。
  证据 RE/validation/training_efficiency_201_300_verified.json。新三组典型 steps/s
  为0.655/0.504/0.400，区间有效值为0.342/0.323/0.343；不能把典型提升当作端到端收益。
- 三组首个 production step500 full DCP 已落盘并独立读取验证：EMA count500、decay0.99、
  FP32 shadow 数量/形状/总参数与 online 对齐，最大张量8分片，weights/optimizer/trainer组件完整。
  证据 RE/validation/production_step500_ema_audit.json。6个online/EMA评测身份均已进队列；
  production_step500_dispatch_audit.json记录启动情况，尚无正式分数，继续等实际采样/score。
  部分新评测 task 沿用 xpre-fvd 前缀，按 receipt / EMA 输出路径辨认，不按前缀误停。
- 主报告速度0.600/0.593来自125M step9970–9999的最近30条非checkpoint日志，
  /first-cross/原表0.528/0.341来自step201–300的典型/有效吞吐，窗口和斜杠含义不同。
  后续预选1201–1300/4201–4300/8201–8300窗口125M均为concat典型速度更快；
  CA反向始终更慢。差异集中在forward，日志包含codec/模型等，不能归因纯CA kernel。
  报告已加入窗口对照；R/validation/conditioning_speed_windows_20260921.json与
  conditioning_speed_main_window_20260921.json保留复算证据。

## 已停、已替代与下一步

不要按历史 Running 段落恢复这些任务：
原始 mask80 **32 GPU** 任务；旧 RGB patch4 / DuET；历史 A/B/C/D/E/G/H condition；
`scaling_first_cross_20260921T075334Z`（SA→CA）；
`scaling_first_cross_pre_20260921T081808Z`（CA→SA、无 EMA）。
两个旧 first-cross campaign 的训练和控制器均已停，保留输出。
新 active 指针均指向 `scaling_first_cross_ema_20260921T084210Z`。

接下来按顺序：

1. 三组 step500 的完整 DCP / EMA 已核对。继续检查 online/EMA 各自实际权重加载、
   首批采样 provenance、2048-case 正式 score 和任务清理；队列创建不等于评测完成。
2. step201–300 效率已核对并显示在 /first-cross/；后续需要新窗口时保持相同统计口径，
   区分典型步时与区间有效吞吐、EMA开销、节点差异和参数变化，不以历史旧顺序替代当前结果。
3. 继续 20B 两组到 10k，发现完整 DCP 后评估；八个小模型已测完。
   用户早先的“小模型优先”已满足，20B 正在继续；不要停掉有效的 20B 采样。
4. 用实际 task 状态/退出码/完整 checkpoint 判定完成；只清理明确属于本 campaign 的资源。
   正式结果检查协议和输入 hash，绘图保留标准图、LOWESS 和 OLS，不用拟合图证明 scaling law。
5. EFS 原子写恢复曾发生并已解决：
   `R/validation/atomic_artifact_recovery/recovery_audit_complete.json`。
   不重复计算/覆盖已审计的 1.56B entry 9000/9500 结果。

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

## 历史归档与维护

整理前全文（611 行）按原字节保存在
[历史 HANDOFF](handoff/archive/20260921T090641Z/HANDOFF.md)，原 AGENTS 同目录保存；
[归档索引](handoff/archive/README.md) 提供原章节行号和 SHA256。
历史文件中的 Running、PIDs、staged-only、batch1/no-batching 等只描述当时状态，
不构成当前操作指令。重要迁移数值语义仍在 AGENTS 与 [模型指南](docs/models/rep-world.md)。
后续更新当前摘要和证据链接；不要继续把整段开发日志无限追加在顶部或尾部。
