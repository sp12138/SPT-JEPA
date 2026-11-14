# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。最新用户要求继续改进独立动作头控制、同会话每小时检查，直到验证可靠稳定的scaling，不在有限pilot后自行HOLD。资源只限dev13/dev15/dev6；旧pilot、长训、goal和旧定时监控不恢复。

## 当前持续实验

- 当前根为`runs/robocasa_reach3d_scaling_20261114`，旧根`runs/robocasa_stable_scaling_20261114`已整阶段停止，全部原始记录及完整训练状态保留。这是同一用户目标下的数据设定修正，不是暂停目标。每小时继续检查；旧goal/旧监控不恢复。实际状态看新根`MASTER.json`、`SESSION_CHECKS.json`、`HOST-*.json`及`training/*`，不从本文推断持续卡利用率。
- 确认了一项此前未核对充分的差异：训练用ObjectPush控制器，z被限在park−.16m至park+.03m；Reach控制器另用空间球/姿态保持。原17920事实转移的动作→实际位移线性响应xyz约.523/.545/.296，较大z指令约45.8%产生不到2mm的z位移。不是完全没有z运动，也未证明这是唯一原因。原状态只用于只读审计，绝不进入学习器/教师；定位旧根`protected-actuator-audit`。
- 已在TRAIN厨房通过Reach接口试验：三个轴±20mm实际移动约19mm。新采集为同公开Reach执行接口、TRAIN/validation厨房、纯随机多尺度当前命令；只记录RGB/实际输入，不记录目标、奖励、状态或专家路径。`runs/robocasa_reach3d_train_20261114`原56lane×20条全部1120、`runs/robocasa_reach3d_val_20261114`8×12全部96，无采集错误。额外采集原计划560条，已提交部分只作未标注经验增长，不改变固定动作标签集合，不把计划数当完成数。
- 原SPT/宽状态/条件率完整10h、原deep/残差完整13h、高噪声完整6h及各64例控制后均已交接，模型/优化器/调度/4rank RNG/游标核验通过。六组各4卡：dev13的0–3联合物理码、4–7像素加法传输；dev15的0–3 deep+仿射Euler、4–7 deep+残差；dev6的0–3原仿射、4–7 deep+仿射RK4，并共用既有普通采集器。实际归属查新根`MASTER.json`及进程，不由本文推断持续利用率。新根`runs/robocasa_reach3d_{joint_pixel,deep_refinements,control_flow,pixel_transport}_20261114`均已真实更新并通过8/16步完整恢复。dev6交接退出检查竞态已修复，原错误保留；验证旧完整13h状态且无旧worker后只启动一次RK4，无旧SGD重放。新加法组交接也无旧SGD重放。Euler/RK4初始NN逐位相同、各36,773,888参数，但RK4计算及dev6共卡开销不同，不称等FLOPs。联合128D对独立2×64D的池化/F初始化与参数量不同。所有新E/F/decoder随机，无退役NN祖先；共享原语料版本/固定标签、冻结外部G例外、FP32小E/action及共同调度。同seed1非独立重复。
- 三机先验证本地1216条缓存SHA，再启动16个保留功能更新与真实全状态恢复；`PRETRAIN_READY.json`是数据门。有限采集尚占卡时controller等待，不把等待算成训练。dev6后续八条Reach事实采集可与训练共用设备；归属和实际GPU看新根，非额外机器。所有恢复含model/optimizer/scheduler/RNG/游标，未保存重放须披露。
- 动作头始终冻结/detach，只用同一转移a_t的17920/1536个XYZ标签；body无动作target。数据从旧Push改为Reach，旧/新协议单列，不把数据变化算算法公平收益。仍同64已见开发案例、5动作、norm .3、native2mm final/any；32确认例保持未打开。固定标签/同案例的多预算曲线、后段稳定及严格完成进步才算目标，不用loss或单点代替。
- 同Reach标签冻结G均值/旧Push-SPT5h/入口8更新为77.67/84.28/58.29%，final0/1/0（各64例），非公开RoboJEPA策略复现。原deep1/3/6/12/13h为92.20/94.30/93.62/93.09/93.08%，末端17.26/12.44/14.30/15.79/15.83mm，后段平台；仿射12/13h94.20/93.94%、12.97/13.59mm、4/2final。新高噪声/深残差/联合码3h92.60/93.53/91.54%，均未超过原deep3h。每完整训练小时固定读出闭环；有限诊断DONE不结束目标。相对线性头、状态条件头不推广：后者deep10/12h92.26/92.20%、16.44/17.29mm，未保持6h收益。原始根`runs/robocasa_reach3d_{relative,deep_response}_20261114`。
- 同容量384参数、同5000次SGD的MAE线性头1/3/6/9/12/13h完整曲线为91.45/93.84/93.90/94.24/94.62/94.54%、18.24/13.23/13.13/13.13/11.61/12.12mm，final0/1/2/2/1/1。后段粗定位较MSE稳定，13h轻微回落、严格成功未scaling；根`runs/robocasa_reach3d_robust_linear_20261114`。压缩三维坐标上的Broyden校准deep12h-MAE/仿射12h均变差（94.31/93.19%、13.62/17.05mm），不推广，原始`runs/robocasa_reach3d_servo_20261114`。
- 冻结F的当前动作响应`B(z)`生成读出坐标`B(z)^+ delta_z`，仅拟合9参数C；E/F无动作target梯度。同一原仿射父模型1/3/6/9/12/13h完整曲线为91.27/93.09/93.50/93.82/94.86/93.67%、19.33/15.10/12.98/12.92/9.70/13.65mm，final1/0/1/3/3/2，每点64开发例；13h回落，未达标。使用了F，不能混作E-only probe。固定C=I且0动作标签的12h为92.97%、13.71mm、1final，弱于校准C。根`runs/robocasa_reach3d_response_{geometry,identity}_20261114`。TRAIN残差协方差虽改善验证NLL，但白化后C9/I的12h为93.37/92.76%、13.22/13.24mm、各2final，整体未改善，不扩展；根`runs/robocasa_reach3d_response_likelihood_20261114`。解析解、默认回归及核心协方差拟合检查通过。
- 投影前保留128维视觉变化，以`B(z) C^-1 + R_episode`作动作响应，R每案例清空、只用本回合最近实际RGB变化/已执行a_t更新。12/13h各64例已齐：94.87/93.86%、9.71/12.66mm，均3final，any7/6；首动作与原策略最大误差0。12h基本持平，13h较固定响应距离改善约.99mm，但仍后段回落、不推广为稳定方案。E/F/C不变、0新body/监督头更新，根`runs/robocasa_reach3d_response_servo_20261114`。原仿射直接头14h回落至91.75%、19.39mm、3final；新Euler/RK4各完成1h（90.87/88.82%），仍需后续曲线。
- 冻结E的正向后果读出，目标delta_z、当前a_t仅作输入；常数512参数及状态条件66048参数，含drift、同ridge网格及验证下一码MSE选择。原仿射12h常数/状态条件为89.43/90.39%、23.54/24.17mm，final0/2；13h常数89.49%、22.89mm、0final，状态条件待收。验证预测MSE约减半不等于控制改善；矩阵条件数正常、去掉动作输入误差显著变大，未证简单数值退化或只靠drift。审计在同根`FACTUAL-DRIFT-AUDIT.json`，零动作预测不是物理反事实。根`runs/robocasa_reach3d_forward_effect_20261114`，不是原inverse probe，另有正向头拟合成本。
- 采集顺序有限对照的全数据/随机前半/反向后半192记录已齐：89.67/85.23/91.29%、23.54/33.09/21.93mm，final0/1/5、any5/4/8。原deep12h同父E、同66048参数、同1536验证；训练17920/8960/8960转移。后两组命令范数多重集相同，但状态覆盖/历史未受控，不能作唯一因果证明。5final是弱开发集信号，回报/距离仍差，不推广整体算法。根`runs/robocasa_reach3d_randomized_effect_20261114`；原回归缺runtime符号链接的导入错误和修复保留，默认全数据真实拟合逐位回归/切片配对/detach检查通过。32确认例仍未打开。
- 新像素加法传输根`runs/robocasa_reach3d_pixel_transport_20261114`已实际启动，保留更新及8/16步完整恢复通过；尚无控制结果。0SGD梯度、原deep E/pool/renderer逐位配对、合成/恒等/样本隔离通过，35,958,528活跃参数。保留像素flow/rate，惟`m_next=m_current+A a_t`，新E/A/renderer随机、原预测NN停用；不是旧feature-transport，也未证明一般操纵有效。高噪声完整6h为92.24%、18.42mm、1final/3any，92606步全状态及64控制保留，CPU模型/优化器加载一致、scheduler/4rank RNG/游标核验后只接班一次，0旧SGD重放；证明在`HANDOFF-dev13.json`及`ACTUAL-STARTUP-VERIFIED.json`。新小时检查器从MASTER取当前assignment，不恢复旧monitor。
- 旧阶段完整结论见[CONCLUSIONS](experiments/CONCLUSIONS.md)及[持续协议](experiments/robocasa_stable_scaling/PROTOCOL.md)。原SPT旧标签1→6h约79–81%且严格成功稀少；多尺度Push标签3h84.40%、5h83.75%仍全部0，非线性/双线性未证持续收益。不可据这批跨控制器迁移结果断言JEPA原理不适合3D。
- 一次加深E的第2h恢复在写JSON临时文件时退出，发生在该阶段任何SGD前；改为独占临时文件，160次本地及160次EFS并发写通过，从14006步完整恢复、0重放，随后随旧阶段正常收尾。原错误、恢复及全部状态保留，非模型数值失败。此前数值守卫与精度修复也保留原版本，不改在跑的冻结源码。

## 最新已完成scaling验证

- EFS根`runs/robocasa_action_scaling_20261113`；入口`experiments/robocasa_action_probe/scaling.py`，预定协议见同目录`SCALING.md`。17个单元全部完成：680条模型—案例记录，656次新控制、24条旧开发回合复用；40个独立物理案例，其中32个首次使用。`results/summary.json`、`analysis.json`、`scaling.png/pdf/csv`及`DONE.json`/`HOLD`为入口。原始记录留EFS，图表与汇总本地镜像在仓库旁`SPT-JEPA-local-archives/robocasa_action_scaling_20261113`。
- 两种SPT各入口8更新、1/2/4/8/12/17/18h，另有视觉均值参考。相同1120/96普通随机日志、ridge协议、当前a_t标签，表示冻结/detach；新增后果学习SGD为0。共享冻结外部V-JEPA，不是整个backbone从零或解冻实验。
- 旧8开发案例单列；32个新案例为厨房1/4/8/10×29301至29308，均在读分前固定，学习厨房分离，全部时间点评全部案例。曲线在这32个案例上为：

| 累计后果训练 | 64D动作头回报 | global16动作头回报 |
| --- | ---: | ---: |
| 入口8更新 | 19.05% | 43.62% |
| 1h | 81.02% | 81.89% |
| 2h | 85.22% | 81.69% |
| 4h | 85.38% | 84.40% |
| 8h | 86.50% | 83.33% |
| 12h | 86.46% | 82.58% |
| 17h | 85.51% | 81.49% |
| 18h | 84.97% | 82.64% |

- 64D距离1h47.61→8h29.37→18h34.20mm。预定18h对1h回报+3.94个百分点，四厨房区组区间[+.27,+8.08]，19/32案例改善；global16仅+.75点，区间[-2.79,+4.28]，增益不明确。64D有前段回报/距离scaling，8h后平台/回落；不能称持续scaling或通用power law。仅一个训练seed，区间不包括训练seed不确定性。
- 只有64D8h出现final/any1/32，厨房1/29304最终1.331mm，物理/初始及目标RGB重放误差0、模型不变；其余所有点严格final/any均0/32。视觉参考69.98%、76.50mm、0/32。尚无稳定精确控制或困难manipulation突破；不能拿不同案例/输入的公开RoboJEPA数字直接排名。
- 旧8案同一64D权重的CEM1→8h为78.85→73.37%，独立头却81.37→85.70%；原控制读数不能直接归因表示没学好。动作头有额外当前动作监督，不能把两控制器当同学习目标排名。动作头仅读取472528个学习参数（全45916946中的约1.03%），没有读动力学预测器/renderer，所以这不是完整F能力的scaling测量。
- 原始`READOUT_DEPENDENCY.json`和`GAUGE_DIAGNOSTIC.json`提供机制诊断：在共享槽内O(16)旋转后，重新拟合同样ridge的验证动作预测最大差约1.92e-15；1632个事实验证轨迹—目标排序中1553个出现严格L1顺序翻转。说明uniform L1缺少该坐标不变性，而校准动作读出保有它；不是完整NN重参数化/控制实验或唯一失败原因证明。
- 三个新worker均已退出，本轮EXIT实际检查三机GPU计算为空、24卡0MiB。旧controller/goal/定时监控不恢复，不自动追加长训。64D9.5h后的同batch/RNG FP32异常重算路径仍需披露；8h点在此之前。8h是完整曲线中的选择后较优点，32新案例今后应视作已见开发案例，不能再次冒称独立确认。

## 已完成独立动作读出pilot

- EFS根`/efs/prescottsun/SPT-JEPA/runs/robocasa_action_probe_20261113`，核心入口`experiments/robocasa_action_probe/probe.py`，协议见同目录`PROTOCOL.md`。最终`results/summary.json`、`paired-comparison.json`、`DONE.json`和`HOLD`；24条trace校验和、物理/RGB重放、5动作、模型不变与实际stopgrad backward均通过。有限pilot已结束，不自动长训或搜索。
- 共享冻结V-JEPA2.1-G（约1.845B）；不是本轮训练大backbone。global16/64D池化表示分别继承同数据17h的321983/326088步父模型；不继承公开动力学或策略。视觉对照只取V-JEPA patch均值，不调用RoboJEPA预测器，不能称为RoboJEPA策略复现。global16在dev15、64D在dev6；视觉对照改由已完成的dev15顺序执行，dev13待机probe在无子进程时先停止，重定位记录保留，无重复实验。
- 同一普通随机命令日志1120train/96validation，17920/1536真实转移、厨房分离。仅以同一次实际转移的输入a_t前三维为标签，输入两个真实RGB编码之差。表示冻结/detach；动作监督不进入后果学习器，没有a_(t+1)、最优动作、专家路径、reward/state/task。不是整个链条都没有动作标签。
- 线性头`W(z_next-z_current)`、无截距，sample MSE+L2；固定ridge候选1e-6/1e-4/1e-2/1，仅按验证厨房RMSE选择，三头均选.01。SPT64/global16/视觉验证RMSE .07089/.07098/.07462，零预测.10035；头拟合成本另计，不能与旧无动作监督CEM作同目标训练的公平排名。
- 相同8个Reach开发案例（厨房1/4/8/10×19301/19302）、224方形两视角、5动作、norm上限.3、2mm规则。推理将真实下一帧替换为目标RGB，直接出动作，不调用CEM。17h global16的CEM→头回报71.99→77.55%，最终88.47→59.57mm；64D为73.56→83.50%、63.59→41.12mm。逐案例回报4/8、6/8改善；final/any全0/8。视觉头回报68.54%、80.96mm，也全0/8。三个模型重复同8个物理案例，不是24独立案例或未选择确认集。
- 每回合实测均值含编码/仿真/记录、不含启动：global16 50.80→11.61秒（约4.38×），64D42.16→11.78秒（约3.58×）；头本身不能代替整回合计时。新头编码/读出约1.08/1.10秒，另有物理执行成本。中位回合约7.37/7.64秒，一例约42秒；不是所有案例均约7秒。
- 有限结论：独立动作读出有用，原CEM成绩不足以代表表示可用于控制的上限。仍有远目标分布变化、末端精度与反复修正；未证明任何一个是唯一瓶颈，未解决manipulation或超过RoboJEPA论文。
- 旧无限期controller已全部在当前完整阶段后退役：dev13/dev15/dev6分别保留21h/21h/18h完整状态，均未杀正在训练的子进程。dev13末次37386步/21h检查点已保存，该点没有新增控制评测，不能当性能结果。归属与完成证明在`boundary-*`、`OLD_STAGE_RETIRED-*`、`RETIRED_CHECKPOINTS.json`和`CLOSED.json`。最新实际检查三机均无本项目进程、GPU计算为空、24卡0MiB；这是检查时状态，不是持续占用承诺。上轮pilot结束后没有继续训练或新增成绩。之后需重新检查进程，不能恢复旧controller或从HOLD自动扩展。

## 原Reach同时间比较

- EFS根`runs/robocasa_time_compare_20261112`，旧冻结训练源码`code-v4`、dev6数值恢复`code-v5`保留。基线为309M RGB-only K2适配，global16/原64D分别约46.6M/45.9M训练参数；均共享上述冻结编码器、随机新预测头、seed1。原目标无任何动作标签或inverse，候选sample-level。不是公开23数据集训练复现。
- 本地1216日志及缓存清单三机一致，manifest`ced8adcea7896427db758ae9bade7423e5e4b8434a91a5fb973d135b6b2d4e68`。每模型8H200/globalbatch32；SPT窗口8、基线16。累计最慢rank数据/前后向/优化墙钟，排除初始化/存盘/评测；SPT每秒更新约11–12倍、图像量约5–6倍，非等容量/FLOPs比较。900秒warmup后恒定LR。完整状态包括model/Adam/时钟调度/8rank RNG及确定性数据游标。
- 原CEM1024候选/H1/首15后10轮/top20/context2/execute1/5动作/repeat1/uniform L1。三算法共同17h回报基线/global16/64D为12.30/71.99/73.56%，最终274.56/88.47/63.59mm；严格final/any全部0/8。旧CEM选择最高为64D2h79.65%，非确认成绩；新动作头不是继续训练原CEM目标得到的改善。更多SGD没有展示持续控制scaling。
- dev6约9.57h的BF16非有限梯度原错误保留，从204871步/9.5h完整状态恢复；同batch/RNG FP32重算不跳样本、不改loss/LR/optimizer，已跨过大量具体异常，不代表唯一数值根因已确定。至少1509个未保存更新重放，确切数量未知；原状态`recovery-dev6-numerics/saved-204871.ckpt`。两个额外seed/两个新假设合计32个启动更新另封存，不计入当前祖先。
- 逐小时原始结果在`results/{run}/hour-NN.json`，三机按需汇总`results/active-latest.json`；旧四方法`latest.json`滞留7h不作主入口。旧根`plan.json`已标交接和禁止自动恢复；不依据旧ACTIVE/LONG_APPROVED或本文旧协议重启。历史grid16及失联不构成当前性能或占用证据。

## 已完成300M Reach四例

- EFS根`runs/robocasa_300m_reach_20261112`。官方300M300k，1237022376字节，SHA256 `19494381dd66b2340421d8732c64526a8d35f469ce50fa5c90a0974ae4ee0838`，下载与四机本地缓存校验通过；沿用官方冻结V-JEPA2.1-G。
- 固定案例：dev13厨房1/seed0、dev15厨房4/seed1、dev6厨房8/seed2、dev6-2c厨房10/seed3；同ID的layout/style，target split，scene_seed=厨房ID-1。先选案例再运行，四物理案例，不是32独立案例。
- 各8rank×128sample×1batch=1024候选，H1、5动作、15/10轮、top20、context2、execute1、action_repeat2，translation std/maxnorm .3，mean/std momentum .15/.70，目标距离.25–.35m。渲染480×640再输入240×320，相机0/2，官方Reach embodiment2全本体参考。
- 采用bf16 eager；复制官方client仅把9处`torch.float16`改为`torch.bfloat16`，原源文件不变；外层只固定每rank规划RNG并记录CEM耗时。无AOT编译和新SGD。公开checkpoint为K10 cooldown，论文Reach曲线为K2配方，不能称同checkpoint严格数值复现；依赖环境版本也沿用既有兼容环境。
- 最终入口`results/summary.json`、`DONE.json`、`HOLD`、`CLOSED.json`。20个真实动作完成，四例数值有效，0控制执行错误、0物理重放、0SGD。厨房1/4/8/10的轨迹平均归一化回报分别.86925/.88974/.89709/.89032，总平均.88660；归一化折扣回报.88499。最终平均进度.98491，末端误差4.703/8.162/4.074/1.896mm，native2mm final/any均1/4。回报不是成功率。
- 正式每例136.99/141.70/138.69/约139.71秒；含模型启动的四机并行墙钟271.30秒=4.52分钟，不含此前下载/配置及事后归档。论文300M高预算Reach图约.95–.96，当前轨迹回报低约6–7个百分点；最终接近目标不等同于已复现累计回报，且K10/K2和四例边界保留。
- dev6-2c在完整5动作后缺ffmpeg可执行路径，原client导出视频失败。原`case03/ERROR.json`保留，NPZ/5条timings/stdout完成记录和SHA验证通过，`PERFORMANCE_COMPLETE.json`补充接管数值结果，不伪造原client exit0或重跑。`collect_final_v2.py`统一收集；其正式时间由NPZ时间戳近似推得并单独标记。该例无视频，其他三例视频存在。
- 仅本轮`bin/ffmpeg`改成兼容四机本地依赖路径的包装，四台`-version`实际通过，不改旧根。2026-11-11 18:01:34–35UTC（北京时间Nov12 02:01）EXIT快照均无本项目进程、无GPU计算、32卡0MiB。旧1B/22M及其他实验保持封存。

## 已完成1B四机测速

- 根：EFS `runs/robocasa_1b_speed_20261111`，最终`results/summary.json`、`DONE.json`、`HOLD`；有效运行`fourhost-v2`。全部driver已退出。
- 官方1B300k，4070931566字节，SHA256 `76011b30cfb39c7b5984293be6813c4b5ce4e87243992cc215a394f29aad0e5e`；下载校验通过。冻结官方V-JEPA2.1-G沿用。
- dev13/dev15/dev6/dev6-2c共32张H200共同分担一个Push案例，非四个独立案例。最快实测每卡32候选×1批，全局1024，H5、action_repeat2、15/10轮、top20、execute1、context2、240×320、相机0/2，原版客户端FP16和embodiment3本体输入参考。
- 四机初态H5每轮16×2／32×1分别5.46／4.23秒；每配置1轮热身加3轮计时。恢复测速前RNG、验证当前输入不变后才正式执行动作。原版官方client/server源SHA与22M相同，包装只测速并选择批量，没有SGD。
- 单机测速已完成并退出：8rank下每轮16×8／32×4／64×2／128×1分别21.52／16.47／14.50／14.81秒，最优64×2；仅初态H5吞吐，不是完整回合。
- 第一版四机elastic rendezvous发布了跨集群无法解析的Kubernetes FQDN，尚未执行动作。已SIGINT四个确切driver并确认旧driver/client/server退出、四机GPU无计算后，改为固定IPv4的static rendezvous。原错误和日志保留在`fourhost`，不计模型负结果。
- 完整18动作已执行、四机NODE_DONE均PASS、0有效执行错误。正式过程843.01秒=14.05分钟，其中CEM698.48秒。启动到退出1014.43秒=16.91分钟，含42.25秒批量测速；后者不含下载、单机测速及早先DNS修复，不能称整个本轮准备耗时。
- target kitchen4/style4/seed0，初始物体距离12.7244cm，与22M同一物理案例。最终进度18.7521%，距离10.3383cm，native2cm final/any均false；物体沿目标方向净位移2.7517cm。单例不代表成功率，尚非论文数值复现或同数据从零比较。
- 原始NPZ/视频在`fourhost-v2/result_`嵌套目录；`CLIENT_DONE.json`与`formal-timings.json`保留完整耗时。2026-11-11 16:07:00–01UTC（北京时间Nov12 00:07）四机EXIT快照均无本项目进程、无GPU计算，32卡0MiB；不是持续可用承诺。此前实验/goal/监控均未恢复。

## 已完成22M参照

- 根：EFS `/efs/prescottsun/SPT-JEPA/runs/robocasa_22m_eval_v2_20261111`。
- 主机：dev13，原controller PID873218已退出；8卡仅分担同一个案例的候选计算，没有第二个模型或案例。
- 官方22M预测器，width512/depth7，公开150k checkpoint；102192971字节，SHA256 `bdd77b212d81e11438fd6bd85525bed72ff1f47a6b3211ac925ddfbc7c8351a6`。22M指预测器，视觉编码器仍是官方共享的冻结V-JEPA2.1-G。
- 仅一个Push案例：target kitchen4、task seed0、18动作。完整240×320、左／腕相机、embodiment3本体输入参考，官方客户端FP16、默认action_repeat2。
- 原版官方client/server，无adapter／记录钩子；CEM总候选1024=8rank×64sample×2batch，15/10轮、top20、H5、execute1。0新SGD。之前单卡16×64每轮约56秒，在实际动作0时停止；旧根日志保留，不算性能失败。
- 2026-11-11 14:24:52UTC完整18步执行结束，0执行错误。最终进度0.1465%，距目标12.7058cm，native2cm final／any均false；物体沿目标方向净位移约0.187mm。本次多卡运行1074.49秒，约17.91分钟。完整数据在results/summary.json，90分钟上限未触及。
- 原始结果在formal下，官方task名会产生嵌套输出目录；用rglob寻找NPZ／视频，不能用平坦result*.npz判断失败。完整回合应有18条env_rewards；技术测试与正式性能分开。
- 这是一个案例，不是论文平均成功率。论文Fig15所选22M150k附近为近零成功区域，定性弱Push相符；单例及未确认相同的二值阈值不足以认定严格数值复现。论文图和当前官方说明分别报告success rate／final reward，不能把它们与2cm完成标志直接等同。
- 依赖沿用兼容环境Python3.12／Torch2.9.1+cu130／MuJoCo3.3.1，与README的Python3.11／Torch2.7.1+cu126不同，未完成逐版本环境复刻。

## 全停已完成

2026-11-11 13:55:20UTC四台实际核对全部无项目／GPU进程后，才启动22M单例。该单例完成后，14:28:01UTC四台再次核对均无本项目实验进程、无GPU计算进程；证明在本轮EXIT-*.json。此前停机证明仍在 `runs/robocasa_official_cohort_20261111/ALL_STOPPED_*.json`。

旧2B四例全部按用户要求停止，正式完成0例，已有日志保留，未完成项不计模型失败。共同根USER_CLOSED.json/HOLD及各根USER_STOP.json/STOP_VERIFIED.json，不恢复旧控制器。

旧会话每小时配置 `runs/robocasa_encoder_budget_20261111/SESSION_CHECKS.json` 已设old_scope_stopped_by_user、next_check_utc=null。旧session-check.py有停止守卫，不能自动恢复检查／实验；旧goal／旧LLM监控仍关闭。当前单例只做有限评测。

## 已封存的最新结论

当前输入inverse轮：六模型到50000、24源回合、0执行错误、HOLD。Control／observed／paired Reach90.98／67.72／68.68%，Push9.31／9.16／9.66%，严格final／any均0。两种inverse未提供可靠操纵提升，不长训。Observed只重建事实转移当前输入a0；paired从预测中也读回同一个a0，没有未来动作target。

40000空间轮六模型48回合与16回合重评分CEM均未解决Push。单卡原版2B技术测试完成18步、官方NPZ／视频、exit0，但仅4候选／2轮，不是性能复现。较早路径配置／记录钩子／输出glob错误均保留，不能作为模型负结果。全部原始证据仍在EFS。

SPT使用自身普通事实转移，不是RoboJEPA完整23数据集混合；同数据／同编码器的本地17h训练对照已完成，参数和FLOPs不等。公开全输入参考和RGB候选分轨，不混作公平算法排名。详见[关键结论](experiments/CONCLUSIONS.md)。本轮无最优配置晋级。
