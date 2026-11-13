# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。最新用户要求继续改进独立动作头控制、同会话每小时检查，直到验证可靠稳定的scaling，不在有限pilot后自行HOLD。资源只限dev13/dev15/dev6；旧pilot、长训、goal和旧定时监控不恢复。

## 当前持续实验

- 新根`runs/robocasa_stable_scaling_20261114`，核心入口`experiments/robocasa_stable_scaling`，[协议](experiments/robocasa_stable_scaling/PROTOCOL.md)。现各4卡：dev13的0–3卡`spt-residual-flow-growing`、4–7卡`spt-future-mean-growing`；dev15的0–3卡`spt-rate-growing`续训、4–7卡固定双线性读出队列；dev6的0–3卡`spt-control-affine-growing`、4–7卡`spt-deep-observation-growing`，与原八条采集共用设备，非额外八卡。有限反对称头已结束，失败分支不恢复。实际进程/GPU只看EFS的`training/*/{OWNER,controller,progress}.json`、`stream/PROGRESS-*.json`及`SESSION_CHECKS.json`，不从本文推断持续利用率。
- 四组均新随机初始化、seed1、global batch32、window8，E/pool/F/action-encoder初始SHA配对一致；各8步启动和8步完整恢复已实际通过。首版共1238轨迹（原1120 TRAIN加118新事实日志），之后仅增长组接受新版本；96验证轨迹不进body训练。动作按物理范围归一化，共同900秒warmup后逆平方根衰减。这些共同改变及不同参数/吞吐须披露，四配方不是四独立种子。
- 共享冻结V-JEPA2.1-G；新的后果头从零，不是整个视觉骨干从零。body无动作标签，独立线性头始终同17920/1536当前a_t事实标签，表示冻结/detach。每累计训练一小时保存完整状态并评真实闭环；同会话每墙钟小时检查、必要时另冻结版本修复，禁止修改已运行源码。首小时`code-train-v1`、第二小时v2；现v6与已测试v5的训练/读出代码SHA完全一致，另优化不可变本地缓存复用。每卡4个评测worker。v5同步各rank的停止判断，四卡实际11步完整停止后精确恢复8步到19，未发布不完整小时，见`SMOKE-distributed-stop.json`。v2不在训练中触发STOP，均在完整2h后只结束父controller、保留控制子进程完成64例；边界记录`RETIRED-2H-BOUNDARY.json`。采集保留v1八条流水线，另v2扩容56条（最多7168新轨迹、索引1000000起、stride56），共64条/八卡；归属`stream/OWNER-throughput.json`，不覆盖原记录。
- 主监测集预定64案例：厨房1/4/8/10×39301至39316；反复查看后属于开发集。另32确认案例预定厨房2/3/5/6×49301至49308，尚未打开控制评测。回报、距离和原2mm final/any分别报告；不把单点、loss下降或最佳点选择叫稳定scaling。结果入口`results/<run>/hour-NNN.json`；完成有限诊断不代表当前持续目标完成。
- 已完成诊断：非线性势函数差分动作头在全部8个旧时间点验证误差改善，却没有闭环收益，40个已见案例8h回报84.76%、18h76.46%，严格成功均0/40，不采用。1h曾因FP32/FP64核对精度守卫失败，另冻结`code-readout-v2`只修匹配算术后重试，原错误保留。旧F的事实验证实际动作MSE/copy在1/8/18h为.334/.205/.263，零动作约.998/.979/.971；支持预测本身也平台，非因果反事实或控制结果。原始在新根`readout-pilot`及`predictive-diagnostic`。
- 新64例对照：旧8h/18h线性头80.19/79.12%、48.02/52.88mm。新SPT固定1→2h回报78.50→79.21%、末端48.84→46.28mm；增长组79.04→80.08%、49.49→43.83mm。线性特征目标固定70.29→61.46%、增长68.33→65.87%，均退化。全部0/64。2h增长组2977轨迹（原1120加1857新日志）；收益小、单训练seed，未证稳定scaling。三弱臂完整收尾，不自动恢复；SPT增长组保留作参照。结果只报同案例，不与旧32例混比。
- 具体瓶颈：当前线性特征解码范围即使使用最优oracle潜码，最小L2仍为copy的1.135–1.146倍；不向训练传oracle码/基。源码在`diagnostics.py`，原始`decoder-span-diagnostic`。F隐式Newton头16例67.24%对同模型线性81.63%、仅2/16改善，不采用；完整原型只归档`code-implicit-v1`及`implicit-readout-pilot`。
- 新三配方从v6新随机状态启动，实际入口8步和完整恢复8步通过；首1h结果与后续v7修复见下。参照续训3h语料为6444轨迹，新三者仍从同首版1238开始、按同版本序列增长。v3/v4的12个启动更新及v5停止恢复的19个更新均丢弃，不继承测试权重；joint训练/读出及旧loss回归逐位一致。停止测试首次仅因dev6缺少118新日志本地缓存而在任何完整更新前退出，补齐后另尝试通过；原错误保留。算法、假设及参数差异见持续协议。
- 三结构方案1h的FP32读出：条件预测70.66%/68.60mm，单视图传输55.31%/122.61mm，joint传输47.03%/145.95mm，均0/64 final/any。首1h为v6训练；只读12轨迹发现传输编码RMS约145/669，71%/95%动作增量坐标被BF16舍掉，动作能量仅占观测编码变化的.0011/.0091。v7在完整1h后精确恢复，保持小编码器/加法FP32，不改参数/损失/优化器；0更新增量误差2.19e-9、编码逐位一致。2h仍仅69.57/55.02/56.00%、70.57/119.57/115.60mm，全部0；不继续弱配方，保留全部状态，未伪发3h成绩。
- 所有新读出显式FP32；旧BF16单列。参考根`runs/robocasa_fp32_reference_20261114`已完成：同64例SPT增长1/2/3/4h回报78.98/81.12/80.47/81.34%，末端52.78/42.21/43.20/40.90mm，final全0、any0/1/1/0；3/4h在主根。旧8/18h80.18/79.23%、48.19/52.11mm，全0。仍未证持续scaling，不用精度切换冒充收益。
- `runs/robocasa_codec_short_20261114`两个新随机有限原型各1GPU/10000更新已完成。保留原SPT噪声/编码率，仅换自然未来特征L1：通用/加法F回报46.95/62.59%，末端159.39/111.58mm，均0/16；同案例SPT1/2h80.99/80.93%。短测不是同预算比较，但无推广依据。完整状态保留，权重不进主线，DONE不结束持续改进目标。
- 编码一致性只读检查：8普通轨迹（两个初始物理分支组），batch2/8/34与原缓存逐位一致，未发现该处偏移。`runs/robocasa_spatial_readout_20261114`的同标签4×4/14×14完整空间线性头16例62.29/54.43%，109.05/133.92mm，全0；参数显著多于64D头，未推广。原型仅归档EFS，不把固定读出失败等同视觉无信息。
- `runs/robocasa_likelihood_readout_20261114`六点TRAIN-only响应校准已完成：线性Gaussian假设下用(WB)^-1 W移除动作先验收缩。验证平均响应近单位阵，但同16例新1/2/3/4h仅57.37/63.11/63.02/62.24%，旧8/18h70.41/65.18%，全0，均差于原头，不采用。与空间头的完整原型仅保留EFS及仓库外归档，不为失败短测增加Git细节文件。
- 新v8只改原SPT像素distortion：`residual-flow`预测相对已观测首帧的RGB变化；`future-mean`用恒零renderer输入直接预测未来RGB条件均值。均保留原Gaussian码/forecast/rate及同64例线性头，原SPT参照语义不变。两组完整初始NN SHA与SPT参照一致，8步入口及8步/16步真实完整恢复已通过，所有更新计入预算。
- 有限读出`runs/robocasa_differential_readout_20261114`：同16例双线性1→4h为81.59→85.20%、39.43→29.74mm；原线性80.99→82.25%、45.27→37.81mm。共享二次势差头77.62→75.68%、49.39→52.21mm，均0成功。仅两个开发点、不同容量，未证稳定scaling或本质非可积性。核心`differential_readout.py`仅固定标签读出，body冻结。
- 新根`runs/robocasa_differential_scaling_20261114`固定双线性方法，dev15 4–7卡评入口8更新及各完整小时的同64开发例；覆盖参照及两像素body，固定队列无算法选择或第二套LLM监控。1/4h原16例仅在模型/头SHA一致后复用，`REUSE-PREFLIGHT.json`通过。扩至64例后入口27.76%，1/2/3/4h80.46/79.60/78.93/79.27%，末端45.01/48.67/47.23/45.40mm，全部0；原16例正结果未推广。近等容量反对称头16例1/4h77.24/66.93%，54.68/84.71mm，全0；不能宣称非可积性收益。32确认例继续未打开。
- 原SPT线性头5h81.45%、39.29mm，全0/64。v9在dev6新随机检验两瓶颈：局部`z+drift(z)+G(z)A*a`替代通用历史F（3588万参数），或仅加两层192宽帧内注意力（4681万）。两者新E/action运算FP32，原code/noise/rate/像素flow不变；共享初始投影及pool SHA，8步及8/16步完整恢复通过。共用采集会影响吞吐，不冒称等FLOPs/独占GPU预算。实际占用归属检查在`cohosting-startup.json`，读出数值验证在`readout-startup-v9`。
- 第一轮扩容56×128有效日志已完成；第二轮同策略56×128已启动，索引2000000起、stride56，归属`stream/OWNER-throughput2.json`。原八条采集继续，日志不覆盖，原始错误保留；21:13实际检查11449条新日志，非持续利用率证明。

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
