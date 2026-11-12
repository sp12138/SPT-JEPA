# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。最新用户在长训平台后建议：后果学习与stop-gradient独立动作头分开。有限pilot已完成三种读出、24回合，回报与速度有改善，严格2mm完成仍全0。资源限dev13/dev15/dev6；dev6-2c及grid16只保留历史记录，旧goal和旧LLM监控不恢复。

## 当前独立动作读出

- EFS根`/efs/prescottsun/SPT-JEPA/runs/robocasa_action_probe_20261113`，核心入口`experiments/robocasa_action_probe/probe.py`，协议见同目录`PROTOCOL.md`。最终`results/summary.json`、`paired-comparison.json`、`DONE.json`和`HOLD`；24条trace校验和、物理/RGB重放、5动作、模型不变与实际stopgrad backward均通过。有限pilot已结束，不自动长训或搜索。
- 共享冻结V-JEPA2.1-G（约1.845B）；不是本轮训练大backbone。global16/64D池化表示分别继承同数据17h的321983/326088步父模型；不继承公开动力学或策略。视觉对照只取V-JEPA patch均值，不调用RoboJEPA预测器，不能称为RoboJEPA策略复现。global16在dev15、64D在dev6；视觉对照改由已完成的dev15顺序执行，dev13待机probe在无子进程时先停止，重定位记录保留，无重复实验。
- 同一普通随机命令日志1120train/96validation，17920/1536真实转移、厨房分离。仅以同一次实际转移的输入a_t前三维为标签，输入两个真实RGB编码之差。表示冻结/detach；动作监督不进入后果学习器，没有a_(t+1)、最优动作、专家路径、reward/state/task。不是整个链条都没有动作标签。
- 线性头`W(z_next-z_current)`、无截距，sample MSE+L2；固定ridge候选1e-6/1e-4/1e-2/1，仅按验证厨房RMSE选择，三头均选.01。SPT64/global16/视觉验证RMSE .07089/.07098/.07462，零预测.10035；头拟合成本另计，不能与旧无动作监督CEM作同目标训练的公平排名。
- 相同8个Reach开发案例（厨房1/4/8/10×19301/19302）、224方形两视角、5动作、norm上限.3、2mm规则。推理将真实下一帧替换为目标RGB，直接出动作，不调用CEM。17h global16的CEM→头回报71.99→77.55%，最终88.47→59.57mm；64D为73.56→83.50%、63.59→41.12mm。逐案例回报4/8、6/8改善；final/any全0/8。视觉头回报68.54%、80.96mm，也全0/8。三个模型重复同8个物理案例，不是24独立案例或未选择确认集。
- 每回合实测均值含编码/仿真/记录、不含启动：global16 50.80→11.61秒（约4.38×），64D42.16→11.78秒（约3.58×）；头本身不能代替整回合计时。新头编码/读出约1.08/1.10秒，另有物理执行成本。中位回合约7.37/7.64秒，一例约42秒；不是所有案例均约7秒。
- 有限结论：独立动作读出有用，原CEM成绩不足以代表表示可用于控制的上限。仍有远目标分布变化、末端精度与反复修正；未证明任何一个是唯一瓶颈，未解决manipulation或超过RoboJEPA论文。
- 旧无限期controller已暂停后交接：dev15/dev6在当前完整阶段结束后退役，保留21h/18h完整状态；dev13仍让已开始的21h子阶段保存退出，由`retire-only-dev13.py`有限交接，之后不启动新实验。实际归属在`boundary-*`、`OLD_STAGE_RETIRED-*`及`launch-retire-only-dev13.json`。不要把pilot完成说成dev13已全部停止；需检查该退出记录和真实进程。本轮末次实际检查dev15/dev6无项目进程、GPU计算为空；dev13当前阶段8卡仍在计算，不是三机满载。检查GPU后再报告新占用。

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
