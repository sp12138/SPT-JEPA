# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。**最新授权四机同时间训练对比，持续至用户叫停，每累计训练一小时评测。三个SPT配方必须来自历史较好规划记录，不跑三个seed。旧实验、goal与旧LLM监控不恢复。**

## 当前Reach同时间比较

- EFS根`/efs/prescottsun/SPT-JEPA/runs/robocasa_time_compare_20261112`；核心入口`experiments/robocasa_time_compare`，协议见该目录`PROTOCOL.md`。以实际`launch-long-{host}-*.json`、各run的`progress.json`和`execution-long-*/ERROR.json`确认运行状态，不凭本文推断GPU占用。
- dev13=`robojepa-seed1`，公开309M参数架构的RGB-only K2适配；dev15=`spt-seed1`，历史global16/256D＋初态空间上下文；dev6=`spt64-seed1`，原64D analytic temporal-forecast/无场景bridge；dev6-2c=`spt-grid16-seed1`，256D固定空间槽位＋上下文。四算法均seed1，不是独立种子重复。历史Reach回报约91.88/86.10/84.35%，不保证新头复现；grid16的旧support代价91.83%单列，本轮统一uniform主要代价。
- 新预测器随机初始化，共享冻结V-JEPA2.1-G；不继承历史高分头或公开动力学。普通随机命令日志1120train/96validation，厨房分离；四机本地1216记录实际SHA清单完全相同，manifest`ced8adcea7896427db758ae9bade7423e5e4b8434a91a5fb973d135b6b2d4e68`。只用RGB/发出并执行的命令，无proprio/reward/task/动作标签；候选sample-level、无inverse。
- 每模型8张H200、globalbatch32；SPT窗口8、基线16，不等参数/图像FLOPs。共同预缓存特征成本排除；累计最慢rank每次数据读取/传输/前后向/优化器迭代墙钟，排除初始化/存盘/评测，另外报告进程墙钟。用户把24h建议改为无限期；900秒warmup后恒定LR，无任意终点cosine。检查点包含model/Adam/时钟调度/8rank RNG/数据游标。
- 每小时八个预固定target案例：厨房1/4/8/10×taskseed19301/19302，实际8个goal/初态物理与RGB检查通过。公开CEM1024候选/H1/15后10轮/top20/context2/execute1/5动作/repeat1；方形224共同输入，uniform latent L1。每卡一个案例并行。报告平均归一化回报、最终mm距离及2mm完成数；不是23数据集完整论文复现，也不把回报当成功率。
- 四run实际入口均PASS：各保留8个真实更新，必要正梯度/有限loss、末尾动作污染不影响loss、缓存重编码一致、真实两动作CEM。RoboJEPA公开K2与本实现FP32 loss误差0。原训练源码`code-v4`和全部旧错误保留。dev13/dev15持续归属`launch-long-{host}-1.json`；dev6的attempt1在约9.57h遇BF16非有限梯度，attempt2从204871步/9.5h的完整model/Adam/scheduler/8rankRNG/cursor恢复，已实际到205120步/34254秒、有限loss/必要正梯度/8卡计算。独立`code-v5`增加同一失败batch/RNG的FP32重算，无丢样本或更改loss/LR/optimizer；此快照尚未触发FP32分支，不能称已复现并消除数值根因。至少1509个未保存更新会重放，确切数量未知；原完整状态另存`recovery-dev6-numerics/saved-204871.ckpt`。dev6-2c SSH35779连续关闭且心跳停在约7.56h，保存到144117步/7.5h；不能声称当前仍跑或已停止，已询问新入口，不自动迁移到其他机器。
- 用户两次细化方向前，两个额外seed及两个新假设各做8步后封存，合计32个启动更新，分别在`superseded-seed-startup`和`superseded-hypothesis-startup`，不计入当前算法祖先或性能。当前只保留基线/global16的原8步，其余两方法重新随机初始化；不是把旧hypothesis续成历史配方。
- 逐小时结果`results/{run}/hour-NN.json`；`results/latest.json`只比较四方法均完成的同一训练小时，并保留各自最新小时。当前共同7h：基线/global16/64D/grid16回报19.33/70.68/74.65/69.45%，最终距离243.5/70.1/56.9/61.5mm，final/any2mm全部0/8；32条实际trace SHA/物理/RGB/NN不变核对通过。当前选择最高64D为2h79.65%，不是未选择确认或成功率。另一个完整配对12h：global16为71.97%、73.9mm，基线为14.36%、271.3mm，均0/8；64D9h71.66%，grid16仅7h，不同小时不配对混比。曲线未显示持续scaling：基线TF/AR损失下降同时回报下降，不能证明唯一原因；SPT早期达到较高回报后波动、严格成功仍0。原始本次审查`reviews/status-20261112/summary.json`。新native worker负责训练/评测，无第二套LLM定时监控；当前资源状态必须查实际进程，不沿用旧32卡满载说法。

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

当前SPT训练使用自身普通事实转移，不是RoboJEPA完整23数据集混合；同数据／同编码器／同预算训练对照尚未完成。公开全输入参考和RGB候选分轨，不混作公平算法排名。详见[关键结论](experiments/CONCLUSIONS.md)。本轮无最优配置晋级。
