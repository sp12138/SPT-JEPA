# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。**最新用户授权的1B四机测速与单例完整验证已完成，四机释放，等待下一步；不自动训练、增加案例或恢复旧监控。**

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
