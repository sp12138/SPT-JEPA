# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。**最新用户要求一件件处理：先全停，再看最小模型评测。本步已完成；四台已停机，没有训练或自动后续，等待下一步。**

## 本步结果

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
