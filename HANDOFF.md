# 当前交接

长期目标是改进SPT并超过RoboJEPA，尚未达成。**最新用户要求一件件处理：先停止所有实验，然后只看最小模型评测。当前只运行一个官方22M模型的Push案例，不训练、不扩分支，不自动进入下一步。**

## 唯一当前任务

- 根：EFS `/efs/prescottsun/SPT-JEPA/runs/robocasa_22m_eval_20261111`。
- 主机：dev13，GPU0，controller PID873179；真实归属以launch.json和实际argv为准。
- 官方22M预测器，width512/depth7，公开150k checkpoint；102192971字节，SHA256 `bdd77b212d81e11438fd6bd85525bed72ff1f47a6b3211ac925ddfbc7c8351a6`。22M指预测器，视觉编码器仍是官方共享的冻结V-JEPA2.1-G。
- 仅一个Push案例：target kitchen4、task seed0、18动作。完整240×320、左／腕相机、embodiment3本体输入参考，官方客户端FP16、默认action_repeat2。
- 原版官方client/server，无adapter／记录钩子；CEM总候选1024=16sample×64batch，15/10轮、top20、H5、execute1。0新SGD。
- plan.json定义90分钟绝对上限，无自动追加。当前已成功加载模型，正式结果尚未齐。
- 原始结果在formal下，官方task名会产生嵌套输出目录；用rglob寻找NPZ／视频，不能用平坦result*.npz判断失败。完整回合应有18条env_rewards；技术测试与正式性能分开。
- 这是一个案例，不是论文平均成功率。报告最终进度与官方运行时2cm完成标志；论文Fig15二值阈值尚未验证相同。
- 依赖沿用兼容环境Python3.12／Torch2.9.1+cu130／MuJoCo3.3.1，与README的Python3.11／Torch2.7.1+cu126不同，未完成逐版本环境复刻。

## 全停已完成

2026-11-11 13:55:20UTC四台实际核对：dev13/dev15/dev6/dev6-2c全部无本项目实验进程、无GPU计算进程。证明在 `runs/robocasa_official_cohort_20261111/ALL_STOPPED_*.json`。之后才启动上述22M单例。

旧2B四例全部按用户要求停止，正式完成0例，已有日志保留，未完成项不计模型失败。共同根USER_CLOSED.json/HOLD及各根USER_STOP.json/STOP_VERIFIED.json，不恢复旧控制器。

旧会话每小时配置 `runs/robocasa_encoder_budget_20261111/SESSION_CHECKS.json` 已设old_scope_stopped_by_user、next_check_utc=null。旧session-check.py有停止守卫，不能自动恢复检查／实验；旧goal／旧LLM监控仍关闭。当前单例只做有限评测。

## 已封存的最新结论

当前输入inverse轮：六模型到50000、24源回合、0执行错误、HOLD。Control／observed／paired Reach90.98／67.72／68.68%，Push9.31／9.16／9.66%，严格final／any均0。两种inverse未提供可靠操纵提升，不长训。Observed只重建事实转移当前输入a0；paired从预测中也读回同一个a0，没有未来动作target。

40000空间轮六模型48回合与16回合重评分CEM均未解决Push。单卡原版2B技术测试完成18步、官方NPZ／视频、exit0，但仅4候选／2轮，不是性能复现。较早路径配置／记录钩子／输出glob错误均保留，不能作为模型负结果。全部原始证据仍在EFS。

当前SPT训练使用自身普通事实转移，不是RoboJEPA完整23数据集混合；同数据／同编码器／同预算训练对照尚未完成。公开全输入参考和RGB候选分轨，不混作公平算法排名。详见[关键结论](experiments/CONCLUSIONS.md)。本轮无最优配置晋级。
