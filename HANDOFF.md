# 当前交接

最新用户授权 RoboCasa 训练厨房／留出厨房的三路线对照，以超过 RoboJEPA 为目标。协议见 [robocasa_compare/PROTOCOL.md](experiments/robocasa_compare/PROTOCOL.md)。dev13＝temporal-forecast，dev15＝空间表示，dev6＝长时域预测；每条两个随机初始化种子，各四卡 DDP。已获取六个官方公开原子任务、642条轨迹；训练／验证厨房分组隔离，十个测试厨房未进入训练。六模型恢复／反传、40个物理案例及官方50M／2B短闭环入口已通过。有限队列为500步诊断→5000步训练→共同案例完整CEM对照；尚无成熟性能突破。原始状态入口为 `/efs/prescottsun/SPT-JEPA/runs/robocasa_compare_20261110/pipeline/`，最终收集为 `results/summary.json`。旧实验、goal 和定时监控不恢复。

源训练状态重放发现正方形训练视场与宽画面评测不匹配，已统一候选模型的中央正方形裁剪。500步评测在修复期间启动，已取消，不能用于路线排名或scaling结论；六个500步检查点直接续训，5000步使用统一修复。原始记录和取消说明全部保留，未重复已完成SGD。另已定位公开手工proprio辅助函数的0.7m原点／四元数符号差异；仅带状态的外部参考改用原生传感器约定，候选保持RGB＋事实控制输入。官方参考有额外预训练与数据，成绩差距不等于同数据从零算法排名。

2026-11-09 最新 causal-inverse 研究已结束：8 个模型各新增 10,000 步，384 个控制评测回合，无执行错误。六项困难 manipulation 在每种方法、每种控制器下均为 0/12；尚无可靠性能突破。每组复用六个物理案例及两个原训练父模型，不能把总回合数当独立案例数。

- [核心结论](experiments/CONCLUSIONS.md)
- [较优配置与适用边界](experiments/best-configs.json)
- [原始记录及清理归档](docs/experiment-archives.md)
- [项目约定](AGENTS.md)

源码保留在 `as_jepa_lewm/`、`as_jepa_tools/`、`jobs/`、`runtime/` 和各 `experiments/` 方法目录。历史实验脚本可能需要从归档恢复配置／数据才能重放；不据旧运行状态自动启动。
