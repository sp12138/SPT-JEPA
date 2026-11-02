# Lingjing 提案的完整问题覆盖

最新性能见 [2026-11-03新方案与旧TF比较](PERFORMANCE_20261103.md)，上一轮复核见 [2026-11-02新结果与决定](REVIEW_20261102_1229.md)，历史结果见 [RESULTS.md](RESULTS.md)，逐提案覆盖与条件路线见 [COVERAGE.md](COVERAGE.md)，机器与完整状态恢复入口见 [RECOVERY.json](RECOVERY.json)。

主要结论：TwoRoom所有主要配方未通过2000步状态门槛，β0等额外诊断也无效，按规则结束该部分；Cube12个配方已完成12000步/48个规划终点，均无延长优势，停止同配置23342续训。PushT原L0三seed10epoch已完成，raw primary/fresh均值90.89/87.33%。总体信息量、数值尺度或低 latent factual MSE 均不能单独保证任务状态/控制有效。

所有小型数值、绘图和来源随Git保存；全检查点、媒体、详细运行日志与共享数据保留EFS。主训练NN源固定0196dc17，原审计helper e9281dd的日志末行误报由d1f5ed1修复，当前I/O与后续派发helper c058745；NN源不变。无最优动作标签，物理探针标签仅用于诊断。135主要配方的存在不等于名义预算全部完成；early-stop和实现错误分别记录。
