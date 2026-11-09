# 当前交接

研究已按用户要求暂停。当前仅整理 Git，不恢复实验、goal 或定时监控。

2026-11-09 最新 causal-inverse 研究已结束：8 个模型各新增 10,000 步，384 个控制评测回合，无执行错误。六项困难 manipulation 在每种方法、每种控制器下均为 0/12；尚无可靠性能突破。每组复用六个物理案例及两个原训练父模型，不能把总回合数当独立案例数。

- [核心结论](experiments/CONCLUSIONS.md)
- [较优配置与适用边界](experiments/best-configs.json)
- [原始记录及清理归档](docs/experiment-archives.md)
- [项目约定](AGENTS.md)

源码保留在 `as_jepa_lewm/`、`as_jepa_tools/`、`jobs/`、`runtime/` 和各 `experiments/` 方法目录。历史实验脚本可能需要从归档恢复配置／数据才能重放；不据旧运行状态自动启动。
