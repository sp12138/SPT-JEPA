# 故障排查

本页只在任务失败时按症状查询，不是 Quick Start 的前置阅读。配置问题优先查看运行
目录中的 `resolved_config.yaml`，不要只根据命令行推断最终值。

## 启动和配置

| 症状 | 先检查 |
| --- | --- |
| 首次训练却恢复了旧任务 | 检查 resolved config；公共默认应为 `resume.path=none` |
| 配置覆盖没有生效 | 检查 experiment、task 和 CLI 的覆盖顺序 |
| 多卡命令只启动一个进程 | 使用 `torchrun -m vforge.tasks.<verb>` |
| world size 与并行维度不匹配 | 检查 `dp × usp × cp × tp = WORLD_SIZE` |
| 多节点启动后 collective 卡住 | 检查各节点的 rendezvous、`NODE_RANK`、进程数和共享路径 |
| 输出目录与预期不同 | 检查 `runtime.output_dir` 及其派生的 checkpoint/samples 目录 |

相关入口：[通用 Override 速查](../reference/configuration-cheatsheet.md)和
[分布式运行](../features/distributed-training.md)。

## 显存和 Attention

| 症状 | 先检查 |
| --- | --- |
| 开了 offload 但指标为 0 | 同时确认 gradient checkpointing、Flash backend，以及 model root 是否实现 `gradient_checkpointing_enable()` |
| 缺少 FA3/FA4 kernel | 检查 `attention.backend` 与 `attention.allow_fallback` |
| FSDP2 recompute dtype 不一致 | 查看下面的专项排障页 |
| 仍然 OOM | 对比 recompute、offload、FSDP reshard 和输入 shape |

- [激活重算与 Offload](../features/activation-recompute-and-offload.md)
- [FSDP2 AC 混合精度](fsdp2-activation-checkpoint-mixed-precision-bug.md)

## 数值和精度

| 症状 | 先检查 |
| --- | --- |
| 换 topology 后结果不再逐位一致 | RNG 状态、确定性/TF32 设置、collective 和 GEMM 顺序 |
| BF16 或 autocast 行为与预期不同 | resolved config 中的 `parallel.mixed_precision` 与 `runtime.autocast` |
| strict 模式启动失败 | 当前 backend/kernel 是否满足确定性要求，不要直接退回 performance 掩盖问题 |

执行模式、mixed precision 和 autocast 的边界见
[随机性与精度](../features/execution-and-precision.md)。

## Checkpoint 和恢复

| 症状 | 先检查 |
| --- | --- |
| 找不到自动恢复目标 | `checkpoint.dirpath`、`latest` 和合法 manifest |
| 目录以 `.dcp.incomplete` 结尾 | 该目录未提交，不可恢复 |
| PT 不能恢复 optimizer | PT 是 weights-only；完整恢复使用 full DCP |
| 远端 staging 很慢 | 先检查 S3 mount 映射和 CRT；selective 默认关闭，只在正确性验证后显式 opt-in |

完整协议见 [Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

## 收集最小证据

提交问题时至少附带：

- `resolved_config.yaml`；
- 模型 experiment、入口命令和 world topology；
- Python、PyTorch、GPU 和 attention backend；
- 首个完整异常栈；
- 是否从 checkpoint 恢复，以及 checkpoint schema/world size；
- 若涉及 offload，附 `offload/packed_tensors`、CPU/GPU 峰值和 step time。
