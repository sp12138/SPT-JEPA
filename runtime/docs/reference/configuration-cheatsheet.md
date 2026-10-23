# 通用 Override 速查

本页只维护跨模型常用配置。模型准备、训练和评估流程在各自
[One Pager](../models/index.md)；字段类型和 schema 默认值见
[配置方式与字段](configuration.md)。

本页不是首次训练的前置阅读。先按模型流程启动；只有需要恢复、周期保存、并行调优或
S3 时再查对应条目。

## 怎么修改

- 临时修改：把本页的 `key=value` 直接加到 CLI 末尾；
- 固定任务配置：把覆盖字段写入 `user/` 目录下的 YAML，通过
  `--config-dir <dir> +user=<name>` 加载；
- 不确定最终值：先加 `--cfg job` 查看，运行后检查
  `${runtime.output_dir}/resolved_config.yaml`。

外部 YAML 的完整写法见[配置方式与字段](configuration.md#custom-config-file)。
框架内部如何合成配置见开发者文档
[配置加载与覆盖](../development/configuration-composition.md)。

## 首次训练与完整续训

公共契约默认：

- `resume.path=none`，首次训练从 step 0 开始；
- `checkpoint.enabled=true`；
- final full DCP，`format=dcp`、`async_save=true`；
- `checkpoint.dirpath` 为空时派生为 `${runtime.output_dir}/checkpoints`；
- `checkpoint.every_n_steps=0` 只关闭周期保存，不影响 final；
- `checkpoint.keep_last=2`。

完整续训只增加恢复选择：

```bash
resume.path=auto
```

`auto` 从派生 checkpoint 目录选择 step 最大且已提交的 checkpoint。保持
weights/optimizer/training state 默认开启；`runtime.max_steps` 是整次训练的总目标，
不是恢复后新增步数。

## Weights-only warm-start

显式指定 PT 或 DCP，并关闭 optimizer 与 training state：

```bash
resume.path=/runs/source/checkpoints/00100.dcp \
resume.optimizer=false \
resume.training_state=false
```

只有结构确实变化时才评估 `runtime.weights_strict=false` 或 `resume.strict=false`；
它们不是常规必填项。

## 周期 checkpoint

长任务在默认 final DCP 上增加：

```bash
checkpoint.every_n_steps=500
```

常用覆盖：

| Override | 用途 |
| --- | --- |
| `checkpoint.keep_last=3` | 调整已提交 checkpoint 保留数 |
| `checkpoint.save_final=false` | 明确不保存 final；通常只用于临时 contract |
| `checkpoint.enabled=false` | CPU/config smoke 不应落盘时关闭持久化 |
| `checkpoint.empty_cache_before_save=false` | 保存前不清 CUDA cache；仅在显存充裕且要省掉 empty_cache 开销时使用 |

完整恢复使用 `full+dcp`。`full+pt` 不支持；PT 只适合权重交付或 warm-start。

## 性能与并行

| Override | 用途与约束 |
| --- | --- |
| `parallel.usp_size=<N>` | Ulysses 宽度；不能与 TP/CP 同时大于 1 |
| `parallel.tp_size=<N>` | Tensor parallel；先确认所用 engine 支持 |
| `parallel.cp_size=<N>` | Megatron context parallel |
| `parallel.reshard_after_forward=false` | 用更多显存换更少通信 |
| `parallel.gradient_checkpointing=true` | activation recompute；多数模型 recipe 已设置 |
| `attention_offload.enable=true attention_offload.async_transfer=true` | FA 算子内 preserved offload（Native preview）；需同时开启 GC 和显式 FA backend |
| `runtime.seed=42` | 固定训练随机流的初始状态；完整续训恢复 RNG 状态 |
| `runtime.deterministic=true` | 统一开启 PyTorch、cuDNN 和 FA4 确定性；不支持 FA3 |

`runtime.autocast`、`runtime.weights_strict` 和 `attention.allow_fallback` 可能由模型
recipe 定义。除归因或兼容加载外，不要把它们机械加入每条命令。

## S3 与环境变量

路径只有命中显式 mount 映射时才被视为 S3：

```yaml
storage:
  s3_mounts:
    - mount_path: /mnt/training
      bucket: training-checkpoints
      region: us-west-2
```

将 `runtime.output_dir` 或显式恢复路径放在该 mount 下。系统不会从目录名或环境变量
猜 bucket。

| 环境变量 | 默认 | 用途 |
| --- | --- | --- |
| `VFORGE_CKPT_LOCAL_ROOT` | `/local-ssd/vforge-checkpoints` | node-local staging 根目录 |
| `VFORGE_STAGE_WORKERS` | `8` | 每节点并发对象数 |
| `VFORGE_S3_TRANSFER_CONCURRENCY` | `32` | 单对象 CRT 并发 |
| `VFORGE_DCP_PLANNER_SELECTIVE` | 关闭 | planner-aware selective staging |

选择性 staging 需要显式开启：

```bash
export VFORGE_DCP_PLANNER_SELECTIVE=1
```

先用默认 full staging 建立正确性，再单独验证 selective 路径。

## 易错组合

- 首次训练设置 `resume.path=auto`：可能恢复同一输出目录的历史任务；
- warm-start 未关闭 `resume.optimizer` 和 `resume.training_state`：会请求不适用的完整状态；
- `checkpoint.state=full checkpoint.format=pt`：不支持；
- `checkpoint.every_n_steps=0 checkpoint.save_final=false`：启用 checkpoint 却没有保存点；
- 只设置 `attention_offload.enable=true`、未开启 gradient checkpointing 或仍走 SDPA：不会产生 preserved offload 收益；
- `usp_size>1` 同时设置 `tp_size>1` 或 `cp_size>1`：启动时拒绝；
- 只改 checkpoint 目录、不改 `runtime.output_dir`：日志、resolved config 与权重分散；
- 把 `model.pretrained_path`、`resume.path` 和 `evaluation.checkpoint_path` 混用：三者
  分别表示基础 snapshot、训练恢复和评估权重。

持久化协议和跨 topology 约束见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。
