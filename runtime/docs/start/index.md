# Quick Start：用 Wan 2.1 跑通一次

下面用稳定支持的 Wan 2.1 跑通一次 Native 最小任务：
**安装 → Prepare → Train → 检查产物**。跑通后再按第五节进入续训、评估或其它模型。
先把示例路径替换成真实路径。

## 1. 安装

要求 Linux、Python 3.10 及以上和 NVIDIA CUDA 环境。以下命令在 vforge 仓库根目录
执行；本文 Train 使用已验收的单机 8 GPU USP8 topology，不是 CPU 或笔记本流程：

```bash
cd /path/to/vforge
pip install -e '.[wan]'
```

访问 S3/OSS 权重或 checkpoint 时安装 `.[wan,remote]`。

### 启动入口 {#launch-entry}

普通单进程任务使用 `vforge-*`；模型需要多个进程时使用
`torchrun -m vforge.tasks.<verb>`。GPU 数量由 launcher 或运行平台决定，
`parallel.*_size` 只切分已有 rank。`--standalone` 仅表示单节点；用卡数、多节点和
并行维度的关系见[分布式运行](../features/distributed-training.md#launch-topology)。
下面的 Wan21 Train 使用 `torchrun`，因为当前验收 topology 是单机 8 GPU USP8。

## 2. Prepare

准备基础模型 snapshot、训练 JSONL 和所有训练进程可见的 cache 目录。最小 JSONL
每行包含视频路径和 prompt，`sample_id` 可省略：

```json
{"sample_id":"sample-000001","video":"/data/videos/000001.mp4","prompt":"a car driving"}
```

将它保存为 `/data/wan21/train.jsonl`，再生成离线 cache：

```bash
vforge-prepare experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache
```

本节用默认的单进程、单 GPU Prepare；大数据集可按样本扩展到多卡或多机，命令和共享
目录要求见[分布式运行](../features/distributed-training.md#distributed-prepare)。
Prepare 在线运行 VAE/文本等 encoder；Train 读取生成的离线 cache。两步必须使用相同
模型版本和预处理参数。Prepare 的 resolved config 和日志默认写入
`outputs/prepare/`。

## 3. Train

用单机 8 GPU 跑一个最小 step，确认模型、数据和 kernel 可以工作。这里使用
`--standalone` 是因为本节复现单机验收 topology，不是 vforge 的多机限制：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.usp_size=8 \
  runtime.max_steps=1
```

首次训练无需配置 resume 或 checkpoint：默认从 step 0 开始，并在最终 step 异步保存
full DCP。正式任务只需提高 `runtime.max_steps`；长任务再增加
`checkpoint.every_n_steps=<N>`。完整保存语义见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

## 4. 检查产物

默认训练目录：

```text
outputs/train/
  resolved_config.yaml
  resolved_config.hash
  train_steps.jsonl
  checkpoints/00001.dcp/
```

先确认 `train_steps.jsonl` 有完整 step，checkpoint 目录已提交且没有
`*.dcp.incomplete`。
排障时以 `resolved_config.yaml` 为最终配置。

## 5. 跑通之后

- 完整续训：保持输入、topology、`runtime.output_dir` 和自定义
  `checkpoint.dirpath` 不变，提高总目标步数，再增加 `resume.path=auto`；完整命令见
  [Wan 2.1 One Pager：完整续训](../models/wan21.md#wan21-resume)；
- 评估或导出：按 [Wan 2.1 One Pager](../models/wan21.md) 的 Eval、DCP→PT 命令执行；
- 换模型：进入[支持模型](../models/index.md)，复制对应模型 One Pager；
- 修改一两个字段：查[通用 Override 速查](../reference/configuration-cheatsheet.md)；
- 保存长期配置：使用
  [自己的 YAML](../reference/configuration.md#custom-config-file)；
- 遇到启动或恢复问题：查看[故障排查](../troubleshooting/index.md)。
