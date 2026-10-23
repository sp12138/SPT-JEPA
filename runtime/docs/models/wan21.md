# Wan 2.1 One Pager

!!! success "稳定"
    `experiment=wan21_i2v_14b` 覆盖 prepare、train、完整 DCP resume、PT/DCP eval、
    DCP→PT 和单机 USP8 acceptance。

## 准备

要求 Linux、NVIDIA CUDA 和足够的 GPU 显存；本页训练命令使用已验收的单机 8 GPU
USP8 topology。安装模型依赖：

```bash
pip install -e '.[wan]'
```

!!! note "本页的 `--standalone`"
    Train、Resume 和 Warm-start 命令用 `--standalone`，只因为当前公开验收是单机
    USP8，不表示 vforge 只能单机。跨机 launcher 参数和两机 16 卡模板见
    [分布式运行](../features/distributed-training.md#launch-topology)；Wan21 Native
    2×8 仍需独立验收。

准备 Wan 2.1 I2V 14B Diffusers snapshot、训练 JSONL 和所有节点可见的 cache 目录。
最小 manifest 每行可写为：

```json
{"sample_id":"sample-000001","video":"/data/videos/000001.mp4","prompt":"a car driving"}
```

`video_path` 可替代 `video`，`text` 可替代 `prompt`。先生成离线 cache：

```bash
vforge-prepare experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache
```

Prepare 入口自动使用 `latent_cache.mode=online`；训练 recipe 读取 offline cache。

## 首次训练

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.usp_size=8 \
  runtime.max_steps=100
```

首次运行不需要设置 `resume.*`、`checkpoint.*` 或 checkpoint 目录。公共契约从 step 0
开始，训练结束异步保存 full DCP 到 `${runtime.output_dir}/checkpoints`；默认只保留
最近 2 个已提交 checkpoint。长任务增加周期容灾：

```bash
checkpoint.every_n_steps=500
```

## 产物

默认 train 输出目录包含：

```text
outputs/train/
  resolved_config.yaml
  resolved_config.hash
  train_steps.jsonl
  checkpoints/<step>.dcp/
```

DCP 目录包含模型、optimizer 和 trainer state；异步保存会在训练退出前等待提交完成。

## 完整续训 {#wan21-resume}

保持模型、数据、cache、topology、`runtime.output_dir` 和自定义
`checkpoint.dirpath` 不变，提高总目标步数，并只增加 `resume.path=auto`：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.usp_size=8 \
  runtime.max_steps=200 \
  resume.path=auto
```

`runtime.max_steps=200` 表示训练到总计 200 step。`auto` 从派生 checkpoint 目录选择
step 最大且已完整提交的 checkpoint，并恢复 weights、optimizer、step、scheduler、
dataloader 和 RNG。

## Weights-only warm-start

从其它任务显式加载 PT 或 DCP 权重，不继承 optimizer 和训练进度：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.usp_size=8 \
  runtime.max_steps=100 \
  resume.path=/runs/source/checkpoints/00100.dcp \
  resume.optimizer=false \
  resume.training_state=false
```

结构变化时再评估是否需要 non-strict 加载；不要把它作为常规必填参数。

## 评估

Condition-only 推理请求可只提供首帧、prompt 和配置启用的条件流。例如 normal
condition：

```json
{"sample_id":"infer-000001","image":"/data/images/000001.png","normal":"/data/normals/000001.mp4","prompt":"a car driving"}
```

```bash
vforge-eval experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  evaluation.data_source=inference \
  evaluation.manifest=/data/wan21/infer.jsonl \
  evaluation.checkpoint_path=outputs/train/checkpoints/00100.dcp \
  evaluation.num_samples=1 \
  evaluation.seed=42
```

有目标视频的 standalone eval 使用 `evaluation.data_source=validation`，manifest
记录与训练数据一致。外部 latent cache 使用 `evaluation.data_source=cache`，并把
`evaluation.manifest` 直接指向 cache 根目录下兼容的 `manifest.json`。在线 Eval
只编码首帧、文本与 normal/grid 等条件，不再编码目标 RGB `frame_latents`。

默认产物写入 `outputs/eval/samples`（即 `${runtime.output_dir}/samples`），包括
MP4、逐样本 sidecar 和 `eval_manifest.json`。Eval 只加载模型权重，DCP 会按当前
评估 topology 重分片，不要求沿用训练时的 USP8。使用任一 manifest 时 Eval 在线编码
条件，不要求提前 Prepare。`data.val_manifest` 和 `data.cache_dir` 不会作为 standalone
eval 输入。

## DCP 转 PT

需要单文件交付时再转换：

```bash
vforge-convert-checkpoint experiment=wan21_i2v_14b \
  convert_checkpoint.input_path=outputs/train/checkpoints/00100.dcp \
  convert_checkpoint.output_path=/runs/wan21/wan21-step100.pt
```

PT 只含模型权重，不能用于完整训练状态恢复。

## 维护者验收（可选）

USP8 表示单节点启动 8 个进程，并把 Ulysses sequence parallel size 设为 8。
`scripts/acceptance_wan21_sp8.sh` 是维护者使用的一次性回归脚本，不是另一篇文档，也
不是普通用户启动训练的必经步骤。它在一次 8 GPU allocation 中检查 prepare/train、
full DCP 保存恢复、DCP→PT 和 eval 闭环。`TRAIN_MANIFEST` 必须与 `CACHE_DIR` 中的
离线 cache 索引一致：

```bash
CACHE_DIR=/data/wan21/cache \
PRETRAINED_PATH=/models/Wan2.1-I2V-14B \
TRAIN_MANIFEST=/data/wan21/train.jsonl \
  bash scripts/acceptance_wan21_sp8.sh
```

脚本中的 eval 使用 USP8 和离线 cache，是固定 gate 路径；普通用户仍优先使用上面的
单进程 `vforge-eval` 与 `evaluation.data_source=validation` 在线评估，不需要复刻 gate
topology。

## 支持边界

- 公开稳定路径是单机 FSDP2/DP/Ulysses、冻结条件 encoder 和
  `gradient_accumulation_steps=1`；
- Native transformer 声明的 Diffusers `_keep_in_fp32_modules`（`time_embedder` /
  `norm1` / `norm2` / `norm3`）把这些模块留在 fp32；`time_proj` 跟 Diffusers 一样走 mixed policy
  （`temb.type_as` 之后）。block 入口不预转 `temb` / RoPE。AdaLN 仍用
  `temb.float()`。`scale_shift_table` 是单个权重而不是模块，落盘保持 fp32，
  前反向跟随 `param_dtype`；
- self-attention Ulysses 已覆盖，cross-attention Ulysses、EP 和 ring attention
  未公开支持；
- Megatron TP/CP 是独立实验路径，不等同于 Wan21 USP8 acceptance；
- cache 必须与模型、conditioning stream 和预处理参数一致；
- 完整恢复要求同 topology 和相同数据 contract；换 topology 使用 weights-only
  warm-start。

通用字段和远端存储见[通用 Override 速查](../reference/configuration-cheatsheet.md)和
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。
