# MiniMax H3 One Pager

!!! warning "预览"
    H3 FL2VA 已通过 1×4 GB300 Native train、full DCP resume 和 eval gate。当前只支持
    `variant=fl2va`、`data.batch_size=1`；接口和验证范围仍可能调整。

!!! tip "首次使用先准备数据"
    先按 [H3 数据与 Cache](minimax-h3-data.md)确认媒体、帧数、画布和 manifest，再回到
    本页执行 Prepare、Train 和 Eval。

## 已验收版本

| 组件 | 固定版本 |
| --- | --- |
| 模型 | `MiniMaxAI/MiniMax-H3` |
| 模型版本（commit） | `9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6` |
| Diffusers baseline | `f53d552036a0d1bd5570782a39cd40cfabf112bc` |

VForge 使用独立 Native H3 Transformer；固定 Diffusers pipeline 只作为数值基线。

## 准备

```bash
pip install -e '.[h3]'
```

!!! note "本页的 `--standalone`"
    Train、Resume、Warm-start 和 Eval 命令用 `--standalone`，只因为 H3 当前验收是
    单节点 4×GB300。通用多节点 launcher 模板见
    [分布式运行](../features/distributed-training.md#launch-topology)；不能据此推导
    H3 已支持多节点。

访问 S3/OSS 权重或 checkpoint 时安装 `.[h3,remote]`。`h3` extra 不安装 CUDA
attention 扩展；运行镜像必须已经提供与 GPU 匹配的 FA3 或 FA4。Recipe 禁止 fallback，
缺少 kernel 会在启动阶段失败。

模型 snapshot 至少包含 transformer、video/audio VAE、text encoder、tokenizer 和
scheduler。`model.pretrained_path` 必须指向上表模型版本
`9ac0dd7aabc2c651fcf0ace4c00b2bffd9c8c8a6` 对应的 snapshot；不要让路径内容与
`model.extras.model_revision` 指纹不一致。

首次准备按以下顺序进行：

1. 下载上面的已验收模型版本；
2. 按 [H3 数据与 Cache](minimax-h3-data.md) 准备媒体、manifest、帧数和画布；
3. 回到本页执行下面的 Prepare 命令。

生成 offline cache：

```bash
vforge-prepare experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache
```

先用单样本检查媒体解码和 snapshot 完整性，再扩大 manifest。

## USP4 首次训练 {#usp4-first-train}

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache \
  parallel.usp_size=4 \
  runtime.max_steps=100
```

首次运行不设置 `resume.*`、`checkpoint.*` 或 checkpoint 目录。公共契约在最终 step
异步保存 full DCP；长任务只增加 `checkpoint.every_n_steps=<N>` 做周期容灾。

H3 recipe 自带 FSDP2、activation recompute、FP32 参数与 optimizer state、BF16 forward
精度 contract 和 attention kernel 约束。持久参数保持 FP32，FSDP2 只在 forward 中按
模型 contract 将普通计算转换为 BF16。这些是模型实现边界，不是用户每次必须覆盖的参数。

## HSDP 定向验收

H3 的常规用户路径仍是 USP4。维护者需要验证 HSDP 时，B300 单机 8 卡使用两个
4-way shard group：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache \
  parallel.usp_size=1 \
  parallel.dp_replicate_size=2 \
  parallel.dp_shard_size=4 \
  attention.backend=flash_attention_4 \
  runtime.max_steps=1 \
  checkpoint.dirpath=/local-ssd/h3-hsdp/checkpoints \
  checkpoint.every_n_steps=1
```

每个 DP rank 至少需要一个 prepared sample；8 卡 smoke 的 cache manifest 因此至少包含
8 个样本。该 topology 已在 8×B300 上验证 H3 33B forward/backward/optimizer、
step-1 full DCP 保存，以及相同 HSDP split 下通过 `resume.path=auto` strict resume
到第二步。它不替代 USP4 的 eval 和 Diffusers parity gate，也不代表 H3 多节点 HSDP
已经验收。维护者 KAP 命令见 [KAP 定向测试](../development/kap.md)。

## 完整续训

保持 USP/world size、manifest、cache fingerprint、模型版本、
`runtime.output_dir` 和自定义 `checkpoint.dirpath` 不变，提高总目标步数，并只增加
`resume.path=auto`：

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache \
  parallel.usp_size=4 \
  runtime.max_steps=200 \
  resume.path=auto
```

H3 33B full DCP 约含 124 GB FP32 weights 和 248 GB FP32 AdamW state。保存前确认
本地或远端空间以及异步 staging 容量。

## 加载已有训练 checkpoint

`model.pretrained_path` 始终指向完整的 MiniMax-H3 基础 snapshot，用于构造模型配置、
VAE、text encoder、tokenizer 和 scheduler；不要把它替换成训练产出的 checkpoint。
训练权重通过 `resume.path` 叠加加载。

当前 FP32 contract 生成的 full DCP 使用上一节的 `resume.path=auto` 或显式 DCP 路径
完整续训，同时恢复 FP32 model weights、FP32 AdamW state、step、scheduler、dataloader
和 RNG。

之前训练得到的 BF16/mixed-dtype DCP 或 weights-only PT 只能用于 weights-only
warm-start，不应恢复旧 optimizer 和 training state。加载后权重存入当前 FP32 参数，
但历史 checkpoint 已经发生的 BF16 舍入无法恢复；optimizer 从新的 FP32 state 开始，
训练 step 从 0 重新计数。

加载历史 DCP：

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  data.train_manifest=/data/h3/train.jsonl \
  data.cache_dir=/data/h3/cache \
  parallel.usp_size=4 \
  runtime.max_steps=100 \
  resume.path=/runs/source/checkpoints/00100.dcp \
  resume.optimizer=false \
  resume.training_state=false
```

加载历史 PT 时使用同一组参数，仅把路径替换为 PT 文件：

```bash
  resume.path=/runs/source/checkpoints/00100.pt \
  resume.optimizer=false \
  resume.training_state=false
```

H3 默认 `runtime.weights_strict=true`，因此 checkpoint 的 Transformer key 和 shape 必须
与当前模型一致。格式不同不会自动放宽结构检查；模型结构已经变化的 checkpoint 需要先
完成显式转换，不能依靠 non-strict 加载掩盖缺失权重。

## Native DCP 评估

H3 当前 eval gate 与训练相同，要求单节点 USP4，因此这里继续使用 `torchrun`；不要套用
Wan21 的单进程 `vforge-eval` 示例。纯推理请求使用 condition-only manifest：

```json
{"sample_id":"h3-001","image":"/data/first.png","prompt":"A singer performs on stage.","num_frames":124}
```

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.eval \
  experiment=minimax_h3_fl2va \
  model.pretrained_path=/models/MiniMax-H3 \
  evaluation.data_source=inference \
  evaluation.manifest=/data/h3/infer.jsonl \
  parallel.usp_size=4 \
  evaluation.checkpoint_path=outputs/train/checkpoints/00200.dcp \
  evaluation.num_samples=1
```

需要 target-backed standalone eval 时使用 `evaluation.data_source=validation`，其
manifest 与训练数据相同，包含 `video` 和可选 `image`/`last_image`。外部 latent
cache 使用 `evaluation.data_source=cache`，并把 `evaluation.manifest` 直接指向
cache 根目录下兼容的 `manifest.json`。Eval 在线编码时不会编码目标视频或音频；训练后的
Transformer 权重来自 `evaluation.checkpoint_path`。`data.val_manifest` 和
`data.cache_dir` 不属于 standalone eval 输入。

## 产物

- train：`resolved_config.yaml`、hash、step JSONL 和
  `${runtime.output_dir}/checkpoints/<step>.dcp`；
- eval：默认 `outputs/eval/samples`（即 `${runtime.output_dir}/samples`）中的
  video/audio 产物和 manifest；
- 默认 final full DCP、异步保存、保留最近 2 个已提交 checkpoint。

## 支持边界

- 只支持 FL2VA；Ref2VA、sparse attention 和动态 batch 不在本轮范围；
- `data.batch_size` 必须为 1；常规公开 gate 是单机 USP4，另有单机 8×B300 HSDP
  定向 train smoke；
- H200/H20 选择 FA3，B200/B300/GB300 选择 FA4；缺少所需 kernel 时 fail-fast；
- FP32 latent/modulation 边界、autocast、strict weights 和 fallback policy 由 recipe
  与模型 contract 管理，不应变成常规命令行步骤；
- 默认固定学习率；需要 warmup 时可使用
  `scheduler.name=constant_with_warmup scheduler.warmup_steps=<N>`；
- 修改帧数、画布或 encoder 配置后必须重新 prepare。

通用恢复和周期保存组合见
[通用 Override 速查](../reference/configuration-cheatsheet.md)；格式、strict resume 和
远端存储行为见 [Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

## 维护者验收（可选）

框架维护者复现 1×4 GB300 gate 时，使用
[KAP 定向测试](../development/kap.md) 中的
KAP 流程，不从本页用户命令推导 CI 资源配置。
