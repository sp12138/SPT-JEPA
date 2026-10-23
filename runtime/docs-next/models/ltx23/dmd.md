# LTX2.3 DMD

## 准备模型与数据 {#dmd-data}

DMD 需要三份 `.safetensors` 文件，各自用途如下：

1. **官方完整 checkpoint**（`LTX_DMD_BASE`）：读取模型结构配置，并加载文本投影层与
   音视频文本连接器（connector）的权重。Transformer 主体参数由下面两份文件提供。
2. **Generator 初始权重**（`LTX_DMD_GENERATOR`）：使用 causal TF 训练后的权重，初始化待训练的生成器。
3. **Score 初始权重**（`LTX_DMD_SCORE`）：使用 bidirectional 模型权重，同时初始化冻结的
   `real_score` 和参与训练的 `fake_score`。

另外准备完整 Gemma 模型目录，以及包含视频 latent 和文本条件的 `.precomputed`
目录。两路缓存文件的相对路径必须相同：

```text
.precomputed/
  latents/<sample>.pt
  conditions/<sample>.pt
```

DMD 会生成音频并计算 audio loss，当前配置使用零值 reference audio，因此不需要
`audio_latents/`。[SFT / TF](sft-tf.md) 仍需三路数据。

把实际路径写入环境变量：

```bash
export LTX_DMD_BASE=/path/to/base.safetensors
export LTX_DMD_GENERATOR=/path/to/generator.safetensors
export LTX_DMD_SCORE=/path/to/score.safetensors
export LTX_DMD_GEMMA=/path/to/gemma-3-12b
export LTX_DMD_CACHE=/path/to/dataset/.precomputed
```

数据分片存放在同一目录下时，将 `LTX_DMD_CACHE` 指向它们的父目录即可：

```text
.precomputed/
  0001/latents/<sample>.pt
  0001/conditions/<sample>.pt
  0002/latents/<sample>.pt
  0002/conditions/<sample>.pt
  ...
```

例如参考配置中的 `0001` 至 `0010`，只需配置共同的 `.precomputed` 路径，无需逐个
列出或合并文件。程序读取父目录下所有包含 `latents/` 和 `conditions/` 的直接子目录，
按目录名、样本相对路径排序；不同分片中的同名样本分别保留。

采样器按实际 rollout 的 block 数分桶。默认 `data.drop_last=true`，每个桶中不足
一个 DP 组的尾部样本会被丢弃；至少一个桶需要有 DP 个样本，训练才能启动。
小数据集可以设置 `data.drop_last=false`，通过重复桶内样本补齐 DP 组。

以下是可直接使用的 DMD 范例，位于 `ieg-infra-efs` 存储卷，挂载路径为 `/efs`。
包含 16 个缓存样本、三份权重和完整 Gemma 依赖：

```bash
export LTX_DMD_ROOT=/efs/alexlshen/vforge/examples/ltx23-dmd
export LTX_DMD_BASE="$LTX_DMD_ROOT/model/base.safetensors"
export LTX_DMD_GENERATOR="$LTX_DMD_ROOT/model/generator.safetensors"
export LTX_DMD_SCORE="$LTX_DMD_ROOT/model/score.safetensors"
export LTX_DMD_GEMMA="$LTX_DMD_ROOT/model/gemma-3-12b"
export LTX_DMD_CACHE="$LTX_DMD_ROOT/data/.precomputed"
```

范例缓存保留原始 **113 帧、480×832**，未裁切或重新序列化。下面将 rollout 帧数上限
设为 113；DMD 按完整 block 生成，实际使用 **97 帧、480×832**。使用自己的数据时，
相应调整 `training.models.generator.config.max_train_frames`。
下方命令用 `LTX_DMD_MAX_FRAMES` 设置这个上限，默认 113；模型 experiment 默认值为 241，
当前支持范围为 25～241。实际生成长度还受缓存长度和完整 block 布局限制。

## 单机 DP8 {#dmd-dp8}

单机 8 卡、不开启 Ulysses 时使用 `DP8 × USP1`。下面运行 100 步；可通过
`LTX_DMD_MAX_STEPS` 调整训练步数。

通过 `training.dmd.rollout.backward_mode` 选择生成器 rollout 的反向方式：

- **`retained_graph`（默认）**：保留各时间 block 的计算图，反向时直接使用，显存占用较高。
- **`block_replay`**：反向时逐个时间 block 重放前向并释放计算图，以额外计算降低显存占用。

下面两条启动命令显式设置 `training.dmd.rollout.backward_mode=block_replay`。
需要使用默认方式时，将这一行改为 `training.dmd.rollout.backward_mode=retained_graph`。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LTX_DMD_MAX_STEPS="${LTX_DMD_MAX_STEPS:-100}"
export LTX_DMD_MAX_FRAMES="${LTX_DMD_MAX_FRAMES:-113}"
export LTX_DMD_OUTPUT="${LTX_DMD_OUTPUT:-/efs/your-name/vforge/runs/ltx23-dmd/dp8}"

torchrun --standalone --nproc_per_node=8 \
  -m vforge.tasks.train \
  experiment=ltx23_dmd \
  training.models.generator.pretrained_path="$LTX_DMD_BASE" \
  training.models.generator.init_weights="$LTX_DMD_GENERATOR" \
  training.models.real_score.init_weights="$LTX_DMD_SCORE" \
  training.models.generator.config.text_encoder_path="$LTX_DMD_GEMMA" \
  data.sources.0.cache_dir="$LTX_DMD_CACHE" \
  runtime.output_dir="$LTX_DMD_OUTPUT" \
  runtime.max_steps="$LTX_DMD_MAX_STEPS" \
  training.models.generator.config.max_train_frames="$LTX_DMD_MAX_FRAMES" \
  parallel.usp_size=1 \
  training.dmd.rollout.backward_mode=block_replay \
  attention.backend=auto \
  resume.path=auto
```

## 单机 DP4 × USP2 {#dmd-sp2}

同样使用单机 8 卡，每两个进程组成一个 Ulysses 组，数据并行度改为 4。
完成前面的模型与数据路径设置后，执行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LTX_DMD_MAX_STEPS="${LTX_DMD_MAX_STEPS:-100}"
export LTX_DMD_MAX_FRAMES="${LTX_DMD_MAX_FRAMES:-113}"
export LTX_DMD_OUTPUT="/efs/your-name/vforge/runs/ltx23-dmd/dp4-usp2"

torchrun --standalone --nproc_per_node=8 \
  -m vforge.tasks.train \
  experiment=ltx23_dmd \
  training.models.generator.pretrained_path="$LTX_DMD_BASE" \
  training.models.generator.init_weights="$LTX_DMD_GENERATOR" \
  training.models.real_score.init_weights="$LTX_DMD_SCORE" \
  training.models.generator.config.text_encoder_path="$LTX_DMD_GEMMA" \
  data.sources.0.cache_dir="$LTX_DMD_CACHE" \
  runtime.output_dir="$LTX_DMD_OUTPUT" \
  runtime.max_steps="$LTX_DMD_MAX_STEPS" \
  training.models.generator.config.max_train_frames="$LTX_DMD_MAX_FRAMES" \
  parallel.usp_size=2 \
  training.dmd.rollout.backward_mode=block_replay \
  attention.backend=auto \
  resume.path=auto
```

`parallel.dp_size` 默认自动推导：上述 8 卡命令设置 `parallel.usp_size=2` 后，DP 为 4。
DMD 当前要求 TP 为 1、每个 DP
rank 的 batch 为 1。改变 DP/SP 会改变全局 batch 和数值结果，不同拓扑的 loss 不要求相同。

## 学习率 {#dmd-learning-rate}

两种命令均由 FSDP 汇集 BF16 参数，关闭 autocast；norm、gradient reduce 和
gradient norm 保留 FP32。generator 每五步更新一次，critic 每步更新。两套 AdamW 都使用**固定学习率**，默认 `1e-5`；
experiment 中 `scheduler.name=none`，没有 warmup 或衰减。

调整学习率时，在上面的启动命令中增加以下参数，例如生成器用 `5e-6`、critic 用 `1e-5`：

```text
optimizers.generator.learning_rate=5e-6
optimizers.critic.learning_rate=1e-5
```

当前 DMD 两路 optimizer 尚未接入通用 scheduler，暂不支持通过 `scheduler.name`
配置 warmup 或线性衰减。

## 保存与恢复 {#dmd-resume}

上述命令从第一次启动就使用 `resume.path=auto`，默认在
`${runtime.output_dir}/checkpoints` 中查找步数最大的完整 checkpoint；设置了
`checkpoint.dirpath` 时，从指定目录查找。没有 checkpoint 时从初始权重开始。
重启时使用原输出目录，新开一轮训练时换一个输出目录。
请将示例中的输出路径替换为自己的持久存储路径。

训练日志位于输出目录的 `train_steps.jsonl`，默认在结束时保存 checkpoint。
需要中途故障恢复时，增加 `checkpoint.every_n_steps=100` 定期保存。
指定 checkpoint 恢复和平台自动容错见
[保存、恢复与自动容错](../../guide/training-recovery.md)。

将 checkpoint 存储在 S3 时，可配置本地 SSD 暂存和并发传输，加速保存与恢复，参考
[S3 保存与恢复加速](../../guide/training-recovery.md#s3)。

## 推理

训练后的 checkpoint 可直接用于 [DMD 推理](eval.md)，支持缓存文本或自定义 prompt，
输出带音轨的视频。
