# Gen Engine 多模态生成

Gen Engine 使用从零训练的原生三流 MMDiT。它支持无参考和多参考生成，target 可以是
视频、音频或配对音视频。Reference 可以是 image、video、audio 或 video_audio。
Dense 与 Visual MoE 变体共用同一套 text/visual/audio joint-attention 主干，不继承或
运行时依赖 H3、Wan、Diffusers 或 MCore MoE。

## 数据

训练前不执行 Prepare。数据团队交付：

```text
samples.jsonl
cache/mme/*.pt
cache/visual/*.pt
cache/audio/*.pt
```

JSONL 的完整契约见
[T2V / R2V 数据组织格式](https://git.woa.com/gen-engine/models/dit/blob/main/docs/t2v-r2v-data-schema.md)。
模型结构与 V0 验证计划同目录：
[MMDiT 多模态生成设计](https://git.woa.com/gen-engine/models/dit/blob/main/docs/mmdit-multimodal-generation-design.md)、
[V0 验证计划](https://git.woa.com/gen-engine/models/dit/blob/main/docs/moe-v0-validation-plan.md)。
运行时只读取已经生成的 MME embedding 和 VAE encoder 输出，不加载 Qwen、video VAE
或 audio VAE。文件既可以是包含 `embedding` / `latent` 字段的 vforge bundle，也可以
是 `torch.save(tensor, path)` 保存的裸 Tensor；默认仍读取 bundle，裸 Tensor 需要显式
选择对应配置。描述符中的 shape、dtype 必须与文件内容一致。

JSONL 本身放在本地；`embedding_path` 和 `latent_path` 支持本地路径及 `s3://bucket/key`。
S3 文件自动下载到本地缓存，复用标准 AWS 凭据链。Dataset 不修改 JSONL、不重新封装
`.pt`，也不为预计算 latent 下载 `source_path` 指向的原始媒体。设置音频输入布局为
`ATC` 时，读取器先按 JSONL 声明校验 `[声道数, 时间长度, 特征维度]`，再在内存中转成
packing 使用的 `ACT`；文件中的数值和排列保持原样。

单份数据使用 `data.train_manifest`；多份数据使用 `data.sources` 中的
`name`、`manifest`、`weight`，两种入口互斥。每个子 Dataset 使用相同的文件格式、布局
和 codec 配置，由 MultiDataset 路由样本地址，MultiSourceSampler 负责按权重选择
Source 及 Source 内 Shuffle。这两种入口都支持裸 Tensor 与 S3 文件。

每条样本读取后，DataLoader worker 执行 CPU packing：

1. 对声明了 `posterior_sampling` 的 visual encoder 输出，从当前 PyTorch RNG 采样；
2. 使用 latent codec 中的 channel-wise mean/std 标准化 VAE latent；
3. visual latent 按模型 patch size 转成 rows；
4. audio latent 按 audio plane 和时间转成 rows；
5. 使用 codec temporal mapping 和媒体 `effective_fps` 生成秒单位的局部时间；
6. reference 使用 `cursor = ceil(end) + 1` 顺排；
7. 生成 position IDs、stream indices 和 reference identity IDs。

Reference 与 clean target rows 保持分离。Target 加噪发生在训练 step，不写回数据集。

### 无真实数据时生成测试样本

从仓库根目录运行以下数据工具，输出目录必须尚不存在：

```bash
python -m vforge.tasks.gen_engine.synthetic \
  --output-dir /data/gen-engine-synthetic \
  --transformer-config vforge/models/gen_engine/dit/configs/mmdit-moe-32b-a4.1b.json \
  --video-latent-codec-config vforge/models/gen_engine/video_vae/latent_codec_configs/minimax-h3.json \
  --audio-latent-codec-config vforge/models/gen_engine/audio_vae/latent_codec_configs/minimax-h3.json \
  --seed 42
```

默认 `smoke` profile 各生成一个视频、音频、音视频、带 image/audio reference 的
音视频和长视频样本。对应总 token 数为 1,024、386、1,154、1,298 和 12,736；
每条均含 256 个文本 token。长视频的 raw moments 为 `[48,32,30,52]`，采样后
`[24,32,30,52]` 按 `[1,2,2]` patch 变为 12,480 个视觉 token。
`--profile long_video --samples-per-kind 200` 可生成 200 个不同的长视频样本。

数据是独立高斯随机张量，MME embedding 不对应 caption 或媒体内容。Visual mean 和
audio latent 使用实际 codec 的 mean/std 还原至 VAE 空间；visual posterior 的 std
在模型空间为 0.1。输出 `samples.jsonl`、带摘要校验的 bundle 和记录配置/seed 的
`generation.json`。相同配置和 seed 重放张量值；文件摘要校验实际序列化文件，不承诺
跨运行或 PyTorch 版本的字节一致性。数据用于链路验证，不能用于判断模型收敛质量。

## Transformer 配置

`model.extras.transformer_config_path` 指向原生 MMDiT 的严格 JSON 配置。配置必须显式
声明 schema 和 architecture；未知字段会直接报错，旧 H3-shaped 配置和 checkpoint
不会自动映射。版本控制的模型结构统一放在
`vforge/models/gen_engine/dit/configs/`；当前包含 32B-A4.1B MoE、
同 FLOP 4.1B Dense 和约 2.5× 的 10.4B Dense 杠杆模型。Demo 目录里的
`transformer*.json` 只用于缩小的 fixture，不作为稳定训练结构。Dense 最小结构如下：

```json
{
  "schema": "vforge.gen_engine.dit",
  "schema_version": 1,
  "architecture": "mmdit_dense",
  "scratch_init_scheme": "mmdit_stable_v1",
  "num_attention_heads": 32,
  "attention_head_dim": 128,
  "hidden_size": 4096,
  "num_layers": 40,
  "num_refiner_layers": 2,
  "ffn_dim": 11008,
  "refiner_ffn_dim": 11008,
  "in_channels": 24,
  "audio_in_channels": 32,
  "patch_size": [1, 2, 2],
  "text_dim": 5120,
  "freq_dim": 256,
  "time_embed_hidden_dim": 4096,
  "time_embed_dim": 2048,
  "rope_freq_dim": 16,
  "max_reference_index": 32,
  "position_time_scale": 1.0,
  "reference_embedding_init_std": 0.02,
  "reference_identity_mode": "modality_index",
  "num_audio_planes": 2,
  "audio_plane_embedding_init_std": 0.02
}
```

`ffn_dim` 是 text/audio FFN 宽度，也是默认 Visual FFN 宽度。Dense isoFLOP
对照可以额外声明 `visual_ffn_dim`，只覆盖 Visual Dense FFN，不改变 text/audio
或 Token Refiner；省略时严格回退到 `ffn_dim`。

MoE JSON 的 `moe.dispatcher` 可显式选择 `vforge_expert`，使用公共 packed Expert；
默认 `torch_reference` 保留逐专家实现。Packed bank 直接堆叠原 SwiGLU 的
`[up, gate]` 权重，router、Top-K、shared branch 和初始化配方仍由 MMDiT 拥有。
两个 dispatcher 的参数键不同，不能直接互相加载 checkpoint。EP 通过模型的
`get_parallel_plan()` 声明主专家 bank；block bank、shared expert 和 router 使用普通
FSDP。拓扑统一由 `parallel.ep_size`、`parallel.usp_size` 配置，不属于模型 JSON。
完整 32B 的 EP4/USP4 长序列训练仍未验证。
Packed 路径保持 FP32 router 和 Top-K 归一化；混合权重在与专家输出相乘前转为
专家实际输出的 dtype，再累加到残差 dtype，保持 reference 的乘法精度。
当前公共 Expert 的 grouped-MM 计算仅支持 CUDA BF16；此混合策略不代表支持 FP16 GEMM。
保持 `runtime.autocast=false`，由模块的混合精度策略控制计算精度，保留要求 FP32 的模块。
显式开启全局 autocast 时，vforge 会在构建训练配置时输出 warning。

`architecture=mmdit_moe` 时必须增加 `moe` object。V0 杠杆实验的稳定配置和精确参数
口径见 `vforge/models/gen_engine/dit/configs/README.md`。256B 生产蓝图仍保留在
`examples/configs/gen-engine/mmdit-moe-256b.json`，核心参数估算约为 255.9B total /
38.4B active。Visual block 1 使用 Dense SwiGLU，block 2 起使用独立 main/block router、
routed experts 和 fused shared branch；text/audio 在所有 block 中保持 Dense。Router
以 FP32 计算 sigmoid 和 top-k，`raw_score + routing_bias` 只用于选择，mixing 使用 raw
sigmoid score。main、block 各自在已选 experts 上独立归一化，再与 gated shared branch
三路相加。`main_routed_scaling_factor` 与 `block_routed_scaling_factor` 在各 bank
归一化后应用；生产基线均显式为 `1.0`，保留凸组合语义。它们只为 V0 定向消融保留，
不是把 DeepSeek-V3 的单 bank `2.5` 同时复制到两个 banks。

MME hidden width、visual row width 和 audio row width 必须分别与 config 中的
`text_dim`、`in_channels × product(patch_size)`、`audio_in_channels` 一致。
Audio latent 第一维固定为 `[left,right]`，两声道共享 `(T,0,0)` RoPE，并分别添加
left/right audio-plane embedding；reference audio 另外再添加 reference modality/index
embedding。模型输入中的 `audio_plane_ids` 与完整 packed sequence 对齐：
`0=非 audio`、`1=left`、`2=right`，embedding row 0 固定为零。

`reference_identity_mode` 支持 `none`、`modality` 和 `modality_index`，默认使用完整
modality/index identity。`num_refiner_layers=0` 表示 Linear-only MME connector；大于
0 时使用本地 text-only Token Refiner。Text token 明确不应用 DiT RoPE。

### 32B MoE 与 4096 维文本输入 {#mmdit-moe32b-pt}

`experiment=gen_engine_mmdit_moe32b` 使用
`examples/configs/gen-engine/mmdit-moe-32b-a4.1b-pt4096.json`。它派生自固定版本 DiT
catalog 的 `mmdit-moe-32b-a4.1b.json`，仅将 `text_dim` 从 5120 改为 4096；28 层、
1536 hidden size、160 个 main experts、top-8 及其余结构保持一致。

该配置选择裸 Tensor、音频 `ATC` 布局，并启用实际文件身份固定，不要求 JSONL 增加
缓存 digest 字段。它接受 4096 维 MME、24 通道视频 latent 和双声道 32 维音频 latent，
不会自动裁剪、补齐文本维度，也不会根据模型名选择 VAE 预处理。

此 profile **没有预设 video/audio codec**。必须由数据导出方确认视频是 posterior
moments 还是已采样 latent、是否已经标准化，以及各模态的时间映射，再提供两份 codec
配置。仅凭 tensor shape 和 `effective_fps` 无法推导这些信息。若确认文件已在模型空间，
可在 codec 中明确设置逐通道 mean=0、std=1；若确认视频已采样，应省略
`posterior_sampling`。这些是数据契约，不是选择 MMDiT 后自动成立的假设。

该完整结构包含 **32,187,187,072** 个参数，已通过 CPU meta 构建和使用 synthetic
数据、模拟 codec 的受限 GPU 训练 Replay。原始数据与数据生产方 codec 的组合、正式
训练规模和 KAP 仍未验证。

## Latent codec 配置

VideoVAE 和 AudioVAE 使用两个独立 contract：

- `model.extras.video_latent_codec_config_path`：video latent channels、posterior
  sampling、normalization、spatial compression 和 temporal mapping；
- `model.extras.audio_latent_codec_config_path`：audio latent channels、planes、
  normalization 和 temporal mapping。

两份配置独立定版和计算 digest，因此任一 VAE 可以单独迭代。运行时才把它们组合成
packing 所需的 `LatentCodecConfig`。
旧的 combined `latent_codec_config_path` 不再接受，也不会隐式回退，以免训练静默使用
错误版本的另一种 VAE contract。

VideoVAE 配置骨架：

```json
{
  "schema": "vforge.video_latent_codec",
  "schema_version": 1,
  "latent_channels": 24,
  "spatial_compression": 16,
  "posterior_sampling": {
    "type": "diagonal_gaussian",
    "logvar_min": -30.0,
    "logvar_max": 20.0
  },
  "normalization": {
    "type": "channelwise_mean_std",
    "mean": ["24 values"],
    "std": ["24 values"]
  },
  "temporal": {
    "input_rate_unit": "frames_per_second",
    "mapping": {
      "type": "periodic_offsets",
      "input_period": 17,
      "latent_offsets": [0, 1, 5, 9, 13],
      "valid_length_remainder": 5
    }
  }
}
```

AudioVAE 配置骨架：

```json
{
  "schema": "vforge.audio_latent_codec",
  "schema_version": 1,
  "latent_channels": 32,
  "planes": 2,
  "normalization": {
    "type": "channelwise_mean_std",
    "mean": ["32 values"],
    "std": ["32 values"]
  },
  "temporal": {
    "input_rate_unit": "samples_per_second",
    "mapping": {
      "type": "stride",
      "input_stride": 800
    }
  }
}
```

完整 H3 contract 位于：

- `vforge/models/gen_engine/video_vae/latent_codec_configs/minimax-h3.json`
- `vforge/models/gen_engine/audio_vae/latent_codec_configs/minimax-h3.json`

`effective_fps` 表示实际送入 VAE 的输入采样率：video 使用 frames/s，audio 使用
samples/s。以 H3 为例，video 使用 24，audio 使用 32000；codec 分别使用周期
offset `[0,1,5,9,13] / 17` 和 stride 800，把输入位置换算成 latent local time。

H3 visual bundle 保存 VideoVAE 原始 encoder moments，shape 为
`[2 * latent_channels, T, H, W]`，前后两半依次是 mean 和 logvar。Gen Engine 从当前
PyTorch RNG 采样成 `[latent_channels, T, H, W]`，再在 patchify 前标准化；codec 不持有
seed，运行级 RNG 和 resume 由 training runtime 管理。当前采样结果直接以 FP32 标准化，
不复现 H3 路径中的 FP16 round。未声明 `posterior_sampling` 的 codec 仍把输入视为已经
得到的 VAE latent。

推理在 unpatchify 后反标准化，因此输出 latent 可直接送入对应 VAE decoder。Codec
`latent_channels` 必须与 Transformer 的 `in_channels` / `audio_in_channels` 一致。

Gen Engine 不加载 MiniMax-H3 DiT 权重。Checkpoint 仍以 `transformer.*` 为模型根，
但内部 key 是原生 MMDiT schema；旧 H3 block key 会 strict-load 失败。首次训练在
FSDP/TP materialize 后执行确定性的 scratch initializer；完整 resume 直接恢复
checkpoint（包括 FP32 routing-bias buffers），不重复初始化。

Scratch 模型固定使用 `mmdit_stable_v1`：

- 普通 Linear weight 在 FP32 中按 `Var(W)=1/fan_in` 的 bounded uniform 采样，bias
  为零，RMSNorm weight 为一；
- 主干每层的 fused AdaLN weight/bias 全零，使每个 block 从严格 identity 开始；
- Visual/audio FinalAdaLN 和 output heads 全零，使初始 flow velocity 精确为零；
- 两层无门 Token Refiner 的 attention output 与 FFN down 各乘
  `1/sqrt(2 * num_refiner_layers)=0.5`；
- main/block router 的目标 logit RMS 为 `0.1`，routing bias 为零；shared sigmoid
  gate weight 为零、初始 gate 为 `0.5`；
- reference/index/audio-plane embedding 非 padding 行的 std 为 `0.02`，padding 行为零。

初始化在 materialize 和 FSDP sharding 之后逐 local shard 执行，不 gather 完整参数。
随机值先以 FP32 生成再转换到参数存储 dtype；module-name seed 让同布局重跑确定，shard
coordinate 区分不同分片而 replica 保持一致。Zero head 的第一个 backward 只更新
output heads，后续两个 optimizer steps 依次打开 FinalAdaLN/visual-audio gates 和对应
trunk branches；前层 text residual branch 再晚一个 gate update，最后一层未被消费的
text residual 参数则是结构性 dead。因此 smoke/canary 必须对非零 flow target 计算
loss，并按这些阶段判断梯度覆盖。

## 训练

`experiment=gen_engine` 声明模型差异；`data.*`、`parallel.usp_size`、
`runtime.max_steps` 等公共字段由训练入口的 `native_defaults` 提供，使用普通
`key=value` 覆盖。

当前 experiment 使用 AdamW：`lr=1e-5`、`betas=(0.9, 0.95)`、`eps=1e-8`、
`weight_decay=0`，global gradient clip 为 `1.0`。默认 `scheduler.name=none`，
不启用 LR warmup。Forward 使用 BF16，router/路由统计和 optimizer moments 保持 FP32。

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=gen_engine \
  model.extras.transformer_config_path=/models/gen-engine/config.json \
  model.extras.video_latent_codec_config_path=/models/video-vae/latent_codec.json \
  model.extras.audio_latent_codec_config_path=/models/audio-vae/latent_codec.json \
  data.train_manifest=/data/mmgen/train.jsonl \
  parallel.usp_size=4 \
  runtime.max_steps=100
```

`latent_cache.mode` 默认就是 `offline`，表示输入已经由数据团队编码；gen_engine 拒绝
其他取值，并且不要求 `data.cache_dir`。第一版要求 `data.batch_size=1`（同为默认值），
但 `.pt` 加载和 packing 可以通过 `data.num_workers`、`persistent_workers` 和
`prefetch_factor` 并行。Experiment 默认 `model.extras.require_digests=true`，因此每个
MME/latent descriptor 必须声明对应的 `embedding_digest`/`latent_digest`。

默认使用 lazy validation：启动时建立 JSONL byte offsets，单样本 schema/路径校验
延迟到 DataLoader 访问。启用 `pin_artifacts` 后，启动阶段还会遍历所需 tensor 描述符，
获取文件并固定实际 SHA-256 与远端对象身份；大数据集的首次启动会产生对应的 I/O。
全量业务验收和重复 `sample_id` 检查仍由数据团队的离线验收工具负责。

使用完整 32B 裸 Tensor profile 时，从仓库根目录运行，并替换为数据生产方确认的 codec
和数据路径。可先用 `--cfg job` 检查组合后的配置，再运行训练：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=gen_engine_mmdit_moe32b \
  model.extras.video_latent_codec_config_path=/models/encoder/video-confirmed.json \
  model.extras.audio_latent_codec_config_path=/models/encoder/audio-confirmed.json \
  model.extras.artifact_cache_dir=/data/mmgen/artifact-cache \
  data.train_manifest=/data/mmgen/train.jsonl \
  parallel.usp_size=1 \
  runtime.max_steps=100
```

读取字段的默认值和可选值见[配置参考](../reference/configuration.md#gen-engine-input)。
多源训练将上面的单 manifest 替换为用户 YAML 中的 `data.sources`，并保留同一组模型
extras：

```yaml
data:
  train_manifest: ""
  sources:
    - name: source_a
      manifest: /data/mmgen/source-a.jsonl
      weight: 11
    - name: source_b
      manifest: /data/mmgen/source-b.jsonl
      weight: 89
```

Source 权重独立于实际数据量。静态训练总样本预算由训练步数、本地 batch size 和 DP
共同确定，恢复需要保持同一采样计划。

Video/audio target 使用同一个基础随机数采样 flow timestep，再分别应用各自的 flow
shift。不存在的 target stream 不参与 loss。Step log 同时记录
`video_output_rms` / `audio_output_rms`，用于确认 zero-velocity 预测在 warmup 中平滑
长出；缺失 stream 的对应 RMS 为零。

MoE 的 routing bias 不在 forward 或 activation-checkpoint 重算中更新。每次 optimizer
step 完成后，训练器通过 `post_optimizer_step` 把每层 detached routing statistics 在
`DP×USP` group 上求和，再对 main/block 两个 bank 独立执行 RMS-normalized load-error update：

```text
bias = (1 - decay) * bias
     + alpha(step) * (mean(count) - count)
       / sqrt(mean((mean(count) - count)^2) + eps)
```

默认 `alpha=4e-5`、`decay=8e-6`，alpha 在总 optimizer steps 的前 1% 线性 warmup。
因此启用 MoE warmup 时必须提供正的 `runtime.max_steps`，否则 warmup 无法定标并直接
报错。日志包含逐层、逐 bank 的 load max/mean、CV、dead fraction、selected affinity、
selection entropy、bias 范围和 main/block/shared output RMS，另含
image-reference/video-reference/target-video 分类负载及 shared-gate 均值、标准差和
0/1 饱和率。设置
`moe.detailed_log_interval>0` 可按间隔输出完整 per-expert count。

## Resume

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=gen_engine \
  model.extras.transformer_config_path=/models/gen-engine/config.json \
  model.extras.video_latent_codec_config_path=/models/video-vae/latent_codec.json \
  model.extras.audio_latent_codec_config_path=/models/audio-vae/latent_codec.json \
  data.train_manifest=/data/mmgen/train.jsonl \
  parallel.usp_size=4 \
  runtime.max_steps=200 \
  resume.path=auto
```

严格 resume 会校验 manifest digest、bundle 声明的 digest、codec digest、
packing/layout version、模型结构和 sampler 状态。数据团队替换 `.pt` 时必须同步更新
manifest digest 字段。

裸 Tensor profile 的 `pin_artifacts=true` 会把实际文件 SHA-256、大小以及 S3 的对象
版本或 ETag 等身份纳入数据恢复契约。即使 JSONL 没有缓存 digest，同一路径换了文件
也不会被静默接受。S3 有 VersionId 时按指定版本下载；无版本时检查下载前后的对象
身份。运行中复用已固定的本地副本并验证字节；新运行重新检查远端身份，再与 checkpoint
中的契约比较。本地缓存目录不参与数据身份，可以在另一台机器重新获取相同数据。

`require_digests=false` 只允许描述符省略缓存 digest，并不代表 `source_digest` 或 MME
`fingerprint` 可以代替缓存 SHA-256。后两者描述原媒体或编码来源，实际 tensor 字节
由文件身份固定机制单独校验。该机制固定数据输入；训练 Replay 还要求相同采样配置、
codec 随机过程和训练随机状态。

## 评估

推理 manifest 使用预计算 MME/reference latent 和 `target_spec`。`evaluation.*` 由
`conf/evaluation/default.yaml` 声明，因此使用常规 Hydra override：

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.eval \
  experiment=gen_engine \
  model.extras.transformer_config_path=/models/gen-engine/config.json \
  model.extras.video_latent_codec_config_path=/models/video-vae/latent_codec.json \
  model.extras.audio_latent_codec_config_path=/models/audio-vae/latent_codec.json \
  evaluation.data_source=inference \
  evaluation.manifest=/data/mmgen/infer.jsonl \
  evaluation.checkpoint_path=/runs/gen-engine/checkpoints/00100.dcp \
  parallel.usp_size=4
```

当前 eval 不加载 decoder，主产物是包含生成 video/audio latent 及几何元数据的 `.pt`。
后续如接入独立 decode 服务，可由相同 latent 生成 MP4/WAV，不改变模型输入契约。

## 第一版限制

- `data.batch_size=1`；
- 不提供 `vforge-prepare` / encode verb；
- 数据团队交付的 reference 必须按 `image → video_audio → video → audio` 排列；
  训练 Dataset 信任该顺序，不重复校验；
- MME、所有 reference latent 和训练 target latent 必须预计算；
- packing 使用 `[2,C_a,T]` 音频，读取器可按明确配置接受 `[2,T,C_a]`；
- 不支持运行时媒体解码或 VAE/MME 编码；
- mixed target modality 可以共用一个训练或评估 manifest；
- MoE 可选择 reference 或 packed dispatcher；不设置 capacity、不丢 token，
  不使用 auxiliary loss；
- 256B 配置只用于结构表达和参数估算，完整规模训练仍未验证；
- `latent_moe_dim` 当前必须为 null，非空配置会 fail-fast。


## EP 与 USP 联合验证

先使用 4 层、hidden size 384、12 heads、expert intermediate size 384 的小模型，
保留目录配置的 160 个主专家、Top-8、两个 Dense 层和视频/音频/文本路径。
验证参数为 seed=42、BF16 计算、FP32 梯度归约、每个数据副本 batch size 1。
USP 纳入 FSDP，主专家同时按 EP 和专家 FSDP 两个维度分片。

| GPU 数 | EP | USP | DP | 每 rank 主专家数/层 | 专家 FSDP 大小 |
| --- | --- | --- | --- | --- | --- |
| 4 | 2 | 2 | 2 | 80 | 2 |
| 8 | 4 | 2 | 4 | 40 | 2 |
| 8 | 4 | 4 | 2 | 40 | 2 |

数值 gate 对比相同逻辑权重和全局 batch 下的完整序列对照，覆盖输出、输入及参数
梯度、梯度裁剪和路由计数。对照使用单 rank FSDP 复用 Native 模块精度边界，
专家计算使用独立的逐 assignment 矩阵乘法；router 保留 FP32，不启用全局 autocast。
前反向按 BF16 容差比较；optimizer 使用已校验的相同逻辑
梯度，要求参数更新与 optimizer state 逐项一致。这样可以单独检验分片更新，避免
AdamW 对接近零梯度的归一化放大 BF16 舍入差异。数值对照的均衡/偏斜路由设置明确的
Top-K 选择间隔，避免分支切换破坏梯度比较；原生训练使用默认自然路由。包含纯音频、空专家、USP padding
以及重算开关。GPU 环境必须包含公共 Expert 所需的 grouped-MM 实现。

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=4 \
  tests/models/gen_engine/mmdit_ep_usp_worker.py --ep 2 --usp 2 --suite
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=8 \
  tests/models/gen_engine/mmdit_ep_usp_worker.py --ep 4 --usp 2 --suite
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=8 \
  tests/models/gen_engine/mmdit_ep_usp_worker.py --ep 4 --usp 4 --suite
```

原生训练和新进程续训使用同一 `max_steps=10`：连续运行在第 5、10 步保存异步 full
DCP，再从第 5 步显式恢复到第 10 步。以下目录必须尚不存在；生成的短序列样本包含
视频、音频、音视频和 reference。它们仅用于训练链路验证。

```bash
python tests/models/gen_engine/mmdit_ep_usp_resume.py prepare /data/mmdit-ep-usp
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=8 \
  tests/models/gen_engine/mmdit_ep_usp_resume.py train /data/mmdit-ep-usp --ep 4 --usp 4
CUBLAS_WORKSPACE_CONFIG=:4096:8 torchrun --standalone --nproc-per-node=8 \
  tests/models/gen_engine/mmdit_ep_usp_resume.py train /data/mmdit-ep-usp --ep 4 --usp 4 --resume
python tests/models/gen_engine/mmdit_ep_usp_resume.py compare /data/mmdit-ep-usp
```

这个测试 worker 使用 `runtime.deterministic=true`，并显式关闭 TF32。
比较器要求最终模型（包含 routing bias）、optimizer、scheduler、各 rank 的数据游标与
RNG 逐项一致，并比较后 5 步的 loss 和训练指标。完整 28 层、12,736 tokens 的训练和
恢复使用下文的完整模型配置，不由小模型结果替代。

### HSDP + EP + USP 多节点启动验证

该组合通过公共并行层实现：普通参数在 `dp_shard × usp` 域内分片，主专家在
同一域内划分为 `专家 FSDP × EP`，两者均沿 `dp_replicate` 维度复制并归约梯度。
EP 大小必须整除 `dp_shard_size × usp_size`。梯度累加仍为 1。

16 节点、每节点 8 卡的目标布局使用以下普通 override：

```text
parallel.dp_replicate_size=16
parallel.dp_shard_size=2
parallel.usp_size=4
parallel.ep_size=4
runtime.autocast=false
```

连续节点 rank 排列下，普通参数在节点内 8 卡分片；每 rank 每层持有 40 个主专家，
专家权重在节点内 2 卡分片。每节点处理 2 份样本，全局 DP=32。
2026-09-15，每节点 8×B300、PyTorch 2.13.0+cu130 上完成以下小模型验证：

- 2 节点：均衡/偏斜路由、纯音频 × 重算开关，共 6 用例、12 次逐步数值对照通过；
- 2 节点和 16 节点：Native loader、meta 初始化、自然路由和连续 3 次 optimizer 更新通过；
- 16 节点的 128 个 rank 均成功退出，loss、梯度范数及路由指标均为有限值。

验证使用 4 层、hidden=384、12 heads、expert intermediate=384，保留 160 专家和
Top-8；seed=42、BF16 计算、FP32 梯度归约、开启重算、每副本 batch=1。
上述 EP4 组合仅完成小模型启动验证；完整 32B 的 EP4 训练与续训仍未验证。

### 完整 MoE 32B：128 卡训练与续训

以下命令从仓库根目录运行，使用 16 节点 × 8 B300、完整 28 层模型、160 个主专家和
Top-8。各节点需要相同代码、公共 Expert grouped-MM 和 FA4 依赖，并能读写同一个
共享目录。USP 固定为 4、全局 DP=32、每个数据副本 batch=1，无梯度累加。

| EP | `dp_replicate_size` | `dp_shard_size` | 普通参数分片组大小 | 每 rank 主专家数/层 | 总步数 / 保存间隔 | 完整模型验证 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 16 | 2 | 8 | 20 | 100 / 50 | 100 步及 50→100 续训通过 |
| 16 | 8 | 4 | 16 | 10 | 20 / 10 | 20 步及 10→20 续训通过 |
| 32 | 4 | 8 | 32 | 5 | 20 / 10 | 20 步及 10→20 续训通过 |

普通参数的分片组大小为 `dp_shard_size × usp_size`。这三组主专家的 FSDP 分片数
均为 1，沿 HSDP 副本维度复制。EP 只在每个 HSDP 分片域内通信；EP16、EP32 的
分片域分别覆盖连续的 2、4 个节点，因此普通参数的 FSDP 通信也会跨节点。

**准备模型配置与数据（只在一个节点执行一次）**：将 `MMDIT_ROOT` 改为共享路径，
生成完整模型配置，选择 packed dispatcher；随后生成 256 个长视频样本，每个样本
包含 12,480 个视觉 token 和 256 个文本 token。数据输出目录必须尚不存在。

```bash
export MMDIT_ROOT=/shared/mmdit-32b
mkdir -p "$MMDIT_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path

source = Path("vforge/models/gen_engine/dit/configs/mmdit-moe-32b-a4.1b.json")
config = json.loads(source.read_text())
config["moe"]["dispatcher"] = "vforge_expert"
config["moe"]["detailed_log_interval"] = 1
target = Path(os.environ["MMDIT_ROOT"]) / "transformer.json"
target.write_text(json.dumps(config, indent=2) + "\n")
PY
python -m vforge.tasks.gen_engine.synthetic \
  --output-dir "$MMDIT_ROOT/data" \
  --transformer-config "$MMDIT_ROOT/transformer.json" \
  --video-latent-codec-config vforge/models/gen_engine/video_vae/latent_codec_configs/minimax-h3.json \
  --audio-latent-codec-config vforge/models/gen_engine/audio_vae/latent_codec_configs/minimax-h3.json \
  --profile long_video --samples-per-kind 256 --seed 42
```

**启动训练（以下 Bash 配置在全部 16 个节点执行）**：`MASTER_ADDR` 为节点 0 可达的
地址，`NODE_RANK` 在各节点分别设为 0～15。所有节点的 `MMDIT_EP`、路径和其他参数
必须一致；下面默认 EP16，可改为表中的 EP8 或 EP32。各组合顺序运行并释放上一组资源。

```bash
export MMDIT_ROOT=/shared/mmdit-32b
export MASTER_ADDR=worker-0
export MASTER_PORT=29500
export NODE_RANK=0  # 每个节点分别设置为 0～15
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MMDIT_EP=16
case "$MMDIT_EP" in
  8)  MMDIT_REPLICATE=16; MMDIT_SHARD=2; MMDIT_STEPS=100; MMDIT_SAVE_STEP=50 ;;
  16) MMDIT_REPLICATE=8;  MMDIT_SHARD=4; MMDIT_STEPS=20;  MMDIT_SAVE_STEP=10 ;;
  32) MMDIT_REPLICATE=4;  MMDIT_SHARD=8; MMDIT_STEPS=20;  MMDIT_SAVE_STEP=10 ;;
  *) echo "MMDIT_EP must be 8, 16 or 32" >&2; exit 1 ;;
esac
MMDIT_RUN_ROOT="$MMDIT_ROOT/ep$MMDIT_EP"

mmdit_train() {
  torchrun --nnodes=16 --nproc-per-node=8 \
    --node-rank="$NODE_RANK" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
    -m vforge.tasks.train \
    experiment=gen_engine \
    model.extras.transformer_config_path="$MMDIT_ROOT/transformer.json" \
    model.extras.video_latent_codec_config_path=vforge/models/gen_engine/video_vae/latent_codec_configs/minimax-h3.json \
    model.extras.audio_latent_codec_config_path=vforge/models/gen_engine/audio_vae/latent_codec_configs/minimax-h3.json \
    data.train_manifest="$MMDIT_ROOT/data/samples.jsonl" \
    data.batch_size=1 data.num_workers=0 \
    parallel.dp_replicate_size="$MMDIT_REPLICATE" parallel.dp_shard_size="$MMDIT_SHARD" \
    parallel.ep_size="$MMDIT_EP" parallel.usp_size=4 \
    parallel.gradient_checkpointing=true \
    runtime.seed=42 runtime.precision=bf16 runtime.autocast=false runtime.deterministic=true \
    attention.backend=flash_attention_4 attention.allow_fallback=false \
    logging.wandb_enabled=false \
    runtime.max_steps="$MMDIT_STEPS" checkpoint.every_n_steps="$MMDIT_SAVE_STEP" \
    runtime.output_dir="$1" resume.path="$2"
}

mmdit_train "$MMDIT_RUN_ROOT/continuous" none
```

使用全新的输出目录开始 scratch 训练。保持公共 checkpoint 默认配置：异步 full DCP、
最终保存、保留最近两个 checkpoint；上述周期分别在第 50/100 或 10/20 步保存。
异步保存仍包含前台快照阶段；退出前会等待后台写入和完整性校验完成。

**从中间 checkpoint 重新启动**：连续运行成功退出后，在所有节点使用相同的 Bash
配置和 `mmdit_train` 函数执行以下命令。它启动新进程、写入独立目录，并保持原来的
`max_steps`，分别继续执行 50 或 10 次更新。

```bash
printf -v MMDIT_SAVE_DIR '%05d.dcp' "$MMDIT_SAVE_STEP"
mmdit_train "$MMDIT_RUN_ROOT/resumed" \
  "$MMDIT_RUN_ROOT/continuous/checkpoints/$MMDIT_SAVE_DIR"
```

这里使用显式 checkpoint 路径，以便与连续运行对照。实际任务中，从原输出目录的
最新 checkpoint 接续只需将 `resume.path` 设为 `auto`，保留同一输出目录及其他配置。

训练 RNG 已改为持续随机流，确定性计算统一通过 `runtime.deterministic` 控制，
TF32 保留单独的精度设置。
本次修改已通过 B300 的 4/8 卡小模型 EP/USP/HSDP 数值回归，包含重算。
下述 128 卡完整模型历史验收不覆盖这些修改；上面的完整模型训练与续训命令仍**未验证**。

2026-09-15 的 EP8、EP16 验证使用 PyTorch 2.13.0+cu130、FA4 4.0.0b29，seed=42、
BF16 计算、FP32 梯度归约及模块精度保留策略、开启重算。额外的验收工具确认了全部
128 个 rank 的最终模型、optimizer、routing bias、数据游标和 RNG 完全一致，恢复
步骤的 loss、路由指标和样本信息也逐项一致。上面的 Native 启动命令负责训练与恢复，
不会自动执行状态比较。随机数据仅验证链路；真实数据收敛、跨拓扑恢复和完整模型的
音频/音视频/reference 长序列仍未验证。
