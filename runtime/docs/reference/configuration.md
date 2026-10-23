# 配置方式与字段

本页回答“这个字段允许什么值”。用户只需要选择一种修改方式：

1. 临时修改一两个字段：直接在 CLI 末尾增加 override；
2. 需要保存、复用或评审一组配置：把覆盖项写入自己的 YAML。

模型完整流程见对应 [One Pager](../models/index.md)，常用组合见
[通用 Override 速查](configuration-cheatsheet.md)。

## 方式一：在 CLI 修改

字段使用 `配置块.字段=值`：

```bash
vforge-train experiment=wan21_i2v_14b \
  runtime.max_steps=1000 \
  optimizer.learning_rate=2e-5 \
  checkpoint.every_n_steps=500
```

适合单次试验。CLI 只写本次与模型默认不同的值，不要复制整个 experiment。
单进程示例使用 `vforge-train`；模型 One Pager 要求多进程时，把相同 override 追加到
`torchrun -m vforge.tasks.train` 命令末尾。两种入口使用同一配置合成逻辑，启动方式见
[Quick Start](../start/index.md#launch-entry)。

## 方式二：使用自己的 YAML {#custom-config-file}

长期任务可以把覆盖项放在仓库外或自己的业务目录。用户 YAML 统一放在 `user/` 子目录，
例如创建 `/workspace/my-configs/user/wan21-local.yaml`：

```yaml
# @package _global_
defaults:
  - override /experiment: wan21_i2v_14b
  - _self_

model:
  pretrained_path: /models/Wan2.1-I2V-14B

data:
  train_manifest: /data/wan21/train.jsonl
  cache_dir: /data/wan21/cache

runtime:
  output_dir: /runs/wan21/local
  max_steps: 100

checkpoint:
  every_n_steps: 500
```

`override /experiment` 选择内置 experiment；`@package _global_` 让字段覆盖根配置，
`_self_` 确保本文件内容在 defaults 之后生效。具体生命周期仍由所用 CLI 决定，例如
训练：

```bash
vforge-train \
  --config-dir /workspace/my-configs \
  +user=wan21-local
```

`--config-dir` 告诉 Hydra 去哪个目录寻找配置，`+user=wan21-local` 加载
`user/wan21-local.yaml`。Prepare、Eval 和 Convert Checkpoint 可复用同一份 YAML；
各入口先加载自己的内置生命周期，再应用用户配置。仓库内模板见
[`examples/configs/user/wan21-local.yaml`](https://git.woa.com/ieg-infra/wmtrain/blob/master/examples/configs/user/wan21-local.yaml)。

这里的 `+` 是 Hydra 语法，不是 shell 运算符。内置 `defaults` 没有 `user` 项，所以
需要用 `+user=...` 把用户 YAML 加入本次配置；如果直接写 `user=wan21-local`，Hydra
会尝试替换一个已经存在的 `user` 项并报错。`runtime.max_steps` 等已经存在的普通字段
不需要加 `+`。

运行前可以只查看最终配置，不启动模型：

```bash
vforge-train \
  --config-dir /workspace/my-configs \
  --cfg job \
  +user=wan21-local
```

CLI override 仍具有最高优先级：

```bash
vforge-train \
  --config-dir /workspace/my-configs \
  +user=wan21-local \
  runtime.max_steps=200
```

无论使用哪种方式，运行时都会把最终结果写入
`${runtime.output_dir}/resolved_config.yaml`。字段覆盖与继承的内部规则见
[配置加载与覆盖](../development/configuration-composition.md)。

## 三种权重路径

| 字段 | 用途 | 是否包含训练状态 |
| --- | --- | --- |
| `model.pretrained_path` | 没有 resume 目标时初始化模型 | 否 |
| `resume.path` | 训练恢复或 warm-start | 取决于 PT/DCP 和选择开关 |
| `evaluation.checkpoint_path` | 评估指定权重 | 否 |

三者不能互换。格式和恢复协议见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

Wan2.2 A14B 训练通过 `model.extras.train_expert=high|low|both` 选择专家。`high` 和
`low` 只构造、加载并训练对应的一座 14B Transformer；MCore 训练只支持这两种单专家
模式，`both` 的奇偶 step 交替仅由 FSDP2 支持。离线 encode 不受该字段影响。

Wan2.2 eval 始终构造完整双专家模型。`evaluation.checkpoint_path` 可以是完整双专家
checkpoint 的字符串，也可以组合两个独立训练任务：

```yaml
evaluation:
  checkpoint_format: auto
  checkpoint_path:
    high: /path/to/high-checkpoint
    low: /path/to/low-checkpoint
```

mapping 模式严格限制 High 只写 `wan.*`、Low 只写 `wan_2.*`。两份 checkpoint 可以有
不同训练 step 和 DP 拓扑，但模型结构、condition layout、boundary 以及 MCore checkpoint
的 CP/TP 必须兼容；eval manifest 会分别记录两侧来源。

## 公共运行契约

- 首次训练 `resume.path=none`；
- final full DCP、异步保存、默认保留最近 2 个；
- 周期保存默认关闭，需要时设置 `checkpoint.every_n_steps=<N>`；
- checkpoint 目录为 `${runtime.output_dir}/checkpoints`；
- eval 产物目录为 `${runtime.output_dir}/samples`。

模型 recipe 已固化正确性和性能约束。用户通常只修改路径、步数、学习率和周期保存；
模型差异见对应 One Pager。

## Schema 字段

以下是 structured schema 的字段类型和基础默认。task YAML、experiment 和用户
override 会继续覆盖；查看一次运行的最终值请使用 `--cfg job` 或
`resolved_config.yaml`。

### `model`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `model.name` | `""` | Registry 中的模型名 | 通常由 experiment 设置 |
| `model.init_device` | `meta` | `meta`、`cpu` | 模型初始构造设备 |
| `model.weight_loading` | `auto` | `auto`、`distributed`、`rank0_broadcast`、`replicated` | 预训练权重加载策略 |
| `model.pretrained_path` | `""` | 模型 snapshot 目录 | 预训练权重与 encoder 资产根目录 |
| `model.extras` | `{}` | YAML mapping | 模型专属字段；由模型指南维护 |

### RepWorld 模型字段

`rep_world` 与其 1D/EUPE/V-JEPA variants 公开声明 `model.extras` 中的以下分组；
默认值与完整结构以对应 experiment YAML 为准。流程和验证范围见
[RepWorld](../models/rep-world.md)。

| 字段（`model.extras.` 前缀） | 语义与约束 |
| --- | --- |
| `codec.family` / `codec.variant` | VideoRAE 的 `3d`/`1d`，V-RAE 的 `eupe`/`vjepa2_1`；决定原生 geometry |
| `codec.checkpoint` / `codec.statistics` | frozen 发布权重及可选 per-channel latent stats；不再传上游 source_root |
| `codec.encode_batch_size` / `codec.decode_batch_size` | frozen codec microbatch，不是独立 training clips 数 |
| `codec.independent_state_decode` | 仅 V-RAE 可选的逐 state 解码；默认保留发布 decoder 的时序行为 |
| `states` / `frame_interval` / `random_clip` | 每 clip 原生 state 数、源视频帧步长、是否随机训练窗口 |
| `conditioner.input` | `rgb` 或 `latent`，同时决定采样时反馈 RGB 还是生成 latent |
| `conditioner.temporal_patch` | RGB 的 2/4 帧 tubelet；latent 不二次 patch，字段必须保持 2 |
| `conditioner.input_size` / `feature_layers` | RGB ViT 输入分辨率、提取的 layer 索引；独立于 frozen codec |
| `world.*` / `conditioner.*` | 两个 trainable transformer 的结构；codec channels/planes 自动派生 |
| `flow.mode` | `no_zprev`、RGB `zprev` 或无 ViT 的 `pure_tf` |
| `flow.mean/std/shift/base_shift` | 原 latent FM logit-normal 时间分布与 sampling shift |
| `use_muon` | true 使用任务局部 Muon+辅助 Adam；false 使用 AdamW |
| `initial_weights_format` | `native` 或显式 `legacy` weights-only；不是完整旧训练恢复 |
| `rollout.*` | 输出 states、初始 history、history window、CFG interval、FPS；steps/CFG scale 使用公共 evaluation 字段 |

### Gen Engine 文件输入 {#gen-engine-input}

以下字段由 `experiment=gen_engine` 声明；表中为该 experiment 的默认值。
`gen_engine_mmdit_moe32b` 继承它并选择 4096 维文本输入的完整 32B MoE 配置。
完整流程见 [Gen Engine](../models/gen-engine.md)。

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `model.extras.tensor_format` | `bundle` | `bundle`、`tensor`、`auto` | `bundle` 读取 vforge 容器中的 `embedding` / `latent`；`tensor` 读取裸 Tensor；`auto` 按实际载荷选择，不猜测轴顺序或编码语义 |
| `model.extras.audio_layout` | `ACT` | `ACT`、`ATC` | 文件中的音频轴顺序，A 为声道、C 为特征、T 为时间。`ATC` 按原声明校验后在内存转成 `ACT`，不修改源文件 |
| `model.extras.artifact_cache_dir` | `null` | 本地目录或 `null` | S3 tensor 下载缓存；`null` 使用 `$XDG_CACHE_HOME/vforge/artifacts`，未设置 XDG 时使用 `~/.cache/vforge/artifacts` |
| `model.extras.pin_artifacts` | `false` | `true`、`false` | 启动时获取所需 tensor 并固定实际 SHA-256、大小和远端对象身份，写入数据恢复契约；首次运行需要相应下载及校验 I/O |
| `model.extras.require_digests` | `true` | `true`、`false` | 是否要求 JSONL tensor 描述符提供 `embedding_digest` / `latent_digest`；不等同于原媒体的 `source_digest` |
| `model.extras.video_latent_codec_config_path` | `""` | 必填本地 JSON 路径 | 数据导出方确认的视频 posterior、归一化、空间压缩与时间映射契约 |
| `model.extras.audio_latent_codec_config_path` | `""` | 必填本地 JSON 路径 | 数据导出方确认的音频声道、归一化与时间映射契约 |

`gen_engine_mmdit_moe32b` 设置 `tensor_format=tensor`、`audio_layout=ATC`、
`pin_artifacts=true`、`require_digests=false`；两份 codec 路径保持空值，必须显式提供。
不能由模型名称或 tensor shape 推断标准化参数、posterior 状态或时间映射。
新 profile 与原 bundle 模式都支持本地文件和 `s3://bucket/key`，S3 使用标准 AWS
凭据链。JSONL 本身仍为本地文件。

单源使用 `data.train_manifest`；多源使用 `data.sources` 的 `name`、`manifest`、
`weight`，两个入口互斥。这里的 manifest 是预计算 `mmgen.sample` JSONL，不是
通用 cache Dataset 的索引子集；不设置 `kind` 或 `cache_dir`。

### `latent_cache`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `latent_cache.mode` | `offline` | `online`、`offline` | encoder 何时运行。`offline` 读 prepare 写好的 cache，训练进程不加载 VAE/text encoder；`online` 每步现场编码，prepare 必须用它。与 `data.sources` 必须一致：写 `cache_dir` 的条目需 `offline`，写 `kind` 的条目需 `online` |

像素空间的数据增强（`normal_rot_aug_*`、`use_partial_normal_aug`）只在 `online`
下生效。cache 是按原样读回的，offline 时这些开关不会报错，只是静默失效。

### `data`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `data.train_manifest` | `""` | JSON / JSONL 文件路径 | Prepare/online train 的原始样本列表 |
| `data.val_manifest` | `""` | JSON / JSONL 文件路径 | 预留给训练过程中的 validation 数据，不是 standalone eval 输入 |
| `data.sources` | `[]` | mapping / 路径列表 | 训练数据源列表，两项及以上按 `weight` 混合。每项可用 `kind` 声明现场解码的原始数据集（见下表）。Prepare 必须恰好一项 |
| `data.cache_dir` | `""` | 路径字符串 | 单个离线 cache 的简写，等价于只有一项的 `data.sources`；方便 CLI 写 `data.cache_dir=/path`。不接受列表，且不能与 `data.sources` 同时非空 |
| `data.mix_seed` | `20260726` | 整数 | 多源 `MixedSampler` 的 seed |
| `data.steps_per_epoch` | `1000000000` | 正整数 | 多源虚拟 epoch 长度 |
| `data.base_seed` | `1234567` | 整数 | 数据基础 seed |
| `data.sampler_seed` | `0` | 整数 | Sampler seed |
| `data.batch_size` | `1` | 正整数 | 每个数据 owner 的 batch；模型可施加更严格限制 |
| `data.num_workers` | `0` | 非负整数 | Dataloader worker 数 |
| `data.shuffle` | `true` | `true`、`false` | 是否 shuffle |
| `data.drop_last` | `true` | `true`、`false` | 是否丢弃最后一个不完整 batch |
| `data.pin_memory` | `true` | `true`、`false` | 是否使用 pinned host memory |
| `data.persistent_workers` | `false` | `true`、`false` | worker 是否跨 epoch 保持 |
| `data.prefetch_factor` | `null` | `null` 或正整数 | 每个 worker 预取数量 |

`data.sources` 项的通用字段：`name`、`weight`（默认 1）、`min_latents` / `max_latents`
（仅双向 CT 截断）。用 `data.sources` 覆盖上层 experiment 时，若上层已经写了非空的
`data.cache_dir`，要一并写 `data.cache_dir: ""` 把简写清掉，否则 compose 阶段报错。

`kind` 决定这一项是哪种数据源，两种模式的字段不通用：

| `kind` | 该项字段 | 含义 |
| --- | --- | --- |
| 留空（默认） | `cache_dir`、`manifest`（indexed 布局的可选子集） | 读离线 encoded cache，需 `latent_cache.mode=offline` |
| `gameplay`、`self_forcing`、`self_forcing_long_video` | `index_files`、`frames`、`height`、`width`、`max_sequence_length`（默认 512）、`extras` | 训练时现场解码原始视频，需 `latent_cache.mode=online` |

一次训练只能全是 cache 或全是 raw，混写会在 compose 阶段报错。`kind` 与 `cache_dir`
同时出现只有一种含义——prepare 把这个 raw 源编码到那个目录——训练侧会拒绝该组合。

`extras` 是自由 mapping，交给具体 dataset 类解释，不在 schema 里逐个列举：把每个
dataset 的私有开关都提升为字段，会让所有数据源都带上彼此无关的选项，而且能通过类型
检查的组合大多数依然是无意义的。代价是 `extras` 内部的拼写错误不会在这一层被发现。
Gameplay 认得 `use_partial_normal_aug`、`normal_rot_aug_*`、`camera_source_height` /
`camera_source_width`、`defer_spatial`；三类都认得 `extract_reference_frame` /
`ref_num_frames` 与 `max_num`。

`defer_spatial: true` 让 gameplay 的 worker 只解码、返回原始 uint8，把 resize、normal
单位化、旋转与 mask 混合挪到主 rank 的 device 上批量做。worker 侧不再产生 1.6 GB 的 fp32 中间量，跨进程
的载荷也从 bf16 降到 uint8。两点代价：产出的像素与 worker 内处理的不是逐比特相同（fp32
保留到最后一步、旋转在 mask 混合之前、GPU 的 antialias 是另一套 kernel），所以两种模式
写出的 cache 不可互换；以及该模式不支持 `extract_reference_frame`。

### `optimizer` 与 `scheduler`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `optimizer.name` | `adamw` | `adamw` | 当前唯一 optimizer |
| `optimizer.learning_rate` | `1e-5` | 非负浮点数 | 学习率 |
| `optimizer.weight_decay` | `0.0` | 非负浮点数 | Weight decay |
| `optimizer.betas` | `[0.9, 0.999]` | 两个 `[0, 1)` 浮点数 | Adam betas |
| `optimizer.eps` | `1e-8` | 正浮点数 | Adam epsilon |
| `optimizers` | `{}` | name -> optimizer mapping | 专用 trainer 的独立 optimizer；普通 Trainer 忽略 |
| `scheduler.name` | `none` | `none`、`constant`、`constant_with_warmup` | Scheduler 类型 |
| `scheduler.factor` | `1.0` | 浮点数 | `constant` 的缩放因子 |
| `scheduler.warmup_steps` | `0` | 非负整数 | `constant_with_warmup` 要求大于 0 |

H3 recipe 使用 `scheduler.name=none`。不要把 Wan 的 scheduler 习惯直接套到 H3。

`optimizers` 中每个 named optimizer 使用与 `optimizer` 相同的
`name/learning_rate/weight_decay/betas/eps` 字段。`wan22_ti2v_5b_dmd` 要求
`generator` 和 `critic` 两项；它们分别只更新 generator 和 fake_score tower。
`optimizer` 单数保留给现有 recipe 和兼容工具，不控制 `PhasedTrainer`。

### `training.recipe`、`training.models` 与 `training.dmd`

多模型训练不再伪装成一个 `model.name`。`training.recipe=dmd` 时，
`training.models` 显式列出算法所需 role；每项包含 `model`、模型插件自行解释的
`config`、`pretrained_path`、`init_weights`、`init_device` 和 `weight_loading`。
标准 `distribution_matching` 算法要求 `generator`、`real_score`、`fake_score`。
`training.dmd.algorithm` 选择完整计算图，其余字段是该算法的配置，都会进入 semantic
config fingerprint。

Wan DMD 不定义专用 cache schema：`latent_cache.mode=offline` 读取普通 CT cache，
`online` 读取 typed raw source 并调用普通 Wan encoder。生产配置优先使用 offline。

| 字段 | 类型 / 默认 experiment 值 | 作用 phase | 实际语义 |
| --- | --- | --- | --- |
| `training.models.generator.init_weights` | path / `generator.pt` | 初始化 | 独立、扁平的 causal generator state |
| `training.models.generator.config.frame_per_block` | int / `3` latent frames | causal generator | 非 anchor block 大小 |
| `training.models.generator.config.first_frame_own_block` | bool / `true` | causal generator | 第 0 帧独立 block |
| `training.models.real_score.init_weights` | path / `real-score.pt` | 初始化 | 独立 frozen teacher，strict load |
| `training.models.fake_score.init_weights` | path / `fake-score.pt` | 初始化 | 独立 critic；可与 real-score 使用同一初始文件 |
| `rollout.denoising_steps` | int list / `[1000,938,833,625,0]` | generator、critic rollout | 五个边界、四次 forward；terminal zero 不 forward |
| `rollout.terminal_zero_is_boundary` | bool / `true` | rollout | 为 true 时数组必须以 0 结束 |
| `rollout.last_step_only` | bool / `true` | generator | 每个 target block 仅最后 denoise forward 保留梯度 |
| `rollout.context_noise` | float / `0` | rollout | 已提交 context 的 renoise 强度 |
| `score.num_train_timesteps` | int / `1000` | 两个 score | flow schedule 长度 |
| `score.timestep_shift` | float / `5.0` | 两个 score | sample 后的 Wan timestep shift |
| `score.sampled_timestep_min` | int / `1` | score sampling | 原始离散 sample 下界 |
| `score.effective_timestep_clamp` | `[min,max]` / `[20,980]` | 两个 score | shift 后 score forward 的有效区间 |
| `score.real_guidance_scale` | float / `5.0` | real_score | `cond + scale*(cond-uncond)` |
| `score.fake_guidance_scale` | float / `0.0` | fake_score | 0 表示只执行 conditional score |
| `score.denoising_loss` | enum / `flow` | critic | 第一阶段只支持 flow matching |
| `score.mse_weight` | float / `0.0` | generator | 额外 MSE 权重，0 表示关闭 |
| `schedule.generator_every_n_steps` | int / `5` global steps | `PhasedTrainer` | step 0 开始；critic 仍每 step |
| `ema.enabled` | bool / `true` | generator update 后 | 是否维护 generator EMA |
| `ema.decay` | float / `0.99` | EMA | `ema=decay*ema+(1-decay)*generator` |
| `ema.start_step` | int / `200` global steps | EMA | 第一次 eligible generator step 直接初始化 |
| `ema.storage` | enum / `sharded_fp32` | checkpoint | 每卡保存当前 FSDP/DTensor shard |

训练长度由 data source 和 generator 的 `max_frames` 决定；`frame_per_block` 的单位是
latent frame。KV cache 的 `current_start/global_end/q_len/kv_len` 单位是 token，
二者不能混用。完整执行顺序见 [DMD 训练](../features/dmd-training.md)。

`ltx23_dmd` 选择 `block_distribution_matching` 配置，基于现有 DMD 扩展数值规则，复用三塔装配、
phase 调度和双 optimizer。标准 `distribution_matching` 的配置与数值语义不变。

- `rollout.denoising_steps=[999,757,522,0]` 定义边界，末尾零不执行 denoise。
  随机退出与 clean-history commit 是该算法的固定行为；
  `same_step_across_blocks=true` 让所有块共用退出位置，跨 rank 广播。
- `score.sampled_timestep_min/max=20/980` 为整数均匀采样的闭区间，
  `num_train_timesteps=1000` 为模型 sigma 表的索引范围。
- `component_guidance_scales={video:3,audio:5}` 分别指定 CFG scale；公式固定为
  `cond+(scale-1)*(cond-uncond)`。分布梯度按模态单独归一化，分母加 `1e-8`，
  并使用任务的有效区域排除干净首帧。
- `component_weights={video:1,audio:1}` 指定模态权重。surrogate 固定先计算各块
  均值再平均；平方、均值保持输入精度，块标量在 FP32 累加。critic loss 同样
  保持输入精度。noising 固定使用 FP32 sigma 运算后转换回输入 dtype。
  这些是算法语义，不再暴露 guidance/precision/mask 等模式切换。
- LTX experiment 的 `data.sampler_seed` 跟随 `runtime.seed`；G/C 的偏移分别为
  0/7919，两流每全局步各推进一次。`data.drop_last=true` 丢弃不足一个 DP 组的
  尾桶，匹配参考分桶策略；小数据专项测试可显式设为 false，但不能与参考默认
  顺序直接比较。两侧仍须使用相同、规范排序的样本列表。
- `training.models.generator.config` 中 `first_block_latent_frames=4`、
  `block_latent_frames=3`、`max_train_frames=241` 定义布局，前两者单位为 latent
  frames，后者单位为 pixel frames。`ref_audio_seconds=1.5`、`ref_position_gap=5`
  单位为秒，`ref_follow=true` 由模型处理 reference RoPE。
- 各角色的 `pretrained_path` 指向 base checkpoint，`init_weights` 指向相应 TF/bid
  单文件 safetensors；`config.source_prefix` 默认为空，可指定 checkpoint 键前缀。
  单文件加载使用 `weight_loading=auto`（解析为 rank0 broadcast）或 `replicated`。
- generator 的 `config.text_encoder_path`、`negative_prompt` 和 `max_text_length`
  用于启动时一次性负向编码。正向条件使用 cache；不支持 DMD raw 编码入口。

完整命令与适用边界见 [LTX-2.3](../models/ltx23.md)。

### `runtime`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `runtime.max_steps` | `0` | 非负整数 | 整次训练目标总步数 |
| `runtime.run_backward` | `true` | `true`、`false` | 是否执行 backward |
| `runtime.seed` | `42` | 整数 | 启动时播种全局 RNG，并初始化训练命名随机流；不随 step 重置 |
| `runtime.deterministic` | `false` | `true`、`false` | 统一控制 PyTorch、cuDNN 和 FA4 确定性；开启时关闭 cuDNN benchmark，不改变 RNG/TF32 |
| `runtime.precision` | `bf16` | `bf16` / `bfloat16`、`fp16` / `float16`、`fp32` / `float32` | 运行计算 dtype |
| `runtime.autocast` | `false` | `true`、`false` | 与 FSDP mixed precision 独立 |
| `runtime.gradient_clip_val` | `0.0` | 非负浮点数 | 0 表示不裁剪 |
| `runtime.log_every_n_steps` | `1` | 正整数 | Step 日志频率 |
| `runtime.log_timing` | `true` | `true`、`false` | 是否记录阶段耗时 |
| `runtime.output_dir` | `outputs` | 目录路径 | Schema 默认；train task 最终默认 `outputs/train` |
| `runtime.weights_strict` | `false` | `true`、`false` | train/eval 的 pretrained、warm-start 和评估 checkpoint 权重是否严格匹配 key/shape；`resume.strict` 另管完整训练状态恢复 |

### `logging`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `logging.jsonl` | `true` | `true`、`false` | 写 JSONL step 日志 |
| `logging.jsonl_filename` | `train_steps.jsonl` | 文件名 | JSONL 日志文件 |
| `logging.wandb_enabled` | `false` | `true`、`false` | 是否启用 W&B |
| `logging.wandb_project` | `""` | 字符串 | 启用 W&B 时必填 |
| `logging.wandb_entity` | `null` | `null` 或字符串 | W&B entity |
| `logging.wandb_group` | `null` | `null` 或字符串 | W&B group |
| `logging.wandb_name` | `""` | 字符串 | 启用 W&B 时必填 |
| `logging.wandb_tags` | `[]` | 字符串列表 | W&B tags |
| `logging.wandb_mode` | `null` | `null` 或 W&B mode；常用 `online`、`offline`、`disabled` | 透传给 W&B，不由 vforge 枚举 |
| `logging.wandb_save_dir` | `""` | 目录路径 | W&B 本地产物目录 |
| `logging.wandb_include_optimizer_in_backward` | `false` | `true`、`false` | backward 计时是否包含 optimizer |
| `logging.wandb_exclude_dataloader_from_step` | `false` | `true`、`false` | step 计时是否排除 dataloader |

### `parallel`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `parallel.engine` | `fsdp2` | `fsdp2`、`megatron` | 训练引擎 |
| `parallel.dp_mode` | `fsdp2` | `fsdp2`、`ddp` | Data parallel 实现 |
| `parallel.dp_size` | `auto` | `auto` 或正整数 | 自动推导或显式指定 DP 宽度 |
| `parallel.dp_replicate_size` | `1` | 正整数 | HSDP 外层 replica 数；大于 1 时必须同时显式设置 shard |
| `parallel.dp_shard_size` | `1` | 正整数 | FSDP/HSDP 内层 shard 宽度 |
| `parallel.usp_size` | `1` | 正整数 | Ulysses 宽度 |
| `parallel.cp_size` | `1` | 正整数 | Context parallel；当前由 Megatron 提供 |
| `parallel.tp_size` | `1` | 正整数 | Tensor parallel 宽度 |
| `parallel.ep_size` | `1` | 正整数，整除 world size 及每个 EP bank 的专家数 | 预览：FSDP2 + EP，可叠加 USP；EP 划分 DP×USP 域，大小无需等于 USP；模型只声明需分片的专家 bank；与 TP/CP/PP/HSDP 互斥 |
| `parallel.include_usp_in_fsdp` | `true` | `true`、`false` | FSDP mesh 是否包含 USP 维度 |
| `parallel.gradient_checkpointing` | `false` | `true`、`false` | Activation recompute；experiment 通常开启 |
| `parallel.reshard_after_forward` | `true` | `true`、`false` | FSDP forward 后是否 reshard |
| `parallel.fsdp_leaf_module_paths` | `[]` | 模块路径字符串列表 | 额外 FSDP leaf |

组合矩阵和选择方法见[分布式运行](../features/distributed-training.md)。

### `parallel.mixed_precision`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `enabled` | `true` | `true`、`false` | 是否应用 FSDP mixed-precision policy |
| `param_dtype` | `bfloat16` | `bfloat16` / `bf16`、`float32` / `fp32`、`float16` / `fp16` 或 `null` | 参数计算 dtype |
| `reduce_dtype` | `float32` | 同上 | 梯度归约 dtype |
| `output_dtype` | `null` | 同上 | 输出 dtype |
| `buffer_dtype` | `null` | 同上 | Buffer dtype |
| `cast_forward_inputs` | `true` | `true`、`false` | 是否预转换 forward 浮点输入 |
| `preserve_module_paths` | `[]` | 模块路径字符串列表 | 独立成不应用 mixed policy 的 FSDP group，保留模块现有参数/buffer dtype。相对 FSDP 执行根；`*` 只匹配**一整段**（`blocks.*.norm1`、`blocks.*`），wrap 时展开，不要写 `blocks.norm*` |
| `preserve_forward_input_paths` | `[]` | 模块路径字符串列表；`.` 表示 execution root | 保留参数/归约 policy，仅关闭指定 group 的 FSDP forward 输入预转换。同样支持 `*` 整段通配 |
| `contract_mode` | `append` | `append`、`replace` | 追加或替换模型 precision contract |

模型默认的保留模块只来自它的 `_keep_in_fp32_modules`，`FSDPPrecisionContract` 只能声明
`preserve_forward_input_paths`；`append` 合并模型声明与配置，`replace` 仅使用配置路径。
两个 preserve 字段的区别、引入原因和 H3 示例见
[随机性与精度](../features/execution-and-precision.md)。

`preserve_module_paths` 只作用于模块：FSDP 让目标单独成一个不带 mixed policy 的
组来保留 dtype，而单个权重没有自己的组。路径指向单个权重（如
`blocks.*.scale_shift_table`）时会被跳过，rank 0 会打一行说明，该权重仍按
`param_dtype` 参与计算；路径什么都匹配不到仍然直接报错。

### `parallel.mcore_ddp`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `overlap_grad_reduce` | `true` | `true`、`false` | Megatron DDP 梯度归约重叠 |
| `overlap_param_gather` | `true` | `true`、`false` | Megatron DDP 参数收集重叠 |

### `attention`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `attention.backend` | `auto` | `auto`、`sdpa`、`flash_attention_3`、`flash_attention_4` | Attention backend |
| `attention.allow_fallback` | `true` | `true`、`false` | 缺少目标 kernel 时是否允许 fallback |
| `attention.debug` | `false` | `true`、`false` | 输出 attention 诊断 |

`attention.allow_fallback=true` 只允许与 `backend=auto` 组合。
确定性由 `runtime.deterministic` 统一控制；开启时拒绝当前不支持确定性的 FA3 路径，
`sdpa`、`sdpa_math` 和 FA4 可使用该字段。

### `attention_offload`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `attention_offload.enable` | `false` | `true`、`false` | FA 算子内 preserved activation CPU offload（Native preview） |
| `attention_offload.preserve_fa_call_indices` | `[0]` | 非负整数列表或 `null` | invocation 内 FA 序号；null=全部 |
| `attention_offload.min_tensor_bytes` | `1048576` | 非负整数 | 小于该字节数的 tensor 不 offload |
| `attention_offload.pin_memory` | `true` | `true`、`false` | CPU offload tensor 是否使用 pinned memory |
| `attention_offload.max_cpu_bytes` | `null` | `null` 或正整数 | 单 step live CPU tensor budget |
| `attention_offload.async_transfer` | `false` | `true`、`false` | 异步 D2H/H2D 与 backward prefetch；true 要求 pin_memory=true |

必须与 activation recompute 和 FA backend 配合才有收益。FA3 支持 dense/varlen，
FA4 还支持 teacher-forcing sparse；真实 GPU gate 完成前 Native 路径为 preview。默认
`[0]` 不增加旧 execution fingerprint 字段，非默认选择会记录。详见
[激活重算与 Offload](../features/activation-recompute-and-offload.md)。

### `checkpoint`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `checkpoint.enabled` | `true` | `true`、`false` | 是否保存 |
| `checkpoint.every_n_steps` | `0` | 非负整数 | 0 表示关闭周期保存 |
| `checkpoint.dirpath` | `""` | 空字符串或目录路径 | 空值解析为 `${runtime.output_dir}/checkpoints` |
| `checkpoint.state` | `full` | `weights`、`full` | 保存模型权重或完整训练状态 |
| `checkpoint.format` | `dcp` | `pt`、`dcp`、`mcore` | `full+pt` 不支持；`mcore` 仅用于 Megatron engine |
| `checkpoint.save_final` | `true` | `true`、`false` | 结束时保存 |
| `checkpoint.keep_last` | `2` | `null` 或正整数 | 保留最近的已提交 checkpoint；null=不清理 |
| `checkpoint.async_save` | `true` | `true`、`false` | DCP 与 MCore 原生异步保存；MCore 需要 NVRx |
| `checkpoint.save_optimizer` | `true` | `true`、`false` | full DCP 是否含 optimizer |
| `checkpoint.save_trainer_state` | `true` | `true`、`false` | 是否含 scheduler/loader/RNG sidecar |
| `checkpoint.empty_cache_before_save` | `true` | `true`、`false` | 保存前是否清 CUDA cache；关掉会让 DCP 的 state dict 拷贝和训练留下的碎片化 reserved 段争显存 |

### `resume`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `resume.path` | `none` | `auto` / `latest`、禁用值或显式 PT/DCP/MCore 路径 | 选择恢复目标 |
| `resume.weights` | `true` | `true`、`false` | 是否恢复 weights |
| `resume.optimizer` | `true` | `true`、`false` | 依赖 `weights=true` |
| `resume.training_state` | `true` | `true`、`false` | 依赖 `optimizer=true`；原子恢复训练状态 |
| `resume.strict` | `null` | `true`、`false`、`null` | null 时由 training_state 推导 |
| `resume.stage_remote` | `false` | `true`、`false` | 未命中 S3 映射时启用 filesystem staging |

选择性恢复、跨 topology 和 strict 规则见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

### `storage.s3_mounts`

每一项描述一个 mount path 到真实对象存储身份的映射。

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `mount_path` | `""` | 容器内绝对目录路径 | 用于匹配 checkpoint 路径 |
| `bucket` | `""` | Bucket 名称 | 真实对象存储 bucket |
| `region` | `us-west-2` | Region 字符串 | Bucket region |
| `key_prefix` | `""` | 空字符串或对象 key 前缀 | 限定 mount 对应的远端前缀 |
| `endpoint_url` | `null` | `null` 或 HTTP(S) URL | 自定义 endpoint；会影响 CRT 可用性 |
| `addressing_style` | `auto` | `auto`、`path`、`virtual` | S3 addressing style |

### `diagnostics`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `diagnostics.level` | `basic` | `basic`、`parity` | 诊断级别 |
| `diagnostics.log_local_parameter_stats` | `false` | `true`、`false` | 是否记录本地参数统计 |

### `evaluation`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `evaluation.checkpoint_format` | `auto` | `auto`、`pt`、`dcp`、`mcore` | 评估权重格式 |
| `evaluation.checkpoint_path` | `""` | checkpoint 路径字符串、`latest`/`auto`，或 Wan2.2 `{high, low}` mapping | 评估权重；`latest`/`auto` 通过 `checkpoint.dirpath` 解析 |
| `evaluation.evaluate_all_checkpoints` | `false` | `true`、`false` | 按 completed step 顺序评估 `checkpoint.dirpath` 内全部完整 checkpoint |
| `evaluation.data_source` | `""` | `validation`、`inference`、`cache` | Standalone eval 输入类型 |
| `evaluation.manifest` | `""` | Manifest 文件路径 | `validation`/`inference` 使用样本 JSON/JSONL；`cache` 直接指向 cache root 的 `manifest.json` |
| `evaluation.output_dir` | `""` | 空字符串或目录路径 | Eval task 最终默认 `${runtime.output_dir}/samples` |
| `evaluation.num_samples` | `null` | `null` 或非负整数 | 限制样本数 |
| `evaluation.sample_index` | `null` | `null` 或非负整数 | 在 DP 分片前选择一个全局样本；设置时 `num_samples` 只能为 `null` 或 `1` |
| `evaluation.sample_selection` | `first` | `first`、`evenly_spaced` | 在 DP 分片前选择前 N 个，或按 manifest 来源均衡分配后在各来源内等间隔选择 |
| `evaluation.seed` | `42` | 整数 | 评估 seed |
| `evaluation.skip_existing` | `true` | `true`、`false` | 是否跳过已有产物 |
| `evaluation.num_inference_steps` | `null` | `null` 或正整数 | 生成采样步数；`null` 使用模型 eval 默认值 |
| `evaluation.guidance_scale` | `null` | `null` 或非负浮点数 | CFG scale；`null` 使用模型 eval 默认值 |
| `evaluation.negative_prompt` | `null` | `null` 或字符串 | CFG negative prompt；`null`/空字符串使用模型默认值 |
| `evaluation.rollout_mode` | `""` | `""`、`full_recompute`、`incremental_kv` | Causal eval backend；空字符串使用模型默认值，只对 causal variant 有效 |
| `evaluation.gridnormal.enabled` | `false` | `true`、`false` | 是否在支持的 bidirectional TI2V eval 上展开 grid/normal 多变体评估 |
| `evaluation.gridnormal.variants` | `null` | `null` 或非空 variant 列表 | 限制 gridnormal 变体集合；设置时必须同时启用 `evaluation.gridnormal.enabled` |
| `evaluation.online_ref.num_frames` | `null` | `null` 或非负整数 | 从 validation 目标视频分层抽取固定数量的 ref；`0` 表示本次不使用 ref |
| `evaluation.online_ref.frame_indices` | `null` | `null` 或非空、唯一的非负整数列表 | 直接指定 validation 目标视频的 ref 帧号，不消费选帧 RNG |

`evaluation.data_source` 和 `evaluation.manifest` 必须同时设置。`cache` manifest
未完成或 codec/schema 不匹配时直接报错；
`data.val_manifest` 和训练数据源（`data.sources` / `data.cache_dir`）不会被 eval
隐式回退使用。
`sample_selection=evenly_spaced` 在 manifest 含 `source` 或 `meta.source` 时，按来源
首次出现顺序前置分配 `num_samples` 配额，再在各来源内部做含首尾端点的等间隔选样；
未标记来源的样本作为一个独立来源。manifest 完全没有来源标记时，对完整列表做全局
等间隔选样。上述选择都先于 DP rank 分片。

模型形态和 eval 执行方式分开配置。`model.extras.variant` 唯一决定模型是
bidirectional 还是 causal；`evaluation.rollout_mode` 只在 causal variant 上选择
`full_recompute` 或 `incremental_kv`。为 bidirectional variant 设置非空
`rollout_mode`，或为 causal variant 启用 gridnormal，都会在 task 路由阶段报错。
当前 Wan TI2V 的 `incremental_kv` 还要求 temporal patch size 为 `1` 且
`parallel.usp_size=1`。

`evaluation.online_ref.num_frames` 与 `frame_indices` 互斥，只适用于 ref-token variant
和 `evaluation.data_source=validation`。manifest 显式 `ref`/`reference` 拥有最高优先级；
仅未提供显式 ref 时才读取目标视频。未设置 override 时，
`model.extras.ref_num_frames_online` 继续表示随机数量上限。

Gridnormal 的完整变体集合是 `full`、`normal_to_video`、
`normal_fg_to_video`、`normal_bg_to_video`、`grid_to_video`、
`ref_normal_to_video`、`ref_grid_to_video` 和 `ti2v`。`variants=null` 时，
Wan Taichu 的 `model.extras.variant=wan22_ti2v_5b_ref` 默认运行全部八个；只运行五个
无-ref 变体时，通过 `evaluation.gridnormal.variants` 显式列出
`normal_to_video`、`normal_fg_to_video`、`normal_bg_to_video`、`grid_to_video` 和
`ti2v`。ref projection 能力只从 `model.extras.variant` 推导。

`evaluation.evaluate_all_checkpoints=true` 时，`evaluation.checkpoint_path` 只能为空、
`auto` 或 `latest`，实际 checkpoint 由训练生命周期共用的
`discover_checkpoints()` 发现，产物写到
`evaluation.output_dir/checkpoint-<step>/`。当同一步同时存在多种完整格式时，
目录名追加格式后缀以避免覆盖。单 checkpoint 的 `latest`/`auto` 同样复用
`resolve_checkpoint_path()`，不会在 eval 内维护第二套 latest 规则。

Eval cache 使用 vforge cache root contract。外部生产方必须把
`evaluation.manifest` 指向 cache 根目录下的 `manifest.json`；该文件至少包含
`schema_version`、`bundle_version`、`codec_name`、
`codec_schema_version`、有序 `sample_ids`、`completed`、`encoding_fingerprint`
和 `field_specs`。`completed` 必须等于样本数；各 bundle 按 manifest 的
`bucket_rule` 定位，并在读取时由模型 codec 校验必需字段。

### `convert_checkpoint`

| 字段 | 默认值 | 可选值 / 格式 | 用途与约束 |
| --- | --- | --- | --- |
| `convert_checkpoint.input_path` | `""` | DCP 目录路径 | 转换输入 |
| `convert_checkpoint.output_path` | `""` | `.pt` 文件路径 | 转换输出 |
| `convert_checkpoint.dry_run` | `false` | `true`、`false` | 只验证，不写输出 |
| `convert_checkpoint.component_path` | `""` | 例如 `generator` | 只导出 RoleModelBundle 中的指定模型 |
| `convert_checkpoint.weights` | `raw` | `raw`、`ema` | 导出即时参数；或指定模型对应的 EMA |

### `limitations`

`limitations` 记录当前公开 recipe 的验收范围，不是 launcher 参数或可调功能开关：

| 字段 | 固定值 | 含义 |
| --- | --- | --- |
| `gradient_accumulation_steps` | `1` | 当前只支持 1 |
| `expert_parallel` | `false`（兼容默认值） | 旧验收标记，不再禁止 EP；实际能力由 `parallel.ep_size`、engine 和模型声明检查 |
| `ring_attention` | `false` | 未公开支持 |
| `train_condition_encoders` | `false` | 条件 encoder 冻结 |
| `cross_attention_ulysses` | `false` | 未公开支持 |
| `padded_cross_attention_requires_varlen` | `true` | Padded cross attention 需 varlen |

除兼容字段 `expert_parallel` 外，修改这些固定值会立即失败。EP 使用普通
`parallel.ep_size=2` override，无需修改 `limitations`。多节点能力由 `parallel`
topology 和各模型验收范围描述，不再
使用 `limitations.multi_node`；旧配置中的 `multi_node=false` 仅作为兼容输入读取并从
resolved config 移除。


### Block DMD backward 方式

`training.dmd.rollout.backward_mode` 仅由 `block_distribution_matching` 消费：
`retained_graph`（默认）保留各块退出步图并统一 backward；`block_replay` 先无梯度
rollout，再逐块重放、立即 backward。两者每个 phase 只执行一次 clip 和 optimizer。
后者当前要求 FSDP2 engine，不能与 activation offload 同用。该字段已在
`ltx23_dmd` experiment 声明，可直接用普通 Hydra override 设置。


梯度裁剪实现由训练组件装配，不提供实现选择开关。公共默认使用逻辑模型范数；
LTX DMD 通过多阶段组件的裁剪工厂使用 PyTorch/DTensor 范数，只裁剪当前 phase
的模型塔，要求 FSDP2、TP=1。其他模型的默认裁剪行为保持不变。

`runtime.gradient_norm_dtype=float64` 仅提高梯度范数的计算精度，不修改 forward、
FSDP reduce、参数梯度或 optimizer 状态的 dtype；默认是 `float32`。当前 FP64
由 LTX DMD 的精度感知裁剪工厂支持；不支持此精度的 recipe 会在运行时初始化前报错。
FP64 路径逐参数计算双精度范数，再用 PyTorch `get_total_norm` 和
`clip_grads_with_norm_` 归约、裁剪，不同时复制整个模型塔的梯度。
实际裁剪语义与范数精度记录在 execution fingerprint 中；原 logical/FP32 和
DMD torch/FP32、torch/FP64 的 checkpoint 字段保持兼容，严格恢复禁止静默切换。
