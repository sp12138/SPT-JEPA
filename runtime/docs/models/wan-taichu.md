# Wan Taichu

`wan_taichu` 是独立维护的 Native 模型系列。模型实现位于 `vforge/models/wan_taichu/`，
任务装配位于 `vforge/tasks/wan_taichu/`，训练使用统一 Native 入口。
评估工作流见[Wan Taichu 评估契约](../development/wan-taichu-evaluation.md)。

## 公开入口

只注册一个模型入口：

```yaml
model:
  name: wan_taichu
  extras:
    variant: wan22_ti2v_5b
```

`variant` 是唯一的模型形态选择器：

- `wan21_i2v_14b`：普通双向 Wan 2.1 I2V 14B；
- `wan21_i2v_14b_audio`：带 speak/listen Audio 注入的 Wan 2.1 I2V 14B；
- `wan22_a14b`：high/low-noise 双 expert Wan 2.2 A14B；
- `wan22_ti2v_5b`：普通双向 Wan 2.2 TI2V 5B；
- `wan22_ti2v_5b_ref`：带 in-context ref token 的 TI2V 5B；
- `wan22_ti2v_5b_causal_teacher`：doubled `[clean | noisy]` block-causal teacher。

不增加 `training_mode` 或 `temporal_mode`。这两个名字要求使用者理解实现差异，却没有
提供比显式 variant 更多的信息。条件流由
有序的 `condition_streams` 表达，例如：

```yaml
model:
  name: wan_taichu
  extras:
    variant: wan22_ti2v_5b_ref
    condition_streams: [grid, normal]
```

同一 `model.name` 可以有多个 experiment preset，但每个 preset 只保存该 variant 的模型
差异。Hydra 公共训练、checkpoint 和运行默认仍以 `vforge/hydra_config.py` 为真源。

## 生产实验模板

仓库提供两份可移植的 user YAML，用于配置多数据源训练：

- `examples/configs/user/wan-taichu-5b-ref-prod.yaml`：5B Ref 的三源
  `0.5/0.3/0.2` mixture，source latent 长度分别为 `21–38`、`21–150`、
  `150–450`，使用 `[grid, normal]`、Ref 在线采样上限 `6`、USP `4`；
- `examples/configs/user/wan-taichu-a14b-prod.yaml`：A14B 的两源 `0.2/0.8`
  mixture，source latent 长度分别为 `21–38`、`38–150`，使用
  `[rgb, grid, normal]`、USP `8`，仅训练 high-noise `wan` expert，并在 step
  `0–4000` 将有效最大长度从 `38` 线性提升到 `150`；同时按生产实验开启
  FlashAttention activation offload，使用 `64 GiB` CPU 上限和异步传输。

Wan Taichu 模型与官方 Wan 一样保留 FSDP forward-input precision contract；生产模板
必须保留 `preserve_forward_input_paths=(".", "norm_out")`。实验仍允许通过
`model.extras.wan_use_fsdp_precision_contract=false` 显式关闭该 contract 来做归因，但
不能把关闭状态写入生产模板，也不能同时取消根模块和 `norm_out` 中的任意一项。

模板中的模型、cache 和输出目录都是明确占位路径。`model.pretrained_path` 直接接受
weights-only `.pt/.pth`；task 从
`model.extras.condition_pretrained_path` 指向的官方 Diffusers 目录读取 Transformer 结构和
冻结 encoder，再把 `.pt` 中的 `wan/wan_2` 权重路由到 Native 模型。这里仍是
weights-only warm-start，不是 Native resume：原始 `.pt` 不能写入 `resume.path`，也不会
恢复 optimizer、scheduler、global step 或 RNG 状态。

两份生产模板读取离线 v3 cache。为支持 CFG condition dropout，模板设置
`load_vae_for_dropout_fallback=true`，使用冻结 VAE 生成
grid/normal 的均匀像素 fallback latent。数据链若已经把
`condition_fallback_latents_grid` 和 `condition_fallback_latents_normal` 写入每条 cache，
可关闭该开关，训练 task 会直接读取预计算 fallback。

## 所有权与依赖方向

`vforge/models/wan_taichu/` 只拥有模型语义：

- 一个由 variant 参数化的 `WanTaichuModel` 和一套 package-local Transformer 实现；新增
  variant 不增加平行 wrapper 类；
- Attention、RoPE、Transformer block、condition embedding 和 Teacher-Forcing
  forward；
- 冻结资产 encoder 的输入/输出语义；
- ref projection、ref RoPE 和 ref token 拼接；
- block-causal teacher attention；
- audio encoder、audio injector 和 block 级 residual；
- Gameplay 相机坐标到 grid pixels 的渲染语义；
- 不 import `vforge.models.wan.v21`、`vforge.models.wan.v22` 或相应版本化 task。

`WanTaichuModel` 是算法开发的统一扩展点。单 expert variant 在该模型下注册
`transformer`，A14B 把同一套 Transformer 实现实例化为 `wan` 和 `wan_2` 两个 expert；
variant 负责提供 architecture、attention、expert、channel 和可选能力参数，不通过新增
模型类表达配置组合。

`vforge/tasks/wan_taichu/` 拥有任务语义：

- variant 配置校验和模型装配；
- raw/cache batch 统一化；
- condition dropout、noise/timestep、target、loss 和 expert 路由；
- Diffusers 权重源、patch widening、ref/audio 初始化。

Attention kernel/backend、Teacher-Forcing sparsity contract、Ulysses、FSDP、Trainer、
checkpoint manager 和 batch broadcast 继续复用 vforge core。`wan_taichu` 不复制这些
公共设施。官方 Wan 与 Wan Taichu 的模型代码
分开维护：前者可持续跟进开源实现，后者作为算法同学直接修改的业务模型，不通过 import
继承前者的版本实现。

## 训练预检

`wan_taichu` 在公共 train builder 旁注册 `train_preflight`。训练入口把已解析
配置转为 `TrainRequest` 后立即调用它，早于 resume 路径解析、process group、CUDA
runtime、模型、dataloader 和权重构造。该回调只复用无分配的 variant/request 校验：

- 普通双向、ref 和 causal teacher variant 都会在这里校验 variant、condition stream、
  `boundary_ratio`、`train_expert`、dropout、ref 采样、causal block 参数和阶段范围等
  静态 contract；
- `wan22_a14b` 使用 Megatron 时不支持 `train_expert=both`，必须显式选择
  `high` 或 `low`；
- FSDP2 可使用 `train_expert=both`，Megatron 的 `high`/`low` 单 expert 也可以
  通过预检；
- 当前 Wan Taichu 不支持 Tensor Parallel（TP），`parallel.tp_size` 必须为 `1`。

这些规则仍由 train builder 的共用解析 helper 作最终校验；preflight 只负责让已知
不可运行的组合在分配 GPU 资源之前失败。通用回调 contract 见
[新增 Native 模型](../development/adding-native-model.md#train-preflight)。

## Variant 基线

### `wan21_i2v_14b`

- latent channel `z_dim=16`，每个 condition stream 是 `4` 个 mask channel 加 `16` 个
  latent channel；
- 默认 `condition_streams=[normal]`，允许 `rgb`、`normal`；
- 默认 flow-match shift 为 `1.0`；
- 可使用冻结 VAE、CLIP 或 DINO 条件。

CLIP、DINO 和 added-KV 只支持 Wan21，Wan22 TI2V/A14B 不启用这些分支。

### `wan21_i2v_14b_audio`

- 继承 Wan 2.1 的 channel 和普通双向 Transformer 语义；
- 默认 `condition_streams=[rgb]`、flow-match shift `5.0`；
- 支持 `wav2vec2` 和 `wav2bert` stage A/B，以及模型内 stage C；
- 支持 speak/listen 两组注入层、局部窗口、逐样本 Audio dropout 和显式 zero-init；
- 当前训练数据把 `audio_features` 接入 speak；模型接口另接受可选
  `audio_features_l` 作为 listen 输入，不会自动把 speak 复制给 listen；
- Audio Ulysses 要求 `T*S` 可被 `usp_size` 整除，不接受 sequence padding；当前不支持
  Audio 与 Context Parallel 或 Tensor Parallel 组合；
- 可选 CLIP 条件；Audio 关闭时普通模型输出必须不受影响。

### `wan22_a14b`

- `z_dim=16`、每流 `4+16` channel；允许 `rgb`、`grid`、`normal`，默认 `[rgb]`；
- 两个 expert 使用 `wan` 和 `wan_2` 命名，边界默认是
  `0.9 * num_train_timesteps`；
- flow-match shift 默认 `3.0`；一个 micro-batch 只能位于边界的一侧；
- widening 只扩展输入 projection，两个 expert 使用相同 channel layout。

### `wan22_ti2v_5b`

- `z_dim=48`，condition stream 没有 mask prefix；允许 `rgb`、`grid`、`normal`，默认
  `[normal]`；
- flow-match shift 默认 `5.0`；
- frame 0 使用干净 `first_frame_latents` 覆盖 noisy latent，对应 token timestep 为
  `0`，且默认不计入 loss；
- 其余帧使用同一个 micro-batch scalar timestep 展开成 frame-major per-token timestep。

### `wan22_ti2v_5b_ref`

- 继承普通 TI2V，默认 `condition_streams=[grid, normal]`；
- ref latent channel 为 `48`，通过独立 `patch_embedding_ref` 和 ref RoPE 生成 token；
- ref latent 使用自己的空间 patch grid，不要求 H/W 与 target latent 相同；生产 sidecar
  可能保留 crop 前的一行 latent，主序列和 ref 序列只要求 batch/channel 一致；
- 外部 `ref_frame_latents` 优先；没有外部 ref 时才允许按配置从 target latent 采样；
- `ref_num_frames_online=0` 关闭在线采样；大于 `0` 时，参考帧数按
  `uniform[0, min(T_lat/20, ref_num_frames_online)]` 规则采样；
- batch 或 rank 没有 ref 时不构造伪 token；参与同一 FSDP reduction 的 rank 必须对
  `patch_embedding_ref` 是否执行达成一致。

### `wan22_ti2v_5b_causal_teacher`

- 继承 TI2V channel、首帧和 flow-match 语义；默认 `[normal]`；
- 模型输入沿时间轴拼成 doubled `[clean | noisy]`，condition stream 同时拼到两半；
- block-causal mask 由 `frame_per_block` 和 `first_frame_own_block` 定义；
- target 与 loss 只对应 noisy half；该 variant 不注册推理或自回归采样能力。

## Forward 与 backward 契约

模型测试除 state-dict key 和 forward shape 外，还应覆盖以下数值契约。

### RoPE

- 生产验收平台是 CUDA；freqs 必须用 `float64` 计算，`theta=10000`；
- `h_dim=w_dim=2*(attention_head_dim//6)`，剩余 channel 属于时间维；
- 时间、H、W freqs 按 frame-major token 顺序组装；
- RoPE 应用公式、interleave 方式以及写回 Q/K dtype 必须相同；
- Teacher-Forcing 只生成 clean half 的 RoPE，然后使用
  `cat([cos, cos])`、`cat([sin, sin])`，使 clean/noisy 同帧共享位置；
- CUDA golden 必须比较 freqs、旋转后的 Q/K，以及 FP32/BF16 输出；不能只断言 shape。

### Teacher-Forcing backward

TF mask 在 forward 和 activation-checkpoint backward 重算中必须完全相同。测试至少覆盖：

- gradient checkpointing 开/关时，固定输入的 output、loss 和所有参与训练参数 gradient；
- reentrant 与 non-reentrant checkpoint 路径；
- backward 重算期间仍安装同一个 block-causal sparsity；
- 连续执行不同序列长度或 block layout 后，不得从 Attention processor 读取旧 mask；
- BF16 下 QK norm 使用 `rms_norm_across_heads`、`eps=1e-6`；
- Ulysses 开启时 backward loss scaling 仍为 `loss / usp_size`。

当前实现把 sparsity 作为 block forward 参数，由本次 forward 新建的 checkpoint closure
显式捕获；Teacher-Forcing processor 不再保存可变 `_sparsity`。因此 backward 重算使用
发起该计算图时的 block layout，不依赖 processor 最近一次被哪个序列调用。该约束由连续
执行不同序列长度后再 backward 的 reentrant/non-reentrant gate 固定。

## 统一编码 batch

这里的“统一”是指，无论数据来自新版 cache、旧版 cache 还是在线编码，Native task 都先
把它转换成同一种 per-stream batch 格式，再调用模型。共同字段是：

- `sample_id`：稳定字符串 ID；collate 后为字符串列表；
- `prompt`：`[B, L, text_dim]`；
- `frame_latents`：`[B, z_dim, T, H, W]`；
- 每个已配置 stream 对应独立的 `condition_latents_<stream>`，shape 为
  `[B, mask_channels + z_dim, T, H, W]`；TI2V 的 `mask_channels=0`；
- `image_condition_latent_vae`、`image_condition_latent_clip`、
  `image_condition_latent_dino` 仅在 variant 和配置需要时出现；
- TI2V 额外要求 `first_frame_latents=[B, 48, 1, H, W]`；
- ref variant 允许 `ref_frame_latents=[B, 48, K, H, W]`；
- partial-normal 启用时要求 `normal_fg_body` 和 `normal_bg_body`；
- audio variant 要求 `audio_features=[B, L, D, F]`；可选 listen 输入使用同形状的
  `audio_features_l`。

stream 的 channel concat 顺序严格等于 `condition_streams`。模型不读取 cache 字段名，task
先按该顺序拼出 `hidden_states=[noisy_latent | stream 0 | stream 1 | ...]`。

训练 dropout 只能发生在转换为统一格式之后。drop 某个 stream 时使用该 stream 的
确定性 fallback block：RGB/grid 对应像素值 `0.0`，normal 对应 `0.5`；带 mask 的
grid/normal fallback mask 为 `1`。缺少生成 fallback 所需的缓存或冻结 VAE 时，在训练
开始前报错，不能静默换成零 tensor。

固定 seed 的 dropout 使用以下随机数消费顺序：RGB 首帧/stream、RGB image
side input、normal、grid、text、逐样本 audio，最后才是 partial-normal。概率为 `0` 或
对应输入不存在时不得额外消费随机数。RGB stream 命中 dropout 时同时清除
`first_frame_latents`，避免被首帧条件旁路重新注入。这个顺序不改变各项边缘概率，但会
决定同一个 seed 具体丢弃哪些条件，因此属于训练复现契约。

Audio dropout 以样本为单位独立抽样；命中后清零该样本完整的 Wav2Vec feature stack，
不改变 tensor shape。在线编码时 `audio_pretrained_path` 指向冻结的 Wav2Vec 资产：stage A
拼接所有 hidden state 并从 50 fps 采样到 `audio_video_rate`，stage B 生成
`[L,D,F]` bucket；离线训练则直接读取同一格式的 v3 cache，不加载 Wav2Vec。

### Gameplay camera 到 grid

配置包含 `grid` condition stream 时，在线 raw manifest 可用两种方式提供 grid 输入：

- `grid`：与 RGB 视频逐帧对齐的 grid 视频路径；
- `c2ws` 和 `Ks`：直接写在 manifest record 中的相机矩阵。`c2ws` 是
  `[T,4,4]` camera-to-world pose，`Ks` 是 `[T,3,3]` 像素坐标内参。

同一个 batch 中已有 `grid` 视频时优先读取视频，不再调用相机渲染；没有 grid 视频时，
每条样本必须同时提供 `c2ws/Ks`。相机序列长于 RGB 时从尾部截断，短于 RGB 时重复最后
一帧，最终与 collate 后的 RGB 帧数严格一致。`Ks` 必须已经对应解码后 RGB 的高宽；
resize/crop 引起的内参变换归数据处理所有，模型侧不猜测原始分辨率。

`camera_to_grid` 先把所有 pose 转成相对第一帧的坐标，再按 2 米网格、6 米
层高、红/蓝地面线和绿色立柱参数现场渲染 `[T,3,H,W]` FP32 pixels，随后走配置中
`grid` stream 的同一个冻结 VAE，cache 仍只保存 `condition_latents_grid`。可选
`grid_random_rot_deg` 在相对 pose 上施加全局随机旋转；默认关闭，Gameplay ref preset
不再默认开启。该增强操作的是相机或像素，只能在 cache 准备或 raw-online 编码阶段生效；
离线 latent cache 已经完成编码，在训练 YAML 中设置该字段不会改变任何数据，因此两份
生产离线模板都不配置它。

## Cache 读取

只支持数据链当前生成的 v3 per-stream cache：

- 每个已配置 stream 都必须存在显式 `condition_latents_<stream>` 字段；通常值为 tensor。
  生产 source 明确不拥有某个 stream 时允许值为 `None`，但训练必须配置冻结 VAE 或在
  cache 中提供对应的 `condition_fallback_latents_<stream>`，Wan Taichu task 会在严格
  shape 校验前补成中性 latent；
- 生产 cache 可沿用 `train/<bucket>/<sample>.pt` indexed 目录而没有 root manifest；如果
  cache 提供 `manifest.json`，其中 codec identity 必须是 vforge Native manifest，不接受
  缺少 codec identity 的 legacy manifest；
- indexed 目录允许原始写入编号存在空洞。通用 Native cache reader 仍只认实际存在的
  文件；Wan Taichu 训练则在 task 内恢复原始 metadata 的稠密 `0..N-1` index
  空间，使 Random Reshuffling 的 permutation 长度和顺序不因少量 cache miss 改变；
- 稠密位置对应的主 bundle 缺失或损坏时，最多在同一 source 内随机重试 `32` 次。resume fingerprint 同时固定稠密 sample count/digest
  与实际写入文件的 count/digest；若 metadata 尾部位置没有任何写入文件，source 可通过
  `extras.indexed_sample_count` 明确提供原始总数；
- ref variant 在读取主 bundle 后检查
  `ref_images/train/<bucket>/<index>.pt` sidecar：主 bundle 已有 `ref_frame_latents` 时优先
  使用；否则把 sidecar 的 `ref_latent` 映射为 `ref_frame_latents`；sidecar 缺失或损坏时
  不终止训练，继续走 target latent 在线 Ref 采样。主 bundle 发生 miss fallback 时，
  sidecar 跟随实际读到的替代 index，而 sampler 身份仍保留原始稠密 index；
- v2 assembled `condition_latents_normal` 不再按 channel 拆包；
- v1 的 `condition_latents_normal` 不再映射到其他配置 stream；
- per-sample tensor 可以缺 batch 维，collate 时只允许提升一次；
- channel、batch 或 latent geometry 与 variant 不一致时立即拒绝。

v1/v2 cache 若仍需使用，应由数据链离线转换为 v3，不在模型 task 内保留 reader adapter。

### Source 长度与动态截断

`data.sources[*].min_latents/max_latents` 是 source 的训练期长度范围，不是 v3 cache
字段。`MixedDataset` 把对应 source 的 bounds 附在每条样本上；Wan Taichu 自有 collate
要求同一 batch 内 bounds 的存在性和值完全一致，再把它们作为运行时元数据传到训练
step。严格 cache codec 只校验 latent 字段，不会把 bounds 写入或要求它们出现在 cache
manifest。

训练顺序固定为：

```text
encode/cache canonicalize
  -> condition dropout
  -> temporal truncate
  -> condition processor / noise / target
```

截断长度从 source 闭区间均匀采样，并被当前 `frame_latents` 的实际长度限制。采样使用
独立的持久 generator，以 `runtime.seed` 初始化一次，随后持续推进；
它不读取 global step，也不会消耗 noise 或 Ref sampler 的 RNG。
所有与 `frame_latents` 时间轴对齐的 latent stream 同步截断，单帧
`first_frame_latents` 和已经独立选出的 `ref_frame_latents` 保持不变。Ref 没有外部 cache
输入时，在线 Ref sampler 看到的是截断后的 target 帧库；Teacher-Forcing 也在截断之后
再组装 doubled `2×T` 序列。

可选 `model.extras.max_latents_schedule` 使用四个整数字段：

```yaml
max_latents_schedule:
  start_step: 0
  end_step: 4000
  start_value: 38
  end_value: 150
```

它按整数线性插值进一步限制 source 上限；当 schedule 上限低于 source 下限时，
有效下限收敛到有效上限。该字段属于 Wan Taichu 的训练语义，不扩展为通用 data schema。

## 训练输入与 loss

普通 variant 使用以下固定数据流：

```text
统一编码 batch
  -> condition dropout / optional partial-normal
  -> sample scheduler index and noise with persistent RNG
  -> noisy = (1 - sigma) * clean + sigma * noise
  -> first-frame inpaint / per-token timestep when applicable
  -> model forward
  -> mean(loss_weight * (output - (noise - clean)) ** 2)
```

所有 Wan Taichu variant 都使用 pre-shift sigma 表：先生成
`linspace(1, 0, num_train_timesteps + 1)[:-1]`，再调用 Diffusers
`FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)` 应用 Wan shift。不能只使用
scheduler 构造函数的默认表；默认表采用 `linspace(1, 1/N, N)`，两者虽然只差一个端点
定义，但固定 seed 下会在部分 index 产生不同的 timestep/sigma，并使后续 optimizer 状态
逐步分叉。

A14B 先根据可训练 expert 和 global-step 奇偶选择 high/low 区间，再从该区间采 timestep；
不能先全区间采样再按第一个元素路由。Ulysses 的梯度缩放、autocast、backward、optimizer、
日志和 checkpoint 由 vforge Trainer/engine 负责。

Causal teacher 生成相同 noise 和 target，但模型输入是 doubled sequence，per-token
timestep 和 attention mask 由 causal variant 构造。Audio 和 ref 只增加 model input，
不得改变未启用时的普通训练结果。

## 权重与 checkpoint

Diffusers 初始化：

- 单 expert 从 `transformer/` 加载；无前缀权重 key 通过
  `model_prefix="transformer."` 映射到现有 Native wrapper 的 `transformer` 子树；
- A14B 从 `transformer/`、`transformer_2/` 加载到 `wan`、`wan_2`；
- widened patch embedding 复制已有输入 channel，新增 tail 为零；
- checkpoint 没有 ref projection 时，必须等主 patch embedding 加载完成，再从其 noisy
  channel warm-start；checkpoint 已包含 ref projection 时不得覆盖；
- checkpoint 没有 Audio 参数时，先按普通模块构造规则初始化
  `casual_audio_encoder.*` 与 `audio_injector.*`，再加载已有参数；checkpoint 已包含这些
  参数时由加载结果覆盖初始值；
- `audio_zero_init=true` 时只在权重加载之后清零 injector output projection，false 时不得
  覆盖 checkpoint 中已有 Audio 权重。

单 expert 的 portable PT 权重通过 `StateDictMapper.strip_prefix("transformer.")` 保持无前缀
导入导出；full DCP 使用 Native wrapper 的 `transformer.*` 内部 key。
A14B 使用 `wan/wan_2` 布局。首次训练保持 `resume.path=none`，完整 DCP 续训使用
`resume.path=auto`；weights-only 初始化不恢复 optimizer、scheduler、global step 或 RNG。

## 验证

训练输入统一使用持续随机流；CT/TF 测试 worker 使用 `runtime.deterministic=true`
并显式关闭 TF32。
loss gate 比较 loss、学习率和样本顺序，不再依赖训练日志中的 tensor hash。
2026-09-16 已重新采集并独立复跑 B300 的 CT 5B（DP2×USP4）、CT A14B 和 TF A14B
（DP1×USP8）基线，各 3 步；原有 8 项 Wan GPU gate 全部通过。环境为
PyTorch 2.13.0+cu130、CUDA 13.0、FA4 4.0.0b29，seed=42、BF16、2 层模型。
数值保存在
[loss JSON](https://git.woa.com/ieg-infra/vforge/blob/master/tests/models/wan_taichu/ct_tf_b300.json)，
保留原 `rtol=1e-5`、`atol=1e-6` 比较。该回归不覆盖完整层数长跑和 GPU 完整续训。

模型与任务的 CPU contract 位于 `tests/models/wan_taichu/` 和
`tests/tasks/wan_taichu/`，覆盖模型输入、条件、权重映射、训练与评估。
GPU 和分布式训练的验证范围见[支持矩阵](index.md)与[本地开发与验证](../development/testing.md)。
