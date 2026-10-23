# RepWorld：World-DiT + causal ViT

RepWorld 将 representation-space World-DiT 主干接入 Native train/eval。训练使用 frozen
VideoRAE 或 V-RAE 编码 RGB，联合优化 World-DiT 与 causal ViT；没有复制研究仓库的
Trainer、attention backend、checkpoint manager、实验启动器或诊断脚本。

## 范围与状态

当前为**内部验证**，不是完整生产验收。支持在线视频训练、原生 checkpoint 恢复、
严格因果视频 rollout、Ulysses sequence parallelism 和旧两塔权重初始化。
已接入已完成的 VideoRAE3D packed-FP32 shard cache（latent conditioner / pure-TF）；
暂不支持 Native prepare、cache eval、其它 codec 的离线 cache、TP/CP、
NextLat、joint pixel renderer、RCG、训练 EMA 或旧 optimizer resume。

| experiment | Frozen codec | RGB 帧/state | latent/state | 默认 states |
| --- | --- | ---: | --- | ---: |
| `rep_world` | VideoRAE 3D | 16 | 1024 × 64，4 × 16 × 16 | 10 |
| `rep_world_1d` | VideoRAE 1D | 16 | 512 × 32，learned slots | 10 |
| `rep_world_eupe` | V-RAE EUPE | 4 | 256 × 768，1 × 16 × 16 | 20 |
| `rep_world_vjepa` | V-RAE V-JEPA2.1 | 4 | 256 × 1024，1 × 16 × 16 | 20 |

1D 的两层 16×16 网格只是兼容 World-DiT 的索引布局，没有物理时间轴。其 16/2
不能当作实际的 temporal compression。其它 codec 的原生压缩由发布权重决定，不能
任意改 stride 后继续声称使用同一 codec。

四种 frozen codec 在 H200 BF16 上与原实现的 encode/decode/逐段 decode 已逐位对齐。
10 组真实旧训练权重的完整 50-step Euler rollout 也已逐位对齐，latent/RGB 最大误差均为 0：
VideoRAE 3D 与 EUPE 各覆盖 RGB patch2/patch4/latent，另含 VideoRAE 3D/1D z-prev、
VideoRAE 16×960 pure-TF 和 V-JEPA2.1 RGB。每组使用固定输入、seed=1234、80 或 160 帧；
双方固定 SDPA，语义模型 CFG=2，pure-TF CFG=1。这不代表不同 attention backend 逐位等价。
四种 codec × RGB patch2 / RGB patch4 / latent 共 12 组完整 12×768 网络已通过
两步 Native 训练、full DCP 保存和短程评估；另 8 组覆盖四种 codec 的 z-prev 与
pure-TF（完整网络，3 states）。测试 seed=42、单卡 H200、FP32 参数/BF16 autocast。
完整 12×768 ViT + 12×768 World-DiT 的 EUPE/RGB 单卡训练、DCP 保存/恢复和短程评估已跑通。
完整网络的单机 H200 DDP2、FSDP2、HSDP4、USP2/4/8、DP2×USP2 和
HSDP4×USP2 已通过短程训练；四种 codec 的 USP2、VideoRAE z-prev/pure-TF USP2、
重算与梯度累积也已覆盖。不同并行度的 loss/梯度在 FP32/BF16 容差内对齐，
不承诺 BF16 跨拓扑逐位相同。验证范围与限制见下文。
不据此承诺多节点吞吐、长程质量或全部组合已验收。
另已完成两节点 B300 的 KAP 验证，具体覆盖范围见下文。

CPU 测试覆盖模型/梯度、因果性、token 排列、重算一致性、Muon 更新、DTensor 全矩阵
正交化，以及 Native full DCP 的逐位续训一致性。正式 rollout 的数值不会因迁移而增加
额外归一化；原模型可能存在的长程漂移并未在此修复。


## B300 的 FA4 attention

RepWorld 保留 `attention.backend=sdpa` 默认值。B300 的 FA4 路径已接入，但独立
Muon 参数更新检查未通过，不能声称与 SDPA 具有相同优化轨迹。需要显式启用
`attention.backend=flash_attention_4 runtime.deterministic=true`；
`attention.allow_fallback=false` 保证主模型不会静默回退。
FA4 需要 B300/SM100 和匹配的 `flash_attn.cute` 环境；H200/CPU 保留 SDPA。
这是 attention kernel 的选择，与 DDP/FSDP 的选择独立。

BF16 算子对 FP64 oracle 的前向及 Q/K/V 梯度检查、精确因果性检查已通过。
小模型的独立 Muon 更新差异超过预设 5% 门槛，不能以较小的 loss 或梯度差异替代
此项验收。FA4 确定性模式的同后端三步重复检查逐位一致；该开关不消除 FA4 与 SDPA
之间的浮点舍入差异。现有训练不会因新增路径自动迁移。
模型只负责描述可见关系，计算复用公共 `vforge.ops.attention`：

- ViT 使用 state 级分段因果可见性；同一 state 内双向可见，未来 state 不可见。
- 80% latent token masking 保留原空间采样和 RoPE，分段长度使用每个 state 的实际保留数。
- 无前缀 DiT self-attention 和 cross-attention 使用 dense FA4。
- z-prev 的两块 mask、pure-TF 的 clean/query mask 和推理 KV 的矩形 mask
  使用公共 `SegmentedSparsity`，不改成 token 级三角 causal mask。
- frozen codec 沿用发布实现的 SDPA，不通过这一开关替换编码器/解码器内部算子。

SDPA 路径保留原有 dense mask，用作独立参考。跨 FA4/SDPA 不承诺逐位相等；
验证区分 mask 的精确等价、BF16 前向/梯度误差与独立 Muon 更新差异。
数值 gate 不把一侧梯度复制到另一侧，不以权重总范数掩盖更新量差异。
GPU gate 保留原阈值并会如实报告未通过的更新检查；`--diagnostic` 收集完整误差，
仍以非零状态报告失败。

CPU 可见性与完整梯度回归：

```bash
pytest -q tests/models/rep_world/test_fa4_visibility.py
```

B300 算子、因果性和模型数值 gate：

```bash
pytest --gpu -q tests/models/rep_world/test_fa4_gpu.py
```

完整 12×768 模型的真实 checkpoint gate 使用
`tests/models/rep_world/fa4_worker.py --stage full --checkpoint <legacy.pt>
--input <fixed_input.pt> --codec-checkpoint <videorae.pth>
--statistics <stats.pt> --output <result.json>`。此阶段需要测试环境中明确配置的
VideoRAE 发布资产；训练实现本身不依赖测试资产路径。
既有并行/续训 gate 显式固定 SDPA，不能用历史 SDPA 验证结果代替 FA4 的验证结果。

## 数据与权重

准备 JSONL 或 JSON 数组，每条记录包含稳定 ID 和视频路径：

```json
{"sample_id":"clip-0","rgb_path":"/data/clip-0.mp4"}
```

也接受 `video`/`video_path`/`rgb`；相对路径以 manifest 所在目录为基准。`clip_id` 可用作
ID。视频按原实现直接 resize 到 256×256，不 crop/letterbox，不重新采样 FPS。
训练选连续随机窗口，再按 `frame_interval` 取帧；窗口长度为 states×原生 RGB 帧/state。
缺帧或解码失败会带 sample ID 报错，不会静默替换成另一条视频。Native 保存数据游标和
worker RNG；不承诺旧训练器与 Native 的逐样本 RNG 顺序相同。

`codec.checkpoint` 指向已下载且受信任的完整发布权重；生产代码不需要上游 checkout。
VideoRAE 使用与训练匹配的 `codec.statistics`（每通道 mean/std）；V-RAE 默认沿用发布
encoder 的输出归一化，不额外添加 stats。许可证与来源见模型包的 `codecs/THIRD_PARTY.md`；
EUPE 含非商业研究限制，不能把本地适配当作对外发布许可。

## 使用已提取的 VideoRAE3D latent 训练

`latent_cache.mode=offline` 读取 `rep_world.videorae3d.packed_fp32.shards.v1`。
缓存对应 frozen codec 输出：每条 FP32 `[256,S,16,16]`，拼 batch 后为
`[B,256,S,16,16]`，channel 顺序是 `plane*64+channel`。mean/std 已在编码时应用；
训练读取后不重复归一化，不加载 VideoRAE，直接进入原有 FM/model step。latent conditioner 不解码 RGB。
causal ViT、mask、condition dropout、timestep 与 noise 仍在每次训练时正常计算。
RGB conditioner 同时读取原视频作为 ViT 输入、缓存作为 FM 目标，仍不加载在线 codec。原视频必须与已审计缓存的帧数、路径、sample ID 和固定窗口完全一致；窗口变化直接失败。每次样本读取只消耗一次 clip RNG，与在线 RGB 路径一致。pure-TF 仅需 latent。

使用现有 `data.sources` 字段给出一个 cache，同时保留原始有序 `data.train_manifest`：

```bash
torchrun --standalone --nproc-per-node=2 -m vforge.tasks.train \
  experiment=rep_world \
  latent_cache.mode=offline \
  'data.sources=[{name:train_latents,cache_dir:/data/videorae3d_cache}]' \
  data.train_manifest=/data/train.jsonl \
  model.extras.conditioner.input=latent \
  model.extras.codec.checkpoint=/models/vjepa21_3d.pth \
  model.extras.codec.statistics=/models/solarwm_latent_stats_32clips.pt \
  runtime.precision=bf16 \
  data.batch_size=8 \
  runtime.max_steps=10000 \
  runtime.output_dir=/runs/repworld_cached
```

`data.sources` 中只需 `name/cache_dir`；不传 raw dataset `extras`，不使用 `+` override。
这里仍声明编码时的权重与统计文件，启动时只读其 SHA256 验证身份，不装载 codec 模型。
同一缓存可供不同大小、mask 比例和 condition 注入方式的 latent 模型复用。

读取要求 cache 根目录存在 `contract.json`、`index.jsonl`、`COMPLETE.json`，以及
全部已提交的 `shards/NNNNNN.npy/.json`。会验证完整性标记、contract fingerprint、
源 manifest 的完整字节摘要、样本顺序、codec/统计文件摘要、BF16 编码精度、state 数、
frame interval、shape、dtype 和 packing。每次读取还校验该行 latent SHA256；
文件缺失或损坏直接报错，不跳样本、不静默回退在线编码。仅 latent conditioner/pure-TF 无需保留原视频；RGB conditioner 必须保留配对源视频。

本版固定窗口必须覆盖该视频的全部请求帧，例如 160-frame 视频的 10 个 16-frame states。
不把长视频的随机窗口替换为固定窗口。对于长度恰好等于请求窗口的原在线 dataset，
`random_clip=true` 仍会消耗一次 `torch.rand`；cached dataset 保留这次消耗。
shuffle、drop_last、worker RNG、DP/USP batch 分发和游标继续由 Native loader 管理。

缓存训练从首次启动起使用上述配置后，可以正常 `resume.path=auto` 完整续训。
online 与 offline 的数据/semantic fingerprints 不同；本接入不绕过 Native 严格校验，
不自动把正在运行的在线训练切换到缓存，也不宣称直接跨模式恢复旧 DCP 游标。

定向验证：

```bash
pytest -q tests/tasks/rep_world/test_rep_world_latent_cache.py
```

CPU contract 覆盖缓存完整性/损坏、原始样本顺序和 RNG、0/2-worker loader 的下一 batch
恢复、在线/缓存 FM loss 与梯度逐位一致、普通 Hydra override。真实 H200 的当前实验
验证范围与结果记录在交接中；不由 reader 存在推断 Native prepare 或 cache eval 已接入。

## 从头训练与续训

以下使用 EUPE；其它 codec 替换 experiment 和资产路径，VideoRAE 还需提供 statistics。
沿用 Native 的完整保存/恢复规则，不在 experiment 中复制 checkpoint 生命周期。

```bash
torchrun --standalone --nproc-per-node=2 -m vforge.tasks.train \
  experiment=rep_world_eupe \
  model.extras.codec.checkpoint=/models/vrae_eupe.pt \
  data.train_manifest=/data/train.jsonl \
  data.batch_size=4 \
  runtime.max_steps=10000 \
  runtime.output_dir=/runs/repworld
```

相同配置增加 `resume.path=auto` 即为 Native full DCP 续训。周期保存使用
`checkpoint.every_n_steps=2500`。Muon 默认开启，`optimizer.learning_rate/betas/eps`
同时控制矩阵分支学习率和辅助 Adam 参数；`model.extras.use_muon=false` 使用普通 AdamW。
Muon 的 Newton–Schulz 在完整矩阵上计算，不对 FSDP 的局部 shard 独立正交化；这种
正确性优先的实现会为每个矩阵 gather 更新量，不保证比 DDP 更快。

默认 FP32 参数 + BF16 autocast，保留源训练精度路径。RepWorld 显式关闭
FSDP 参数/input 的 mixed-precision cast（`parallel.mixed_precision.enabled=false`），
避免把 FM 随机采样、归一化等也改成 BF16。旧 DCP 若使用该字段为 true，严格续训时
仍须沿用旧 resolved config；不要以新默认值覆盖旧训练的精度 contract。
`parallel.gradient_checkpointing=true`
对 World/ViT blocks 使用 Native 注入的非 reentrant 重算 context；默认关闭。

### 数据并行与序列并行

`parallel.dp_mode=ddp` 使用复制的数据并行，要求 `usp_size=1`。
`parallel.dp_mode=fsdp2` 可独立使用，也可组合 `parallel.usp_size=2/4/8`。
本库的 `usp_size` 指 **Ulysses all-to-all** 路径；论文中的 Unified Sequence
Parallelism 还可组合 Ring Attention，不应把本次接入理解成 Ring/CP 已支持。

例如在 4 张卡上使用 DP2×USP2：

```bash
torchrun --standalone --nproc-per-node=4 -m vforge.tasks.train \
  experiment=rep_world_eupe \
  model.extras.codec.checkpoint=/models/vrae_eupe.pt \
  data.train_manifest=/data/train.jsonl \
  data.batch_size=1 \
  parallel.dp_mode=fsdp2 \
  parallel.usp_size=2 \
  parallel.gradient_checkpointing=true \
  runtime.max_steps=10000 \
  runtime.output_dir=/runs/repworld_usp
```

8 卡 HSDP2×2×USP2 在上述命令中使用 `--nproc-per-node=8`，并增加
`parallel.dp_replicate_size=2 parallel.dp_shard_size=2`。
此时 DP size=4；FSDP 内层同时沿 DP shard 与 USP 分片，外层做数据复制。
同一 USP group 共享视频、FM noise/timestep 和 condition dropout；
不同 DP group 读取不同 clips。每步独立 clip 数为
`dp_size × data.batch_size × runtime.gradient_accumulation_steps`，
不能再乘 USP size；一个 clip 的 S−1 个预测位置也不是独立 clips。

ViT 与 World block 的隐藏 token 按序列切分，RoPE 保留全局坐标，causal mask 保留
全局语义；self-attention 复用 Native Ulysses，cross-attention 使用本地 Q 与完整
condition K/V。模型输出恢复原有全局 token 排列。token 长度须能被 USP size 整除；
heads 不整除时由 Native padding 处理，12 heads/USP8 已验证。
纯 TF 推理沿用全局 KV cache，并复制到各 rank；此路径没有按 USP 倍数减少 KV 显存。
frozen codec、完整输入与部分边界特征仍复制，不承诺整条流水线显存均按卡数线性缩减。

单机 8×H200、PyTorch 2.9.1/CUDA 13.0、seed=42 的验证包括：

- 完整 12×768 ViT + 12×768 World：DDP2、FSDP2、HSDP4、USP2/4/8、
  DP2×USP2、HSDP4×USP2；每项 3 optimizer steps。
- 四种 codec 的 USP2；VideoRAE latent 的重算与两次梯度累积；
  VideoRAE z-prev/pure-TF 的 USP2（3 states）。
- 5 种拓扑各 20 组数值用例，覆盖 RGB patch2/4、latent、z-prev、pure-TF，
  FP32/BF16、重算开/关及两次 Muon 更新。独立比较完整 loss/梯度后，使用同一份
  已验证梯度比较 Muon 分片更新，后者逐位一致；不代表独立 BF16 训练轨迹逐位一致。
- 完整 VideoRAE latent、FSDP2×USP2、重算和两次梯度累积下，
  `runtime.deterministic=true` 的连续 4 步与第 3 步 DCP 恢复至第 4 步逐位一致，
  覆盖全部模型/optimizer tensors、loss、数据游标与 RNG。普通非确定性执行可有微小差异；
  逐位复现实验必须从首次训练起使用同一确定性配置与硬件拓扑。
- Native DCP 短程 eval：单卡、DP2、USP2 各 4 cases；CFG2、5 Euler steps、3 states。
  DP2 的解码 MP4 与单卡逐位相同；USP2 存在数值差异，不作逐位 rollout 保证。

TP、EP、pipeline 与 Megatron engine 未接入 RepWorld；DDP+USP 被配置检查拒绝。
CP 保持现状，未在本次接入中修改。多机结果见下一节；短程正确性测试不代表吞吐验收。

### 两节点验证与远端入口

在两台各 8 卡 B300、PyTorch 2.9.1+cu130 上，已通过 Native KAP 多机验证：

- DDP16、FSDP16 和跨机 USP16 的完整 RGB 模型短程训练。
- DP8×USP2、HSDP(2 replicate×4 shard)×USP2 的完整模型训练；
  四种 codec 的 RGB 路径均覆盖 DP8×USP2。
- USP16、DP8×USP2、HSDP×USP2 各 20 组数值检查，覆盖前述五种模型路径、
  FP32/BF16、重算、完整梯度、Muon 更新、随机量同步与 KV cache。
- 完整 VideoRAE latent、HSDP×USP2、重算＋累积2，在确定性配置下，
  连续 4 步与第 3 步恢复到第 4 步的全部权重、optimizer、16 rank 的 RNG/数据游标
  及 loss 逐位一致。
- 从多机 HSDP DCP 在 DP8×USP2 下完成 8 cases 的短程 Native eval，
  CFG2、5 Euler steps、3 states，所有视频完整解码。

USP16 的一个 group 实际跨两个物理节点；USP2 group 留在机内，数据并行通信跨机。
HSDP 的每台机器包含 4 个 DP shards×USP2，两个节点之间同步 replica。
测试使用 FP32 参数＋BF16 autocast；数值 oracle 关闭 TF32。未放宽单机数值阈值。

可复用的多机 gate 为 `tests/tasks/rep_world/multinode_worker.py`，
模板为 `scripts/ci/kap/rep_world.task.yaml`。它复用 Native CLI 和上述数值 worker，
要求 revision bundle、共享 codec 资产与 manifest、独立输出目录，以及两个节点上的一致参数。
按 [KAP 流程](../development/kap.md) 使用仓库外 machine profile/overlay，
通过模板的 `REP_WORLD_*` 环境变量提供这些路径和 rendezvous 地址；
先运行 worker 的 `--help` 查看参数。必须检查平台最终 Succeeded、worker 的成功标记、
每阶段结果文件及任务清理，不能只看调度成功。

以上为两节点指定组合的短程正确性验证。更多节点、其它硬件组合、跨拓扑逐位恢复、
长程收敛和隔离吞吐仍需单独验证；完整尺寸 z-prev/pure-TF 的多机训练尚未覆盖。

### Conditioner 与 FM 模式

所有覆盖项已在 experiment 声明，不需要 `+` 前缀：

```text
model.extras.conditioner.input=rgb
model.extras.conditioner.temporal_patch=4

model.extras.conditioner.input=latent
model.extras.conditioner.temporal_patch=2

model.extras.flow.mode=no_zprev
model.extras.flow.mode=zprev
model.extras.flow.mode=pure_tf
model.extras.flow.mode=pure_df
```

RGB temporal patch 允许 2/4。latent 输入不会二次 patchify，temporal_patch 在该模式必须
保持 2（不参与其运算）。RGB patch4 与 latent 输入的 token 数在 3D VideoRAE、EUPE 上
对齐，但两者闭环反馈不同，不能据此认定端到端速度/数值等价。

`no_zprev` 支持两种 ViT 输入。`zprev` 额外给 World-DiT clean previous state，保持 RGB
conditioner；`pure_tf` 不构建 ViT，用 World-DiT 的全历史 teacher forcing/KV rollout。
`pure_df` 也不构建 ViT，固定首个 state 为干净上下文，其后逐 state 采样非递减的噪声时间，并用 state-causal attention 预测后续 state。两种无 ViT 模式可共用参数拓扑与逐 state KV rollout；评估都必须设置 guidance_scale=1。latent + zprev 不作为既有消融路径混入。

RGB 训练输入 `[B,3,S*F,256,256]`，packed latent `[B,P*C,S,16,16]`。语义模式将
`S-1` 个 target 展平成 `[B*(S-1),C,P,16,16]`；condition 为 `[B*(S-1),N,768]`，FM
仍回归 clean−noise。RGB condition 来自 GT RGB prefix；latent condition 来自 detached
GT latent prefix，clean 当前 target 不进入自己的 condition。S 个位置不等于 S 个独立 clips。

## 评估与旧权重初始化

```bash
python -m vforge.tasks.eval \
  experiment=rep_world_eupe \
  model.extras.codec.checkpoint=/models/vrae_eupe.pt \
  evaluation.checkpoint_path=/runs/repworld/checkpoints/10000.dcp \
  evaluation.data_source=inference \
  evaluation.manifest=/data/eval.jsonl \
  evaluation.num_inference_steps=50 \
  evaluation.guidance_scale=2 \
  runtime.output_dir=/runs/repworld_eval
```

`inference` 只读取 observed prefix；`validation` 读取完整 clip，但生成仍只看 seed。
输出为逐 sample MP4 和 Native provenance JSON；随机噪声使用独立 generator，seed 按
Native sample ID 规则稳定派生，不依赖前一个 case 的采样。

批量 rollout 使用 `model.extras.rollout.batch_size=4`（默认 1），表示每个 DP rank
同时生成的独立 case 数；`data.batch_size` 不控制这个推理窗口。每个 case 保留按 sample ID
派生的独立随机流，codec encode 仍逐 case 执行。Native 仍逐 case 校验 provenance、
跳过已完成结果并写出视频；不足一个窗口时保留尾批和分布式 padding。
修改 batch 会进入解析后的配置，因此已有结果是否复用仍由 Native provenance 校验决定。
该选项处于实验验证阶段；不同 batch 下浮点 kernel 的逐位一致性需要在实际模型、
设备和 attention backend 上确认，不能由固定 seed 单独保证。

旧 RepMiniWorld checkpoint 仅用于**weights-only**初始化：把 evaluation.checkpoint_path
留空，设置 `model.pretrained_path=/models/old_run.pt` 和
`model.extras.initial_weights_format=legacy`，并提供与旧模型一致的 codec、conditioner
输入域、feature layers、输入分辨率、temporal patch 和 FM 模式。train 也支持这两个字段。
导入严格检查所有 active tensors，只去掉原 ViT 未参与此路径的 image branch；不恢复
旧 optimizer、epoch、step、EMA 或 NextLat predictor。旧 metadata 缺省时与源采样器
保持一致：no-zprev、RGB temporal patch2、最后一层特征。pure-TF 的持久化 RoPE
预计算表按训练窗口校验后转换为 Native 的逐 state 缓存，base frequencies 与全部
learned tensors 保持原值。不要将旧 `.pt` 放到 `resume.path`
来模拟无缝续训。

World 网络尺寸也必须匹配。例如源 `0.22B` pure-TF 使用 width=960、depth=16、
heads=15、mlp_ratio=4.3125；这些都通过 `model.extras.world.*` 显式覆盖，不能只改
`flow.mode` 后加载不同尺寸的旧 checkpoint。

## 验证入口

```bash
OMP_NUM_THREADS=4 pytest -q tests/models/rep_world tests/tasks/rep_world
```

Ulysses 的无外部资产 GPU 数值 gate 可直接运行（需 2 张 CUDA GPU）：

```bash
torchrun --standalone --nproc-per-node=2 \
  tests/models/rep_world/sequence_parallel_worker.py \
  --usp 2 --output /tmp/repworld-usp2.json
```

4 卡可测试 `--usp 2` 或 `--usp 4`；8 卡可测试 `--usp 8` 或
`--usp 2 --replicate 2`。该 gate 固定 SDPA math，FP32 oracle 关闭 TF32，
覆盖 head padding、完整梯度、分片 Muon 更新、重算、随机量同步和 pure-TF KV。
对应 pytest wrapper 位于 `tests/models/rep_world/test_sequence_parallel_gpu.py`。

published parity tests 是额外资产门禁，需要通过测试文件声明的环境变量提供发布权重与
仅用于参考比较的旧 checkout；这些 checkout 不是 Native 运行时依赖。

`tests/tasks/rep_world/test_legacy_rollout_gpu.py` 用持久化 RGB input 与真实训练权重
比较完整 50-step Euler rollout。双方使用同一 SDPA backend；源实现可选的 FlashAttention
快捷路径仅在对照测试中关闭。跨 attention backend 不承诺逐位相同。CFG 系数与 Euler
步长必须保留张量广播后的 dtype promotion，不能替换为零维 scalar 运算。
pure-TF 的有效 rollout prefix 短于训练窗口，RoPE 沿用源实现的 autocast 动态计算，
首个 state 也不能复用 FP32 预计算表，否则首层 KV cache 就会发生数值偏移。


## Latent context token masking

Set `model.extras.conditioner.input=latent` and
`model.extras.conditioner.context_mask_ratio=0.8` to drop context tokens before
the causal ViT blocks during training. The default ratio is zero and retains
the original model path and checkpoint parameter layout.

Sampling follows DuET's independent spatial masks for each sample, state and
latent plane: keep `max(1, floor(H*W*(1-ratio)+1e-8))` positions per plane.
At 80% masking on VideoRAE 3D this is 51 of 256 positions per plane, or 204 of
1024 tokens per state. Visible tokens keep their original global 3D RoPE
coordinates and state-causal boundaries. DiT cross-attention reads only the
visible features; this option does not add DuET's dense prefix completion
module, clean shortcut or spatial injection. Evaluation always uses all tokens.

The unsplit masked training path requires latent input; split E/C also supports
RGB input. Both masked paths require USP=1. Existing unmasked USP behavior is
unchanged. This feature changes the training objective's condition
distribution, so start a separate experiment rather than treating it as an
equivalent continuation of an unmasked run.

对于 `architecture=split_ec`，默认 `mask_stage=after_e`：每个完整 state 先经过
ViT-E，然后在进入 ViT-C 前抽取可见 token。设置
`model.extras.conditioner.mask_stage=before_e` 后，同一抽样会提前到 ViT-E 前；
latent 输入会在 patch 线性投影前 gather，RGB 输入则在 3D patch embedding 后、
ViT-E attention 前 gather。ViT-E 与 ViT-C 都只处理可见 token，DiT 仍接收
原始 state 内的 token 索引。前后两种方式使用相同的抽样比例和模型参数结构，
但训练条件不同，应该从相同初始权重分别训练，不要当作等价续训。

RoPE 不会将可见 token 重新编号。设 state 为 `s`、latent plane 为 `p`、
网格坐标为 `(h,w)`，则 ViT-E 的三个坐标是 `(p,h,w)`，每个 state
重新从 `p=0` 开始；ViT-C 是 `(s*P+p,h,w)`，其中 `P` 是每个 state 的
plane 数。移动 mask 前后，保留 token 的这两组位置编码完全相同；变化只在于
ViT-E 的 self-attention 是否看过最终被丢弃的 token。eval 模式不施加
训练 mask，仍使用完整 token 和原有 KV cache 路径。


## Entry concat conditioning (Native preview)

Set `model.extras.world.condition_mode=entry_concat` with latent input,
`flow.mode=no_zprev`, a spatially ordered codec, and `world.patch_size=1`.
The default is `cross_attention`; its parameter layout is unchanged.

Entry concat joins each noisy latent patch's raw channels with the projected
ViT feature at the same spatial position, then applies a Linear–SiLU–Linear
fusion before the first World block. This mode does not instantiate the
ordinary noisy-patch projection or cross-attention layers. It does not add
per-block spatial injection or a clean latent shortcut.

With context masking, the conditioner also returns state-local visible
indices. Features are scattered back to their original positions after
projection; masked positions contain zeros, including when the projector has
a bias. Evaluation uses all conditioner tokens. Entry concat rejects zprev,
teacher-forcing/cache prefixes, non-spatial codecs and patch sizes other than
one. Legacy cross-attention weights are not interchangeable with this topology.

Hopper FA4 segmented attention is available when the installed CuTe package
provides SM90 forward and backward kernels. The validated H200 path uses
`attention.backend=flash_attention_4`, `attention.allow_fallback=false` and
`runtime.deterministic=false`. Dense/cross/masked-causal forward and backward
and eight-rank FSDP2 training with activation recomputation and full DCP saving
have been exercised. Deterministic segmented SM90, other head dimensions and
other installation versions require separate validation. Frozen codec attention
retains its original implementation.

### Cross-attention order

`model.extras.world.cross_attention_position=after_self` preserves the default
SA → CA → FFN block order. Set `before_self` for CA → SA → FFN in blocks
that have cross-attention. `cross_attention_every` still selects block indices
0, N, 2N, …; setting N equal to DiT depth enables only block 0. Other blocks
remain SA → FFN. Cross-attention excludes clean prefix tokens in either order;
CA before SA changes the input used to compute self-attention keys and values.
Training and evaluation must use the same setting. This option changes no
parameter shapes and does not make old checkpoints equivalent under a new order.

## Joint ViT/DiT EMA

Enable model.extras.ema.enabled=true and set model.extras.ema.decay=0.99.
The task creates FP32 shadows after materialization/sharding and before Native
full-resume loading. All trainable ViT and DiT parameters are included; the
frozen codec is excluded. Shadows start from the initial online weights. Every
successful optimizer step applies ema = decay * ema + (1 - decay) * online,
with no warmup or delayed first copy. EMA buffers and the update count are part
of the same full DCP checkpoint as the online weights and optimizer.

For evaluation, select model.extras.ema.evaluation_weights=online or ema.
Use a distinct evaluation.output_dir for each selection. Both online towers
are loaded strictly with Native component loading; the EMA selection also loads
and validates the EMA subtree, saved decay, parameter order and completed-step
count before copying its weights into the inference model. Missing EMA state is
an error. Training resumes from online weights plus the saved EMA state, never
from the EMA-selected inference model. EMA is disabled by default.

CPU tests cover the exact update formula, strict online/EMA DCP selection and
continuous-versus-resumed joint Muon training. GPU validation is tracked with
the experiment's topology, precision and checkpoint evidence.


## Same-spatial cross-attention ablation

Set `model.extras.world.cross_attention_scope=same_spatial` to restrict each
DiT query to visible condition tokens at the same `(h,w)` across temporal
planes. The default `global` preserves the original path and parameters.
This composes with `cross_attention_every` and `cross_attention_position`;
using `every=depth` and `position=before_self` gives first-block CA → SA → FFN.

This mode requires spatially ordered latent input, `flow.mode=no_zprev`,
`patch_size=1`, and USP=1. It retains the original state-local indices after
conditioner masking. Training groups queries by spatial location and groups
visible keys by count; each group runs through the configured Native attention
backend. There is no SDPA override or dense masked-attention fallback.
Unrestricted temporal-plane candidates at the same location are allowed.
Missing keys do not enter softmax; an empty location returns zero, including
the output projection bias. Classifier-free dropout still replaces visible
condition values with the learned null, retaining the visibility mask.
Dense inference has all locations available; `condition=None` broadcasts the
learned null to all locations.

For VideoRAE 3D, each location has four temporal tokens. With 51/256 visible
tokens independently selected per plane (nominal mask ratio 0.8), roughly
`(205/256)^4 = 41.1%` of locations have no visible keys during training.
This ablates direct cross-patch retrieval by CA; the conditioner ViT and DiT
self-attention still mix spatial information. Q/K/V, normalization, projection,
initialization, and trainable parameter counts are unchanged.

Targeted validation: `tests/models/rep_world/test_same_spatial_cross_attention.py`
covers an independent dense-softmax reference for outputs and gradients,
forbidden-key influence/gradients, sparse-coordinate permutation, empty
locations, BF16 autocast, model loss/backward with checkpointing, dense and null
inference, exact parameter initialization, and unsupported input rejection.


### Cross-attention 的 3D RoPE

`model.extras.world.cross_attention_3d_rope=true` 显式旋转 CA 的 Q/K，默认关闭。
复用模型的三轴 RoPE 实现，不新增可训练参数或持久 buffer，attention 仍由 Native backend 调度。
当前支持 spatially ordered latent conditioner、no_zprev、patch_size=1 和 USP=1；
完整网格及带原始索引的 masked condition 均可用。每对相邻 state 使用统一 latent-plane 坐标：
condition 时间为 [0,P)，目标为 [P,2P)，两边空间网格相同。稀疏 key 使用原始 token 索引，
不能按压缩序列重新编号。全局 CA 建模三轴相对位移；same_spatial CA 的空间位移为零，
因此空间旋转在理想精度的 QK 内积中抵消，时间轴仍区分各 latent plane。
该开关不改变 ViT/DiT self-attention 的 RoPE，也不等同于 context token masking。
CPU/GPU 验证结果以当前测试记录为准；旧实验的 frozen runtime 不随此开关实现更新。


## DuET 随机预测距离（Native preview）

在现有训练命令中添加 `model.extras.flow.distant_target_prob=0.25`，
每个 clip 的每个 context 独立选择预测下一个、第 2 个或第 3 个 codec state；
概率分别为 `1-p`、`p/2`、`p/2`。这对应 DuET 的 `distant_target_prob`，
内部/公开 offset 统一从零开始：K=3 的候选是 0/1/2，offset=0 是原有下一 state 预测。
不是同时预测多帧，也不是预测概率分布。VideoRAE 3D 一个 state 对应 16 个 RGB 帧，
因此三个预测距离为 16、32、48 个采样帧；数据 `frame_interval` 仍决定原视频间隔。

该功能支持 `flow.mode=no_zprev`，RGB / latent conditioner、入口 concat、
global CA 和 same-spatial CA。原有各模式的约束仍适用；zprev / pure-TF
会明确拒绝。`p=0` 是默认值，保留原来的 `S-1` 个目标、RNG 消耗和 loss 路径。
`p>0` 至少需要 4 个 state，并与 DuET 一样固定使用前 `S-3` 个 origin：
origin `i` 只读取 `0..i` 的历史，目标取 `i+1+offset`，跳过的未来状态不会进入
condition。10-state clip 因而从 9 个训练目标变为 7 个。训练日志的
`prediction_positions` 和 `local_tokens` 使用实际目标数；每个 horizon 的比例
及对总 FM loss 的贡献也会记录，三个 loss contribution 相加等于 `loss_fm`。

融合后的 DiT hidden 加上参数无关的
`sin_cos(offset+1) - sin_cos(1)`，频率与 DuET 一致。预测相邻 state 时此项严格为零。
如果启用 CA 3D RoPE，K 的时间坐标保持 `[0,P)`，Q 使用
`[(offset+1)*P,(offset+2)*P)`，其中 P 是每个 state 的 latent plane 数；
same-spatial 的空间可见性不变。没有开启 CA RoPE 的模型仍由显式 gap embedding
区分预测距离。ViT 的因果 RoPE 和 DiT self-attention 路径不变。
RepWorld 原有 `clean-noise` velocity 符号与采样积分方向也保留，
没有照搬 DuET 的相反噪声时间约定。

默认 rollout / FVD 仍逐相邻 state 生成，等价于 `offset=0`，
不随机跳过未来。模型的 `operation=velocity` / `operation=world` 可显式传
`target_offsets`（同 device 的 int64 `[B]`，值 0–2）来查询其他距离。
训练 loss 的显式 `noise` / `uniform` 分别对应展开后的
`[B*(S-3),C,P,H,W]` / `[B*(S-3),1]`；`target_uniform` 为 `[B,S-3]`，
只用于固定目标选择的对照测试。

该开关不新增参数或持久 buffer，旧权重可严格加载；改变开关会改变训练目标和
RNG 路径，不能当作同一优化轨迹的续训。保持配置不变时，随机选择使用 Native
checkpoint 管理的 PyTorch RNG；USP sample-sharing 使用现有随机张量同步。
测试覆盖见 `test_distant_target.py`、`test_distant_target_config.py` 及 Native
DCP resume 测试。GPU / 分布式验证范围以测试记录为准。


## K=4 四象限未来目标（Native preview）

在现有训练命令中设置 `model.extras.flow.target_layout=quadrant_mosaic`。
保持 `model.extras.flow.distant_target_prob=0`；本模式每个目标同时包含
四个 horizon，不再随机选择单一 horizon。默认 `target_layout=single_state`
保留原有行为。该模式是自定义训练目标，不是 DuET 原有的四象限实现。

对每个 clip、每个 origin `i`，独立随机生成 `[0,1,2,3]` 的排列，
依次指定左上、右上、左下、右下的额外预测 offset。K=4 表示四个候选，
offset=0 就是普通下一 state，offset=3 是第四个未来 state。
每个象限从 `state[i+1+offset]`
取相同空间区域，保持原始空间位置及全部通道、latent planes，不缩放、不移动。
四个象限共用一个 diffusion timestep，但使用逐元素高斯噪声；
FM 的 velocity 符号、积分方向和均值归约保持不变。condition 只读取 `0..i`。

这里的距离单位是 codec state，而不是单张 RGB 帧。VideoRAE 3D 的每个
state 有 4 个 latent planes，对应 16 个采样 RGB 帧；horizon 1..4 的
state 起点差为 16、32、48、64 个采样帧，原视频间隔还受 frame_interval 影响。

至少需要 5 个 state；每个 clip 使用前 `S-4` 个 origin。
10-state clip 因此产生 6 个 composite targets。日志的
`prediction_positions` 是 composite target 数；`prediction_quadrants`
是它的四倍。日志的 horizon 仍表示实际未来 state 距离 1..4（即 offset+1）。
四个 `target_horizon_N_fraction` 恒为 0.25；
每个 `loss_fm_horizon_N_contribution` 是该 horizon 对总 loss 的贡献，
四项相加等于 `loss_fm`。

### 每个 token 的时间坐标

设每个 state 有 P 个 latent planes，token 在 state 内的 plane 为 p，
其所在象限的零起始 offset 为 o。SA 与 CA 沿用各自普通预测的坐标原点：

- DiT 每层 self-attention 的 Q、K 都使用 `t=o*P+p`；h/w 保持真实空间位置。
- 开启 CA 3D RoPE 时，以 condition 为原点，CA Q 使用 `t=(o+1)*P+p`；K 保持
  `t=0..P-1`，被 mask 的 key 仍使用原始 token 索引。
- ViT 的历史因果 RoPE 不变，不把未来时间写入历史 condition。
- 原有 gap embedding 改为逐 token 的 `sin_cos(o+1)-sin_cos(1)`，
  在入口 fusion 后、第一个 DiT block 前加入。diffusion timestep/AdaLN 不变。

例如 TL/TR/BL/BR 的 offset 为 `[2,0,3,1]`，P=4 时 SA 分别使用
时间坐标 8..11 / 0..3 / 12..15 / 4..7。CA Q 相应为 12..15 / 4..7 / 16..19 / 8..11，
CA K 保持 0..3；等价于将 CA Q 改用 SA 原点并同时把 K 改为 -4..-1。
不能只把 CA Q 减去一个 state 而保留 K，否则会错误地少算一个 state 的时间差。
Self-attention 仍允许各象限互相查询；没有新增时间因果 mask。
所有 token 同时平移同一时间会在理想精度的 RoPE 相对位置中抵消，
但不同象限的时间差不会抵消，因此不能只修改 CA 的 Q。

该模式要求 `flow.mode=no_zprev`、spatially ordered codec 和偶数 DiT
patch 网格；patch 不得跨越象限边界。RGB/global CA 可以沿用已有路径，
入口 concat 及 position-aware CA 的原有 latent-input 限制保持；
same-spatial/CA 3D RoPE 仍只支持 USP=1。VideoRAE 1D slots 无真实空间
对应，明确拒绝。此模式不新增参数或持久 buffer。

训练测试可传 `quadrant_offsets`，形状为同 device 的 int64
`[B,S-4,4]`，每行必须是 0..3 的排列；不能同时传 target_uniform。
显式 FM noise/uniform 的 batch 维为 `B*(S-4)`。
低层 `operation=velocity` 可传 `target_offsets` 为 int64 `[B,N]`
（0..3，N=P*h*w），以同样位置约定查询 composite target。
默认 rollout/FVD 仍生成普通相邻完整 state，不生成四象限拼接视频。
普通整帧 rollout 的效果需要重新评测，训练代码可运行不代表该分布变化无影响。

随机排列复用 Native checkpoint 管理的 PyTorch RNG。
`test_quadrant_future.py` 覆盖独立 source gather 与复数 RoPE oracle、
condition 因果性、mask0/80%、activation recompute 梯度和普通推理；
DCP resume 测试核对连续/恢复运行的排列、参数、optimizer、EMA 和 loss。
`test_distant_target_gpu.py` 同时覆盖原有单 horizon 和本模式的 FA4 路径。
具体硬件与分布式验证范围以当前测试证据为准；新模式不宣称多机/USP 已验证。


## 分离表征与因果建模：ViT-E / ViT-C（Native preview）

对 conditioner 设置 `model.extras.conditioner.architecture=split_ec`，支持 latent 或网格对齐的 RGB 输入。
`causal` 为默认值，保留原单 ViT 模型；拆分模型有新的参数命名和中间归一化，
不自动把旧单 ViT checkpoint 当作拆分模型加载。两段与 DiT 一起训练，梯度不截断。

数据路径为：

```text
packed latent [B,P*C,S,H,W]
    -> ViT-E：每个 state 独立、完整 attention
    -> 表征 [B,S,P*H*W,D]
    -> 可选 token mask
    -> ViT-C：沿 state 序列做因果 attention
    -> 最后一个 state 的整组 token [B,N_visible,D]
    -> 原有 DiT condition 注入方式
```

ViT-E 把 state 折叠到 batch 维，所有 state 共享同一套参数。
它没有 causal mask，也不丢 token；每个 state 内的所有 latent planes / 空间位置
可相互读取，不读取其他 state。E 的三轴 RoPE 在每个 state 内从时间 0 开始，
所以同一 latent 放在不同历史位置、单独编码或组成 batch 编码，其表示不依赖历史。
浮点实现的不同 batch 形状可能有数值舍入差异。

ViT-E 输出经过 LayerNorm，保持 token 的 plane/h/w 顺序；
不使用 CLS pooling 或 prefix expander。ViT-C 直接接收这些表示，不再次 patchify。
C 的 RoPE 沿序列使用连续 latent-plane 时间，state 内双向可见，state 之间只看当前
及过去。训练可一次计算所有合法 prefix 的结果；推理取最后一个 state 的所有 token。
这两段复用当前 ViT block 和 Native attention backend。

`context_mask_ratio` 只在 E 输出之后、C 的 transformer blocks 之前生效。
空间采样仍按 sample/state/plane 独立进行，并保留原始索引供 DiT 的入口 concat /
same-spatial CA / CA 3D RoPE 使用。被丢 token 在 E 中已参与表示计算，
因此这种 mask 与原来在单 ViT 前丢 token 的信息可见性及计算量不同。
eval 不执行随机 token mask。

### 层数与参数预算

`conditioner.depth` 表示 E+C 的总 transformer 层数，宽度、heads、MLP ratio 共用。
`encoder_depth=null` 时默认把约四分之一层数分给 E，其余给 C；
也可以显式设置 E 层数。两段都至少需要一层。
这近似保持原 ViT:DiT=2:1 下的 E:C:DiT=0.5:1.5:1，
最终比例应按参数统计核对，层数取整及 projection/norm 会带来偏差。

例如在原 19 层 latent ViT 的配置上增加：

```yaml
model:
  extras:
    conditioner:
      input: latent
      architecture: split_ec
      depth: 19
      encoder_depth: 5
      feature_layers: [18]
```

得到 E5 / C14。feature_layers 仍按 E+C 合并后的零起始层号解释，
但必须选择 C 内的层；[18] 表示最后一个 C block 的输出。
CA / entry concat、普通 / distant / quadrant 目标及 EMA 配置分别沿用各自开关。

### 分开调用与推理缓存

单次 `operation=condition` 仍执行 E+C 并返回最后 state 的 condition。
额外提供两个经 Native engine_forward 调用的操作：

- `operation=encode_states, source=packed_latents`：返回 E 表征 [B,S,N,D]。
- `operation=prefill_condition, source=encoded_states`：C prefill 并返回 [B,N,D]。

默认 split rollout 缓存 E 表征：观测历史只编码一次，后续只编码新生成的 state，
缓存按 history_window_states 截断。C 每轮对当前表示窗口重新 prefill；
本轮没有实现 C 的增量 K/V cache。所有 trainable forward 均经过 Native engine，
不绕过 FSDP 模型入口；E 表征缓存仅在一次固定权重的 rollout 内复用。

本模式当前接受 latent input、USP=1，使用原有 DDP/FSDP 生命周期；
不添加 USP/TP/CP 支持。Native Muon、activation checkpointing、EMA 和 full DCP
自动包含两段参数。旧单 ViT 的 legacy weights-only 映射不适用于此结构；
拆分模型之间的 Native strict load/full resume 按相同结构配置进行。

验证覆盖 state 独立性、完整 E attention、C 因果输出/梯度、post-E mask、
原始 token 索引、分开调用与联合调用、缓存/重算 rollout、meta materialization、
三种 DiT 注入方式及普通/distant/quadrant loss、重算梯度和带 EMA 的 full resume。
FA4 与 DDP/FSDP 的具体硬件验证以当前测试记录为准。


### K=2、左右拼接与等历史预算

Native 支持在同样的 condition 历史范围内比较普通 next-state、随机未来 state 和左右两半拼接：

- `flow.future_target_choices=2` 时，`distant_target_prob=p` 对应 offset 0/1 的概率 1-p/p；默认 choices=3 保留原来的 0/1/2 语义。offset=0 总是普通 next state。
- `flow.target_layout=left_right_mosaic` 时，左右两半各取 offset 0/1，逐 clip、逐预测起点独立等概率交换。原空间坐标、所有 channel/latent plane 的对应关系不变；token-wise RoPE 和 gap embedding 使用每半边的真实 offset。该布局要求 no_zprev、spatially ordered codec、偶数 patch 网格、distant_target_prob=0。
- `flow.target_origin_reserve=2` 将每条 S-state clip 的预测起点固定为 S-2 个，包括 baseline。10-state clip 因而使用 origins 0..7，各自 condition 仅包含 states 0..origin；目标取 origin+1+offset。此字段不截断缓存数据，也不把未来 target 输入 conditioner。
- task 的 `next_state_tuning_start_step=10000` 在完成 10000 次 optimizer update 后，切换为 offset=0 的整 state 目标。必须显式配置 target_origin_reserve；切换保持原预测起点数。optimizer、scheduler、EMA、RNG、data cursor 继续由同一个 Native Trainer/DCP 管理。日志 target_phase 区分 main / next_state_tuning。
- K=2 和左右拼接不改变推理协议：常规 rollout 仍预测 next state，不生成拼接目标。

可直接使用普通 Hydra overrides：

```bash
vforge-train experiment=rep_world \
  model.extras.conditioner.input=latent \
  model.extras.flow.future_target_choices=2 \
  model.extras.flow.distant_target_prob=0.5 \
  model.extras.flow.target_origin_reserve=2 \
  model.extras.next_state_tuning_start_step=10000 \
  runtime.max_steps=12000
```

以上省略环境对应的数据和 codec 路径；与已完成的 VideoRAE3D offline cache 训练兼容。
baseline 使用 distant_target_prob=0；左右拼接同时设置 target_layout=left_right_mosaic 和 distant_target_prob=0。


### K=2 随机等量 token 目标

设置 `model.extras.flow.target_layout=random_token_mosaic`、
`model.extras.flow.distant_target_prob=0`；使用 `no_zprev` 和 spatially ordered codec。
每个 clip 的每个 origin 独立随机选择恰好一半 DiT tokens 取 offset=0（next state），
其余取 offset=1（next-next state）。分配范围是完整 `P × h × w` 网格，包含 latent planes；
总 token 数必须为偶数。每个 token 的所有 channel 与 patch 内像素选择同一个源 state，
保留原 plane/空间坐标，不是左右连续区域，也不是有数量波动的独立 Bernoulli 抽样。
例如 `4 × 16 × 16 = 1024` tokens 时，各时间精确使用 512 个 tokens。

目标由 `source[origin+1+offset, channel, plane, y, x]` 构成；Self-attention RoPE 的
时间为 `offset*P+plane`，3D cross-attention query 为 `(offset+1)*P+plane`，
condition key 为原 `plane`。gap embedding 同样逐 token 使用 offset；H/W 不变。
使用 `target_origin_reserve=2` 时与 baseline 一样保留 `S-2` 个预测起点，condition
不接触未来 state。`next_state_tuning_start_step` 切换后关闭混合、不消耗分配 RNG，
常规 inference 仍只生成 next state。CPU 测试覆盖严格比例、源索引、RoPE、无泄漏、
切换为 baseline 的数值/RNG以及 full DCP 恢复；GPU 状态见对应验证记录。


### 从 EMA 开始 next-state tuning

`model.extras.next_state_tuning_init=ema_reset_optimizer` 要求开启 EMA，并设置
`next_state_tuning_start_step`。在该边界的第一个 tuning 前向前，先把完整 ViT/DiT
EMA shadow 拷贝给 online 参数，再清空优化器状态（Muon momentum、Adam moments 和
optimizer-local step）。保留参数组、学习率、scheduler、全局步数、数据游标与 RNG。
EMA shadow/count 保持边界状态，之后继续按配置 decay 更新。默认 `online` 不执行此重置。

边界前的 checkpoint 同时保存 pretrain online 和 EMA。tuning 的 0-step 模型对应这个
checkpoint 的 EMA；其后 500/1000/1500/2000 步分别评测新的 online/EMA。
同一步的多次 gradient-accumulation 前向只初始化一次。full resume 从边界 checkpoint
恢复时重做此初始化；从已进入 tuning 的 checkpoint 恢复时不再清空优化器。
此选项必须在进入边界前启用，不用于追溯修正已经从 online 开始的 tuning checkpoint。


### RGB split ViT 与缓存目标

使用 `conditioner.input=rgb`、`architecture=split_ec`、`temporal_patch=4`、
`spatial_patch=16`、`input_size=256` 时，VideoRAE3D 的每个 16-frame state
经 RGB Conv3d patch embedding 变成 4×16×16=1024 tokens。E 独立编码每个
RGB state，C 处理同样的 latent-plane 时间单位与 h/w 网格；输出宽度仍为
`conditioner.width`。它与 latent 输入具有相同 token 压缩率，并非 codec 数值等价。
网格不一致会明确拒绝。RGB split 可在 ViT-E 完整编码每个 state 后、ViT-C 前执行 token mask；
被保留 token 的原空间位置会传给首层 CA 的 3D RoPE。首层全空间 CA +3D RoPE
沿用相同坐标，RGB 路径不擅自降低位置编码或转为 SDPA。

训练可用 `latent_cache.mode=offline` 配对原 RGB 与已缓存的目标 latent。
ViT-E patch 投影由 latent C→D 变为 3×Tpatch×Hpatch×Wpatch→D，
所以固定其它层宽和深度时参数量略有增加，应记录实际参数数。
推理的首段 E 读取 observed RGB，之后读取新生成 latent 解码出的 RGB；
缓存 E 表征仅追加新 state，C 仍对当前 history 窗口 prefill。未来 GT 不进入 condition。
RGB/latent cached dataset 使用不同 resume contract，不能绕过校验混用数据游标。

验证覆盖 CPU 的 RGB/cache 逐元素及 worker RNG/游标对齐、E state 独立性、
生成 RGB 历史与重新编码 rollout 一致性，以及 H200 BF16 FA4 前后向/rollout。
RGB split 的单机 8×H200 FSDP2/FA4 训练首步已验证；多机、USP 与完整 DCP 续训尚未验证。

## 直接 RGB4 PixelDiT 实验（2026-09-24）

`codec.family=pixel, codec.variant=rgb4` 不加载 VideoRAE/V-RAE：每个 state 是连续 4 帧、256×256 RGB，按 16×16 空间 patch 无损重排为 `[B,3072,S,16,16]`。这只是张量布局转换，预测和反馈都直接在像素空间进行。ViT-E 用 `temporal_patch=4` 编码同一组 4 帧，mask 在 ViT-C 前；WorldDiT 的首层 global CA 使用 3D RoPE。两级 PixelDiT 先让粗粒度 patch token 做全局注意力，再由 PiT 把每个 patch 的 256 个像素 token 压到 patch 宽度、跨 patch 注意、展开回像素，并输出每像素 12 通道的速度（4 帧×RGB）。直接积分后无损还原 RGB，不经过神经解码器。

355M 消融使用 4 个粗粒度 DiT block + 1 个 PiT block、宽度 768/16，总可训练参数 353,584,192；ViT-E/ViT-C/DiT 分别约 42.1M/234.0M/77.5M。40 帧组成 10 个 state；相比 16 帧一个 VideoRAE state 的原 RGB-input 实验，state 数保持 10，但真实视频时长仅为四分之一。两者 FVD 的生成长度不同，不应将原始 FVD 数值直接视为同等视频长度比较。当前 H200 训练任务使用 10k mask90/K2 预训练、2k mask0/next-state tuning；性能与收敛以任务实测为准。

## Wan2.2 VAE RGB4 消融（2026-09-24）

`codec.family=wan22, variant=ti2v5b` 使用官方 Wan2.2 TI2V-5B VAE 权重（48 latent 通道、时空压缩 4×16×16），冻结 VAE，仅训练 ViT-E/ViT-C/DiT。ViT 仍输入原始 RGB，每个 state 对应连续 4 张 RGB 帧；DiT 在 `[B,48,S,16,16]` 的 Wan latent 空间预测速度。官方 VAE 的首帧独立于后续四帧组编码：适配器先重复真实首帧作为单帧起点，编码后丢弃起点 latent，因此 40 个真实 RGB 帧恰好对应 10 个训练 state；解码时恢复该起点 latent，丢弃起点输出帧。起点不作为 DiT 目标。归一化沿用官方 VAE 内置 48 通道 mean/std，不复用 VideoRAE 统计。

355M 任务的 ViT-E/ViT-C/DiT 总可训练参数 354,974,240，保持首层 global CA+3D RoPE、mask90、K2、batch8/卡、FA4 strict、FSDP2、EMA0.99、10k+2k tuning。VAE 在线编码会增加每步计算，模型计算速度和总吞吐应分开报告。推理解码目前逐 state 重算全部已生成 latent 前缀以保持官方 causal VAE 语义，尚未优化为流式 cache。

官方 VAE 源码在 `vforge/models/rep_world/vendor/wan22/`，附原始 Apache-2.0 许可与 SHA256；模型权重来自 `/efs/stormyxie/model_zoo/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。端到端任务状态见 `/efs/stormyxie/reports/wan22_rgb4_355m_20260924/`。
