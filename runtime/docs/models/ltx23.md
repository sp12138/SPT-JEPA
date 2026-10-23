# LTX-2.3

LTX-2.3 已支持 causal teacher forcing（TF）和 DMD 训练，统一使用 vForge Native 入口。

| 训练模式 | Experiment | 状态 |
| --- | --- | --- |
| Stage 1 causal teacher forcing | `ltx23_causal_ar_tf` | 已支持 |
| TI2AV SFT | `ltx23_ti2av_sft` | 待支持 |
| DMD | `ltx23_dmd` | 已支持 |

SFT experiment 目前仍属于开发中的内部入口，不应作为已验收能力使用。本文先说明
Stage 1 TF 流程，再介绍 DMD 训练的使用方法。

TF 的对齐背景和验证记录见：

<https://doc.weixin.qq.com/smartpage/a1_Af8AxQb1AGUCN906NOmA1SzKHekDZ?scode=AJEAIQdfAAo13EcsI0>

## 前置条件

训练需要：

1. LTX-2.3 单文件 `.safetensors` checkpoint；
2. 与 checkpoint 配套生成的 `.precomputed` 数据；
3. 支持 FlashAttention-4 的 CUDA 环境；
4. 使用新的输出目录，避免覆盖已有训练结果。

数据目录至少包含以下三路文件：

```text
/data/ltx23/train/
  .precomputed/
    latents/
      shard-000/sample-000001.pt
    audio_latents/
      shard-000/sample-000001.pt
    conditions/
      shard-000/sample-000001.pt
```

`latents` 与 `audio_latents` 使用相同的相对路径。`conditions` 通常也使用相同路径；
旧格式的 `latent_<name>.pt` 可以匹配 `condition_<name>.pt`。

## 单机 8 卡运行示例

下面的命令运行 `DP1 × USP8` Stage 1 TF。视频/音频的 structured attention 使用
FlashAttention-4，文本 cross-attention 使用 PyTorch SDPA。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 \
  -m vforge.tasks.train \
  experiment=ltx23_causal_ar_tf \
  model.pretrained_path=/models/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  data.cache_dir=/data/ltx23/train/.precomputed \
  data.num_workers=0 \
  data.persistent_workers=false \
  data.prefetch_factor=null \
  runtime.output_dir=/runs/ltx23-causal-ar-tf \
  runtime.max_steps=10 \
  scheduler.total_iters=10 \
  parallel.dp_size=1 \
  parallel.usp_size=8 \
  attention.backend=flash_attention_4 \
  attention.allow_fallback=false \
  runtime.deterministic=true \
  attention.blas_backend=cublas \
  checkpoint.enabled=false \
  resume.path=none
```

## 运行流程

1. vForge 从 checkpoint metadata 构造 LTX-2.3 Transformer，并加载训练权重；
2. dataloader 读取配对的 video latent、audio latent 和预计算文本条件；
3. 每个逻辑样本分别组装 video/audio causal teacher-forcing packed sequence；
4. 一个 USP group 共同处理一个逻辑样本，序列按 `usp_size` 尾部补齐并连续分片；
5. structured video/audio attention 通过 Ulysses 和 segmented FlashAttention-4 执行；
6. forward 后 gather 完整 prediction，移除 padding，只在 noisy token 上计算 loss；
7. `train_steps.jsonl` 记录 loss、GradNorm、step timing、吞吐和显存峰值。

默认配置关闭 autocast，使用 BF16 计算、FSDP2、gradient checkpointing、全参数 AdamW，并要求：

```text
world_size = dp_size × usp_size
tp_size = 1
cp_size = 1
include_usp_in_fsdp = true
data.batch_size = 1
```

`runtime.output_dir` 必须使用新的路径。需要保存 checkpoint 时，移除
`checkpoint.enabled=false`，并按通用 checkpoint 配置设置保存周期和恢复路径。

通用 checkpoint 行为见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)，分布式启动参数见
[分布式运行](../features/distributed-training.md)。

## Loss 回归基线

Stage 0/1 的 B300、DP2×USP2、10 步 loss golden 随仓库保存：

- [Stage 0 JSON](https://git.woa.com/ieg-infra/vforge/blob/master/tests/models/ltx/stage0_loss_b300_dp2_usp2.json)
- [Stage 1 JSON](https://git.woa.com/ieg-infra/vforge/blob/master/tests/models/ltx/stage1_loss_b300_dp2_usp2.json)

测试直接读取仓库中的 JSON，模型权重和数据仍从外部存储读取；后续重算通过
Git diff review 数值变化。

训练 RNG 当前改为持续推进；Stage 1 噪声流继续按 DP 副本区分、由 USP peers 共享。
2026-09-16 已重新采集并独立复跑全部 6 项 B300 loss gate：Stage 0/1 各 10 步，
DMD 在 DP8×USP1、DP4×USP2 下各 10 步，Long Tuning 在两种拓扑下各 18 步。
全部保留 `float.hex` 逐位比较；环境为 PyTorch 2.13.0+cu130、CUDA 13.0、
FA4 4.0.0b29，seed=42、BF16 计算、FP32 reduce/norm，启用确定性并关闭 TF32。
DMD 与 Long Tuning 的数值分别保存在
[DMD JSON](https://git.woa.com/ieg-infra/vforge/blob/master/tests/models/ltx/dmd_b300.json) 和
[Long Tuning JSON](https://git.woa.com/ieg-infra/vforge/blob/master/tests/models/ltx/long_tuning_b300.json)。
旧的逐 step 随机序列已改变，下文历史恢复验收不覆盖这次修改；
Long Tuning 的完整 GPU checkpoint/resume 仍**未验证**。

## DMD 训练

可直接使用 ieg-infra-efs（挂载为 `/efs`）上的 **DMD 范例数据**：
`/efs/alexlshen/vforge/examples/ltx23-dmd`。其中 `data/.precomputed` 包含 16 个
固定样本，`model` 包含 base、generator、score 权重和完整 Gemma；
原始缓存为 113 帧，完整 DMD rollout 为 97 帧。
完整可复制命令与 CI 用法见[新版 LTX 指南](https://ieg-infra-vforge.devcloud.woa.com/models/ltx23/dmd/)。

DMD 使用方法、原始尺寸与不同 SP 的对齐记录见 [DMD 验证记录](https://doc.weixin.qq.com/smartpage/a1_AGYA7nhqAAkCNl7hbK9cYS0qLBxXZ_a?scode=AJEAIQdfAAoVAzst8PAGYA7nhqAAk)。

`ltx23_dmd` 使用已有的 TF generator 和 bidirectional score checkpoint，复用
三塔装配、双 optimizer、G/C step 和 rollout 主流程，使用 `PhasedTrainer`，
通过 `block_distribution_matching` 选择数值规则与 replay 扩展。
标准 `distribution_matching` 保留原数值语义。

Generator 默认使用 `training.dmd.rollout.backward_mode=retained_graph`，保留各块
退出步计算图后统一 backward。对照参考的执行方式时可选择
`training.dmd.rollout.backward_mode=block_replay`：先无梯度 rollout 并保存退出步
输入，再重建 reference/clean KV，每块退出步重放后立即 backward，全部块完成后
只裁剪和更新一次。历史 KV 始终截断梯度，两种方式使用相同的逐块 loss 权重。
该选项属于 block DMD，不改变标准 `distribution_matching`。

`block_replay` 目前只支持 FSDP2 engine，不支持 activation offload；普通梯度重算
可开启。它额外执行退出步 forward 和历史 prefill，以降低同时存活的激活图数量。
`ltx23_dmd` 通过训练组件自动装配 PyTorch 裁剪，保留 DTensor，
只裁剪当前 phase 的模型塔。其他 experiment 保持公共 `logical` 默认。
PyTorch 裁剪要求 FSDP2、TP=1，无需配置实现选择开关。
对照实验可设置 `runtime.gradient_norm_dtype=float64`，只提高 norm 计算精度，
forward 与 reduce 仍分别保持 BF16/FP32。参考侧需使用对应的 FP64 norm 配置。
改变 backward 或裁剪方式会改变数值轨迹，历史报告不能作为新路径逐位对齐的证明。


验证必须区分三种范围：

- **正式入口端到端对齐**：参考保留原始 DMDTrainer、Accelerate/FSDP1 和独立
  teacher；vForge 使用 Native Trainer/FSDP2。两侧各自加载数据、编码负向条件并
  采样随机张量，不使用输入记录/重放。已完成 FP32 reduce、MATH 小尺寸十步，
  以及 FA4、49 帧、480×832、重算开启的各 100 步长跑（20G/100C）。后者稳定性
  通过，loss 最大相对差异仍为 16.41%，尚未通过完整参数梯度和更新权重对齐。
- **原始缓存尺寸短测**：同机 8×B300、DP8×SP1、BF16 compute、FP32 reduce/norm、
  deterministic FA4、重算开启；Native 使用 `block_replay`。16 个原始缓存文件
  保持字节一致，latent `[128,15,15,26]`、空间 480×832；完整块规则选择
  `4+3+3+3=13` latent frames，实际 rollout 为 97 帧。十步结果 G 2/2、C 6/10
  的全局平均 loss 精确相同；C 从 step 6 开始分歧，最大相对差 4.091%。该步
  grad norm 为 0.86371505 / 2.63285875，仍为 `DIFFERENCE`，不能作为梯度对齐。
  两侧 loader 独立核验 704 项、条件/位置/sigma/RNG 核验 332 项通过；未记录正式
  训练全程所有运行时输入。原始尺寸 SP8 尚未运行。详细记录见外部对齐文档。
- **固定输入控制实验**：已有目标尺寸 forward/loss/分布梯度检查，以及两侧 FSDP2
  的梯度和更新诊断。这些实验绕开了部分数据或训练生命周期，不能替代正式入口验收。
- **保存恢复**：历史 BF16 reduce、MATH 和重算开启的连续十步与 5+5 DCP 比较已通过，
  generator 导出严格加载也已验证。当前默认 FP32 reduce 的完整恢复尚待复验，
  不沿用历史精度配置的报告作为当前配置验收。

FA4 可通过公共 `attention.backend=flash_attention_4` 选择；上述长跑显式开启
`runtime.deterministic=true`，稳定性通过不代表数值对齐通过。带 mask 的 connector
与 transformer 路径需要另外核对其
SDPA 策略，不能把配置了 FA4 理解为模型所有 attention 均使用 FA4。
专项结果、参考补丁和运行配置保存在上述 DMD 验证记录中。历史对齐工具保留在
MR 精简前的提交 `7d23660`，不属于当前分支的 CI 入口。当前沿用主线统一 pytest CI，
已接入 DMD CPU loss/梯度用例和八卡 B300、DP8/SP1 十步 GPU 回归；12/12 loss
与固定 Native 基线精确一致，输入和模型权重哈希校验通过。

该流程要求 FSDP2、TP/CP 为 1、batch size 为 1；DMD 可通过
`parallel.usp_size=8` 启用 Ulysses，默认仍为 1。每五个全局步执行
`G+C, C, C, C, C`，即 **critic:G=5:1**；十步应有 2 次 G 和 10 次 C 更新。
generator 分块自回归 forward；默认 retained_graph 汇总各块 loss 后统一 backward，
block_replay 则逐块重放并 backward。历史 KV 截断梯度；重算复用当前块原始的 KV 快照和 FP32 positions。

DMD Ulysses 复用现有序列切分、all-to-all 和输出 gather。生成块的 Transformer 激活
按 token 切分，clean KV 在 all-to-all 后按 head 缓存。无梯度的 reference prefill
在各 USP rank 使用完整形状计算，再仅保留本 rank 的 KV heads，以减少短 reference
因局部行数和布局改变产生的 BF16 数值差异。该阶段临时保留完整 reference KV；
目标分辨率下的临时显存和性能尚未验证。head 数可整除 USP 时，持久 KV 元素数为
USP=1 的 `1/usp_size`。仅交换当前生成块，不重复交换完整历史 KV。
视频和音频分别补齐序列；attention 前移除无效 token，输出再补齐供反向通信。
文本条件在 USP 组内复制，reference-follow RoPE 和重算快照语义保持不变。

SP 对拍必须显式固定 `parallel.dp_size`；`WORLD_SIZE = dp_size * usp_size`。
例如 `DP1×USP1` 对 `DP1×USP8` 分别需要 1 与 8 个进程，不能用八卡
`DP8×USP1` 代替前者。DMD 使用完整样本 loss：输出 gather 的 backward 为 SUM，
FSDP 在 DP×USP 上取均值，两者抵消 USP 倍率，不再额外除以 `usp_size`。
B300 小型独立三塔的 DP1×USP1/USP8、FP32 MATH 十步梯度与权重对拍已通过固定
门限，`retained_graph` 和 `block_replay` 均有验证。BF16 FA4 十步存在两个 generator
输出投影梯度超出门限，不能标为精度通过；零学习率控制实验通过同一门限，所有 loss
一致，说明权重更新会放大两种分片下已有的低精度梯度差异。固定 seed 和确定性
attention 不保证不同 SP 拓扑逐位一致。
真实三塔 DP1×USP8 在 49 帧、128×128、BF16 FA4、重算开启下完成十步（2G/10C）；
它证明该配置的执行稳定性，不代替真实 checkpoint 的 SP1/SP8 数值对拍。
目标尺寸长跑与 SP DCP 恢复尚未验证。

DMD cache 只需要 `latents/` 和 `conditions/`。视频提供首帧参考、生成长度和空间
尺寸，待生成音视频均从噪声开始，不要求目标 audio latent。SFT/TF 的三路数据要求
不变。每个 rank 启动时按生成块数分桶；G/C 分别维护可恢复的数据流，
每个全局步均推进两流。sampler seed 跟随 runtime seed，G/C 分别加 0/7919；
默认丢弃不足一个 DP 组的尾桶。与参考库比较时，在参考验证侧规范化样本排序，
并适配为相同的双流推进方式；不修改公共 loader 的消费规则。

base checkpoint 提供结构和冻结 connector，generator 从 TF 权重初始化，
real/fake score 默认从相同的 bid checkpoint 初始化。rank 0 仅在启动时用 Gemma
编码 negative prompt 并广播，三塔装配前释放 Gemma。训练参数保留 FP32 master，
forward 使用 BF16；teacher 与 connector 冻结。

正向 connector 在 autocast 外执行，生成的双模态条件保持 BF16，避免训练入口的
autocast 将条件归一化提升为 FP32。LTX 的 `input` loss 精度也显式保留 BF16
平方和均值运算，各块 generator loss 最后在 FP32 中累加。参考项目在模型 forward
之外计算 loss，因此直接沿用整个 step 的 autocast 会产生不同的舍入；这属于精度
执行边界差异，并不表示 FP32 loss 本身不正确。

当前 DMD 使用 `parallel.mixed_precision.param_dtype=bfloat16`，由 FSDP 汇集 BF16
参数执行前向和反向，FP32 master 参数由 optimizer 更新；`runtime.autocast=false`。
历史对齐使用 FP32 汇集加 BF16 autocast。模型侧显式保留 FP32 norm 输出及后续调制，
只在 Linear 输入和 attention kernel 入口转换计算 dtype。独立 DMD connector
继续使用原来的 BF16 norm；TF latent 和位置数据不做整体降精度转换。
FSDP 参数 dtype 的变化仍需分别验证 SP1、SP2，不能由相同 seed 推断 bitwise 一致。

`parallel.mixed_precision.reduce_dtype` 沿用框架默认的 `float32`，experiment 不覆盖。
历史对拍显式使用 `bfloat16` 匹配参考 Accelerate/FSDP1 的梯度通信精度。
统一 backward 在累加块梯度后舍入，参考逐块 backward 则逐块
归约后再累加，因此两者仍可能有 BF16 舍入差异；FP32 本地累加并不保证最终
梯度逐位一致。验收需同时记录原始差异及其来源，不能把 loss 相同当作梯度相同。
这种初始梯度差异可能经 AdamW 更新放大，使后续 loss 偏离。两侧 FSDP2 的历史
控制实验中，FP32 reduce 消除了该次多步 loss 差异；保留参考 FSDP1 的正式入口
实验仍有差异。归约精度相同也不保证裁剪范数的求和顺序相同。与参考比较时需同时
记录 attention、梯度累加、通信和裁剪策略，逐项定位，不能仅凭相同 seed 或 FP32
reduce 宣称数值一致。

LTX 模型在 attention 调用边界将 Q/K 转为 V 的计算 dtype，不提前降低 norm、
RoPE 和存储的 KV 的精度。位置坐标、RoPE 系数及各训练路径的通信入口保持原样。

数值对拍也要匹配各 attention 路径的 kernel 策略和 tensor 布局。`sdpa` 允许
PyTorch 自动选择 fused kernel，不能视为显式 MATH；connector 与 masked attention
还可能采用不同的 dispatch 策略。先排除这些设置差异，再判断模型或算法实现误差。

定位重复训练的数值差异时，可设置 `attention.backend=sdpa_math`。LTX transformer
的 masked attention，以及 DMD 独立加载的正负条件 connector masked attention，
均遵循该选择。Gemma 编码器仍使用其 SDPA 配置。默认 `sdpa` 允许 cuDNN 等路径，并不承诺独立
运行之间逐位一致。恢复报告分别比较 checkpoint 刚加载完成的全部训练状态和
后续训练轨迹，避免把保存之前已有的数值分歧误判为恢复错误。

以下命令用于内部验收，路径需替换为已有输入和新的输出目录：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=ltx23_dmd \
  training.models.generator.pretrained_path=/models/ltx-base.safetensors \
  training.models.generator.init_weights=/models/ltx-tf.safetensors \
  training.models.real_score.init_weights=/models/ltx-bid.safetensors \
  training.models.generator.config.text_encoder_path=/models/gemma-3 \
  data.sources.0.cache_dir=/data/ltx/.precomputed \
  runtime.output_dir=/runs/ltx-dmd \
  runtime.max_steps=10 \
  resume.path=none
```

默认布局为 `4,3,3,…` latent frames，最多 241 pixel frames，音频按 fps 对齐。
rollout 的 `[999,757,522]` 查询 LTX 40 步 sigma 表最近点，退出步默认跨块、跨 rank
同步。score 在整数 `[20,980]` 均匀采样，查询反转的 LTX 1000 步 sigma 表；video/audio
CFG 分别为 3 和 5，使用 `cond+(scale-1)*(cond-uncond)`。双模态梯度分别归一化，
分母加 `1e-8`；score 使用干净首帧并排除该区域，generator 按块求均值再平均。

两套 AdamW 默认 lr=`1e-5`、betas=`(0,0.999)`、weight decay=0，梯度裁剪为 1，
关闭 EMA 和学习率衰减。完整 DCP 与恢复使用通用训练接口，不另设模型专用 Trainer。
配置字段见 [配置参考](../reference/configuration.md)，公共 CI 调度见
[KAP 远程任务](../development/kap.md)。

## DMD 推理

独立推理使用 `ltx23_dmd_eval`，支持 generator safetensors 和 vForge DCP/PT，
输出带音轨的视频。
完整命令见[新版 LTX 指南](https://ieg-infra-vforge.devcloud.woa.com/models/ltx23/eval/)。

## Long Tuning

使用 `experiment=ltx23_long_tuning` 从已训练的 DMD student 开始段链训练。
启动、段参数和恢复配置见[新版 LTX 使用指南](https://git.woa.com/ieg-infra/vforge/-/blob/feature/ltx-long-tuning/docs-next/models/ltx23.md)。
真实 GPU 百步参考对齐、SP2 与链中途恢复验收尚未完成。
