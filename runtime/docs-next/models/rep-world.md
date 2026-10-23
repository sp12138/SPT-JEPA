# RepWorld

RepWorld 是 World-DiT 与 causal ViT 的 Native 适配，支持 VideoRAE 3D/1D、V-RAE
EUPE/V-JEPA2.1，以及 RGB 或 latent 输入的 trainable ViT。目前处于内部验证阶段。
已验证 H200 单机与 B300 两节点的 DDP、FSDP2、HSDP、Ulysses 指定组合；CP 暂未接入。

训练、推理、数据 contract、短名 experiment、权重初始化和已验证边界统一维护在
[RepWorld Native 指南](https://git.woa.com/ieg-infra/vforge/blob/xj_dev/docs/models/rep-world.md)。
不要将旧研究 checkpoint 的 weights-only 初始化当作 optimizer/data cursor 的完整恢复。

保存与恢复的通用规则见[保存、恢复与自动容错](../guide/training-recovery.md)。


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

The masked training path currently requires latent input and USP=1; attempts
to combine masking and Ulysses fail during preflight. Existing unmasked USP
behavior is unchanged. This feature changes the training objective's condition
distribution, so start a separate experiment rather than treating it as an
equivalent continuation of an unmasked run.

## Split ViT 推理 KV cache

ViT-E/ViT-C 拆分模型可在评估时设置
`model.extras.rollout.use_vit_c_kv_cache=true`。ViT-C 每层保留已生成 state 的
K/V，新 state 只计算自己的 Q/K/V，并继续使用原来的 3D RoPE 时间坐标。训练和
checkpoint 不变；默认关闭。当前实现要求完整历史，即
`model.extras.rollout.history_window_states` 至少为 `total_states - 1`，否则
在构造推理器时直接报错。该开关不缓存 DiT 的 self-attention 或 cross-attention K/V。

单张 H200、BF16、FA4、batch 8、9 state、CFG1、50 步的模型前向测速已验证三个
355M 拆分模型的 ViT-C condition 与无 cache 版逐元素一致；测量不含 VideoRAE
编码、解码和视频特征提取。Native 分布式评估路径尚未单独验证。
