# CD 训练

!!! warning "当前状态"
    `wan22_ti2v_5b_cd` 已接入 VForge Native CD runtime，真实 Wan2.2 TI2V-5B
    checkpoint、148-latent cache、FSDP2、USP4 和 FA4 已完成 20-step
    forward/backward/optimizer smoke。固定权重计算可以与 Taichu 逐位一致；生产非零
    LR 的 20-step loss 高度相关但尚未达到全程 bitwise。当前能力可用于迁移验证，
    不能把现有短程结果外推为完整训练收敛等价。

    `wan22_a14b_cd` 已接入双专家算法、family adapter、独立 EMA 和 39-latent
    配置。目前只完成 CPU contract；USP8/FSDP2 真实 checkpoint 的 GPU 闭环与
    Taichu parity 仍是发布门禁。

## 训练对象

Wan2.2 TI2V-5B Causal Consistency Distillation 同时持有三个同构 transformer：

- `teacher`：冻结的 causal teacher，预测 velocity 并执行一步 Euler；
- `student`：唯一可训练模型，也是唯一 optimizer target；
- `ema_model`：冻结的 target model，每一步 forward 前从 FP32 EMA shadow 同步。

三个角色都使用 `wan22_ti2v_5b_causal_teacher`，但它们是三个独立模型实例和三个
FSDP2 execution root。teacher、student 与 EMA model 不会共享 parameter object。

CD 使用额外的 `student_ema` 保存 FP32 shadow。它不是第四个 transformer，不执行
forward，也不作为 FSDP execution root。

## 配置：CD 是 recipe，不是 model variant

通过 `training.recipe=cd` 选择训练方式，并显式声明三个角色：

```yaml
training:
  recipe: cd
  models:
    teacher:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b_causal_teacher
        condition_streams: [grid, normal]
      init_weights: /path/to/causal-teacher.pt
    student:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b_causal_teacher
        condition_streams: [grid, normal]
      init_weights: /path/to/causal-teacher.pt
    ema_model:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b_causal_teacher
        condition_streams: [grid, normal]
      init_weights: /path/to/causal-teacher.pt
  cd:
    algorithm: causal_consistency_tf
```

三个 role 的结构配置、pretrained root、初始化 checkpoint、加载模式和 source prefix
必须一致。当前版本只允许：

```yaml
training:
  cd:
    algorithm: causal_consistency_tf
    discrete_levels: 40
    sigma_grid: wan
    pair_sampling: adjacent_uniform
    consistency_shift: 5.0
    guidance_scale: 1.0
    ema:
      decay: 0.995
      start_step: 0
```

未知字段或改变生产数学语义的取值会在模型分配前 fail-fast。

## 数据与条件

第一阶段只支持 offline Wan Taichu latent cache：

```yaml
latent_cache:
  mode: offline

data:
  sources:
    - cache_dir: /path/to/gameplay-cache
      min_latents: 148
      max_latents: 148
  batch_size: 1
```

cache 必须提供：

- `frame_latents`；
- grid condition latent；
- normal condition latent；
- `first_frame_latents`；
- prompt embedding。

每一种 latent 都是 48 channels。模型输入为：

```text
noisy/clean latent  48 channels
grid condition      48 channels
normal condition    48 channels
total              144 channels
```

生产长度为 148 latent frames，并满足 `148 = 1 + 3 × 49`。frame 0 是 clean
image anchor；其余 latent 每三个组成一个 causal block。

USP group 内只有 batch leader 读取和处理 cache，随后向同组 rank 广播完全相同的
batch。pair index 和 noise 使用 `(seed, global_step)` 派生的 step-local generator，
先采样 adjacent pair，再对完整 latent shape 采样 noise。

## 一个 student phase 的计算顺序

每个 global step 只包含一个 `student` phase：

```text
prepare one cache batch
  → sample adjacent (t, t_next)
  → add flow noise at t and restore frame-0 anchor
  → frozen teacher forward at t
  → one Euler step to x_next
  → trainable student forward at t
  → frozen EMA-model forward at t_next
  → convert both velocities to x0
  → MSE over frames 1..T-1
  → student backward / clip / optimizer / scheduler
  → update FP32 student EMA
```

teacher 与 EMA model 始终处于 `eval`、`requires_grad=False` 和 `no_grad` 边界。
只有 student 接收梯度。

## 40 档 Wan schedule

CD 使用不带 terminal-zero sentinel 的 40 档单次 shift grid。对
`i = 0, ..., 39`：

```text
s_i     = 1 - i / 40
sigma_i = 5 * s_i / (1 + 4 * s_i)
t_i     = 1000 * sigma_i
```

每一步从 `[0, 38]` 均匀采样 `i`，得到相邻下降 pair：

```text
(t, t_next) = (t_i, t_{i+1})
```

加噪和 teacher Euler 分别为：

```text
x_t    = (1 - sigma_t) * x0 + sigma_t * noise
x_next = x_t - ((t - t_next) / 1000) * teacher_velocity
```

两个结果都会重新写入 clean frame-0 anchor。

## Doubled teacher-forcing

teacher、student 和 EMA model 都通过 `forward_teacher_forcing` 执行完整 doubled
forward：

```text
clean_cond = concat(clean_context, grid, normal)  # [B, 144, T, H, W]
noisy_cond = concat(noisy,         grid, normal)  # [B, 144, T, H, W]
hidden     = concat(clean_cond, noisy_cond, time) # [B, 144, 2T, H, W]
```

clean half 的 timestep 全为 0；noisy half 的 frame 0 为 0，其余 token 使用当前
scalar timestep。transformer 最终只返回 noisy half 对应的 48-channel velocity。

velocity 转 x0 时使用 Taichu 的 FP64 中间精度：

```text
x0 = double(noisy) - double(sigma) * double(velocity)
```

结果再转回 velocity dtype。loss 在 BF16 prediction dtype 中对 frame 1..T-1 做
unweighted mean MSE；frame 0 不参与 loss。

## EMA

CD 同时维护：

- `ema_model`：可执行 forward 的冻结 transformer；
- `student_ema`：保持当前 DTensor placement 的 FP32 `ShardedEmaState`。

执行顺序是：

```text
step 0 target forward:
  ema_model 使用初始化 checkpoint

step 0 optimizer 后:
  student_ema 第一次 update 直接复制更新后的 student

step N target forward:
  student_ema copy_to ema_model

step N optimizer 后:
  shadow = 0.995 * shadow + 0.005 * student
```

`initialized`、`num_updates` 和所有 shadow buffers 都是 persistent state。
`copy_to` 支持 FSDP2 DTensor student shadow 与 replicated EMA model 之间的显式布局
转换。

## 并行与显存

生产 experiment 使用：

```yaml
parallel:
  usp_size: 4
  dp_size: auto
  dp_mode: fsdp2
  include_usp_in_fsdp: true
  gradient_checkpointing: true
  reshard_after_forward: true

attention:
  backend: flash_attention_4
  allow_fallback: false
```

148-latent doubled sequence 通过 USP4 分摊激活压力；三个 transformer 分别作为
FSDP2 root。`forward_teacher_forcing` 被显式注册为 FSDP forward method，保证从该
入口调用时也执行正确的 unshard/reshard。

当前校验允许 `usp_size in {1, 2, 4, 8}`，且 doubled token 数必须能被 USP size
整除。2026-09-10 的 4×B300 实测：

- USP4、DP1：20-step 非零 LR 正常完成，VForge 峰值约 83.3 GiB allocated、
  98.9 GiB reserved；
- USP1、DP4：148-latent 两步正常完成，峰值约 189.2 GiB allocated、
  214.7 GiB reserved。

USP1/DP4 会让每个 DP rank 读取不同样本，global batch 和 Taichu SP4/DP1 不同，
因此该 smoke 只证明容量和执行闭环，不能用于直接 loss parity。

## Optimizer 与 LR

student 默认使用：

```yaml
optimizers:
  student:
    name: adamw
    learning_rate: 5.0e-6
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1.0e-15

scheduler:
  name: constant_with_warmup
  warmup_steps: 500
```

第一个 optimizer step 的 LR 为 0，之后每步增加 `1e-8`，第 500 步才达到
`5e-6`。因此 20-step 短程对比只覆盖 warmup 初段，不能代表 full-LR 训练轨迹。

Taichu 生产 CD 使用 DeepSpeed stage 1 的 BF16 working parameters 和 FP32 master
更新；VForge 当前使用 FSDP2 的 FP32 optimizer-facing parameters、BF16 mixed
precision forward 和 `torch.optim.AdamW`。两条路径数学等价，但不承诺更新后参数
逐位一致。

## 确定性与 Taichu parity

严格排查建议固定：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=42 \
VFORGE_ULYSSES_GATHER_BACKWARD=reduce_scatter \
NPROC_PER_NODE=4 \
bash scripts/train.sh \
  experiment=wan22_ti2v_5b_cd \
  runtime.deterministic=true \
  runtime.max_steps=20 \
  checkpoint.enabled=false
```

`reduce_scatter` 使 Ulysses gather backward 的 collective 与 Taichu 数值路径一致。
它不会自动统一 DeepSpeed 与 FSDP2 的 optimizer parameter layout。

2026-09-10 在同一台 4×B300、PyTorch `2.13.0+cu130`、FA4 `4.0.0b29` 上得到：

- LR=0：Taichu 两次、VForge 两次各 10 step，40/40 loss float32 bits 一致；
- 生产 scheduler：20 step 中 14 step loss bitwise 一致；
- 平均绝对误差 `7.391e-7`；
- 最大绝对误差 `5.722e-6`；
- 最大相对误差 `1.887%`；
- Pearson correlation `0.999948`。

这证明 CD 数据、RNG、teacher/student/EMA forward 与 loss 主路径已经高度对齐；
它不是生产训练轨迹 bitwise 证明。跨 engine 的严格多步 bitwise 仍需对齐 gradient
reduction、clipping、FP32 master、AdamW kernel 和 BF16 working-copy 回写。

## Checkpoint 与恢复

CD 复用 `PhasedTrainer` 的 full DCP 生命周期，逻辑状态包括：

- teacher、student、ema_model；
- student optimizer 与 scheduler；
- FP32 `student_ema` shadow、初始化状态和更新次数；
- dataloader cursor；
- RNG 与 global step；
- topology、执行和配置 fingerprint。

当前 CPU/tiny contract 已覆盖 role 冻结、单 student gradient、首次 EMA copy 和
配置校验。真实 148-latent full-DCP save/resume 后下一步 bitwise，以及
student/EMA 的 Taichu-compatible weights-only 导出，仍需单独完成生产 gate。

## Wan2.2 A14B 双专家 CD

A14B 使用 `causal_consistency_moe_tf`，三个 role 各自持有 `wan` 和 `wan_2`，
共暴露六个 FSDP execution roots。单个 optimizer 同时持有两个 student expert，
但每步只有一个 expert 进入计算图：

```text
even step -> wan   -> t,t_next >= 900
odd step  -> wan_2 -> t,t_next < 900
```

相邻 timestep pair 必须完整落在一个 noise band。两个 student expert 分别维护
`student_ema_wan` 和 `student_ema_wan_2`，optimizer 后只更新当前 step 的 EMA。

A14B 条件顺序固定为 `[rgb, grid, normal]`，输入宽度为
`16 + 3 × (4 + 16) = 76` channels。它没有 5B 的 first-frame inpaint 语义：
frame 0 正常加噪、noisy timestep 不清零，并参与 full-frame consistency loss。
生产配置使用 39 latent、shift 3.0、boundary 0.9、local attention 30、sink 9
和 USP8。

完整配置见 `experiment=wan22_a14b_cd`。

四卡 B300 验证已用真实 checkpoint/cache 完成最小合法 3-latent 的
USP4/SP4 forward bitwise：输入、noise、timestep condition、block 0/39、
model output、student/EMA x0 和 loss 的哈希全部一致。生产 39-latent 仍要求
USP8；Taichu DeepSpeed stage 2 在四卡会因复制 85.7B 参数并创建 Adam state
而 OOM，这不是 CD forward 语义错误。

## 当前限制

- 仅 offline latent cache；
- 仅 FSDP2；
- 只有一个 student phase 和一个 optimizer；
- 不支持 KV-cache rollout、offline ODE trajectory 或 CD/DMD 联合训练；
- A14B 尚未完成生产 39-latent USP8/FSDP2、DCP resume 和发布导出门禁；
- USP4 生产 shape 已运行，但非零 LR 跨 Taichu 全程 bitwise 尚未完成；
- full-LR 长程收敛、真实 DCP resume 和发布导出仍需独立验收。

算法和模型家族的职责边界见
[CD 架构设计](../development/cd-architecture.md)，模型接入约束见
[Wan CD 接入说明](../development/wan22-5b-cd-extension-plan.md)。

## CD 裁剪作用域与数值复核

CD 沿用 trainer 按参数身份建立的梯度贡献范围，在当前 student 子模型上执行
FP32 逻辑范数裁剪。贡献过滤不依赖 `student.*` 等名称前缀，也无需在更新时
重新构造；所有本地梯度仍使用同一个裁剪系数。

数值验收使用的临时张量/梯度采集 probe 已从开发代码移除，历史结果与实验脚本
归档在仓库外。正常训练保留梯度范数、裁剪系数等公共诊断指标；
首步 bitwise 结论不等于完整训练状态全程等价。
