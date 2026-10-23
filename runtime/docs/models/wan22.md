# Wan 2.2 内部验证

!!! warning "内部验证"
    Wan 2.2 当前用于框架开发和回归，不承诺 prepare/train/eval 的完整用户闭环。不要把
    registry、CPU contract 或单个 loss gate 解读为生产支持。

## Experiment 差异

| Experiment | 模型与训练语义 | 当前证据 |
| --- | --- | --- |
| `wan22_a14b` | I2V 双 expert；high/low noise 按 step 交替；默认 RGB condition stream | Native tiny CPU E2E；历史 adapter 验证不作为 Native GPU 证据 |
| `wan22_ti2v_5b` | 双向 TI2V；首帧 in-paint；首帧不计 loss；默认 normal stream | encode/process/step/sample CPU contract |
| `wan22_ti2v_5b_tf` | 与 TI2V 共享权重、cache、encode；train 使用 block-causal teacher forcing，eval 使用 generated-history causal rollout | teacher-forcing、full-recompute 与 incremental-KV causal eval CPU contract；无模型专属 GPU gate |
| `wan22_ti2v_5b_dmd` | `wan_taichu` Self-Forcing DMD，三个网络（生成器、参照、打分）；通用 phased runtime + Wan adapter；生成器每 5 步更新一次；双 optimizer；分片保存的 EMA | 生产尺寸（121 帧 / 704×1280）与太初参照实现逐位比对：单卡 50 步训练误差全部相同，八卡序列并行第 1 步相同 |
| `wan22_a14b_dmd` | A14B MoE Self-Forcing DMD，六个网络（每个角色各持高/低噪 expert）；逐 expert 独立 KV cache 与 FSDP2 执行单元；随机抽取生成过程的退出档位，expert 选择全 rank 一致 | 完整 40 层模型在 8×B300、704×1280、21 latent frames 上跑通（需 attention offload）；逐位比对只在每个网络 2 层的缩小模型上做过 |

TF cache 必须用 `wan22_ti2v_5b` prepare 路径生成；TF 训练从权重兼容的双向 TI2V
checkpoint 初始化。在其它已有 Native 插件上做同样的训练变体，见
[在已有模型上适配 Teacher Forcing](../development/adapting-teacher-forcing.md)。

TF eval 默认 `evaluation.rollout_mode=full_recompute`。首帧 `z0` 先提交到 generated
history，之后每个 causal block 从自己固定的初始噪声出发走完整 scheduler，再提交到
history；validation 目标 latent 不参与 rollout。输出 sidecar 的 `metadata` 记录实际
`rollout_mode` 和 post-patch latent-frame `block_layout`。`incremental_kv` 每层缓存已经
提交的 clean K/V，并在 block 完成后用 timestep `0` 重新提交 clean context；它当前要求
temporal patch size `p_t=1` 且 `usp_size=1`。需要 Ulysses 时应显式使用默认的
`full_recompute`，框架不会静默切换模式。

官方 Wan 曾用于迁移验证的双向 GridNormal/ref-token 原型已经移除；当前 GridNormal、
在线 ref 和八变体入口属于 `model.name=wan_taichu`。官方
`wan22_ti2v_5b_tf` 的 dedicated teacher-forcing causal eval 仍保留。完整 manifest
字段、命令与输出布局见
[Wan Taichu 评估契约](../development/wan-taichu-evaluation.md)。

### A14B expert 选择

`model.extras.train_expert=high|low|both` 控制 A14B 训练的 expert，字段缺失时默认为
`both`，以保持现有 FSDP2 配置的双 expert 交替训练语义。`high` 和 `low` 只构造、
加载并训练对应的一座 14B Transformer。Megatron engine 当前不支持 `both`；使用
Megatron 训练 A14B 时必须显式指定 `high` 或 `low`，否则会在初始化 distributed
runtime 和加载 GPU 权重前报错。

## 最小 CPU contract

先安装开发与 Wan 依赖：

```bash
pip install -e '.[dev,wan]'
```

运行现有 tiny-model contract，不需要下载发布权重：

```bash
python -m pytest -q \
  tests/tasks/wan/v22/a14b/test_a14b_contracts.py \
  tests/tasks/wan/v22/a14b/test_a14b_end_to_end.py \
  tests/tasks/wan/v22/ti2v_5b/test_ti2v_contracts.py \
  tests/tasks/wan/v22/ti2v_5b/teacher_forcing/test_tf_contracts.py
```

只验证 Hydra/runtime contract 时显式禁用持久 checkpoint，避免内部 smoke 产生无意义
的 final DCP：

```bash
python - <<'PY'
from vforge.tasks.common import compose_standalone_config
from vforge.tasks.translate import build_runtime

for name in (
    "wan22_a14b",
    "wan22_ti2v_5b",
    "wan22_ti2v_5b_tf",
    "wan22_ti2v_5b_dmd",
):
    cfg = compose_standalone_config(
        "train",
        overrides=[f"experiment={name}", "checkpoint.enabled=false"],
    )
    runtime = build_runtime(cfg, world_size=1)
    print(
        name,
        runtime.config.training.recipe,
        sorted(runtime.config.training.models),
        runtime.trainer.checkpoint.enabled,
    )
PY
```

该命令只证明配置合成和 runtime translation，不加载模型、不训练，也不产生 checkpoint。

## TI2V-5B DMD 架构

DMD 从 causal generator、real-score 和 fake-score 三个独立模型原生 checkpoint
启动，不运行 prepare ODE 或 ODE init。模型包含三个独立 FSDP execution root。训练由
通用 `PhasedTrainer` 和 `vforge/algorithms/dmd/` 驱动；offline 优先读取普通 CT cache，
online 则复用 Wan raw data 和 encoder。DMD 不增加 prepare/cache 格式，
`vforge/tasks/wan_taichu/dmd/inputs.py` 只做输入与噪声转换，其余 DMD 文件是两个
Adapter 与组装入口。一个 global
step 内的 generator→critic 顺序、增量 Q/K、EMA、strict resume 与 generator 导出见
[DMD 训练](../features/dmd-training.md)。

这里的 experiment 使用 `training.recipe=dmd` 和
`training.models.{generator,real_score,fake_score}` 逐个网络配置。DMD 不再注册成
`model.name`，也不再占用 `wan_taichu` 单模型 variant；因果 generator 的 attention
能力名为 `block_causal_teacher`。

默认配置 `wan22_ti2v_5b_dmd` 使用以下训练配置：

- 轨迹边界 `[1000, 938, 833, 625, 0]`，实际四次 transformer forward；
- 121 pixel frames 对应 31 latent frames，第一个 latent frame 是 clean anchor；
- generator 在 step 0、5、10、… 更新，critic 每 step 更新；
- generator/fake_score LR 分别为 `2e-6`/`4e-7`；
- 使用固定学习率。

## Eval 输入 contract

Wan 2.2 与 Wan 2.1 共用 `evaluation.data_source` + `evaluation.manifest`：

- `validation`：manifest 与训练相同，包含目标 `video`；
- `inference`：manifest 只包含 `image`、`prompt` 和已启用的 `normal`/`grid`
  等条件；无 temporal condition 时还要提供 `num_frames`；
- `cache`：manifest 直接指向外部 latent cache 根目录下兼容的 `manifest.json`。

`ref`/`reference` 可以是一个图片/视频路径，也可以是有序路径列表；每个视频路径取首帧，
所有 ref 分别作为单帧 VAE 样本编码。manifest 使用 `source` 或 `meta.source` 标记来源时，
`evaluation.sample_selection=evenly_spaced` 会先均衡来源配额，再在各来源内部等间隔选样。

两个字段必须同时设置。在线 eval 只编码推理条件，不生成训练目标
`frame_latents`。`data.val_manifest` 和 `data.cache_dir` 不属于 standalone eval 输入。

## 当前边界

- A14B 发布目录需要 `transformer` 与 `transformer_2` 两个 expert；
- Native transformer（含 TF 子类继承自 CT 的声明）的 Diffusers
  `_keep_in_fp32_modules`（`time_embedder` / `norm1` / `norm2` / `norm3`）把这些
  模块留在 fp32；`time_proj` 跟 Diffusers 一样在 `temb.type_as`
  之后走 mixed policy。block 入口不预转 `temb` / RoPE。AdaLN 仍用
  `temb.float()`。`scale_shift_table` 是单个权重而不是模块，落盘保持 fp32，
  前反向跟随 `param_dtype`；
- A14B condition stream 变化会改变输入宽度，cache 不能跨 stream 列表复用；
- TI2V/TF 的 stream 列表、首帧语义和 cache schema 必须一致；
- DMD 只支持 FSDP2；序列并行已打开，但 `usp_size` 必须整除注意力头数和一个去噪块的
  token 数，16 卡下两个模型的头数都不能整除，未验证；
- TF causal eval 复用训练用 transformer，因此沿用 doubled sequence 的 USP 整除和
  FA4 sparse metadata 约束；CPU tiny contract 可经 dense SDPA attention 执行；
- 单卡 GB300 已验证八变体 Gridnormal、TF full-recompute 与 incremental-KV 各一个样本；
  在线多-ref、来源分组等间隔选样及 `usp_size=4` 组合仍未执行模型专属 GPU gate；
- 本页不提供伪造的用户训练、恢复、评估或交付命令；达到公开闭环后再提升状态并补齐。

通用 schema 仍见[配置方式与字段](../reference/configuration.md)，但内部验证不等于
这些字段组合都已在 Wan 2.2 上验收。
