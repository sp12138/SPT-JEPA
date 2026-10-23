# Wan Taichu Eval 适配状态

行为契约以
[Wan Taichu 模型指南](../../models/wan-taichu.md)和
[Wan Taichu 评估契约](../wan-taichu-evaluation.md)为准。

## 当前结论

WanTaichu eval 代码迁移和入口切换已经完成：

- `model.name=wan_taichu` 已注册 train 与 eval；
- `scripts/evaluate_gridnormal.sh` 要求通过 CLI 显式选择 `experiment`；
- 普通 bidirectional infer、八变体 GridNormal、显式/在线 ref、
  `full_recompute` causal rollout 和 `incremental_kv` 均走统一
  `EvalComponents` contract；
- 官方 Wan 的双向 GridNormal/ref-token 过渡原型已经移除；
- 官方 `wan22_ti2v_5b_tf` dedicated teacher-forcing causal eval 继续保留；
- 官方 Wan 与 WanTaichu 不共享 task assembler，只共享中立的 evaluation、prepared、
  flow-match 和 causal KV helper。

因此本文记录的是已完成实现，不再把脚本默认值、官方 GridNormal 原型或
`current_implementation_target` 切换列为未来阶段。

## 已完成的入口与职责

WanTaichu registry 只公开：

```python
vforge.tasks.wan_taichu.eval:build_eval_components
```

私有 workflow 路由位于 `vforge.tasks.wan_taichu.eval_task`：

```text
vforge/tasks/wan_taichu/
├── eval.py
├── model.py
├── train.py
└── eval_task/
    ├── __init__.py
    ├── base.py
    ├── reference.py
    ├── checkpoint.py
    ├── bidirectional.py
    ├── causal.py
    └── gridnormal/
        ├── __init__.py
        ├── variants.py
        └── artifacts.py
```

早期 eval 私有 `shared.py` 已按所有权拆为：

- `base.py`：WanTaichu model、codec、condition encoder、dataloader、inference 和
  published/checkpoint source 装配；
- `reference.py`：显式/在线 reference 选择、RNG state 和 provenance；
- `checkpoint.py`：只用于训练 checkpoint 的兼容回调。

早期独立 `weights.py` 已删除。published 权重按 MiniMax H3 已采用的模式，直接复用
train 拥有的 published weight source；eval 不复制 condition widening、ref warm-start
或 after-load 初始化规则。训练 checkpoint 则使用 `checkpoint.py` 的缺失 ref
projection 兼容回调，checkpoint 自带参数始终优先。

## 模型和 task 保持独立

`vforge.models.wan_taichu` 不 import、继承或包装 `vforge.models.wan`。WanTaichu 的
condition channel、per-token timestep、ref projection、teacher-forcing attention 和
incremental KV forward 都留在自己的模型目录。

同样，`vforge.tasks.wan_taichu` 不 import 官方 Wan task assembler。两套 task 可以复用：

- `vforge.evaluation` 的 runner、sampler、multi-variant 和 output contract；
- `vforge.tasks.prepared_eval` 的 prepared batch helper；
- `vforge.models.video_diffusion.flow_match` 的可选 scheduler/CFG loop；
- `vforge.models.video_diffusion.wan` 的 Wan asset/latent helper；
- `vforge.models.causal` 的 KV cache、attention context 和 neutral executor。

它们通过行为 contract 保持一致，不建立跨 task 依赖。

## 已完成的 workflow

### Bidirectional 与 GridNormal

`model.extras.variant` 决定模型能力：

- `wan22_ti2v_5b`：普通 bidirectional TI2V；
- `wan22_ti2v_5b_ref`：增加 ref projection，可运行含 ref 的 GridNormal variants。

`evaluation.gridnormal.enabled` 只决定是否展开 GridNormal workflow；
`evaluation.gridnormal.variants` 显式缩小运行集合。模型能力、评估矩阵和 sampling
backend 不再由隐式进程开关混合表达。

八种 variant、fallback condition 和纯 encoded-bundle 转换位于
`eval_task/gridnormal/variants.py`；multi-variant 执行复用公共
`MultiVariantInference`，MP4、sidecar、共享输入和 completion marker 由
`eval_task/gridnormal/artifacts.py` 拥有。

reference 在 VAE 编码前选择。manifest 的显式 `ref`/`reference` 优先；否则 validation
样本可按配置从目标 RGB 视频抽帧，每帧独立 VAE 编码。实际使用的 ref、frame index 和
选择 provenance 会随共享输入 artifact 记录。

### Causal

WanTaichu causal variant 使用自己的 generated-history block rollout：

- 5B 使用独占的 clean block 0；A14B 的 `first_frame_own_block=false`，从 block 0
  开始去噪，首帧条件来自 RGB condition；
- `full_recompute` 是 doubled teacher-forcing golden/fallback；
- A14B 按 timestep `900` 切换 high/low expert；`incremental_kv` 切换 expert 时
  清空 cache，并用新 expert、timestep `0` replay 已提交的 clean blocks；
- `local_attn_size=30`、`sink_size=9` 时，KV 仅保留固定 sink 和最近历史；
- `incremental_kv` 的 cond/uncond cache 相互独立，并支持 Ulysses head-sharded KV；
- 每个 block 完成去噪后才以 timestep `0` 提交 clean history；
- 所有 transformer 调用仍经过 evaluation engine。

官方 Wan 的 teacher-forcing causal eval 是独立保留的 dedicated 实现，不依赖
WanTaichu task assembler，也不随官方 GridNormal/ref-token 原型一起删除。

### Wan Taichu indexed latent cache

A14B 99-lat train-set causal eval 直接复用 Wan Taichu 训练 cache reader，读取
`train/<shard>/<index>.pt` 中的 `frame_latents`、prompt embedding 和
RGB/grid/normal condition latents，并按 source 配置截到 99 latent。该路径不重新运行
VAE、文本编码或 grid 渲染，因此输入与训练缓存一致。

它只属于 `model.name=wan_taichu`，不会改变通用
`evaluation.data_source=cache + evaluation.manifest=<cache>/manifest.json`：

- `.pt` 是实际 encoded tensor bundle；
- `manifest.json` 是 vforge-native hashed-bucket cache 的索引和 schema 元数据，不包含
  latent tensor；
- 两者不能仅通过改名互换。完整迁移需要重排 `.pt` 路径并生成或消费 manifest；
- causal eval 无需迁移，直接配置一个无 `kind` 的 `evaluation.sources[].cache_dir`。

## 配置边界

配置按职责分开：

- `model.extras.variant`：模型结构和 ref/causal 能力；
- `evaluation.gridnormal.enabled`：是否运行 GridNormal；
- `evaluation.gridnormal.variants`：实际 variant 集合；
- `evaluation.rollout_mode`：`full_recompute` 或 `incremental_kv`；
- `evaluation.online_ref.num_frames` / `frame_indices`：在线 ref 选择；
- `latent_cache.mode`：`online` 或 `offline` 输入路径。

用户命令见[Taichu Wan 命令替换](wan-normal-command-equivalents.md)。

## 验证状态

已建立的可执行 contract 覆盖：

- registry、workflow 路由和非法组合；
- 八种 variant 的 stream/ref 映射与 deterministic fallback；
- 固定 seed、显式/在线 ref 和 shared artifact provenance；
- published 权重与训练 checkpoint 的 ref 初始化边界；
- generated history、full recompute、incremental KV 和 CFG cache 隔离；
- output ownership 与 `skip_existing`。

现有记录还包括 published Wan 5B 的单卡 BF16 GridNormal/causal smoke，以及双卡 FSDP
显式 ref 与在线 K=0 混合样本验证。A14B 99-lat 的真实 checkpoint bitwise、40-step
视频效果和 incremental-KV Ulysses GPU smoke 仍需单独执行。

## 保持非目标

- 修改训练 loss、cache schema 或 checkpoint 布局；
- ODE trajectory、DMD 或 self-forcing 训练；
- 服务化 API、质量指标平台或批量调度系统。
