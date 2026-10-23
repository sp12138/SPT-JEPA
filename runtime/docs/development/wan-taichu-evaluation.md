# Wan Taichu 评估契约

本页说明 `wan_taichu` 的评估、推理、可视化和 causal KV rollout。
模型、训练与 cache 契约见[Wan Taichu 模型指南](../models/wan-taichu.md)。

## 不变量与所有权

- `vforge/models/wan/` 与 `vforge/models/wan_taichu/` 平行维护，互不 import、互不包装，
  也不以继承对方的 task/pipeline 来共享 Wan Taichu 语义；
- 可复用的 scheduler、eval runner、multi-variant writer、数据读取和分布式能力必须下沉
  到无模型偏好的公共模块，不能复制进两个模型目录；
- `vforge/models/wan_taichu/` 只拥有 variant forward、条件注入和 causal block/KV
  模型语义；`vforge/tasks/wan_taichu/` 拥有工作流装配、消融、输入校验和 artifact
  provenance；
- Wan Taichu 专属选择器先放在 model/task extras。若必须修改
  `EvaluationConfig`、`EvalRequest`、runner/component protocol、CLI 或公共 dataset
  schema，同一变更必须迁移并测试 Wan、Wan Taichu、MiniMax H3，不能让任一模型继续
  使用旧接口；
- `wan_taichu` 直接由 `model.extras.variant=wan22_ti2v_5b_ref` 声明 ref projection
  能力。启用
  `patch_embedding_ref` 后，`full`、`ref_normal_to_video`、`ref_grid_to_video`
  可以消费样本中的显式 ref，或由 validation 目标视频在线抽取 ref。禁止让 ref
  配置静默无效；
- 官方 Wan 和 WanTaichu 只共享中立 helper，不共享 task assembler。

## 当前可用公共入口

当前入口为 `scripts/evaluate_gridnormal.sh`、`scripts/infer.sh`、
`scripts/visualize.sh` 和 causal `scripts/eval.sh`。

`evaluate_gridnormal` 由同名脚本显式装配。ref 能力从
`model.extras.variant=wan22_ti2v_5b_ref` 推导；默认运行五个原生变体，
完整八变体需要通过 `evaluation.gridnormal.variants` 显式列出。A14B 的五个无-ref
变体是 `full`、`normal_to_video`、`normal_fg_to_video`、
`normal_bg_to_video` 和 `grid_to_video`。
`wan22_ti2v_5b_tf` 的 eval builder 不复用双向 CT sampler，可显式选择 causal
`full_recompute` 或 `incremental_kv`。

## 三个独立工作流

三个工作流使用稳定 ID：`evaluate_gridnormal`、`infer`、`visualize`。实现可以共享公共
runner，但入口语义、输入约束和产物不能合并成一个由隐式参数猜测行为的模式。

### `evaluate_gridnormal`

这是 checkpoint × sample × variant 的八种条件消融，不是单次普通 eval。每个 sample
的八个 variant 必须复用相同初始噪声，seed 为 `base_seed + sample_index`；因此输出差异
只来自条件块/ref 开关。

这里的 `grid` 是相机内外参渲染得到的 camera Hilbert-RGB 条件，主要表达相机轨迹和
空间射线；`normal` 表达目标视频的几何/表面朝向；`ref` 是额外的外观参考帧 token。
“关闭”某个 grid/normal 条件并不是直接删除 channel，而是用同一冻结 VAE 编码确定性的
fallback 像素，保持模型输入 shape 和 channel layout 不变：

- fallback grid：全像素 `0.0`，语义上关闭真实相机 grid；
- fallback normal：全像素 `0.5`，语义上提供中性 normal；
- foreground/background-only normal：只替换 normal body，保留既定 mask/channel layout；
- 无 ref：不向 transformer 注入 ref token。

八种 variant 的固定语义如下：

1. `full`：真实 grid + 完整 normal + ref。保留全部可用条件，作为完整条件基线。
2. `normal_to_video`：fallback grid + 完整 normal，无 ref。只观察完整 normal 对生成的
   约束能力，排除真实相机 grid 和参考帧。
3. `normal_fg_to_video`：fallback grid + 前景 normal，无 ref。只保留前景几何，用于观察
   主体形状/动作约束。
4. `normal_bg_to_video`：fallback grid + 背景 normal，无 ref。只保留背景几何，用于观察
   场景结构和背景运动约束。
5. `grid_to_video`：真实 grid + fallback normal，无 ref。只观察相机轨迹条件，不使用
   有效 normal 或参考帧。
6. `ref_normal_to_video`：fallback grid + 完整 normal + ref。观察 normal 与参考外观的
   联合作用，不使用真实相机 grid。
7. `ref_grid_to_video`：真实 grid + fallback normal + ref。观察相机轨迹与参考外观的
   联合作用，不使用有效 normal。
8. `ti2v`：fallback grid + fallback normal，无 ref。只保留 text 和 first frame，是
   不使用 grid/normal/ref 的 TI2V 基线。

因此常用对照关系是：`full` 对比 `normal_to_video`/`grid_to_video` 可以观察两类几何
条件的贡献；`normal_to_video` 对比前景/背景两个变体可以拆分 normal 的区域贡献；
带 ref 与对应无-ref 变体对比可以观察参考外观 token 的作用；`ti2v` 是统一的最低条件
基线。

`normal_fg_to_video` 和 `normal_bg_to_video` 替换 normal body，但保留 normal stream 的
既定 channel layout。三个 ref variant 要求
`model.extras.variant=wan22_ti2v_5b_ref`。样本显式提供 `ref`/`reference` 时显式输入
优先；否则
`model.extras.ref_num_frames_online=K` 会对 validation 目标视频先计算
`k_max=min(T_rgb // 80, K)`，再从 `[0, k_max]` 均匀采样本次 ref 数量，并在每个时间分段
内随机抽一帧。抽到零帧时 ref variant 按无 ref 执行。
每张 ref 独立作为单帧样本经过冻结 VAE，再沿 token 时间轴拼接，不能把多张 ref 当作一段
连续视频编码。`inference`/`cache` 没有目标像素时必须提供显式 ref 或关闭在线抽取。

旧 checkpoint 没有 `patch_embedding_ref` 时只 warm-start 缺失权重，checkpoint 自带
ref 权重始终优先。

条件 fallback 是确定性数据：

- grid/rgb 先生成全像素 `0.0` block；
- normal 先生成全像素 `0.5` block；
- 再通过与真实条件相同的冻结 VAE/预处理路径生成 latent；
- 带 mask 的 grid/normal fallback mask 固定为 `1`；
- 同一 shape/device/dtype 的结果可以缓存，但它是输入数据，不是参数、buffer、
  checkpoint 权重或可训练 embedding；
- 缺少生成 fallback 所需的冻结 encoder 时应在采样前报错，禁止以零 latent 代替。

共享输入 artifact 每个 sample 只写一次：`gt`、`normal`、`normal_fg`、`normal_bg`、
`grid`、`ref`。它们写入本次评估根目录的同名子目录，并以
`.gridnormal-inputs/<sample_id>.json` 作为原子完整性 marker；因此多个 checkpoint
不会重复导出输入。`evaluation.data_source=cache` 没有原始像素路径时，marker 会明确记录
对应 stream 为 unavailable，不会用 VAE decode 冒充原始数据。生成视频放在
`checkpoint-<step>/<variant>/<sample_id>.mp4`，并携带 JSON sidecar。sidecar 至少记录
sample ID/index、checkpoint、variant、seed、条件/ref 开关、resolved config hash 和
fallback pixel contract。每个 variant 还记录 grid/normal 的实际选择、对应输入
源路径、ref selection 以及 fallback pixel value，不能只靠目录名推断条件。`skip_existing`
只有在 MP4 与 sidecar 都完整且 provenance 匹配时才可跳过。

训练 preset 中的 `grid_random_rot_deg` 是数据增强，不进入 standalone eval。Eval 使用
inline `c2ws/Ks` 渲染 grid 时强制设为 `0`，保证相同输入、seed 和 provenance 可重放。

`evaluation.sample_selection=evenly_spaced` 在单个 source 或扁平 manifest 没有来源
信息时做全局等间隔选择；配置多个 `evaluation.sources`，或 manifest 存在 `source` /
`meta.source` 时，按照来源顺序前置分配全局 `num_samples` 配额，再在各来源内部
等间隔选择，最后才进行 DP 分片。

显式 `ref`/`reference` 接受一个路径或有序路径列表；图片直接作为一帧，每个视频路径
读取首帧。一个 batch 内样本必须具有相同 ref 数量和空间尺寸，编码器将每一帧作为独立
单帧 VAE 样本编码。多个显式 ref 的共享产物写为
`ref/<sample_id>-000.png`、`ref/<sample_id>-001.png` 等。

### 权重与随机性

Wan Taichu eval 使用 PT/auto loader，并为无前缀 transformer 权重补 `transformer.` 前缀。
Manifest 可给样本提供 `eval_seed_offset`；gridnormal seed 与在线 ref generator 使用
`evaluation.seed + eval_seed_offset`。未提供时回退到扁平 manifest 的全局 index。
在线 ref 抽样后的 generator state 继续用于初始 latent noise；八个 variant 从相同
seed/state 开始。

实际抽中的在线 ref 帧按选择顺序写入共享 `ref/<sample_id>-<slot>.png`，
`.gridnormal-inputs/<sample_id>.json` 同时记录原始帧 index。写入直接读取目标视频像素。

Manifest 显式 `ref`/`reference` 优先；没有显式 ref 时才从目标视频在线抽取。
`evaluation.online_ref.num_frames` 和 `evaluation.online_ref.frame_indices` 分别用于
指定抽帧数量和帧号，两者互斥。未设置时，`model.extras.ref_num_frames_online` 表示随机
抽样上限，`0` 表示关闭在线抽取。训练从 latent 时间轴抽 ref；eval 从目标视频像素抽帧，
再逐帧 VAE 编码。

### `infer`

`infer` 是单样本推理 smoke，不会评估目录内全部 checkpoint，也不自动运行八种消融：

- 必须明确一个非负 `sample_index`；
- `num_samples` 只能未设置或为 `1`；
- checkpoint 必须明确解析为 published weights、指定 checkpoint 或“最新 checkpoint”
  之一，sidecar 记录最终解析结果，禁止静默换权重；
- 使用所选 variant 的正常 condition/ref 语义，只生成一个 MP4 和一个 JSON sidecar；
- 固定 seed 时，同一输入、权重和配置可重放；该 smoke 验证装配和输出，不作为感知质量
  gate。

### `visualize`

`visualize` 是不加载模型、不加载 checkpoint、不运行 tokenizer/VAE/denoiser 的原始数据
检查：

- 直接读取 `train`、`val` 或 `test` split，并支持前 N 个样本或一个 sample index；
- 自动导出样本中 video-like `[T,C,H,W]` 流和 image-like `[1,C,H,W]` 流，同时保存
  prompt 与 sample metadata；
- 至少覆盖 target video、grid、normal、normal foreground/background、first frame/ref
  中样本实际提供的字段；缺失可选流应记录为 absent，不能生成 fallback 冒充原始数据；
- 输出只证明 dataset decode、字段、shape/range 和时间对齐可观察，不证明模型可推理。

由于该工作流明确“无模型”，实现不得为了复用 eval builder 而实例化空模型或读取模型
配置中的 pretrained path。

## Causal eval 与 KV rollout

本节正式从基础契约接管 `kv_cache_rollout`。WanTaichu 路径只适用于声明 causal
inference 能力的 variant；`wan22_ti2v_5b_causal_teacher` 的训练入口仍按基础契约约束。
官方 `wan22_ti2v_5b_tf` 继续提供独立的 teacher-forcing causal eval。

### Generated history

Causal eval 必须按 block 自回归生成，历史来自前面已经生成并提交的 clean blocks，而
不是 ground-truth future、teacher-forcing clean half 或预先编码的完整 target：

1. 首帧 latent 是 block 0，以 timestep `0` 提交为初始 clean history；
2. 当前 block 从固定初始噪声开始完成全部 scheduler steps；
3. 当前 block 完成后才提交为 clean history；
4. 后续 block 只能 attend 已提交历史和自身，不能看到 future blocks；
5. condition stream 可以覆盖完整时间范围，但不能借 condition 通道注入 target latent。

有 target 的 validation 数据只用于生成后的比较/共享 artifact，不得进入 causal history。
可执行测试和数值 parity 必须使用会区分 generated history 与 ground-truth history 的
非平凡 fixture。

### 主路径与 golden

`incremental_kv` 是生产主路径：

- 模型每次只处理当前 video block；
- 每层缓存已经提交的 clean K/V；当前 block 去噪步骤可覆盖自己的临时 slot；
- block 去噪完成后，以 timestep `0` 做 context re-pass，将 slot 覆盖为 clean K/V；
- RoPE 使用完整 clip 的绝对 frame 位置后按 block 切片；
- CFG 的 cond/uncond 各自维护独立 cache，禁止跨分支共享；
- cache 生命周期限定为一次 sample rollout，sample、shape 或 checkpoint 改变时必须重建。

`full_recompute` 是 golden/fallback，也是 Ulysses 的首版实现：

- 每个 denoise step 重算完整 doubled `[clean | noisy]` sequence；
- clean half 只包含首帧和已提交的 generated clean history；当前 noisy block 写入 noisy
  half，future 使用不可见占位；
- per-token timestep 的 clean half 为 `0`，noisy half 只有当前 block 取 scheduler
  timestep；
- mask、RoPE tile、输出截取与 teacher-forcing forward 完全一致；
- 首版 Ulysses 只要求 `full_recompute`；`incremental_kv + Ulysses` 在单独 parity 和
  cache-sharding 契约建立前必须显式拒绝，不能静默回退。

两条路径使用相同 initial noise、scheduler、block layout、条件与 CFG。小模型 gate 按
block/step 比较预测、scheduler 更新和最终 latent；CUDA BF16 gate 另行记录容差。若
incremental 路径越界、cache 不完整或 parity 不通过，可显式选择 full recompute，但必须
在 provenance 中记录实际模式。

## 可执行测试分层与门禁

工作流行为由贴近实现边界的可执行测试分层覆盖，不再维护一份与代码重复同步的
JSON/Python 快照：

- config 测试覆盖配置解析、默认值和非法组合；
- routing 测试覆盖工作流入口、模型 variant 与 causal backend 的路由；
- variants 测试覆盖八种消融的 stream/ref 映射和 deterministic pixel fallback；
- inference 测试覆盖 seed、在线/显式 ref、条件装配与生成调用；
- artifacts 测试覆盖共享输入、sidecar provenance、完整性 marker 和 `skip_existing`；
- causal 测试覆盖 generated history、incremental KV、full recompute、Ulysses 限制及
  KV/full 数值 parity。

这些测试直接验证运行时行为和产物边界，避免同一事实同时存在于实现、机器可读快照和
快照断言中而产生重复维护。

当前已通过 published Wan 5B、单卡 GB300、CUDA BF16 的八变体和两种 causal backend
KAP smoke，并以两张 GB300 验证显式 ref 与在线 K=0 混合的 FSDP eval。Training
checkpoint、Ulysses 多卡、TP/CP output leader、长视频 cache 内存和 MP4 视觉质量仍
分别标记为未验证。

## 不包含的能力

- 修改训练 loss、cache schema、checkpoint 布局或基础 variant 集合；
- ODE-init、DMD/self-forcing generator 训练；
- `incremental_kv + Ulysses` 的 cache shard/all-gather 设计；
- 服务化 API、Web UI、批量推理调度和质量指标平台；
