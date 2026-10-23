# 架构与所有权

本文是训练入口、所有权边界和模型生命周期的**参考页**，用于判断一段代码该由谁拥有、
在生命周期的哪个阶段执行；它不是新增模型的接入步骤。配置字段见
[配置方式与字段](../reference/configuration.md)，checkpoint 恢复见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)，配置的加载和覆盖顺序见
[配置加载与覆盖](configuration-composition.md)。

## 按任务查阅

- 新增或迁移模型：直接进入[新增 Native 模型](adding-native-model.md)；
- 修改并行、TP 或 engine 生命周期：进入[训练引擎](training-engines.md)；
- 修改保存、恢复、manifest 或远端 staging：进入
  [Checkpoint 与 S3 内部机制](checkpoint-internals.md)；
- 不确定代码归属或执行顺序：继续阅读本页。

## Native 入口与运行时

Native Hydra 是用户入口；`TrainComponents` 只出现在其内部装配链路：

```text
Hydra entry (vforge.tasks.train.main / run_train)
  -> registry lookup by model.name
  -> vforge.tasks.wan.v21.train.build_train_components(request)
  -> TrainComponents(model=ModelSpec | nn.Module)
  -> Trainer.fit(...)
  -> session_from_components(...)
  -> TrainingSession
  -> Trainer._fit_session(...)
      -> create_engine(...)
      -> session.setup(...) / resolve ModelSpec
      -> engine.prepare_model(...)
```

`TrainingSession` 当前仍负责把 Native 组件标准化为 Trainer 消费的 callbacks。
`Trainer.fit_session()` 保留为标准化输入接口。
模型和新增训练能力统一进入 Native，Trainer 不 import 业务项目。

这里的三个核心契约分别负责：

- **`ModelSpec`**：延迟构造真实模型，让 Trainer 先拒绝非法引擎/topology；
- **`TrainComponents`**：Native task 的声明式装配，包含 model/spec、数据、step 和
  optimizer/scheduler factory；
- **`TrainingSession`**：Native 组件标准化后的生命周期输入，供 Trainer 消费。

## 原生路径

Hydra 的 `vforge.tasks.train` 负责 compose/校验配置、初始化共享 runtime、解析 resume path 和 reporters。入口按 `model.name` 从 `vforge.tasks.registry` 取到 builder 路径并惰性导入；Wan21 对应 `vforge.tasks.wan.v21.train.build_train_components()`。

Wan21 装配代码拥有业务构建：创建延迟模型构造所需的 `ModelSpec`，并装配 dataloader、
optimizer/scheduler factory、step function、预训练权重来源和 state-dict mapper，最后
返回 `TrainComponents`。它不执行并行化、materialize、权重加载或训练循环。

`Trainer.fit()` 调用 `session_from_components()`。该函数用 holder 保存
`ModelSpec | nn.Module`，并把 factories、step function、reporters、resume 元数据包装为
`TrainingSession` callbacks。Trainer 创建引擎后，`session.setup()` 才通过
`resolve_model()` 构造真实模型；此后再进入并行化和 materialize。

通过 Python API 启用参数分桶诊断时，调用方应显式设置
`TrainerConfig.diagnostic_parameter_bucket` 分类函数；默认值为 `None`。
模型上的私有属性不再用于传递该函数。

## 所有权边界

- **Hydra task entry**：配置、runtime 初始化/销毁、日志输出和调用编排。
- **Task assembly (`vforge.tasks.wan.v21`)**：模型与数据的业务语义、单步 loss、optimizer factories、源权重格式。按动词分成 `train` / `eval` / `encode` 三个模块，加一个 `shared` 放多个动词共用的模型构造、条件编码器和 collate。同一 family 的多个版本共用一层：版本无关的部分放 `vforge.tasks.wan` 的 `shared` / `encode` / `eval`，版本只提供一个 spec 说明自己的 codec、模型和采样默认值。
- **Model definition (`vforge.models.wan.v21`)**：只有 `nn.Module`、transformer、encoder 和推理；不得 import 训练层。
- **`TrainComponents`**：原生路径的声明式输入；包含 `ModelSpec` 或已有 model，不拥有执行循环。
  optimizer/scheduler 通常由 factory 提供；需要 engine-neutral 描述时可同时提供
  `OptimizerSpec` / `SchedulerSpec`。
- **`TrainingSession`**：Trainer 消费的标准 callback contract；不实现模型逻辑。
- **`Trainer`**：设备与并行策略、模型生命周期编排、权重/resume、optimizer 创建、forward/backward/step、持久 checkpoint 和最终清理。
- **`TrainingEngine`**：模型并行的施加方式、后端专属 materialize 收尾和整个 optimizer step；每进程选一次。fsdp2 与 megatron 之间什么共用、什么必须分叉，见[训练引擎](training-engines.md)。
- **Distributed/checkpoint/ops 包**：提供机制；不构建模型或业务 application。

依赖方向保持为 model task -> training contracts -> Trainer。core training 不得反向依赖业务项目。

## Core 目录与放置规则

这里的 core 指不依赖具体业务项目、可被不同 Native 模型复用的框架代码。新增文件前先按它负责的生命周期选择目录：

```text
vforge/
├── training/              # 训练状态机、recipe contract、上报与确定性
│   ├── trainer.py         # 唯一训练循环
│   ├── recipe.py          # TrainComponents / TrainingSession
│   ├── reporting.py       # 通用指标与 reporter
│   └── checkpoint/        # 跨运行的 PT/DCP 保存与恢复
├── distributed/           # DeviceMesh、FSDP2、Ulysses、CP backend、parallelize
├── checkpointing/         # 单次 forward/backward 内的激活重算与 offload
├── ops/                   # 公共算子接口和 backend dispatch
│   └── _attention/        # FA/SDPA 等 attention backend 内部实现
├── data/                  # 通用 dataset、cache、resumable/leader-broadcast loader
├── evaluation/            # 与模型无关的分布式评估 runner
├── models/                # 通用权重加载与原生模型定义
│   ├── video_diffusion/   # 单 latent flow-match 模型可选的中立推理组件
│   └── wan/               # Wan family 公共件；版本实现在 v21/ 与 v22/，不 import 训练层
├── offline_encoder/       # encode 动词的执行引擎与 EncodeComponents
├── tasks/                 # Hydra 薄入口、模型注册表、schema→运行时翻译
│   └── wan/               # Wan 装配：版本无关的 shared / encode / eval，版本差异在 v21/ 与 v22/
├── utils/                 # 无生命周期所有权的纯工具
└── hydra_config.py        # Native Hydra structured schema、解析与序列化
```

按改动性质选择位置：

- 模型结构、条件编码或推理语义：`models/<family>/<version>/`，family 共用的基类和契约放 `models/<family>/`。
- 至少两个 family 共享且满足单 latent、单 scheduler contract 的 flow-match 机制可放
  `models/video_diffusion/`；它只能提供可选 helper，不能扩大
  `EvalComponents` 或要求 H3 等模型采用 Wan sample schema。具体边界见
  [视频推理扩展边界](video-inference-extension.md)。
- 某个模型的 train/eval/encode 装配、step 语义、optimizer factory：`tasks/<family>/<version>/`；`tasks/` 与 `models/` 的目录层级一一对应，知道模型在哪就知道装配在哪。
- Hydra 字段本身：`hydra_config.py`；字段到运行时对象的映射：`tasks/translate.py`。schema 不得 import 任何 vforge 模块（有测试守住）：新配置块若已有运行时类型，就在 schema 里写一个只有字段和默认值的 `*HydraConfig` 孪生体，校验交给 `translate` 里构造的运行时对象，不要两边各写一遍不变量。
- Trainer step 顺序、recipe contract、通用上报：`training/`。
- 持久训练状态、manifest、PT/DCP I/O：`training/checkpoint/`。
- FSDP、DeviceMesh、序列并行或 meta materialize：`distributed/`。
- 激活重算、saved-tensor preservation、CPU offload：`checkpointing/`。
- attention 或其它算子 backend：内部实现放 `ops/_attention/`，只从 `ops/` 暴露稳定接口。
- 通用 loader、cache schema 或数据恢复能力：`data/`；模型专属字段转换仍留在 `models/<name>/`。
- 新 CLI 只有在所有模型共享同一生命周期时才放 `tasks/`；模型差异通过 `tasks/registry.py` 注册，不新增模型专属 Trainer 或 runtime。

仍无法判断时，先看能力是否至少被两个模型或项目共享、语义是否稳定。否则先留在模型插件；不要用 `utils/` 规避所有权判断。新增模型和框架能力的具体步骤见[新增 Native 模型](adding-native-model.md)和[新增框架能力](adding-framework-feature.md)。

## 并行配置与运行时状态

并行层区分“描述什么 topology”“如何改造模型”和“当前进程实际属于哪些通信域”：

```text
ParallelHydraConfig
  → tasks.translate.build_parallel_config()
  → ParallelConfig
  → ParallelizeConfig(parallel=..., fsdp2=..., mcore_ddp=...)
  → parallelize_model()
  → init_parallel_state()
  → ParallelState
  → RuntimeContext.parallel_state
```

- **`ParallelConfig`** 是运行时 topology 值对象，负责维度大小、默认值归一化和组合
  校验，不创建模型 wrapper；
- **`ParallelizeConfig`** 是模型改造 policy，组合 topology、FSDP2/MCore policy 和
  activation checkpoint；它只在执行并行化时消费，不保存 rank-local 状态；
- **`ParallelState`** 是唯一的进程级并行状态，持有归一化后的 `ParallelConfig`、
  完整 `DeviceMesh`、派生子 mesh 和当前 rank 的 process group；
- **`RuntimeContext`** 统一持有 device、`ParallelState` 和 ops policy。它不会创建
  第二份并行状态，框架也不存在 `ParallelizeState`。

完整 `DeviceMesh` 可以包含 `(pp, dp_replicate, dp_shard, usp, cp, tp)` 多个轴。不同
机制只消费自己的视图：USP/TP/CP 通过 `get_group("<axis>")` 取得单轴通信组，组合域
通过 `_flatten()` 派生；FSDP2 只接收 `fsdp_mesh`。普通 FSDP 的视图是一维
`dp_shard_usp`，HSDP 的视图是二维
`(dp_replicate, dp_shard_usp)`，其中维度顺序而不是名字决定 replicate/shard 语义。

`ParallelState` 使用进程级 singleton，是因为一个 torchrun worker 在一次 task 中只
运行一个 topology。初始化相同配置是幂等的，冲突配置会失败，task cleanup 调用
`destroy_parallel_state()`。只有未来需要同一进程并存多个独立 topology 时，才应取消
singleton；不要通过新增另一个 state 类型绕过当前生命周期。

代码真源分别是
[`parallel_state.py`](https://git.woa.com/ieg-infra/wmtrain/blob/master/vforge/distributed/parallel_state.py)、
[`parallelize.py`](https://git.woa.com/ieg-infra/wmtrain/blob/master/vforge/distributed/parallelize.py)
和
[`runtime.py`](https://git.woa.com/ieg-infra/wmtrain/blob/master/vforge/distributed/runtime.py)。

## 模型生命周期概览

Trainer 对同一个模型按固定所有权顺序执行：

```text
创建 engine/runtime → session.setup() 解析模型 → engine.prepare_model()
  → 安装可选 offload wrapper → materialize
  → 按需加载 pretrained baseline → engine.prepare_data_parallel()
  → session.set_model() 回写 live model
  → 创建 optimizer/scheduler/dataloader 与 execution contract
  → 加载 DCP full resume 或 PT warm-start → 训练与保存 → cleanup
```

pretrained baseline 在 data-parallel wrapper 之前加载；DCP full resume 必须等 optimizer
和 dataloader 创建后再恢复其状态。该顺序保证引擎先于模型存在，optimizer 始终绑定
最终 live model，task/adapter 通过 `set_model()` 接受替换后的模型。具体实现分别由以下
页面维护：

- 引擎选择、并行化和 materialize 顺序：[训练引擎](training-engines.md)；
- non-persistent buffer 重建：[Meta runtime buffer 合约](meta-runtime-buffers.md)；
- 预训练权重、PT/DCP 保存恢复和 S3 staging：
  [Checkpoint 与 S3 内部机制](checkpoint-internals.md)；
- activation checkpoint 与 offload 的用户行为：
  [激活重算与 Offload](../features/activation-recompute-and-offload.md)。

## 下一步

- 新增或迁移模型：先读[配置加载与覆盖](configuration-composition.md)和
  [Meta runtime buffer 合约](meta-runtime-buffers.md)，再进入
  [新增 Native 模型](adding-native-model.md)；
- 修改并行或 engine 生命周期：进入[训练引擎](training-engines.md)；
- 修改保存、恢复或远端 staging：进入
  [Checkpoint 与 S3 内部机制](checkpoint-internals.md)；
- 开始验证：进入[本地开发与验证](testing.md)。

## DMD 算法与 LTX 接入

DMD 使用既有 `Algorithm` → `PhasedTrainComponents` → `PhasedTrainer` 链路。
标准与 blockwise 配置共用装配、G/C step 和 rollout 遍历；模型适配器负责将通用
latent 操作映射到具体模型。代码按以下边界放置：

- `algorithms/dmd/algorithm.py`：三塔装配、optimizer phase、EMA，以及唯一的
  generator/critic 主流程。公开的 `DistributionMatchingSteps` 提供 rollout、score
  gradient、generator/critic loss 和 backward replay 扩展方法；标准 DMD 是默认实现。
- `algorithms/dmd/runtime.py`：共用的双数据流、RNG、加噪、块遍历和 KV 提交。
  `DistributionMatchingRuntime` 提供 scheduler、退出步和 timestep 采样的扩展方法。
  历史是否加噪独立声明：标准路径即使 sigma=0 也保留随机数消耗，干净历史路径不抽噪声。
- `algorithms/dmd/blockwise.py`：一个算法扩展文件，保留随机退出步、干净历史、
  逐块等权 loss、各模态 CFG 和 replay。配置类型归公共 `config.py`；score 调用和
  校验复用公共 objectives，BF16 舍入及 block 算术由扩展方法保留。
  不重写 G/C step 或 rollout 遍历，配置名称继续使用 `block_distribution_matching`。
- `algorithms/dmd/segment_chain.py`：继承 blockwise，数据流持有当前样本和段间状态。
  G/C 使用同一旧前缀，critic 更新后的既有回调提交一次状态；有 G 更新时采用 G 的
  rollout。评分窗口通过 adapter 的默认恒等方法扩展，原 DMD 行为保持不变。
- `tasks/ltx/dmd/`：LTX 的三塔构造、数据准备、sigma 映射、generator/critic 适配器，
  以及确保各 rank 执行相同块数的 sampler。Long Tuning 的前缀写入、评分窗口和
  生成区裁切仍由这些 adapter 处理；不拥有训练循环或 optimizer 更新。
- `models/ltx/video_vae/`：段间视频转换所需的卷积 VAE 编码、解码和权重加载。
  后续段的图像条件必须由末尾 block 解码后的最后一个像素帧重新编码，不能直接
  截取视频 latent。VAE 冻结，USP 组源 rank 执行转换，再复用公共广播传递结果。
- `models/ltx/kv_cache.py` 与 `cached_attention.py`：LTX 的历史 KV 存储和带缓存的
  attention 语义。序列并行通信和 attention kernel 继续调用公共 distributed/ops，
  task 注入的 attention kernel 不再次执行 Ulysses。

`StepOutput.backward_losses` 是已有 step contract 的可选扩展：算法只产生惰性 loss，
engine 负责逐块 backward，并在全部块完成后执行一次裁剪和更新。
公共梯度裁剪默认仍为 logical；LTX DMD 通过
`PhasedTrainComponents.gradient_clipper_factory` 装配 PyTorch 裁剪。
工厂接收 `runtime.gradient_norm_dtype`，返回现有 `GradientClipper` 回调；
`PhasedTrainer` 转接到共享 Trainer/engine 链路，engine 不选择模型专属实现。

LTX DMD 的 generator/critic adapter 负责模型调用前后的模态切分和输出还原；
字段切分与 gather/crop 在现有 `tasks/ltx/dmd/inputs.py` 共用，padding、slice 和 gather
复用 distributed 的公共 USP。generator 的 prefill 不收集输出，模型层继续负责
attention 通信、KV cache 和 padding 屏蔽；SFT/TF 的训练路径保持独立。
`slice_sequence(gather_grad=False)` 保持原本的本地切片梯度，不引入反向通信；
公共默认仍收集梯度。`models/ltx/ops.py` 拥有模型的精度边界：Linear 输入匹配 FSDP 参数 dtype，
norm 和后续调制保持 FP32，Q/K 在 attention 调用边界转为 V 的 dtype，
RoPE 和存储的 KV 不提前降精度；SFT/TF 的既有通信入口保持不变。
独立 DMD connector 保留原来的 BF16 norm；公共 ops 不依赖模型类型。
配置字段和限制见[配置方式与字段](../reference/configuration.md)。

纯 loss 与 replay 等价性测试放在 `tests/algorithms/dmd/`；engine 执行顺序测试放在
`tests/training/`；双数据流恢复测试放在 `tests/data/`。LTX 的真实权重和原始缓存
端到端回归继续由 `tests/models/ltx/test_dmd_training.py` 负责。

### DMD 推理

`ltx23_dmd_eval` 使用既有 eval runner、generator 子模块权重加载和公共 DMD rollout。
模型和 published 权重构建由 `tasks/ltx/dmd/shared.py` 共同提供，直接接收模型参数；
eval 不构造训练请求，也不依赖训练集成层。生成函数归既有 generator 模块。任务层只适配缓存条件、推理日程和输出；`models/ltx/decoding.py` 管理解码组件的加载与
驻留，固定使用模型库源码，不依赖参考 Trainer。KV 滑窗与固定 sink 属于模型 cache，
默认关闭。公共 rollout 的逐 block 噪声回调默认关闭，训练保留已有随机数顺序。

标准 DMD 推理通过现有 `TransformerOpsConfig` 保留参考入口的 BF16 norm 输出与调制，
包括 Q/K norm、block norm 和输出 LayerNorm；训练继续使用默认的 FP32 norm 策略。
两种路径共用模型和权重格式，精度策略由任务装配，不新增用户配置开关。

多条 prompt 共用一次 Gemma 加载，逐条编码后将条件保存在 CPU，并在构建 generator
之前释放编码器。解码产物在文件写入完成后原子替换。
