# 概念与术语

本页是按需查询的术语表，不是运行任务前必须顺序阅读的教程。通用组合见
[Override 速查](configuration-cheatsheet.md)，字段见
[配置方式与字段](configuration.md)。

## 入口和配置

**Native Hydra**
: 由 `vforge-{prepare,train,eval,...}` 或 `python -m vforge.tasks.<verb>` 启动的原生
  路径。普通单进程任务优先使用公开的 `vforge-*` CLI；只有并行维度要求多个进程时
  才使用 `torchrun -m vforge.tasks.<verb>`，不要把模块入口机械替换为所有任务的默认
  写法。新模型和新能力统一走 Native。

**Wan Taichu Native**
: 在 vforge 内实现的 Wan Taichu 模型与任务，使用 Native 配置和训练入口。模型契约见
  [Wan Taichu 模型指南](../models/wan-taichu.md)。

**Experiment**
: 一套模型 recipe，位于 `vforge/conf/experiment/`。它覆盖 schema 默认，但仍可被 task
  和 CLI 覆盖。

**Resolved config**
: Schema、experiment、task 和 CLI 合成后的最终配置，保存在
  `resolved_config.yaml`。

## 并行

**ParallelConfig**
: 运行时拓扑配置，描述 DP/HSDP/USP/CP/TP/PP 等维度的大小和组合约束；它不持有
  process group。

**ParallelizeConfig**
: 模型并行化策略，组合 `ParallelConfig`、FSDP2/MCore policy 和 activation
  checkpoint 开关；它是 `parallelize_model()` 的输入，不是运行时状态。

**ParallelState**
: 根据 `ParallelConfig` 初始化后的进程级运行时视图，持有完整 `DeviceMesh`，并提供
  `dp_group`、`usp_group`、`tp_group`、`fsdp_mesh` 等当前 rank 所属的通信域。

**RuntimeContext**
: task 初始化后的共享上下文，持有 device、`ParallelState` 和算子 backend policy。
  `RuntimeContext.parallel_state` 引用的是同一个进程级 `ParallelState`，框架不存在
  独立的 `ParallelizeState`。

**DP**
: Data Parallel。不同 replica 处理不同样本并归约梯度。

**Ulysses**
: 将一个长序列切到多个 rank，并在 attention 前后执行 all-to-all 的算法。

**USP**
: VForge 对 Ulysses 并行维度的命名，对应 `usp_size`、`usp_group`。它会占用 rank，
  与 TP/CP 不能同时大于 1。

!!! note "不要使用 SP 缩写"
    Megatron sequence parallelism 是 TP 组内切 activation 的另一种机制，不形成独立
    并行维度。文档和配置统一写 `usp`、`Ulysses` 或完整的
    `Megatron sequence parallelism`。

**TP**
: Tensor Parallel。沿权重维度切分线性层；FSDP2 和 Megatron 都提供实现。

**CP**
: MCore Context Parallel。使用 zigzag 布局切 token 序列，可与 Megatron TP 组合。

**Batch group / leader**
: 必须看到同一 batch 的 rank group。组内 leader 持有并推进真实 dataloader，其余 rank
  接收广播。

组合选择见[分布式运行](../features/distributed-training.md)。

## 执行与精度

**FSDP2**
: PyTorch composable FSDP，按 DP mesh 分片参数、梯度和 optimizer state。

**HSDP**
: Hybrid Sharded Data Parallel。FSDP2 的二维 DP mesh：内层 `dp_shard` 分片，
  外层 `dp_replicate` 在 shard group 之间复制并归约梯度。

**训练随机流**
: 使用固定 seed 初始化、随后持续推进的随机状态。命名 generator 可隔离不同随机输入。
  重放依赖 RNG 状态和操作顺序，不依赖 step；完整续训同时恢复全局 RNG 和命名流。

**算子确定性**
: 固定输入与环境下，重复运算是否得到相同结果。`runtime.deterministic` 统一控制
  PyTorch、cuDNN 和 FA4 确定性，与随机 seed 和 TF32 精度选择分开。

**FSDP mixed precision**
: 分别声明 parameter、reduce、output、buffer 和 forward input dtype 的 policy。

**runtime.autocast**
: 独立的算子级 AMP 开关，不等于 FSDP parameter dtype。

更多语义见[随机性与精度](../features/execution-and-precision.md)。

## 两类 Checkpoint

**Activation recompute / activation checkpoint**
: 在 backward 重算 block forward，以计算换 GPU activation 显存；不落盘。

**Preserved-kernel offload**
: 将 replay 所需的 Flash Attention output/auxiliary 保存到 CPU，使 backward 跳过该
  kernel 的 forward 重算。配置名是 `attention_offload`。

这两项见[激活重算与 Offload](../features/activation-recompute-and-offload.md)。

**Weights-only checkpoint**
: 只保存模型权重，用于评估或 warm-start。

**Full checkpoint**
: 保存模型和可选完整训练状态的 DCP，用于故障恢复。

**PT**
: 单文件 weights-only 导出，不能完整 resume。

**DCP**
: PyTorch Distributed Checkpoint 目录，支持分片保存、reshard 和 full resume。

**Warm-start**
: 加载权重后从 step 0 开始新训练。

**Resume**
: 从已完成 step 继续，并按计划恢复 optimizer、scheduler、dataloader 和 RNG。

完整协议见[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

## 数据恢复

**StatefulDataLoader**
: 可通过 `state_dict/load_state_dict` 恢复 sampler、worker/prefetch 和 committed cursor
  的 torchdata loader。

**Dataloader fingerprint / contract**
: 描述 dataset、sampler、batch、worker 和 sample stream 语义的稳定元数据。Strict
  resume 要求一致。

**Production numerical correctness**
: 续跑指标在业务容差内，且没有遗漏训练状态的证据。

**Bit-exact / bitwise**
: 在声明边界内逐位一致；要求状态、数据流、topology 和相关 kernel/collective 都满足
  确定性条件。
