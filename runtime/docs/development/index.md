# 开发者导读

本节解释代码所有权、扩展方式和验证门禁。用户只想运行模型时不需要阅读这里，应从
[Quick Start](../start/index.md) 或[支持矩阵](../models/index.md)开始。

## 快速接入

按要完成的改动直接进入对应接口指南，不要求先读完整架构：

- 新增模型：直接按[新增 Native 模型](adding-native-model.md)实现当前需要的 builder、
  注册项和 experiment；
- 新增模型推理：先确认
  [视频推理扩展边界](video-inference-extension.md)中的 `infer_sample` 必选面和可选
  flow-match helper；
- 新增跨模型能力：从[新增框架能力](adding-framework-feature.md)判断所有权和复用边界；
- 修改完成后：从[本地开发与验证](testing.md)选择最低成本、足以证明行为的检查；
- 只有真实 GPU、机器或多节点行为需要远端资源时，才进入
  [KAP 远程任务](kap.md)。

模型与数据语义留在 `models/<family>/` 和 `tasks/<family>/`；训练循环、分布式和
checkpoint 不复制到模型插件。只有改动跨越这些边界时，才需要读
[架构与所有权](architecture.md)。

## 场景指南

### 新增或迁移模型

以[新增 Native 模型](adding-native-model.md)为主线。它给出真实接口、实现顺序、完成
标准和 Wan21 参考路径。experiment 合成、meta materialize 和 engine 细节都不是开始
接入的前置阅读：

- 需要增加或调试模型配置时，再读[配置加载与覆盖](configuration-composition.md)；
- 模型含 `persistent=False` 派生 buffer 时，再读
  [Meta runtime buffer 合约](meta-runtime-buffers.md)；
- 模型需要特殊 FSDP/TP/CP 行为时，再读[训练引擎](training-engines.md)；
- 修改 Wan Taichu 模型时，参照[模型指南](../models/wan-taichu.md)中的 variant、
  cache、训练输入和权重布局，以及[评估契约](wan-taichu-evaluation.md)。

完成插件和 contract test 后，更新模型 One Pager 和[支持矩阵](../models/index.md)。

### 在已有模型上增加训练变体

增加 block-causal 训练变体时，按
[在已有模型上适配 Teacher Forcing](adapting-teacher-forcing.md)复用已有 encode、
eval 和权重加载，只替换训练 step。

### 新增跨模型框架能力

从[新增框架能力](adding-framework-feature.md)开始。先确认能力至少被两个模型或项目
共享；否则优先留在模型插件，不要提前下沉 core。实现时按涉及的机制选择内部文档，
最后回到[本地开发与验证](testing.md)和按风险选择的 [KAP gate](kap.md)。

### 修复或验证已有功能

先定位改动所属机制：

- FSDP2、TP、CP 或 Megatron：[训练引擎](training-engines.md)；
- 保存、恢复或 S3 staging：[Checkpoint 与 S3 内部机制](checkpoint-internals.md)；
- meta 模型 materialize：[Meta runtime buffer 合约](meta-runtime-buffers.md)；
- schema、experiment 或 CLI 合成：[配置加载与覆盖](configuration-composition.md)；

测试命令和远端资源操作只在[本地开发与验证](testing.md)和 [KAP](kap.md)维护。

## 内部机制

以下页面用于维护框架边界，不是所有模型作者的必读前置：

- [架构与所有权](architecture.md)：目录职责、依赖方向和跨层改动；
- [视频推理扩展边界](video-inference-extension.md)：`EvalComponents`、模型推理和可选
  flow-match helper 的所有权；
- [配置加载与覆盖](configuration-composition.md)：Hydra compose、schema 和 translate；
- [Meta runtime buffer 合约](meta-runtime-buffers.md)：meta 构造后的派生状态恢复；
- [训练引擎](training-engines.md)：并行化、mixed precision 和 engine 生命周期；
- [Checkpoint 与 S3 内部机制](checkpoint-internals.md)：保存、恢复、格式和 staging。

## 用户行为与内部实现的边界

| 用户文档 | 开发者文档 |
| --- | --- |
| [分布式运行](../features/distributed-training.md) | [训练引擎](training-engines.md) |
| [配置方式与字段](../reference/configuration.md) | [配置加载与覆盖](configuration-composition.md) |
| [Checkpoint 与 S3](../features/checkpoint-and-storage.md) | [Checkpoint 与 S3 内部机制](checkpoint-internals.md) |
| [激活重算与 Offload](../features/activation-recompute-and-offload.md) | `checkpointing/` 与 `ops/` 所有权见[架构](architecture.md) |

用户页说明“怎么用和有什么边界”，开发者页说明“谁负责、如何实现和怎么验收”。不要在
两边复制同一段实现细节。

## 开始修改前

提交前至少按[本地开发与验证](testing.md)完成 changed-file lint、定向测试和文档检查；
真实 GPU 行为再按风险进入 [KAP](kap.md)。面向开发者的网页文档只维护公开流程和
contract，不反向依赖 Agent 专用指令。
