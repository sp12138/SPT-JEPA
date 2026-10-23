# vforge 文档

vforge 是面向视频大模型训练的原生 PyTorch 分布式框架。第一次使用先通过 Quick Start 了解安装、数据准备、训练和产物检查的共同流程；开始正式任务时，再进入对应模型的
One Pager 获取训练、恢复和评估命令。

## 按模型开始

| 模型 | 状态 |
| --- | --- |
| [Wan 2.1 I2V 14B](models/wan21.md) | **稳定** |
| [MiniMax H3 FL2VA](models/minimax-h3.md) | **预览** |
| [Wan 2.2](models/wan22.md) | **内部验证** |

完整状态定义和 experiment 名称见[支持矩阵](models/index.md)。本轮文档范围不声明其它
已注册模型的支持状态。

## 推荐阅读顺序

1. 第一次使用：按 [Quick Start](start/index.md) 跑通 Wan 2.1 最小任务；
2. 正式任务：进入[支持矩阵](models/index.md)，从对应模型 One Pager 复制完整命令；
3. 需要调并行、显存或恢复时，再查“用户指南”中的专题；
4. 命令失败时，从[故障排查](troubleshooting/index.md)按症状进入。

## Wan Taichu 模型

Wan Taichu 模型使用 Native 入口，配置与能力边界见
[Wan Taichu 模型指南](models/wan-taichu.md)。

## 按需查阅

- **修改配置**：[配置方式与字段](reference/configuration.md)和
  [通用 Override 速查](reference/configuration-cheatsheet.md)；
- **理解运行语义**：[概念与术语](reference/concepts.md)和
  [分布式运行](features/distributed-training.md)；
- **保存与远端存储**：[Checkpoint 与 S3](features/checkpoint-and-storage.md)，包括
  S3 mount 路由、异步上传，以及需显式开启的按需下载优化；
- **Activation 显存优化**：
  [激活重算与 Offload](features/activation-recompute-and-offload.md)；
- **执行与数值验证**：[随机性与精度](features/execution-and-precision.md)。

## 我想开发 vforge

从[开发者导读](development/index.md)开始：先确认架构与代码所有权，再按当前任务进入
“新增或迁移模型”“开发跨模型框架能力”或“修复与验证已有机制”三条路径。

模型和新增能力统一走 Native，模型定义与任务装配的边界见
[架构与所有权](development/architecture.md)。

## 在哪里找什么

按问题类型直接进入对应页面：

- 跑通某个模型：对应模型 One Pager；
- 查常用配置组合：[通用 Override 速查](reference/configuration-cheatsheet.md)；
- 查字段和配置文件写法：[配置方式与字段](reference/configuration.md)；
- 判断模型是否稳定支持：[支持矩阵](models/index.md)；
- 了解功能行为和使用边界：用户指南中的功能专题；
- 修改框架实现：开发者文档。
