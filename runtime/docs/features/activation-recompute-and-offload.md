# 激活重算与 Offload 支持边界

!!! warning "支持状态"
    Activation recompute 是 Native experiment 的常用能力。Preserved-kernel offload 在
    Native 使用标准 gradient-checkpointing context，并在 FA 算子内部完成保存和恢复。
    目前 CPU 测试已覆盖；H200/FA3、B300/FA4、USP8 和 FA4 sparse 仍需对应 GPU 验证。

这两个功能都发生在一次 forward/backward 内，用于降低 GPU activation 显存。它们不是
训练恢复用的 PT/DCP checkpoint。

模型 recipe 已决定默认 recompute 策略。首次运行不要机械添加开关；显存不足或做专项
性能验证时再按本页调整。

## 先选哪一种

| 目标 | 配置 | 代价 |
| --- | --- | --- |
| 用计算换显存 | `parallel.gradient_checkpointing=true` | backward 重算 block forward |
| 少重算昂贵的 Attention kernel | Native preview 使用 `attention_offload.enable=true` | 占用 CPU 内存并增加传输 |
| 降低参数和 optimizer 显存 | 当前没有 CPU offload 配置 | 未交付 |

通常先开 activation recompute；只有 Flash Attention 重算成本明显且 CPU 内存足够时，
再评估 preserved-kernel offload。

## 工作过程

```text
Forward checkpoint block
  → 保存 Flash Attention output/aux
  → CPU offload
  → backward 前预取最后一个 checkpoint block
  → replay 在首次消费时恢复 output 并跳过 FA forward
  → FA backward 首次消费时恢复 LSE
  → 预取前一个 block，与当前 FA backward 重叠
```

## 当前推荐的 Native 方法

### 核心对象及生命周期

`ActivationOffloadManager` 在一次训练过程中负责：

- 保存和恢复选中的 FA output、LSE 和 metadata；
- 区分首次 forward 与 checkpoint replay；
- 在 backward 前预取数据，并在使用后释放。

Manager 使用 `ContextVar` 临时记录当前 checkpoint block。模型和 FA 不需要传递这个
变量；FA 在执行时通过 manager 查询当前 block。每个 FA 的保存结果直接放在其 autograd
ctx 上，直到对应 backward 完成。

首次 forward 和 replay 必须以相同顺序执行 FA。当前不支持 checkpoint 内再次进入另一个
使用同一 manager 的 checkpoint；检测到这种嵌套时会直接报错。顺序执行多个 checkpoint
blocks 不受影响。

### 接入原理

当前方法分为模型接入和 FA 算子执行两部分。Trainer 不查找模型内部的 FA 模块，也不
修改 block。

#### 模型接入

```text
Trainer 创建 ActivationOffloadManager
  → 通过 gradient_checkpointing_kwargs 传入 manager.checkpoint_contexts
  → parallelize_model() 调用 model.gradient_checkpointing_enable()
  → model root 将配置转发给内部 transformer
  → transformer 在执行 block 时调用 PyTorch checkpoint
```

Model root 必须提供以下接口：

```python
def gradient_checkpointing_enable(
    self,
    gradient_checkpointing_kwargs: dict | None = None,
) -> None:
    ...
```

这个接口只是统一入口，不会把整个 model 包进一次 checkpoint。具体 checkpoint 哪些
blocks，仍由内部 transformer 决定。

`ContextVar` 是 manager 的内部实现。进入 checkpoint block 时写入当前状态，离开后
立即恢复。模型和 FA 都不需要显式传递它。

#### FA 算子执行

FA 通过 `claim_checkpoint_call()` 判断当前调用是否需要保存。未选中的调用走普通
Flash Attention；选中的调用分三个阶段处理：

1. **首次 forward**：调用 FA private forward，得到 output 和 LSE；manager 将保存
   结果搬到 CPU；
2. **checkpoint replay**：恢复 output，跳过 FA forward；
3. **backward**：恢复 LSE，调用 FA private backward，然后释放保存结果。

Dense、varlen 和 sparse FA 统一使用 `AttentionCallMetadata`。Varlen 调用在其中保存
序列边界；sparse 调用只保留 backward 仍需要的 mask 和 tile 信息。

异步模式会在 backward 开始前预取最后一个 checkpoint block，并在处理当前 block 时
预取前一个 block。同步模式则在使用时直接恢复。

#### 扩展到新算子的当前边界

新算子可以复用 manager 的 CPU 传输、预取和 checkpoint 生命周期，但仍需实现自己的
autograd Function，说明 replay 和 backward 分别需要保存哪些 tensor。当前选择配置和
保存接口仍以 FA 为主，尚未形成完全通用的算子 API。

### 推荐用法

先用显式 backend、关闭 fallback，并保留每次 checkpoint block 调用中的第一个 FA：

```bash
vforge-train --config-name <experiment> \
  parallel.gradient_checkpointing=true \
  attention.backend=flash_attention_4 \
  attention.allow_fallback=false \
  attention_offload.enable=true \
  attention_offload.async_transfer=true \
  'attention_offload.preserve_fa_call_indices=[0]'
```

FA3 使用 `attention.backend=flash_attention_3`，但不支持 teacher-forcing sparse，也不
支持 preserved backward 的 deterministic 模式。以上 Native 路径在真实 GPU gate
完成前只作为 preview 使用。

#### FA 调用选择

- `[0]`：默认值，只保存当前 checkpoint block 调用中第一个受 backend 支持的 FA；
- `[1]`：只 preserve 第二个 FA；
- `[0, 2]`：preserve 指定的多个 FA；
- `null`：preserve 全部 eligible FA。

序号在首次 forward 和 replay 中分别从 0 重置。FA4 teacher-forcing sparse 参与编号；
FA3 不支持 sparse。

### 前置条件

1. 同时启用 `parallel.gradient_checkpointing=true`；
2. Native model root 必须实现 `gradient_checkpointing_enable()`，并将参数转发给需要
   重算的 transformer；
3. 目标 attention 调用必须走受支持的 Flash Attention 路径；
4. 节点需要足够的 CPU 内存；offload 不会把参数或 optimizer 移到 CPU。

当前不支持受同一 manager 管理的 checkpoint 相互嵌套，检测到后会直接报错。顺序执行
多个 checkpoint blocks 不属于嵌套，不受此限制。

SDPA/native attention 不会获得 preserved-kernel 收益。FA4 teacher-forcing sparse
已经接入同一个 FA Function，但真实 B300 数值与性能仍未验证。

### 模型边界

Native Wan 2.1、Wan 2.2 和 MiniMax-H3 已接入当前接口，但仍需完成对应硬件上的数值和
性能验证。配置使用 `attention_offload.*`。

### 如何确认生效

训练 step payload 会写入 `attention_offload`，W&B 使用 `offload/` 前缀。

| 指标 | 判断方式 |
| --- | --- |
| `offload/packed_tensors` | 大于 0 表示有 tensor 被保存并 offload |
| `offload/packed_gb` | 当前 step 搬到 CPU 的累计数据量 |
| `offload/peak_live_gb` | 当前 step 的 CPU live 峰值 |

不要只看 GPU OOM 是否消失。至少同时比较 peak GPU memory、CPU memory、step time 和
上述指标；`packed_tensors=0` 通常表示 backend、checkpoint callable 或开关组合不满足。
指标不替代对应硬件和拓扑的数值 gate。

### `AttentionOffloadPolicy` 的作用

`AttentionOffloadPolicy` 是 Trainer 使用的只读配置对象，不执行 tensor 传输，也不保存
forward/backward 状态。Hydra 配置经过校验后会转换成该对象；Trainer 再根据它决定是否
创建 `ActivationOffloadManager`，以及用哪些参数创建 manager。

它包含以下成员：

- `enable`：是否启用 attention activation offload。为 `false` 时不创建 manager；
- `preserve_fa_call_indices`：选择每次 checkpoint block 调用中需要保存的 FA。默认
  `(0,)` 表示第一个，多个序号表示保存指定调用，`None` 表示保存全部符合条件的 FA；
- `config`：传给 `ActivationOffloadManager` 的 tensor 传输配置，类型为
  `ActivationOffloadConfig`，其中：
    - `min_tensor_bytes`：tensor 小于该字节数时保留在 GPU，不搬到 CPU；
    - `pin_memory`：CPU tensor 是否使用 pinned memory；
    - `max_cpu_bytes`：一个训练 step 中允许同时占用的最大 CPU 字节数，`None` 表示
      不设置上限；
    - `async_transfer`：是否使用独立 CUDA stream 异步传输并预取。

### 完整配置

```yaml
attention_offload:
  enable: false
  preserve_fa_call_indices: [0]
  min_tensor_bytes: 1048576
  pin_memory: true
  max_cpu_bytes: null
  async_transfer: false
```

`async_transfer=false` 使用同步恢复，没有 prefetch overlap；GPU 性能评估应显式设为
`true`，且此时 `pin_memory` 必须为 `true`。`max_cpu_bytes` 是单 step live CPU tensor
预算，超出时直接失败。这些字段只改变显存、CPU 内存和传输调度，不进入 strict
resume fingerprint；恢复训练时可以调整。

## 常见问题

- FSDP2 mixed precision 在 activation recompute 中出现 dtype metadata 不一致：
  [FSDP2 AC 混合精度排障](../troubleshooting/fsdp2-activation-checkpoint-mixed-precision-bug.md)；
- 想调整持久 checkpoint 或远端存储：
  [Checkpoint 与 S3](checkpoint-and-storage.md)；
- 想了解配置字段：
  [通用 Override 速查](../reference/configuration-cheatsheet.md)和
  [配置方式与字段](../reference/configuration.md)。
