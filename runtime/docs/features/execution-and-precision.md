# 随机性与精度

训练随机状态、算子确定性和 mixed precision 分别控制随机输入、计算路径和 dtype。
模型 recipe 提供各自的精度默认值；正常训练只需通过 `runtime.seed` 指定随机流的初始状态。

## 训练随机状态

Trainer 在启动时使用 `runtime.seed` 播种一次全局 RNG（随机数生成器）。模型输入通过
`StepContext.random_streams` 获取命名 generator，首次创建时播种，后续持续消费。
长度、噪声等独立随机输入可以使用不同名字，避免相互影响。

RNG 与训练 `step` 无关。每次随机操作推进自身状态，课程和学习率仍按更新步推进。
固定 seed 的对照需要相同的初始 RNG 状态和随机操作顺序，不能靠重用某个 step 重放。
完整续训同时恢复全局 RNG 和显式 generator 状态。

训练日志保留 loss、workload 和 task 提供的样本信息，不自动计算每步输入、输出及
loss 的 tensor hash。需要逐张量排查时，可在测试中使用 `vforge.utils.comparison`。

## 确定性计算

通过一个运行级字段统一开启：

```bash
vforge-train experiment=wan21_i2v_14b runtime.deterministic=true
```

`runtime.deterministic` 默认 `false`。运行初始化时，它同时控制：

- `torch.use_deterministic_algorithms(enabled, warn_only=False)`；
- `torch.backends.cudnn.deterministic = enabled`；
- FA4 的确定性 forward/backward 路径，包括重算和 offload 使用的 kernel。

开启时还会关闭 cuDNN benchmark，并在未指定 `CUBLAS_WORKSPACE_CONFIG` 时设置为
`:4096:8`，供确定性 cuBLAS 运算使用。关闭时取消上述三项确定性要求；不修改 seed、
随机流状态或 TF32 设置。TF32 属于精度选择，数值测试可单独设置。

该开关适用于训练、评估和 prepare 的共享 runtime。`sdpa`/`sdpa_math` 由 PyTorch
确定性机制约束；显式 FA4 和 `auto` 选中的 FA4 使用同一个开关。当前 FA3 路径不支持
确定性：显式选择会在配置校验时报错，`auto` 选中 FA3 时会在 backend 解析时报错。

开启后遇到不支持确定性的算子会直接报错，也可能降低性能；它不保证不同硬件、
CUDA kernel、collective 或 topology 逐位一致。

!!! warning "RNG 行为变更"
    `runtime.execution` 和三档执行模式已删除；原 `attention.deterministic` 移到
    `runtime.deterministic`，旧 override 会报错。
    当前统一使用持续随机流，不再保留旧的逐 step 播种序列。
    旧执行策略的 checkpoint 不接受严格续训；可通过 weights-only 开始新训练。
    CPU contract 和 B300 CUDA RNG 状态恢复测试已通过；LTX 与 Wan 的 loss golden
    已重新采集并独立复跑。模型验证范围分别见 [LTX](../models/ltx23.md)
    与 [Wan Taichu](../models/wan-taichu.md)。完整模型 GPU 续训尚未重新验证。

## FSDP mixed precision

常用字段：

```yaml
parallel:
  mixed_precision:
    enabled: true
    param_dtype: bfloat16
    reduce_dtype: float32
    output_dtype: null
    buffer_dtype: null
    cast_forward_inputs: true
```

必须保留 dtype 的模块来自模型的 `_keep_in_fp32_modules`，`FSDPPrecisionContract` 只
声明不能被预转换的 forward input 边界。保留只作用于模块，字段语义见
[`parallel.mixed_precision`](../reference/configuration.md#parallelmixed_precision)。

### 为什么需要模型级精度边界 {#model-precision-boundaries}

FSDP mixed-precision policy 不只决定参数在 forward 中使用什么 dtype；当
`cast_forward_inputs=true` 时，它还会在每个 FSDP group 入口转换浮点输入。对结构统一的
Transformer block，这可以让大部分计算稳定运行在 BF16。但多模态模型通常还包含两类
例外：

1. 输入/输出投影、timestep embedding 等数值敏感模块必须按模型原有约定使用 FP32；
2. block 的主 hidden state 可以是 BF16，但 timestep、RoPE、调制参数等辅助输入需要保留
   模型在 forward 中选择的 dtype，不能被 FSDP 在模块入口统一转换。

只让 checkpoint 中的参数保持 FP32 不能解决第一个问题：FSDP 仍可能在 forward 时临时把
参数转换为 `param_dtype`。全局关闭 mixed precision 或输入转换又会让所有 block 失去
BF16 的显存和吞吐收益。把 dtype 写死在通用 FSDP 代码中也不成立，因为哪些模块敏感是
模型语义，不是并行框架能够推导的事实。

因此模型声明少量例外，框架再把它们转换为明确的 FSDP group 边界：

| 边界 | 模型声明位置 | FSDP 行为 | 适用场景 |
| --- | --- | --- | --- |
| 保留模块 | `_keep_in_fp32_modules` | 该模块树形成独立 FSDP group，不应用 mixed-precision policy | 模块参数需要沿用现有 dtype，输入转换由模型负责 |
| 保留 forward input | `FSDPPrecisionContract.preserve_forward_input_paths` | 参数、归约和输出 policy 保持不变，只对该 group 设置 `cast_forward_inputs=false` | 参数仍按 BF16 等 policy 计算，但辅助输入不能被 FSDP 预转换 |

模型侧只有 `_keep_in_fp32_modules` 能决定哪些模块被保留，合同不能另外声明模块路径：
同一份名单同时决定参数的存储 dtype 和 FSDP 豁免范围，两者不会各说各话，也不存在“存成
FP32 但仍被 policy 转换”这种只在数值上表现出来的分歧。没有声明这份名单的模型不会得到
任何豁免。

保留模块不等于“强制 FP32”：它保留模块已有的参数和 buffer dtype，且 FSDP 不再替该
group 转换输入；模型仍要保证输入与参数 dtype 兼容。保留 forward input 也不是“让模块
运行在 FP32”，它只禁止 FSDP 在 group 入口统一转换参数以外的浮点输入；模块内部的显式
转换和 autocast 仍然生效。两者指向同一边界时，保留模块优先。

路径相对于模型声明的 FSDP execution root。`.` 只可用于
`preserve_forward_input_paths`，表示不要预转换进入 execution root 的浮点参数；整个
execution root 不能从 mixed-precision policy 中排除。不存在的路径会在模型分片前失败；
配置中的空路径和重复路径会在配置校验时失败；`preserve_module_paths` 中相互重叠的
module tree 也会被拒绝。

路径支持 `*` 通配一整段，`blocks.*.norm1` 覆盖每一层的 `norm1`，wrap 时展开成具体
路径，不必按层手写。`*` 不做段内匹配，`blocks.norm*` 不合法，`blocks.*.norm1` 也不会
匹配 `blocks.0.foo.norm1`。

`preserve_module_paths` 只能指向模块：保留 dtype 靠的是让目标单独成一个不带 policy 的
group，而单个权重没有属于自己的 group。路径指向单个权重时会被跳过，rank 0 打一行说明，
该权重仍按 `param_dtype` 参与前反向。

### 模型声明与配置覆盖

模型默认值应由模型自己声明。例如 H3 用 `_keep_in_fp32_modules` 让输入/输出投影和
timestep embedder 保持 FP32，再用合同阻止 FSDP 改写 execution root、Transformer block
和输出 norm 的辅助输入 dtype：

```python
_keep_in_fp32_modules = [
    "proj_in",
    "audio_proj_in",
    "time_embedder",
    "proj_out",
    "audio_proj_out",
    "rope",
]


def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
    return FSDPPrecisionContract(
        preserve_forward_input_paths=(
            ".",
            "norm_out",
            *(f"transformer_blocks.{index}" for index in range(len(self.transformer_blocks))),
        ),
    )
```

`parallel.mixed_precision.preserve_*` 是运行配置覆盖。`contract_mode=append` 会在模型
默认值后追加并去重；`replace` 会完全替换模型声明，主要用于模型接入和数值归因，不应
作为普通训练配置复制。不同模型的敏感边界不同，不能复用上述 H3 路径。

这套 precision contract 与 FSDP2 + activation checkpoint 的兼容补丁解决不同问题：
前者表达模型有意保留的 dtype 边界，后者保证原始 forward 与 backward 重算执行同一份
已生效 policy。相关故障见
[FSDP2 AC 混合精度排障](../troubleshooting/fsdp2-activation-checkpoint-mixed-precision-bug.md)。

## Autocast

`runtime.autocast` 是独立开关。Structured schema 默认关闭；Wan21 保持关闭，部分
Wan22 experiment 会显式开启。不要把 autocast 与 FSDP parameter
dtype 当成同一配置，最终值以 resolved config 为准。

## 显存不足时怎么选

| 手段 | 主要解决 | 当前范围 | 主要代价或限制 |
| --- | --- | --- | --- |
| BF16 mixed precision | 参数、activation 和计算显存 | 模型 recipe 已提供验收默认值 | 改变数值路径，不要只为省显存随意覆盖 |
| FSDP reshard | 参数驻留显存 | FSDP2 已支持；按模型验收 topology 使用 | 增加 all-gather 等通信 |
| Activation recompute | Activation 显存 | Native experiment 常用，通常已由 recipe 开启 | backward 重算 forward |
| Preserved-kernel offload | 减少 Attention 重算 | Native preview；配置与验证边界见[激活重算与 Offload](activation-recompute-and-offload.md) | 占用 CPU 内存并增加传输 |

先保持模型 recipe 的精度和并行默认值，只针对确认的瓶颈调整。修改 BF16 或 autocast
时，固定 seed、输入和计算环境做数值对照，并保存 resolved config；每次只调整一个变量。

Recompute/offload 的组合、模型边界和指标见
[激活重算与 Offload](activation-recompute-and-offload.md)。它们都不是落盘 checkpoint；
训练恢复见 [Checkpoint 与 S3](checkpoint-and-storage.md)。

`runtime.gradient_clip_val` 只控制梯度裁剪，不属于精度或显存策略。
