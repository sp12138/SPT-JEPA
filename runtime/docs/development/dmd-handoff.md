# VForge DMD 开发交接

本文只描述 DMD 训练当前的状态：哪些配置验证过、验证到什么程度、还剩哪些边界。
三层结构和新增模型/算法的做法见 [DMD 架构设计](dmd-architecture.md)；配置字段和
运行时行为见 [DMD 训练](../features/dmd-training.md)。

## 一句话状态

Wan2.2 TI2V-5B 和 Wan2.2 A14B 两条 DMD 配置都能训练。TI2V-5B 在生产尺寸上、单卡和
八卡序列并行两种方式下，都与太初（Taichu）的参照实现比对过每一步的训练误差。A14B
的完整模型能跑完，但它的数值比对是在缩小成每个网络 2 层的模型上做的。

## 训练时同时持有几个网络

DMD 需要三个角色：生成器（`generator`，可训练）、给真实分布打分的网络
（`real_score`，始终冻结、不算梯度）、给生成分布打分的网络（`fake_score`，可训练）。
TI2V-5B 每个角色一个 transformer，所以是三个；A14B 每个角色各有高噪声和低噪声两个
专家（expert），所以是六个。

这些 transformer 各自是独立的 FSDP2 执行单元，前向时不会同时把全部参数聚合到一张
卡上。A14B 尤其要注意：生成过程直接驱动内层的专家，不经过持有它们的外层容器，所以
外层容器的参数聚合钩子不会触发。必须让角色把每个专家单独登记成执行单元，否则顶层
参数会留在切分状态，`patch_embedding` 会以 `mixed torch.Tensor and DTensor` 失败。

## 数值比对结果

比对方法：两套框架喂完全相同的数据、随机种子和初始权重，跑在同一个 Python 环境里，
逐步取完整精度的训练误差做逐位比较。50 步为一组。

| 配置 | 规模 | 单卡逐位相同 | 八卡序列并行 |
| --- | --- | --- | --- |
| TI2V-5B | 完整模型，121 帧 / 31 个 latent 帧，704×1280 | 50 / 50 步 | 仅第 1 步；之后偏离中位 0.213%、最大 3.379% |
| A14B | 每个网络 2 层（完整是 40 层），81 帧 / 21 个 latent 帧，704×1280 | 50 / 50 步 | 仅第 1 步；之后偏离中位 0.122%、最大 2.011% |
| A14B | 完整的 40 层模型，尺寸同上 | 未比对，能跑完 | 未比对 |

单卡逐位相同是很强的结论：下一步的训练误差依赖上一步更新后的权重，能连续 50 步都
一位不差，说明前向、反向和优化器更新三者都一致。

八卡下只有第 1 步逐位相同，这是预期结果而不是缺陷。第 1 步是初始权重上的纯前向，
序列并行的注意力是精确重排，所以前向必须一致。从第 2 步起带上了反向的差别：把一段
求和拆到八张卡上分别做再汇总，加法顺序变了，而浮点加法不满足结合律。这个差别改不
成零，除非放弃拆分。

只看反向本身（第 1 步的梯度大小，此时权重还没更新过，不含逐步累积）：

| 第 1 步生成器梯度大小 | TI2V-5B | A14B |
| --- | --- | --- |
| 单卡（两套框架互差） | 1.2e-08 | 1.1e-07 |
| 太初八卡，相对单卡 | 1.648% | 7.252% |
| VForge 八卡，相对单卡 | 0.077% | 0.200% |

VForge 这一侧比太初更贴近单卡，而且它的拆分更彻底：太初那份配置把八张卡全用于切
序列，参数在每张卡上仍是完整一份；VForge 还额外把参数按八份切开。所以两套框架在
八卡下的差别主要由太初那一侧带来。太初内部各项占比不在本次范围内。

## 并行方式

只支持 FSDP2，其它训练引擎会在校验阶段直接报错。

序列并行（Ulysses）对 TI2V-5B 和 A14B 都已打开。生成过程走增量 KV 缓存，所以并行
要同时覆盖缓存和主干：缓存每张卡存 `注意力头数 / usp_size` 个头，块的前向按同样的
比例切分序列。两个约束由 `_validate_dmd_usp` 和块前向分别检查：

- `usp_size` 必须整除注意力头数。两个模型在 8 卡下都满足。
- `usp_size` 必须整除一个去噪块的 token 数。块含几帧由因果布局决定，所以这一条放在
  块前向里查。

`usp_size=16` 时两个模型的头数都不能整除，需要先补齐再切分。补齐这条路径可能另外
引入前向差别，没有验证过。**如果哪天 16 卡下前向不再逐位相同，先查这里。**

## 显存

生产尺寸必须开激活重算（`gradient_checkpointing: true`）。这个开关本身不改数值：在
A14B 的缩小模型上单独翻它，两个网络的训练误差和全部 69 个参数的梯度都逐位相同。

A14B 完整模型在 704×1280 / 21 个 latent 帧上贴着单节点上限。两个原因：

- `usp_size=1` 时每张卡自己扛一份按专家分开的全片缓存，40 层 × K/V × 73920 token ×
  5120 × bf16，单个专家约 60 GB，两个专家就是峰值的大头。开序列并行后缓存按头切分，
  这部分随卡数下降。
- AdamW 的状态是**用到才分配**的。一个专家只有第一次拿到梯度时才建状态，所以显存会
  随着退出档位和打分侧轮到不同专家而逐步爬升。前几步看起来富余不代表稳定，要等最后
  一个专家的状态落地才算到峰值。

因此 A14B 的配置默认打开 `attention_offload`，每步把约 57 GB 的重算激活搬到 CPU 换
回余量。爬到上限时的表现是所有卡停在同一个梯度范数汇总上、功耗掉到空转水平。

## 激活重算与增量 KV 的两个约束

生成过程每个块单独做一次前向，而激活重算发生在 `backward` 内部，此时块的调用环境已
经退出。由此有两个必须处理的问题，代码里都已处理，改动这条路径时不要退回去：

1. KV 缓存通过 `use_incremental_kv_cache` 这个 ContextVar 传递，重算时上下文已解除，
   `attn1` 取不到缓存。`run_transformer_blocks` 因此接受 `block_context`，由重算边界
   在重算时重新绑定缓存。
2. `update_and_attend` 原本读缓存的实时 `committed_tokens`。后续块会继续提交，等到
   反向重算时这个计数已经推进，重算会看到当时并不存在的未来帧。
   `IncrementalKVAttentionContext.history_tokens` 在前向时把可见历史固定住，重算据此
   复现同一个窗口。

第 2 点原本是**静默的错梯度**。这个例子里 PyTorch 恰好因为 KV 长度的形状变化报出
`CheckpointError`（前向 12 个 token、重算 16 个），但定长滑动窗口不改变形状，就不会
被发现。所以 `local_attn_size != -1` 与激活重算同时出现时直接报错，而不是返回可疑
梯度。

## 关键代码位置

- `vforge/algorithms/base.py`：通用 `Algorithm` 抽象基类；
- `vforge/algorithms/dmd/__init__.py`：DMD 公共入口与算法注册；
- `vforge/algorithms/dmd/model.py`：适配器、角色模型、家族声明与构建结果；
- `vforge/algorithms/dmd/algorithm.py`：标准 Distribution Matching 的阶段与损失；
- `vforge/algorithms/dmd/runtime.py`：成对数据加载、随机数、打分噪声与逐块生成；
- `vforge/algorithms/dmd/objectives.py`：打分梯度与损失公式；
- `vforge/algorithms/dmd/config.py`：标准算法配置；
- `vforge/tasks/dmd/train.py`：DMD 训练入口；
- `vforge/tasks/wan_taichu/dmd/integration.py`：Wan 侧钩子、校验与唯一的家族声明；
- `vforge/tasks/wan_taichu/dmd/algorithm.py`：Wan 配置、x0 舍入和 critic loss；
- `vforge/tasks/wan_taichu/dmd/runtime.py`：Wan 随机数、时间步和 KV rollout 数值规则；
- `vforge/models/wan_taichu/inference/incremental_kv.py`：Wan 的增量 KV 前向；
- `vforge/models/causal/cache.py`：定容量 KV 缓存；
- `vforge/models/causal/context.py`：一次块前向的可见历史；
- `vforge/training/recipe.py`：`PhaseSpec`、`PhasedTrainComponents`；
- `vforge/training/phased_trainer.py`：多阶段执行框架。

## 还没做的

- A14B 完整 40 层模型的数值比对。现有结论来自 2 层的缩小模型。
- `usp_size=16` 及更高。头数不能整除，要走补齐路径，未验证。
- 张量并行、Megatron 引擎。
- 生成前的 ODE 预热和 ODE 初始化。
- 跨 global step、每个片段各自做一次优化器更新的长序列训练。参数更新后旧 KV 来自旧
  权重，不能直接复用，需要能保存语义上下文并用新权重重建 KV。
- 生成器与打分网络来自不同模型家族时的输入桥接。
