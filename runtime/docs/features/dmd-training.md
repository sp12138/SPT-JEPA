# DMD 训练

!!! note "当前状态"
    `wan22_ti2v_5b_dmd` 已在生产尺寸（121 帧 / 704×1280）上与太初（Taichu）的参照
    实现逐位比对：单卡 50 步的训练误差全部一位不差；八卡序列并行下第 1 步一位不差，
    之后因为拆分改变了加法顺序而分开，50 步内偏离中位 0.213%、最大 3.379%。

    `wan22_a14b_dmd` 的完整 40 层模型能在 8×B300、704×1280、21 个 latent 帧上跑完，
    但它的逐位比对是在缩小成每个网络 2 层的模型上做的，完整模型未比对。

    详细结果与剩余边界见 [DMD 开发交接](../development/dmd-handoff.md)。
    10|
## 训练对象

DMD 需要三个角色，每个角色是一个可以独立执行、独立按 FSDP2 切分的大型 transformer：

- `generator`：因果生成器，可训练；
- `real_score`：双向网络，给真实分布打分，始终处于 `eval` 且不计算梯度；
- `fake_score`：双向网络，给生成分布打分，可训练。

前向时这几个网络不会同时把完整参数聚合到一张卡上。指数滑动平均（EMA）也按现有
DTensor 分布保存成 FP32 分片，不会聚合出完整的 5B 生成器。

A14B 的每个角色各持高噪声和低噪声两个专家（expert），所以一共六个网络，也是六个
独立的 FSDP2 执行单元。生成过程直接驱动内层的专家，不经过持有它们的外层容器，因此
外层容器的参数聚合钩子不会触发；必须让角色把每个专家单独登记成执行单元，否则顶层
参数会留在切分状态，`patch_embedding` 会以 `mixed torch.Tensor and DTensor` 失败。

## 配置：DMD 是训练方式，不是模型名

配置通过 `training.recipe=dmd` 选择训练方式，并逐个声明模型角色：

```yaml
training:
  recipe: dmd
  models:
    generator:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b_causal_teacher
      init_weights: /path/to/causal.pt
    real_score:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b
      init_weights: /path/to/teacher.pt
    fake_score:
      model: wan_taichu
      config:
        variant: wan22_ti2v_5b
      init_weights: /path/to/teacher.pt
  dmd:
    algorithm: distribution_matching
```

`wan22_ti2v_5b_dmd` 只是 experiment 名，不出现在模型 variant 列表里。`model` 和它
私有的 `config` 描述单个网络；`algorithm` 决定需要哪些角色以及计算图长什么样。以后
生成器和打分网络来自不同模型家族时，只替换对应角色的引用，不需要造一个组合模型名。

输入优先复用普通的 latent 缓存：

```yaml
latent_cache:
  mode: offline
data:
  sources:
    - cache_dir: /path/to/wan-taichu-ct-cache
```

调试时也可设为 `online` 并提供带 `kind`、`index_files` 的原始数据源。两条路径使用
同一个 DMD 批次转换，不需要 DMD 专用的预处理命令。

## 通用框架与模型适配

三层边界、新模型接入方式和代码归属见
[DMD 架构设计](../development/dmd-architecture.md)。本页只描述现有 Wan 配置和运行
行为：模型插件提供适配器与输入转换，算法生成 `PhaseSpec`，`PhasedTrainer` 执行。

## 一个 global step 的顺序

`PhasedTrainer` 按 `PhaseSpec` 决定业务顺序，训练引擎不感知阶段名称，只执行一次
`zero_grad -> forward -> backward -> clip -> optimizer.step`：

```text
step 0: generator 阶段 -> EMA -> critic 阶段
step 1:                          critic 阶段
step 2:                          critic 阶段
step 3:                          critic 阶段
step 4:                          critic 阶段
step 5: generator 阶段 -> EMA -> critic 阶段
```

默认 `generator_every_n_steps=5`，也就是生成器每 5 步更新一次。critic 的生成过程必须
在生成器的优化器更新之后重新构造，因此 step 0/5 的 critic 一定看到刚更新的生成器，
而不是旧权重。生成器和 critic 使用不同的 AdamW 与不同的数据流；
`MultiStreamStatefulDataLoader` 会把所有命名数据流的读取位置一起写进完整的 DCP。

## 去噪轨迹与末档的 0

默认轨迹边界是：

```yaml
denoising_steps: [1000, 938, 833, 625, 0]
terminal_zero_is_boundary: true
```

数组里是五个**边界**，不代表五次 transformer 前向。最后那个 `0` 表示干净的终点；实际
transformer 调用的 timestep 是 `1000, 938, 833, 625`，共四次。把末档的 0 再执行一次会
改变训练语义，配置校验会拒绝这种写法。

121 个像素帧经 Wan VAE 得到 31 个 latent 帧。第 0 个 latent 帧是干净的图像锚点，不参与
DMD 与 critic 的损失。`frame_per_block=3` 时，后续 latent 帧按块自回归生成。
`frame_per_block` 与 `first_frame_own_block` 属于 `training.models.generator.config`，
不是 DMD 算法配置。

## 增量 Q/K 与 KV 缓存

因果生成每次的 query 只包含当前块：

- 首块：1 个 latent 帧，704×1280 下是 880 个 token；
- 后续块：通常 3 个 latent 帧，即 2640 个 token。

key/value 包含“历史缓存 + 当前块”，所以 `q_len != kv_len`。随着块推进，`kv_len` 最终
达到完整 31 帧序列的 27280 个 token。这是密集的矩形注意力，不是 teacher forcing 那种
整段带稀疏掩码的形式。

`StaticLayerKvCache` 的索引全部以 **token** 为单位：

- `current_start`：当前 query 块在全局 token 序列里的起点；
- `global_end`：历史缓存加当前块的右边界；
- `q_len`：当前块的 token 数；
- `kv_len`：当前可见历史与当前块的 token 总数。

B300 上必须显式选一个已经验证过 `q_len != kv_len` 前反向行为的密集 FlashAttention
实现，数值比对用的是 `attention.backend=flash_attention_4`。DMD 配置不允许静默换成会
改变掩码语义的实现。

## 梯度边界

生成器阶段需要生成器的梯度。默认 `last_step_only=true` 时，每个非锚点块只有最后一次
去噪前向保留自动求导，更早的去噪与缓存提交都在 no-grad 下进行。critic 阶段的整个生成
过程都是 no-grad，只有 `fake_score` 接收梯度。`real_score` 永远冻结。

这个边界同时决定数值语义和显存：如果意外保留了整个生成过程的计算图，31 帧训练即使在
8×B300 上也可能显存不足。

## 随机选择退出档位：`last_step_only=false`

档位之间的重新加噪在 no-grad 下进行，所以只有生成过程**退出的那一个档位**连着损失。
如果恒定停在末档，只服务更早档位的专家永远拿不到梯度 —— A14B 的高噪声专家就是这种
情况，它只出现在 1000 和 900 两档，而末档 477 恒在低噪声侧。`last_step_only=false` 让
每次迭代抽一个退出档位，梯度轮流覆盖每一档。

抽样必须所有卡一致，理由和下面的专家选择相同：档位决定跑哪个专家，各卡抽得不同就会在
不同的 FSDP 实例上发起参数聚合。抽样由 0 号卡完成后广播给全体，一次生成过程内所有块
共用同一个档位。

`context_full_trajectory=true` 让提前退出的块在 no-grad 下补完剩余档位再提交 KV 上下
文，避免把粗糙的中间结果喂给后续块。

`score.timestep_floor_at_rollout_exit=true` 是它的配套项：退出档只把样本降噪到下一档的
噪声水平，比这更干净的水平对该样本来说是分布外的，所以打分 timestep 的采样下界取该
水平。下界从 `scheduler.timesteps` 反查，与采样的重映射读同一张表。

## A14B 的专家路由

A14B 按 `boundary_ratio * num_train_timesteps = 900` 路由：`t >= 900` 走高噪声专家。
**每个专家各有自己的 KV 缓存**，所以一个块完成后要把它的上下文按各自的权重写进每一个
专家的缓存，而不只是这次前向用到的那个。

打分 timestep 的采样区间跨越 900 时，各数据并行卡独立抽样会把一部分卡送进高噪声专家、
另一部分送进低噪声专家，双方聚合不同专家的分片，集合通信随即错配（NCCL 只报出「同一
SeqNum 上 OpType 不同」）。做法是**先选边再在边内抽样**：0 号卡以
`(hi - split) / (hi - lo)` 的概率选高噪声侧并广播，因此 timestep 的边缘分布不变。单
专家的模型、以及区间本身已经落在一侧时短路，不额外抽样、不额外通信。`split` 同样从
`scheduler.timesteps` 反查得到。

A14B 的参考帧走 y 的掩码通道而不是独立的首块，所以 `first_frame_own_block=false`，
DMD 的首帧锚点关闭，首帧参与损失。

## 并行方式与显存

只支持 FSDP2。序列并行（Ulysses）对 TI2V-5B 和 A14B 都已打开，`usp_size` 需要同时整除
注意力头数和一个去噪块的 token 数；两个模型在 8 卡下都满足，16 卡下头数不能整除，需要
先补齐再切分，这条路径未验证。

生产尺寸必须开激活重算（`gradient_checkpointing: true`）。这个开关不改数值：单独翻它，
训练误差和每个参数的梯度都逐位相同。

A14B 完整模型在 704×1280 / 21 个 latent 帧上贴着单节点显存上限，有两个原因：

- `usp_size=1` 时每张卡自己扛一份按专家分开的全片缓存，40 层 × K/V × 73920 token ×
  5120 × bf16，单个专家约 60 GB，两个专家就是峰值的大头。开序列并行后缓存按头切分，
  这部分随卡数下降。
- AdamW 的状态**用到才分配**：一个专家只有第一次拿到梯度时才建状态，所以显存会随着
  退出档位和打分侧轮到不同专家而逐步爬升。前几步看起来富余不代表稳定。

所以 A14B 的 experiment 默认打开 `attention_offload`，每步把约 57 GB 的重算激活搬到
CPU 换回余量。爬到上限时的表现是所有卡停在同一个梯度范数汇总上、功耗掉到空转水平。

## EMA

每次生成器的优化器更新之后：

```text
ema = decay * ema + (1 - decay) * generator
```

默认 `decay=0.99`、`start_step=200`。step 200 之前不更新；第一次符合条件的生成器更新
直接用当前生成器初始化 EMA，之后才应用上式。EMA 的缓冲区从训练启动时就存在，这样
DCP 记录的模型结构指纹不会在 step 200 变化。

## Checkpoint 与严格恢复

完整的 DCP 包含：

- generator、real_score、fake_score；
- 生成器与 critic 两个优化器；
- 分片保存的 FP32 EMA；
- 多数据流加载器的所有命名读取位置；
- 随机数状态、global step、执行与配置指纹。

新任务分别从生成器、参照网络和打分网络各自独立的模型原生 checkpoint 初始化；之后完整
的 DCP 负责恢复全部网络、优化器、数据加载器、随机数和训练状态。offline 模式优先复用
普通 latent 缓存，online 模式从原始批次调用同一个 Wan 编码器。DMD 不额外增加预处理
命令或缓存格式。

最终模型用 `convert_checkpoint.component_path=generator` 从 DCP 单独导出；
`convert_checkpoint.weights=ema` 导出 EMA 版本，`raw` 导出训练中的即时生成器。
