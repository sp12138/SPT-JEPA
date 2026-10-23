# 分布式运行：Prepare、Train 与 Eval

Prepare、Train 和 Eval 共用同一套分布式 runtime，但并行语义不同：Prepare 按样本
分片，Train 由 FSDP2 或 Megatron engine 执行模型并行，Eval 组合 DP 与模型已支持的
序列并行。

首次启动先使用模型 One Pager 的验收 topology；只有需要扩大卡数、序列长度或切换
引擎时再阅读本页。

## 启动进程和指定卡数 {#launch-topology}

`torchrun` 负责创建进程，vforge 只消费它注入的 rank 环境：

- `--nproc_per_node`：每个节点的进程数，通常等于每节点 GPU 数；
- `--nnodes`：节点数；
- `--node_rank`：当前节点序号，从 0 开始；
- `--master_addr` / `--master_port`：所有节点共同使用的 rendezvous 地址。

`--standalone` 只适用于单节点，不要带到多节点命令。下面是两机、每机 8 卡的启动
模板；两台机器执行相同命令，只把 `NODE_RANK` 分别设为 0 和 1：

```bash
export MASTER_ADDR=node0.example.internal
export MASTER_PORT=29500
export NODE_RANK=0  # 第二个节点设为 1

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/shared/models/Wan2.1-I2V-14B \
  data.train_manifest=/shared/data/wan21/train.jsonl \
  data.cache_dir=/shared/data/wan21/cache \
  runtime.output_dir=/shared/runs/wan21-2x8 \
  parallel.usp_size=8 \
  runtime.max_steps=100
```

此时总 `WORLD_SIZE=16`，`parallel.dp_size=auto` 推导为 2，即 USP8 × DP2。权重、cache、
output 和 checkpoint 路径必须对所有节点可见。该命令展示 Native 多节点拉起语法；
当前 Wan21 公开验收仍是单机 USP8，不能把语法可运行直接解释为 2×8 已验收。

当前自动 CI 的多节点证据是 GB300 2×4 卡模型无关 HSDP gate；它证明 HSDP mesh、
跨节点 collective 和 DCP roundtrip 可工作，不加载业务模型。该证据不等于
Wan21 Native 2×8、USP16 或多节点 full DCP resume 已通过验收。

## HSDP：节点内分片、节点间复制

HSDP 是 FSDP2 的二维 DP mesh。`dp_shard` 在组内分片参数、梯度和 optimizer state，
`dp_replicate` 在多个 shard group 间复制模型并归约梯度：

```text
dp_size = dp_replicate_size × dp_shard_size
WORLD_SIZE = dp_size × usp_size × cp_size × tp_size
```

框架不会猜测节点拓扑。未配置两个字段时保持普通 FSDP：全部 DP rank 属于同一个 shard
group。只有 `dp_replicate_size>1` 且 `dp_shard_size>1` 才启用 HSDP，并且只支持
`parallel.engine=fsdp2, parallel.dp_mode=fsdp2`。

两节点、每节点 4 卡的推荐布局是节点内 4-way shard、节点间 2-way replicate：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/shared/models/Wan2.1-I2V-14B \
  data.train_manifest=/shared/data/wan21/train.jsonl \
  data.cache_dir=/shared/data/wan21/cache \
  parallel.dp_replicate_size=2 \
  parallel.dp_shard_size=4 \
  runtime.max_steps=100
```

torchrun 连续分配 global rank 时，`dp_shard` 是快变维度，因此每组连续 4 个 rank
构成节点内 shard group。自定义 rank 映射时必须自行保证物理布局。HSDP 可与 USP
组合，此时 FSDP 内层 mesh 是 `dp_shard × usp`；USP 仍不能与 TP 或 CP 组合。

Full DCP strict resume 会把 replicate/shard split 纳入 topology fingerprint，因此要求
split 不变。Weights-only 或 non-strict warm-start 可由标准 DCP 在普通 FSDP 与 HSDP
之间 reshard，不承诺 bitwise 轨迹。

不加载业务模型的维护者 smoke：

```bash
torchrun --standalone --nproc_per_node=4 scripts/smoke_fsdp2_hsdp.py \
  --dp-replicate-size=2 \
  --dp-shard-size=2 \
  --checkpoint-dir=/tmp/hsdp-dcp \
  --checkpoint-mode=roundtrip
```

当前实机证据包括单机 4×B300 的 2×2 HSDP 训练、DCP roundtrip、HSDP→FSDP
reshard，以及双机 2×4 GB300 的跨节点 2×4 HSDP DCP roundtrip 和
HSDP→FSDP load。H3 33B 还通过了单机 8×B300 的 2×4 HSDP full DCP 保存与
strict resume。通用 MR gate 使用
[KAP 定向测试](../development/kap.md)。

## Prepare Latent 的多卡和多机边界 {#distributed-prepare}

`vforge-prepare` 是默认单进程入口。大数据集可用 `torchrun -m vforge.tasks.prepare`
启动多个 rank；runner 按 manifest 顺序枚举样本，并以 `index % WORLD_SIZE` 分配 rank，
每个样本只由一个 rank 编码。下面是两机、每机 8 卡的命令，两台机器只需分别设置
`NODE_RANK=0/1`：

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m vforge.tasks.prepare \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/shared/models/Wan2.1-I2V-14B \
  data.train_manifest=/shared/data/wan21/train.jsonl \
  data.cache_dir=/shared/data/wan21/cache \
  parallel.usp_size=1
```

所有节点必须看到相同的模型 snapshot、manifest 和共享 `cache_dir`。每个 rank 都加载
完整 condition encoder，Prepare 不使用 USP/TP/CP 拆分单个样本，因此：

- 多卡/多机提高的是样本吞吐，不能解决 encoder 或单样本 OOM；
- 保持 `parallel.usp_size=1`、`tp_size=1`、`cp_size=1`，`dp_size=auto` 自动使用全部
  rank；
- 全部 rank 完成写入后，rank 0 校验所有 bundle 并原子发布 cache manifest；
- 当前有通用分布式 runner 支持，但没有 Wan21/H3 多节点 Prepare 的公开性能或稳定性
  gate；正式扩容前先用小 manifest 检查样本数、fingerprint 和共享文件系统语义。

## Eval 的多卡和多机边界

`vforge-eval` 只启动一个进程；多卡 Eval 使用相同配置和 `torchrun` 模块入口。例如
Wan21 单机 USP8：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.eval \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  evaluation.data_source=validation \
  evaluation.manifest=/data/wan21/val.jsonl \
  parallel.usp_size=8 \
  evaluation.checkpoint_path=outputs/train/checkpoints/00100.dcp \
  evaluation.num_samples=8
```

当前 Eval runner 的分布式行为：

- **DP**：不同 DP rank 处理不同样本；
- **Ulysses**：一个 DP group 内的 rank 协同处理同一样本，只由 batch leader 编码并写
  产物；
- **TP / CP**：通用 Eval 路径尚未接入 engine-specific `prepare_model`，设置
  `parallel.tp_size>1` 或 `parallel.cp_size>1` 会显式失败。

多节点 Eval 使用本页开头相同的 `--nnodes`、`--node_rank` 和 rendezvous 参数，并把
`-m vforge.tasks.train` 改为 `-m vforge.tasks.eval`；runner 可以跨节点做 DP/Ulysses
collective。但当前 Native 模型没有公开的多节点 Eval gate，因此不能从 launcher 和
HSDP train smoke 推导多节点 Eval 已验收。当前公开证据是 Wan21 单机 USP8 gate和
H3 单机 USP4 gate。

## 怎么选

```text
默认或首次运行
  └─ FSDP2
      ├─ 序列太长：增加 USP
      └─ 需要 tensor parallel：使用 Torch TP

明确需要 MCore TP/CP
  └─ Megatron + DDP
      ├─ TP
      ├─ CP
      └─ CP × TP
```

先选引擎，再选引擎支持的并行维度。USP 不能与 TP 或 CP 组合；TP × CP 只在
Megatron 中支持。非法组合会在模型大规模分配前 fail-fast。

## 能力矩阵

| 需求 | FSDP2 | Megatron |
| --- | --- | --- |
| HSDP `dp_replicate_size>1` | 支持 | 拒绝 |
| Ulysses `usp_size>1` | 支持 | 拒绝 |
| Tensor parallel `tp_size>1` | 支持 | 支持 |
| Context parallel `cp_size>1` | 拒绝 | 支持 |
| CP × TP | 拒绝 | 支持 |
| Expert parallel `ep_size>1` | 预览：FSDP2，可叠加 USP；需模型 bank 声明 | 拒绝 |
| 保存 optimizer state | 支持 | 暂不支持 |

公开支持组合以此表为准。实现和数值证据见
[训练引擎](../development/training-engines.md)。

## FSDP2 + Expert Parallel（预览）

Expert Parallel（EP）将一个专家池的专家分配到不同 rank，并把 token 发送到对应
专家。模型可直接复用 CUDA BF16 的 `Expert`，通过 `activation_fn` 选择 gate 激活函数
（默认 SiLU，即 SwiGLU），通过 `get_parallel_plan()` 选择要分片的
专家池，再调用专家模块的 `dispatch()`；也可提供自定义 bank。未声明的模型启用
EP 会直接报错。模型写法见[专家模块契约](../development/training-engines.md#fsdp2-ep-bank-contract)。
目前提供单 bank 与双 bank 验证模型，真实 Visual MoE 尚未接入。

设总 rank 数为 W、EP 大小为 P，W 必须能被 P 整除。每个专家 FSDP 坐标只有一个
EP 通信域，需要分片的专家池共用此域。双专家池模型只把 Main 声明为 EP bank；
Block 是普通模块，由模型直接执行本地路由：

| 参数 | 计算视图 | 参数、梯度和 AdamW 状态 |
| --- | --- | --- |
| Main（EP bank） | 本 rank 的连续专家区间 | 相同 EP 坐标之间以 W/P 做 FSDP |
| Block（普通模块） | 每 rank 完整专家池，本地 Top-K | 以 W 做普通 FSDP |
| Router、Shared、Attention | 模型原有本地计算 | 以 W 做普通 FSDP |

Block 的完整池在计算前由 FSDP 收集；训练状态没有常驻 W 份完整副本。Block token
不走 all-to-all，也不重复发送到多个副本。Main 专家数必须整除 P，各 rank 持有
等长连续专家区间。400 个专家可使用 EP=20 或 EP=40，400/32 会报错。
EP 大小不要求是 2 的幂，但必须整除总卡数；40 卡可采用 EP20×专家FSDP2 或
EP40×专家FSDP1。Token 的路由数量仍可不均匀，因此通信使用变长 all-to-all。
专家 FSDP 大小为 1 时只接受 FP32/BF16 梯度归约，避免 PyTorch 2.13 在其他
归约 dtype 下重复应用梯度除数；默认 BF16 计算、FP32 归约可正常使用。

EP 可与 USP 组合。USP 组内共享一份样本，并在模型中切分序列；EP 路由这些本地
token，计算结果回到发送它们的源 rank，之后仍由模型完成序列收集。EP 将普通
FSDP 的 `DP × USP` 域重新划分为 `专家FSDP × EP`，不额外增加总卡数。
例如 8 卡可设置 `parallel.usp_size=2 parallel.ep_size=4`：4 个独立样本组，
每组 2 卡切序列，同时采用 EP4×专家FSDP2。EP 和 USP 大小无需相同。

首期不组合 TP、CP、PP、HSDP 或梯度累积；USP 必须包含在 FSDP 域内。Block 部分
副本、稀疏部署、非整除专家分配、动态迁移与吞吐优化不在此版本范围内。配置字段见
[配置参考](../reference/configuration.md#parallel)。

沿用公共 checkpoint 默认值：首次 `resume.path=none`，结束时异步保存完整 DCP；
完整续训使用 `resume.path=auto`，拓扑兼容性沿用公共 resume contract，严格续训会
校验包括 EP 在内的并行配置。Main 参数和 AdamW 状态以 EP×FSDP 的二维
DTensor 直接交给标准 DCP 保存，保留全局形状和原参数名。预训练权重加载、完整 PT
收集及 DCP-to-PT 转换复用公共入口，无需 EP 专用切片或重组。

CPU gate 覆盖 400/20 与 400/40 布局、39 层声明和同拓扑恢复。B300 4/8 卡的
FP32/BF16 与 activation checkpointing 验收由
`tests/distributed/test_expert_gpu.py` 定义。所有专家 GPU 用例均带 GPU marker，由公共
CI GPU worker 自动执行。当前 EP DTensor 实现已通过 B300 / PyTorch 2.13 的 4 卡
`(EP, USP)=(2,1)/(2,2)/(4,2)/(2,4)` 和 8 卡 `(4,1)/(4,2)` 数值、重算与
同拓扑完整恢复。真实 Visual MoE、
EP=20/40、多机运行及跨拓扑完整续训仍为**未验证**。

## 单机 FSDP2 示例

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.engine=fsdp2 \
  parallel.usp_size=8 \
  parallel.dp_mode=fsdp2 \
  runtime.max_steps=1
```

## 单机 Megatron CP × TP 示例

```bash
torchrun --standalone --nproc_per_node=4 -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  model.pretrained_path=/models/Wan2.1-I2V-14B \
  data.train_manifest=/data/wan21/train.jsonl \
  data.cache_dir=/data/wan21/cache \
  parallel.engine=megatron \
  parallel.dp_mode=ddp \
  parallel.cp_size=2 \
  parallel.tp_size=2 \
  checkpoint.enabled=false \
  runtime.max_steps=1
```

运行镜像需安装 `.[wan,megatron]`。CP 当前要求 self-attention 无 mask、非 causal，
序列长度可被 `2 * cp_size` 整除。

## 模型验收边界

- Wan21 是公开稳定主路径，FSDP2/USP 的用户流程最完整；
- Wan22 的代码组合不等于普通用户稳定支持；
- H3 预览 gate 使用 USP4 + FSDP2；

通用开关见[Override 速查](../reference/configuration-cheatsheet.md)，模型成熟度见
[支持模型](../models/index.md)。
