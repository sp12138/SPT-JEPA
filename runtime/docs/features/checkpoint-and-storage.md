# Checkpoint 与 S3

本文集中说明用户侧持久 checkpoint 的保存、恢复和远端存储行为。字段默认值见
[配置方式与字段](../reference/configuration.md)，可复制命令见
[模型 One Pager](../models/index.md)，通用组合见
[Override 速查](../reference/configuration-cheatsheet.md)。

Activation checkpoint 和 preserved-kernel offload 只影响一次 forward/backward，与
这里的持久恢复不同。显存优化见
[激活重算与 Offload](activation-recompute-and-offload.md)。

!!! tip "S3 路径会自动启用远端优化"
    `storage.s3_mounts` 不只是路径别名。Checkpoint 路径命中映射后会自动使用
    node-local staging、异步 DCP 提交和 boto3/CRT 并发上传。Planner-aware selective
    staging 不会随 mount 自动开启，需在正确性验证后显式 opt-in。

```text
写入 <step>.dcp.incomplete/
  → 生成并校验 manifest
  → 原子提交为 <step>.dcp/
  → 本地直接可恢复；命中 S3 mount 时异步上传远端
```

## 先选择用途

| 目标 | 推荐格式 | 恢复内容 |
| --- | --- | --- |
| 评估或交付权重 | `weights + pt` | 模型权重 |
| 跨 topology warm-start | `weights + dcp` | 模型权重 |
| 同任务故障恢复 | `full + dcp` | 模型、optimizer 和训练状态 |
| Megatron 故障恢复 | `full + mcore` | MCore 模型、distributed optimizer 和训练状态 |

`full + pt` 不支持。PT 不能恢复 optimizer、step、scheduler、dataloader 或 RNG。
Weights-only PT 中的浮点权重统一保存为 FP32；BF16 mixed precision 只影响 forward
计算，不改变 portable checkpoint 的存储精度。

文件名中的 step 是**已完成的 optimizer step 数**。`00060.dcp` 表示前 60 步已完成，
恢复后从 step 60 继续；`runtime.max_steps` 始终是整次训练的目标总步数。

## 保存

公共训练契约默认启用 final full DCP：

```yaml
checkpoint:
  enabled: true
  every_n_steps: 0
  dirpath: ""
  state: full
  format: dcp
  save_final: true
  keep_last: 2
  async_save: true
```

- 空 `dirpath` 解析为 `${runtime.output_dir}/checkpoints`；
- `every_n_steps=0` 关闭周期保存，但不影响 `save_final`；
- 长任务设置 `checkpoint.every_n_steps=<N>` 增加周期容灾；
- `save_optimizer` 和 `save_trainer_state` 控制 full DCP 中的可选组件；
- `async_save=true` 支持 Torch DCP 和 MCore native；
- `keep_last` 只清理同格式且已提交的 checkpoint。

完整字段见 [`checkpoint`](../reference/configuration.md#checkpoint)。

### DCP 目录

当前保存采用 **DCP v2 目录协议**。这里的 v1/v2 指 vforge 的 checkpoint 目录 schema，
不是模型版本、PyTorch DCP 版本或用户配置项。新任务始终写 v2；恢复时仍可读取历史
v1 目录。两代协议的组件布局和兼容差异见
[DCP v1 与 v2 目录协议](../development/checkpoint-internals.md#dcp-schema-versions)。

```text
00060.dcp/
  manifest.json
  weights/
  optimizer/             # save_optimizer=true
  trainer/rank-00000.pt  # save_trainer_state=true
```

保存先写 `<step>.dcp.incomplete/`，所有组件成功后才生成 manifest 并原子提交。自动恢复
不会选择 `.incomplete` 目录。

### MCore native 目录

`checkpoint.format=mcore` 只允许 `engine=megatron`。模型侧由 vForge
递归桥接普通容器，并让 MCore TP module 描述 TP shard；optimizer 侧直接使用
`DistributedOptimizer.sharded_state_dict` 的 `dp_reshardable` schema，保存 FP32
master parameter、Adam moments、param-group metadata、step 与 grad scaler。vForge
不重新实现 optimizer bucket range，也不把它转换成 Torch optimizer state。

```text
00060.mcore/
  manifest.json
  mcore/                 # MCore native model + optimizer payload
  trainer/rank-00000.pt  # 可选的 rank-local sidecar
```

同步保存和 NVRx 异步保存使用相同目录协议；native payload 完成后才写 manifest 并原子
提交。恢复时 CP、TP、PP 必须保持不变，DP 可以由 `dp_reshardable` 重分片。改变 DP
时必须关闭 `resume.training_state`，因为 scheduler、dataloader cursor 和 RNG 是
rank-local sidecar。第一版不支持 MCore remote upload、跨 engine optimizer 恢复或
HF 导出。

## 选择恢复目标

`resume.path` 支持：

| 值 | 行为 |
| --- | --- |
| `auto` / `latest` | 从 `checkpoint.dirpath` 选择 step 最大的完整 checkpoint |
| `none` / `off` / `false` / `no` / 空值 | 禁止恢复 |
| 显式路径 | 加载指定 PT 或带合法 manifest 的 DCP/MCore 目录；不存在时失败 |

公共默认是 `resume.path=none`，首次训练不需要额外 override。同一步同时存在 PT 和
DCP 时，显式或自动选择优先 DCP。

## 选择恢复内容

```yaml
resume:
  path: auto
  weights: true
  optimizer: true
  training_state: true
  strict: null
```

依赖关系：

- optimizer 或 training state 依赖 weights；
- training state 依赖 optimizer；
- `training_state` 原子恢复 completed step、scheduler、dataloader 和 RNG；
- PT 或 weights-only DCP 不能满足 training state 请求；
- `strict=null` 时，完整续训默认 strict，warm-start 默认 non-strict。

跨 world size 默认只允许 weights-only warm-start。Weights+optimizer 重新分片要求
`training_state=false` 且 effective strict 为 false。

HSDP 的 `dp_replicate_size` 和 `dp_shard_size` 都进入 topology fingerprint。Full DCP
strict resume 要求 split 不变；weights-only 或上述 non-strict warm-start 可以在普通
FSDP 与 HSDP、以及不同 HSDP split 之间 reshard，但不承诺 bitwise 轨迹。

`runtime.weights_strict` 是 train/eval 统一的权重结构策略，控制 pretrained、
weights-only warm-start 和评估 checkpoint 的 key/shape 匹配；声明式 weight source
本身不再携带另一份 strict 开关。它不替代 `resume.strict`，后者控制 optimizer、
进度、dataloader、RNG 和拓扑 fingerprint 等完整训练状态能否严格恢复。

## S3 Mount 路由与异步优化

VForge 只根据显式 `storage.s3_mounts` 判断对象存储身份：

```yaml
storage:
  s3_mounts:
    - mount_path: /mnt/training
      bucket: training-checkpoints
      region: us-west-2

checkpoint:
  dirpath: /mnt/training/user/run/checkpoints
```

未命中映射的路径按本地/EFS 文件系统处理。系统不会从 mount 名、环境变量或路径猜
bucket。

`checkpoint.dirpath` 命中映射时，DCP 保存会自动切换为异步本地 staging + 远端上传，
即使 YAML 中 `checkpoint.async_save=false`。这是远端提交协议的一部分，不表示训练退出
时可以跳过等待最终 commit。

生产推荐路径是：

```text
S3 持久对象
  → node leader 使用 boto3/CRT staging
  → node-local SSD
  → 标准 PyTorch DCP reader
```

Mountpoint 适合统一路径、枚举 manifest 和 metadata；大对象 payload 不建议由 DCP
reader 直接随机读取 mount。

### 远端 staging

路径命中 S3 映射时自动 staging。未命中时，`resume.stage_remote=true` 只启用普通
filesystem staging，不能触发 bucket 推导。

常用环境变量：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `VFORGE_CKPT_LOCAL_ROOT` | `/local-ssd/vforge-checkpoints` | 本地 staging 根目录 |
| `VFORGE_STAGE_WORKERS` | `8` | 节点并发对象数 |
| `VFORGE_S3_TRANSFER_CONCURRENCY` | `32` | 单对象 CRT 并发 |

并发默认值不是所有机器的固定最优值。

## Planner-aware 按需下载优化

默认行为是**每个节点完整 staging 一份 DCP**。选择性 staging 已实现，但需显式开启：

```bash
export VFORGE_DCP_PLANNER_SELECTIVE=1
```

启用后：

- 同 topology 根据保存布局预选本节点 weights/optimizer payload；
- 跨 world size 先运行 metadata-only planner pass，再按当前 topology 的 node union
  下载文件；
- schema、optimizer 布局、plan 或覆盖关系异常会 fail-fast，不静默回退。

该路径要求 schema v2；optimizer 需要 payload/aux 布局。跨 world size 还受上一节的
warm-start 条件约束。

只观察 plan、不改变 I/O：

```bash
export VFORGE_DCP_PLANNER_OBSERVE=1
export VFORGE_DCP_PLANNER_OBSERVE_DIR=/runs/planner-observe
```

详细算法、日志 marker、兼容性边界和 benchmark 见
[Checkpoint 与 S3 内部机制](../development/checkpoint-internals.md)。

## 严格恢复保证

同 topology 的 full DCP strict 恢复会校验：

- Python/PyTorch、world size 和 parallel topology；
- seed、precision、autocast 和 recompute policy；
- model、optimizer、scheduler、ops 和数据 fingerprint；
- 请求组件、manifest 和完整性信息；
- dataloader 类型、cursor 和 resume contract。

这些校验是正确 continuation 的必要条件，不承诺所有 distributed kernel 和 collective
端到端逐位确定。Activation offload 只改变内存和传输调度，因此不参与 strict
resume 校验。

## 常见选择

- **第一次训练**：使用公共默认，不设置 `resume.*`；
- **同任务续训**：full DCP + `resume.path=auto`；
- **换 topology 或换模型版本**：weights-only warm-start；
- **评估/交付**：PT，或将 weights DCP 转 PT；
- **S3 多节点恢复**：先用默认 full staging 建立正确性，再单独验证 selective opt-in。

故障入口见[故障排查](../troubleshooting/index.md)。实现、实验数据和开发者测试见
[Checkpoint 与 S3 内部机制](../development/checkpoint-internals.md)。
