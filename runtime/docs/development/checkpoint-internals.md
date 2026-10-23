# Checkpoint 与 S3 内部机制

本文面向修改 `vforge/training/checkpoint/` 的开发者。用户配置和行为见
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)。

## 所有权

| 能力 | 所有者 | 说明 |
| --- | --- | --- |
| PT/DCP 读写 | `training/checkpoint/storage.py` | 逻辑 state dict 与物理格式 |
| Manifest/提交 | `training/checkpoint/manifest.py` | Schema、fingerprint、完整性 |
| Resume 计划 | `training/checkpoint/resume.py` | 选择组件和 strict contract |
| Planner staging | `training/checkpoint/planner_staging.py` | Node union、observe、replay |
| 远端保存 | `training/checkpoint/remote_save.py` | 独立 worker 与 S3 上传 |
| Node staging | `utils/node_staging.py` | 远端到本地 SSD |

### 常见维护任务 → 文件/章节

- 修改恢复组件选择、strict contract 或 fingerprint 校验：从
  `training/checkpoint/resume.py` 和 [Trainer resume 状态机](#trainer-resume-state-machine)
  开始；
- 修改异步远端保存、上传 worker 或 S3 提交顺序：看
  `training/checkpoint/remote_save.py` 和 [原子提交](#atomic-commit)；
- 修改同拓扑/跨拓扑的按需下载、planner observe/replay：看
  `training/checkpoint/planner_staging.py` 和
  [Planner-aware staging](#planner-aware-staging)；
- 修改 manifest schema、组件存在性、hash 或提交标记：看
  `training/checkpoint/manifest.py`、[DCP v1 与 v2 目录协议](#dcp-schema-versions)
  和[原子提交](#atomic-commit)；
- 修改 PT/DCP 逻辑 state dict 与物理读写：看
  `training/checkpoint/storage.py` 和
  [DCP v1 与 v2 目录协议](#dcp-schema-versions)。

模型和 task 不拥有第二套 checkpoint manager。

## DCP v1 与 v2 目录协议 {#dcp-schema-versions}

这里的 v1/v2 是 **vforge checkpoint 目录 schema**，不是 PyTorch DCP 版本。Writer 只
生成 v2；reader 继续兼容历史 v1。

| 差异 | v1（历史只读） | v2（当前写入） |
| --- | --- | --- |
| 组件目录 | weights 与 optimizer 共用 `state/` | `weights/`、`optimizer/`、`trainer/` 分离 |
| Trainer sidecar | 位于 checkpoint 根目录 | 位于 `trainer/` |
| Manifest | 不声明组件路径和存在性 | 使用 `components` / `components_present` 显式声明 |
| Fingerprint | 没有 model/optimizer 专属 fingerprint | 记录 model、optimizer、data 和 execution fingerprint |
| Planner-aware staging | 不支持，按完整目录 staging | 可按组件和 payload 选择性 staging |

两种协议都保存逻辑 state dict，并由标准 PyTorch DCP reader 读取。v2 的主要变化不是
模型权重格式，而是把组件布局、存在性和兼容性信息显式写入 manifest，使选择性恢复、
严格校验和远端按需下载可以 fail-fast。

v2 manifest 还记录 schema、completed steps、topology、execution policy、组件文件大小
和 SHA256。

Optimizer DCP 每 rank 最多使用 payload/aux 两类文件：

- 大 tensor 写入 `*_payload.distcp`；
- 标量、BYTE_IO、零长度和小 tensor 写入 `*_aux.distcp`。

该布局仍使用标准 DCP metadata，不引入私有逻辑 schema。它使 planner staging 可以只
下载本节点 payload，同时复制体积较小的跨节点 aux。

## 原子提交 {#atomic-commit}

本地/EFS：

1. 所有 rank 写 `<step>.dcp.incomplete/`；
2. rank 0 汇总组件并写 manifest；
3. 原子 rename 为 `<step>.dcp/`；
4. 更新 `latest`，再执行 retention。

S3：

1. 每节点 rank 将本地 shard 写入 node-local upload root；
2. local rank 0 启动独立 spawn worker，使用 boto3 CRT 上传；
3. global rank 0 汇总文件大小和 SHA256；
4. 最后上传 manifest 作为提交标记，再更新 `latest`。

Worker 不加入训练 NCCL/Gloo group。没有 manifest 的 prefix 不可恢复；重试会清理未提交
prefix，已提交 checkpoint 不允许覆盖。

## 远端存储选型证据

历史 Wan2.2、单机 8×B300、约 319.48 GiB full DCP：

| 路径 | Staging + load |
| --- | ---: |
| S3 Mountpoint direct | 909.6s |
| EFS direct | 257.4s |
| S3 boto3/CRT → SSD | 78.79s |

这些数字只解释当前设计，不是其它节点、bucket 或客户端版本的 SLA。原生 S3 多对象下载
绕过 FUSE，随后仍由标准 DCP reader 读取真实本地文件。

## Planner-aware staging {#planner-aware-staging}

### Same topology

`build_node_preselection()` 根据 manifest world size、ranks-per-node 和物理文件名预选：

- 本节点 ownership 的 weights files；
- 本节点 optimizer payload；
- 所有必要 optimizer aux；
- 本节点 trainer sidecars。

Final `LoadPlan` 在 payload read 前验证预选覆盖关系。

### Cross topology

跨 world size 不能按保存 rank 文件名猜 ownership：

1. 对 weights 和 optimizer 运行 metadata-only planner pass；
2. 在 `StorageReader.read_data()` 边界捕获每 rank final `LoadPlan`；
3. 合并当前节点 ReadItems 对应的文件 union；
4. staging 后由 `ReplayLoadPlanner` 重放 final plan；
5. destination state dict resolve/commit 仍委托给 `DefaultLoadPlanner`。

PyTorch 没有公开“直接注入 final LoadPlan”的 API。当前实现只使用公开 `LoadPlanner`
扩展点，但 replay 语义需要在升级 PyTorch 时重新验证。

### 日志 marker

| Marker | 含义 |
| --- | --- |
| `VFORGE_DCP_STAGE mode=full` | 每节点完整 staging |
| `mode=planner-preselect` | Same-topology 物理预选 |
| `mode=planner-deferred` | 推迟到当前 topology planner |
| `mode=planner-only` | Planner pass 后 staging |
| `VFORGE_PLANNER_ONLY` | 捕获 final plan |
| `VFORGE_PLANNER_REPLAY` | 正式 load 重放 final plan |

缺 marker、ReadItems 改变、planner-only 读取远端 payload 或 selective 文件缺失都应视为
兼容性失败，不得静默回退。

### 验证数据

Wan2.2 4→8 验证：

- 每节点下载 212.96 GiB → 106.51 GiB；
- planner 3.19s；
- staging + full component 关键路径 71.96s → 59.38s。

只证明该 checkpoint、topology 和 PyTorch 2.9.1 路径。

## Trainer resume 状态机 {#trainer-resume-state-machine}

1. 创建 engine，`session.setup()` 解析 `ModelSpec`；
2. `engine.prepare_model()` 应用 parallelize，Trainer 安装可选 offload wrapper；
3. materialize 参数，并在无 full-resume 或 compatible overlay 时加载 pretrained 权重；
4. `prepare_data_parallel()` 后回写 live model；
5. 创建 optimizer、scheduler 和 dataloader，并生成 execution/data contract；
6. DCP full resume：stage 目录、读取 manifest、校验 fingerprint，再按 plan 加载 model、
   optimizer 和当前 rank sidecar；
7. PT warm-start：只加载模型权重，step 从 0 开始；
8. full resume 原子恢复 scheduler、dataloader 和 RNG，再从 manifest
   `completed_steps` 进入唯一训练循环。

RNG 最后恢复，避免模型、optimizer 和 iterator 构造消耗随机流。
Trainer 将同一个 `RandomStreams` 交给 `StepContext` 和 `CheckpointManager`。
保存只调用 `random_streams.state_dict()`，恢复只调用 `random_streams.load_state_dict()`；
这两个方法同时处理全局 RNG 和显式 generator，无需独立的 capture/restore 函数。
rank-local `rng` 除 Python、NumPy、torch CPU/CUDA 状态外，还保存 `streams` 中
各命名 generator 的设备类型和状态；CUDA generator 恢复到当前 rank 的设备。
task 首次取 generator 时播种，后续一直复用，不从 step 推导 seed。
CPU checkpoint/resume contract 已通过；B300 上命名 generator 和全局 CUDA RNG 的
状态恢复已通过逐位比较。完整模型的 GPU checkpoint/resume 尚**未验证**。

Native `StatefulDataLoader` 直接保存/恢复 state dict。Leader-broadcast loader 只有 group
leader 持有真实 cursor；follower 保存 owner=false stub。Full DCP 不支持普通 DataLoader
的 replay+skip 回退。

`MultiStreamStatefulDataLoader` 独立保存各数据流的 epoch 和 cursor。它的外层迭代器
持续循环，不向 Trainer 暴露单一 sampler；恢复时先设置各 sampler 的已保存 epoch，
再加载各 loader 的 state dict，避免外层 epoch 覆盖已经恢复的采样状态。

## 权重 dtype 如何写出 {#weight-storage-dtype}

DCP 按参数当前的存储 dtype 原样写出：`_keep_in_fp32_modules` 命中的参数是 fp32，其余按
模型的全局存储 dtype。Wan 与 H3 的全局存储 dtype 都是 fp32，所以两者的 DCP 全是 fp32，
`_keep_in_fp32_modules` 对它们只决定 FSDP 豁免范围，不改变落盘 dtype。

portable PT weights-only 走另一套规则，把所有浮点统一成 fp32，与存储 dtype 无关。两条
路径目前对 Wan 和 H3 的结果一致；全局存储 dtype 低于 fp32 的模型会在这里被升到 fp32，
DCP 与 PT 因此可能不一致。

## PyTorch 升级门禁

升级 DCP 相关 PyTorch 版本至少验证：

1. Planner/storage-reader callback 顺序和签名；
2. `LoadPlan`、`ReadItem` 和 storage data 的可重放语义；
3. Same-world 与 cross-world 的 per-rank ReadItems；
4. weights+Adam 逐 tensor 一致；
5. 完整保存、远端 staging、恢复一步和新 DCP 提交。

定向测试：

```bash
pytest -q \
  tests/checkpoint/test_checkpoint.py \
  tests/checkpoint/test_planner_staging.py \
  tests/checkpoint/test_resume.py
```

用户可见行为变化时，同时更新功能页、配置 reference 和任务模板。

## 下一步

先按[本地开发与验证](testing.md)运行 checkpoint 定向测试；涉及真实 S3、跨节点或
大模型保存恢复时，再进入 [KAP](kap.md)。用户行为只在
[Checkpoint 与 S3](../features/checkpoint-and-storage.md)维护。
