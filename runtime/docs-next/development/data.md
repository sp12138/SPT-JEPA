# DataSpec 与多源 JSONL 读取

状态：预览。已完成 CPU 数据读取、采样和恢复验证，以及完整 MMDiT 32B 的
8 卡与 128 卡训练、确定性重放及中点 checkpoint 恢复验证。128 卡实验覆盖 A/B
两条数据路线；在相同配置下，样本序列、模型输入和 loss 的重放比较一致，已完成的
模型及 AdamW 状态比较逐位一致。128 卡训练稳定性验证已通过。

## 调用关系与职责

```text
已有配置解析 → TrainRequest
                  ↓
gen_engine.data.build_data_spec
                  ↓
DataSpec(dataset, sampler_factory, collate_fn)
                  ↓
build_resumable_loader → StatefulDataLoader → Trainer
```

`DataSpec` 保存已经创建好的组件，不负责解析 YAML，也不直接读取样本。任务层根据
已解析的配置创建它，公共 loader 根据当前 DP rank 调用 `sampler_factory`。

多源路径中三个组件各有职责：

| 组件 | 输入 | 输出与职责 |
| --- | --- | --- |
| `MultiSourceSampler` | Source 权重、数据量、版本、种子、总采样量 | 生成 `(source_id, sample_index)`，拥有两层 Shuffle 和 DP 切片 |
| `MultiDataset` | 上述样本地址 | 找到对应的子 Dataset，调用它的 `__getitem__`，原样返回样本 |
| `MMGenDataset` | 一个 Source 的 JSONL 清单及行索引 | 校验该行，读取其预计算 MME/latent 文件，返回模型定义的样本 |

`collate_fn` 负责将读取的样本组装为模型 Batch。任务层使用现有的 `RuntimePacker`；
本次没有修改模型输入格式。

一个 Source 在当前任务接入中对应一个 JSONL；同一条记录中的 `references` 是该样本
的条件数据。Source 的数据量和采样权重独立配置。`len(MultiDataset)` 是所有子数据集
的实际记录数之和，训练采样总量由 sampler 的 `total_samples` 决定。

## 两层 Shuffle

### 第一层：生成 Source 顺序

先将整数权重除以最大公约数。例如 `11:89` 得到长度为 100 的一组配额。
票号 `[0, 11)` 对应 A，`[11, 100)` 对应 B。每组调用：

```python
g = torch.Generator(device="cpu").manual_seed(group_seed)
tickets = torch.randperm(group_size, generator=g, device="cpu", dtype=torch.int64)
```

按票号所属区间映射回 Source，得到随机顺序，但完整组恰好包含 11 个 A 和 89 个 B。
组消费完后，使用下一组的 seed 重新 Shuffle。组的边界独立于 Batch 边界。

采样前固定总样本数 `T`。最后不足一组时，只对剩余名额计算整数配额：先向下取整，
再按小数余数从大到小补齐，余数相同则按 Source 配置顺序决定。随后对这些名额单独
调用 `randperm`。例如 `T=320`，前三组分配 `33:267`，尾部 20 条分配 `2:18`，
最终是 `35:285`。

因此在预定终点，每个 Source 的总次数与理想次数之差严格小于 1 条。中途截断或
单个 Batch 的占比仍会波动。实现使用整数运算，不通过浮点概率决定名额。

### 第二层：生成 Source 内部索引

选中 Source 后，根据它在全局序列中此前累计消费的次数 `c` 和数据量 `N` 计算：

```python
source_epoch, offset = divmod(c, N)
sample_index = source_order[source_epoch][offset]
```

每个 Source 的每轮 `source_order` 由独立的 CPU `torch.randperm(N)` 生成，缓存当前轮，
用完才创建下一轮。一轮中所有索引恰好出现一次，跨轮允许再次消费。

两层均使用独立的 `torch.Generator`。固定 JSON 序列化、SHA-256 和前 8 字节转换只
用于生成 seed；实际排列由 `randperm` 产生。第一层的 seed 绑定权重、倍率、组号及
尾部长度；第二层绑定 Source ID、清单版本、数据量和轮次。给定相同输入和运行环境，
可以重新计算相同地址序列，不依赖 Python hash seed 或 Torch 全局随机状态。

## DP 切分与恢复

采样器先定义唯一的全局样本序列，每个 Global Batch 连续切成各 DP rank 的本地
Batch。例如本地 Batch 为 2、DP 为 2，rank 0 取全局位置 `0,1,4,5,...`，rank 1
取 `2,3,6,7,...`。各 Source 的内部消费次数按全局前缀计算，不能只数本 rank 的样本。
在 `T` 和其他采样参数固定时，改变 DP/Batch size 只改变分配方式；任务入口若固定
`max_steps` 而改变 DP，则会重新推导 `T`，尾部配额也可能改变。

传入自定义 sampler 后，公共 loader 不再额外套 `DistributedSampler`。仍复用现有
`StatefulDataLoader` 的 worker、预取与快照机制，以及现有 leader 广播封装。

采样器迭代器记录下一个位置和恢复契约。恢复时由完整组配额加当前组的已消费前缀
直接重建各 Source 计数，无需扫描全部历史。实际训练应保存和恢复 **loader 的
`state_dict()`**，它同时记录已经交付的 Batch 和 worker 状态；预取时不能用单独
采样器的位置代替训练进度。

对于 map-style Dataset（包括多 Source JSONL），`num_workers>0` 时公共 Loader
还会保存每个 worker 的 PyTorch CPU、Python `random` 和 NumPy 默认 RNG 状态。
实现通过 Dataset 状态包装层接入 torchdata：worker 完成读取及 `collate_fn` 后取快照，
恢复时先加载 Dataset/collate 自身的状态，再恢复 RNG，最后由 torchdata 重建待消费数据。
因此 codec 中的 posterior 采样可以继续使用 `torch.randn_like()`，无需持有 seed。
样本地址一致并不自动保证模型输入一致：posterior 采样还依赖读取进程的 RNG 状态。
训练侧使用的 `RandomStreams` 命名随机流由 Trainer 保存和恢复，不能替代 worker
进程内的 RNG 快照。
支持状态保存的 Dataset 和 collate 对象应同时提供 `state_dict/load_state_dict`。

恢复需要相同的 worker 数、本地 batch size、seed、预取及持久 worker 配置。
旧多 worker checkpoint 没有保存这些 RNG 状态，会明确拒绝恢复；需重新开始训练来生成
新 checkpoint。`num_workers=0` 仍由 Trainer 恢复主进程 RNG，旧行为保持不变。
本机制不保证不同 worker 数之间的输入相同，不支持 worker 内 CUDA RNG；私有 Generator
或其他有状态变换需要由其所属 Dataset/collate 状态接口保存。IterableDataset 仍使用其
自身状态协议，不在这个 map-style 包装范围内。

恢复契约校验数据顺序与版本、权重、总量、种子、算法、运行环境和 DP/Batch 配置。
当前版本要求这些参数保持一致，暂不支持改变总采样量、GBS 或 DP 拓扑后直接加载
旧 checkpoint。组选择只缓存当前组的排列，各 Source 另行缓存各自当前轮的排列。

## 不依赖模型的 CPU 示例

下面先用两个简单 Dataset 观察职责，真实 JSONL 使用相同的路由方式：

```python
from functools import partial
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset

from vforge.data.loader import build_resumable_loader
from vforge.data.multi_source import DatasetSource, MultiDataset, build_multi_source_sampler
from vforge.data.spec import DataSpec

dataset = MultiDataset([
    DatasetSource("A", TensorDataset(torch.arange(100)), "a-v1", 11),
    DatasetSource("B", TensorDataset(torch.arange(200)), "b-v1", 89),
])
spec = DataSpec(
    dataset=dataset,
    sampler_factory=partial(build_multi_source_sampler, total_samples=320, mix_seed=7),
    collate_fn=list,
)
state = SimpleNamespace(
    config=SimpleNamespace(dp_size=1),
    dp_rank=0,
    batch_group=None,
    is_batch_leader=True,
)
loader = build_resumable_loader(
    spec, parallel_state=state, batch_size=32,
    num_workers=0, drop_last=False, seed=42,
)

# sampler 只生成地址；这里的新迭代器不改变 loader 的读取进度。
addresses = list(loader.sampler)
assert len(addresses) == 320
assert sum(source_id == "A" for source_id, _ in addresses) == 35

# loader 根据地址读取样本，连续每 32 条组成一个 Batch。
assert len(list(loader)) == 10
```

## 接入 gen_engine

`gen_engine.data.build_data_spec` 接收已有解析器生成的 `TrainRequest`。单清单继续使用
原有 `MMGenDataset` 及分布式采样路径；多源通过已有 `data.sources` 中的 `name`、
`manifest`、`weight` 字段配置。两个入口互斥。

任务层把十进制权重转换为精确的整数比例，例如 `0.11:0.89` 转为 `11:89`。Source
名称必须唯一且非空，JSONL 的完整内容摘要参与数据版本。多源路径拒绝未支持的
cache 等 Source 字段，避免配置被忽略。

清单及其引用的文件在训练期间应保持不变。声明的工件摘要始终校验；
`require_digests=false` 只允许省略摘要。启用 `pin_artifacts` 时，启动阶段获取所有
消费文件并固定实际 SHA-256 与远端对象身份；格式、音频布局及这些身份也参与 Source
版本与恢复校验。缓存目录不参与数据身份。读取时还校验固定的 JSONL 行内容和工件
字节，不能仅依赖文件时间戳。未固定工件且未声明摘要时，只凭清单版本无法保证跨运行
的文件内容一致。

读取器支持本地文件与 S3 上的裸 Tensor 或 bundle；音频可按明确配置从 `ATC` 转为
模型消费的 `ACT`，原始文件保持不变。公开字段与完整 32B profile 见主站的
`docs/reference/configuration.md` 和 `docs/models/gen-engine.md`。

当前接入固定倍率为 1，`data.mix_seed` 控制 Source 顺序，`data.sampler_seed` 控制
Source 内顺序。当前多源任务按
`runtime.max_steps × 本地 Batch size × DP size` 推导总样本数，要求正数
`max_steps`。该预算针对无梯度累积、每个优化步消费一个本地 Batch 的接入方式。
虽然 Trainer 已有梯度累积能力，这个预算公式不能直接用于累积次数大于 1 的训练；
本文配置与验证均限定梯度累积次数为 1。

通用采样器支持 Batch 内混合多个 Source；当前 gen_engine 的模型 collate 仍要求
本地 Batch size 为 1。在 DP 大于 1 时，全局 Batch 可以包含不同 Source。本地
Batch 为 32 的混合采样由通用 loader 测试覆盖，模型端的多样本打包尚未接入。

为了避免意外分配巨大的排列，规范化配额组最多为 1,000,000。DP 大于 1 时，采样
总量必须能被 Global Batch size 整除，当前实现不会通过重复或丢弃样本凑齐各 rank。

### 三源训练配置

准备三个非空的 `mmgen.sample` v1 JSONL，以及它们引用的预计算 MME、latent 和
reference 工件。每个清单对应一个 Source，所有 Source 使用同一套模型与打包配置。
运行环境需要安装 vforge，并准备好 [gen_engine 子模块](../guide/images.md)。

将下面的用户配置保存为 `/workspace/my-configs/user/gen-engine-multi.yaml`，把
`/data/` 下的路径替换为实际文件路径：

```yaml
# @package _global_
defaults:
  - override /experiment: gen_engine
  - _self_

model:
  extras:
    transformer_config_path: /data/configs/transformer.json
    video_latent_codec_config_path: /data/configs/video-latent-codec.json
    audio_latent_codec_config_path: /data/configs/audio-latent-codec.json

data:
  train_manifest: ""
  sources:
    - name: source_a
      manifest: /data/source-a.jsonl
      weight: 5
    - name: source_b
      manifest: /data/source-b.jsonl
      weight: 3
    - name: source_c
      manifest: /data/source-c.jsonl
      weight: 2
  mix_seed: 7
  sampler_seed: 42
  shuffle: true
  batch_size: 1

runtime:
  max_steps: 10000
  output_dir: outputs/gen-engine-multi
```

Transformer 配置须显式声明 `schema: vforge.gen_engine.dit`、`schema_version: 1`
和 `architecture`；视频与音频 codec 分别使用 `vforge.video_latent_codec` 和
`vforge.audio_latent_codec` schema。

权重填写在 `data.sources` 每个条目的 `weight` 中。上例的抽样比例是
`5:3:2 = 50%:30%:20%`，与各 JSONL 的实际行数无关。权重不必加起来等于 1，
写成 `0.5:0.3:0.2` 也得到同样的配额。权重必须非负，且至少一个大于零；权重为零
的来源不参与抽样，但仍需提供有效的非空清单。

在 DP=1 时，上例共读取 10,000 次，三个来源分别贡献 5,000、3,000、2,000 次。
某个来源的一轮数据用完后会重新洗牌，继续读取。增加第四个或更多来源时，继续向
`sources` 列表添加包含唯一 `name`、`manifest` 和 `weight` 的条目即可。

### 查看配置与启动

先查看合成后的来源列表、权重和模型配置路径：

```bash
vforge-train --config-dir /workspace/my-configs --cfg job \
  +user=gen-engine-multi
```

这里 `+user=gen-engine-multi` 加载已有机制支持的用户配置组。公共 `data` 字段由
`vforge/hydra_config.py` 注册，经公共默认配置加载；用户 YAML 用于组织 Source 列表
和模型路径，不负责注册这些字段。后续字段覆盖使用普通 `key=value`；按索引覆盖
`data.sources.0.weight` 时，需要先配置对应的 Source 条目。

`--cfg job` 只展示 Hydra 合成结果，不读取样本或检查模型配置文件内容。确认文件已
准备好后，启动单卡训练：

```bash
vforge-train --config-dir /workspace/my-configs \
  +user=gen-engine-multi
```

多卡通过同一个训练入口启动，例如在四张可用 GPU 上：

```bash
torchrun --standalone --nproc-per-node=4 -m vforge.tasks.train \
  --config-dir /workspace/my-configs +user=gen-engine-multi
```

默认并行配置下，这时 DP=4、全局总样本数为 `10000 × 1 × 4 = 40000`；三个来源
分别贡献 20,000、12,000、8,000 次。运行时将示例路径替换为实际 JSONL 和模型、
codec 配置路径，并按训练拓扑设置并行参数。

### 修改权重与恢复训练

可以直接修改 YAML，也可以为一次新训练覆盖已有条目的权重：

```bash
vforge-train --config-dir /workspace/my-configs \
  +user=gen-engine-multi data.sources.0.weight=7 \
  runtime.output_dir=outputs/gen-engine-multi-7-3-2
```

`data.sources.0` 指列表中的第一个来源；这里新的比例是 `7:3:2`。输出目录单独保存
这次运行，实际使用的配置保存在该目录的 `resolved_config.yaml` 中。

默认在训练结束时保存 checkpoint；如果需要中途恢复，在启动时增加
`checkpoint.every_n_steps=1000`，每 1,000 步保存一次。保持来源、顺序、权重、种子、
`max_steps` 和 DP/Batch 配置不变时，可以从原输出目录恢复已有 checkpoint。
单卡示例的恢复命令为：

```bash
vforge-train --config-dir /workspace/my-configs \
  +user=gen-engine-multi resume.path=auto
```

该命令使用 YAML 中的原始输出目录与 `5:3:2` 配比。多卡恢复时，保留原来的
`torchrun` 启动方式和进程数，再增加 `resume.path=auto`。

当前实现不支持把新增来源、修改权重或扩大 `max_steps` 当作原采样计划的直接续训；
这些修改需要配置新的训练计划。

## 阅读与验证

按这个顺序阅读实现：

1. `vforge/data/spec.py`：描述数据组件。
2. `vforge/tasks/gen_engine/data.py`：从任务配置创建子 Dataset 和 DataSpec。
3. `vforge/data/multi_source.py`：样本地址、两层 Shuffle、DP 切分和恢复。
4. `vforge/data/loader.py`：把 DataSpec 接入现有 loader。
5. `vforge/tasks/gen_engine/train.py`：训练入口装配与 resume fingerprint。

CPU 定向检查：

```bash
python -m pytest -q tests/data tests/tasks/gen_engine
```

用例覆盖尾部配额、Source 逐轮不重不漏、相同配置重放、全局随机状态隔离、DP
地址重组、带 worker 预取的 loader 恢复，以及临时 JSONL 与预计算文件的实际读取。
这些测试不替代用户真实 JSONL 的读取检查或 GPU 训练验证。
