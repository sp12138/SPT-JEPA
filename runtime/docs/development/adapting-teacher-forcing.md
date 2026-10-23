# 在已有模型上适配 Teacher Forcing

本指南面向已经有一份稳定 Native 插件的开发者：不需要再复制一套 encode / eval /
权重加载，只把训练 step 改成 block-causal teacher forcing。新模型仍先走
[新增 Native 模型](adding-native-model.md)；参考实现是 Wan 2.2 TI2V 5B 的 `ct` → `teacher_forcing` 变体，
运行状态见 [Wan 2.2 内部验证](../models/wan22.md)。

Teacher forcing 在 vforge 里**不是** YAML 开关，而是一个只改训练前向的模型变体：
新的 `model.name`、同一份 cache、同一份发布权重、同一个 eval sampler。

## 只需要改什么

1. **模型侧**：新增 `teacher_forcing` transformer 子类，只替换训练 self-attention、
   构造块因果 sparsity 并丢掉 clean 输出半边；不新增或改名权重模块。
2. **任务侧**：新增训练 condition processor，把基座 recipe 已生成的 clean/noisy
   输入和 per-token timestep 组装成 `[clean | noisy]`；不复制 dataset、cache、
   noising、target、loss、encode 或 eval。
3. **接线与验证**：注册新的 `model.name`，增加 experiment 的两个 mask 超参，并补
   CPU contract；真实 SM100 + FA4 精度仍需独立 KAP gate，不能由代码存在推导。

## 开始前检查

先确认基座插件已经能独立 prepare/train/eval，且其 CPU contract 通过。适配前还要
逐项确认：

- 基座 transformer 的 token flatten 是 frame-major，或能提供与实际布局一致的
  token→frame 映射；
- 基座 checkpoint 能按相同参数名和 shape 严格加载到计划中的子类；TF 超参不会写入
  Diffusers config 或 checkpoint；
- TF 与基座共用 cache schema、噪声、target、loss 权重和 eval sampler；
- 训练配置可以关闭 context parallel / Ring Attention；目标 GPU 路径具备 SM100 和
  FA4，CPU dense SDPA 只用于小序列 contract。

在 TI2V 参考实现上，可从仓库根目录检查基座和目标测试路径：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
test -f vforge/models/wan/v22/ti2v_5b/ct/transformer.py
test -f vforge/tasks/wan/v22/ti2v_5b/ct/train.py
test -f vforge/tasks/wan/v22/ti2v_5b/encode.py
test -f vforge/tasks/wan/v22/ti2v_5b/eval.py
test -f tests/tasks/wan/v22/ti2v_5b/teacher_forcing/test_tf_contracts.py
```

任一检查失败都表示参考树或工作目录与本文不一致，应先修正基座，不要开始复制
teacher-forcing 变体。

本文把适配前、self-attention 对完整 token 序列全可见的原始 Native model/task
实现称为**基座插件**。Wan 2.2 TI2V 示例的基座 transformer 位于
`vforge/models/wan/v22/ti2v_5b/ct/`；下文的“基座 transformer”“基座训练
recipe”“基座 encode/eval”都指这套适配前的原实现。

## 方案：加倍序列 + 块因果 mask

Transformer 在训练时看到的是沿帧轴加倍的 `[clean | noisy]` 序列。Clean 半边是
ground-truth 上下文；noisy 半边是与基座训练 recipe **同一套噪声** 加出来的 denoising
输入。Self-attention 只允许：

| Query | 允许看到的 Key |
| --- | --- |
| clean 帧 | 本块及更早的 **clean** 帧（块因果） |
| noisy 帧 | **本块 noisy**，以及 **严格更早块的 clean** |

因此 noisy 帧既读不到自己的 clean 答案，也读不到未来。Cross-attention（文本等）
保持原样，不走这张 mask。

因果块由两个训练超参决定，它们描述训练方案而不是权重，**不得**写入 Diffusers
config 或 checkpoint：

- `frame_per_block`：每块多少 latent 帧。越大，同块 noisy token 互相看见得越多，
  离它要监督的自回归 student 越远；
- `first_frame_own_block`：`true` 时第 0 帧独占 block 0（适合首帧 in-paint）；
  `false` 时按 `f // frame_per_block` 均匀分块。

Forward 将实际 token 网格和上述块布局组成 `TeacherForcingSparsity`。其中
`num_latent_frames` 与 `tokens_per_frame` 定义半边序列长度和 token→frame 映射，
`frame_per_block` 与 `first_frame_own_block` 定义 frame→block 映射。它是完整块因果
mask 的结构化描述；`vforge.ops.attention` 据此生成 FA4 sparse metadata，SDPA
fallback 也由同一份 spec 生成语义一致的 dense mask。

`frame_per_block=1`、三帧时的允许关系（`✓` 可见，`.` 不可见）：

```text
              clean0  clean1  clean2 | noisy0  noisy1  noisy2
clean0          ✓       .       .    |   .       .       .
clean1          ✓       ✓       .    |   .       .       .
clean2          ✓       ✓       ✓    |   .       .       .
noisy0          .       .       .    |   ✓       .       .
noisy1          ✓       .       .    |   .       ✓       .
noisy2          ✓       ✓       .    |   .       .       ✓
```

输出头只保留 noisy 半边，所以模型仍从 `2T` 帧输入预测 `T` 帧；target 与
loss 权重的 shape 与基座训练 step 相同。Clean 半边的 per-token timestep 必须是 `0`：
它是从未加噪的上下文，不能再被当成 denoising 条件。

## 所有权：mask 在 ops，变体只拥有 train step

| 层 | 负责 |
| --- | --- |
| `vforge.ops` | `TeacherForcingSparsity`、FA4 tile metadata、SDPA dense fallback |
| `vforge.models.<plugin>/teacher_forcing` | 子类 transformer：换 self-attn processor、铺 RoPE、构造 spec、丢掉 clean 半边 |
| `vforge.tasks.<plugin>/teacher_forcing` | 加倍 `hidden_states`、干净半边 timestep=0、读 extras、注册新 `model.name` |
| encode / eval / cache | 基座插件继续拥有 |

Sparse mask 的生产路径需要 FA4 和 SM100（Blackwell）。没有 FA4 时 ops 会把 spec
物化成 dense SDPA mask，但序列超过 8192 token 会直接拒绝——那正是这份 spec 要避开
的 `O((2S)²)` 显存。CPU contract 用小序列 + SDPA；真实训练不要指望 fallback。

Teacher-forcing sparse 已接入 FA4 算子内 preserved-kernel offload，并使用 forward
KV 方向和 backward 转置 Q 方向的两套 block-sparse metadata；该组合目前是 preview，
真实 B300 数值与性能 gate 尚未完成。FA3 仍明确不支持 sparse preservation。见
[激活重算与 Offload](../features/activation-recompute-and-offload.md)。

Context parallel / Ring Attention 与 `sparsity=` 目前不能组合：公开 `attention()` 在
`cp_enabled` 时不会把 spec 传下去。TF experiment 必须把对应 limitation 关掉。

## 适配步骤（对照 TI2V）

参考树：

```text
vforge/models/<family>/<version>/
  ct/                         # 已有基座 transformer（全序列 self-attention）
  teacher_forcing/
    __init__.py               # 只写文档字符串
    transformer.py            # 子类：加倍前向 + TF processor
    model.py                  # build_empty，把 extras 传进 transformer
    inference.py              # 可选：只放 [B, 2S] timestep 布局

vforge/tasks/<family>/<version>/
  shared.py / encode.py / eval.py   # 两个变体共用，不要复制
  ct/train.py                       # 已有基座训练 step
  teacher_forcing/train.py          # 只改组装与模型构造

vforge/conf/experiment/<name>_tf.yaml
vforge/tasks/registry.py            # 新 model.name；encode/eval 复用基座插件
tests/tasks/.../teacher_forcing/test_tf_contracts.py
```

只有训练 step 不同的两个变体共用一层，与
[新增 Native 模型](adding-native-model.md) 的推荐结构一致。

### 1. 子类 transformer，不改权重模块

继承基座 transformer，保留 patch embedding、block、RoPE 表和输出头的模块名与权重
shape。`frame_per_block` / `first_frame_own_block` 作为普通 `__init__` 参数留下，
不要 `@register_to_config`。构造后给每个 block 的 self-attention 换上 TF attention
processor；TI2V 对应 `WanTeacherForcingSelfAttentionProcessor`。它是挂在 attention
模块上的计算适配器，通过 `block.attn1.set_processor(...)` 安装；它复用原 attention
模块的 QKV projection、norm 和输出 projection，不新增权重，只改变 self-attention
如何调用 `vforge.ops.attention`。

当前 `TeacherForcingSparsity` 按 frame-major 布局解释 token：一帧有
`tokens_per_frame` 个 token，帧 `f` 对应
`[f * tokens_per_frame, (f + 1) * tokens_per_frame)`，并通过
`(i % S) // tokens_per_frame` 反推帧号。基座模型若不是这种 flatten 顺序，不能直接
复用这份 spec，需要先让 token→frame 映射与实际布局一致。

Forward 相对基座父类只做这些事：

1. 断言帧轴为偶数，`T = 2T_in / 2`；拒绝调用方再传入 `attention_mask`。
2. **RoPE 只对半边计算再 tile**：`rope(hidden[:, :, :T])` 后
   `cat([cos, cos], dim=1)`。同一 latent 帧的 clean / noisy token 必须落在同一
   位置，否则 mask 按帧比较、位置却在说另一帧。
3. Patch embed 整段 `2T`；timestep 必须是 `[B, 2S]`，再交给原来的
   condition embedder。
4. 构造 `TeacherForcingSparsity(num_latent_frames, tokens_per_frame, ...)`，在
   进入 trunk **之前**写到每个 self-attn processor 上。
5. Ulysses：切 `2S`，但**禁止 padding**。Spec 和 FA4 tile metadata 按一个精确
   序列长度编译；pad 会让 kernel 计算一张 metadata 没描述的 mask。`2S` 不能整除
   `usp_size` 时 fail-fast。
6. 不要调用会 `pad_to_multiple` 的 `prepare_ulysses_inputs`。
7. Gather 回 `2S` **之后**再丢掉 clean 半边。Shard 边界与 clean/noisy 分界不必
   对齐（`usp_size=2` 时 rank 0 会拿到整段 clean），先切半边会取错 token。
8. `temb` 若按 token 对齐，也在 gather 之后再切 noisy 半边；不要对它做 Ulysses
   slice。
9. 输出头与 `unpatchify` 仍按 `T` 帧，与基座模型相同。

这个 self-attention processor 的 `__call__` 读取 transformer forward 写入的
`TeacherForcingSparsity`，依次执行 QKV projection、QK norm、RoPE，再调用：

```python
vforge_attention(query, key, value, ulysses=True, sparsity=self._sparsity)
```

不要传 `attn_mask` 或 `is_causal`。Ulysses all-to-all 会把完整 `2S` 还到每个
rank（切的是 head），所以 spec 始终按全局序列写。

TI2V 参考：

- `vforge/models/wan/v22/ti2v_5b/teacher_forcing/transformer.py`
- `vforge/models/wan/v22/ti2v_5b/teacher_forcing/model.py`

### 2. 在训练 step 中构造 Teacher Forcing 的模型输入

这里的“构造模型输入”是指生成传给 transformer 的 `hidden_states` 和 `timestep`，
不是修改 dataset、collate 或 cache。TF 训练继续复用基座 recipe 生成的噪声、sigma、
target 和 loss 权重，并保证同一 seed 下逐位相同；只把模型输入改为
`[clean | noisy]` 和对应的 per-token timestep。

将这两部分拆成可覆盖方法，由 task 侧的 condition processor（TI2V 对应
`_TfConditionProcessor`）override：

- `_assemble_hidden_states`：clean = ground truth（含模型自己的首帧 in-paint
  等条件），noisy = 已加噪 latent；两半拼接同一组 condition stream，再沿**帧轴**
  `cat`。
- `_assemble_timestep`：返回 `[B, 2S]`。Clean 半边全 `0`；noisy 半边复用基座
  布局（TI2V 的 noisy 第 0 帧也是 `0`，因为它是 in-paint 参考而不是预测）。

训练循环、collate、codec、dropout、pretrained widener 全部复用基座插件
`build_train_components` 的零件。Loss 仍是
`mean(loss_weights * (output - targets) ** 2)`，因为 output 已经是 `T` 帧。

TI2V 参考：`vforge/tasks/wan/v22/ti2v_5b/teacher_forcing/train.py`，其中
`_TfConditionProcessor` 覆盖上述两个方法，step 仍调用 `_wan22_train_step`。

模型若没有首帧 in-paint：`first_frame_own_block=false`，noisy 半边不必把第 0 帧
timestep 钉成 `0`，loss 也不必把第 0 帧置零。这些是 TI2V 条件契约，不是 mask
契约。

### 3. 注册新名字，encode 复用基座插件，eval 使用 causal builder

```python
"wan22_ti2v_5b_tf": ModelTasks(
    train="...teacher_forcing.train:build_train_components",
    eval="...teacher_forcing.eval:build_eval_components",
    encode="...ti2v_5b.encode:build_encode_components",
),
```

新 experiment YAML 复制基座实验，改 `model.name`，并增加：

```yaml
model:
  extras:
    frame_per_block: 3
    first_frame_own_block: true   # 仅当第 0 帧是给定参考时
evaluation:
  rollout_mode: full_recompute
```

Prepare 仍用基座插件的 `model.name` 写 cache，因为 TF 训练读取相同的 encoded 字段
和 cache schema。TF 模型与基座模型的参数名和 shape 相同，因此训练可以直接用基座
checkpoint 初始化。

Eval 继续复用 WanTaichu 包内的 `build_base_eval_components`、encoder、cache codec
和权重来源，并由 causal workflow 传入专用 pipeline/settings。`full_recompute`
rollout 先提交首帧 `z0`，再按 causal block 生成：每个 block 从对应的初始噪声出发走完
全部 scheduler step，完成后才写入 clean history。每次 forward 都构造完整的
`[generated clean history | current noisy block]`，两半拼相同 condition stream；clean
半 timestep 全零，noisy 半只有 active block 使用当前 scheduler timestep。模型输出只
取 active block 更新 scheduler。CFG 的 cond/uncond forward 分别使用命名 cache
context。

Validation 的目标视频只用于确定输出尺寸，在线 encoder 不产生 `frame_latents`，cache
collate 也会在进入 sampler 前丢弃它，因此 ground truth 不会进入 history。输出 metadata
记录实际 rollout mode 和 block layout。`incremental_kv` 每次只执行当前 block，按绝对
RoPE slot 临时覆盖 K/V，block 完成后以 timestep `0` context re-pass 提交 clean K/V；
CFG 的 cond/uncond 分支持有完全独立的 cache。它当前要求 temporal patch size `p_t=1`
和 `usp_size=1`，不满足时显式报错，不能静默回退。

`full_recompute` 复用训练用 TF transformer，所以 USP 仍要求 doubled token sequence
可被 `usp_size` 整除，FA4 仍使用 `TeacherForcingSparsity` 的 sparse metadata。
`incremental_kv` 走固定容量 cache 与 dense attention。CPU tiny fixture 会比较两条路径
的 block 输出和最终 latent；真实长序列显存、吞吐及 GPU BF16 误差仍需模型 gate 验证。

增量执行的中立边界位于 `vforge.models.causal`。`IncrementalKVExecutor` 只负责
`IncrementalKVTransformer` protocol、`p_t=1` / `usp_size=1` 门禁、固定容量 cache
分配和完整 clip RoPE；Wan 与 Wan Taichu 各自保留
`forward_incremental_kv_block`，继续明确拥有 patch layout、timestep embedding、
condition 解包、transformer trunk 和 output projection。Executor 不是 `nn.Module`，
不会把 evaluation engine 已持有的 transformer 参数再次注册。

该边界新增的公开/运行时变量必须按以下含义传递：

- `IncrementalKVExecutor.transformer`：engine 已拥有的具体 transformer 引用；
- `IncrementalKVExecutor.block_forward`：唯一的模型专属单 block forward；
- `IncrementalKVExecutor(..., usp_size=...)`：可选的显式 USP 拓扑；未传时读取当前
  parallel runtime；
- `IncrementalKVGuidanceCaches.conditional` / `.unconditional`：一个 rollout 生命周期
  内的 cond cache 与可选 uncond cache；`use_cfg` 决定是否分配后者，二者禁止同一对象；
- `allocate_incremental_kv_guidance_caches` 的 `batch_size` / `num_cache_tokens` /
  `device` / `dtype`：两个分支共用的 cache geometry、设备和初始 dtype；
- `IncrementalKVAttentionContext.cache_layer` / `.write_start` / `.commit`：每层当前
  cache、绝对 token 写入起点和 clean-history 提交开关；
- `forward_kv_block` 的 `kv_cache` / `full_rotary_emb` / `kv_write_start` /
  `commit_kv`：通过 evaluation engine 路由到模型单 block forward，不进入配置或权重。

只把 CFG cache 的分配和隔离生命周期提取为
`allocate_incremental_kv_guidance_caches`。Scheduler loop、block layout、per-token
timestep、condition channel 拼接、engine cache context，以及 timestep `0` 的
clean-history re-pass 仍在 Wan 各自的 pipeline 中显式表达；把这些行为藏进 callback
集合会模糊模型协议，因此本阶段不抽取通用 rollout controller。

## GPU 精度验证范围

静态检查、CPU 测试和 MR 流水线按[本地开发与验证](testing.md)执行。真实 GPU 精度
需要先新增模型专属的 KAP loss-regression gate；任务结构和提交方式见
[KAP 模型 gate](kap.md#kap-model-gate)，新增模板的要求见
[KAP 远程任务](kap.md)。仓库当前没有 teacher-forcing 专属 task template，不要
引用一个并不存在的 gate 名称。

TF 精度 gate 应固定代码与依赖版本、基座 checkpoint、cache、seed、训练步数、
execution/precision、attention backend 和并行拓扑；在生产所需的 SM100 + FA4 路径上
重复运行并测量波动后，按机器分别记录短程 loss golden 和 tolerance。B300 与 GB300
不能共用 baseline。任务必须产出 `verdict.txt` 和摘要，稳定后再接入 CI 的
`gpu-gates` stage。

当前 Wan 2.2 TF **没有**模型专属 KAP 精度 gate，因此仍属于 GPU 精度未验证状态。
模型无关的 HSDP 验证不能作为 TF 的验证证据；历史 A14B adapter loss 记录也不覆盖
Native TF，旧 adapter 回归入口已移除。

## 参考实现

| 角色 | 路径 |
| --- | --- |
| 块因果谓词与 FA4 metadata | `vforge/ops/_attention/teacher_forcing.py` |
| 公开入口 `attention(..., sparsity=)` | `vforge.ops.attention` |
| TI2V TF transformer | `vforge/models/wan/v22/ti2v_5b/teacher_forcing/` |
| TI2V TF train step | `vforge/tasks/wan/v22/ti2v_5b/teacher_forcing/train.py` |
| Experiment | `vforge/conf/experiment/wan22_ti2v_5b_tf.yaml` |
| Registry | `vforge/tasks/registry.py` 中 `wan22_ti2v_5b_tf` |
| CPU contract | `tests/tasks/wan/v22/ti2v_5b/teacher_forcing/test_tf_contracts.py` |

完成后更新对应[模型 One Pager](../models/index.md) 和 experiment 表，写清 cache
仍由基座插件 prepare 生成、训练从基座 checkpoint 初始化。

## 完成前的 CPU 定向验证

**前提**：从含 `.git` 的仓库根目录运行，使用 Python 3.10 以上，并安装完整 Wan
测试依赖：

```bash
python3 -m pip install -e '.[dev,remote,wan]'
```

**运行**：

```bash
CI_STRICT=1 \
TEST_PATHS="tests/tasks/wan/v22/ti2v_5b/teacher_forcing" \
  ./scripts/ci/test.sh
```

**预期**：命令以 0 退出并打印 `[test] PASS`。这组测试证明小序列 CPU SDPA 路径下：

- clean/noisy 加倍输入、clean timestep=0、target/loss shape 和基座 noising 一致；
- 块因果 mask 阻止 noisy token 读取本块 clean 答案，同时允许读取更早 clean 块；
- transformer 只返回 noisy 半边，基座权重可严格加载；
- Ulysses 切分不 padding，非法序列长度 fail-fast；
- 真实训练 step 产出有限 loss，registry 继续把 encode/eval 路由到基座插件。

**失败或尚未证明**：`[test] FAIL`、pytest failure/error 或依赖缺失均表示 CPU contract
未通过，不能继续申请 GPU。即使成功，也不证明 FA4 sparse kernel、SM100 数值、
长序列显存、真实 checkpoint/cache、多卡/多机训练或 loss baseline。当前这些 GPU
结论仍是“未验证”；要改变该状态，必须按上节要求新增并实际运行 TF 专属 KAP gate，
取得 `kap_run.py` exit code 0 和对象存储 `verdict=PASS`，不能复用现有 A14B/HSDP
gate 的结果。
