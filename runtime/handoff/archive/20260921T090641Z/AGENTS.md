# vforge Agent Guide

本文件约束在 vforge 仓库中工作的 coding agent。目标是先找到正确的所有权边界，用
最低成本证明改动，再按需升级到真实 GPU 和 KAP。

## 硬规则

1. 新模型、新数据语义和新增 train/eval 能力走 Native：
   `vforge/models/` + `vforge/tasks/` + `vforge/conf/experiment/`。
2. 不新增外部业务框架的 application adapter；模型、任务和 DMD 统一通过 Native 接入。
3. core（`training`、`distributed`、`ops`、`data`、`evaluation`）不得 import
   业务项目。
4. 不复制 Trainer、parallel state、checkpoint manager 或 attention backend。通用
   contract 只有一份实现。
5. Native 通用运行默认以 `vforge/hydra_config.py` 为真源，experiment 只保存模型
   差异。不要在 schema、experiment 或 task 两侧复制默认值。
6. `vforge/mcore` 是带校验清单的 vendored upstream，不做顺手修改。
7. 未运行真实 GPU/KAP 时，结论必须写为“未验证”，不能把代码存在等同于生产支持。
8. 用户可见的 Hydra override 字段必须在命令行解析前进入 compose 配置：公共字段由
   `vforge/hydra_config.py` 经 `native_defaults` 注册，入口 defaults 必须加载它；
   模型公开 extras 在 experiment 中声明，不为公开 override 重复抄写公共默认值。脚本、文档和
   测试使用普通 `key=value`，不得用 `+key=value` 或 `++key=value` 绕过字段声明。
   一次性本地调试确需动态字段时可以临时使用，但不得提交为正式用法。

## 先判断任务属于哪里

| 任务 | 默认位置 | 必读 |
| --- | --- | --- |
| 新模型或模型版本 | `models/<family>/`、`tasks/<family>/`、experiment YAML | `docs/development/adding-native-model.md` |
| 通用训练生命周期 | `training/` | `docs/development/adding-framework-feature.md` |
| 并行与设备拓扑 | `distributed/` | `docs/features/distributed-training.md` |
| Attention/backend | 公共 API 在 `ops/`，实现在 `ops/_attention/` | `docs/development/architecture.md` |
| Meta runtime buffer | `distributed/` 与模型回调 | `docs/development/meta-runtime-buffers.md` |
| 持久 checkpoint/resume | `training/checkpoint/` | `docs/development/checkpoint-internals.md` |
| 单次前反向重算/offload | `checkpointing/` | `docs/features/activation-recompute-and-offload.md` |
| 文档与发布 | `docs-next/`、`mkdocs-next.yml`、`scripts/ci/docs_build.sh` | `docs-next/development/maintenance/documentation.md` |

模型定义不 import 训练层。模型装配、step、optimizer factory 和数据 contract 放在
`tasks/`；`models/` 只拥有模型、encoder 和 inference 语义。

## 最快把任务跑起来

要求 Python 3.10+。按任务安装最小依赖：

```bash
pip install -e '.[dev]'
pre-commit install

# 完整 CPU suite
pip install -e '.[dev,remote,wan]'

# H3 开发
pip install -e '.[dev,h3]'
```

Native 用户入口是：

```text
vforge-prepare
vforge-train
vforge-eval
vforge-convert-checkpoint
```

先用已有 experiment 和极小 `runtime.max_steps` 证明配置与数据路径，不要先复制一个新
Trainer 或模型专属 CLI。多进程使用 `torchrun -m vforge.tasks.<verb>`。

## 最快反馈阶梯

按顺序升级成本；当前层足以证明改动时不要跳到更高层。

### Tier 0：静态检查

```bash
CI_STRICT=1 LINT_SCOPE=changed BASE_REF=origin/master ./scripts/ci/lint.sh
```

### Tier 1：定向 CPU contract

```bash
CI_STRICT=1 \
TEST_PATHS="tests/training/test_contracts.py tests/tasks/wan/v21" \
  ./scripts/ci/test.sh
```

按改动替换 `TEST_PATHS`。模型改动同时覆盖 `tests/models/<family>` 和
`tests/tasks/<family>`；配置改动覆盖 compose 与 translation。

### Tier 2：全量 CPU

```bash
CI_STRICT=1 ./scripts/ci/test.sh
```

更高层的 Pod smoke、模型专属 GPU gate 和 KAP 命令只在
`docs/development/testing.md` 与 `docs/development/kap.md` 维护。不要从脚本名推断
覆盖范围。

## Native 模型清单

新增 Native 模型至少需要：

- `models/<family>/<version>/` 中的 meta-safe 模型与权重映射；
- `tasks/<family>/<version>/` 中的 train/eval/encode builders；
- `vforge/tasks/registry.py` 注册；
- `vforge/conf/experiment/<name>.yaml`；
- raw sample、cache、batch、loss 和 inference 的显式 contract；
- CPU contract tests；
- 模型专属 GPU gate 或清楚的未验证声明；
- 用户指南、配置 reference 和 `docs/models/index.md` 能力状态同步更新。

不要为模型新增 Trainer、checkpoint manager 或全局 parallel state。只有至少两个模型
共享且语义稳定的能力才提升到 core。

## Native 接入边界

模型接入统一使用 Native，外部业务框架不能成为 vforge 运行时依赖；业务 YAML 字段
不得直接进入 core schema。`wan_taichu` 是保留的 Native 模型实现，与外部框架接入无关。

Native activation offload 使用 `ActivationOffloadManager` 与标准 gradient-checkpointing
context。FA 调用选择、内部 benchmark 和 smoke 脚本统一使用这一接口。
文档维护当前接口、模型用法和必要的命令对照；已移除的框架接口不得作为
运行依赖。保留 Wan Taichu 的命令替换、评估启动命令和 Eval 适配状态文档。

## KAP 操作规范

KAP 的稳定抽象是：

```text
machine profile + task template + code bundle
        -> kap_run.py
        -> task status + worker logs
        -> cleanup
```

- machine profile 位于 `scripts/ci/kap/machines/`，拥有 cluster、image、topology、
  mounts 和 code store；
- task template 位于 `scripts/ci/kap/*.task.yaml`，保持机器无关；
- `scripts/ci/bundle.py` 从指定 Git revision 生成 bundle；
- `scripts/ci/kap_run.py` 负责 render、submit、follow、保存日志和 finally cleanup；
- 不按项目、用户或一次性存储绑定复制 machine profile；这类差异使用仓库外 overlay；
- worker 以退出状态报告结果，runner 要求 task 最终状态为 `Succeeded`；
- 凭据只通过环境或 CI settings 注入，永不写入仓库；
- 不确定环境时先用与目标测试相容的最小拓扑验证；
- golden 按 machine 维护，不跨 GPU/云环境比较。

不要把个人路径、密钥、临时 task name、一次性 manifest 或 cluster 细节写进本文件。

## 文档规范

- 面向使用者写“如何完成任务”，面向开发者写 contract 和所有权；
- 使用准确、专业且易懂的词汇，不使用只有少数人理解的黑话、缩写堆砌或含义不清的
  内部说法；必要术语首次出现时用一句话解释；
- 稳定、预览、内部验证、存量兼容必须明确标记；一次性实验不进入主文档；
- `docs/development/index.md` 维护开发者阅读路径，不在各专题重复完整导航；
- 模型完整可复制流程只在对应 `docs/models/<model>.md` One Pager 维护；
- `docs/reference/configuration-cheatsheet.md` 只维护跨模型通用 override；
- schema 字段和用户修改方式只在 `docs/reference/configuration.md` 维护；
- Defaults List、外部配置搜索路径和覆盖顺序只在
  `docs/development/configuration-composition.md` 维护；
- 功能行为在 `docs/features/`，实现 contract 在 `docs/development/`；
- 模型成熟度只在 `docs/models/index.md` 维护；
- 可复制命令使用仓库真实入口，必要参数与默认值对照代码；
- 站内文档使用相对链接，源码使用仓库网页绝对链接；
- 新增、删除或移动站点页面同步 `mkdocs-next.yml`；
- 提交前运行 `./scripts/ci/docs_build.sh`。

Native 统一训练契约是：首次运行 `resume.path=none`；默认启用 final async full DCP，
目录从 `${runtime.output_dir}/checkpoints` 派生并保留最近 2 个；完整续训只增加
`resume.path=auto`；weights-only warm-start 使用显式路径并关闭 optimizer 与
training state；周期容灾只增加 `checkpoint.every_n_steps=<N>`。模型 experiment
不得重复或改写这些公共默认；内部 CPU/config smoke 不应落盘时必须显式设置
`checkpoint.enabled=false`。

## RepWorld 接入约束

- 模型与推理语义位于 `models/rep_world/`；配置、视频读取、frozen codec 装配和 Muon
  位于 `tasks/rep_world/`，复用 Native Trainer、attention、DCP 和 data cursor。
- 不引入原研究仓库、VideoRAE/V-RAE checkout 的运行时 import；published checkpoint
  已含 frozen encoder 权重。保留 codec 的许可证、来源和各自数值语义。
- 区分 codec 原生 state、latent planes 与 RGB ViT temporal patch。VideoRAE 1D 的
  512 个 learned slots 排为两层网格只是索引约定，不代表真实的时间压缩率。
- RGB conditioner 的 rollout 读取解码后的 RGB；latent conditioner 直接读取生成
  latent。不能为了改善闭环效果在迁移时添加归一化、重编码、CFG 或采样策略变更。
- CFG 系数与 Euler 步长保留有维度的广播张量；FP32 state/BF16 velocity 下，零维
  标量会改变 PyTorch dtype promotion。pure-TF 也逐 state 解码，不能让 full-attention
  decoder 提前看到后续生成 state。数值对照须固定两边的 attention backend。
- pure-TF 推理的 RoPE 按当前 autocast 动态计算，包括 offset=0 的首段；不能因
  Native 的预计算表恰好只有一个 state 就复用 FP32 表而改变源模型的 KV prefill。
- 本次迁移保留源主路径的 online train/eval；offline cache、混合数据源与 prepare
  未接入且明确拒绝，不以通用 Native 清单中的 encode 项暗示已支持。
- 旧两塔 checkpoint 只能通过显式 `initial_weights_format=legacy` 作 weights-only
  初始化；不宣称恢复旧 optimizer、训练步数或数据游标。Native full resume 走 DCP。
- 实验诊断、一次性采样报告和历史启动器不进入本模型包。能力与验证范围见
  `docs/models/rep-world.md`，继续工作前核对 `HANDOFF.md` 与实际测试结果。

## 提交前检查

- 改动位置与依赖方向正确；
- changed-file lint 通过；
- 定向测试通过，必要时全量 CPU 通过；
- 用户可见行为已更新文档；
- GPU/KAP 结论包含机器、拓扑、seed、dtype 和未覆盖边界；
- KAP 任务已清理；
- MR 标题符合 `[<scope>] <type>: <title>`。

## 准备 Merge Request

- 不直接从 `master` 提交 MR；从最新 `origin/master` 创建描述单一目标的分支；
- 标题严格使用 `[<scope>] <type>: <title>`，例如
  `[train] feat: expand FSDP2 HSDP support`；
- `scope` 选择主要所有权边界，如 `train`、`model`、`data`、`infra`、`ci`；
- `type` 使用 `feat`、`fix`、`refactor`、`docs`、`test` 或 `chore`，不要用
  `feat` 描述修复；
- 描述分开写背景、变更、验证和未覆盖边界，不把多个问题或方案混在一段；
- 验证项写明 CPU suite、文档构建以及 GPU/KAP 的机器和 topology；未执行的 gate
  明确标记为待 CI 验证；
- push 前确认 diff 不含凭据、个人 overlay、临时 bundle、一次性 task 或调试产物。

## CUDA profiling on B300

- With PyTorch 2.9.x + CUDA 13.0 on B300, the first Kineto profiling session can
  silently omit all CUDA activity. This is tracked by `pytorch/pytorch#169464`;
  the relevant Kineto fix (`pytorch/kineto#1115`) is not included in that build.
- Before the real CPU/CUDA profile, consume one disposable profiling session by
  calling `vforge.training.profiling._warmup_cuda_profiler()`.
- A scheduled profiler warmup does not fix this issue because it remains inside
  the same broken Kineto session.
- Validate every trace before drawing conclusions: `key_averages()` must contain
  CUDA timing columns and the trace must contain CUDA kernel events. Treat a
  CPU-only trace as invalid and rerun after the disposable prewarm.
- Prefer profiling a focused model region or step. Full H3 prepare-sample traces
  create excessive Kineto overhead and can take a long time to finalize.
