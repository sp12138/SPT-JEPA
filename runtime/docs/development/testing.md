# 本地开发与验证

验证按成本逐级增加。先运行能证明当前改动的最小检查，只有低成本门禁通过后才占用
GPU 或提交 KAP 任务。

## 开始前

所有命令都从 Git 仓库根目录运行，而不是从 `docs/`、`scripts/` 或安装目录运行：

```bash
cd "$(git rev-parse --show-toplevel)"
test -f pyproject.toml
test -x scripts/ci/lint.sh
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
```

项目要求 Python 3.10 以上；当前 CI 使用 Python 3.11。只跑静态检查可安装：

```bash
python3 -m pip install -e '.[dev]'
pre-commit install
```

完整 CPU 测试会导入远端存储和 Wan 依赖，应安装：

```bash
python3 -m pip install -e '.[dev,remote,wan]'
```

H3 定向测试改用 `.[dev,h3]`。不要因为一个模型的依赖而要求所有开发环境安装所有
extra。

changed-scope lint 需要真实且可达的目标分支引用。首次运行或长时间未同步时先执行：

```bash
git fetch --no-tags origin master
git rev-parse --verify origin/master
git merge-base origin/master HEAD
```

三条命令都应以 0 退出。若仓库没有 `origin`、无法拉取 `master`，或没有 merge base，
先修复 clone/remote/分支历史；不能把 `BASE_REF` 指向任意旧提交来换取绿色结果。

## 反馈阶梯

### 1. 静态检查

**前提**：位于仓库根目录，`.[dev]` 已安装，且上面的 `origin/master` 校验通过。

**运行**：

```bash
CI_STRICT=1 LINT_SCOPE=changed BASE_REF=origin/master ./scripts/ci/lint.sh
```

**预期**：命令以 0 退出，最后打印 `[lint] PASS`。这表示改动涉及的 Python 文件通过
ruff check/format，项目 Python 通过 compileall，vendored MCore 完整性检查通过；若
当前环境有 torch，import smoke 也通过。

**失败或尚未证明**：`[lint] FAIL` 或非零退出表示至少一项静态契约失败，应按前面的
具体报错修复。它不运行 pytest，也不检查 MR 标题和 MkDocs，因此成功不代表 `unit`
或完整 CI 通过。

### 2. 定向 CPU 测试

**前提**：已安装目标测试所需 extra；完整 Wan 测试环境使用 `.[dev,remote,wan]`。

**运行**：

```bash
CI_STRICT=1 \
TEST_PATHS="tests/training/test_contracts.py tests/tasks/wan/v21" \
  ./scripts/ci/test.sh
```

把 `TEST_PATHS` 换成改动对应的真实测试路径。模型改动至少覆盖对应 `tests/models/`
与 `tests/tasks/`；配置改动至少覆盖 Hydra compose 和 translation contract。

**预期**：pytest 选中的用例全部通过，脚本以 0 退出并打印 `[test] PASS`；`-rs`
输出会列出仍被跳过的用例及原因。

**失败或尚未证明**：`[test] FAIL`、pytest failure/error，或严格模式下缺少 pytest/
torch，均表示定向门禁未通过。成功只证明 `TEST_PATHS` 中实际收集并执行的契约，不
证明全量测试，也不证明 GPU 路径。

### 3. 全量 CPU 测试

**前提**：已安装 `.[dev,remote,wan]`。容器需为 DataLoader 和真实多进程 Gloo case
提供足够的 `/dev/shm`；Docker 环境建议至少 1 GiB。

**运行**：

```bash
CI_STRICT=1 ./scripts/ci/test.sh
```

**预期**：完整 `tests/` 套件结束，脚本以 0 退出并打印 `[test] PASS`。GPU-gated
用例在 CPU 主机上可能跳过，必须检查 `-rs` 的原因。

**失败或尚未证明**：非零退出表示 CPU suite、依赖或本地多进程环境有问题；共享内存
不足常表现为 DataLoader bus error，不应当作随机 flake 忽略。成功不执行 CUDA kernel，
不证明 GPU 数值、显存或多机拓扑。

### 4. 已准备 Pod 中的多卡 smoke

**前提**：这一步只在已经准备好的训练 Pod 内运行，不是本地 torch 门禁。Pod 应有
项目 Python 环境、测试依赖、`torchrun`、4 张可见 GPU，以及脚本所需的数据/挂载和
共享内存；进入挂载的 vforge 仓库根目录后再运行。

**运行**：

```bash
bash scripts/cloud_regression_standalone.sh
```

**预期**：脚本以 0 退出，最后打印成功 marker
`[standalone-regression] done`。这表示 Native import、指定 CPU 回归和 4-rank
Ulysses runtime smoke 均已走完。

**失败或尚未证明**：未出现 marker 或命令非零退出，都表示 Pod 环境或 Native
基础回归未通过，以停止处的命令为准。成功不代表所有模型完成 GPU 验收，也不证明
模型专属数值、保存恢复和产物契约。

### 5. 远程 GPU gate

**前提**：前四层通过，而且改动确实需要真实 GPU/多机拓扑或模型数值证据；KAP
客户端、配置、凭据和 bundle 前提见 [KAP 远程任务](kap.md)。

**运行**：按 KAP 页面通过 `scripts/ci/run.py mr --phase gpu --gpu-group ltx` 准备
bundle 并运行指定组；另外两组为 `wan` 和 `core`。MR 流水线并行运行全部三组。

**预期**：runner 以 0 退出，远端任务状态为 `Succeeded`。

**失败或尚未证明**：runner 非零退出、超时或任务状态非 `Succeeded`，都算失败。一个 gate 成功只证明其模板、机器 profile 和
固定验收范围；没有运行的模型、机器和拓扑必须明确写“未验证”。

## 蓝盾 CI

`.ci/ci.yml` 是唯一 MR 流水线，按顺序调用 `scripts/ci/run.py mr` 的 CPU、
文档 debug 和 GPU 阶段。实际测试范围以仓库中的 runner 和 task template 为准。

流水线先检出 MR 源提交，再初始化该提交锁定的 Gen Engine `dit`、`audio_vae` 和
`video_vae` 子模块。CPU 与 GPU 阶段都通过 `--include-submodule` 将它们打入
KAP 代码包；worker 根据 bundle metadata 要求 DiT 存在，缺失时测试失败而非跳过。

流水线通过不代表所有 Native 模型的用户闭环。手工 KAP 用于新增测试和现有回归无法覆盖的模型/拓扑验证。

## 文档检查

文档构建、预览与发布统一见[构建与发布文档](../../docs-next/development/maintenance/documentation.md)。

## 完成标准

- 改动所属层级和依赖方向正确；
- 最小定向测试与受影响的上一级门禁通过；
- 新配置有 compose/translation 测试；
- 新模型走 Native，并具备 prepare/train/eval contract；
- 用户可见行为同步更新指南或 reference；
- GPU/KAP 未运行时明确写为“未验证”，不从代码存在推导生产承诺。
