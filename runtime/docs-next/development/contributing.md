# 提交代码

一个 Merge Request（MR，合并请求）应围绕一个明确目标，让审查者能够理解改动原因、行为变化和验证方式。提交前先[准备开发环境](environment.md)。

## 分支管理与提交规范

从最新 `origin/master` 创建工作分支。下面以修复 checkpoint 问题为例，分支名称应改为实际任务名称：

```bash
git fetch --no-tags origin master
git switch -c fix-checkpoint origin/master
```

MR 标题和源分支新增的**每一个 commit 首行**都使用同一格式：

```text
[scope] type: description

[train] fix: restore checkpoint state correctly
[ci] refactor: simplify test execution
```

scope 选择改动的主要所属模块；type 表达改动性质，例如缺陷修复使用 `fix`。标题应描述具体行为，允许值和校验规则以 [mr_check.py](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/mr_check.py) 为准。

检查范围是目标分支没有、源分支新增的提交。目标分支已有提交和平台生成的临时合并提交不参与检查；自己创建的 merge commit 仍需符合格式。推送前应整理 `fixup!`、`squash!` 等临时命名。

## 提交前检查

先运行成本低且与改动直接相关的检查，再根据影响范围扩展测试。

### 1. 代码静态检查

```bash
git fetch --no-tags origin master
CI_STRICT=1 LINT_SCOPE=changed BASE_REF=origin/master \
  uv run --no-sync bash scripts/ci/lint.sh
```

该入口检查改动 Python 文件的 Ruff 规则和格式，并检查项目语法、vendored MCore 完整性及可用环境中的导入路径。本地会纳入未提交改动和新文件；命令应以 0 退出。

### 2. 相关模块回归测试

从相关模块的定向测试开始。涉及公共数据、训练或分布式逻辑时，扩大到受影响的测试集合；涉及 CUDA 或多卡行为时，在 GPU 环境补充验证。

具体命令与结果判读统一见[运行测试](testing/running.md)。新增测试前参照[编写测试](testing/writing.md)，用断言说明需要保护的行为。

### 3. 文档构建与变更审查

用户可见行为变化时同步修改对应文档。文档改动执行：

```bash
uv run --no-sync bash scripts/ci/docs_build.sh
git diff --check
git diff --stat
```

检查待提交文件中没有凭据、本地产物或无关修改。依赖变化需要同时提交锁文件，并按[镜像构建流程](environment.md#custom-image)更新环境。

## 创建合并请求

推送工作分支，在工蜂创建目标为 `master` 的 MR。描述应包含：

- **背景**：用户在什么场景下遇到什么问题。
- **变更**：修改后的行为和主要实现选择。
- **验证**：执行了哪些命令，覆盖哪些输入和运行环境。
- **未覆盖项**：还需要哪些验证；GPU 行为未实际运行时明确标注。

GPU 验证说明实际卡数、拓扑和相关精度设置；运行记录留在 MR 或测试报告中。使用文档维护稳定的操作方式和约束。

## CI 结果检查与代码审查

创建或更新 MR 后，查看关联的 CI 检查。蓝盾依次执行 CPU 检查、文档 debug 和 GPU 测试；前一步失败时后续步骤停止。触发条件以[流水线配置](https://git.woa.com/ieg-infra/vforge/blob/master/.ci/ci.yml)为准。

「文档 debug」成功后，可在[预览站点](https://ieg-infra-vforge.devcloud.woa.com/debug/)审查页面效果。所有 MR 共用此地址，只保留最后一次部署的内容，详见[MR 文档预览](maintenance/documentation.md#mr-preview)。

失败后查看对应步骤日志，修复并提交到同一分支。命名失败时同时检查 MR 标题与新增 commit 标题；测试失败时可按[运行测试](testing/running.md)中的定向命令复现。

完成审查并合入后，master 的文档发布由独立流水线处理。CI 内部职责和资源管理见[持续集成设计](maintenance/ci.md)。
