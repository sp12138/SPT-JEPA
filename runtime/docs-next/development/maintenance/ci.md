# 持续集成设计

本文面向修改 CI 的维护者，说明执行顺序、脚本职责和资源生命周期。提交者需要的命名规范与本地检查见[提交代码](../contributing.md)；测试命令见[运行测试](../testing/running.md)。

## 执行顺序与资源复用

MR 流水线的触发条件和执行步骤由 [`.ci/ci.yml`](https://git.woa.com/ieg-infra/vforge/blob/master/.ci/ci.yml)定义。

```text
蓝盾检出与编排
  │
  ├─ CPU checks：申请一个 CPU worker
  │    命名检查 → 增量 lint → CPU 测试 → 文档构建
  │    全部通过后释放 CPU
  │
  ├─ 文档 debug：CI 将 CPU 构建的文档覆盖到 /debug/
  │
  ├─ GPU tests：申请一个 GPU worker
  │    pytest 执行 GPU 用例 → 释放 GPU
  │
  └─ Finally：独立执行资源清理
```

前一步失败即停止，后续步骤不会启动。CPU、文档 debug、GPU 在蓝盾中显示为独立步骤。CPU 与 GPU 各自复用一个 worker；文档 debug 复用 CPU 阶段的产物，由 CI 主机部署，不额外申请机器。

编排环境以流水线配置为准，远端资源分别由 [CPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/cpu.yaml)和 [GPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/b300.yaml)维护。

## CI 脚本职责划分

CI 脚本位于 [`scripts/ci/`](https://git.woa.com/ieg-infra/vforge/tree/master/scripts/ci)，调用方向为 **蓝盾 → 编排 → KAP runner → worker → 检查工具 / pytest**。

- **编排层**确定待验证的源码，准备检查元数据，安排 CPU、GPU 和发布顺序。
- **KAP runner**负责提交、等待和清理远端资源，并传播任务失败。
- **worker**初始化环境，按顺序调用检查工具和 pytest。
- **检查工具与测试**拥有具体校验逻辑，通过退出状态报告结果。

任务模板只声明工作目录与 worker 入口。机器配置维护 cluster、资源、镜像和挂载。修改测试逻辑应进入 `tests/`，不让编排代码识别模型、解释 loss 或解析 PASS 文本。

## 源码分发与运行环境准备 {#source-environment}

CI 验证源分支的固定提交。编排层从 Git 提交生成源码包，附上 MR 标题、源分支新增 commit 和改动文件清单；包中不携带 `.git` 或凭据。

KAP runner 提交源码包，由平台将其放入 worker 的工作目录。

worker 先检查源码的 `uv.lock` 与镜像中保存的锁文件一致，再离线安装当前 vforge 源码。第三方依赖来自镜像。镜像如何选择与更新，统一见[配置开发环境](../environment.md)。

增量 Ruff 检查使用源提交与 merge base 之间的 Python 改动清单。worker 根据清单执行，不需要连接 Git 服务；同时执行语法、vendored MCore 完整性及导入检查。

## 执行结果判定与资源清理 {#result-cleanup}

检查命令失败会使 worker 非零退出；KAP runner 要求任务终态为 `Succeeded`。任务失败、等待超时或清理失败，都会使流水线失败。日志与 JUnit 用于展示和诊断。

任务名称由阶段和蓝盾 build ID 确定。runner 在正常结束、失败或中断时清理任务；蓝盾 Finally job 使用相同 build ID 再执行清理，不依赖上一步留下的本地文件。

排查时先定位蓝盾失败步骤，再查看 worker 日志、具体测试异常和 KAP 终态。Finally 失败需要单独处理，避免遗留资源持续占用配额。

## CI 变更约束

- 增加检查：在相应 worker 阶段安排顺序，前一步失败必须阻断后续步骤。
- 调整卡数或挂载：修改机器配置；不要把集群信息写进测试。
- 更新 Python 依赖：先更新锁文件和镜像，再切换机器配置中的 digest。
- 修改凭据：通过蓝盾 settings 注入，源码包和测试文件不得保存凭据。

master 的文档发布由独立流水线处理，复用 CPU worker 和源码打包能力，不重复执行 MR 测试。构建、传输和部署配置见[构建与发布文档](documentation.md)。

## 陈旧分支扫描（预览）

[`.ci/stale-branches.yml`](https://git.woa.com/ieg-infra/vforge/blob/master/.ci/stale-branches.yml)
定义独立的 Stream 任务 `vforge-stale-branches-preview`，每天按 `0 10 * * *`
执行，也支持手动运行。定时配置由默认分支上的 YAML 决定；需要先启用项目的 Stream CI，
再将配置合入 master。`always: true` 保证仓库没有新提交时也扫描。

任务默认筛选 60 天以上的分支，可修改 YAML 中的 `STALE_DAYS`。扫描使用 Stream
为当前 job 提供的临时 `ci.token`，通过环境变量 `GONGFENG_TOKEN` 传入脚本，
不需要在仓库保存个人令牌。使用公共 Docker 构建环境，不申请 GPU 或 KAP worker。

扫描排除默认分支、保护分支、`KEEP_PATTERNS` 中的长期分支，以及 opened/reopened
MR 在当前项目内的源和目标分支。分支保护状态不明、缺少创建时间或提交时间时也跳过。
日期取分支创建时间、可用的更新时间和 HEAD 提交时间中的最新值，避免把从旧提交新建
的分支当成过期分支。这些字段不等同于完整的最后推送记录，报告仅供人工核对。

任务日志显示候选清单，工作目录的 `.ci-output/stale-branches/` 写入 `report.md`
和包含全部分支及排除原因的 `report.json`。这两个文件暂不上传为构建产物。
脚本只有 GET 请求，不发送通知、不删除分支。API 失败会使任务失败，不会把查询失败
显示为“无候选分支”。通知、保留申请、24 小时等待和删除前复查尚未实现。

本地验证：

```bash
python -m pytest tests/infra/test_stale_branches.py
```
