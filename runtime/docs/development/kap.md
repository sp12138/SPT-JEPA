# KAP 远程任务

KAP 用于本地和 CPU CI 无法覆盖的 GPU 与多节点测试。低成本检查见
[本地开发与验证](testing.md)。实际执行范围由仓库中的 task template 和 worker 决定。

## 任务组成

- `scripts/ci/kap/machines/`：machine profile，配置集群、镜像、拓扑和挂载；
- `scripts/ci/kap/*.task.yaml`：机器无关的 CPU、文档和 GPU 任务；
- `scripts/ci/bundle.py`：从指定 Git commit 打包 vforge，并写入 source revision、
  target revision 和 changed-file lint 清单；未提交修改不会进入 bundle；
- `scripts/ci/run.py`：组织 MR 验证与文档发布；
- `scripts/ci/kap_run.py`：渲染、提交、轮询、保存 worker 日志和清理 KAP task；
- `scripts/ci/worker.py`：在远端执行检查，并通过进程退出状态报告结果。

## 运行 MR GPU 检查 {#kap-model-gate}

从仓库根目录运行。提交机需要 Python、PyYAML、已授权的 `koala-cli`，以及目标
profile 的集群、镜像和挂载权限。凭据通过环境或 CI settings 注入。

```bash
python scripts/ci/run.py mr \
  --source "$(git rev-parse HEAD)" \
  --target "$(git rev-parse origin/master)" \
  --mr-title '[train] refactor: update training runtime' \
  --phase gpu \
  --gpu-group ltx \
  --run-id "manual-$(date +%Y%m%d%H%M%S)"
```

把标题替换为实际 MR 标题，每次运行使用不同的 `run-id`。该入口自动准备 bundle，
使用 `gpu.task.yaml` 和 `b300` profile 运行指定组；`--gpu-group` 可选 `ltx`、
`wan`、`core`（通用算子与分布式测试）。

正常 MR 流水线在 CPU 和文档检查完成后，并行运行 **GPU Core、GPU LTX、GPU Wan**
三个独立 Job，分别展示状态、耗时和日志。每组分配一台八卡 B300，最多同时占用
24 张 GPU，实际启动时间取决于集群空闲资源。每组 KAP 任务总超时为 7200 秒，
组内用例串行执行，保留各用例自身的超时。所有组成功后 GPU 阶段才通过；某一组
失败时，其余组继续执行。测试集合以 worker 和 pytest marker 为准；通过这一阶段
不表示所有模型和拓扑都已验证。

CPU 阶段使用 `--phase cpu`，也会构建文档预览产物；它需要 profile 对应的对象存储
权限。正常 MR 流水线由 `.ci/ci.yml` 调用这些阶段，无需重复手工提交。

## 定向测试与新增任务

先确认测试是否已包含在当前 GPU worker 中。若需要新的模型或拓扑测试，新增或调整
机器无关的 task template，并复用合适的 machine profile。已有准备好的 bundle 时，
可直接调用 runner：

```bash
python scripts/ci/kap_run.py \
  --config scripts/ci/kap/gpu.task.yaml \
  --machine b300 \
  --bundle /path/to/vforge-bundle \
  --log-dir .ci-results \
  --timeout-min 120
```

Bundle 的目录布局必须满足模板中的工作目录和启动命令。CPU、文档和 GPU 模板使用
bundle 内的 `vforge/`。不要直接用包含未提交文件的工作目录替代 revision bundle。

新增测试应在分配大模型之前校验依赖和输入，失败时返回非零，并注明覆盖的模型、
机器、拓扑、seed、dtype 与未覆盖边界。先用最小配置验证环境，再增加资源规模。

## 如何判断结果与清理资源

`kap_run.py` 等待 KAP task 结束；最终状态为 `Succeeded` 才算执行成功。任务失败、
超时、日志获取或清理阶段的致命错误会使 runner 返回非零。Worker 日志保存在
`--log-dir` 指定的目录，供定位失败步骤。

Runner 在提交后的 `finally` 中保存日志并删除任务。进程被强杀或机器断电时无法保证
清理完成；应按任务名检查残留资源。CI 另有独立 cleanup job，用相同 `run-id` 清理。
不要将个人路径、凭据或一次性环境参数写入通用 profile。

没有运行的模型、硬件和拓扑标记为“未验证”。测试模板存在、任务成功调度或进程退出
成功，都不能代替该任务未覆盖的模型数值与训练行为验证。
