# 运行测试

vforge 使用 pytest 执行测试。CPU 与 GPU 用例都位于 `tests/`，通过命令行选择运行集合。以下命令从仓库根目录执行，环境准备见[配置开发环境](../environment.md)。

## CPU 测试执行

```bash
# 全量 CPU 测试
uv run --no-sync python -m pytest tests -q

# 定向运行训练模块的 CPU 测试
uv run --no-sync python -m pytest tests/training -q
```

不传 `--gpu` 时，pytest 排除所有带 `gpu` 标签的用例。将第二条命令的目录替换为受影响的测试文件或目录，即可缩小范围。

部分 CPU 测试会启动多进程。容器需要可写临时目录和足够的共享内存；资源分配以 [CPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/cpu.yaml)为准。

## GPU 测试执行 {#gpu-tests}

先进入已分配 CUDA 设备的环境。运行全部 GPU 测试需要满足其中最大卡数需求；CI 的设备分配以 [GPU 机器配置](https://git.woa.com/ieg-infra/vforge/blob/master/scripts/ci/kap/machines/b300.yaml)为准。

```bash
# 全量 GPU 测试
uv run --no-sync python -m pytest tests --gpu -q

# 定向运行 checkpoint 集成测试
uv run --no-sync python -m pytest tests/integration/checkpoint --gpu -q
```

`--gpu` 只选择带 `gpu` 标签的用例，不申请机器、不安装依赖。设备数量不足会使相应用例失败。新增用例如何声明资源，见[GPU 资源声明](writing.md#gpu-resources)。

MR 在 CPU 和文档检查完成后，并行运行 **GPU Core、GPU LTX、GPU Wan** 三个独立
Job，各自展示状态、耗时和日志。每组使用一台八卡 B300，KAP 任务总超时各为
7200 秒。LTX 和 Wan 组收集对应模型及任务目录下的 GPU 用例，
其余 GPU 用例归入 core；全部组通过才算成功。组内仍串行执行。
在已分配 GPU 的环境中可加 `--gpu-group=ltx`（或 `wan`、`core`）单独运行一组；
不传此参数仍运行全部 GPU 用例。

## 测试用例收集与范围确认

```bash
uv run --no-sync python -m pytest tests --collect-only -q
uv run --no-sync python -m pytest tests --gpu --collect-only -q
```

`--collect-only` 列出按路径和标签选中的用例，不执行测试，也不校验运行时 GPU 数量。适合检查新增用例是否被发现、选择范围是否符合预期。

执行时可以将目录进一步缩小为 pytest node ID，例如：

```bash
uv run --no-sync python -m pytest \
  tests/integration/checkpoint/test_hsdp.py::test_hsdp_checkpoint_roundtrip_and_reshard \
  --gpu -q
```

需要在首个失败处停止并显示详细用例名时，可将 `-q` 换成 `-x -vv`。

## 测试结果判读与报告导出

| 输出 | 含义 |
| --- | --- |
| `passed` | 用例执行成功 |
| `failed` / `error` | 断言、初始化、环境或执行失败，需要检查异常 |
| `deselected` | 被路径、CPU/GPU 选择规则或标签条件排除，没有执行 |
| `skipped` | 用例被跳过，可加 `-rs` 查看原因 |

命令成功应以 0 退出。没有选中用例也不是通过：先检查路径和选择条件。CPU/GPU 分组产生的 `deselected` 是正常的，不表示另一组已经得到验证。

需要保存报告时，使用 pytest 自带选项：

```bash
uv run --no-sync python -m pytest tests -q --junitxml=/tmp/vforge-cpu.xml
```

报告用于定位问题和展示结果，CI 的通过条件见[执行结果判定与资源清理](../maintenance/ci.md#result-cleanup)。

测试原则和多卡用例示例见[编写测试](writing.md)。
