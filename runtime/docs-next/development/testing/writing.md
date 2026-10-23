# 编写测试

测试应保护 vforge 的主要行为，并能在行为被破坏时给出明确失败。写用例前先说明：输入是什么、预期是什么、哪种错误应被它发现。

## 测试覆盖范围与断言依据

优先覆盖公共接口、关键状态转换、边界输入和已经修复的真实缺陷。例如：

- 数据字段、形状、dtype 与设备是否符合约定。
- loss、输出或梯度是否落在有依据的误差范围内。
- checkpoint 保存、恢复和重新分片后，状态是否正确。
- 分布式错误是否传播，失败后资源是否释放。

断言需要独立依据，例如小规模手算结果、稳定数学性质、明确的数据契约或可信参考实现。不要把被测实现复制一遍作为预期结果。

仅检查源码字符串、文件名、PASS 日志或函数调用次数，通常不能证明业务行为正确。薄脚本通过静态检查和真实入口运行验证即可，不要求给每个包装脚本、fixture 或测试再编写一套测试。

## 测试目录组织与用例设计

单一模块的逻辑测试放在对应的 `tests/<模块>/` 下；需要多个模块协作的测试放在 `tests/integration/`。共享 fixture 就近放在 `conftest.py`，不要复制训练器、checkpoint 管理器或并行状态实现。

先判断行为是否依赖真实设备：可在 CPU 表达的数据与状态契约使用 CPU；CUDA kernel、设备迁移及 GPU 分布式行为使用 GPU。

每个用例保持小而明确：准备输入、调用真实代码、断言结果。数值用例明确 seed、dtype 和误差容限，临时文件使用 `tmp_path`，避免依赖上一次运行的产物。

## GPU 资源声明 {#gpu-resources}

给用例添加 `gpu(gpus=N)`，声明所需卡数。下面演示单卡测试的结构；实际用例应调用需要保护的 vforge 逻辑：

```python
import pytest
import torch


@pytest.mark.gpu(gpus=1)
def test_gradient_matches_expected():
    value = torch.tensor([2.0], device="cuda", requires_grad=True)
    value.square().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([4.0], device="cuda"))
```

`gpus` 必须是正整数。收集阶段会拒绝无效参数；运行阶段检查可见设备数量，不足时失败。新 GPU 用例由 CI 自动收集，无需增加模型专属脚本或 KAP 配置。

资源声明不会自动扩容。需要超过 CI worker 卡数的测试，应先调整执行资源；测试不根据机器型号静默跳过。设备或必要后端缺失时应暴露失败。

## 多卡分布式测试编写

使用公共 `torchrun` fixture，根据 `gpu(gpus=N)` 启动 N 个 rank。rank 是分布式进程的编号，每个进程执行同一份 worker 代码。

下面是一组通信断言示例，两个文件都放在测试目录中。pytest 入口只负责声明资源和启动：

```python
# test_collective.py
from pathlib import Path

import pytest


@pytest.mark.gpu(gpus=2)
def test_all_reduce(torchrun):
    torchrun(Path(__file__).with_name("collective_worker.py"), timeout=120)
```

worker 中放每个 rank 的逻辑和断言：

```python
# collective_worker.py
import os

import torch
import torch.distributed as dist


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        value = torch.tensor([dist.get_rank() + 1.0], device="cuda")
        dist.all_reduce(value)
        size = dist.get_world_size()
        expected = torch.tensor([size * (size + 1) / 2], device="cuda")
        torch.testing.assert_close(value, expected)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

将示例中的通信操作替换为需要验证的 vforge 行为。按测试需要设置 `timeout`，默认行为见[公共 fixture](https://git.woa.com/ieg-infra/vforge/blob/master/tests/conftest.py)。任一 rank 非零退出使测试失败，超时或中断时清理整个子进程组。

已有项目用例可参考 [MCore 分布式测试](https://git.woa.com/ieg-infra/vforge/blob/master/tests/distributed/test_mcore_data_parallel.py)和 [HSDP checkpoint 测试](https://git.woa.com/ieg-infra/vforge/blob/master/tests/integration/checkpoint/test_hsdp.py)。

用例完成后先定向运行，再检查它是否进入预期的 CPU/GPU 集合。提交说明应写清它保护的行为，便于后续维护者判断是否仍有保留价值。
