# Meta 模块的运行时缓冲区契约

原生模型可以在 `torch.device("meta")` 上构造。Trainer 先调用
`engine.prepare_model()` 执行 TP、activation checkpoint 和 FSDP shard 布局，再按需
安装 activation offload wrapper，最后调用 `engine.materialize_model()`。各 engine
复用 `materialize_meta_module()`，由它调用 `to_empty(device)` 分配存储空间。
`to_empty()` 不会生成缓冲区的实际数值。

完整 materialize 顺序见[训练引擎](training-engines.md)。

## 适用条件

只有同时满足以下条件时，模块才需要实现本页接口：

1. 模型会在 meta 设备上构造并经过 `materialize_meta_module()`；
2. 模块在 meta 上注册了 `persistent=False` 的 tensor buffer；
3. 该 buffer 的值不会从 `state_dict()` 加载，且 materialize 后需要按配置和形状确定性重建。

不满足这些条件就无需实现 `reset_non_persistent_buffers()`。例如模型不走 meta 构造、
buffer 会由持久 checkpoint 恢复，或该值在首次使用前由别的明确生命周期创建，都不属于
本接口。后一种情况不要让一个无需重建的 meta non-persistent tensor buffer 进入
materialize 边界；当前实现会把它视为必须重建并 fail-fast。

## 必须实现的契约

满足上述三个条件时，该缓冲区所属的模块必须实现：

```python
def reset_non_persistent_buffers(self) -> None:
    """在当前 buffer 的 device/dtype 上确定性重建运行时数据。"""
```

`materialize_meta_module()` 会在调用 `to_empty()` 前收集这些回调，并在存储空间分配完成后逐个调用。如果发现 meta 非持久化缓冲区但未找到回调，则会立即报错。由于这些缓冲区不在 `state_dict()` 中，后续加载预训练权重或恢复训练都无法修复它们。

回调必须满足以下要求：

- 仅依赖稳定的配置和形状，不读取尚未加载的参数值；
- 在缓冲区当前的设备和数据类型上生成确定性内容；
- 不注册新参数，也不改变 `state_dict()` 的键；
- 不创建进程组，也不执行跨 rank 通信；
- 可重复生成相同结果，不依赖未初始化的存储内容。

建议让构造逻辑和重置逻辑复用同一个纯构建函数：

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cos, sin = self._build_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def reset_non_persistent_buffers(self) -> None:
        device, dtype = self.cos.device, self.cos.dtype
        with torch.device("cpu"):
            cos, sin = self._build_tables()
        self.cos = cos.to(device=device, dtype=dtype)
        self.sin = sin.to(device=device, dtype=dtype)
```

不要在调用 `torch.empty()` 后假定其中的内容有效，也不要将可派生数据改为 `persistent=True`；后者会改变对外的 checkpoint 契约。

## 验证门禁

新增模型至少需要验证：

1. 在 meta 设备上构造后，参数和派生缓冲区确实位于 meta 设备；
2. 实体化后，参数和缓冲区均不再位于 meta 设备；
3. 运行时缓冲区的关键值与直接在 CPU 上构造的结果一致，不能只检查形状；
4. 运行时缓冲区不出现在 `state_dict()` 中；
5. 缺少回调时，`materialize_meta_module()` 会报错；
6. 使用固定输入时，首个使用该缓冲区的 block 与参考路径输出一致。

Wan21 的 RoPE 表曾在 `to_empty()` 后保持未初始化状态，导致首层 attention 出现偏差；参数覆盖率检查无法发现此类错误。通用测试位于 `tests/distributed/test_parallelize_materialize.py`。

## 下一步

新增模型继续进入[新增 Native 模型](adding-native-model.md)；修改 materialize 或
wrapper 顺序时，先运行上述定向测试，再按[本地开发与验证](testing.md)补覆盖。
