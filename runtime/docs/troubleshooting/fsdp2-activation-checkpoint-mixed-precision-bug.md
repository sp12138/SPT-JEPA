# FSDP2 与激活检查点组合使用时的混合精度数据类型不匹配

该问题适用于启用 `parallel.gradient_checkpointing=true` 的 Native 路径，不只存量
adapter。功能配置和术语见
[激活重算与 Offload](../features/activation-recompute-and-offload.md)。

## 问题现象

第一次执行真实的反向传播时，在非重入式激活检查点重新计算期间失败：

```text
torch.utils.checkpoint.CheckpointError:
saved metadata:      dtype=torch.bfloat16
recomputed metadata: dtype=torch.float32
```

触发此问题需要同时满足以下条件：

- 使用 composable FSDP2；
- 启用了带输入转换的 `MixedPrecisionPolicy`；
- 使用激活检查点重新计算。

此问题并非由序列并行或模型损失引起。

## 根本原因

受影响的 PyTorch FSDP2 hook 会在 `TrainingState.PRE_BACKWARD` 状态下提前返回。因此，原始前向传播会应用混合精度输入/输出策略，而重新计算可能会保留 fp32 输入或输出。激活检查点的元数据验证会正确地拒绝这种不一致。

使用 autocast 包裹 `loss.backward()` 无法恢复 FSDP2 的模块边界策略。在模型内部转换激活检查点参数也不是可接受的修复方式：这会硬编码数据类型，并将分布式运行时策略侵入模型代码。

该问题对应 PyTorch issue [`#159359`](https://github.com/pytorch/pytorch/issues/159359)，上游修复为[`#182580`](https://github.com/pytorch/pytorch/pull/182580)。

## vforge 的处理方式

仅当启用了 FSDP2 混合精度且模型正在使用激活检查点时，`parallelize_model()` 才会安装兼容性补丁。随后，`vforge.distributed.fsdp2_patch` 会：

1. 如果上游已具备 `_cast_forward_inputs` 行为，则不执行任何操作；
2. 根据已知的易受影响代码结构校验私有 hook 源码；
3. 将重新计算的输入转换为 `self._mp_policy.param_dtype`；
4. 配置输出数据类型时，将重新计算的输出转换为 `self._mp_policy.output_dtype`；
5. 每个进程只安装一次；
6. 遇到未知 hook 结构时，以安全失败方式终止。

补丁遵循当前生效的策略；调用方不得硬编码 bf16、fp16、fp32 或禁用转换等行为。

## 处置清单

出现此错误时：

1. 确认 `ParallelizeConfig.gradient_checkpointing` 和 FSDP2 混合精度均已启用。
2. 记录 `torch.__version__`，以及完整的 saved/recomputed 数据类型元数据。
3. 保持模型的激活检查点输入不变，并移除本地强制数据类型转换的临时方案。
4. 运行 `tests/distributed/test_fsdp2_patch.py`，以及分布式冒烟测试 `scripts/smoke_fsdp2_activation_checkpoint.py`。
5. 如果 vforge 报告不支持的私有代码结构，请升级到包含 PR #182580 的 PyTorch 构建版本。不要放宽源码校验，也不要修补未知的内部实现。
6. 验证过程必须至少包含一次真实的反向传播和优化器更新；仅执行前向传播或仅计算 loss 的冒烟测试无法触发重新计算。

只有当所有受支持的训练镜像均已包含上游修复，并且移除补丁后分布式激活检查点冒烟测试仍能通过时，才可删除该兼容性补丁。
