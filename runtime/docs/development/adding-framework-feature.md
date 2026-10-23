# 新增框架能力

本页面向修改 vForge 公共生命周期或跨模型机制的开发者。只被一个模型使用的结构、
数据或 loss 语义应留在模型插件；新增模型本身见
[新增 Native 模型](adding-native-model.md)。

完成一个框架能力通常意味着：

1. 找到并扩展已有 contract；
2. 把配置、运行时对象和执行位置接通；
3. 在分配大模型或初始化通信前拒绝非法组合；
4. 用 CPU contract 固定语义，需要真实硬件时再增加 GPU gate；
5. 更新用户行为和开发者实现文档。

不要增加第二个 Trainer、checkpoint manager、parallel state 或模型专属公共入口。

## 1. 找到扩展入口

先按能力选择已有接口，不要从目录名反推一套新生命周期：

| 要修改的能力 | 主要接口或文件 | 执行位置 |
| --- | --- | --- |
| Native 训练 recipe | `TrainComponents`、`TrainingSession` | `vforge/training/recipe.py` |
| 训练引擎 | `TrainingEngine` | `vforge/training/engine/base.py` |
| 并行与 FSDP policy | `ParallelConfig`、`ParallelizeConfig` | `vforge/distributed/` |
| Attention kernel 与 dispatch | `vforge.ops.attention`、`varlen_attention` | `vforge/ops/_attention/` |
| Forward/backward 内重算或 offload | checkpoint wrapper、saved-tensor hooks | `vforge/checkpointing/` |
| 持久 checkpoint 与恢复 | manifest、storage、resume contract | `vforge/training/checkpoint/` |
| Hydra 字段 | `*HydraConfig` + `tasks.translate` | `vforge/hydra_config.py`、`vforge/tasks/translate.py` |

完整代码归属见[架构与所有权](architecture.md)。持久 checkpoint 与 activation
checkpoint 是两套机制，不能因为名称相同而放进同一个包。

### 新增训练引擎

实现 `TrainingEngine` 的五个方法：

| 方法 | 必须保证的结果 |
| --- | --- |
| `prepare_model()` | 在 optimizer 创建前应用引擎拥有的模型改造 |
| `materialize_model()` | 分片后分配真实存储，并恢复后端所需 metadata |
| `prepare_data_parallel()` | 权重加载后施加复制式 data parallel wrapper |
| `configure_optimizers()` | 构造引擎专属 optimizer，或返回 `None` 使用 recipe factory |
| `run_step()` | 完成一次不可分割的 forward、backward 和 update |

引擎选择、共享与分叉原则见[训练引擎](training-engines.md)。不要让模型 import
Megatron/MCore，也不要让 Trainer 根据模型名称选择分支。

### 新增 Attention backend

当前 attention 没有独立的第三方注册 Protocol。新增 backend 需要同时修改：

1. `vforge/ops/_attention/` 下的 backend 实现和 capability；
2. `dispatch.py` 的选择与 fail-fast；
3. 公开 `vforge.ops.attention` / `varlen_attention` 路径；
4. dense、varlen、mask、deterministic、fallback 和目标 topology 的定向测试。

跨 rank 的通信实现放在 `vforge/distributed/`，模型仍只调用公共 ops。已有
block-causal 训练复用 `TeacherForcingSparsity`；模型适配见
[在已有模型上适配 Teacher Forcing](adapting-teacher-forcing.md)。

### 修改持久状态

先定义 schema 版本、组件存在性、fingerprint、旧版本读取和 crash consistency，再修改
I/O。按改动选择 `manifest.py`、`storage.py`、`resume.py` 或
`planner_staging.py`，具体入口见
[Checkpoint 与 S3 内部机制](checkpoint-internals.md)。

## 2. 接通配置与运行时

新增用户字段时按固定链路实现：

```text
hydra_config.py
  → tasks.translate
  → runtime dataclass / contract
  → 唯一消费位置
  → resolved_config.yaml 与 execution fingerprint
```

Hydra schema 只声明字段、类型和基础默认值，不 import runtime 模块。跨字段校验由
`tasks.translate` 构造运行时对象时完成。字段加载顺序和测试方式见
[配置加载与覆盖](configuration-composition.md)。

实现前明确：

- 能力作用于 build、parallelize、materialize、load、step、save 还是 cleanup；
- 能力是否需要模型侧适配；模型差异放在 task，不能反向成为 core 依赖；
- 哪些非法组合可以在模型分配和 process-group 初始化前拒绝；
- 哪些字段必须进入 resolved config、checkpoint fingerprint 或 eval manifest。

## 3. 完整示例：增加模型级 FSDP 精度边界

现有 `FSDPPrecisionContract` 展示了一项公共能力如何贯穿各层：

1. `vforge/distributed/parallelize.py` 定义运行时 contract 和执行语义；
2. `MixedPrecisionHydraConfig` 提供同名配置孪生字段；
3. `tasks.translate` 把 Hydra 值转换为 `MixedPrecisionConfig`；
4. 模型通过 `get_fsdp_precision_contract()` 声明自身例外，不感知 Hydra；
5. `parallelize_model()` 在 FSDP group 建立前合并模型默认值与配置覆盖；
6. `tests/distributed/test_parallelize_materialize.py` 覆盖正常路径、优先级和非法路径；
7. 用户语义记录在[随机性与精度](../features/execution-and-precision.md#model-precision-boundaries)，
   字段格式记录在[配置方式与字段](../reference/configuration.md)。

这个例子中的关键点不是 precision 本身，而是同一事实只有一个运行时 contract：
Hydra、模型声明、执行实现和 checkpoint fingerprint 都引用它，不通过隐式属性或 import
side effect 传递状态。

## 4. 验证

先运行受影响模块的最小 CPU contract，再逐级扩大：

```bash
CI_STRICT=1 \
TEST_PATHS="<受影响的 tests/... 路径>" \
  ./scripts/ci/test.sh

CI_STRICT=1 LINT_SCOPE=changed BASE_REF=origin/master \
  ./scripts/ci/lint.sh
```

通过表示接口、成功路径和 fail-fast 行为已在当前 CPU 环境成立，不代表 CUDA kernel、
真实 topology、吞吐或数值基线已验证。后者按风险进入 [KAP](kap.md)。

排障日志、逐层 hook 和 tensor dump 必须默认关闭；稳定指标应定义单位、聚合口径和
rank 范围。

## 合入前检查

- 能力至少被两个模型或项目共享；否则仍留在模型插件。
- 配置有单一所有者、默认值、校验和 resolved-config 可见性。
- 没有复制 Trainer、runtime、checkpoint manager 或顶层 Hydra task。
- 模型、task 与 core 边界明确，无 core → 业务项目反向依赖。
- 大资源分配前拒绝不支持组合。
- CPU 测试覆盖成功和 fail-fast；GPU gate 与风险相称。
- 用户文档说明如何使用和有什么限制，开发者文档说明接口和验证方式。

## 按块 forward/backward

`StepOutput.backward_losses` 可提供惰性 loss 工厂；`StepOutput.loss` 仍表示整个
optimizer update 的标量日志。FSDP engine 在 precision context 下请求一个 loss，
退出该 context 后立即 backward，再请求下一个。梯度在各块间累积，最后只裁剪、
执行 optimizer/scheduler 一次，不由算法调用 backward 或 optimizer。
迭代失败时关闭 iterator，异常上抛且不更新参数。空 loss 序列报错。

当前消费者为 block DMD 的重放模式；Megatron engine 和 activation offload 拒绝
该可选路径，原单 loss contract 保持不变。普通 activation checkpoint 仍在模型层。
