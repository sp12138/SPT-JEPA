# 配置从哪里来、谁覆盖谁

本文面向维护 Native 配置入口的开发者，说明配置从哪里来、不同配置的覆盖顺序，以及
最终配置如何校验和保存。普通用户只需在 CLI 覆盖字段或使用
[自己的配置文件](../reference/configuration.md#custom-config-file)。
模型插件作者只需先读“模型插件作者最少需要知道什么”；只有修改配置层级、schema、
translate 或用户 YAML 加载方式时，才需要阅读其余机制。

## 配置层

```text
字段默认值
  → 模型配置（experiment）
  → Train / Prepare / Eval 等任务配置
  → 用户 YAML
  → CLI 临时覆盖
  → resolved_config.yaml
```

各层职责：

1. `vforge/hydra_config.py` 声明完整字段、类型和基础默认；
2. `vforge/conf/experiment/<name>.yaml` 只声明模型配置和模型约束；
3. `vforge/conf/{train,prepare,eval,convert_checkpoint}.yaml` 声明各任务的公共行为；
4. 用户 YAML 选择模型配置，并只保存当前任务的差异；
5. CLI 是一次运行的最终覆盖；
6. `write_resolved_config()` 保存校验和解析后的配置及 hash。

不要让模型配置重复声明公共的保存与恢复规则，也不要把模型精度约束放进用户配置模板。

## 模型插件作者最少需要知道什么

每个内置模型在 `vforge/conf/experiment/<name>.yaml` 有一份 experiment 配置。新增模型
时只需保证：

1. experiment 中的 `model.name` 与 `vforge/tasks/registry.py` 的注册表键完全一致；
2. `model.extras` 只放模型专属参数，由该模型的 task builder 解释；
3. 只注册模型当前支持的 train/eval verb；存量 encode builder 仍直接接收
   `StandaloneConfig`，但 prepare latent 后续会迁入数据管线，新模型不建议再新增。

experiment 选择模型和声明模型约束，不负责训练循环、保存恢复或任务入口的公共行为。
完整接入步骤和各 builder 接口见[新增 Native 模型](adding-native-model.md)。只有需要
修改配置层级、schema、translate 或用户 YAML 加载方式时，才需要继续阅读下面的机制。

## 内置配置如何组合

以 `train.yaml` 为例：

```yaml
defaults:
  - native_defaults
  - experiment: wan21_i2v_14b
  - _self_
```

`native_defaults` 由 `vforge.conf` 通过 Hydra ConfigStore 注册，内容直接来自
`StandaloneConfig`，不另写一份默认值 YAML。四个入口都先加载公共默认值，再载入
模型配置，最后通过 `_self_` 载入入口本身。因此入口的公共生命周期配置覆盖模型配置。
公共字段在解析 CLI 前已经存在，例如 `parallel.mixed_precision.param_dtype=float32`
可以直接覆盖；experiment 只需声明与公共默认不同的精度要求。

用户仍可在 CLI 选择其它模型配置：

```bash
vforge-train experiment=minimax_h3_fl2va
```

`prepare`、`eval` 和 `convert_checkpoint` 使用同一结构，但分别拥有数据处理模式、
输出目录和任务专属字段。

## 用户 YAML 如何加载

用户 YAML 放在 `user/` 子目录中。例如
`/workspace/my-configs/user/wan21-local.yaml`：

```yaml
# @package _global_
defaults:
  - override /experiment: wan21_i2v_14b
  - _self_

runtime:
  max_steps: 100
```

入口仍然只有一个标准 `@hydra.main`。启动时通过 `--config-dir` 告诉 Hydra 去哪里寻找
用户 YAML，再用 `+user=wan21-local` 加载这个文件：

```bash
vforge-train \
  --config-dir /workspace/my-configs \
  +user=wan21-local
```

Hydra 原生完成以下步骤：

```text
加载当前命令的任务配置
  → 加载任务默认的模型配置
  → 加载 user/wan21-local.yaml
  → 按用户 YAML 的 defaults 替换模型配置
  → 应用用户 YAML 中的业务字段
  → 应用 CLI 临时覆盖
```

`# @package _global_` 使用户字段合入根配置，而不是落到 `user.*`；
`override /experiment` 替换任务默认选择的模型配置；`_self_` 使当前文件中的业务
覆盖在其 defaults 之后生效。Train、Prepare、Eval 和 Convert Checkpoint 仍分别保留自己的
任务配置，因此同一份用户 YAML 可以复用，生命周期不会被用户文件替换。

这里没有自定义参数解析、`OmegaConf.merge()` 分支或第二套打印逻辑。`--cfg job`、CLI
override 及 Hydra 其它标准参数都由同一个入口处理。

## 最终配置如何校验

ConfigStore 注册的是 schema 导出的普通配置节点，避免 frozen dataclass 使 Hydra
合并只读，同时保留模型 extras 的扩展能力。字段转换与运行时校验仍沿用现有路径。
Hydra 完成组合后，`build_runtime()` 将 `DictConfig` 转成
`StandaloneConfig`：

```text
DictConfig
  → config_from_dict()
  → StandaloneConfig
  → tasks.translate.build_runtime()
  → TrainerConfig / OpsConfig / task requests
```

因此：

- 未声明字段应在 `StandaloneConfig` 转换阶段失败；
- schema 不 import `vforge` runtime 模块；
- 跨字段不变量由 translate 构造的 runtime dataclass 校验；
- 新字段同时更新 schema、translate、字段 Reference 和 compose 测试；
- 配置合成测试优先使用 `compose_standalone_config()`；用户 YAML 还要通过真实入口
  subprocess 测试，避免只验证内部 compose helper。

## 调试与测试

只看最终 Hydra job：

```bash
vforge-train \
  --config-dir /workspace/my-configs \
  --cfg job \
  +user=wan21-local
```

开发回归至少覆盖：

1. 每个内置 task 可以 compose；
2. 每个公开 experiment 可以替换默认 experiment；
3. 用户 YAML 可以保留当前任务行为并替换模型配置；
4. CLI 可以覆盖用户 YAML；
5. `resolved_config.yaml` 与 runtime 实际消费的值一致。

任务真正启动后，以 `${runtime.output_dir}/resolved_config.yaml` 和
`resolved_config.hash` 作为复现依据。完整反馈阶梯和测试命令见
[本地开发与验证](testing.md)。

## 下一步

- 新增 model/experiment：继续读[新增 Native 模型](adding-native-model.md)；
- 修改 `defaults`、用户 YAML 加载方式或 schema：按
  [本地开发与验证](testing.md)运行配置定向测试。
