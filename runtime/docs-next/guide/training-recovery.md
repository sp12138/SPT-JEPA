# 保存、恢复与自动容错

训练的保存、恢复和故障重启通过以下配置完成：

- **保存 checkpoint**：默认在训练结束时保存；通过 `checkpoint.every_n_steps` 设置保存间隔，
  `checkpoint.keep_last` 设置保留数量。
- **恢复训练**：`resume.path=auto` 自动加载最新完整 checkpoint，也可以指定路径；
  关闭优化器和训练状态恢复后，可只加载权重开始新训练。
- **故障后自动续训**：平台负责重新拉起任务，vForge 配合定期保存和 `resume.path=auto`
  恢复训练进度。下文以 KAP 的 `recoverLevel: 2` 为例。
- **加速 S3 保存与恢复**：通过 `storage.s3_mounts` 配置已有挂载与 bucket 的对应关系，
  启用本地 SSD 暂存和 S3 并发传输。

## 定期保存 checkpoint

默认只在训练结束时异步保存 checkpoint，目录为 `${runtime.output_dir}/checkpoints`，
保留最近两份。checkpoint 包含模型权重、优化器和训练状态。

新 checkpoint 保存成功后，会删除超出 `checkpoint.keep_last` 数量的旧 checkpoint。
保存在 EFS 上时，删除的是 EFS 中的实际文件；S3 目前通过挂载目录清理，需挂载支持删除
且具备删除权限，否则远端可能保留超过指定数量的 checkpoint。

每 100 步保存一次，在启动命令中增加：

```text
checkpoint.every_n_steps=100
```

也可以写入训练 YAML：

```yaml
checkpoint:
  enabled: true
  every_n_steps: 100
  save_final: true
  keep_last: 2
```

保留全部 checkpoint、不自动删除时，将 `keep_last` 改为 `null`；命令行使用
`checkpoint.keep_last=null`。不要设为 `0`。

自定义保存目录使用 `checkpoint.dirpath=/efs/your-name/my-run/checkpoints`。
目录要放在重启后仍可访问的存储上；多机训练时，所有节点都要能访问。

## 从最新 checkpoint 续训

首次启动和重启都使用：

```text
resume.path=auto
```

`auto` 从保存目录加载步数最大的完整 checkpoint。目录中没有 checkpoint 时，从初始权重
开始训练，所以第一次启动也可以使用 `auto`。

重启时保留相同的 `runtime.output_dir` 或显式 `checkpoint.dirpath`。例如：

```text
runtime.output_dir=/efs/your-name/vforge/runs/my-run
checkpoint.every_n_steps=100
resume.path=auto
```

续训默认恢复模型权重、优化器、学习率调度器、训练步数、数据读取位置和随机数状态。
`runtime.max_steps` 表示总步数，例如从第 600 步恢复并设置为 1000，会再训练 400 步。
续训时保持原模型、数据、精度和 DP/SP 配置。需要严格校验 checkpoint 的配置指纹和
执行条件时，显式设置 `resume.strict=true`；该选项不影响选择恢复哪些训练状态。

## 指定 checkpoint 或只加载权重

恢复指定步数的 checkpoint，将 `resume.path` 设为对应的 `.dcp` 目录：

```text
resume.path=/efs/your-name/vforge/runs/my-run/checkpoints/00600.dcp
```

指定的路径不存在时会报错。恢复训练进度需要完整 checkpoint，只有模型权重的 `.pt` 文件
无法恢复优化器、数据读取位置和随机数状态。

只加载模型权重、重新开始训练时，使用新输出目录，并关闭优化器和训练状态恢复：

```text
runtime.output_dir=/efs/your-name/vforge/runs/new-run
resume.path=/efs/your-name/vforge/runs/my-run/checkpoints/00600.dcp
resume.optimizer=false
resume.training_state=false
```

## 平台自动容错

在 KAP 任务 YAML 中启用故障重启：

```yaml
recoverLevel: 2
```

`recoverLevel` 是 KAP 配置，不写入训练 YAML。训练命令使用固定的输出目录，并设置
定期保存和自动续训：

```text
runtime.output_dir=/efs/your-name/vforge/runs/my-run
checkpoint.every_n_steps=100
resume.path=auto
```

任务失败后，KAP 会重新启动进程，vForge 从最近一次保存成功的 checkpoint 继续训练。
该 checkpoint 之后尚未保存的步骤会重跑。使用其他平台时，需要配置该平台的任务重试策略。

保存过程中失败产生的不完整 checkpoint 会被跳过。如果没有开启定期保存，首次训练
中途失败后只能从头开始。

## S3 保存与恢复加速

以当前 B300 环境为例，`/threed-code` 挂载的是 `ieg-infra-code-us-west-2` bucket。
将 checkpoint 放在该目录下，并在训练 YAML 中加入以下配置，即可启用本地 SSD 暂存和
S3 并发传输，加速保存与恢复：

```yaml
runtime:
  output_dir: /threed-code/vforge-checkpoints/your-name/ltx-dmd

checkpoint:
  every_n_steps: 100

resume:
  path: auto

storage:
  s3_mounts:
    - mount_path: /threed-code
      bucket: ieg-infra-code-us-west-2
      region: us-west-2
```

把 `your-name/ltx-dmd` 换成自己的用户名和任务名。`checkpoint.dirpath` 可以省略，
默认保存到 `${runtime.output_dir}/checkpoints`。上例实际保存到：

```text
s3://ieg-infra-code-us-west-2/vforge-checkpoints/your-name/ltx-dmd/checkpoints/
```

`mount_path` 填挂载根目录 `/threed-code`，其下的子目录都会自动映射，不需要逐个配置。
`key_prefix` 默认是空字符串，这里可以省略；只有挂载的是 bucket 中的某个子目录时，
才填写那个子目录的前缀。

这段配置复用平台已有挂载，不会创建挂载。换环境时使用对应的挂载目录和 bucket，
并确保训练进程具备 S3 读写权限。
