# Wan Taichu CD

CD（Consistency Distillation，一致性蒸馏）训练 student，使相邻噪声时间点的干净样本
预测保持一致。冻结 teacher 执行一步 Euler 推进，EMA（指数移动平均）模型提供目标，
只有 student 执行反向传播和优化器更新。

本页针对经过世界模型适配的 Wan Taichu causal teacher：5B 使用 grid/normal 条件，
A14B 使用 RGB/grid/normal 条件。CD 复用这些条件输入和按时间块组织的 teacher forcing。

## 准备模型与数据

在已安装 vForge 与 Wan 依赖的 GPU 环境中执行；基础镜像和安装方式见
[镜像资源](../../guide/images.md)。准备以下模型与数据：

1. 对应模型的 Diffusers 目录，用于读取模型配置。
2. 与条件通道、模型变体匹配的 causal teacher checkpoint，用于初始化 teacher、
   student 和 EMA。5B 三个角色使用相同初始权重；A14B 的 student 与 EMA 必须一致，
   teacher 可以使用独立权重。
3. 已编码的 Wan Taichu 离线缓存。当前 CD 仅支持 `latent_cache.mode=offline`，
   不在训练过程中编码原始视频。

缓存使用分路条件字段：`frame_latents`、`prompt` 和 `condition_latents_<stream>`。
5B 还需要 `first_frame_latents`。各条件 latent 的时间和空间尺寸需要与目标视频匹配，
5B 与 A14B 的 latent 通道及 mask 布局不同，不能混用缓存。
字段校验以[缓存 schema](https://git.woa.com/ieg-infra/vforge/blob/master/vforge/tasks/wan_taichu/batch.py)为准。

Taichu indexed cache 的样本文件布局如下；保留缓存附带的 manifest 和索引元数据：

```text
cache/
  train/
    0000/
      00000000.pt
      00000001.pt
```

选择 experiment，并把下面的路径替换为本机可访问的模型与数据：

```bash
# Wan 2.2 TI2V-5B
export WAN_CD_EXPERIMENT=wan22_ti2v_5b_cd

# Wan 2.2 A14B：同时切换下方模型、权重和缓存路径
# export WAN_CD_EXPERIMENT=wan22_a14b_cd

export WAN_CD_MODEL_PATH=/path/to/wan-diffusers
export WAN_CD_INIT_WEIGHTS=/path/to/causal-teacher.pt
export WAN_CD_CACHE_DIR=/path/to/wan-taichu-cache
export WAN_CD_OUTPUT_DIR=/path/to/runs/${WAN_CD_EXPERIMENT}
```

两份完整配置分别为
[`wan22_ti2v_5b_cd`](https://git.woa.com/ieg-infra/vforge/blob/master/vforge/conf/experiment/wan22_ti2v_5b_cd.yaml)
和
[`wan22_a14b_cd`](https://git.woa.com/ieg-infra/vforge/blob/master/vforge/conf/experiment/wan22_a14b_cd.yaml)。
数据的 latent 长度筛选、条件 dropout、优化器和 EMA 参数以配置为准；若更换数据集，
需同步检查 `data.sources.0.min_latents` / `max_latents` 是否适合该缓存。

## 单机 8 卡 / SP8

以下命令使用 8 个进程组成一个序列并行组（SP8），逻辑 DP 为 1，FSDP2 在这 8 张卡上
分片。模型结构由 `WAN_CD_MODEL_PATH` 中的 Transformer 配置决定。
下面运行 1000 步，可通过 `WAN_CD_MAX_STEPS` 调整训练步数。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WAN_CD_MAX_STEPS="${WAN_CD_MAX_STEPS:-1000}"
export WAN_CD_CHECKPOINT_ENABLED="${WAN_CD_CHECKPOINT_ENABLED:-true}"

torchrun --standalone --nproc_per_node=8 \
  -m vforge.tasks.train \
  experiment="${WAN_CD_EXPERIMENT}" \
  training.models.teacher.pretrained_path="${WAN_CD_MODEL_PATH}" \
  training.models.student.pretrained_path="${WAN_CD_MODEL_PATH}" \
  training.models.ema_model.pretrained_path="${WAN_CD_MODEL_PATH}" \
  training.models.teacher.init_weights="${WAN_CD_INIT_WEIGHTS}" \
  training.models.student.init_weights="${WAN_CD_INIT_WEIGHTS}" \
  training.models.ema_model.init_weights="${WAN_CD_INIT_WEIGHTS}" \
  data.sources.0.cache_dir="${WAN_CD_CACHE_DIR}" \
  parallel.usp_size=8 \
  runtime.seed=123 \
  runtime.max_steps="${WAN_CD_MAX_STEPS}" \
  runtime.output_dir="${WAN_CD_OUTPUT_DIR}" \
  checkpoint.enabled="${WAN_CD_CHECKPOINT_ENABLED}" \
  checkpoint.every_n_steps=100 \
  resume.path=none
```

首次训练使用新的输出目录。正常推进时，`train_steps.jsonl` 会记录 student loss；
5B 固定首帧锚点并从 loss 中排除首帧，A14B 交替更新高低噪声专家，各专家维护自己的
EMA 和优化器计数。

首次试跑可设置 `WAN_CD_MAX_STEPS=2`，让 A14B 高、低专家各训练一步。
无需保存时设置 `WAN_CD_CHECKPOINT_ENABLED=false`；需要续训时保留默认的 `true`，
并将输出目录设为有足够空间的持久存储路径。

## 保存与恢复

开启 checkpoint 时，上述命令每 100 步保存一次完整训练状态，训练结束时也会保存。
默认目录为 `${runtime.output_dir}/checkpoints`；设置了 `checkpoint.dirpath` 时，
使用指定目录。

继续训练时，保持模型、数据配置和并行拓扑不变，使用原输出目录，将 `resume.path=none`
改为 `resume.path=auto`。程序会查找最新完整 checkpoint，恢复模型、优化器和训练进度；
没有 checkpoint 时从初始权重开始。`runtime.max_steps` 表示目标总步数，应大于已完成步数。
新开一轮训练时换一个输出目录。

指定 checkpoint 和平台自动容错见
[保存、恢复与自动容错](../../guide/training-recovery.md)。
