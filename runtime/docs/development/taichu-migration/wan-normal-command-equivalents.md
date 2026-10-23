# Taichu Wan 命令替换

本文集中维护 Taichu Wan 三个工作流在 vforge 中的直接替代方式。设计与实现边界见
[Wan Taichu 评估契约](../wan-taichu-evaluation.md)。

以下命令都从 vforge 仓库根目录执行。通用 JSONL manifest 入口仍然可用；迁移
Taichu Gameplay 时直接配置 `.pkl`，不再转换 manifest 或生成中间 MP4。

`grid` 是由相机内外参渲染的 camera Hilbert-RGB 条件视频，不是 Transformer patch
token 网格。`ref`/`reference` 可以是单一路径或有序路径列表；图片各贡献一帧，视频取
首帧。

`evaluation.data_source=inference` 是通用 manifest 的 condition-only 模式。
Taichu `.pkl` 包含目标视频，必须使用 `evaluation.data_source=validation`。

两个 production 数据配置在运行时直接读取 PKL 的 `interval`、normal/mask 和 camera：

```bash
evaluation=wan_taichu_5b_ref_prod
evaluation=wan_taichu_a14b_20s_prod
```

production evaluation preset 已包含 PKL 和数据尺寸，命令中不需要重复填写。完整生产命令见
[评估启动命令](wan-normal-evaluation-launch-commands.md)。

## `evaluate_gridnormal.sh`

Taichu：

```bash
CHECKPOINT_DIR=/runs/train/checkpoints \
OUTPUT_DIR=/runs/evaluation \
NUM_GPUS=8 \
bash scripts/evaluate_gridnormal.sh <taichu-config>
```

vforge：

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref_prod \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/evaluation \
  evaluation.evaluate_all_checkpoints=true \
  checkpoint.dirpath=/runs/train/checkpoints
```

CLI 显式选择 `wan_taichu_wan22_ti2v_5b_ref_prod`。它设置 16 个全局等间隔
样本、40 个采样 step、`guidance_scale=1.0`、seed 42，并在 manifest 未提供 ref 时从
validation 目标视频在线选择最多 6 帧。无需再通过环境开关选择 ref 能力：
`model.extras.variant=wan22_ti2v_5b_ref` 已明确声明该能力。

A14B 生产配置使用 `[rgb,grid,normal]`，直接改用：

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_a14b_prod \
  evaluation=wan_taichu_a14b_20s_prod \
  model.pretrained_path=/models/Wan2.2-I2V-A14B-Diffusers \
  latent_cache.mode=online \
  parallel.usp_size=2 \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/a14b-evaluation \
  evaluation.evaluate_all_checkpoints=true \
  checkpoint.dirpath=/runs/a14b/checkpoints
```

A14B 没有 ref-token，运行五个无-ref 变体；所有变体都保留 RGB 首帧条件。

5B Ref 默认与 Taichu `evaluate_gridnormal.sh` 对齐，运行以下五个原生变体：
`full`、`normal_to_video`、`normal_fg_to_video`、`normal_bg_to_video`、
`grid_to_video`。要运行 vforge 支持的完整八变体时显式配置：

```bash
NPROC_PER_NODE=8 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref_prod \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/evaluation \
  'evaluation.gridnormal.variants=[full,normal_to_video,normal_fg_to_video,normal_bg_to_video,grid_to_video,ref_normal_to_video,ref_grid_to_video,ti2v]'
```

也可用脚本环境变量传递同一份显式列表：

```bash
GRIDNORMAL_VARIANTS='[full,normal_to_video,normal_fg_to_video,normal_bg_to_video,grid_to_video,ref_normal_to_video,ref_grid_to_video,ti2v]' \
NPROC_PER_NODE=8 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref_prod \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/evaluation
```

只评估一个 checkpoint 时，删除 `evaluation.evaluate_all_checkpoints` 和
`checkpoint.dirpath`，改用：

```bash
evaluation.checkpoint_path=/runs/train/checkpoints/00020.dcp
```

常用参数对应关系：

- `NUM_GPUS` → `NPROC_PER_NODE`；
- `SP_SIZE` → `parallel.usp_size`；
- `NUM_SAMPLES` → `evaluation.num_samples`；
- `NUM_INFERENCE_STEPS` → `evaluation.num_inference_steps`；
- `GUIDANCE_SCALE` → `evaluation.guidance_scale`；
- `SEED` → `evaluation.seed`；
- `NEGATIVE_PROMPT` → `evaluation.negative_prompt`；
- `SKIP_EXISTING=1` → `evaluation.skip_existing=true`；
- `REF_NUM_FRAMES=K` → `evaluation.online_ref.num_frames=K`；
- `REF_FRAME_IDX=0,10,20` → `evaluation.online_ref.frame_indices=[0,10,20]`；
- `GRIDNORMAL_VARIANTS='[...]'` → `evaluation.gridnormal.variants=[...]`。

生成视频位于
`<output_dir>/checkpoint-<step>/<variant>/<sample_id>.mp4`。共享的 `gt`、`normal`、
`normal_fg`、`normal_bg`、`grid`、`ref` 每个样本只导出一次。

## `infer.sh`

Taichu：

```bash
GPU=0 bash scripts/infer.sh <taichu-config> \
  --sample_idx 17 \
  --output_dir /runs/infer
```

vforge 优先使用 WanTaichu：

```bash
CUDA_VISIBLE_DEVICES=0 SAMPLE_INDEX=17 NPROC_PER_NODE=1 bash scripts/infer.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/infer \
  evaluation.checkpoint_path=latest \
  checkpoint.dirpath=/runs/train/checkpoints
```

`scripts/infer.sh` 固定只推理一个样本；`SAMPLE_INDEX` 是所有 source 拼接后从零开始的
全局索引。
只使用 published 权重时，删除 `evaluation.checkpoint_path` 和 `checkpoint.dirpath`。
Ulysses 同时设置 `NPROC_PER_NODE=<SP_SIZE>` 和 `parallel.usp_size=<SP_SIZE>`。
通用 condition-only 请求仍可改用 `evaluation.data_source=inference` 与
`evaluation.manifest=/data/infer.jsonl`。

A14B standalone 使用同一个入口：

```bash
SAMPLE_INDEX=0 NPROC_PER_NODE=2 bash scripts/infer.sh \
  experiment=wan_taichu_wan22_a14b \
  evaluation=wan_taichu_a14b_20s_prod \
  model.pretrained_path=/models/Wan2.2-I2V-A14B-Diffusers \
  'model.extras.condition_streams=[rgb,grid,normal]' \
  latent_cache.mode=online \
  parallel.usp_size=2 \
  evaluation.data_source=validation \
  evaluation.output_dir=/runs/a14b-infer \
  evaluation.checkpoint_path=latest \
  checkpoint.dirpath=/runs/a14b/checkpoints
```

## `visualize.sh`

Taichu：

```bash
bash scripts/visualize.sh <dataset-config> \
  --num_samples 5 \
  --fps 24 \
  --output /runs/visualize
```

vforge 直接读取 Taichu PKL：

```bash
bash scripts/visualize.sh \
  --evaluation wan_taichu_5b_ref_prod \
  --num-samples 5 \
  --fps 24 \
  --output-dir /runs/visualize
```

只检查一个样本：

```bash
bash scripts/visualize.sh \
  --evaluation wan_taichu_5b_ref_prod \
  --sample-index 17 \
  --fps 24 \
  --output-dir /runs/visualize
```

该命令只解码并导出原始媒体、prompt 和 metadata，不加载模型、checkpoint 或
distributed runtime，也不生成 fallback 条件。缺失字段会记录为 absent。

## Causal eval

官方 Wan 的 `wan22_ti2v_5b_tf` dedicated teacher-forcing causal eval 仍保留。它不属于
已移除的官方双向 GridNormal/ref-token 过渡实现：

```bash
NPROC_PER_NODE=1 bash scripts/eval.sh \
  experiment=wan22_ti2v_5b_tf \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.rollout_mode=incremental_kv \
  evaluation.data_source=inference \
  evaluation.manifest=/data/causal.jsonl \
  evaluation.output_dir=/runs/causal \
  evaluation.checkpoint_path=latest \
  checkpoint.dirpath=/runs/train/checkpoints
```

WanTaichu causal 使用对应 experiment：

```bash
NPROC_PER_NODE=1 bash scripts/eval.sh \
  experiment=wan_taichu_wan22_ti2v_5b_causal_teacher \
  model.pretrained_path=/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  evaluation.rollout_mode=incremental_kv \
  evaluation.data_source=inference \
  evaluation.manifest=/data/causal.jsonl \
  evaluation.output_dir=/runs/causal
```

A14B 99-lat SWA train-cache eval 直接读取训练阶段生成的 indexed `.pt`，无需生成
`manifest.json`：

```bash
A14B_LATENT_CACHE=/path/to/gameplay-production-a14b-20s \
NPROC_PER_NODE=2 bash scripts/eval.sh \
  experiment=wan_taichu_wan22_a14b_causal_teacher \
  evaluation=wan_taichu_a14b_causal_teacher_99lat_swa_train \
  model.pretrained_path=/efs/shared/models/Wan2.2-I2V-A14B-Diffusers \
  evaluation.checkpoint_path=/efs/your-name/a14b-teacher/checkpoints/00100.pt \
  evaluation.output_dir=/runs/a14b-causal-teacher \
  parallel.usp_size=2 \
  attention.backend=flash_attention_4 \
  attention.allow_fallback=false
```

扫目录内所有 checkpoint 时，将 `evaluation.checkpoint_path` 替换为：

```bash
evaluation.evaluate_all_checkpoints=true \
checkpoint.dirpath=/efs/your-name/a14b-teacher/checkpoints
```

这里 `.pt` 保存实际 latent bundle，`manifest.json` 只描述 vforge-native
hashed-bucket cache 的样本索引和 schema；两者不能仅通过改名互换。该 eval 路径直接读取
`train/<shard>/<index>.pt`，不需要转换。

将 `evaluation.rollout_mode` 改为 `full_recompute` 即使用无 cache 的
golden/fallback。`incremental_kv` 要求 temporal patch size 为 `1`，并支持通过
`parallel.usp_size` 开启 Ulysses。A14B 默认按 timestep 900 切换 high/low expert，
使用 `local_attn_size=30`、`sink_size=9` 的滑窗 KV。

## 迁移回归基线

2026-09-01 在 B300 上验证了上述公开入口：

- `evaluate_gridnormal.sh` 的完整八变体使用真实 checkpoint、USP4、40-step 与 Taichu
  对照，121 帧视频 pixel MAE 为 `0.00928`–`0.02073`；
- `infer.sh` 使用非空 prompt、同输入、同 checkpoint、同 RNG、USP4、40-step 与
  Taichu 对照，121 帧视频 pixel MAE 为 `0.01371`；
- WanTaichu 与官方 Wan teacher-forcing 的 `full_recompute`、`incremental_kv`
  均通过 1-step GPU smoke，并核验 sidecar 中的 rollout mode；
- 五个无-ref 变体的环境变量/Hydra 两种写法、single/latest/all checkpoint、
  `negative_prompt`、固定 ref frame、`skip_existing`、published-only 和
  `visualize.sh` 的 first-N/sample-index/sparse manifest 均通过入口回归。
