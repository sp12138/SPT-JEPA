# Taichu 迁移 vforge 评估启动命令

vforge 的 `evaluate_gridnormal.sh`、`infer.sh` 和 `visualize.sh` 均可直接读取
Taichu Gameplay `.pkl`，不需要转换 manifest，也不会生成中间 MP4。

以下命令使用 `/efs`：

- evaluation preset 已包含 PKL 路径和数据尺寸；
- 最终结果写入 `/efs/alexlshen/vforge/wan-command-validation/outputs`。

数据尺寸、PKL 路径和挂载映射均由 evaluation preset 管理。评估和推理使用
`latent_cache.mode=online`，两侧均从原始像素输入编码。

## Wan2.2 TI2V 5B Ref

Taichu 配置：

```text
exp/ct/prod_v1/wan22-5b-diffusers/exp_ref_prod.yaml
```

vforge 使用 `evaluation=wan_taichu_5b_ref_prod` 直接加载三个数据来源。
`evaluation.num_samples=16` 与 `evaluation.sample_selection=evenly_spaced`
会按来源选出 6、5、5 个样本。

### evaluate_gridnormal.sh

Taichu：

```bash
cd taichu-video

CHECKPOINT_DIR=/efs/jwphuang/experiments/prod_v1_ref_prod/prod_v1_ref_prod/checkpoints \
OUTPUT_DIR=/efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/gridnormal/taichu \
NUM_GPUS=4 SP_SIZE=4 \
bash scripts/evaluate_gridnormal.sh \
  exp/ct/prod_v1/wan22-5b-diffusers/exp_ref_prod.yaml
```

vforge：

```bash
cd vforge

NPROC_PER_NODE=4 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref_prod \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/efs/shared/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  parallel.usp_size=4 \
  evaluation.data_source=validation \
  evaluation.num_samples=16 \
  evaluation.sample_selection=evenly_spaced \
  evaluation.output_dir=/efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/gridnormal/vforge \
  evaluation.checkpoint_format=pt \
  evaluation.evaluate_all_checkpoints=true \
  checkpoint.dirpath=/efs/jwphuang/experiments/prod_v1_ref_prod/prod_v1_ref_prod/checkpoints
```

5B 默认运行与 Taichu 相同的五个 variant。运行 vforge 全部八个 variant 时增加：

```bash
'evaluation.gridnormal.variants=[full,normal_to_video,normal_fg_to_video,normal_bg_to_video,grid_to_video,ref_normal_to_video,ref_grid_to_video,ti2v]'
```

### infer.sh

Taichu：

```bash
cd taichu-video

SP_SIZE=2 bash scripts/infer.sh \
  exp/ct/prod_v1/wan22-5b-diffusers/exp_ref_prod.yaml \
  --sample_idx 0 \
  --checkpoint_path /efs/jwphuang/experiments/prod_v1_ref_prod/prod_v1_ref_prod/checkpoints/27000.pt \
  --output_dir /efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/infer/taichu \
  --num_inference_steps 40 --guidance_scale 1.0 --seed 42
```

vforge：

```bash
cd vforge

CUDA_VISIBLE_DEVICES=0,1 SAMPLE_INDEX=0 NPROC_PER_NODE=2 \
bash scripts/infer.sh \
  experiment=wan_taichu_wan22_ti2v_5b_ref \
  evaluation=wan_taichu_5b_ref_prod \
  model.pretrained_path=/efs/shared/models/Wan2.2-TI2V-5B-Diffusers \
  latent_cache.mode=online \
  parallel.usp_size=2 \
  evaluation.data_source=validation \
  evaluation.num_samples=1 \
  evaluation.output_dir=/efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/infer/vforge \
  evaluation.num_inference_steps=40 \
  evaluation.guidance_scale=1.0 \
  evaluation.seed=42 \
  evaluation.checkpoint_format=pt \
  evaluation.checkpoint_path=/efs/jwphuang/experiments/prod_v1_ref_prod/prod_v1_ref_prod/checkpoints/27000.pt
```

### visualize.sh

Taichu：

```bash
cd taichu-video

bash scripts/visualize.sh \
  prod_v1/gameplay-production-ti2v5b-20s.yaml \
  --num_samples 5 --fps 24 \
  --output /efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/visualize/taichu
```

vforge：

```bash
cd vforge

bash scripts/visualize.sh \
  --evaluation wan_taichu_5b_ref_prod \
  --num-samples 5 --fps 24 \
  --output-dir /efs/alexlshen/vforge/wan-command-validation/outputs/5b-ref/visualize/vforge
```

## Wan2.2 A14B 20s

Taichu 配置：

```text
exp/ct/prod_v1/wan22-a14b-diffusers/exp_20s_prod.yaml
```

vforge 使用 `evaluation=wan_taichu_a14b_20s_prod` 直接加载 20s 数据。

### evaluate_gridnormal.sh

Taichu：

```bash
cd taichu-video

CHECKPOINT_DIR=/efs/jwphuang/experiments/prod_v1_5s_prod_a14b_20s-onlygameplay/prod_v1_5s_prod_a14b_20s-onlygameplay/checkpoints \
OUTPUT_DIR=/efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/gridnormal/taichu \
NUM_GPUS=4 SP_SIZE=4 \
bash scripts/evaluate_gridnormal.sh \
  exp/ct/prod_v1/wan22-a14b-diffusers/exp_20s_prod.yaml \
  --ckpt_step 100001
```

vforge：

```bash
cd vforge

NPROC_PER_NODE=4 bash scripts/evaluate_gridnormal.sh \
  experiment=wan_taichu_wan22_a14b_prod \
  evaluation=wan_taichu_a14b_20s_prod \
  model.pretrained_path=/efs/shared/models/Wan2.2-I2V-A14B-Diffusers \
  latent_cache.mode=online \
  parallel.usp_size=4 \
  evaluation.data_source=validation \
  evaluation.num_samples=16 \
  evaluation.sample_selection=evenly_spaced \
  evaluation.output_dir=/efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/gridnormal/vforge \
  evaluation.num_inference_steps=40 \
  evaluation.guidance_scale=3.5 \
  evaluation.seed=42 \
  evaluation.checkpoint_format=pt \
  evaluation.checkpoint_path=/efs/jwphuang/experiments/prod_v1_5s_prod_a14b_20s-onlygameplay/prod_v1_5s_prod_a14b_20s-onlygameplay/checkpoints/100001.pt \
  attention.backend=flash_attention_4 \
  attention.allow_fallback=false
```

A14B 无 ref-token，运行五个无-ref variant；RGB 首帧始终保留。

### infer.sh

Taichu：

```bash
cd taichu-video

SP_SIZE=4 bash scripts/infer.sh \
  exp/ct/prod_v1/wan22-a14b-diffusers/exp_20s_prod.yaml \
  --sample_idx 0 \
  --checkpoint_path /efs/jwphuang/experiments/prod_v1_5s_prod_a14b_20s-onlygameplay/prod_v1_5s_prod_a14b_20s-onlygameplay/checkpoints/100001.pt \
  --output_dir /efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/infer/taichu \
  --num_inference_steps 40 --guidance_scale 3.5 --seed 42
```

vforge：

```bash
cd vforge

SAMPLE_INDEX=0 NPROC_PER_NODE=4 bash scripts/infer.sh \
  experiment=wan_taichu_wan22_a14b_prod \
  evaluation=wan_taichu_a14b_20s_prod \
  model.pretrained_path=/efs/shared/models/Wan2.2-I2V-A14B-Diffusers \
  latent_cache.mode=online \
  parallel.usp_size=4 \
  evaluation.data_source=validation \
  evaluation.num_samples=1 \
  evaluation.output_dir=/efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/infer/vforge \
  evaluation.num_inference_steps=40 \
  evaluation.guidance_scale=3.5 \
  evaluation.seed=42 \
  evaluation.checkpoint_format=pt \
  evaluation.checkpoint_path=/efs/jwphuang/experiments/prod_v1_5s_prod_a14b_20s-onlygameplay/prod_v1_5s_prod_a14b_20s-onlygameplay/checkpoints/100001.pt \
  attention.backend=flash_attention_4 \
  attention.allow_fallback=false
```

### visualize.sh

Taichu：

```bash
cd taichu-video

bash scripts/visualize.sh \
  prod_v1/gameplay-production-a14b-20s.yaml \
  --num_samples 5 --fps 24 \
  --output /efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/visualize/taichu
```

vforge：

```bash
cd vforge

bash scripts/visualize.sh \
  --evaluation wan_taichu_a14b_20s_prod \
  --num-samples 5 --fps 24 \
  --output-dir /efs/alexlshen/vforge/wan-command-validation/outputs/a14b-20s/visualize/vforge
```

## Checkpoint 选择

评估目录中全部 checkpoint：

```bash
evaluation.checkpoint_format=pt \
evaluation.evaluate_all_checkpoints=true \
checkpoint.dirpath=/path/to/checkpoints
```

只评估最新 checkpoint：

```bash
evaluation.checkpoint_path=latest \
checkpoint.dirpath=/path/to/checkpoints
```

只评估指定 checkpoint：

```bash
evaluation.checkpoint_format=pt \
evaluation.checkpoint_path=/path/to/checkpoints/100001.pt
```

## 最终验证结果

六个入口均已完成运行验证并生成对应产物：

- 5B：`evaluate_gridnormal.sh`、`infer.sh`、`visualize.sh`；
- A14B：`evaluate_gridnormal.sh`、`infer.sh`、`visualize.sh`。

### A14B 对齐结论

A14B 已完成 bitwise 对齐验证。此前的不一致来自：

- grid 增强：Taichu eval 继承 `grid_random_rot_deg=30.0`，vforge eval 为 `0.0`；
  Taichu eval 通过 overlay 关闭随机旋转后对齐；
- 文本条件：两边的 negative prompt 和文本清洗配置不同；统一后对齐；
- 运行配置：统一双 expert checkpoint、seed、采样步数、CFG、SP 和 attention backend。

完成上述对齐后，2-step 前向的 31/31 个阶段完全一致。

bitwise 对齐后又完成了 40-step `full` 视频生成，任务
`ieg-a14b-gen-r2-09041708-6324` 成功。两边视频均为 597 帧、`1280x720`、24 FPS，
PSNR `40.39 dB`、MAE `1.70/255`，生成效果基本对齐；剩余差异主要来自 VAE 解码、
视频编码和像素取整。

### 5B 对齐结论

5B 已完成 bitwise 对齐验证。不对齐由两个数值执行差异引入：

- reference tensor layout：Taichu 传给 `patch_embedding_ref` 的输入是
  `channels_last_3d`，vforge 默认是连续 NCDHW；在 vforge 的
  `patch_embedding_ref` 前转为 `channels_last_3d` 后对齐；
- eval weight dtype：Taichu Transformer 使用 BF16 working weights，vforge 默认是
  FP32 参数存储配合 autocast；将 vforge eval Transformer 构建为 BF16 后，
  RMSNorm 的数值内核与 Taichu 对齐。

增加这两项后中间值完全一致。它们只作为 bitwise 门禁条件记录，不合入默认训练代码。

40-step eval 视频与 Taichu 对齐，PSNR `39.84 dB`、MAE `1.64/255`。剩余小幅
像素差异主要来自在线 VAE 编解码、视频编码和像素取整，不影响整体生成效果。
