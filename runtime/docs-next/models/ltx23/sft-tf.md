# LTX2.3 SFT / TF

Stage 0 SFT 和 Stage 1 TF 共用模型、数据准备及启动方式，通过 experiment 选择阶段。

## 准备模型与数据

1. LTX-2.3 单文件 `.safetensors` checkpoint；
2. 离线 `.precomputed` 数据。

`.precomputed` 目录必须包含三路相同相对路径的 `.pt` 文件：

```text
.precomputed/
  latents/<sample>.pt
  audio_latents/<sample>.pt
  conditions/<sample>.pt
```

把本机实际路径写入环境变量：

```bash
export LTX23_MODEL_PATH=/path/to/ltx-2.3-22b-dev.safetensors
export LTX23_PRECOMPUTED_ROOT=/path/to/dataset/.precomputed
```

以下提供一份示例：

```bash
export LTX23_MODEL_PATH=/efs/dannsu/vforge/ci/ltx23-stage1/model/ltx-2.3-22b-dev.safetensors
export LTX23_PRECOMPUTED_ROOT=/efs/dannsu/vforge/ci/ltx23-stage1/data/.precomputed
```

选择训练阶段。以下命令默认运行 Stage 0；运行 Stage 1 时改用后一项：

```bash
# Stage 0 full SFT
export LTX23_EXPERIMENT=ltx23_ti2av_sft

# Stage 1 causal teacher forcing
# export LTX23_EXPERIMENT=ltx23_causal_ar_tf
```

## 单机 DP8

单机 8 卡、不开启 Ulysses 时，使用 `DP8 × USP1`。`LTX23_MAX_STEPS` 默认采用
experiment 的 200000 step；验证启动命令时可以在外部把它设为较小值。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LTX23_MAX_STEPS="${LTX23_MAX_STEPS:-200000}"
export LTX23_OUTPUT_DIR="${LTX23_OUTPUT_DIR:-/efs/dannsu/vforge/runs/${LTX23_EXPERIMENT}/dp8}"
export LTX23_CHECKPOINT_ENABLED="${LTX23_CHECKPOINT_ENABLED:-true}"

torchrun --standalone --nproc_per_node=8 \
  -m vforge.tasks.train \
  experiment="${LTX23_EXPERIMENT}" \
  runtime.output_dir="${LTX23_OUTPUT_DIR}" \
  runtime.max_steps="${LTX23_MAX_STEPS}" \
  scheduler.total_iters="${LTX23_MAX_STEPS}" \
  parallel.usp_size=1 \
  checkpoint.enabled="${LTX23_CHECKPOINT_ENABLED}" \
  resume.path=auto
```

## 双机 DP8 × USP2

`DP8 × USP2` 需要 16 个进程。下面以两台机器、每台 8 卡为例；在 KAP 中
`INDEX`/`CHIEF_IP` 会提供 node rank 和 rendezvous 地址，其他平台可以直接设置
`NODE_RANK`/`MASTER_ADDR`。两台机器执行相同命令：

```bash
export NODE_RANK="${NODE_RANK:-${INDEX:-${RANK:-0}}}"
export MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:?CHIEF_IP unset}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export LTX23_MAX_STEPS="${LTX23_MAX_STEPS:-200000}"
export LTX23_OUTPUT_DIR="${LTX23_OUTPUT_DIR:-/efs/dannsu/vforge/runs/${LTX23_EXPERIMENT}/dp8-usp2}"
export LTX23_CHECKPOINT_ENABLED="${LTX23_CHECKPOINT_ENABLED:-true}"

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m vforge.tasks.train \
  experiment="${LTX23_EXPERIMENT}" \
  runtime.output_dir="${LTX23_OUTPUT_DIR}" \
  runtime.max_steps="${LTX23_MAX_STEPS}" \
  scheduler.total_iters="${LTX23_MAX_STEPS}" \
  parallel.usp_size=2 \
  checkpoint.enabled="${LTX23_CHECKPOINT_ENABLED}" \
  resume.path=auto
```

模型、`.precomputed` 数据和多机输出目录必须对所有节点可见。`runtime.output_dir` 应使用
新目录，避免覆盖已有任务；配置默认读取 `LTX23_MODEL_PATH` 和
`LTX23_PRECOMPUTED_ROOT`，无需在命令行重复模型与数据路径。

Stage 0 和 Stage 1 都支持 USP 序列尾部不能整除的输入。运行时会先补齐到 USP 并行度
的整数倍，在 self-attention 和 audio↔video cross-attention 中屏蔽补齐 token，并在
计算 loss 前恢复逻辑长度。
