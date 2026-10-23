# LTX2.3 DMD 推理

在[安装好 vForge 的 CUDA 环境](../../development/environment.md)中，
使用 `ltx23_dmd_eval` 加载训练后的 generator，根据缓存中的参考首帧和文本生成带音轨的
MP4。可以直接使用 DMD / Long Tuning 训练保存的 checkpoint，无需先导出权重。

## 安装解码依赖

首次运行 eval 前，在运行 vForge 的同一个 Python 环境中安装 `ltx-core`。
下面的命令用于当前标准 Linux CUDA 容器（PyTorch 2.13、transformers 5.16.1），
从 Lightricks 公开仓库安装固定版本，不修改仓库的 `pyproject.toml`、`uv.lock` 或镜像。
LTX GPU CI 会在测试前自动执行下面的安装与导入检查，每个任务执行一次。

先补齐 SciPy 和 TorchAudio 的构建工具，再用当前环境的 PyTorch 编译 TorchAudio：

```bash
uv pip install --python "$(command -v python)" --no-deps \
  'scipy==1.18.1' cmake ninja

BUILD_VERSION=2.11.0 USE_CUDA=0 USE_ROCM=0 \
BUILD_CUDA_CTC_DECODER=0 BUILD_RIR=0 BUILD_RNNT=0 BUILD_ALIGN=0 \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
uv pip install --python "$(command -v python)" --no-deps --no-build-isolation \
  'torchaudio @ git+https://github.com/pytorch/audio.git@34c52a67e8941bbd8e6adaca0eb0b9eabec11d78'

uv pip install --python "$(command -v python)" --no-deps \
  'ltx-core @ git+https://github.com/Lightricks/LTX-2.git@10885e01526a99c17aaf83566c8962609ace1789#subdirectory=packages/ltx-core'
```

TorchAudio 首次编译需要数分钟；已有按当前 PyTorch 构建、可正常导入的 TorchAudio 时，
可跳过它的编译和构建工具安装。`--no-deps` 保留 vForge 已安装的 PyTorch 和 transformers；
这里使用 Gemma 3，不使用 `ltx-core` 的 Gemma 4 编码器。

安装后检查音视频解码模块能否导入：

```bash
python - <<'PY'
import torch
import torchaudio
from ltx_core.model.audio_vae import AudioDecoderConfigurator, VocoderConfigurator
from ltx_core.model.video_vae import VideoDecoderConfigurator

print("LTX decoder ready; torch:", torch.__version__, "torchaudio:", torchaudio.__version__)
PY
```

## 准备权重与输入

推理需要以下资产：

- **官方 BASE checkpoint**：提供模型结构、文本 connector，以及视频、音频解码权重。
- **训练后的 generator**：支持 vForge 完整 DCP、带 `generator` 参数的 PT，以及参考实现
  导出的单文件 generator safetensors。DCP/PT 只读取 generator，不加载 teacher、critic
  或优化器。
- **数据缓存**：使用与 [DMD 训练](dmd.md#dmd-data) 相同的 `latents/` 和 `conditions/`
  目录，提供参考首帧、空间尺寸、帧率和文本条件。生成时长不受缓存视频时长限制。
- **Gemma 3 模型目录**：仅在使用自定义 prompt 时需要；使用缓存文本时无需加载 Gemma。

`ieg-infra-efs` 挂载为 `/efs` 时，BASE、缓存和 Gemma 可使用以下范例路径。
将 checkpoint 和输出路径替换为自己的路径：

```bash
export LTX_DMD_ROOT=/efs/alexlshen/vforge/examples/ltx23-dmd
export LTX_DMD_BASE="$LTX_DMD_ROOT/model/base.safetensors"
export LTX_DMD_CACHE="$LTX_DMD_ROOT/data/.precomputed"
export LTX_DMD_GEMMA="$LTX_DMD_ROOT/model/gemma-3-12b"
export LTX_DMD_EVAL_CHECKPOINT=/efs/your-name/ltx-dmd/checkpoints/01000.dcp
export LTX_DMD_EVAL_OUTPUT=/efs/your-name/ltx-dmd-eval
```

这里的 checkpoint 指向某一步的 `.dcp` 目录。使用 safetensors 或 PT 时，改为文件路径：

```bash
export LTX_DMD_EVAL_CHECKPOINT=/path/to/trained-generator.safetensors
# 或 export LTX_DMD_EVAL_CHECKPOINT=/path/to/dmd-model.pt
```

范例中的 `model/generator.safetensors` 是 DMD 训练用的 **TF 初始化权重**；评估蒸馏效果时，
应换成 DMD 训练后的 generator。TF 训练的 DCP 不属于这里的 DMD checkpoint 格式。

## 使用缓存文本生成

以下命令在单机 8 卡运行，使用第 0 个缓存样本生成约 30 秒视频：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.eval \
  experiment=ltx23_dmd_eval \
  model.pretrained_path="$LTX_DMD_BASE" \
  evaluation.checkpoint_path="$LTX_DMD_EVAL_CHECKPOINT" \
  evaluation.sources.0.cache_dir="$LTX_DMD_CACHE" \
  evaluation.sample_index=0 \
  runtime.output_dir="$LTX_DMD_EVAL_OUTPUT"
```

`evaluation.sample_index` 从 0 开始，默认取第 0 个样本。程序自动识别 checkpoint 格式，
不需要设置训练的 `resume.path`。

默认使用 SP1；增加 `parallel.usp_size=2` 即可使用 SP2，8 卡时 DP 自动推导为 4。
各节点需要能访问相同的权重、缓存与输出路径。

生成参数可追加到命令末尾：

```text
model.extras.seconds=10
evaluation.seed=123
```

时长默认 **30 秒**，seed 默认 **42**。去噪日程默认为 `[999, 757, 522, 0]`，不使用 CFG。
空间尺寸和 FPS 来自所选缓存，长度按完整 block 对齐；例如 24 FPS 下设置 30 秒，
实际生成 721 帧，约 30.04 秒。

## 使用自定义文本

参考首帧仍使用所选缓存样本，文本替换为 `model.extras.prompt`：

```bash
torchrun --standalone --nproc_per_node=8 -m vforge.tasks.eval \
  experiment=ltx23_dmd_eval \
  model.pretrained_path="$LTX_DMD_BASE" \
  evaluation.checkpoint_path="$LTX_DMD_EVAL_CHECKPOINT" \
  evaluation.sources.0.cache_dir="$LTX_DMD_CACHE" \
  evaluation.sample_index=0 \
  model.extras.text_encoder_path="$LTX_DMD_GEMMA" \
  model.extras.prompt='A person turns toward the camera and says hello.' \
  runtime.output_dir="$LTX_DMD_EVAL_OUTPUT/prompt"
```

多个 prompt 写入 UTF-8 文本文件，每行一条，空行忽略：

```bash
cat > /tmp/ltx-prompts.txt <<'PROMPTS'
A person turns toward the camera and says hello.
A person smiles and waves at the camera.
PROMPTS

torchrun --standalone --nproc_per_node=8 -m vforge.tasks.eval \
  experiment=ltx23_dmd_eval \
  model.pretrained_path="$LTX_DMD_BASE" \
  evaluation.checkpoint_path="$LTX_DMD_EVAL_CHECKPOINT" \
  evaluation.sources.0.cache_dir="$LTX_DMD_CACHE" \
  evaluation.sample_index=0 \
  model.extras.text_encoder_path="$LTX_DMD_GEMMA" \
  model.extras.prompt_file=/tmp/ltx-prompts.txt \
  runtime.output_dir="$LTX_DMD_EVAL_OUTPUT/prompts"
```

`prompt` 和 `prompt_file` 二选一。每条文本使用同一个参考样本和 seed，分别输出视频。

## 批量选择缓存样本

使用缓存各自的文本，评估前 8 个样本时，在“使用缓存文本生成”的命令末尾增加：

```text
evaluation.sample_index=null
evaluation.num_samples=8
```

批量运行时，同一并行批次的样本应具有相同 FPS 和 rollout block 数。
自定义 prompt 则保留具体的 `evaluation.sample_index`，用它选择共同的参考首帧。

## 长视频的历史缓存

默认 `model.extras.kv_cache_blocks=0`、`model.extras.num_sink_blocks=0`，保留完整历史 KV。
生成更长的视频、需要限制缓存显存时，可追加：

```text
model.extras.seconds=60
model.extras.kv_cache_blocks=4
model.extras.num_sink_blocks=1
```

`kv_cache_blocks` 以最大 block 的 token 数为单位设置滑窗容量。`num_sink_blocks`
额外固定保留最开始的生成 block，参考首帧单独保留；启用 sink 时需要正数的滑窗容量。
滑窗会改变模型可见的历史，生成效果可能与完整历史不同。

Long Tuning 权重也使用这个推理入口，时长和 KV 参数按上述方式设置；训练中的
`training.dmd.chain.*` 不会自动成为推理参数。

## 查看结果

结果保存在 `${runtime.output_dir}/samples/`：每个样本输出一个带音轨的 `.mp4`，
同名 `.mp4.json` 记录 seed、FPS、实际时长和 latent 帧数。prompt 文件生成的视频名称
带有对应的 prompt 编号。

使用相同配置和输出目录重启时，已完成且记录匹配的样本会被跳过。对比不同权重或参数时，
使用不同输出目录，便于同时查看生成结果。
