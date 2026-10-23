#!/usr/bin/env bash
# Extra dependencies for LTX GPU CI only; keep the dependency image and lock unchanged.
set -euo pipefail
python_bin="${1:?Pass the CI worker Python executable}"

uv pip install --python "$python_bin" --no-deps \
  'scipy==1.18.1' cmake ninja

# Build against the image's PyTorch instead of installing a mismatched binary wheel.
BUILD_VERSION=2.11.0 USE_CUDA=0 USE_ROCM=0 \
BUILD_CUDA_CTC_DECODER=0 BUILD_RIR=0 BUILD_RNNT=0 BUILD_ALIGN=0 \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
uv pip install --python "$python_bin" --no-deps --no-build-isolation \
  'torchaudio @ git+https://github.com/pytorch/audio.git@34c52a67e8941bbd8e6adaca0eb0b9eabec11d78'

uv pip install --python "$python_bin" --no-deps \
  'ltx-core @ git+https://github.com/Lightricks/LTX-2.git@10885e01526a99c17aaf83566c8962609ace1789#subdirectory=packages/ltx-core'

# Fail before loading large checkpoints if the decoder environment is incomplete.
"$python_bin" - <<'PY'
import torch
import torchaudio
from ltx_core.model.audio_vae import AudioDecoderConfigurator, VocoderConfigurator
from ltx_core.model.video_vae import VideoDecoderConfigurator

print("LTX decoder ready; torch:", torch.__version__, "torchaudio:", torchaudio.__version__)
PY
