import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vforge.models.minimax_h3.encoder import (
    MiniMaxH3ConditionEncoder,
    _configure_qwen_patch_embed,
    _patchify_video_latents,
    _prepare_keyframes,
    _resolve_canvas,
    _video_pixels,
    resolve_qwen_patch_embed_mode,
)
from vforge.models.minimax_h3.inference import _unpatchify_video_rows
from vforge.models.minimax_h3.trace import H3TraceRecorder
from vforge.tasks.minimax_h3.shared import H3_ENCODED_BATCH_CODEC


class _PatchEmbed(nn.Module):
    in_channels = 3
    temporal_patch_size = 2
    patch_size = 2
    embed_dim = 4

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


def _text_encoder_with_patch_embed(
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(visual=SimpleNamespace(patch_embed=_PatchEmbed(dtype)))
    )


@pytest.mark.parametrize("mode", ["default", "fp32", "last_channel"])
def test_qwen_patch_embed_mode_resolves_supported_values(mode: str) -> None:
    assert resolve_qwen_patch_embed_mode({"qwen_patch_embed_mode": mode}) == mode


def test_qwen_patch_embed_mode_defaults_and_rejects_unknown_values() -> None:
    assert resolve_qwen_patch_embed_mode({}) == "default"
    with pytest.raises(ValueError, match="qwen_patch_embed_mode"):
        resolve_qwen_patch_embed_mode({"qwen_patch_embed_mode": "fastest"})


def test_qwen_fp32_patch_embed_matches_explicit_fp32_conv3d() -> None:
    text_encoder = _text_encoder_with_patch_embed(torch.bfloat16)
    patch_embed = text_encoder.model.visual.patch_embed
    inputs = torch.randn(5, 24, dtype=torch.bfloat16)
    weight = patch_embed.proj.weight.detach().float()
    bias = patch_embed.proj.bias.detach().float()
    patch_kernel = (
        patch_embed.temporal_patch_size,
        patch_embed.patch_size,
        patch_embed.patch_size,
    )
    expected = F.conv3d(
        inputs.float().view(-1, patch_embed.in_channels, *patch_kernel),
        weight,
        bias,
        stride=patch_kernel,
    ).view(-1, patch_embed.embed_dim)
    expected = expected.to(torch.bfloat16)

    _configure_qwen_patch_embed(text_encoder, mode="fp32")
    actual = patch_embed(inputs)

    assert patch_embed.proj.weight.dtype is torch.float32
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_qwen_last_channel_patch_embed_preserves_conv3d_values() -> None:
    text_encoder = _text_encoder_with_patch_embed()
    patch_embed = text_encoder.model.visual.patch_embed
    inputs = torch.randn(5, 24)
    expected = patch_embed(inputs)

    _configure_qwen_patch_embed(text_encoder, mode="last_channel")
    actual = patch_embed(inputs)

    torch.testing.assert_close(actual, expected)


def test_fl2va_canvas_and_keyframe_follow_official_geometry() -> None:
    first = Image.new("RGB", (1024, 683), color=(1, 2, 3))

    height, width = _resolve_canvas(first, {})
    keyframes, anchors = _prepare_keyframes(
        first,
        None,
        height=height,
        width=width,
    )

    assert (height, width) == (768, 1152)
    assert [image.size for image in keyframes] == [(1152, 768)]
    assert anchors == ("first",)


def test_video_row_patchification_roundtrips() -> None:
    latents = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 2, 3, 4, 6)
    patch_size = (1, 2, 2)
    rows = _patchify_video_latents(latents, patch_size)

    restored = _unpatchify_video_rows(
        rows,
        num_frames=3,
        latent_height=4,
        latent_width=6,
        patch_size=patch_size,
    )

    assert torch.equal(restored, latents)


def test_video_pixels_resize_to_resolved_canvas() -> None:
    frames = [Image.new("RGB", (8, 4), color=(index, 2, 3)) for index in range(2)]

    pixels = _video_pixels(frames, height=32, width=64)

    assert pixels.shape == (1, 3, 2, 32, 64)
    assert pixels.dtype == torch.uint8


def test_h3_inference_preparation_accepts_image_without_video(tmp_path) -> None:
    image_path = tmp_path / "first.png"
    Image.new("RGB", (1024, 683), color=(1, 2, 3)).save(image_path)
    encoder = object.__new__(MiniMaxH3ConditionEncoder)
    encoder.extras = {"num_frames": 124}

    prepared = encoder.prepare_inference_sample(
        {
            "image": str(image_path),
            "prompt": "A singer performs.",
        }
    )

    assert prepared.prompt == "A singer performs."
    assert prepared.num_frames == 124
    assert (prepared.height, prepared.width) == (768, 1152)
    assert prepared.keyframe_anchors == ("first",)


def test_h3_inference_preparation_requires_image_or_validation_video() -> None:
    encoder = object.__new__(MiniMaxH3ConditionEncoder)
    encoder.extras = {}

    with pytest.raises(ValueError, match="require 'image'"):
        encoder.prepare_inference_sample({"prompt": "missing condition"})
    with pytest.raises(ValueError, match="require 'image'"):
        encoder.prepare_inference_sample({"video": "/data/target.mp4"})


def test_fl2va_cache_schema_invalidates_unconditioned_bundles() -> None:
    assert H3_ENCODED_BATCH_CODEC.schema_version == 2


def test_h3_trace_records_stable_tensor_summary(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = H3TraceRecorder(path)

    recorder.tensor("example", torch.tensor([1.0, -2.0]), step=3)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["name"] == "example"
    assert record["step"] == 3
    assert record["shape"] == [2]
    assert record["sum"] == -1.0
    assert record["square_sum"] == 5.0
