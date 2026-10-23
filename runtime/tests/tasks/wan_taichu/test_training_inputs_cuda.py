from __future__ import annotations

import pytest
import torch

from vforge.tasks.wan_taichu.batch import stream_field
from vforge.tasks.wan_taichu.train import (
    _build_condition_processor,
    _processor_input,
)
from vforge.tasks.wan_taichu.variants import require_variant


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_wan21_flow_matching_inputs_are_built_on_cuda_when_available() -> None:
    device = _device()
    spec = require_variant("wan21_i2v_14b")
    streams = ("normal",)
    processor = _build_condition_processor(
        spec=spec,
        streams=streams,
        extras={"noise_shift": 1.0},
        precision="bf16",
    )
    encoded = {
        "prompt": torch.randn(1, 8, 24, device=device, dtype=torch.bfloat16),
        "frame_latents": torch.randn(1, 16, 3, 4, 4, device=device, dtype=torch.bfloat16),
        stream_field("normal"): torch.randn(1, 20, 3, 4, 4, device=device, dtype=torch.bfloat16),
    }
    processed = processor.process(
        _processor_input(encoded, spec=spec, streams=streams),
        step=0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(123),
        apply_dropouts=False,
    )

    assert processed["model_inputs"]["hidden_states"].shape == (1, 36, 3, 4, 4)
    assert processed["targets"].shape == encoded["frame_latents"].shape
    assert processed["model_inputs"]["hidden_states"].device.type == device.type


def test_wan21_audio_processor_forwards_speak_features() -> None:
    device = _device()
    spec = require_variant("wan21_i2v_14b_audio")
    processor = _build_condition_processor(
        spec=spec,
        streams=("rgb",),
        extras={"noise_shift": 5.0},
        precision="bf16",
    )
    audio = torch.randn(1, 25, 1024, 8, device=device, dtype=torch.bfloat16)
    encoded = {
        "prompt": torch.randn(1, 8, 24, device=device, dtype=torch.bfloat16),
        "frame_latents": torch.randn(1, 16, 3, 4, 4, device=device, dtype=torch.bfloat16),
        stream_field("rgb"): torch.randn(1, 20, 3, 4, 4, device=device, dtype=torch.bfloat16),
        "audio_features": audio,
    }

    processed = processor.process(
        _processor_input(encoded, spec=spec, streams=("rgb",)),
        step=0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(123),
    )

    assert torch.equal(processed["model_inputs"]["audio_input_s"], audio)
    assert processed["model_inputs"]["audio_input_l"] is None


def test_a14b_selects_expert_range_before_timestep_draw() -> None:
    device = _device()
    spec = require_variant("wan22_a14b")
    processor = _build_condition_processor(
        spec=spec,
        streams=("rgb",),
        extras={
            "noise_shift": 3.0,
            "boundary_ratio": 0.9,
            "train_expert": "both",
        },
        precision="bf16",
    )
    processor._ensure_diffusion_helpers()
    high = processor._sample_timestep_index(
        step=0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(7),
    )
    low = processor._sample_timestep_index(
        step=1,
        device=device,
        generator=torch.Generator(device=device).manual_seed(7),
    )

    assert float(processor._scheduler.timesteps[high]) >= 900.0
    assert float(processor._scheduler.timesteps[low]) < 900.0


def test_ti2v_inpaints_clean_first_frame_and_builds_per_token_timestep() -> None:
    device = _device()
    spec = require_variant("wan22_ti2v_5b")
    streams = ("normal",)
    processor = _build_condition_processor(
        spec=spec,
        streams=streams,
        extras={
            "condition_streams": list(streams),
            "noise_shift": 5.0,
            "exclude_first_frame_from_loss": True,
        },
        precision="bf16",
    )
    first = torch.randn(1, 48, 1, 4, 4, device=device, dtype=torch.bfloat16)
    encoded = {
        "prompt": torch.randn(1, 8, 24, device=device, dtype=torch.bfloat16),
        "frame_latents": torch.randn(1, 48, 3, 4, 4, device=device, dtype=torch.bfloat16),
        "first_frame_latents": first,
        stream_field("normal"): torch.randn(1, 48, 3, 4, 4, device=device, dtype=torch.bfloat16),
    }
    processed = processor.process(
        encoded,
        step=0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(321),
        apply_dropouts=False,
        patch_size=(1, 2, 2),
    )

    hidden = processed["model_inputs"]["hidden_states"]
    assert hidden.shape == (1, 96, 3, 4, 4)
    assert torch.equal(hidden[:, :48, 0:1], first)
    assert processed["model_inputs"]["timestep"].shape == (1, 12)
    assert torch.count_nonzero(processed["loss_weights"][:, :, 0:1]) == 0


@pytest.mark.gpu(gpus=1)
def test_ref_processor_prefers_external_latents_and_samples_online_fallback() -> None:
    device = torch.device("cuda")
    spec = require_variant("wan22_ti2v_5b_ref")
    streams = ("grid", "normal")
    processor = _build_condition_processor(
        spec=spec,
        streams=streams,
        extras={
            "condition_streams": list(streams),
            "noise_shift": 5.0,
            "ref_num_frames_online": 2,
        },
        precision="bf16",
    )
    frame_latents = torch.randn(1, 48, 40, 4, 4, device=device, dtype=torch.bfloat16)
    encoded = {
        "prompt": torch.randn(1, 8, 24, device=device, dtype=torch.bfloat16),
        "frame_latents": frame_latents,
        "first_frame_latents": frame_latents[:, :, 0:1].clone(),
        stream_field("grid"): torch.randn_like(frame_latents),
        stream_field("normal"): torch.randn_like(frame_latents),
    }
    external = torch.randn(1, 48, 2, 4, 4, device=device, dtype=torch.bfloat16)
    encoded["ref_frame_latents"] = external
    processed = processor.process(
        encoded,
        step=0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(7),
        patch_size=(1, 2, 2),
    )
    assert torch.equal(processed["model_inputs"]["ref_hidden_states"], external)

    encoded.pop("ref_frame_latents")
    sampled = None
    for seed in range(16):
        processed = processor.process(
            encoded,
            step=0,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed),
            patch_size=(1, 2, 2),
        )
        sampled = processed["model_inputs"]["ref_hidden_states"]
        if sampled is not None:
            break
    assert sampled is not None
    assert sampled.shape[:2] == (1, 48)
    assert 1 <= sampled.shape[2] <= 2
