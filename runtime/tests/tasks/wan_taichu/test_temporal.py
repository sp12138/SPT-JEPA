from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan_taichu.batch import build_encoded_batch_codec, stream_field
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.processors import (
    WanTaichuTeacherForcingConditionProcessor,
    WanTaichuTi2vConditionProcessor,
    _build_taichu_bsmntw_weights,
    _build_taichu_flow_match_scheduler,
)
from vforge.tasks.wan_taichu.temporal import (
    MaxLatentsSchedule,
    truncate_encoded_latents,
)
from vforge.tasks.wan_taichu.train import _train_step
from vforge.tasks.wan_taichu.variants import require_variant
from vforge.training.recipe import StepContext


def _schedule() -> MaxLatentsSchedule:
    schedule = MaxLatentsSchedule.from_config(
        {
            "start_step": 0,
            "end_step": 4000,
            "start_value": 38,
            "end_value": 150,
        }
    )
    assert schedule is not None
    return schedule


def _encoded(*, frames: int = 8, channels: int = 2) -> dict[str, object]:
    shape = (1, channels, frames, 2, 2)
    return {
        "sample_id": ["sample"],
        "prompt": torch.zeros(1, 3, 4),
        "frame_latents": torch.arange(torch.tensor(shape).prod()).reshape(shape).float(),
        "condition_latents_rgb": torch.full(shape, 1.0),
        "condition_latents_grid": torch.full(shape, 2.0),
        "condition_latents_normal": torch.full(shape, 3.0),
        "condition_fallback_latents_normal": torch.full(shape, 4.0),
        "first_frame_latents": torch.full((1, channels, 1, 2, 2), 5.0),
        "ref_frame_latents": torch.full((1, channels, 2, 2, 2), 6.0),
        "min_latents": 3,
        "max_latents": 6,
    }


def test_flow_match_scheduler_uses_taichu_pre_shift_sigma_table() -> None:
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = _build_taichu_flow_match_scheduler(noise_shift=5.0)
    count = int(scheduler.config.num_train_timesteps)
    expected = FlowMatchEulerDiscreteScheduler(shift=5.0)
    expected.set_timesteps(
        sigmas=torch.linspace(1.0, 0.0, count + 1, dtype=torch.float32)[:-1].tolist()
    )
    constructor_default = FlowMatchEulerDiscreteScheduler(shift=5.0)

    torch.testing.assert_close(scheduler.timesteps, expected.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(scheduler.sigmas, expected.sigmas, rtol=0, atol=0)
    assert scheduler.timesteps[178] != constructor_default.timesteps[178]


def test_bsmntw_weights_match_taichu_diffsynth_formula() -> None:
    scheduler = _build_taichu_flow_match_scheduler(noise_shift=3.0)
    actual = _build_taichu_bsmntw_weights(scheduler, noise_shift=3.0)
    sigmas = scheduler.sigmas[:1000]
    positions = sigmas * 1000
    bell = torch.exp(-2 * ((positions - 500) / 1000) ** 2)
    shifted = bell - bell.min()
    expected = shifted * (1000 / shifted.sum())

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (1000,)
    assert actual[0] == 0
    assert float(actual.sum()) == pytest.approx(1000.0)


def test_max_latents_schedule_uses_taichu_integer_interpolation() -> None:
    schedule = _schedule()

    assert schedule.value_at(-1) == 38
    assert schedule.value_at(0) == 38
    assert schedule.value_at(2000) == 94
    assert schedule.value_at(4000) == 150
    assert schedule.value_at(5000) == 150


def test_truncation_uses_generator_and_slices_every_aligned_stream() -> None:
    encoded = _encoded()
    first = truncate_encoded_latents(
        encoded,
        step=17,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    second = truncate_encoded_latents(
        encoded,
        step=99,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )

    selected = int(first["frame_latents"].shape[2])
    assert 3 <= selected <= 6
    assert int(second["frame_latents"].shape[2]) == selected
    for field in (
        "frame_latents",
        "condition_latents_rgb",
        "condition_latents_grid",
        "condition_latents_normal",
        "condition_fallback_latents_normal",
    ):
        assert int(first[field].shape[2]) == selected
        torch.testing.assert_close(first[field], second[field], rtol=0, atol=0)
    assert first["first_frame_latents"] is encoded["first_frame_latents"]
    assert first["ref_frame_latents"] is encoded["ref_frame_latents"]


def test_schedule_below_source_minimum_collapses_to_the_effective_cap() -> None:
    encoded = _encoded(frames=150)
    encoded["min_latents"] = 100
    encoded["max_latents"] = 150

    truncated = truncate_encoded_latents(
        encoded,
        step=0,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
        schedule=_schedule(),
    )

    assert truncated["frame_latents"].shape[2] == 38


def test_random_draws_do_not_advance_length_curriculum() -> None:
    encoded = _encoded(frames=150)
    encoded["min_latents"] = 100
    encoded["max_latents"] = 150
    generator = torch.Generator().manual_seed(42)
    torch.rand(5000, generator=generator)
    for _ in range(4):
        truncated = truncate_encoded_latents(
            encoded,
            step=0,
            generator=generator,
            device=torch.device("cpu"),
            schedule=_schedule(),
        )
        assert truncated["frame_latents"].shape[2] == 38


def test_ref_sampling_receives_the_truncated_frame_bank(monkeypatch) -> None:
    encoded = _encoded(frames=25)
    encoded.pop("ref_frame_latents")
    encoded["min_latents"] = 21
    encoded["max_latents"] = 21
    truncated = truncate_encoded_latents(
        encoded,
        step=0,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    observed: list[int] = []

    def sample(frame_latents, *, max_frames, generator):
        del max_frames, generator
        observed.append(int(frame_latents.shape[2]))
        return frame_latents[:, :, :1]

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.processors.sample_online_ref_frame_latents",
        sample,
    )
    processor = WanTaichuTi2vConditionProcessor(
        extras={"ref_num_frames_online": 6},
        precision="fp32",
        streams=("normal",),
        ref_token=True,
    )

    result = processor._resolve_ref_hidden_states(
        truncated,
        frame_latents=truncated["frame_latents"],
        generator=None,
    )

    assert observed == [21]
    assert result is not None and result.shape[2] == 1


def test_teacher_forcing_doubles_the_already_truncated_sequence() -> None:
    processor = WanTaichuTeacherForcingConditionProcessor(
        extras={},
        precision="fp32",
        streams=("normal",),
    )
    frame_latents = torch.zeros(1, 2, 3, 2, 2)
    condition = torch.ones_like(frame_latents)

    hidden = processor._assemble_hidden_states(
        noisy_latent=frame_latents,
        frame_latents=frame_latents,
        first_frame_latents=frame_latents[:, :, :1],
        conditions=[condition],
    )

    assert hidden.shape == (1, 4, 6, 2, 2)


class _EchoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(patch_size=(1, 1, 1))

    def forward(self, *, hidden_states, **kwargs):
        del kwargs
        return hidden_states * self.scale


class _CapturingProcessor:
    def __init__(self) -> None:
        self.lengths: list[int] = []

    def process(self, encoded, **kwargs):
        del kwargs
        frames = encoded["frame_latents"]
        self.lengths.append(int(frames.shape[2]))
        return {
            "model_inputs": {"hidden_states": torch.zeros_like(frames)},
            "targets": torch.zeros_like(frames),
            "loss_weights": torch.ones_like(frames),
        }


def test_train_step_truncates_after_canonicalization_before_processor() -> None:
    spec = require_variant("wan22_a14b")
    streams = ("rgb",)
    shape = (1, spec.z_dim, 5, 2, 2)
    batch = {
        "sample_id": ["a14b"],
        "prompt": torch.zeros(1, 3, 4),
        "frame_latents": torch.zeros(shape),
        stream_field("rgb"): torch.zeros(1, spec.z_dim + spec.mask_channels, 5, 2, 2),
        "min_latents": 3,
        "max_latents": 3,
    }
    request = TrainRequest(
        extras={"variant": spec.name, "condition_streams": list(streams)},
        precision="fp32",
    )
    processor = _CapturingProcessor()
    state = SimpleNamespace(
        is_batch_leader=True,
        batch_group=None,
        fsdp_group=None,
        config=SimpleNamespace(usp_size=1),
    )

    output = _train_step(
        _EchoModel(),
        batch,
        0,
        StepContext(device=torch.device("cpu"), parallel_state=state),
        request=request,
        spec=spec,
        streams=streams,
        codec=build_encoded_batch_codec(spec, streams),
        processor=processor,
        dropout=ConditionDropoutPolicy(spec=spec, streams=streams, extras=request.extras),
        encoder=None,
    )

    assert processor.lengths == [3]
    assert output.workload == {"local_tokens": 12, "local_batch": 1}


def test_train_step_preserves_taichu_bf16_residual_square_order() -> None:
    class PrecisionLossProcessor:
        def process(self, encoded, **kwargs):
            del encoded, kwargs
            hidden = torch.tensor([0.1], dtype=torch.bfloat16).reshape(1, 1, 1, 1, 1)
            return {
                "model_inputs": {"hidden_states": hidden},
                "targets": torch.zeros_like(hidden),
                "loss_weights": torch.ones_like(hidden, dtype=torch.float32),
            }

    spec = require_variant("wan22_a14b")
    streams = ("rgb",)
    shape = (1, spec.z_dim, 1, 1, 1)
    batch = {
        "sample_id": ["loss-order"],
        "prompt": torch.zeros(1, 1, 1),
        "frame_latents": torch.zeros(shape),
        stream_field("rgb"): torch.zeros(
            1,
            spec.z_dim + spec.mask_channels,
            1,
            1,
            1,
        ),
    }
    request = TrainRequest(
        extras={"variant": spec.name, "condition_streams": list(streams)},
        precision="bf16",
    )
    state = SimpleNamespace(
        is_batch_leader=True,
        batch_group=None,
        fsdp_group=None,
        config=SimpleNamespace(usp_size=4),
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = _train_step(
            _EchoModel(),
            batch,
            0,
            StepContext(device=torch.device("cpu"), parallel_state=state),
            request=request,
            spec=spec,
            streams=streams,
            codec=build_encoded_batch_codec(spec, streams),
            processor=PrecisionLossProcessor(),
            dropout=ConditionDropoutPolicy(spec=spec, streams=streams, extras=request.extras),
            encoder=None,
        )

    residual = torch.tensor([0.1], dtype=torch.bfloat16)
    taichu_order = (residual**2).float().mean()
    promoted_order = residual.float().square().mean()
    torch.testing.assert_close(output.loss, taichu_order, rtol=0, atol=0)
    assert output.loss != promoted_order
