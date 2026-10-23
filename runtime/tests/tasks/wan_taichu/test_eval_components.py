from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vforge.models.wan_taichu.encoder import WanTaichuConditionEncoder
from vforge.models.wan_taichu.inference.bidirectional import (
    WAN_TAICHU_GENERATOR_STATE_FIELD,
    WanTaichuBidirectionalInference,
    WanTaichuBidirectionalPipeline,
    _encode_reference_frames,
    _global_ref_present,
    build_per_token_timestep,
    encode_inference_batch,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task.base import resolve_bidirectional_request
from vforge.tasks.wan_taichu.eval_task.checkpoint import a14b_checkpoint_mapper
from vforge.tasks.wan_taichu.eval_task.reference import _prepare_reference_batch
from vforge.tasks.wan_taichu.variants import require_variant
from vforge.training.checkpoint import StateDictMapper, load_model_weights


def test_bidirectional_request_accepts_standalone_variants_and_rejects_causal() -> None:
    for variant in ("wan22_a14b", "wan22_ti2v_5b", "wan22_ti2v_5b_ref"):
        spec, streams, extras = resolve_bidirectional_request(
            EvalRequest(extras={"variant": variant})
        )
        assert spec.name == variant
        assert streams == spec.default_condition_streams
        assert extras["condition_streams"] == list(streams)

    with pytest.raises(NotImplementedError, match="currently supports"):
        resolve_bidirectional_request(
            EvalRequest(extras={"variant": "wan22_ti2v_5b_causal_teacher"})
        )


def test_eval_disables_random_camera_rotation() -> None:
    _, _, extras = resolve_bidirectional_request(
        EvalRequest(
            extras={
                "variant": "wan22_ti2v_5b_ref",
                "grid_random_rot_deg": 30.0,
            }
        )
    )

    assert extras["grid_random_rot_deg"] == 0.0


def test_online_ref_overrides_require_validation_ref_variant() -> None:
    with pytest.raises(ValueError, match="data_source=validation"):
        resolve_bidirectional_request(
            EvalRequest(
                extras={"variant": "wan22_ti2v_5b_ref"},
                data_source="inference",
                online_ref_num_frames=2,
            )
        )
    with pytest.raises(ValueError, match="ref-token"):
        resolve_bidirectional_request(
            EvalRequest(
                extras={"variant": "wan22_ti2v_5b"},
                data_source="validation",
                online_ref_frame_indices=(1,),
            )
        )


def test_per_token_timestep_keeps_first_frame_clean() -> None:
    latent = torch.zeros(1, 2, 3, 2, 2)

    timestep = build_per_token_timestep(latent, torch.tensor(700), (1, 1, 1))

    assert timestep.shape == (1, 12)
    assert torch.equal(timestep[0, :4], torch.zeros(4))
    assert torch.equal(timestep[0, 4:], torch.full((8,), 700.0))


class _Encoder:
    training_dtype = torch.float32
    raw_stream_keys = {"normal": "normal_condition"}

    def __init__(self) -> None:
        self.vae = nn.Conv3d(1, 1, 1)
        self.encoded_batch_sizes: list[int] = []

    def encode_prompt(self, _tokens, _mask, batch_size):
        return torch.ones(batch_size, 2, 3)

    @staticmethod
    def scale_pixels(value):
        return value * 2 - 1

    def encode_image(self, value):
        self.encoded_batch_sizes.append(int(value.shape[0]))
        return value.mean(dim=1, keepdim=True)


def test_inference_encoder_encodes_references_independently() -> None:
    encoder = _Encoder()
    batch = {
        "image_condition": torch.ones(1, 1, 3, 2, 2),
        "reference_condition": torch.rand(1, 2, 3, 2, 2),
        "normal_condition": torch.rand(1, 3, 3, 2, 2),
        "text_inputs": torch.ones(1, 2),
        "text_mask": torch.ones(1, 2),
        "num_frames": 3,
    }

    encoded = encode_inference_batch(
        encoder,
        batch,
        streams=("normal",),
        stream_fields={"normal": "condition_latents_normal"},
    )

    assert encoded["first_frame_latents"].shape == (1, 1, 1, 2, 2)
    assert encoded["ref_frame_latents"].shape == (1, 1, 2, 2, 2)
    assert encoded["condition_latents_normal"].shape == (1, 1, 3, 2, 2)
    assert 2 in encoder.encoded_batch_sizes


def test_inference_reference_latents_preserve_channels_last_3d() -> None:
    encoder = SimpleNamespace(
        scale_pixels=lambda value: value,
        encode_image=lambda value: value,
    )

    encoded = _encode_reference_frames(
        encoder,
        torch.rand(1, 2, 3, 4, 5),
    )

    assert encoded.shape == (1, 3, 2, 4, 5)
    assert encoded.is_contiguous(memory_format=torch.channels_last_3d)


def test_a14b_inference_encoder_does_not_require_target_video() -> None:
    encoder = object.__new__(WanTaichuConditionEncoder)
    nn.Module.__init__(encoder)
    encoder.encoding = "wan21"
    encoder.vae = nn.Conv3d(3, 16, 1)
    encoder.vae.config = SimpleNamespace(scale_factor_temporal=4)
    encoder.training_dtype = torch.float32
    encoder.streams = ("rgb", "grid", "normal")
    encoder.raw_stream_keys = {
        "grid": "grid_condition",
        "normal": "normal_condition",
    }
    encoder.mask_channels = 4
    encoder.use_vae_condition = False
    encoder.use_clip_condition = False
    encoder.use_dino_condition = False
    encoder.scale_pixels = lambda value: value
    encoder.encode_prompt = lambda _tokens, _mask, batch: torch.ones(batch, 2, 3)
    encoder.encode_image = lambda value: value.mean(dim=1, keepdim=True)[:, :, ::4].repeat(
        1, 16, 1, 1, 1
    )
    batch = {
        "image_condition": torch.ones(1, 1, 3, 2, 2),
        "normal_condition": torch.ones(1, 5, 3, 2, 2),
        "normal_fg_condition": torch.ones(1, 5, 3, 2, 2),
        "text_inputs": torch.ones(1, 2),
        "text_mask": torch.ones(1, 2),
        "num_frames": 5,
    }

    encoded = encoder.encode_inference_batch(batch)

    assert "frame_latents" not in encoded
    assert encoded["condition_latents_rgb"].shape == (1, 20, 2, 2, 2)
    assert encoded["condition_latents_grid"].shape == (1, 20, 2, 2, 2)
    assert torch.all(encoded["condition_latents_grid"][:, :4] == 1)
    assert torch.count_nonzero(encoded["condition_latents_grid"][:, 4:]) == 0
    assert encoded["condition_latents_normal"].shape == (1, 20, 2, 2, 2)
    assert encoded["normal_fg_body"].shape == (1, 16, 2, 2, 2)


def test_online_ref_selection_preserves_taichu_generator_state(monkeypatch) -> None:
    video = torch.arange(240, dtype=torch.float32).view(240, 1, 1, 1)
    monkeypatch.setattr(
        "vforge.data.video_io.read_video_tensor",
        lambda *_args, **_kwargs: video,
    )
    batch = {}

    context = _prepare_reference_batch(
        batch,
        {
            "video": "/data/target.mp4",
            "__vforge_eval_index__": 0,
            "__vforge_eval_seed_offset__": 114,
        },
        base_seed=42,
        spec=require_variant("wan22_ti2v_5b_ref"),
        extras={"ref_num_frames_online": 6},
        device=torch.device("cpu"),
    )

    expected = torch.Generator().manual_seed(156)
    count = int(torch.randint(0, 4, (1,), generator=expected).item())
    segment = 240 / count if count else 0
    expected_indices = []
    for slot in range(count):
        lower = int(slot * segment)
        upper = max(int((slot + 1) * segment), lower + 1)
        expected_indices.append(
            lower + int(torch.randint(0, upper - lower, (1,), generator=expected).item())
        )

    selection = context["__wan_taichu_ref_selection__"]
    assert selection["seed"] == 156
    assert selection["seed_offset"] == 114
    assert selection["indices"] == expected_indices
    assert torch.equal(context[WAN_TAICHU_GENERATOR_STATE_FIELD], expected.get_state())
    if expected_indices:
        assert torch.equal(batch["reference_condition"][0], video[expected_indices])


def test_online_ref_selection_accepts_native_gameplay_video_tensor(monkeypatch) -> None:
    video = torch.arange(20, dtype=torch.float32).view(5, 1, 2, 2)
    monkeypatch.setattr(
        "vforge.data.video_io.read_video_tensor",
        lambda *_args, **_kwargs: pytest.fail("native tensor must not be decoded again"),
    )
    batch = {}

    context = _prepare_reference_batch(
        batch,
        {
            "video_frames": video,
            "__vforge_eval_seed_offset__": 3,
        },
        base_seed=42,
        spec=require_variant("wan22_ti2v_5b_ref"),
        extras={"ref_num_frames_online": 6},
        device=torch.device("cpu"),
        frame_indices_override=(1, 4),
    )

    assert context["__wan_taichu_ref_selection__"]["indices"] == [1, 4]
    assert torch.equal(batch["reference_condition"][0], video[[1, 4]])


def test_explicit_ref_priority_and_overrides_control_target_video_selection(monkeypatch) -> None:
    video = torch.arange(240, dtype=torch.float32).view(240, 1, 1, 1)
    reads = []

    def read_video(*_args, **_kwargs):
        reads.append(True)
        return video

    monkeypatch.setattr(
        "vforge.data.video_io.read_video_tensor",
        read_video,
    )

    explicit = torch.ones(1, 2, 1, 1, 1)
    explicit_batch = {"reference_condition": explicit}
    explicit_context = _prepare_reference_batch(
        explicit_batch,
        {"video": "/data/target.mp4"},
        base_seed=42,
        spec=require_variant("wan22_ti2v_5b_ref"),
        extras={"ref_num_frames_online": 6},
        device=torch.device("cpu"),
        frame_indices_override=(1, 3),
    )
    assert explicit_context["__wan_taichu_ref_selection__"]["source"] == "explicit"
    assert explicit_batch["reference_condition"] is explicit
    assert reads == []

    for num_frames, frame_indices, expected_count, expected_mode in (
        (2, None, 2, "num_frames"),
        (None, (3, 9), 2, "frame_indices"),
        (0, None, 0, "num_frames"),
    ):
        batch = {}
        context = _prepare_reference_batch(
            batch,
            {"video": "/data/target.mp4"},
            base_seed=42,
            spec=require_variant("wan22_ti2v_5b_ref"),
            extras={"ref_num_frames_online": 6},
            device=torch.device("cpu"),
            num_frames_override=num_frames,
            frame_indices_override=frame_indices,
        )
        selection = context["__wan_taichu_ref_selection__"]
        assert (selection["count"], selection["mode"]) == (expected_count, expected_mode)
        if frame_indices is not None:
            assert selection["indices"] == list(frame_indices)
        if selection["indices"]:
            assert torch.equal(batch["reference_condition"][0], video[selection["indices"]])
        else:
            assert "reference_condition" not in batch


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=2, patch_size=(1, 1, 1))


class _Scheduler:
    timesteps = torch.tensor([2.0, 1.0])

    def set_timesteps(self, _steps, *, device):
        self.timesteps = self.timesteps.to(device)

    def step(self, prediction, _timestep, latents, *, return_dict):
        assert return_dict is False
        return (latents - prediction * 0.1,)


def test_bidirectional_pipeline_keeps_inpaint_and_passes_cfg_ref() -> None:
    model = _Model()
    calls = []

    def engine_forward(**kwargs):
        calls.append(kwargs)
        prompt_value = kwargs["encoder_hidden_states"].mean()
        shape = (kwargs["hidden_states"].shape[0], 1, *kwargs["hidden_states"].shape[2:])
        return torch.ones(shape) * prompt_value

    pipeline = WanTaichuBidirectionalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=_Scheduler(),
        vae_scale_factor_temporal=1,
        vae_scale_factor_spatial=1,
    )
    first = torch.full((1, 1, 1, 2, 2), 7.0)
    reference = torch.ones(1, 1, 2, 2, 2)

    result = pipeline.call_with_latent(
        torch.ones(1, 2, 3),
        condition=torch.ones(1, 1, 3, 2, 2),
        first_frame_latents=first,
        height=2,
        width=2,
        num_frames=3,
        num_inference_steps=2,
        guidance_scale=2.0,
        generator=torch.Generator().manual_seed(1),
        negative_prompt_embeds=torch.zeros(1, 2, 3),
        ref_hidden_states=reference,
        latents=torch.zeros(1, 1, 3, 2, 2),
        decode_output=False,
    )

    assert torch.equal(result[:, :, :1], first)
    assert len(calls) == 4
    assert {call["global_has_ref"] for call in calls} == {True}
    assert all(call["ref_hidden_states"] is reference for call in calls)
    assert all(torch.count_nonzero(call["timestep"][:, :4]) == 0 for call in calls)


def test_a14b_pipeline_routes_scalar_timesteps_across_experts() -> None:
    class DualExpertModel(_Model):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.config = SimpleNamespace(in_channels=4, patch_size=(1, 1, 1))

        @property
        def dtype(self):
            return self.weight.dtype

        @staticmethod
        def cache_context(_name):
            return nullcontext()

        @staticmethod
        def route(timestep):
            return "high" if float(timestep.flatten()[0]) >= 500 else "low"

    class Scheduler(_Scheduler):
        def __init__(self) -> None:
            self.timesteps = torch.tensor([900.0, 100.0])
            self.latent_dtypes = []

        def step(self, prediction, timestep, latents, *, return_dict):
            self.latent_dtypes.append(latents.dtype)
            return super().step(prediction, timestep, latents, return_dict=return_dict)

    model = DualExpertModel()
    routes = []

    def engine_forward(**kwargs):
        timestep = kwargs["timestep"]
        assert timestep.shape == (1,)
        assert kwargs["hidden_states"].dtype is torch.bfloat16
        routes.append(model.route(timestep))
        hidden = kwargs["hidden_states"]
        return torch.zeros(hidden.shape[0], 1, *hidden.shape[2:])

    pipeline = WanTaichuBidirectionalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=Scheduler(),
        first_frame_inpaint=False,
        vae_scale_factor_temporal=1,
        vae_scale_factor_spatial=1,
    )
    result = pipeline.call_with_latent(
        torch.ones(1, 2, 3, dtype=torch.bfloat16),
        condition=torch.ones(1, 3, 1, 2, 2),
        first_frame_latents=None,
        encoder_hidden_states_image_vae=None,
        encoder_hidden_states_image_clip=None,
        encoder_hidden_states_image_dino=None,
        height=2,
        width=2,
        num_frames=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(1),
        negative_prompt_embeds=None,
        latents=torch.zeros(1, 1, 1, 2, 2),
        decode_output=False,
    )

    assert result.shape == (1, 1, 1, 2, 2)
    assert result.dtype is torch.float32
    assert pipeline.scheduler.latent_dtypes == [torch.float32, torch.float32]
    assert routes == ["high", "low"]


def test_eval_ref_presence_uses_the_fsdp_group(monkeypatch) -> None:
    group = object()
    observed = []

    monkeypatch.setattr(
        "vforge.models.wan_taichu.inference.bidirectional.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vforge.models.wan_taichu.inference.bidirectional.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(fsdp_group=group),
    )

    def all_reduce(flag, *, op, group):
        observed.append((op, group))
        flag.fill_(1)

    monkeypatch.setattr(
        "vforge.models.wan_taichu.inference.bidirectional.dist.all_reduce",
        all_reduce,
    )

    assert _global_ref_present(None, device=torch.device("cpu")) is True
    assert observed == [(torch.distributed.ReduceOp.MAX, group)]


def test_bidirectional_decode_uses_the_batch_leader(monkeypatch) -> None:
    observed = {}
    advanced = torch.Generator().manual_seed(1)
    torch.rand(3, generator=advanced)
    generator_state = advanced.get_state()
    expected_next = torch.rand(1, generator=advanced)

    class Pipeline:
        def call_with_latent(self, *_args, **kwargs):
            observed["decode_output"] = kwargs["decode_output"]
            observed["next_random"] = torch.rand(1, generator=kwargs["generator"])
            return torch.ones(1, 3, 1, 1, 1)

    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(is_batch_leader=False),
    )
    inference = WanTaichuBidirectionalInference(
        model=_Model(),
        encode_condition=lambda _sample: {},
        pipeline=Pipeline(),
        stream_fields={"normal": "condition_latents_normal"},
        num_inference_steps=1,
        guidance_scale=1.0,
        negative_prompt="",
    )
    assert inference.resolve_sample_seed({"__vforge_eval_seed_offset__": 7}, 42) == 49
    assert inference.resolve_sample_seed({"__vforge_eval_index__": 3}, 42) == 45
    result = inference.infer_sample(
        {
            "sample_id": "sample",
            "__height__": 1,
            "__width__": 1,
            "__num_frames__": 1,
            WAN_TAICHU_GENERATOR_STATE_FIELD: generator_state,
            "__encoded__": {
                "prompt": torch.zeros(1, 1, 1),
                "first_frame_latents": torch.zeros(1, 1, 1, 1, 1),
                "condition_latents_normal": torch.zeros(1, 1, 1, 1, 1),
            },
        },
        seed=1,
    )

    assert observed["decode_output"] is False
    assert torch.equal(observed["next_random"], expected_next)
    assert result["video"] is None


def test_taichu_weights_only_pt_loads_directly_into_native_model(tmp_path: Path) -> None:
    class NativeRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(2, 2)

    model = NativeRoot()
    taichu_state = {
        key: torch.full_like(value, index + 1)
        for index, (key, value) in enumerate(model.transformer.state_dict().items())
    }
    checkpoint = tmp_path / "11400.pt"
    torch.save(taichu_state, checkpoint)

    load_model_weights(
        model,
        str(checkpoint),
        format="auto",
        state_dict_mapper=StateDictMapper.strip_prefix("transformer."),
        strict=True,
    )

    assert set(taichu_state) == set(model.transformer.state_dict())
    for key, expected in taichu_state.items():
        assert torch.equal(model.transformer.state_dict()[key], expected)


def test_a14b_eval_checkpoint_mapper_loads_both_taichu_experts(tmp_path: Path) -> None:
    class DualRoot(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wan = nn.Linear(2, 2, bias=False)
            self.wan_2 = nn.Linear(2, 2, bias=False)

    model = DualRoot()
    checkpoint = tmp_path / "a14b.pt"
    torch.save(
        {
            "module.wan.weight": torch.full_like(model.wan.weight, 3),
            "module.wan_2.weight": torch.full_like(model.wan_2.weight, 7),
        },
        checkpoint,
    )

    load_model_weights(
        model,
        str(checkpoint),
        format="pt",
        state_dict_mapper=a14b_checkpoint_mapper(),
        strict=True,
    )

    assert torch.all(model.wan.weight == 3)
    assert torch.all(model.wan_2.weight == 7)


def test_new_eval_modules_do_not_import_official_wan_eval_or_inference() -> None:
    root = Path(__file__).parents[3] / "vforge"
    paths = [
        root / "models" / "wan_taichu" / "inference" / "bidirectional.py",
        root / "tasks" / "wan_taichu" / "eval.py",
        root / "tasks" / "wan_taichu" / "eval_task" / "base.py",
        root / "tasks" / "wan_taichu" / "eval_task" / "reference.py",
        root / "tasks" / "wan_taichu" / "eval_task" / "bidirectional.py",
    ]
    imported = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )

    assert not any(
        name == "vforge.models.wan" or name.startswith("vforge.models.wan.") for name in imported
    )
    assert not any(
        name == "vforge.tasks.wan" or name.startswith("vforge.tasks.wan.") for name in imported
    )
