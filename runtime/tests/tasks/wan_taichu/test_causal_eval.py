from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn

from vforge.models.causal import get_incremental_kv_cache
from vforge.models.wan_taichu.inference.causal import (
    FULL_RECOMPUTE,
    INCREMENTAL_KV,
    WanTaichuCausalInferencePipeline,
    WanTaichuCausalPipeline,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task.causal import (
    build_causal_eval_components,
    causal_pipeline_kwargs,
)


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(patch_size=(1, 1, 1), in_channels=1)

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    @contextmanager
    def cache_context(self, _name: str):
        yield


class _OneStepScheduler:
    timesteps: torch.Tensor

    def set_timesteps(self, _steps: int, *, device: torch.device) -> None:
        self.timesteps = torch.tensor([1.0], device=device)

    def step(self, prediction, _timestep, sample, *, return_dict: bool):
        assert return_dict is False
        return (sample - prediction,)


def test_causal_inference_uses_taichu_dataset_index_seed() -> None:
    inference = WanTaichuCausalInferencePipeline(
        model=_Transformer(),
        encode_condition=lambda _sample: {},
        pipeline=None,
        stream_fields={},
    )

    assert inference.resolve_sample_seed({"__vforge_eval_seed_offset__": 7}, 42) == 49
    assert inference.resolve_sample_seed({"__vforge_eval_index__": 3}, 42) == 45


def test_full_recompute_uses_generated_history_and_zero_future() -> None:
    transformer = _Transformer()
    calls: list[torch.Tensor] = []

    def engine_forward(**kwargs):
        hidden = kwargs["hidden_states"].detach().clone()
        calls.append(hidden)
        frames = hidden.shape[2] // 2
        return torch.ones(hidden.shape[0], 1, frames, 1, 1)

    pipeline = WanTaichuCausalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        engine_forward=engine_forward,
        scheduler=_OneStepScheduler(),
        frame_per_block=1,
        first_frame_own_block=True,
        rollout_mode=FULL_RECOMPUTE,
        usp_size=1,
    )
    result = pipeline.call_with_latent(
        torch.zeros(1, 2, 4),
        y=None,
        first_frame_latents=torch.full((1, 1, 1, 1, 1), -5.0),
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.0,
        latents=torch.full((1, 1, 3, 1, 1), 10.0),
        decode_output=False,
    )

    assert len(calls) == 2
    second_clean = calls[1][:, :, :3]
    torch.testing.assert_close(second_clean[:, :, 0], torch.tensor([[[[-5.0]]]]))
    torch.testing.assert_close(second_clean[:, :, 1], torch.tensor([[[[9.0]]]]))
    torch.testing.assert_close(second_clean[:, :, 2], torch.tensor([[[[0.0]]]]))
    torch.testing.assert_close(result.flatten(), torch.tensor([-5.0, 9.0, 9.0]))


def test_a14b_full_recompute_generates_block_zero_without_first_frame() -> None:
    calls: list[torch.Tensor] = []

    def engine_forward(**kwargs):
        hidden = kwargs["hidden_states"].detach().clone()
        calls.append(hidden)
        frames = hidden.shape[2] // 2
        return torch.ones(hidden.shape[0], 1, frames, 1, 1)

    pipeline = WanTaichuCausalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=_Transformer(),
        engine_forward=engine_forward,
        scheduler=_OneStepScheduler(),
        frame_per_block=1,
        first_frame_own_block=False,
        rollout_mode=FULL_RECOMPUTE,
        usp_size=1,
    )
    result = pipeline.call_with_latent(
        torch.zeros(1, 2, 4),
        y=None,
        first_frame_latents=None,
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.0,
        latents=torch.full((1, 1, 3, 1, 1), 10.0),
        decode_output=False,
    )

    assert len(calls) == 3
    torch.testing.assert_close(result.flatten(), torch.tensor([9.0, 9.0, 9.0]))


class _RecordingIncremental:
    def __init__(self) -> None:
        self.caches: list[object] = []
        self.cache_dtypes: list[torch.dtype] = []
        self.calls: list[object] = []

    def init_kv_cache(self, **kwargs):
        cache = object()
        self.caches.append(cache)
        self.cache_dtypes.append(kwargs["dtype"])
        return cache

    def full_clip_rotary_emb(self, hidden_states):
        tokens = hidden_states.shape[2]
        rope = torch.zeros(1, tokens, 1, 1)
        return rope, rope

    def forward_kv_block(self, hidden_states, _timestep, _text, **kwargs):
        self.calls.append(kwargs["kv_cache"])
        return torch.zeros(
            hidden_states.shape[0],
            1,
            hidden_states.shape[2],
            hidden_states.shape[3],
            hidden_states.shape[4],
        )


class _ResettableCache:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class _RecordingMoeIncremental(_RecordingIncremental):
    def init_kv_cache(self, **kwargs):
        cache = _ResettableCache()
        self.caches.append(cache)
        self.cache_dtypes.append(kwargs["dtype"])
        return cache


def test_incremental_cfg_allocates_independent_branch_caches() -> None:
    executor = _RecordingIncremental()

    def engine_forward(**kwargs):
        assert kwargs.pop("incremental_kv") is True
        kwargs["kv_cache"] = get_incremental_kv_cache()
        return executor.forward_kv_block(
            kwargs.pop("hidden_states"),
            kwargs.pop("timestep"),
            kwargs.pop("encoder_hidden_states"),
            **kwargs,
        )

    pipeline = WanTaichuCausalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=_Transformer(),
        engine_forward=engine_forward,
        scheduler=_OneStepScheduler(),
        frame_per_block=1,
        first_frame_own_block=True,
        rollout_mode=INCREMENTAL_KV,
        usp_size=1,
        incremental_transformer=executor,
    )
    pipeline.call_with_latent(
        torch.ones(1, 2, 4, dtype=torch.bfloat16),
        y=None,
        first_frame_latents=torch.zeros(1, 1, 1, 1, 1),
        height=16,
        width=16,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=2.0,
        negative_prompt_embeds=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        latents=torch.ones(1, 1, 3, 1, 1),
        decode_output=False,
    )
    assert len(executor.caches) == 2
    assert executor.cache_dtypes == [torch.bfloat16, torch.bfloat16]
    assert executor.caches[0] is not executor.caches[1]
    assert set(executor.calls) == set(executor.caches)


def test_a14b_incremental_resets_and_replays_on_expert_switch() -> None:
    class DualTransformer(_Transformer):
        wan_2 = object()
        boundary_timestep = 900.0

        def primary_transformer(self):
            return self

    class TwoExpertScheduler(_OneStepScheduler):
        def set_timesteps(self, _steps: int, *, device: torch.device) -> None:
            self.timesteps = torch.tensor([950.0, 850.0], device=device)

    executor = _RecordingMoeIncremental()
    expert_calls: list[str | None] = []

    def engine_forward(**kwargs):
        assert kwargs.pop("incremental_kv") is True
        kwargs["kv_cache"] = get_incremental_kv_cache()
        expert_calls.append(kwargs.pop("expert_name"))
        return executor.forward_kv_block(
            kwargs.pop("hidden_states"),
            kwargs.pop("timestep"),
            kwargs.pop("encoder_hidden_states"),
            **kwargs,
        )

    pipeline = WanTaichuCausalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=DualTransformer(),
        engine_forward=engine_forward,
        scheduler=TwoExpertScheduler(),
        frame_per_block=1,
        first_frame_own_block=False,
        rollout_mode=INCREMENTAL_KV,
        usp_size=1,
        incremental_transformer=executor,
    )
    pipeline.call_with_latent(
        torch.ones(1, 2, 4),
        y=None,
        first_frame_latents=None,
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        latents=torch.ones(1, 1, 2, 1, 1),
        decode_output=False,
    )

    assert executor.caches[0].resets == 4
    assert expert_calls.count("wan") >= 2
    assert expert_calls.count("wan_2") >= 2


def test_causal_decode_uses_the_batch_leader(monkeypatch) -> None:
    observed = {}

    class Pipeline:
        rollout_mode = FULL_RECOMPUTE
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 16
        frame_per_block = 1
        first_frame_own_block = True

        @staticmethod
        def call_with_latent(*_args, **kwargs):
            observed["decode_output"] = kwargs["decode_output"]
            return torch.ones(1, 3, 1, 1, 1)

        @staticmethod
        def align_num_frames(num_frames):
            return num_frames

        @staticmethod
        def block_layout(_latent_frames):
            return ((0, 1),)

    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(is_batch_leader=False),
    )
    inference = WanTaichuCausalInferencePipeline(
        model=_Transformer(),
        encode_condition=lambda _sample: {},
        pipeline=Pipeline(),
        stream_fields={"normal": "condition_latents_normal"},
        num_inference_steps=1,
        guidance_scale=1.0,
    )
    result = inference.infer_sample(
        {
            "sample_id": "sample",
            "__height__": 16,
            "__width__": 16,
            "__num_frames__": 1,
            "__encoded__": {
                "prompt": torch.zeros(1, 1, 1),
                "first_frame_latents": torch.zeros(1, 1, 1, 1, 1),
                "condition_latents_normal": torch.zeros(1, 1, 1, 1, 1),
            },
        },
        seed=1,
    )

    assert observed["decode_output"] is False
    assert result["video"] is None


def test_causal_task_builder_passes_validated_pipeline_settings(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def build_base(request, **kwargs):
        captured["request"] = request
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.eval_task.causal.build_base_eval_components",
        build_base,
    )

    request = EvalRequest(
        extras={
            "variant": "wan22_ti2v_5b_causal_teacher",
            "frame_per_block": 2,
        },
        rollout_mode=FULL_RECOMPUTE,
        usp_size=4,
    )
    result = build_causal_eval_components(request)
    assert result is sentinel
    assert captured["pipeline_kwargs"]["frame_per_block"] == 2
    assert captured["pipeline_kwargs"]["usp_size"] == 4
    assert captured["pipeline_kwargs"]["execution_dtype"] is torch.bfloat16
    assert captured["supported_variants"] == (
        "wan22_ti2v_5b_causal_teacher",
        "wan22_a14b_causal_teacher",
    )

    settings = causal_pipeline_kwargs(
        EvalRequest(
            extras={"variant": "wan22_ti2v_5b_causal_teacher"},
            rollout_mode=INCREMENTAL_KV,
            usp_size=2,
        )
    )
    assert settings["usp_size"] == 2


def test_a14b_causal_task_uses_uniform_blocks_and_spatial_factor_eight(monkeypatch) -> None:
    captured = {}

    def build_base(_request, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "vforge.tasks.wan_taichu.eval_task.causal.build_base_eval_components",
        build_base,
    )
    request = EvalRequest(
        extras={
            "variant": "wan22_a14b_causal_teacher",
            "normal_drop_prob": 1.0,
        },
        rollout_mode=FULL_RECOMPUTE,
    )

    build_causal_eval_components(request)

    assert captured["supported_variants"] == (
        "wan22_ti2v_5b_causal_teacher",
        "wan22_a14b_causal_teacher",
    )
    assert captured["pipeline_kwargs"]["first_frame_own_block"] is False
    assert captured["pipeline_kwargs"]["vae_scale_factor_spatial"] == 8
    assert captured["pipeline_kwargs"]["execution_dtype"] is torch.bfloat16
    normal = torch.ones(1, 20, 3, 1, 1)
    prepared = captured["prepare_encoded"](
        {
            "condition_latents_rgb": normal.clone(),
            "condition_latents_grid": normal.clone(),
            "condition_latents_normal": normal,
            "condition_fallback_latents_normal": torch.zeros_like(normal),
        },
        None,
        8,
        8,
        9,
    )
    torch.testing.assert_close(
        prepared["condition_latents_normal"],
        torch.zeros_like(normal),
    )
