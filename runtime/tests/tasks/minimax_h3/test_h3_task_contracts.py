"""CPU contracts for MiniMax-H3 cache and flow semantics."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from vforge.data.dataset import EncodedCacheDataset
from vforge.offline_encoder import EncodeComponents
from vforge.tasks.minimax_h3.eval import _prepare_model_for_eval, build_eval_components
from vforge.tasks.minimax_h3.shared import (
    H3_ENCODED_BATCH_CODEC,
    _build_h3_model,
)
from vforge.tasks.minimax_h3.train import (
    _h3_pretrained_source,
    _optimizer_factory,
    flow_noisy_and_target,
    shifted_sigma,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan.v21.shared import WAN21_ENCODED_BATCH_CODEC


def test_h3_builds_native_implementation(tmp_path) -> None:
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    (transformer_dir / "config.json").write_text(
        json.dumps(
            {
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "hidden_size": 24,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": 32,
                "in_channels": 4,
                "audio_in_channels": 6,
                "patch_size": [1, 2, 2],
                "text_dim": 8,
                "freq_dim": 8,
                "time_embed_hidden_dim": 24,
                "time_embed_dim": 16,
                "rope_freq_dim": 2,
            }
        ),
        encoding="utf-8",
    )

    model = _build_h3_model(
        extras={},
        init_device="meta",
        pretrained_path=str(tmp_path),
    )

    assert ".native." in type(model).__module__
    transformer = model.transformer
    assert {
        parameter.dtype for parameter in transformer.parameters() if parameter.is_floating_point()
    } == {torch.float32}
    assert transformer.proj_in.weight.dtype is torch.float32
    assert transformer.time_embedder.linear_1.weight.dtype is torch.float32
    assert transformer.proj_out.weight.dtype is torch.float32
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert transformer.gradient_checkpointing
    assert transformer._gradient_checkpointing_func.keywords == {"use_reentrant": False}


def test_dual_shift_uses_one_base_sample() -> None:
    u = torch.tensor(0.25)
    video = shifted_sigma(u, 12.0)
    audio = shifted_sigma(u, 3.0)
    torch.testing.assert_close(video, torch.tensor(0.8))
    torch.testing.assert_close(audio, torch.tensor(0.5))
    assert video > audio > u


def test_h3_velocity_target_is_data_ward_and_sigma_enters_input() -> None:
    clean = torch.tensor([2.0])
    noise = torch.tensor([-1.0])
    noisy, target = flow_noisy_and_target(clean, noise, torch.tensor(0.25))
    torch.testing.assert_close(noisy, torch.tensor([1.25]))
    torch.testing.assert_close(target, torch.tensor([3.0]))
    torch.testing.assert_close(noisy + 0.25 * target, clean)


def test_h3_codec_does_not_project_through_wan_schema() -> None:
    bundle = {key: torch.tensor(0) for key in H3_ENCODED_BATCH_CODEC.required_keys}
    canonical = H3_ENCODED_BATCH_CODEC.canonicalize(bundle)
    assert set(canonical) == set(H3_ENCODED_BATCH_CODEC.keys)
    with pytest.raises(ValueError, match="wan21"):
        WAN21_ENCODED_BATCH_CODEC.canonicalize(bundle)


def test_codec_is_mandatory_at_both_cache_boundaries() -> None:
    with pytest.raises(TypeError):
        EncodedCacheDataset(".", [])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        EncodeComponents(  # type: ignore[call-arg]
            dataset=[],
            sample_ids=(),
            encode_condition=lambda sample: sample,
            cache_dir=".",
            encoding_fingerprint={},
        )


def test_h3_stages_transformer_weights_to_node_local_storage() -> None:
    source = _h3_pretrained_source("/models/MiniMax-H3", "distributed")
    assert source.path == "/models/MiniMax-H3/transformer"
    assert source.subfolder == ""
    assert source.stage_to_local


def test_h3_fallback_loads_fp32_weights(monkeypatch) -> None:
    observed = {}

    class FakeTransformer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            observed.update(args=args, kwargs=kwargs)
            return nn.Linear(1, 1)

    monkeypatch.setattr("diffusers.MiniMaxH3Transformer3DModel", FakeTransformer)
    source = _h3_pretrained_source("/models/MiniMax-H3", "auto")

    assert source.fallback_model_factory is not None
    source.fallback_model_factory()
    assert observed["kwargs"]["torch_dtype"] is torch.float32


def test_h3_adamw_state_is_fp32() -> None:
    model = nn.Linear(2, 2, dtype=torch.float32)
    optimizer = _optimizer_factory(
        learning_rate=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )(model)
    assert optimizer is not None

    model(torch.ones(1, 2)).square().mean().backward()
    optimizer.step()

    for state in optimizer.state.values():
        assert state["exp_avg"].dtype is torch.float32
        assert state["exp_avg_sq"].dtype is torch.float32


def test_eval_freezes_mixed_dtype_parameters_before_fsdp() -> None:
    model = nn.Sequential(
        nn.Linear(2, 2, dtype=torch.bfloat16),
        nn.LayerNorm(2, dtype=torch.float32),
    ).train()

    _prepare_model_for_eval(model)

    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_h3_eval_rejects_unknown_data_source() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation.data_source"):
        build_eval_components(
            EvalRequest(
                data_source="unknown",
                manifest="/data/input.jsonl",
            )
        )


def test_h3_eval_cache_requires_internal_manifest(tmp_path) -> None:
    with pytest.raises(ValueError, match="manifest.json"):
        build_eval_components(
            EvalRequest(
                data_source="cache",
                manifest=str(tmp_path / "manifest.json"),
            )
        )
