from __future__ import annotations

import json

import pytest
import torch

from vforge.tasks.requests import TrainingModelRequest, TrainRequest
from vforge.tasks.wan_taichu.dmd.integration import _validate_dmd_usp


@pytest.mark.parametrize("architecture,z_dim", [("wan22_a14b", 16), ("wan22_ti2v_5b", 48)])
@pytest.mark.parametrize("role", ["generator", "real_score", "fake_score"])
def test_dmd_role_weights_stay_fp32_with_bf16_compute(tmp_path, architecture, z_dim, role):
    from vforge.tasks.wan_taichu.dmd.integration import build_wan_role_model

    config = {
        "patch_size": [1, 2, 2],
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": z_dim,
        "out_channels": z_dim,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }
    for subfolder in ("transformer", "transformer_2"):
        directory = tmp_path / subfolder
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
    request = TrainRequest(
        precision="bf16",
        training_models={
            name: TrainingModelRequest(
                model="wan_taichu",
                pretrained_path=str(tmp_path),
                config={
                    "variant": architecture + ("_causal_teacher" if name == "generator" else ""),
                },
            )
            for name in ("generator", "real_score", "fake_score")
        },
    )
    model = build_wan_role_model(request, role)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    assert all(parameter.is_meta for parameter in model.parameters())


def _request(*, usp_size: int, heads: int) -> TrainRequest:
    return TrainRequest(
        usp_size=usp_size,
        training_models={
            "generator": TrainingModelRequest(
                model="wan_taichu",
                config={"num_attention_heads": heads},
            )
        },
    )


@pytest.mark.parametrize(
    ("architecture", "heads"),
    (("wan22_a14b", 40), ("wan22_ti2v_5b", 24)),
)
def test_both_dmd_generators_shard_their_rollout_under_ulysses(
    architecture: str,
    heads: int,
) -> None:
    """Both DMD generators are Taichu towers behind one block forward.

    The 5B was refused here while that forward had no slice or gather, so the
    whitelist earned a test: the two architectures now differ only in geometry,
    and naming one of them sends the other back to ``usp_size=1`` with an
    error message that blames the model instead of the list.
    """
    _validate_dmd_usp(_request(usp_size=8, heads=heads), architecture)


def test_dmd_keeps_refusing_ulysses_where_no_rollout_can_shard() -> None:
    _validate_dmd_usp(_request(usp_size=1, heads=40), "wan21")
    with pytest.raises(NotImplementedError, match=r"parallel\.usp_size=1"):
        _validate_dmd_usp(_request(usp_size=8, heads=40), "wan21")


def test_dmd_rejects_a_ulysses_width_that_splits_the_heads_unevenly() -> None:
    """24 heads divide by 8 but not by 5, and the cache is allocated per head."""
    _validate_dmd_usp(_request(usp_size=8, heads=24), "wan22_ti2v_5b")
    with pytest.raises(ValueError, match="num_attention_heads=24"):
        _validate_dmd_usp(_request(usp_size=5, heads=24), "wan22_ti2v_5b")


@pytest.mark.parametrize("lightning", [False, True])
def test_a14b_expert_weights_use_existing_loader(tmp_path, lightning):
    import torch
    from torch import nn

    from vforge.models.loading import load_pretrained_weights
    from vforge.tasks.wan_taichu.dmd.integration import build_wan_weight_sources

    prefixes = (
        ("generator.model.transformer.", "generator_2.model.transformer.")
        if lightning
        else ("wan.", "wan_2.")
    )
    path = tmp_path / "weights.pt"
    state = {
        prefix + "weight": torch.full((2, 2), float(i + 1)) for i, prefix in enumerate(prefixes)
    }
    torch.save(state, path)
    request = TrainRequest(
        training_models={
            "generator": TrainingModelRequest(
                model="wan_taichu",
                init_weights=str(path),
                config={"variant": "wan22_a14b_causal_teacher"},
            ),
        }
    )
    root = nn.Module()
    root.generator = nn.ModuleDict({name: nn.Linear(2, 2, bias=False) for name in ("wan", "wan_2")})
    for source in build_wan_weight_sources(request, ("generator",)):
        load_pretrained_weights(root, source, strict=True)
    for i, name in enumerate(("wan", "wan_2")):
        torch.testing.assert_close(
            root.generator[name].weight, state[prefixes[i] + "weight"], rtol=0, atol=0
        )


@pytest.mark.parametrize("experiment", ["wan22_a14b_dmd", "wan22_ti2v_5b_dmd"])
def test_wan_config_keeps_task_options_out_of_shared_config(experiment):
    from omegaconf import OmegaConf

    from vforge.algorithms.dmd.config import DmdRolloutConfig, DmdScoreConfig
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.wan_taichu.dmd.algorithm import WanDistributionMatchingConfig

    cfg = compose_standalone_config("train", overrides=[f"experiment={experiment}"])
    config_from_dict(cfg)
    mapping = OmegaConf.to_container(cfg.training.dmd)
    config = WanDistributionMatchingConfig.from_mapping(mapping)
    assert config.rollout.context_full_trajectory == (experiment == "wan22_a14b_dmd")
    assert config.score.timestep_floor_at_rollout_exit == (experiment == "wan22_a14b_dmd")
    assert "context_full_trajectory" not in DmdRolloutConfig.__dataclass_fields__
    assert "timestep_floor_at_rollout_exit" not in DmdScoreConfig.__dataclass_fields__
    mapping["score"]["unrecognized_option"] = True
    with pytest.raises(TypeError, match="unrecognized_option"):
        WanDistributionMatchingConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("family", "name", "expected"),
    [
        ("wan_taichu", "distribution_matching", "WanDistributionMatchingAlgorithm"),
        ("ltx23", "distribution_matching", "DistributionMatchingAlgorithm"),
        ("ltx23", "block_distribution_matching", "BlockDistributionMatchingAlgorithm"),
    ],
)
def test_dispatch_preserves_public_algorithm_name(monkeypatch, family, name, expected):
    from vforge.algorithms.dmd import create_algorithm
    from vforge.algorithms.dmd.algorithm import DistributionMatchingAlgorithm
    from vforge.algorithms.dmd.blockwise import BlockDistributionMatchingAlgorithm
    from vforge.tasks.dmd.train import create_dmd_components_builder
    from vforge.tasks.wan_taichu.dmd.algorithm import WanDistributionMatchingAlgorithm

    validated = []
    for cls in (
        DistributionMatchingAlgorithm,
        BlockDistributionMatchingAlgorithm,
        WanDistributionMatchingAlgorithm,
    ):
        monkeypatch.setattr(
            cls, "validate_request", lambda self, *a, **kw: validated.append(type(self).__name__)
        )
        monkeypatch.setattr(cls, "build_components", lambda self, *a, **kw: type(self).__name__)
    request = TrainRequest(
        dmd={"algorithm": name},
        training_models={"generator": TrainingModelRequest(model=family)},
    )
    build = create_dmd_components_builder(request, engine="fsdp2")
    assert validated == [expected]
    assert build() == expected
    assert type(create_algorithm("distribution_matching")) is DistributionMatchingAlgorithm


@pytest.mark.parametrize("fail", [False, True])
def test_offline_negative_prompt_uses_cpu_and_restores_threads(monkeypatch, fail):
    from types import SimpleNamespace

    import transformers

    from vforge.tasks.wan_taichu.dmd import integration

    encoder = torch.nn.Linear(2, 2)
    prompt = "色调艳丽，过曝，JPEG压缩残留"

    def tokenizer(texts, **kwargs):
        assert texts == [prompt]
        assert kwargs["max_length"] == 512
        return SimpleNamespace(input_ids=torch.ones(1, 512), attention_mask=torch.ones(1, 512))

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(
        transformers.UMT5EncoderModel, "from_pretrained", lambda *args, **kwargs: encoder
    )
    threads = SimpleNamespace(count=8)
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads.count)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: setattr(threads, "count", count))

    def encode(host, input_ids, mask, **kwargs):
        assert host.text_encoder is encoder
        assert next(host.text_encoder.parameters()).device.type == "cpu"
        assert input_ids.shape == mask.shape == (1, 512)
        assert host.training_dtype == torch.float32
        assert threads.count == 1
        assert not encoder.training
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        if fail:
            raise RuntimeError("encoding failed")
        return torch.ones(1, 512, 2)

    monkeypatch.setattr(integration.WanTaichuConditionEncoder, "encode_prompt", encode)
    request = TrainRequest(
        pretrained_path="model",
        precision="bf16",
        latent_cache_mode="offline",
        extras={"negative_prompt": prompt},
    )
    if fail:
        with pytest.raises(RuntimeError, match="encoding failed"):
            integration._build_negative_prompt_embedding(request, encoder=None)
    else:
        result = integration._build_negative_prompt_embedding(request, encoder=None)
        assert result.device.type == "cpu"
        assert result.dtype == torch.float32
    assert threads.count == 8


def test_offline_dmd_cache_uses_compute_dtype(monkeypatch):
    from vforge.tasks.wan_taichu.dmd import integration

    monkeypatch.setattr(
        integration,
        "_build_negative_prompt_embedding",
        lambda *args, **kwargs: torch.ones(1, 4, 8),
    )
    request = TrainRequest(
        precision="bf16",
        latent_cache_mode="offline",
        training_models={
            "generator": TrainingModelRequest(
                model="wan_taichu",
                config={
                    "variant": "wan22_a14b_causal_teacher",
                    "condition_streams": ["rgb", "normal"],
                },
            )
        },
    )
    data = integration.build_wan_data_support(request)
    prepared = data.prepare_batch(
        {
            "frame_latents": torch.randn(1, 16, 3, 4, 4),
            "condition_latents_rgb": torch.randn(1, 20, 3, 4, 4),
            "condition_latents_normal": torch.randn(1, 20, 3, 4, 4),
            "prompt": torch.randn(1, 4, 8),
        },
        torch.device("cpu"),
    )
    assert prepared.clean["latent"].dtype == torch.bfloat16
    assert prepared.model_inputs.condition_latents.dtype == torch.bfloat16
    assert prepared.model_inputs.prompt_embeds.dtype == torch.bfloat16
    assert prepared.model_inputs.negative_prompt_embeds.dtype == torch.bfloat16


def test_dmd_rollout_returns_fp32_for_score_arithmetic():
    from vforge.tasks.wan_taichu.dmd.generator import WanDmdGeneratorAdapter

    adapter = WanDmdGeneratorAdapter(
        frame_per_block=3,
        first_frame_own_block=False,
        first_frame_inpaint=False,
    )
    block = torch.randn(1, 3, 16, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    generated = adapter.concat_blocks([(None, {"latent": block})], None)["latent"]
    assert generated.dtype == torch.float32
    torch.testing.assert_close(generated, block.float(), rtol=0, atol=0)
    generated.sum().backward()
    torch.testing.assert_close(block.grad, torch.ones_like(block))
