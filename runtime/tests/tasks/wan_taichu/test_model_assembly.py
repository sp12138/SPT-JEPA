"""Model assembly contracts for supported Wan Taichu train variants."""

from __future__ import annotations

import pytest
import torch


def _tiny_config(*, in_channels: int, out_channels: int) -> dict:
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }


@pytest.mark.parametrize(
    ("variant", "configs", "expected_architecture", "expected_attention", "expected_prefixes"),
    [
        (
            "wan21_i2v_14b",
            {"": {**_tiny_config(in_channels=36, out_channels=16), "use_vae_condition": False}},
            "wan21",
            "bidirectional",
            ("transformer.",),
        ),
        (
            "wan21_i2v_14b_audio",
            {
                "": {
                    **_tiny_config(in_channels=36, out_channels=16),
                    "use_vae_condition": False,
                    "use_audio_condition": True,
                    "audio_dim": 4,
                    "audio_num_layers": 3,
                    "num_audio_token": 2,
                    "audio_inject_layers_s": [0],
                }
            },
            "wan21",
            "bidirectional",
            ("transformer.",),
        ),
        (
            "wan22_a14b",
            {
                "wan": {
                    **_tiny_config(in_channels=36, out_channels=16),
                    "use_vae_condition": False,
                },
                "wan_2": {
                    **_tiny_config(in_channels=36, out_channels=16),
                    "use_vae_condition": False,
                },
            },
            "wan22_a14b",
            "bidirectional",
            ("wan.", "wan_2."),
        ),
        (
            "wan22_a14b_causal_teacher",
            {
                "wan": {
                    **_tiny_config(in_channels=76, out_channels=16),
                    "use_vae_condition": False,
                },
                "wan_2": {
                    **_tiny_config(in_channels=76, out_channels=16),
                    "use_vae_condition": False,
                },
            },
            "wan22_a14b",
            "block_causal_teacher",
            ("wan.", "wan_2."),
        ),
        (
            "wan22_ti2v_5b",
            {"": _tiny_config(in_channels=96, out_channels=48)},
            "wan22_ti2v_5b",
            "bidirectional",
            ("transformer.",),
        ),
        (
            "wan22_ti2v_5b_ref",
            {
                "": {
                    **_tiny_config(in_channels=144, out_channels=48),
                    "use_ref_token": True,
                    "in_channels_ref": 48,
                }
            },
            "wan22_ti2v_5b",
            "bidirectional",
            ("transformer.",),
        ),
        (
            "wan22_ti2v_5b_causal_teacher",
            {"": _tiny_config(in_channels=96, out_channels=48)},
            "wan22_ti2v_5b",
            "block_causal_teacher",
            ("transformer.",),
        ),
    ],
)
def test_build_empty_model_preserves_native_root_layout(
    variant,
    configs,
    expected_architecture,
    expected_attention,
    expected_prefixes,
) -> None:
    from vforge.tasks.wan_taichu.model import build_empty_model

    model = build_empty_model(
        variant=variant,
        transformer_configs=configs,
        parameter_dtype=torch.bfloat16,
    )

    assert type(model).__name__ == "WanTaichuModel"
    assert model.variant == variant
    assert model.architecture == expected_architecture
    assert model.attention == expected_attention
    assert {type(transformer).__name__ for transformer in model.cache_transformers()} == {
        "WanTaichuTransformer3DModel"
    }
    assert all(
        transformer.taichu_architecture == expected_architecture
        and transformer.taichu_attention == expected_attention
        for transformer in model.cache_transformers()
    )
    assert all(
        transformer.get_fsdp_precision_contract().preserve_forward_input_paths == (".", "norm_out")
        for transformer in model.cache_transformers()
    )
    assert all(parameter.is_meta for parameter in model.parameters())
    keys = tuple(model.state_dict())
    assert all(key.startswith(expected_prefixes) for key in keys)


def test_build_empty_model_rejects_wrong_expert_layout() -> None:
    from vforge.tasks.wan_taichu.model import build_empty_model

    with pytest.raises(ValueError, match="must have keys"):
        build_empty_model(
            variant="wan22_a14b",
            transformer_configs={"": _tiny_config(in_channels=36, out_channels=16)},
        )


def test_build_empty_model_plumbs_explicit_precision_contract_opt_out() -> None:
    from vforge.distributed.parallelize import FSDPPrecisionContract
    from vforge.tasks.wan_taichu.model import build_empty_model

    model = build_empty_model(
        variant="wan22_ti2v_5b",
        transformer_configs={
            "": _tiny_config(in_channels=36, out_channels=16),
        },
        use_fsdp_precision_contract=False,
    )

    assert model.transformer.get_fsdp_precision_contract() == FSDPPrecisionContract()


def test_ref_variant_owns_a_separate_projection() -> None:
    from vforge.tasks.wan_taichu.model import build_empty_model

    model = build_empty_model(
        variant="wan22_ti2v_5b_ref",
        transformer_configs={
            "": {
                **_tiny_config(in_channels=144, out_channels=48),
                "use_ref_token": True,
                "in_channels_ref": 48,
            }
        },
    )

    assert model.transformer.patch_embedding_ref.in_channels == 48
    assert "transformer.patch_embedding_ref.weight" in model.state_dict()


def test_audio_variant_preserves_taichu_checkpoint_paths() -> None:
    from vforge.tasks.wan_taichu.model import build_empty_model

    model = build_empty_model(
        variant="wan21_i2v_14b_audio",
        transformer_configs={
            "": {
                **_tiny_config(in_channels=36, out_channels=16),
                "use_vae_condition": False,
                "use_audio_condition": True,
                "audio_dim": 4,
                "audio_num_layers": 3,
                "num_audio_token": 2,
                "audio_inject_layers_s": [0],
                "audio_inject_layers_l": [0],
            }
        },
    )

    keys = set(model.state_dict())
    assert "transformer.casual_audio_encoder.weights" in keys
    assert "transformer.audio_injector.injector_s.0.q.weight" in keys
    assert "transformer.audio_injector.injector_l.0.o.bias" in keys


@pytest.mark.parametrize("wrapped", [False, True])
def test_taichu_weights_only_pt_loads_both_experts(tmp_path, wrapped) -> None:
    from vforge.models.loading import load_pretrained_weights
    from vforge.models.wan.model import WanChannelLayout
    from vforge.tasks.wan_taichu.model import build_empty_model
    from vforge.tasks.wan_taichu.train import _a14b_pretrained_sources

    names = ("wan", "wan_2")
    model = build_empty_model(
        variant="wan22_a14b",
        transformer_configs={name: _tiny_config(in_channels=36, out_channels=16) for name in names},
    ).to_empty(device="cpu")
    expected = {
        key: torch.full_like(value, 1 if key.startswith("wan.") else 2)
        for key, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "experts.pt"
    state = {f"module.{key}" if wrapped else key: value for key, value in expected.items()}
    torch.save({"state_dict": state} if wrapped else state, checkpoint)
    sources = _a14b_pretrained_sources(
        layouts={
            name: WanChannelLayout(z_dim=16, pretrained_in_channels=36, in_channels=36)
            for name in names
        },
        pretrained_path=str(checkpoint),
        weight_loading="auto",
        expert_names=names,
    )
    for source in sources:
        load_pretrained_weights(model, source, strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key]), key
