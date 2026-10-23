from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vforge.models.wan.model import WanChannelLayout
from vforge.tasks.registry import require_registered
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan_taichu.batch import stream_field
from vforge.tasks.wan_taichu.train import (
    _global_ref_present,
    _pretrained_sources,
    _processor_input,
    _resolve_request,
    _warm_start_ref_projection,
    validate_train_request,
)
from vforge.tasks.wan_taichu.variants import require_variant


def test_public_registry_exposes_train_and_eval() -> None:
    tasks = require_registered("wan_taichu")
    assert tasks.train == "vforge.tasks.wan_taichu.train:build_train_components"
    assert tasks.train_preflight == "vforge.tasks.wan_taichu.train:validate_train_request"
    assert tasks.eval == "vforge.tasks.wan_taichu.eval:build_eval_components"


def test_wan_taichu_task_does_not_import_versioned_wan_implementations() -> None:
    import vforge.tasks.wan_taichu

    root = Path(vforge.tasks.wan_taichu.__file__).parent
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(
                name.startswith(("vforge.models.wan.v", "vforge.tasks.wan.v")) for name in names
            ):
                violations.append(path.relative_to(root))
    assert violations == []


def test_wan_taichu_training_owns_its_collate_contract() -> None:
    import vforge.tasks.wan_taichu

    root = Path(vforge.tasks.wan_taichu.__file__).parent
    imports = []
    for path in (root / "batch.py", root / "train.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")

    assert "vforge.data.collate" not in imports


def test_preflight_rejects_tensor_parallelism_before_allocation() -> None:
    request = TrainRequest(extras={"variant": "wan21_i2v_14b"}, tp_size=2)

    with pytest.raises(NotImplementedError, match=r"parallel\.tp_size=1"):
        validate_train_request(request, "fsdp2")


def test_a14b_preflight_rejects_megatron_dual_expert_before_allocation() -> None:
    request = TrainRequest(extras={"variant": "wan22_a14b", "train_expert": "both"})

    with pytest.raises(NotImplementedError, match="train_expert=both"):
        validate_train_request(request, "megatron")
    validate_train_request(request, "fsdp2")


@pytest.mark.parametrize("expert", ("high", "low"))
def test_a14b_preflight_accepts_megatron_single_expert(expert: str) -> None:
    validate_train_request(
        TrainRequest(extras={"variant": "wan22_a14b", "train_expert": expert}),
        "megatron",
    )


@pytest.mark.parametrize(
    ("name", "shift"),
    (
        ("wan21_i2v_14b", 1.0),
        ("wan21_i2v_14b_audio", 5.0),
        ("wan22_a14b", 3.0),
        ("wan22_ti2v_5b", 5.0),
        ("wan22_ti2v_5b_causal_teacher", 5.0),
    ),
)
def test_variant_resolves_fixed_training_defaults(name: str, shift: float) -> None:
    spec, streams, extras = _resolve_request(TrainRequest(extras={"variant": name}))

    assert spec.name == name
    assert streams == spec.default_condition_streams
    assert extras["noise_shift"] == shift


def test_causal_teacher_resolves_its_block_layout() -> None:
    spec, streams, extras = _resolve_request(
        TrainRequest(extras={"variant": "wan22_ti2v_5b_causal_teacher"})
    )

    assert spec.attention == "block_causal_teacher"
    assert streams == ("normal",)
    assert extras["frame_per_block"] == 3
    assert extras["first_frame_own_block"] is True


def test_audio_variant_resolves_taichu_injection_defaults() -> None:
    spec, streams, extras = _resolve_request(
        TrainRequest(extras={"variant": "wan21_i2v_14b_audio"})
    )

    assert spec.audio is True
    assert streams == ("rgb",)
    assert extras["audio_dim"] == 1024
    assert extras["audio_num_layers"] == 25
    assert extras["num_audio_token"] == 4
    assert extras["audio_inject_layers_s"] == list(range(0, 40, 2))


def test_audio_pretrained_source_owns_missing_keys_and_post_load_zero_init() -> None:
    class Transformer:
        _audio_zero_init = True

        def __init__(self) -> None:
            self.zeroed = False

        def zero_audio_output_projections(self) -> None:
            self.zeroed = True

    class Model:
        def __init__(self) -> None:
            self.transformer = Transformer()

        def state_dict(self):
            return {
                "transformer.casual_audio_encoder.weights": torch.empty(1),
                "transformer.audio_injector.injector_s.0.o.weight": torch.empty(1),
                "transformer.patch_embedding.weight": torch.empty(1),
            }

        def primary_transformer(self):
            return self.transformer

    model = Model()
    source = _pretrained_sources(
        spec=require_variant("wan21_i2v_14b_audio"),
        model=model,
        layouts={
            "": WanChannelLayout(
                z_dim=16,
                pretrained_in_channels=36,
                in_channels=36,
            )
        },
        pretrained_path="/checkpoint/wan-audio",
        weight_loading="auto",
    )

    assert "transformer.casual_audio_encoder.weights" in source.allowed_missing_model_keys
    assert "transformer.audio_injector.injector_s.0.o.weight" in source.allowed_missing_model_keys
    assert source.initialize_missing_after_load is not None
    source.initialize_missing_after_load(model, frozenset())
    assert model.transformer.zeroed is True

    model.transformer._audio_zero_init = False
    preserving_source = _pretrained_sources(
        spec=require_variant("wan21_i2v_14b_audio"),
        model=model,
        layouts={
            "": WanChannelLayout(
                z_dim=16,
                pretrained_in_channels=36,
                in_channels=36,
            )
        },
        pretrained_path="/checkpoint/wan-audio",
        weight_loading="auto",
    )
    assert preserving_source.initialize_missing_after_load is None


def test_causal_teacher_rejects_a_degenerate_block_layout() -> None:
    with pytest.raises(ValueError, match="frame_per_block"):
        _resolve_request(
            TrainRequest(
                extras={
                    "variant": "wan22_ti2v_5b_causal_teacher",
                    "frame_per_block": 0,
                }
            )
        )


def test_processor_input_concatenates_streams_in_config_order() -> None:
    spec = require_variant("wan22_a14b")
    encoded = {
        stream_field("normal"): torch.full((1, 20, 1, 2, 2), 2.0),
        stream_field("grid"): torch.full((1, 20, 1, 2, 2), 1.0),
    }

    output = _processor_input(encoded, spec=spec, streams=("grid", "normal"))

    first, second = output["condition_latents_normal"].split(20, dim=1)
    assert torch.equal(first, encoded[stream_field("grid")])
    assert torch.equal(second, encoded[stream_field("normal")])


def test_single_expert_pretrained_source_maps_into_native_wrapper() -> None:
    spec = require_variant("wan22_ti2v_5b")
    source = _pretrained_sources(
        spec=spec,
        layouts={
            "": WanChannelLayout(
                z_dim=48,
                pretrained_in_channels=48,
                in_channels=96,
            )
        },
        pretrained_path="/checkpoint/wan",
        weight_loading="auto",
    )

    assert source.model_prefix == "transformer."
    assert source.model_path == ""
    assert source.subfolder == "transformer"


def test_ref_variant_registers_checkpoint_warm_start_after_load() -> None:
    source = _pretrained_sources(
        spec=require_variant("wan22_ti2v_5b_ref"),
        layouts={
            "": WanChannelLayout(
                z_dim=48,
                pretrained_in_channels=48,
                in_channels=144,
            )
        },
        pretrained_path="/checkpoint/wan",
        weight_loading="auto",
    )

    assert source.allowed_missing_model_keys == frozenset(
        {
            "transformer.patch_embedding_ref.weight",
            "transformer.patch_embedding_ref.bias",
        }
    )
    assert source.initialize_missing_after_load is not None


def test_ref_projection_warm_start_uses_loaded_noisy_channels_only() -> None:
    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embedding = nn.Conv3d(6, 3, 1)
            self.patch_embedding_ref = nn.Conv3d(2, 3, 1)

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = Transformer()

    model = Wrapper()
    with torch.no_grad():
        model.transformer.patch_embedding.weight.copy_(
            torch.arange(18, dtype=torch.float32).reshape(3, 6, 1, 1, 1)
        )
        model.transformer.patch_embedding.bias.copy_(torch.tensor([3.0, 4.0, 5.0]))
        model.transformer.patch_embedding_ref.weight.fill_(-1.0)
        model.transformer.patch_embedding_ref.bias.fill_(-1.0)

    _warm_start_ref_projection(
        model,
        frozenset(
            {
                "transformer.patch_embedding_ref.weight",
                "transformer.patch_embedding_ref.bias",
            }
        ),
    )

    assert torch.equal(
        model.transformer.patch_embedding_ref.weight,
        model.transformer.patch_embedding.weight[:, :2],
    )
    assert torch.equal(
        model.transformer.patch_embedding_ref.bias,
        model.transformer.patch_embedding.bias,
    )


def test_global_ref_presence_uses_the_fsdp_group(monkeypatch) -> None:
    group = object()
    state = type("State", (), {"fsdp_group": group})()
    observed = []
    monkeypatch.setattr("vforge.tasks.wan_taichu.train.dist.is_available", lambda: True)
    monkeypatch.setattr("vforge.tasks.wan_taichu.train.dist.is_initialized", lambda: True)

    def fake_all_reduce(flag, *, op, group):
        observed.append((op, group))
        flag.fill_(1)

    monkeypatch.setattr("vforge.tasks.wan_taichu.train.dist.all_reduce", fake_all_reduce)
    assert _global_ref_present(
        None,
        hidden_states=torch.zeros(1),
        parallel_state=state,
    )
    assert observed == [(torch.distributed.ReduceOp.MAX, group)]


def test_wan21_missing_image_embedding_keys_use_native_wrapper_prefix() -> None:
    source = _pretrained_sources(
        spec=require_variant("wan21_i2v_14b"),
        layouts={
            "": WanChannelLayout(
                z_dim=16,
                pretrained_in_channels=16,
                in_channels=36,
            )
        },
        pretrained_path="/checkpoint/wan",
        weight_loading="auto",
    )

    assert source.allowed_missing_model_keys == frozenset(
        {
            "transformer.patch_embedding_image.weight",
            "transformer.patch_embedding_image.bias",
        }
    )
