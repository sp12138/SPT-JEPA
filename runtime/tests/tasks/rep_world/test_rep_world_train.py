import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from vforge.models.rep_world.config import CODECS
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.shared import parse_config, validate_train_request
from vforge.tasks.rep_world.train import build_train_components
from vforge.tasks.requests import TrainRequest
from vforge.training.recipe import StepContext


def tiny_extras(domain="rgb"):
    return dict(
        codec=dict(family="vrae", variant="eupe"),
        states=3,
        world=dict(width=48, depth=1, heads=4, condition_dropout=0),
        conditioner=dict(
            input=domain, input_size=32, width=24, depth=2, heads=4, feature_layers=[1]
        ),
    )


@pytest.mark.parametrize("domain,mask_ratio", [("rgb", 0), ("latent", 0), ("latent", 0.8)])
def test_task_step_uses_joint_root_and_keeps_codec_frozen(
    tmp_path, monkeypatch, domain, mask_ratio
):
    state = SimpleNamespace(
        batch_group=None, is_batch_leader=True, dp_rank=0, config=SimpleNamespace(dp_size=1)
    )
    monkeypatch.setattr("vforge.tasks.rep_world.train.get_parallel_state", lambda: state)
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(json.dumps(dict(video="clip.mp4")))
    extras = tiny_extras(domain)
    extras["conditioner"]["context_mask_ratio"] = mask_ratio
    request = TrainRequest(
        extras=extras,
        train_manifest=str(manifest),
        precision="fp32",
        init_device="meta",
        latent_cache_mode="online",
        betas=(0.9, 0.95),
    )
    calls = []

    class Codec:
        def encode(self, rgb):
            assert not torch.is_grad_enabled()
            calls.append(tuple(rgb.shape))
            return torch.randn(rgb.shape[0], 768, 3, 16, 16)

    monkeypatch.setattr(
        "vforge.tasks.rep_world.shared.CodecProvider.get", lambda self, device: Codec()
    )
    components = build_train_components(request)
    model = components.model.build()
    assert all(parameter.is_meta for parameter in model.parameters())
    model.to_empty(device="cpu")
    components.model_initializer(model)
    optimizer = components.optimizer_factory(model)
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    try:
        batch = dict(sample_id=["first"], rgb=torch.rand(1, 3, 12, 256, 256) * 2 - 1)
        for index in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = components.step_fn(
                model, batch, index, StepContext(torch.device("cpu"), state)
            )
            assert torch.isfinite(output.loss)
            assert output.metrics["loss_fm"] == output.loss.detach()
            assert output.metadata["prediction_positions"] == 2
            output.loss.backward()
            if index == 1:
                assert (
                    model.causal_semantic_encoder.encoder.blocks[0].attn.qkv.weight.grad.abs().sum()
                    > 0
                )
            optimizer.step()
        assert calls == [(1, 3, 12, 256, 256)] * 2
    finally:
        reset_ops()


@pytest.mark.parametrize("family,variant", [key for key in CODECS if key[0] not in {"pixel", "wan22"}])
@pytest.mark.parametrize("domain,patch", [("rgb", 2), ("rgb", 4), ("latent", 2)])
def test_codec_and_input_geometry_contract(family, variant, domain, patch):
    config = parse_config(
        dict(
            codec=dict(family=family, variant=variant),
            conditioner=dict(input=domain, temporal_patch=patch),
        )
    )
    assert config.world.channels == config.geometry.channels
    assert config.world.state_planes == config.geometry.planes
    expected = (
        config.geometry.tokens_per_state
        if domain == "latent"
        else config.geometry.rgb_frames // patch * 256
    )
    assert config.conditioner.tokens_per_state(config.geometry) == expected


@pytest.mark.parametrize(
    "experiment,family,variant",
    [
        ("rep_world", "videorae", "3d"),
        ("rep_world_1d", "videorae", "1d"),
        ("rep_world_eupe", "vrae", "eupe"),
        ("rep_world_vjepa", "vrae", "vjepa2_1"),
    ],
)
def test_experiments_compose_with_declared_normal_overrides(experiment, family, variant):
    # Importing common installs the shared native_defaults ConfigStore schema.
    from vforge.tasks.common import build_and_snapshot

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "model.extras.conditioner.temporal_patch=4",
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    build_and_snapshot(cfg, world_size=1, write_snapshot=False)
    assert (parsed.geometry.family, parsed.geometry.variant) == (family, variant)
    assert cfg.parallel.dp_mode == "ddp" and cfg.runtime.autocast
    assert not cfg.parallel.mixed_precision.enabled
    assert cfg.resume.path == "none" and cfg.checkpoint.state == "full"


def test_preflight_rejects_unimplemented_routes_before_allocations():
    request = TrainRequest(
        extras=tiny_extras(), train_manifest="not-opened.jsonl", latent_cache_mode="online"
    )
    validate_train_request(request, "fsdp2")
    for candidate, engine in [
        (request, "megatron"),
        (replace(request, tp_size=2), "fsdp2"),
        (replace(request, latent_cache_mode="offline"), "fsdp2"),
    ]:
        with pytest.raises(ValueError):
            validate_train_request(candidate, engine)


def test_ulysses_preflight_contract():
    from dataclasses import replace

    from vforge.tasks.rep_world.shared import validate_train_request
    from vforge.tasks.requests import TrainRequest

    request = TrainRequest(train_manifest="videos.jsonl", latent_cache_mode="online")
    validate_train_request(replace(request, usp_size=2), "fsdp2")
    for field in ("tp_size", "cp_size"):
        with pytest.raises(ValueError, match="TP/CP"):
            validate_train_request(replace(request, **{field: 2}), "fsdp2")


def test_mask_config_compose_and_preflight():
    from vforge.tasks.common import build_and_snapshot

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=rep_world",
                "model.extras.conditioner.input=latent",
                "model.extras.conditioner.context_mask_ratio=0.8",
                "data.batch_size=8",
                "checkpoint.enabled=false",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    build_and_snapshot(cfg, world_size=32, write_snapshot=False)
    assert parsed.conditioner.context_mask_ratio == 0.8
    assert parsed.flow.mode == "no_zprev" and cfg.data.batch_size == 8
    request = TrainRequest(
        train_manifest="not-opened.jsonl",
        latent_cache_mode="online",
        extras=OmegaConf.to_container(cfg.model.extras, resolve=True),
    )
    validate_train_request(request, "fsdp2")
    with pytest.raises(ValueError, match="masking.*USP=1"):
        validate_train_request(replace(request, usp_size=2), "fsdp2")
