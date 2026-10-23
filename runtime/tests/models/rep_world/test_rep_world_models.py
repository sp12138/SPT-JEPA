from dataclasses import replace

import pytest
import torch

from vforge.distributed.parallelize import materialize_meta_module
from vforge.models.rep_world.conditioner import CausalViT
from vforge.models.rep_world.config import (
    CodecGeometry,
    ConditionerConfig,
    WorldConfig,
    codec_geometry,
)
from vforge.models.rep_world.layers import VisionRotaryEmbeddingFast3D
from vforge.models.rep_world.world import WorldDiT
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def test_codec_geometry_and_token_counts():
    expected = [
        ("videorae", "3d", 2048, 1024, 1024),
        ("videorae", "1d", 2048, 1024, 512),
        ("vrae", "eupe", 512, 256, 256),
        ("vrae", "vjepa2_1", 512, 256, 256),
    ]
    for family, variant, patch2, patch4, latent in expected:
        geo = codec_geometry(family, variant)
        assert ConditionerConfig().tokens_per_state(geo) == patch2
        assert ConditionerConfig(temporal_patch=4).tokens_per_state(geo) == patch4
        assert ConditionerConfig(input="latent").tokens_per_state(geo) == latent
    assert codec_geometry("videorae", "1d").temporal_compression is None
    assert codec_geometry("videorae", "3d").temporal_compression == 4
    with pytest.raises(ValueError, match="temporal_patch"):
        ConditionerConfig(input="latent", temporal_patch=4)


@pytest.mark.parametrize("mode", ["no_zprev", "zprev", "pure_tf"])
def test_world_shape_gradient_and_meta(mode):
    config = WorldConfig(
        4, 2, grid=4, width=48, depth=2, heads=4, condition_width=0 if mode == "pure_tf" else 24
    )
    with torch.device("meta"):
        model = WorldDiT(config)
    assert all(p.is_meta for p in model.parameters())
    materialize_meta_module(model, "cpu")
    model.initialize_weights()
    assert all(not x.is_meta and torch.isfinite(x).all() for x in model.state_dict().values())
    reference = WorldDiT(config)
    for name, buffer in model.feat_rope.named_buffers():
        torch.testing.assert_close(
            buffer, dict(reference.feat_rope.named_buffers())[name], rtol=0, atol=0
        )
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
    frames = {"no_zprev": 2, "zprev": 4, "pure_tf": 8}[mode]
    x = torch.randn(2, 4, frames, 4, 4, requires_grad=True)
    h = torch.randn(2, 7, 24, requires_grad=True) if mode != "pure_tf" else None
    y = model(
        x,
        torch.rand(2),
        condition=h,
        prefix_frames=2 if mode == "zprev" else 0,
        teacher_forcing_planes=2 if mode == "pure_tf" else 0,
    )
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    if h is not None:
        assert h.grad.abs().sum() > 0


@pytest.mark.parametrize("domain,patch", [("rgb", 2), ("rgb", 4), ("latent", 2)])
def test_conditioner_causality_and_training_shift(domain, patch):
    cfg = ConditionerConfig(
        input=domain,
        input_size=32,
        temporal_patch=patch,
        width=24,
        depth=2,
        heads=4,
        feature_layers=(1,),
    )
    geo = CodecGeometry("test", "tiny", 4, 4, 1, grid=2)
    model = CausalViT(cfg, geo)
    source = torch.randn(2, 3, 12, 32, 32) if domain == "rgb" else torch.randn(2, 4, 3, 2, 2)
    source.requires_grad_(True)
    output = model(source)
    assert output.shape == (2, 3, cfg.tokens_per_state(geo), 24)
    changed = source.detach().clone()
    changed[:, :, 8 if domain == "rgb" else 2 :] += torch.randn_like(
        changed[:, :, 8 if domain == "rgb" else 2 :]
    )
    torch.testing.assert_close(output[:, :2], model(changed)[:, :2], atol=0, rtol=0)
    output[:, 1, :, 0].sum().backward()
    assert source.grad[:, :, 8 if domain == "rgb" else 2 :].count_nonzero() == 0
    assert source.grad[:, :, : 8 if domain == "rgb" else 2].abs().sum() > 0
    context = model.training_context(source)
    torch.testing.assert_close(context, output[:, :-1])
    context.square().mean().backward()
    assert model.encoder.blocks[0].attn.qkv.weight.grad is not None


def test_feature_sum_and_unused_rgb_branch_removed():
    cfg = ConditionerConfig(input="latent", width=24, depth=2, heads=4, feature_layers=(0, 1))
    geo = CodecGeometry("test", "tiny", 4, 4, 1, grid=2)
    model = CausalViT(cfg, geo)
    source = torch.randn(1, 4, 3, 2, 2)
    both = model(source)
    single = CausalViT(replace(cfg, feature_layers=(0,)), geo)
    single.load_state_dict(model.state_dict())
    first = single(source)
    single.config = replace(cfg, feature_layers=(1,))
    torch.testing.assert_close(both, first + single(source))
    assert not any(
        "patch_embed_img" in name or "img_mod_embed" in name for name in model.state_dict()
    )


def test_pure_tf_cached_prefill_equivalence():
    model = WorldDiT(
        WorldConfig(4, 2, grid=2, width=48, depth=2, heads=4, condition_width=0)
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "final_layer" in name or "modulation" in name or "shared_mod" in name:
                parameter.normal_(std=0.03)
    x = torch.randn(1, 4, 6, 2, 2)
    timestep = torch.zeros(1, 6)
    full = model(x, timestep, causal_state_planes=2)
    _, cache = model(x[:, :, :4], timestep[:, :4], causal_state_planes=2, return_key_values=True)
    last = model(
        x[:, :, 4:], timestep[:, 4:], causal_state_planes=2, past_key_values=cache, frame_offset=4
    )
    torch.testing.assert_close(full[:, :, 4:], last, atol=1e-6, rtol=1e-5)


@pytest.mark.gpu(gpus=1)
def test_pure_tf_first_state_rope_preserves_autocast_reference_window():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cuda"),
        parallel_state=None,
    )
    config = WorldConfig(4, 2, grid=4, width=192, depth=1, heads=3, condition_width=0)
    model = WorldDiT(config).cuda().eval()
    # The original model precomputes the complete training window, then
    # dynamically computes any shorter inference prefix under autocast.
    reference_rope = VisionRotaryEmbeddingFast3D(64, 6, 4, 4).cuda()
    normalized_keys = []
    handle = model.blocks[0].attn.k_norm.register_forward_hook(
        lambda _module, _inputs, output: normalized_keys.append(output)
    )
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, cache = model(
                torch.randn(1, 4, 2, 4, 4, device="cuda"),
                torch.zeros(1, 2, device="cuda"),
                causal_state_planes=2,
                return_key_values=True,
            )
            expected = reference_rope(normalized_keys[0], num_frames_override=2)
    finally:
        handle.remove()
    torch.testing.assert_close(cache[0][0], expected, atol=0, rtol=0)
