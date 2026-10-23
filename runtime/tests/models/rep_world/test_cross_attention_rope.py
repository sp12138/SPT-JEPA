"""Cross-attention 3D RoPE: independent complex reference and end-to-end routing."""

from dataclasses import replace
from functools import partial

import pytest
import torch

from vforge.models.rep_world.attention import CrossAttention
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.layers import VisionRotaryEmbeddingFast3D
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def cpu_math_ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def independent_rotate(value, indices, planes, grid, *, next_state):
    """Complex multiplication, independent of production cos/sin and rotate_half."""
    width = value.shape[-1]
    spatial_width = 2 * ((width // 3) // 2)
    dims = (width - 2 * spatial_width, spatial_width, spatial_width)
    positions = (
        indices // (grid * grid) + (planes if next_state else 0),
        indices // grid % grid,
        indices % grid,
    )
    pieces = []
    offset = 0
    for dim, position in zip(dims, positions, strict=True):
        part = value[..., offset : offset + dim].float()
        z = torch.view_as_complex(part.reshape(*part.shape[:-1], dim // 2, 2).contiguous())
        frequency = 10000.0 ** (-torch.arange(dim // 2, device=value.device).float() * 2 / dim)
        angle = position.float()[..., None] * frequency
        phase = torch.polar(torch.ones_like(angle), angle).unsqueeze(2)
        pieces.append(torch.view_as_real(z * phase).flatten(-2))
        offset += dim
    return torch.cat(pieces, -1).to(value.dtype)


def reference(module, x, context, positions, planes, grid, same_patch):
    batch, length, width = x.shape
    q = module.q_norm(module.to_q(x).reshape(batch, length, module.num_heads, module.head_dim))
    k, v = module.to_kv(context).reshape(batch, -1, 2, module.num_heads, module.head_dim).unbind(2)
    k = module.k_norm(k)
    qi = torch.arange(length, device=x.device).expand(batch, -1)
    q = independent_rotate(q, qi, planes, grid, next_state=True).transpose(1, 2)
    k = independent_rotate(k, positions, planes, grid, next_state=False).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / module.head_dim**0.5
    visible = torch.ones(batch, length, 1, dtype=torch.bool, device=x.device)
    if same_patch:
        allowed = qi[:, :, None] % (grid * grid) == positions[:, None, :] % (grid * grid)
        visible = allowed.any(-1, keepdim=True)
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        scores = torch.where(visible[:, None], scores, torch.zeros_like(scores))
    out = scores.softmax(-1) @ v.transpose(1, 2)
    out = module.proj(out.transpose(1, 2).reshape(batch, length, width))
    return out * visible.to(out.dtype)


@pytest.mark.parametrize("same_patch", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_forward_backward_matches_independent_reference(same_patch, sparse):
    torch.manual_seed(17)
    module = CrossAttention(48, 3, True)
    rope = VisionRotaryEmbeddingFast3D(16, 2, 3, 3)
    indices = (
        torch.tensor([[0, 3, 7, 9, 12], [2, 5, 8, 11, 14]])
        if sparse
        else torch.arange(18).expand(2, -1)
    )
    x = torch.randn(2, 18, 48, requires_grad=True)
    c = torch.randn(2, indices.shape[1], 48, requires_grad=True)
    actual = module(
        x,
        c,
        spatial_size=9 if same_patch else None,
        context_positions=indices if sparse else None,
        q_rope=partial(rope, start_frame=2),
        k_rope=partial(rope, token_position_ids=indices if sparse else None),
    )
    expected = reference(module, x, c, indices, 2, 3, same_patch)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    variables = (x, c, *module.parameters())
    ga = torch.autograd.grad((actual * upstream).sum(), variables, retain_graph=True)
    gb = torch.autograd.grad((expected * upstream).sum(), variables)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-5)


def test_global_ca_uses_content_coordinates_and_preserves_joint_permutation():
    torch.manual_seed(9)
    ca = CrossAttention(48, 3, True)
    rope = VisionRotaryEmbeddingFast3D(16, 2, 3, 3)
    x, c = torch.randn(2, 18, 48), torch.randn(2, 18, 48)
    perm = torch.randperm(18)
    kwargs = dict(q_rope=partial(rope, start_frame=2), k_rope=rope)
    original = ca(x, c, **kwargs)
    wrong = ca(x, c[:, perm], **kwargs)
    correct = ca(
        x,
        c[:, perm],
        q_rope=kwargs["q_rope"],
        k_rope=partial(rope, token_position_ids=perm.expand(2, -1)),
    )
    torch.testing.assert_close(original, correct, rtol=2e-5, atol=2e-6)
    assert (original - wrong).abs().max() > 1e-3


def make_model(scope="global", mask=0.0, enabled=True):
    geometry = CodecGeometry("test", "test", 8, 8, 2, grid=3)
    encoder = ConditionerConfig(
        input="latent", width=24, depth=2, heads=3, feature_layers=(1,), context_mask_ratio=mask
    )
    world = WorldConfig(
        channels=8,
        state_planes=2,
        grid=3,
        width=24,
        depth=2,
        heads=3,
        condition_width=24,
        adaln_lora_dim=8,
        cross_attention_every=2,
        cross_attention_position="before_self",
        cross_attention_scope=scope,
        cross_attention_3d_rope=enabled,
    )
    return RepWorld(geometry, world, encoder, FlowConfig())


def test_rope_flag_changes_no_parameters_buffers_or_initialization():
    torch.manual_seed(23)
    baseline = make_model(enabled=False)
    torch.manual_seed(23)
    candidate = make_model()
    assert baseline.state_dict().keys() == candidate.state_dict().keys()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)
    assert [i for i, b in enumerate(candidate.net.blocks) if b.cross_attn is not None] == [0]


@pytest.mark.parametrize("scope", ["global", "same_spatial"])
@pytest.mark.parametrize("mask", [0.0, 0.8])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_training_inference_and_null_use_adjacent_state_rope(scope, mask, checkpoint):
    torch.manual_seed(31)
    model = make_model(scope, mask).train()
    if checkpoint:
        model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    captured = []
    handle = model.net.blocks[0].cross_attn.register_forward_pre_hook(
        lambda m, args, kw: captured.append(kw), with_kwargs=True
    )
    latent = torch.randn(2, 16, 3, 3, 3)
    loss = model(latents=latent, condition_drop=torch.tensor([False, True, False, True]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.causal_semantic_encoder.latent_patch_embed.weight.grad.abs().sum() > 0
    assert captured[0]["q_rope"].keywords["start_frame"] == 2
    ids = captured[0]["k_rope"].keywords["token_position_ids"]
    assert ids is None if mask == 0 else ids.shape == (4, 2)
    handle.remove()
    model.eval()
    condition = model(operation="condition", source=latent[:, :, :1])
    x, t = torch.randn(2, 8, 2, 3, 3), torch.full((2, 1), 0.5)
    for context in [condition, None]:
        value = model(operation="velocity", noisy=x, timestep=t, condition=context)
        assert value.shape == x.shape and torch.isfinite(value).all()


def test_invalid_rope_requests_are_rejected():
    model = make_model()
    with pytest.raises(ValueError, match="cross_attention"):
        replace(model.net.config, condition_mode="entry_concat")
    with pytest.raises(ValueError, match="spatially ordered"):
        RepWorld(
            replace(model.geometry, spatially_ordered=False),
            model.net.config,
            model.causal_semantic_encoder.config,
            FlowConfig(),
        )
    with pytest.raises(ValueError, match="both query and key"):
        model.net.blocks[0].cross_attn(
            torch.randn(1, 18, 24), torch.randn(1, 18, 24), q_rope=lambda x: x
        )


@pytest.mark.parametrize("verb", ["train", "eval"])
def test_public_rope_option_composes_without_plus(verb):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from vforge.tasks.common import build_and_snapshot
    from vforge.tasks.rep_world.shared import parse_config

    assert build_and_snapshot
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[
                "experiment=rep_world",
                "model.extras.conditioner.input=latent",
                "model.extras.world.cross_attention_3d_rope=true",
            ],
        )
    parsed = parse_config(OmegaConf.to_container(cfg.model.extras, resolve=True))
    assert parsed.world.cross_attention_3d_rope
