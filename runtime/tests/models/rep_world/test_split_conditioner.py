"""ViT-E state independence, ViT-C causality and Native integration contracts."""

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vforge.distributed.parallelize import materialize_meta_module
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.ops._attention import dispatch


@pytest.fixture(autouse=True)
def ops(request):
    gpu = request.node.get_closest_marker("gpu") is not None
    reset_ops()
    initialize_ops(
        OpsConfig(
            backend="flash_attention_4" if gpu else "sdpa_math",
            allow_fallback=False,
        ),
        device=torch.device("cuda", 0) if gpu else torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def make_model(mode="entry_concat", *, mask=0.0, mask_stage="after_e", width=24, flow=None, input="latent"):
    geo = CodecGeometry("test", "test", 4, 8, 2, grid=4)
    cfg = ConditionerConfig(
        input=input,
        **(dict(input_size=8, spatial_patch=2) if input == "rgb" else {}),
        architecture="split_ec",
        width=width,
        depth=4,
        heads=3,
        feature_layers=(3,),
        context_mask_ratio=mask,
        mask_stage=mask_stage,
    )
    world = WorldConfig(
        8,
        2,
        grid=4,
        width=width,
        depth=2,
        heads=3,
        condition_width=width,
        condition_dropout=0,
        adaln_lora_dim=8 if width == 24 else 64,
        condition_mode="entry_concat" if mode == "entry_concat" else "cross_attention",
        cross_attention_scope="same_spatial" if mode == "same_spatial" else "global",
        cross_attention_3d_rope=mode != "entry_concat",
        cross_attention_every=2,
        cross_attention_position="before_self",
    )
    return RepWorld(geo, world, cfg, flow or FlowConfig())


def test_quarter_three_quarters_layer_budget():
    cfg = ConditionerConfig(input="latent", architecture="split_ec", depth=19, feature_layers=(18,))
    assert (cfg.vit_e_depth, cfg.vit_c_depth) == (5, 14)
    explicit = replace(cfg, encoder_depth=4)
    assert (explicit.vit_e_depth, explicit.vit_c_depth) == (4, 15)
    assert ConditionerConfig().vit_e_depth == 0
    # Preserve the preexisting positional input argument.
    assert ConditionerConfig("latent").input == "latent"


@pytest.mark.parametrize(
    "options",
    [
        {"architecture": "bad"},
        {"input": "latent", "architecture": "split_ec", "depth": 1, "feature_layers": (0,)},
        {"encoder_depth": 2},
        {"input": "latent", "architecture": "split_ec", "encoder_depth": 0},
        {"input": "latent", "architecture": "split_ec", "encoder_depth": True},
        {"input": "latent", "architecture": "split_ec", "encoder_depth": 12},
        {"input": "latent", "architecture": "split_ec", "feature_layers": (0,)},
        {"input": "latent", "architecture": "split_ec", "mask_stage": "unknown"},
        {"input": "latent", "architecture": "causal", "mask_stage": "before_e"},
    ],
)
def test_invalid_architecture_rejected(options):
    with pytest.raises(ValueError):
        ConditionerConfig(**options)


def test_e_state_independence_permutation_full_attention_and_no_rng():
    torch.manual_seed(62)
    encoder = make_model(mask=0.8).causal_semantic_encoder.train()
    source = torch.randn(2, 16, 4, 4, 4, requires_grad=True)
    masks = []
    hook = encoder.vit_e.blocks[0].register_forward_pre_hook(
        lambda m, a, kw: masks.append(kw["mask"]), with_kwargs=True
    )
    rng = torch.get_rng_state().clone()
    together = encoder.encode_states(source)
    assert torch.equal(rng, torch.get_rng_state())
    assert masks == [None]
    hook.remove()
    singles = torch.cat([encoder.encode_states(source[:, :, i : i + 1]) for i in range(4)], dim=1)
    torch.testing.assert_close(together, singles, rtol=1e-5, atol=1e-6)
    perm = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(
        encoder.encode_states(source[:, :, perm]), together[:, perm], rtol=0, atol=0
    )
    changed = source.detach().clone()
    changed[:, :, 1:] += 100
    torch.testing.assert_close(
        encoder.encode_states(changed)[:, :1], together[:, :1], rtol=0, atol=0
    )
    together[:, 0, :, 0].sum().backward()
    assert source.grad[:, :, 1:].count_nonzero() == 0
    assert source.grad[:, :, :1].abs().sum() > 0
    # Within one state, the first temporal plane can read the last plane.
    altered = source.detach().clone()
    altered[:, 8:, 0] += torch.randn_like(altered[:, 8:, 0]) * 5
    assert (encoder.encode_states(altered)[:, 0, :16] - together[:, 0, :16]).abs().max() > 1e-5


@pytest.mark.parametrize("mask", [0.0, 0.8])
def test_c_causality_preserves_original_indices_and_masks_after_e(mask):
    encoder = make_model(mask=mask).causal_semantic_encoder.train()
    source = torch.randn(2, 16, 4, 4, 4)
    encoded = encoder.encode_states(source).detach().requires_grad_(True)
    seen = []
    h = encoder.vit_c.blocks[0].register_forward_pre_hook(
        lambda m, a, kw: seen.append((a[0], kw)), with_kwargs=True
    )
    torch.manual_seed(90)
    output, indices = encoder.prefill(encoded, return_indices=True)
    if mask:
        keep = 2 * max(1, int(16 * (1 - mask) + 1e-8))
        assert output.shape == (2, 4, keep, 24)
        assert indices.min() >= 0 and indices.max() < 32
        expected = encoded.gather(2, indices[..., None].expand(-1, -1, -1, 24))
        torch.testing.assert_close(seen[0][0], expected.flatten(1, 2), rtol=0, atol=0)
        # ViT-E always returned all 32 tokens despite configured 80% masking.
        assert encoded.shape == (2, 4, 32, 24)
    else:
        assert indices is None
    changed = encoded.detach().clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:]) * 100
    torch.manual_seed(90)
    other = encoder.prefill(changed)
    torch.testing.assert_close(output[:, :2], other[:, :2], rtol=0, atol=0)
    output[:, 1, :, 0].sum().backward()
    assert encoded.grad[:, 2:].count_nonzero() == 0
    assert encoded.grad[:, :1].abs().sum() > 0
    h.remove()


@pytest.mark.parametrize("condition_input", ["latent", "rgb"])
def test_mask_before_e_uses_original_positions_in_both_modules(condition_input):
    torch.manual_seed(124)
    model = make_model(
        mode="global" if condition_input == "rgb" else "entry_concat",
        mask=0.75,
        mask_stage="before_e",
        input=condition_input,
    ).train()
    encoder = model.causal_semantic_encoder
    geo = model.geometry
    states = 3
    source = (
        torch.randn(2, 3, states * geo.rgb_frames, 8, 8)
        if condition_input == "rgb"
        else torch.randn(2, geo.packed_channels, states, geo.grid, geo.grid)
    )
    seen_e, seen_c = [], []
    hooks = [
        encoder.vit_e.blocks[0].register_forward_pre_hook(
            lambda _, args, kw: seen_e.append((args[0].shape, kw["positions"])),
            with_kwargs=True,
        ),
        encoder.vit_c.blocks[0].register_forward_pre_hook(
            lambda _, args, kw: seen_c.append((args[0].shape, kw["positions"])),
            with_kwargs=True,
        ),
    ]
    try:
        torch.manual_seed(517)
        encoded, local = encoder.encode_states(source, return_indices=True)
        output, returned = encoder.prefill(encoded, selected_indices=local, return_indices=True)
    finally:
        for hook in hooks:
            hook.remove()
    visible = geo.planes * max(1, int(geo.grid**2 * 0.25 + 1e-8))
    assert local.shape == (2, states, visible)
    assert encoded.shape == output.shape == (2, states, visible, 24)
    torch.testing.assert_close(local, returned, rtol=0, atol=0)
    assert seen_e[0][0] == (2 * states, visible, 24)
    assert seen_c[0][0] == (2, states * visible, 24)

    from vforge.models.rep_world.split_conditioner import _positions

    e_full = _positions(1, geo, encoder.config, source.device)
    c_full = _positions(states, geo, encoder.config, source.device)
    global_indices = local + torch.arange(states)[None, :, None] * geo.tokens_per_state
    for axis, observed in enumerate(seen_e[0][1]):
        torch.testing.assert_close(
            observed, e_full[axis][local.reshape(2 * states, visible)].unsqueeze(1),
            rtol=0, atol=0,
        )
    for axis, observed in enumerate(seen_c[0][1]):
        torch.testing.assert_close(
            observed, c_full[axis][global_indices.reshape(2, -1)].unsqueeze(1),
            rtol=0, atol=0,
        )
    # A joint training pass must draw the mask only once and pass it through E and C.
    torch.manual_seed(517)
    joint, joint_indices = encoder(source, return_indices=True)
    torch.testing.assert_close(joint_indices, local, rtol=0, atol=0)
    torch.testing.assert_close(joint, output, rtol=0, atol=0)


def test_mask_before_e_cannot_read_discarded_latent_tokens():
    torch.manual_seed(63)
    model = make_model(mask=0.75, mask_stage="before_e").train()
    encoder = model.causal_semantic_encoder
    geo = model.geometry
    source = torch.randn(1, geo.packed_channels, 2, geo.grid, geo.grid)
    torch.manual_seed(18)
    output, indices = encoder(source, return_indices=True)
    tokens = source.reshape(1, geo.planes, geo.channels, 2, geo.grid, geo.grid)
    tokens = tokens.permute(0, 3, 1, 4, 5, 2).reshape(1, 2, geo.tokens_per_state, geo.channels)
    modified = tokens.clone()
    discarded = torch.ones(1, 2, geo.tokens_per_state, dtype=torch.bool)
    discarded.scatter_(2, indices, False)
    modified[discarded] += 100
    modified = modified.reshape(1, 2, geo.planes, geo.grid, geo.grid, geo.channels)
    modified = modified.permute(0, 2, 5, 1, 3, 4).reshape_as(source)
    torch.manual_seed(18)
    other, other_indices = encoder(modified, return_indices=True)
    torch.testing.assert_close(indices, other_indices, rtol=0, atol=0)
    torch.testing.assert_close(output, other, rtol=0, atol=0)


def test_mask_stage_keeps_dense_eval_and_parameters_identical():
    torch.manual_seed(24)
    old = make_model(mask=0.8, mask_stage="after_e").eval()
    new = make_model(mask=0.8, mask_stage="before_e").eval()
    new.load_state_dict(old.state_dict(), strict=True)
    source = torch.randn(1, 16, 3, 4, 4)
    with torch.no_grad():
        torch.testing.assert_close(
            old(operation="condition", source=source),
            new(operation="condition", source=source),
            rtol=0, atol=0,
        )


def test_mask_before_e_full_training_backward_and_recompute():
    torch.manual_seed(62)
    original = make_model("global", mask=0.8, mask_stage="before_e").train()
    torch.nn.init.normal_(original.net.final_layer.linear.weight, std=0.02)
    recomputed = copy.deepcopy(original)
    recomputed.gradient_checkpointing_enable()
    latents = torch.randn(2, 16, 5, 4, 4)
    for model in (original, recomputed):
        torch.manual_seed(8)
        loss = model(latents=latents)
        assert torch.isfinite(loss)
        loss.backward()
        for parameter in (
            model.causal_semantic_encoder.vit_e.patch_embed.weight,
            model.causal_semantic_encoder.vit_e.blocks[0].attn.qkv.weight,
            model.causal_semantic_encoder.vit_c.blocks[0].attn.qkv.weight,
        ):
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    for (name, first), (_, second) in zip(
        original.named_parameters(), recomputed.named_parameters(), strict=True
    ):
        if first.grad is not None:
            torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0, msg=name)


@pytest.mark.gpu(gpus=1)
def test_mask_before_e_fa4_bf16_training():
    model = make_model("global", mask=0.8, mask_stage="before_e", width=192).cuda().train()
    model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    latents = torch.randn(2, 16, 5, 4, 4, device="cuda")
    with patch.object(dispatch, "_run_sdpa", side_effect=AssertionError("Unexpected SDPA")):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(latents=latents)
        assert torch.isfinite(loss)
        loss.backward()
    for parameter in (
        model.causal_semantic_encoder.vit_e.patch_embed.weight,
        model.causal_semantic_encoder.vit_c.blocks[0].attn.qkv.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_separate_api_matches_joint_and_each_causal_prefix():
    m = make_model("global").eval()
    x = torch.randn(2, 16, 4, 4, 4)
    encoded = m(operation="encode_states", source=x)
    prefill = m(operation="prefill_condition", source=encoded)
    joint = m(operation="condition", source=x)
    torch.testing.assert_close(prefill, joint, rtol=0, atol=0)
    all_states = m.causal_semantic_encoder.prefill(encoded)
    torch.testing.assert_close(joint, all_states[:, -1], rtol=0, atol=0)
    for n in range(1, 5):
        actual = m(operation="prefill_condition", source=encoded[:, :n])
        torch.testing.assert_close(actual, all_states[:, n - 1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial"])
@pytest.mark.parametrize("layout", ["ordinary", "distant", "quadrant"])
def test_joint_loss_backprop_both_modules_and_recompute(mode, layout):
    flow = {
        "ordinary": FlowConfig(),
        "distant": FlowConfig(distant_target_prob=0.25),
        "quadrant": FlowConfig(target_layout="quadrant_mosaic"),
    }[layout]
    torch.manual_seed(42)
    a = make_model(mode, mask=0.8, flow=flow).train()
    torch.nn.init.normal_(a.net.final_layer.linear.weight, std=0.02)
    b = copy.deepcopy(a)
    b.gradient_checkpointing_enable()
    x = torch.randn(2, 16, 6, 4, 4)
    for m in [a, b]:
        torch.manual_seed(77)
        loss = m(latents=x)
        loss.backward()
        assert torch.isfinite(loss)
        for p in [
            m.causal_semantic_encoder.vit_e.patch_embed.weight,
            m.causal_semantic_encoder.vit_e.blocks[0].attn.qkv.weight,
            m.causal_semantic_encoder.vit_c.blocks[0].attn.qkv.weight,
            m.net.final_layer.linear.weight,
        ]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if p.grad is None:
            # CFG dropout is disabled in this gate, so the null condition is unused.
            assert name == "net.null_semantic_context" and q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)


def test_meta_materialization_and_strict_roundtrip():
    a = make_model("global")
    with torch.device("meta"):
        b = make_model("global")
    assert all(p.is_meta for p in b.parameters())
    materialize_meta_module(b, "cpu")
    b.initialize_weights()
    assert all(torch.isfinite(x).all() for x in b.state_dict().values())
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(1, 16, 3, 4, 4)
    torch.testing.assert_close(
        a(operation="condition", source=x), b(operation="condition", source=x), rtol=0, atol=0
    )


class ToyCodec:
    def __init__(self, geo):
        self.geo = geo
        self.inputs = []

    def encode(self, rgb):
        self.inputs.append(rgb.clone())
        return rgb.new_full(
            (
                rgb.shape[0],
                self.geo.packed_channels,
                rgb.shape[2] // self.geo.rgb_frames,
                self.geo.grid,
                self.geo.grid,
            ),
            0.1,
        )

    def stream(self):
        return SimpleNamespace(
            prime=lambda _: None,
            step=lambda z: z.new_zeros(z.shape[0], 3, self.geo.rgb_frames, 8, 8),
        )


@pytest.mark.parametrize("window", [1, 2, 4])
@pytest.mark.parametrize("condition_input", ["latent", "rgb"])
def test_cached_e_rollout_matches_reencoding_and_obeys_engine_boundary(window, condition_input):
    torch.manual_seed(83)
    model = make_model("global", input=condition_input).eval()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    calls = []

    def engine(**kwargs):
        calls.append(kwargs)
        return model(**kwargs)

    cfg = RolloutConfig(5, history_states=2, history_window_states=window, steps=2, cfg_scale=1)
    cond = model.causal_semantic_encoder.config
    codec = ToyCodec(model.geometry)
    cached = RepWorldInference(
        engine, codec, model.geometry, model.flow, cond, model.timestep_shift, cfg
    )
    # Only the inference switch changes; both paths use the SAME split model weights.
    uncached = RepWorldInference(
        model,
        ToyCodec(model.geometry),
        model.geometry,
        model.flow,
        replace(cond, architecture="causal", encoder_depth=None),
        model.timestep_shift,
        cfg,
    )
    rgb = torch.randn(1, 3, 20, 8, 8)
    sample = {"sample_id": "one", "rgb": rgb}
    got = cached.infer_sample(sample, seed=99)
    expected = uncached.infer_sample(sample, seed=99)
    torch.testing.assert_close(got["latents"], expected["latents"], rtol=1e-5, atol=1e-6)
    e = [x for x in calls if x["operation"] == "encode_states"]
    c = [x for x in calls if x["operation"] == "prefill_condition"]
    unit = model.geometry.rgb_frames if condition_input == "rgb" else 1
    assert [x["source"].shape[2] for x in e] == [min(2, window) * unit, unit, unit]
    if condition_input == "rgb":
        assert torch.equal(e[0]["source"], rgb[:, :, max(0, 8 - window * unit) : 8])
        assert all(x["source"].count_nonzero() == 0 for x in e[1:])
    assert [x["source"].shape[1] for x in c] == [min(n, window) for n in [2, 3, 4]]
    assert not any(x["operation"] == "condition" for x in calls)
    assert codec.inputs[0].shape[2] == 8
    changed = rgb.clone()
    changed[:, :, 8:] += 100
    other = cached.infer_sample({"sample_id": "one", "rgb": changed}, seed=99)
    torch.testing.assert_close(other["latents"], got["latents"], rtol=0, atol=0)


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial"])
@pytest.mark.parametrize("mask", [0.0, 0.8])
@pytest.mark.parametrize("quadrant", [False, True])
def test_split_fa4_bf16_backward_and_separate_inference(mode, mask, quadrant):
    flow = FlowConfig(target_layout="quadrant_mosaic" if quadrant else "single_state")
    model = make_model(mode, mask=mask, width=192, flow=flow).cuda().train()
    model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    x = torch.randn(2, 16, 6, 4, 4, device="cuda")
    with (
        patch.object(dispatch, "_run_sdpa", side_effect=AssertionError("Unexpected SDPA")),
        patch.object(
            dispatch, "_run_flash_attention", wraps=dispatch._run_flash_attention
        ) as dense,
        patch.object(
            dispatch, "_run_segmented_attention", wraps=dispatch._run_segmented_attention
        ) as causal,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(latents=x)
        loss.backward()
        assert torch.isfinite(loss)
        for p in [
            model.causal_semantic_encoder.vit_e.patch_embed.weight,
            model.causal_semantic_encoder.vit_c.blocks[0].attn.qkv.weight,
        ]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        assert dense.call_count > 0 and causal.call_count > 0
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model(operation="encode_states", source=x[:, :, :2])
            c = model(operation="prefill_condition", source=encoded)
            direct = model(operation="condition", source=x[:, :, :2])
        torch.testing.assert_close(c, direct, rtol=0, atol=0)


def test_rgb_split_grid_state_independence_and_gradient():
    torch.manual_seed(62)
    model = make_model("global", input="rgb").train()
    encoder = model.causal_semantic_encoder
    source = torch.randn(2, 3, 16, 8, 8, requires_grad=True)
    encoded = encoder.encode_states(source)
    assert encoded.shape == (2, 4, 32, 24)
    singles = torch.cat(
        [encoder.encode_states(source[:, :, i * 4 : (i + 1) * 4]) for i in range(4)], dim=1
    )
    torch.testing.assert_close(encoded, singles, rtol=1e-5, atol=1e-6)
    changed = source.detach().clone()
    changed[:, :, 4:] += 100
    torch.testing.assert_close(
        encoder.encode_states(changed)[:, :1], encoded[:, :1], rtol=0, atol=0
    )
    encoded[:, 0, :, 0].sum().backward()
    assert source.grad[:, :, 4:].count_nonzero() == 0
    assert source.grad[:, :, :4].abs().sum() > 0
    assert encoder.training_context(source).shape == (2, 3, 32, 24)


def test_rgb_split_rejects_unmatched_grid_and_matches_videorae_compression():
    from vforge.models.rep_world.split_conditioner import SplitViTConditioner

    geo = CodecGeometry("videorae", "3d", 16, 64, 4, grid=16)
    cfg = ConditionerConfig(
        input="rgb",
        architecture="split_ec",
        temporal_patch=4,
        spatial_patch=16,
        width=24,
        depth=2,
        heads=3,
        feature_layers=(1,),
    )
    model = SplitViTConditioner(cfg, geo)
    assert model.tokens_per_state == cfg.tokens_per_state(geo) == 1024
    assert model.vit_e.patch_embed.proj.kernel_size == (4, 16, 16)
    with pytest.raises(ValueError, match="P/H/W"):
        SplitViTConditioner(replace(cfg, temporal_patch=2), geo)


@pytest.mark.gpu(gpus=1)
def test_rgb_split_fa4_backward_and_generated_rgb_rollout():
    model = make_model("global", width=192, input="rgb").cuda().train()
    model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    rgb = torch.randn(2, 3, 16, 8, 8, device="cuda")
    latents = torch.randn(2, 16, 4, 4, 4, device="cuda")

    def forbid(*args, **kwargs):
        raise AssertionError("Unexpected SDPA")

    with patch.object(dispatch, "_run_sdpa", new=forbid):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(latents=latents, rgb=rgb)
        loss.backward()
        assert torch.isfinite(loss)
        for param in [
            model.causal_semantic_encoder.vit_e.patch_embed.proj.weight,
            model.causal_semantic_encoder.vit_c.blocks[0].attn.qkv.weight,
        ]:
            assert (
                param.grad is not None
                and torch.isfinite(param.grad).all()
                and param.grad.abs().sum() > 0
            )
        model.eval()
        cond = model.causal_semantic_encoder.config
        runner = RepWorldInference(
            model,
            ToyCodec(model.geometry),
            model.geometry,
            model.flow,
            cond,
            model.timestep_shift,
            RolloutConfig(3, history_states=1, history_window_states=2, steps=2, cfg_scale=1),
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = runner.infer_sample({"sample_id": "rgb", "rgb": rgb[:1]}, seed=99)
        assert torch.isfinite(output["latents"]).all()


def test_rgb_split_mask_after_full_state_encoder():
    model = make_model(mode="cross_attention", mask=0.9, input="rgb").train()
    source = torch.randn(1, 3, 3 * model.geometry.rgb_frames, 8, 8, requires_grad=True)
    encoder = model.causal_semantic_encoder
    full = encoder.encode_states(source)
    assert full.shape == (1, 3, model.geometry.tokens_per_state, 24)
    torch.manual_seed(7)
    visible, indices = encoder.prefill(full, return_indices=True)
    assert visible.shape[0:2] == (1, 3)
    assert 0 < visible.shape[2] < full.shape[2]
    assert indices.shape == visible.shape[:-1]
    visible.sum().backward()
    assert source.grad is not None and source.grad.abs().sum() > 0


def test_rgb_split_masked_k2_full_loss_and_gradients():
    flow = FlowConfig(future_target_choices=2, target_origin_reserve=2, distant_target_prob=0.5)
    model = make_model(mode="cross_attention", mask=0.9, input="rgb", flow=flow).train()
    geo = model.geometry
    latents = torch.randn(1, geo.packed_channels, 10, geo.grid, geo.grid)
    rgb = torch.randn(1, 3, 10 * geo.rgb_frames, 8, 8)
    loss = model(latents=latents, rgb=rgb)
    assert torch.isfinite(loss)
    loss.backward()
    # The DiT output projection starts at zero, so encoder gradients become
    # nonzero only after the first update to that projection.
    output = model.net.final_layer.linear
    assert output.weight.grad is not None and output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        output.weight.add_(output.weight.grad, alpha=-1e-3)
    model.zero_grad(set_to_none=True)
    model(latents=latents, rgb=rgb).backward()
    grad = model.causal_semantic_encoder.vit_e.patch_embed.proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_vit_c_only_mlp_ratio_does_not_change_vit_e():
    baseline = ConditionerConfig(
        input="latent", architecture="split_ec", width=24, depth=5, heads=3,
        encoder_depth=2, feature_layers=(4,), mlp_ratio=2.0,
    )
    expanded = replace(baseline, depth=7, feature_layers=(6,), vit_c_mlp_ratio=3.0)
    geometry = CodecGeometry("test", "test", 4, 8, 2, grid=4)
    from vforge.models.rep_world.split_conditioner import SplitViTConditioner
    old, new = SplitViTConditioner(baseline, geometry), SplitViTConditioner(expanded, geometry)
    assert len(old.vit_e.blocks) == len(new.vit_e.blocks) == 2
    assert len(old.vit_c.blocks) == 3 and len(new.vit_c.blocks) == 5
    assert old.vit_e.blocks[0].mlp.fc1.out_features == 48
    assert new.vit_e.blocks[0].mlp.fc1.out_features == 48
    assert old.vit_c.blocks[0].mlp.fc1.out_features == 48
    assert new.vit_c.blocks[0].mlp.fc1.out_features == 72
    with pytest.raises(ValueError, match="vit_c_mlp_ratio"):
        replace(baseline, vit_c_mlp_ratio=0)
