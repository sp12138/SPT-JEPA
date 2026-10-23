import torch

from vforge.models.rep_world.config import (
    ConditionerConfig,
    RepaConfig,
    WorldConfig,
    codec_geometry,
)
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.models.rep_world.pixel_codec import PixelRGB4Codec
from vforge.models.rep_world.repa import TemporalREPAProjector
from vforge.ops import OpsConfig, initialize_ops, reset_ops


def test_rgb16_pack_is_exact_and_keeps_16_frame_state_boundaries():
    geometry = codec_geometry("pixel", "rgb16")
    codec = PixelRGB4Codec(geometry)
    rgb = torch.arange(3 * 32 * 256 * 256, dtype=torch.float32).reshape(1, 3, 32, 256, 256)
    packed = codec.encode(rgb)
    assert packed.shape == (1, 12288, 2, 16, 16)
    torch.testing.assert_close(codec.decode(packed), rgb, atol=0, rtol=0)
    torch.testing.assert_close(codec.decode(packed[:, :, 1:]), rgb[:, :, 16:], atol=0, rtol=0)


def test_conv3d_student_alignment_upsamples_only_time_and_backpropagates():
    hidden = torch.randn(1, 4 * 16 * 16, 32, requires_grad=True)
    projector = TemporalREPAProjector(32, 16)
    output = projector(hidden, planes=4)
    assert output.shape == (1, 16, 8, 16, 16)
    output.square().mean().backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_rgb16_pixeldit_flow_and_repa_reach_student_modules():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"), parallel_state=None,
    )
    try:
        geometry = codec_geometry("pixel", "rgb16")
        world = WorldConfig(
            channels=3072, state_planes=4, width=96, depth=2, heads=3,
            mlp_ratio=2, condition_width=64, cross_attention_every=2,
            cross_attention_position="before_self", cross_attention_3d_rope=True,
            pixel_depth=1, pixel_width=16,
        )
        conditioner = ConditionerConfig(
            input="rgb", input_size=256, temporal_patch=4, spatial_patch=16,
            width=64, depth=3, heads=4, feature_layers=(2,), mlp_ratio=2,
            context_mask_ratio=0.9, architecture="split_ec", encoder_depth=1,
        )
        flow = FlowConfig(
            mode="no_zprev", distant_target_prob=0.5,
            target_origin_reserve=2, future_target_choices=2,
        )
        repa = RepaConfig(
            enabled=True, weight=0.5, align_layer=1,
            teacher_checkpoint="test-checkpoint",
        )
        model = RepWorld(geometry, world, conditioner, flow, repa).train()
        model.initialize_weights()
        rgb = torch.randn(1, 3, 48, 256, 256).clamp(-1, 1)
        packed = PixelRGB4Codec(geometry).encode(rgb)
        teacher = torch.randn(1, 3, 768, 8, 16, 16)
        result = model(
            latents=packed, rgb=rgb, teacher_features=teacher,
            return_metrics=True,
        )
        assert set(("loss_fm", "loss_repa")) <= result["metrics"].keys()
        torch.testing.assert_close(
            result["loss"],
            result["metrics"]["loss_fm"] + 0.5 * result["metrics"]["loss_repa"],
        )
        result["loss"].backward()
        assert model.causal_semantic_encoder.vit_e.patch_embed.proj.weight.grad is not None
        assert model.net.pixel_refiner.output.weight.grad is not None
        assert model.repa_projector.up.weight.grad is not None
        assert model.repa_projector.up.weight.grad.abs().sum() > 0
    finally:
        reset_ops()
