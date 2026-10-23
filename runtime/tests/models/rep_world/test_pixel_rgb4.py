import pytest
import torch

from vforge.models.rep_world.config import WorldConfig, codec_geometry
from vforge.models.rep_world.pixel_codec import PixelRGB4Codec


def test_rgb4_codec_is_exact_and_preserves_frame_order():
    codec = PixelRGB4Codec(codec_geometry("pixel", "rgb4"))
    video = torch.arange(3 * 8 * 256 * 256, dtype=torch.float32).reshape(1, 3, 8, 256, 256)
    packed = codec.encode(video)
    assert packed.shape == (1, 3072, 2, 16, 16)
    torch.testing.assert_close(codec.decode(packed), video, atol=0, rtol=0)
    stream = codec.stream()
    stream.prime(packed[:, :, :1])
    torch.testing.assert_close(stream.step(packed[:, :, 1:]), video[:, :, 4:], atol=0, rtol=0)
    with pytest.raises(ValueError, match="4\\*S"):
        codec.encode(video[:, :, :-1])


def test_pixel_dit_requires_rgb4_geometry():
    with pytest.raises(ValueError, match="RGB4"):
        WorldConfig(channels=64, state_planes=4, pixel_depth=1)


def test_pixel_dit_two_level_forward_uses_direct_rgb_velocity():
    from vforge.models.rep_world.world import WorldDiT
    from vforge.ops import OpsConfig, initialize_ops, reset_ops

    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"), parallel_state=None,
    )
    try:
        config = WorldConfig(
            channels=3072, state_planes=1, width=192, depth=1, heads=6,
            mlp_ratio=2, condition_width=64, cross_attention_every=1,
            cross_attention_position="before_self", cross_attention_3d_rope=True,
            pixel_depth=1, pixel_width=16,
        )
        model = WorldDiT(config)
        noisy = torch.randn(1, 3072, 1, 16, 16)
        context = torch.randn(1, 4, 64)
        positions = torch.tensor([[0, 1, 17, 18]])
        result = model(
            noisy, torch.rand(1, 1), condition=context,
            condition_positions=positions, target_offsets=torch.zeros(1, dtype=torch.long),
        )
        assert result.shape == noisy.shape
        assert result.abs().max() == 0  # direct velocity head starts at zero
        result.sum().backward()
        assert model.pixel_refiner.output.weight.grad is not None
    finally:
        reset_ops()
