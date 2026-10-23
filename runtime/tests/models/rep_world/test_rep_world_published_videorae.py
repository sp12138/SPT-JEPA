"""Opt-in published-asset/source parity. Production never imports this reference.

Uses REP_WORLD_ASSET_ROOT, REP_WORLD_REFERENCE_ROOT, REP_WORLD_VIDEORAE_SOURCE.
REP_WORLD_PARITY_DEVICE=cpu (default) tests FP32; cuda tests BF16 with autocast.
"""

import gc
import os
from pathlib import Path

import pytest
import torch

from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.codec import load_videorae

_ASSETS = os.environ.get("REP_WORLD_ASSET_ROOT")
_REFERENCE = os.environ.get("REP_WORLD_REFERENCE_ROOT")
_SOURCE = os.environ.get("REP_WORLD_VIDEORAE_SOURCE")
_DEVICE = os.environ.get("REP_WORLD_PARITY_DEVICE", "cpu")
pytestmark = pytest.mark.skipif(
    not (_ASSETS and _REFERENCE and _SOURCE), reason="Requires opt-in local assets/reference paths"
)


@pytest.mark.parametrize("variant", ["3d", "1d"])
def test_published_video_codec_encode_decode_and_stream(variant, monkeypatch):
    monkeypatch.syspath_prepend(_REFERENCE)
    from rep_miniworld.vae.codec import StreamingVAEDecoder
    from rep_miniworld.vae.codec import load_videorae as load_reference

    cuda = _DEVICE == "cuda"
    if not cuda and torch.cuda.is_available():
        pytest.skip("Use CUDA_VISIBLE_DEVICES='' for CPU source-wrapper parity")
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device(_DEVICE),
        parallel_state=None,
    )
    root = Path(_ASSETS)
    name = "vjepa21_3d.pth" if variant == "3d" else "vjepa_rae.pth"
    checkpoint = root / "VideoRAE" / name
    stats = (
        root
        / "VideoRAE"
        / (
            "solarwm_latent_stats_32clips.pt"
            if variant == "3d"
            else "solarwm_1d_latent_stats_32clips.pt"
        )
    )
    teacher = (
        root / "vjepa2-1-vitb16-384/vjepa2_1_vitb_dist_vitG_384.pt"
        if variant == "3d"
        else root / "vjepa2-vitl-fpc64-256/original/model.pth"
    )
    old = load_reference(
        dict(
            videorae_checkpoint=str(checkpoint),
            videorae_variant=variant,
            videorae_source_root=_SOURCE,
            videorae_vjepa_checkpoint=str(teacher),
            videorae_latent_stats=str(stats),
            mixed_precision="bf16" if cuda else "fp32",
        )
    )
    new = load_videorae(
        checkpoint,
        variant,
        device=_DEVICE,
        dtype=torch.bfloat16 if cuda else torch.float32,
        statistics_path=stats,
    )
    try:
        torch.manual_seed(314)
        rgb = torch.rand(1, 3, 32, 256, 256, device=_DEVICE) * 2 - 1
        with (
            torch.no_grad(),
            torch.autocast(_DEVICE, dtype=torch.bfloat16, enabled=cuda),
        ):
            # The upstream teacher opens its own legacy sdp_kernel() context,
            # re-enabling every backend (including cuDNN). Keep the Native
            # process on the same default SDPA policy; an outer math/flash-only
            # context would apply only to Native and invalidate this comparison.
            reference, actual = old.encode(rgb), new.encode(rgb)
            # FP32 native graph can select algebraically equivalent reduction
            # kernels; BF16 published parity must stay exact.
            tolerance = 0 if cuda else 1e-5
            print(variant, "encode max error", (reference - actual).abs().max().item())
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
            old_rgb, new_rgb = old.decode(reference), new.decode(reference)
            print(variant, "decode max error", (old_rgb - new_rgb).abs().max().item())
            torch.testing.assert_close(new_rgb, old_rgb, atol=tolerance, rtol=tolerance)
            old_stream, new_stream = StreamingVAEDecoder(old), new.stream()
            old_stream.begin()
            old_stream.prime(reference[:, :, :1])
            new_stream.prime(reference[:, :, :1])
            # Source VideoRAE stream accepts explicit per-state time planes.
            planes, channels = new.geometry.planes, new.geometry.channels
            state = reference[:, :, 1:].reshape(1, planes, channels, 1, 16, 16)
            explicit = state.permute(0, 2, 1, 3, 4, 5).reshape(1, channels, planes, 16, 16)
            torch.testing.assert_close(
                new_stream.step(reference[:, :, 1:]),
                old_stream.step(explicit),
                atol=tolerance,
                rtol=tolerance,
            )
            old_stream.end()
    finally:
        reset_ops()
        del old, new
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()
