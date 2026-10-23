"""Opt-in GPU migration parity gate using explicitly supplied research assets.

REP_WORLD_ASSET_ROOT points to the directory containing V-RAE/vrae and
V-RAE/pretrained. REP_WORLD_REFERENCE_ROOT points to the original rep_miniworld
checkout; REP_WORLD_VRAE_SOURCE points to its V-RAE dependency. None is needed
by the Native implementation, and no assets are downloaded by these tests.
"""

import os
from pathlib import Path

import pytest
import torch

from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.codec import load_vrae

_ASSETS = os.environ.get("REP_WORLD_ASSET_ROOT")
_REFERENCE = os.environ.get("REP_WORLD_REFERENCE_ROOT")
_VRAE_SOURCE = os.environ.get("REP_WORLD_VRAE_SOURCE")
pytestmark = pytest.mark.skipif(
    not (_ASSETS and _REFERENCE and _VRAE_SOURCE and torch.cuda.is_available()),
    reason="Requires opt-in local published assets, original reference checkout and CUDA",
)


@pytest.mark.parametrize("variant,release", [("eupe", "eupe"), ("vjepa2_1", "vjepa2.1")])
def test_published_bf16_encode_decode_and_stream_parity(variant, release, monkeypatch):
    monkeypatch.syspath_prepend(_REFERENCE)
    from rep_miniworld.vae.codec import StreamingVAEDecoder
    from rep_miniworld.vae.codec import load_vrae as load_reference

    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cuda"),
        parallel_state=None,
    )
    root = Path(_ASSETS) / "V-RAE"
    path = root / "vrae" / f"vrae_{release}.pt"
    old = load_reference(
        {
            "vrae_checkpoint": str(path),
            "vrae_source_root": _VRAE_SOURCE,
            "vrae_encoder_checkpoint": str(root / "pretrained" / "encoders" / variant / "model.pt"),
            "mixed_precision": "bf16",
        }
    )
    new = load_vrae(path, variant, device="cuda", dtype=torch.bfloat16)
    try:
        assert all(not p.requires_grad for p in new.model.parameters())
        assert not any(p.is_meta for p in new.model.parameters())
        torch.manual_seed(314)
        rgb = torch.rand(1, 3, 8, 256, 256, device="cuda") * 2 - 1
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            original_latent, native_latent = old.encode(rgb), new.encode(rgb)
            torch.testing.assert_close(native_latent, original_latent, atol=0, rtol=0)
            torch.testing.assert_close(
                new.decode(original_latent), old.decode(original_latent), atol=0, rtol=0
            )
            original_stream, native_stream = StreamingVAEDecoder(old), new.stream()
            original_stream.begin()
            original_stream.prime(original_latent[:, :, :1])
            native_stream.prime(original_latent[:, :, :1])
            torch.testing.assert_close(
                native_stream.step(original_latent[:, :, 1:]),
                original_stream.step(original_latent[:, :, 1:]),
                atol=0,
                rtol=0,
            )
            original_stream.end()
    finally:
        reset_ops()
        del old, new
        torch.cuda.empty_cache()
