from types import SimpleNamespace

import pytest
import torch

from vforge.models.rep_world.config import codec_geometry
from vforge.models.rep_world.wan22_codec import Wan22RGB4Codec


class FakeWanVAE:
    def __init__(self):
        self.scale = [0, 1]
        self.model = self
        self.last_encode = None
        self.last_decode = None

    def encode(self, rgb, _scale):
        self.last_encode = rgb.clone()
        batch, _, frames, _, _ = rgb.shape
        assert (frames - 1) % 4 == 0
        values = rgb[:, :1, ::4, :1, :1]
        return values.expand(batch, 48, (frames - 1) // 4 + 1, 16, 16).clone()

    def decode(self, latents, _scale):
        self.last_decode = latents.clone()
        frames = [latents[:, :3, :1, :1, :1].expand(-1, -1, -1, 256, 256)]
        for state in range(1, latents.shape[2]):
            frames.append(latents[:, :3, state:state + 1, :1, :1].expand(-1, -1, 4, 256, 256))
        return torch.cat(frames, dim=2)


def test_wan22_rgb4_uses_duplicate_anchor_and_four_frame_groups():
    fake = FakeWanVAE()
    codec = Wan22RGB4Codec(codec_geometry("wan22", "ti2v5b"), "", "cpu", torch.float32, vae=fake)
    rgb = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1).expand(1, 3, 8, 256, 256)
    packed = codec.encode(rgb)
    assert packed.shape == (1, 48, 2, 16, 16)
    assert fake.last_encode.shape[2] == 9
    torch.testing.assert_close(fake.last_encode[:, :, :2], rgb[:, :, :1].expand(-1, -1, 2, -1, -1))
    assert packed[0, 0, :, 0, 0].tolist() == [3.0, 7.0]
    decoded = codec.decode(packed)
    assert decoded.shape == rgb.shape
    assert fake.last_decode.shape[2] == 3
    stream = codec.stream()
    stream.prime(packed[:, :, :1])
    generated = stream.step(packed[:, :, 1:])
    assert generated.shape == (1, 3, 4, 256, 256)


def test_wan22_rgb4_refuses_partial_groups():
    codec = Wan22RGB4Codec(codec_geometry("wan22", "ti2v5b"), "", "cpu", torch.float32, vae=FakeWanVAE())
    with pytest.raises(ValueError, match="4\\*S"):
        codec.encode(torch.zeros(1, 3, 5, 256, 256))
