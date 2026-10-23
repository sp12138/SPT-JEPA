import json
from types import SimpleNamespace

import pytest
import torch

from vforge.data.loader import build_resumable_loader
from vforge.tasks.rep_world.data import VideoClipDataset, frame_indices


class Reader:
    def __len__(self):
        return 40

    def get_batch(self, indices):
        pixels = torch.tensor(indices, dtype=torch.uint8)[:, None, None, None].expand(
            -1, 256, 256, 3
        )
        return SimpleNamespace(asnumpy=lambda: pixels.numpy())


def dataset(tmp_path, monkeypatch):
    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(dict(sample_id=str(i), rgb_path=f"{i}.mp4")) for i in range(8))
    )
    monkeypatch.setattr("vforge.tasks.rep_world.data.open_video", lambda path: Reader())
    return VideoClipDataset(manifest, frames=4, frame_interval=2)


def test_original_round_random_start_stride_resize_and_layout(tmp_path, monkeypatch):
    clips = dataset(tmp_path, monkeypatch)
    assert clips.records[0]["path"] == str(tmp_path / "0.mp4")
    torch.manual_seed(18)
    fraction = torch.rand(()).item()
    expected = torch.tensor([round(fraction * 33) + i * 2 for i in range(4)])
    torch.manual_seed(18)
    item = clips[0]
    assert item["rgb"].shape == (3, 4, 256, 256)
    torch.testing.assert_close(
        item["rgb"][0, :, 0, 0], expected.float() / 255 * 2 - 1, atol=0, rtol=0
    )
    assert frame_indices(40, 4, 2, random_clip=False) == [0, 2, 4, 6]
    with pytest.raises(ValueError, match="requires 7"):
        frame_indices(6, 4, 2, random_clip=False)


def test_resumable_loader_restores_next_sample_and_clip(tmp_path, monkeypatch):
    clips = dataset(tmp_path, monkeypatch)
    state = SimpleNamespace(
        batch_group=None, is_batch_leader=True, dp_rank=0, config=SimpleNamespace(dp_size=1)
    )

    def build():
        return build_resumable_loader(
            clips,
            parallel_state=state,
            batch_size=2,
            num_workers=0,
            drop_last=True,
            seed=12,
            shuffle=True,
        )

    loader = build()
    iterator = iter(loader)
    next(iterator)
    checkpoint, rng = loader.state_dict(), torch.get_rng_state()
    expected = next(iterator)
    restored = build()
    restored.load_state_dict(checkpoint)
    torch.set_rng_state(rng)
    actual = next(iter(restored))
    assert actual["sample_id"] == expected["sample_id"]
    torch.testing.assert_close(actual["rgb"], expected["rgb"], atol=0, rtol=0)


def test_short_or_broken_sample_is_not_silently_replaced(tmp_path, monkeypatch):
    clips = dataset(tmp_path, monkeypatch)
    clips.frames = 50
    with pytest.raises(RuntimeError, match="Sample 0"):
        clips[0]


def test_manifest_index_does_not_stat_each_video(tmp_path, monkeypatch):
    import os

    manifest = tmp_path / "clips.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(
                dict(
                    sample_id=str(i),
                    rgb_path=str(tmp_path / "assets" / f"{i}.mp4") if i % 2 else f"assets/{i}.mp4",
                )
            )
            for i in range(200)
        )
    )
    original_lstat = os.lstat

    def metadata_lookup(path, *args, **kwargs):
        if "assets" in os.fspath(path).split("/"):
            raise AssertionError("Video paths must not be traversed while indexing a manifest")
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", metadata_lookup)
    clips = VideoClipDataset(manifest, frames=4)
    assert len(clips) == 200
    assert clips.records[0]["path"] == str(tmp_path / "assets" / "0.mp4")
    assert clips.records[1]["path"] == str(tmp_path / "assets" / "1.mp4")
    with pytest.raises(RuntimeError, match="Sample 0"):
        clips[0]
