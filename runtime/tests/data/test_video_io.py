"""CPU tests for the shared media decode/encode helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch


class _StubVideoReader:
    """Stand-in for decord.VideoReader that records which frames were asked for."""

    requested: list[list[int]] = []
    opened: list[str] = []

    def __init__(self, path, total=10, height=4, width=6):
        self.path = path
        _StubVideoReader.opened.append(str(path))
        self.total = total
        self.height = height
        self.width = width

    def __len__(self):
        return self.total

    def get_batch(self, indices):
        _StubVideoReader.requested.append(list(indices))
        return torch.zeros(len(indices), self.height, self.width, 3, dtype=torch.uint8)


def _stub_decord(monkeypatch, **reader_kwargs):
    _StubVideoReader.requested = []
    _StubVideoReader.opened = []
    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(
            bridge=SimpleNamespace(set_bridge=lambda _name: None),
            VideoReader=lambda path: _StubVideoReader(path, **reader_kwargs),
        ),
    )


def test_read_video_window_seeks_only_the_requested_frames(monkeypatch) -> None:
    from vforge.data.video_io import read_video_window

    _stub_decord(monkeypatch, total=1000)
    frames = read_video_window("chunk.mp4", 100, 104)

    assert _StubVideoReader.requested == [[100, 101, 102, 103, 104]]
    assert frames.shape == (5, 3, 4, 6)
    assert frames.dtype == torch.uint8


def test_read_video_window_clamps_a_window_past_the_end(monkeypatch) -> None:
    from vforge.data.video_io import read_video_window

    _stub_decord(monkeypatch, total=10)
    frames = read_video_window("chunk.mp4", 8, 20)

    assert _StubVideoReader.requested == [[8, 9]]
    assert frames.shape[0] == 2


def test_read_video_window_reports_a_missing_decord(monkeypatch) -> None:
    import builtins

    from vforge.data.video_io import read_video_window

    monkeypatch.delitem(sys.modules, "decord", raising=False)
    real_import = builtins.__import__

    def _no_decord(name, *args, **kwargs):
        if name == "decord":
            raise ImportError("No module named 'decord'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_decord)

    with pytest.raises(RuntimeError, match="requires decord"):
        read_video_window("chunk.mp4", 0, 1)


@pytest.mark.parametrize("batched", [False, True])
def test_write_video_mp4_round_trip_preserves_frames_and_color(tmp_path, batched) -> None:
    import av
    import numpy as np

    from vforge.data.video_io import write_video_mp4

    video = torch.full((3, 4, 16, 24), -2.0)
    for channel in range(3):
        video[channel, channel] = 2.0
    video[:, 3] = 0.0
    path = tmp_path / "sample.mp4"
    write_video_mp4(video.unsqueeze(0) if batched else video, str(path), fps=12.0)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.pix_fmt == "yuv420p"
        assert float(stream.average_rate) == 12.0
        frames = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(stream)])

    assert frames.shape == (4, 16, 24, 3)
    assert frames.dtype == np.uint8
    expected = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]])
    # H.264 is lossy; retain color, clipping and frame order within codec rounding.
    np.testing.assert_allclose(frames.mean(axis=(1, 2)), expected, atol=3)


def test_read_video_tensor_resolves_gwm_asset_uri_to_mount(monkeypatch) -> None:
    import vforge.data.video_io as video_io

    monkeypatch.setenv("VFORGE_ASSET_MOUNT", "/mounted-assets")
    monkeypatch.setattr(video_io.os.path, "isfile", lambda path: path.endswith("clip.mp4"))
    _stub_decord(monkeypatch, total=5, height=2, width=4)

    frames = video_io.read_video_tensor(
        "s3://gwm-asset-us-west-2/videos/clip.mp4",
        max_frames=3,
    )

    assert _StubVideoReader.opened == ["/mounted-assets/videos/clip.mp4"]
    assert _StubVideoReader.requested == [[0, 1, 2]]
    assert frames.shape == (3, 3, 2, 4)


def test_read_video_tensor_rejects_zero_frame_decode(monkeypatch) -> None:
    from vforge.data.video_io import read_video_tensor

    _stub_decord(monkeypatch, total=0)

    with pytest.raises(ValueError, match="zero frames"):
        read_video_tensor("/asset/broken.mp4")


def test_read_reference_frames_preserves_manifest_order(monkeypatch) -> None:
    import vforge.data.video_io as video_io

    values = {"ref-a.png": 0.2, "ref-b.png": 0.3}
    monkeypatch.setattr(
        video_io,
        "read_image_tensor",
        lambda path: torch.full((3, 4, 4), values[str(path)]),
    )

    frames = video_io.read_reference_frames({"ref": ["ref-a.png", "ref-b.png"]})

    assert frames.shape == (2, 3, 4, 4)
    assert frames[0].unique().tolist() == [pytest.approx(0.2)]
    assert frames[1].unique().tolist() == [pytest.approx(0.3)]


def test_mp4_with_audio_roundtrip(tmp_path):
    import av
    import torch

    from vforge.data.video_io import write_video_mp4

    path = str(tmp_path / "audio.mp4")
    write_video_mp4(
        torch.zeros(3, 4, 16, 16),
        path,
        fps=4.0,
        audio=torch.zeros(2, 24000),
        audio_sample_rate=24000,
    )
    with av.open(path) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert len(list(container.decode(video=0))) == 4
    with av.open(path) as container:
        samples = sum(frame.samples for frame in container.decode(audio=0))
        assert 24000 <= samples < 24000 + 2048
