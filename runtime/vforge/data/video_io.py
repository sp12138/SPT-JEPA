"""Local media decode and encode helpers.

These know about pixels and containers, not about any model's sample schema:
the evaluation runner writes MP4s with them and model-specific collates decode
their condition clips with them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_ASSET_BUCKET_PREFIX = "s3://gwm-asset-us-west-2/"


def _resolve_video_path(path: str) -> str:
    if not path.startswith(_ASSET_BUCKET_PREFIX):
        return path
    asset_mount = os.environ.get(
        "VFORGE_ASSET_MOUNT",
        os.environ.get("TAICHU_ASSET_MOUNT", "/asset"),
    )
    local_path = os.path.join(asset_mount, path.removeprefix(_ASSET_BUCKET_PREFIX))
    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"asset URI {path!r} is not available at its local mount {local_path!r}"
        )
    return local_path


def write_video_mp4(
    video: torch.Tensor,
    path: str,
    *,
    fps: float = 16.0,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
) -> None:
    """Write decoded ``[B,C,T,H,W]`` or ``[C,T,H,W]`` video in ``[-1, 1]``."""
    import av

    if audio is not None:
        if audio_sample_rate is None or audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive when exporting audio")
        if audio.ndim != 2 or audio.shape[0] not in {1, 2}:
            raise ValueError("audio must have shape [1|2, samples]")
    frames = video.detach().cpu()
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError(f"MP4 export expects batch size 1, got {frames.shape[0]}")
        frames = frames[0]
    if frames.ndim != 4 or frames.shape[0] not in {1, 3, 4}:
        raise ValueError(f"MP4 export expects [C,T,H,W] video, got {tuple(frames.shape)}")
    frames = (
        ((frames.float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
    )
    # Preserve the previous writer's integer frame rate and H.264 pixel format.
    rate = int(round(fps)) if isinstance(fps, float) else fps
    with av.open(path, mode="w") as container:
        stream = container.add_stream("h264", rate=rate)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        audio_stream = None
        if audio is not None:
            audio_stream = container.add_stream("aac", rate=audio_sample_rate)
            audio_stream.layout = "mono" if audio.shape[0] == 1 else "stereo"
        for pixels in frames.numpy():
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio_stream is not None:
            from fractions import Fraction

            samples = audio.detach().float().cpu().contiguous().numpy()
            for start in range(0, samples.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(samples[:, start : start + 1024]),
                    format="fltp",
                    layout=audio_stream.layout.name,
                )
                frame.sample_rate = audio_sample_rate
                frame.pts = start
                frame.time_base = Fraction(1, audio_sample_rate)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def write_video_frames_mp4(
    frames: torch.Tensor,
    path: str,
    *,
    fps: float = 16.0,
) -> None:
    """Write ``[T,C,H,W]`` float video frames in ``[0, 1]`` to MP4."""
    if frames.ndim != 4 or frames.shape[1] not in {1, 3, 4}:
        raise ValueError(f"MP4 frame export expects [T,C,H,W] video, got {tuple(frames.shape)}")
    video = frames.permute(1, 0, 2, 3).contiguous().float().mul(2.0).sub(1.0)
    write_video_mp4(video, path, fps=fps)


def write_image_png(image: torch.Tensor, path: str) -> None:
    """Write a ``[C,H,W]`` float image in ``[0, 1]`` to PNG."""
    pixels = image.detach().cpu()
    if pixels.ndim == 4:
        if pixels.shape[0] != 1:
            raise ValueError(f"PNG export expects batch size 1, got {pixels.shape[0]}")
        pixels = pixels[0]
    if pixels.ndim != 3 or pixels.shape[0] not in {1, 3, 4}:
        raise ValueError(f"PNG export expects [C,H,W] image, got {tuple(pixels.shape)}")
    pixels = (
        pixels.float()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    if pixels.shape[-1] == 1:
        pixels = pixels[..., 0]
    Image.fromarray(pixels).save(path, format="PNG")


def _decode_frames(
    resolved_path: str,
    *,
    start: int = 0,
    end_inclusive: int | None = None,
) -> torch.Tensor:
    """Decode a frame range as ``[N, C, H, W]`` uint8. ``None`` end reads on.

    decord is the only decoder here: torchvision removed ``io.read_video`` in
    0.24, and routing both the whole-file and the sought-window read through
    one backend keeps their pixels comparable. There is deliberately no
    whole-file fallback for the windowed read -- decoding a long video to keep
    one window is exactly what that entry point exists to avoid, so a fallback
    would hide the memory blow-up instead of reporting it.
    """
    try:
        import decord
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "video decode requires decord (available via uv sync on Linux "
            "x86_64); torchvision.io.read_video was removed in torchvision 0.24"
        ) from exc

    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(resolved_path)
    try:
        total = len(reader)
        if total == 0:
            raise ValueError(f"video decoder returned zero frames for {resolved_path!r}")
        first = max(0, min(int(start), total - 1))
        last = total - 1 if end_inclusive is None else int(end_inclusive)
        last = max(first, min(last, total - 1))
        frames = reader.get_batch(list(range(first, last + 1)))
    finally:
        del reader

    if frames.ndim == 4 and frames.shape[-1] in {1, 3}:
        frames = frames.permute(0, 3, 1, 2)
    if frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)
    # The permuted channels-last strides are load-bearing, so do not flatten
    # them to NCHW: reductions over the channel dim and convolutions select
    # kernels by memory format, and the two orders round differently in the
    # last bit. A decoder that hands back TCHW pixels (torchvision's
    # ``read_video``, which this replaced) leaves the permute unmaterialized,
    # so materializing it here would shift every normal map and every VAE
    # latent by an ULP against corpora built that way.
    return frames.clone()


def read_video_tensor(path: str, *, max_frames: int | None = None) -> torch.Tensor:
    """Read a local video as ``[T, C, H, W]`` float32 in ``[0, 1]``."""
    resolved_path = _resolve_video_path(path)
    frames = _decode_frames(
        resolved_path,
        end_inclusive=None if max_frames is None else max_frames - 1,
    )
    return frames.float() / 255.0


def read_video_window(path: str, start: int, end_inclusive: int) -> torch.Tensor:
    """Read a sought frame window as ``[N, C, H, W]`` uint8."""
    return _decode_frames(
        _resolve_video_path(str(path)),
        start=start,
        end_inclusive=end_inclusive,
    )


def read_image_tensor(path: str) -> torch.Tensor:
    """Read a local image as ``[C, H, W]`` float32 RGB in ``[0, 1]``."""
    with Image.open(Path(_resolve_video_path(path))) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def resolve_reference_paths(sample: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve compatible ``ref``/``reference`` aliases to ordered paths."""

    def normalize(value: Any, *, field: str) -> tuple[str, ...]:
        if value is None or value == "":
            return ()
        if isinstance(value, (str, Path)):
            values: Sequence[Any] = (value,)
        elif isinstance(value, Sequence):
            values = value
        else:
            raise TypeError(f"Wan {field!r} must be a path or a list of paths")
        paths = tuple(str(item) for item in values)
        if not paths or any(not path for path in paths):
            raise ValueError(f"Wan {field!r} must contain at least one non-empty path")
        return paths

    ref = normalize(sample.get("ref"), field="ref")
    reference = normalize(sample.get("reference"), field="reference")
    if ref and reference and ref != reference:
        raise ValueError("Wan sample declares different 'ref' and 'reference' paths")
    return ref or reference


def read_reference_frames(sample: Mapping[str, Any]) -> torch.Tensor | None:
    """Read ordered reference image paths or first frames of reference videos."""
    paths = resolve_reference_paths(sample)
    if not paths:
        return None
    frames: list[torch.Tensor] = []
    for path in paths:
        if Path(path).suffix.lower() in _IMAGE_SUFFIXES:
            frames.append(read_image_tensor(path))
            continue
        video = read_video_tensor(path, max_frames=1)
        if video.shape[0] < 1:
            raise ValueError(f"Wan reference video has no frames: {path}")
        frames.append(video[0])
    geometry = tuple(frames[0].shape)
    if any(tuple(frame.shape) != geometry for frame in frames[1:]):
        raise ValueError(
            f"Wan reference paths must share [C, H, W] geometry, got "
            f"{[tuple(frame.shape) for frame in frames]}"
        )
    return torch.stack(frames, dim=0)


__all__ = [
    "read_image_tensor",
    "read_reference_frames",
    "read_video_tensor",
    "read_video_window",
    "resolve_reference_paths",
    "write_image_png",
    "write_video_frames_mp4",
    "write_video_mp4",
]
