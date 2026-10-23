"""Self-forcing corpora: paired RGB and normal chunks, JSON-indexed.

Two datasets share one index schema. :class:`SelfForcingDataset` treats each
entry as an independent training clip and subsamples it down to ``frames``;
:class:`SelfForcingLongVideoDataset` groups entries by uuid and concatenates
them back into the full take, which the ODE-init inference path walks causally.

Both stretch to the target size instead of cover-and-crop, so the aspect ratio
is not preserved. That is inherited from taichu rather than chosen here, and it
is why :func:`vforge.data.raw_dataset.resize_exact` exists next to gameplay's
cover-and-crop variant.

The two diverge on normals, and it is taichu that splits them.
:class:`SelfForcingDataset` renormalizes, because interpolation blends
neighboring normals and leaves the vectors short of unit length.
:class:`SelfForcingLongVideoDataset` does not: taichu overrides the shared
resize there with one that accepts ``is_normal`` and discards it. Keeping the
split is what makes each class agree with the latents already cached for it.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from typing import Any

import torch

from vforge.data.raw_dataset import (
    RawVideoDataset,
    align_4np1,
    prompt_clean,
    resize_exact,
    unit_normalize_normal,
)
from vforge.data.video_io import read_video_tensor

logger = logging.getLogger(__name__)

_FRAME_SAMPLE_MODES = ("uniform", "head")


def resample_by_index(
    frames: torch.Tensor,
    target_frames: int,
    *,
    mode: str = "uniform",
) -> torch.Tensor:
    """Trim ``[T, C, H, W]`` to the VAE stride, then subsample down to target.

    Taichu's rule, and the reason this is not
    :func:`vforge.data.raw_dataset.pad_or_trim`: the clip is first rounded down
    to ``4n + 1``, and from there it is only ever shortened. A source at or
    below the target comes back at its own aligned length instead of being
    padded up to it, so mixed-length sources stay ragged -- which the
    self-forcing recipes tolerate by running at batch size 1.

    ``uniform`` spreads the kept indices across the whole clip, matching how
    the ODE trajectories were collected. ``head`` keeps the leading frames,
    for the callers that need the clip's start rather than its span.
    """
    if mode not in _FRAME_SAMPLE_MODES:
        raise ValueError(f"frame_sample_mode must be one of {_FRAME_SAMPLE_MODES}, got {mode!r}")

    aligned = align_4np1(frames.shape[0])
    if aligned < 5:
        raise ValueError(
            f"source clip is too short for the 4n+1 VAE stride: {frames.shape[0]} frames "
            f"align down to {aligned} (need >= 5)"
        )
    frames = frames[:aligned]
    if aligned <= target_frames:
        return frames
    if mode == "head":
        return frames[:target_frames]
    indices = torch.linspace(0, aligned - 1, target_frames).round().long().clamp_(0, aligned - 1)
    return frames[indices]


@dataclasses.dataclass
class SelfForcingRecord:
    id: str
    video_path: str
    normal_path: str
    prompt: str = ""
    ode_path: str | None = None
    uuid: str = ""
    source: str = ""


@dataclasses.dataclass
class LongVideoChunk:
    video_path: str
    normal_path: str
    num_frames: int
    chunk_id: str


@dataclasses.dataclass
class LongVideoRecord:
    id: str
    uuid: str
    prompt: str
    chunks: list[LongVideoChunk]
    source: str = ""


def _video_info(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return raw.get("file", {}).get("video", {}) or {}


class SelfForcingDataset(RawVideoDataset):
    """One training clip per index entry, subsampled down to ``frames``.

    Where an ``ode_path`` is present the sample carries ``ode_pairs``, the
    solver trajectory the distillation loss scores against.

    Frame timing and preprocessing follow the same rules the ODE trajectories
    were collected under, because the condition latents re-encoded here have to
    line up with them frame for frame: align to the VAE's ``4n + 1`` stride,
    subsample but never pad up, and renormalize the normal stream after the
    resize.

    The prompt is the index's first caption, cleaned. An entry without one
    carries an empty prompt rather than being dropped, which is what the
    collection side does -- the corpora that do have captions have them on
    every entry, so dropping would only ever fire on a malformed index.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frame_sample_mode = str(self.extras.get("frame_sample_mode", "uniform"))
        if self._frame_sample_mode not in _FRAME_SAMPLE_MODES:
            raise ValueError(
                f"frame_sample_mode must be one of {_FRAME_SAMPLE_MODES}, "
                f"got {self._frame_sample_mode!r}"
            )
        # Last clip decoded per stream, reused when a later read fails. The two
        # are tracked apart so an unreadable normal does not also discard the
        # RGB that decoded fine beside it. This is per-worker state, so which
        # clip stands in depends on the worker's history -- acceptable for a
        # rare-failure path, but a failing run is not bit-reproducible.
        self._last_video: torch.Tensor | None = None
        self._last_normal: torch.Tensor | None = None

    def parse_record(self, raw: Any) -> SelfForcingRecord | None:
        info = _video_info(raw)
        video_path = info.get("video_path")
        normal_path = info.get("normal_path")
        if not video_path or not normal_path:
            return None
        captions = (raw.get("file", {}) or {}).get("text_caption") or []
        prompt = str(captions[0].get("caption", "")) if captions else ""
        return SelfForcingRecord(
            id=str(raw.get("id", "")),
            uuid=str(raw.get("uuid", "")),
            video_path=str(video_path),
            normal_path=str(normal_path),
            ode_path=str(info["ode_path"]) if info.get("ode_path") else None,
            prompt=prompt_clean(prompt),
        )

    def _decode_stream(self, path: str, *, is_normal: bool) -> torch.Tensor:
        # read_video_tensor already scales to [0, 1], so unlike taichu there is
        # no second division by 255 here.
        frames = resample_by_index(
            read_video_tensor(path), self.frames, mode=self._frame_sample_mode
        )
        frames = resize_exact(frames, self.height, self.width)
        if is_normal:
            unit_normalize_normal(frames)
        return frames

    def _load_clip_pair(self, record: SelfForcingRecord) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            video = self._decode_stream(record.video_path, is_normal=False)
            self._last_video = video
        except Exception:
            if self._last_video is None:
                raise
            logger.exception(
                "self-forcing video decode failed for id=%s, reusing the previous clip",
                record.id,
            )
            video = self._last_video

        try:
            normal = self._decode_stream(record.normal_path, is_normal=True)
            self._last_normal = normal
        except Exception:
            if self._last_normal is None:
                raise
            logger.exception(
                "self-forcing normal decode failed for id=%s, reusing the previous clip",
                record.id,
            )
            normal = self._last_normal

        if video.shape[0] != normal.shape[0]:
            raise ValueError(
                f"video/normal frame counts disagree for id={record.id}: "
                f"{video.shape[0]} vs {normal.shape[0]}"
            )
        return video, normal

    def build_sample(self, record: SelfForcingRecord) -> dict[str, Any]:
        video, normal = self._load_clip_pair(record)
        sample: dict[str, Any] = {
            "sample_id": record.id,
            "video_frames": video,
            "normal_condition": normal,
            "image_condition": video[0:1],
            "prompt": record.prompt,
            "height": self.height,
            "width": self.width,
        }
        if record.ode_path is not None:
            # Index 0 strips the batch axis the solver dump was written with.
            ode = torch.load(record.ode_path, map_location="cpu", weights_only=True)
            sample["ode_pairs"] = ode[0]
        return sample


class SelfForcingLongVideoDataset(RawVideoDataset):
    """One record per uuid: every chunk of a take, concatenated end to end.

    Chunk boundaries overlap by a frame, so all but the first chunk drop their
    leading frame during the join. ``frames`` is the fallback length for a
    chunk whose index entry omits ``num_frames``; it does not bound the
    result, which is as long as the take.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("mode", "val")
        super().__init__(**kwargs)
        self._last_good: tuple[torch.Tensor, torch.Tensor] | None = None
        logger.info(
            "long-video dataset: %d takes, %d chunks total",
            len(self.records),
            sum(len(record.chunks) for record in self.records),
        )

    def build_records(self, raw_entries: list[Any]) -> list[LongVideoRecord]:
        grouped: dict[str, list[Any]] = defaultdict(list)
        for raw in raw_entries:
            uuid = raw.get("uuid") if isinstance(raw, dict) else None
            if uuid:
                grouped[str(uuid)].append(raw)

        records: list[LongVideoRecord] = []
        for uuid, entries in grouped.items():
            # Chunk order is the id's lexical order, which is what taichu
            # relies on; the ids are zero-padded so it matches capture order.
            ordered = sorted(entries, key=lambda entry: str(entry.get("id", "")))
            chunks: list[LongVideoChunk] = []
            for entry in ordered:
                info = _video_info(entry)
                video_path = info.get("video_path")
                normal_path = info.get("normal_path")
                if not video_path or not normal_path:
                    continue
                chunks.append(
                    LongVideoChunk(
                        video_path=str(video_path),
                        normal_path=str(normal_path),
                        num_frames=int(info.get("num_frames", self.frames)),
                        chunk_id=str(entry.get("id", "")),
                    )
                )
            if not chunks:
                continue
            captions = ordered[0].get("file", {}).get("text_caption", []) or []
            records.append(
                LongVideoRecord(
                    id=uuid,
                    uuid=uuid,
                    prompt=str(captions[0].get("caption", "")) if captions else "",
                    chunks=chunks,
                )
            )
        return records

    def _load_chunk(self, chunk: LongVideoChunk) -> tuple[torch.Tensor, torch.Tensor]:
        video = read_video_tensor(chunk.video_path)
        normal = read_video_tensor(chunk.normal_path)
        clips = []
        for clip in (video, normal):
            missing = chunk.num_frames - clip.shape[0]
            if missing > 0:
                clip = torch.cat([clip, clip[-1:].repeat(missing, 1, 1, 1)], dim=0)
            clips.append(resize_exact(clip[: chunk.num_frames], self.height, self.width))
        return clips[0], clips[1]

    def build_sample(self, record: LongVideoRecord) -> dict[str, Any]:
        videos: list[torch.Tensor] = []
        normals: list[torch.Tensor] = []
        for chunk in record.chunks:
            try:
                video, normal = self._load_chunk(chunk)
                self._last_good = (video, normal)
            except Exception:
                if self._last_good is None:
                    raise
                logger.exception(
                    "long-video chunk decode failed for uuid=%s id=%s, reusing the previous chunk",
                    record.uuid,
                    chunk.chunk_id,
                )
                video, normal = self._last_good
            videos.append(video)
            normals.append(normal)

        video_frames = torch.cat([videos[0], *(clip[1:] for clip in videos[1:])], dim=0)
        normal_frames = torch.cat([normals[0], *(clip[1:] for clip in normals[1:])], dim=0)
        return {
            "sample_id": record.uuid,
            "video_frames": video_frames,
            "normal_condition": normal_frames,
            "image_condition": video_frames[0:1],
            # Empty for the same reason as SelfForcingDataset: taichu never
            # feeds the parsed caption through on this path.
            "prompt": "",
            "uuid": record.uuid,
            "num_chunks": len(record.chunks),
            "height": self.height,
            "width": self.width,
        }


__all__ = [
    "LongVideoChunk",
    "LongVideoRecord",
    "SelfForcingDataset",
    "SelfForcingLongVideoDataset",
    "SelfForcingRecord",
    "resample_by_index",
]
