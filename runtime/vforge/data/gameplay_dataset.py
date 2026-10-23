"""Gameplay corpus: caption windows cut out of long chunk videos.

The index is a pickle written by taichu's analysis pass. Each entry names a
frame interval inside a chunk mp4 plus, optionally, an aligned normal video, a
foreground mask, and per-frame camera intrinsics and poses. ``__getitem__``
seeks that interval, resizes with a cover-then-crop, and resamples the cameras
to match.

Clips come back in bfloat16. At 149x3x704x1280 a float32 clip is 1.6 GB, and
with several loader workers in flight that alone can exhaust host memory; the
values themselves came from uint8, so bfloat16's ~1/256 resolution loses
nothing that was in the file.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from vforge.data.preprocess import (
    DEFERRED_SPATIAL_KEY,
    NORMAL_MISSING_KEY,
    NORMAL_ROTATION_KEY,
    RAW_NORMAL_MASK_KEY,
    TARGET_HEIGHT_KEY,
    TARGET_WIDTH_KEY,
    NormalRotationAugmentor,
    rotate_normal_frames,
)
from vforge.data.raw_dataset import (
    RawVideoDataset,
    align_4np1,
    pad_or_trim,
    prompt_clean,
    remap_path,
    resize_center_crop,
    unit_normalize_normal,
)
from vforge.data.video_io import read_video_window

logger = logging.getLogger(__name__)

# The index stores intrinsics in the resolution the capture pipeline produced,
# which is not necessarily what the mp4 decodes to, so the crop adjustment
# cannot read it off the frames. Overridable via extras for corpora built at a
# different capture size.
_DEFAULT_CAMERA_SOURCE_HEIGHT = 720
_DEFAULT_CAMERA_SOURCE_WIDTH = 1280
GAMEPLAY_MEDIA_SOURCES_KEY = "__vforge_gameplay_media_sources__"
_GAMEPLAY_MEDIA_SOURCE_KIND = "gameplay_window"


def decode_gameplay_media_source(source: Mapping[str, Any]) -> torch.Tensor:
    """Recreate one decoded gameplay stream from lightweight provenance."""
    if source.get("kind") != _GAMEPLAY_MEDIA_SOURCE_KIND:
        raise ValueError(f"unsupported gameplay media source kind: {source.get('kind')!r}")
    stream = str(source.get("stream") or "")
    if stream not in {"video", "normal", "normal_fg", "normal_bg"}:
        raise ValueError(f"unsupported gameplay media stream: {stream!r}")
    start = int(source["start_frame"])
    end = int(source["end_frame"])
    height = int(source["height"])
    width = int(source["width"])
    raw = read_video_window(str(source["path"]), start, end)
    frames = resize_center_crop(raw.to(torch.float32).mul_(1.0 / 255.0), height, width)
    if stream == "video":
        return frames
    unit_normalize_normal(frames)
    if stream == "normal":
        return frames
    mask_path = source.get("mask_path")
    if not mask_path:
        raise ValueError(f"gameplay {stream} source is missing mask_path")
    raw_mask = read_video_window(str(mask_path), start, end)
    mask = resize_center_crop(
        raw_mask.to(torch.float32).mul_(1.0 / 255.0),
        height,
        width,
    )
    if mask.shape[1] > 1:
        mask = mask.mean(dim=1, keepdim=True)
    mask = mask.clamp_(0.0, 1.0)
    if stream == "normal_fg":
        return (frames * mask).add_((1.0 - mask) * 0.5).clamp_(0.0, 1.0)
    return (frames * (1.0 - mask)).add_(mask * 0.5).clamp_(0.0, 1.0)


@dataclasses.dataclass
class GameplayRecord:
    id: str
    video_path: str
    prompt: str
    start_frame: int
    end_frame: int
    normal_path: str | None = None
    mask_path: str | None = None
    intrinsics: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    fps: float = 30.0
    source: str = ""


class GameplayDataset(RawVideoDataset):
    """Windowed gameplay clips with normal, mask, and camera conditions.

    Recognized ``extras``:

    * ``normal_rot_aug_enabled`` / ``normal_rot_aug_prob`` /
      ``normal_rot_aug_max_angle`` / ``normal_rot_aug_full_so3`` /
      ``normal_rot_aug_sources`` -- train-only normal rotation augmentation.
    * ``use_partial_normal_aug`` -- split the normal into foreground and
      background using the mask video.
    * ``camera_source_height`` / ``camera_source_width`` -- capture resolution
      the stored intrinsics belong to.
    * ``max_num`` -- truncate the index, for smoke runs.
    * ``defer_spatial`` -- hand back raw ``uint8`` frames and leave the resize,
      normal processing, and mask blend to
      :func:`~vforge.data.preprocess.preprocess_deferred_batch`
      on the main rank's device. See that module for why the split exists; the
      pixels it produces are not bit-identical to the in-worker path.

    ``extract_reference_frame`` / ``ref_num_frames`` are handled by the base
    class and work here too, except under ``defer_spatial``.
    """

    def __init__(self, **kwargs: Any) -> None:
        extras = dict(kwargs.get("extras") or {})
        self.path_replacements = tuple(
            (str(source), str(target)) for source, target in extras.get("path_replacements", ())
        )
        super().__init__(**kwargs)

        self.defer_spatial = bool(self.extras.get("defer_spatial", False))
        if self.defer_spatial and self.extract_reference_frame:
            # The base class would slice reference frames off the raw uint8
            # clip, and the deferred stage has no contract for that key -- they
            # would reach the encoder unresized while everything else is scaled.
            raise ValueError(
                "extract_reference_frame is not supported under defer_spatial; "
                "the deferred stage does not process 'reference_frames'"
            )

        self.use_partial_normal_aug = bool(self.extras.get("use_partial_normal_aug", False))
        self.camera_source_height = int(
            self.extras.get("camera_source_height", _DEFAULT_CAMERA_SOURCE_HEIGHT)
        )
        self.camera_source_width = int(
            self.extras.get("camera_source_width", _DEFAULT_CAMERA_SOURCE_WIDTH)
        )
        self.normal_augmentor: NormalRotationAugmentor | None = None
        self.normal_aug_sources: set[str] = set()
        if self.mode == "train" and bool(self.extras.get("normal_rot_aug_enabled", False)):
            self.normal_augmentor = NormalRotationAugmentor(
                prob=float(self.extras.get("normal_rot_aug_prob", 1.0)),
                max_angle_deg=float(self.extras.get("normal_rot_aug_max_angle", 0.0)),
                full_so3=bool(self.extras.get("normal_rot_aug_full_so3", True)),
            )
            self.normal_aug_sources = {
                str(source) for source in (self.extras.get("normal_rot_aug_sources") or [])
            }
            logger.info(
                "gameplay normal rotation augmentation: %s; sources=%s",
                self.normal_augmentor.describe(),
                sorted(self.normal_aug_sources) or "<all>",
            )

    def parse_record(self, raw: Any) -> GameplayRecord | None:
        video_path = raw.get("video_path")
        prompt = raw.get("text")
        if not video_path or prompt is None:
            return None

        interval = raw.get("interval")
        if interval is not None:
            try:
                start, end = int(interval[0]), int(interval[1])
            except (IndexError, TypeError, ValueError):
                return None
        else:
            start = raw.get("start_frame")
            end = raw.get("end_frame")
            if start is None or end is None:
                # No window recorded: take the video from the top and let the
                # decoder's clamp find the real end. -1 is the sentinel for it.
                start, end = 0, -1
        start, end = int(start), int(end)
        if end >= 0 and end < start:
            return None

        span = f"{start}-{end}" if end >= 0 else "full"
        return GameplayRecord(
            id=f"{raw.get('source_id')}:{raw.get('chunk_id')}:{span}",
            video_path=self._map_path(str(video_path)),
            prompt=str(prompt),
            start_frame=start,
            end_frame=end,
            normal_path=(
                self._map_path(str(raw["normal_path"])) if raw.get("normal_path") else None
            ),
            mask_path=self._map_path(str(raw["mask_path"])) if raw.get("mask_path") else None,
            intrinsics=raw.get("Ks"),
            extrinsics=raw.get("c2ws"),
            fps=float(raw.get("fps") or 30.0),
        )

    def _map_path(self, path: str) -> str:
        return remap_path(path, self.path_replacements)

    def plan_window(self, record: GameplayRecord) -> tuple[int, int, int]:
        """Return the absolute inclusive frame range and its aligned length.

        Windows are truncated from the head rather than resampled: the caption
        describes the start of the interval, so dropping the tail keeps text
        and pixels aligned where subsampling would not.
        """
        start = record.start_frame
        available = self.frames if record.end_frame < 0 else record.end_frame - start + 1
        length = align_4np1(min(available, self.frames))
        if length <= 0:
            raise ValueError(
                f"gameplay window too short for the 4n+1 VAE stride: "
                f"available={available}, id={record.id}"
            )
        return start, start + length - 1, length

    def _decode_clip(
        self,
        path: str,
        start: int,
        end_inclusive: int,
        *,
        is_normal: bool,
    ) -> tuple[torch.Tensor, int, int]:
        raw = read_video_window(path, start, end_inclusive)
        source_height, source_width = int(raw.shape[-2]), int(raw.shape[-1])
        if self.defer_spatial:
            return raw, source_height, source_width
        scaled = raw.to(torch.float32).mul_(1.0 / 255.0)
        del raw
        resized = resize_center_crop(scaled, self.height, self.width)
        del scaled
        if is_normal:
            unit_normalize_normal(resized)
        return resized.to(torch.bfloat16), source_height, source_width

    def _decode_mask(self, path: str, start: int, end_inclusive: int) -> torch.Tensor:
        raw = read_video_window(path, start, end_inclusive)
        if self.defer_spatial:
            return raw
        scaled = raw.to(torch.float32).mul_(1.0 / 255.0)
        del raw
        resized = resize_center_crop(scaled, self.height, self.width)
        del scaled
        if resized.shape[1] > 1:
            resized = resized.mean(dim=1, keepdim=True)
        return resized.clamp_(0.0, 1.0).to(torch.bfloat16)

    @staticmethod
    def _resample_camera(array: Any, length: int) -> torch.Tensor | None:
        if array is None:
            return None
        tensor = torch.from_numpy(np.asarray(array)).float()
        have = tensor.shape[0]
        if have >= length:
            return tensor[:length]
        padding = tensor[-1:].expand(length - have, *tensor.shape[1:])
        return torch.cat([tensor, padding], dim=0)

    def _adjust_intrinsics(self, intrinsics: torch.Tensor) -> torch.Tensor:
        """Map intrinsics through the same cover-then-crop the frames took."""
        scale = max(
            self.height / self.camera_source_height,
            self.width / self.camera_source_width,
        )
        adjusted = intrinsics.clone()
        adjusted[:, 0, 0] *= scale
        adjusted[:, 1, 1] *= scale
        # The crop is centered, so the principal point lands at the center of
        # the target regardless of how much overflow was trimmed.
        adjusted[:, 0, 2] = self.width // 2
        adjusted[:, 1, 2] = self.height // 2
        return adjusted

    def build_sample(self, record: GameplayRecord) -> dict[str, Any]:
        start, end_inclusive, length = self.plan_window(record)

        try:
            video, _, _ = self._decode_clip(
                record.video_path, start, end_inclusive, is_normal=False
            )
            video = pad_or_trim(video, length)
        except Exception:
            # One unreadable file should not kill a multi-hour run; a black
            # clip keeps the shape contract and shows up in the loss.
            logger.exception(
                "gameplay video decode failed, substituting black frames: %s [%d:%d]",
                record.video_path,
                start,
                end_inclusive,
            )
            video = torch.zeros(
                length,
                3,
                self.height,
                self.width,
                dtype=torch.uint8 if self.defer_spatial else torch.bfloat16,
            )

        normal: torch.Tensor | None = None
        if record.normal_path:
            try:
                normal, _, _ = self._decode_clip(
                    record.normal_path, start, end_inclusive, is_normal=True
                )
                normal = pad_or_trim(normal, length)
            except Exception:
                logger.exception("gameplay normal decode failed: %s", record.normal_path)
                normal = None
        if normal is None:
            if self.defer_spatial:
                # Empty rather than None because a collate cannot carry None,
                # and empty rather than uint8 grey because 0.5 has no exact
                # uint8 spelling -- the marker lets the deferred stage build the
                # neutral normal at full precision instead of decoding 127/255.
                normal = torch.empty(0, dtype=torch.uint8)
            else:
                # 0.5 is the neutral normal: it decodes to the zero vector, which
                # every rotation fixes, so it stays neutral through augmentation.
                normal = torch.full_like(video, 0.5)

        mask: torch.Tensor | None = None
        if self.use_partial_normal_aug and record.mask_path and record.normal_path:
            try:
                mask = pad_or_trim(
                    self._decode_mask(record.mask_path, start, end_inclusive), length
                )
            except Exception:
                logger.exception("gameplay mask decode failed: %s", record.mask_path)
                mask = None

        normal_fg: torch.Tensor | None = None
        normal_bg: torch.Tensor | None = None
        if mask is not None and not self.defer_spatial:
            # normal * mask + 0.5 * (1 - mask), written in place: a full_like
            # constant at this size would be another ~0.8 GB.
            inverse = 1.0 - mask
            normal_fg = (normal * mask).add_(inverse * 0.5).clamp_(0.0, 1.0)
            normal_bg = (normal * inverse).add_(mask * 0.5).clamp_(0.0, 1.0)
            # Dropped rather than deleted: the deferred branch below reads the
            # name, and releasing the reference frees the same ~0.8 GB.
            del inverse
            mask = None

        # Drawn in both modes so the two paths consume the RNG identically and a
        # given seed picks the same rotation either way; deferred only postpones
        # applying it, because the blend it has to precede happens on the device.
        rotation: torch.Tensor | None = None
        if self.normal_augmentor is not None and (
            not self.normal_aug_sources or record.source in self.normal_aug_sources
        ):
            if self.normal_augmentor.should_apply():
                rotation = self.normal_augmentor.sample_matrix()
                if not self.defer_spatial:
                    normal = rotate_normal_frames(normal, rotation)
                    if normal_fg is not None:
                        normal_fg = rotate_normal_frames(normal_fg, rotation)
                        normal_bg = rotate_normal_frames(normal_bg, rotation)

        sample: dict[str, Any] = {
            "sample_id": record.id,
            "video_frames": video,
            "image_condition": video[0:1],
            "normal_condition": normal,
            "prompt": prompt_clean(record.prompt),
            "height": self.height,
            "width": self.width,
        }
        source_base = {
            "kind": _GAMEPLAY_MEDIA_SOURCE_KIND,
            "start_frame": start,
            "end_frame": end_inclusive,
            "height": self.height,
            "width": self.width,
        }
        media_sources: dict[str, dict[str, Any]] = {
            "gt": {
                **source_base,
                "stream": "video",
                "path": record.video_path,
            }
        }
        if record.normal_path:
            media_sources["normal"] = {
                **source_base,
                "stream": "normal",
                "path": record.normal_path,
            }
        if normal_fg is not None:
            sample["normal_fg_condition"] = normal_fg
            sample["normal_bg_condition"] = normal_bg
            assert record.normal_path is not None
            assert record.mask_path is not None
            for name in ("normal_fg", "normal_bg"):
                media_sources[name] = {
                    **source_base,
                    "stream": name,
                    "path": record.normal_path,
                    "mask_path": record.mask_path,
                }
        sample[GAMEPLAY_MEDIA_SOURCES_KEY] = media_sources
        if self.defer_spatial:
            sample[DEFERRED_SPATIAL_KEY] = True
            sample[TARGET_HEIGHT_KEY] = self.height
            sample[TARGET_WIDTH_KEY] = self.width
            sample[NORMAL_MISSING_KEY] = normal.numel() == 0
            if rotation is not None:
                sample[NORMAL_ROTATION_KEY] = rotation
            if mask is not None:
                sample[RAW_NORMAL_MASK_KEY] = mask

        intrinsics = self._resample_camera(record.intrinsics, length)
        if intrinsics is not None:
            sample["Ks"] = self._adjust_intrinsics(intrinsics)
        extrinsics = self._resample_camera(record.extrinsics, length)
        if extrinsics is not None:
            sample["c2ws"] = extrinsics
        return sample


__all__ = [
    "GAMEPLAY_MEDIA_SOURCES_KEY",
    "GameplayDataset",
    "GameplayRecord",
    "decode_gameplay_media_source",
]
