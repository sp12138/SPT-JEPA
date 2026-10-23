"""Deterministic CPU layout construction for multimodal generation streams."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .codec import StrideTemporalMapping, TemporalMapping

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

NO_REFERENCE = 0
IMAGE_REFERENCE = 1
VIDEO_REFERENCE = 2
AUDIO_REFERENCE = 3
REFERENCE_TYPE_IDS = {
    "image": 0,
    "video_audio": 1,
    "video": 2,
    "audio": 3,
}

PatchSize = tuple[int, int, int]


def _validate_patch_size(patch_size: Sequence[int]) -> PatchSize:
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must have three entries, got {tuple(patch_size)}")
    normalized = tuple(int(value) for value in patch_size)
    if any(value <= 0 for value in normalized):
        raise ValueError(f"patch_size entries must be positive, got {normalized}")
    return normalized  # type: ignore[return-value]


def _validate_effective_fps(value: float | None, *, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} requires effective_fps")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{label} effective_fps must be finite and positive, got {value}")
    return fps


@dataclass(frozen=True)
class VisualGeometry:
    """Shape and VAE-input clock rate for one ``[C,T,H,W]`` visual latent."""

    shape: tuple[int, int, int, int]
    effective_fps: float | None
    is_image: bool = False

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 4 or any(value <= 0 for value in shape):
            raise ValueError(f"visual latent shape must be positive [C,T,H,W], got {shape}")
        object.__setattr__(self, "shape", shape)
        if self.is_image:
            if shape[1] != 1:
                raise ValueError(f"image visual latent must have T=1, got shape {shape}")
        else:
            object.__setattr__(
                self,
                "effective_fps",
                _validate_effective_fps(self.effective_fps, label="visual latent"),
            )

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def num_frames(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[2]

    @property
    def width(self) -> int:
        return self.shape[3]

    def grid_shape(self, patch_size: Sequence[int]) -> tuple[int, int, int]:
        patch_t, patch_h, patch_w = _validate_patch_size(patch_size)
        if self.num_frames % patch_t or self.height % patch_h or self.width % patch_w:
            raise ValueError(
                f"visual shape {self.shape} is not divisible by patch {(patch_t, patch_h, patch_w)}"
            )
        return (
            self.num_frames // patch_t,
            self.height // patch_h,
            self.width // patch_w,
        )

    def row_count(self, patch_size: Sequence[int]) -> int:
        grid_t, grid_h, grid_w = self.grid_shape(patch_size)
        return grid_t * grid_h * grid_w

    def row_width(self, patch_size: Sequence[int]) -> int:
        patch_t, patch_h, patch_w = _validate_patch_size(patch_size)
        return self.channels * patch_t * patch_h * patch_w


@dataclass(frozen=True)
class AudioGeometry:
    """Shape and VAE-input clock rate for one ``[A,C,T]`` audio latent."""

    shape: tuple[int, int, int]
    effective_fps: float

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError(f"audio latent shape must be positive [A,C,T], got {shape}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "effective_fps",
            _validate_effective_fps(self.effective_fps, label="audio latent"),
        )

    @property
    def num_channels(self) -> int:
        return self.shape[0]

    @property
    def row_width(self) -> int:
        return self.shape[1]

    @property
    def num_frames(self) -> int:
        return self.shape[2]

    @property
    def row_count(self) -> int:
        return self.num_channels * self.num_frames


@dataclass(frozen=True)
class ReferenceGeometry:
    """Geometry of one already-canonical reference item."""

    type: str
    visual: VisualGeometry | None = None
    audio: AudioGeometry | None = None

    def __post_init__(self) -> None:
        if self.type not in {"image", "video_audio", "video", "audio"}:
            raise ValueError(f"unknown canonical reference type {self.type!r}")
        expected = {
            "image": (True, False),
            "video_audio": (True, True),
            "video": (True, False),
            "audio": (False, True),
        }[self.type]
        actual = (self.visual is not None, self.audio is not None)
        if actual != expected:
            raise ValueError(
                f"reference type {self.type!r} requires visual/audio={expected}, got {actual}"
            )
        if self.visual is not None and self.visual.is_image != (self.type == "image"):
            raise ValueError(
                f"reference type {self.type!r} disagrees with visual is_image="
                f"{self.visual.is_image}"
            )


@dataclass(frozen=True)
class TargetGeometry:
    """Target stream geometry, whether or not clean target latents exist."""

    visual: VisualGeometry | None = None
    audio: AudioGeometry | None = None

    def __post_init__(self) -> None:
        if self.visual is None and self.audio is None:
            raise ValueError("target geometry must contain a visual or audio stream")
        if self.visual is not None and self.visual.is_image:
            raise ValueError("generation target visual geometry cannot be an image")


@dataclass(frozen=True)
class PackedLayout(Mapping[str, Any]):
    """All row-wise metadata for a physically packed multimodal sequence."""

    position_ids: torch.Tensor
    token_tags: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    reference_modality_ids: torch.Tensor
    reference_index_ids: torch.Tensor
    audio_plane_ids: torch.Tensor
    reference_item_types: torch.Tensor
    reference_item_image_indices: torch.Tensor
    reference_item_video_indices: torch.Tensor
    reference_item_audio_indices: torch.Tensor
    reference_item_time_ranges: torch.Tensor
    video_loss_mask: torch.Tensor
    audio_loss_mask: torch.Tensor
    sequence_loss_mask: torch.Tensor
    num_reference_video_rows: int
    num_reference_audio_rows: int
    num_target_video_rows: int
    num_target_audio_rows: int
    target_start: float
    target_end: float
    target_visual_shape: tuple[int, int, int, int] | None
    target_audio_shape: tuple[int, int, int] | None
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_channels: int
    num_audio_latents: int

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


@dataclass
class _StreamParts:
    positions: list[torch.Tensor]
    modality_ids: list[torch.Tensor]
    index_ids: list[torch.Tensor]
    loss_masks: list[torch.Tensor]
    plane_ids: list[torch.Tensor]


def _visual_positions(
    geometry: VisualGeometry,
    *,
    start: float,
    patch_size: PatchSize,
    temporal_mapping: TemporalMapping,
) -> tuple[torch.Tensor, float]:
    grid_t, grid_h, grid_w = geometry.grid_shape(patch_size)
    patch_t = patch_size[0]
    if geometry.is_image:
        times = torch.full((grid_t,), float(start), dtype=torch.float32)
    else:
        input_rate = _validate_effective_fps(
            geometry.effective_fps,
            label="visual latent",
        )
        times = start + temporal_mapping.local_times(
            geometry.num_frames,
            input_rate=input_rate,
            patch_t=patch_t,
        )
    spatial_rows = grid_h * grid_w
    temporal = times.repeat_interleave(spatial_rows)
    height = torch.arange(grid_h, dtype=torch.float32).repeat_interleave(grid_w).repeat(grid_t)
    width = torch.arange(grid_w, dtype=torch.float32).repeat(grid_t * grid_h)
    return torch.stack((temporal, height, width), dim=1), float(times[-1])


def _audio_positions(
    geometry: AudioGeometry,
    *,
    start: float,
    temporal_mapping: TemporalMapping,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    local_times = start + temporal_mapping.local_times(
        geometry.num_frames,
        input_rate=geometry.effective_fps,
    )
    # Audio rows are A-major then time-major, so each channel repeats the same clock.
    times = local_times.repeat(geometry.num_channels)
    positions = torch.zeros((geometry.row_count, 3), dtype=torch.float32)
    positions[:, 0] = times
    plane_ids = torch.arange(
        1,
        geometry.num_channels + 1,
        dtype=torch.long,
    ).repeat_interleave(geometry.num_frames)
    return positions, float(local_times[-1]), plane_ids


def _append_identity(
    parts: _StreamParts,
    positions: torch.Tensor,
    *,
    modality_id: int,
    reference_index: int,
    is_target: bool,
) -> None:
    rows = int(positions.shape[0])
    parts.positions.append(positions)
    parts.modality_ids.append(torch.full((rows,), modality_id, dtype=torch.long))
    parts.index_ids.append(torch.full((rows,), reference_index, dtype=torch.long))
    parts.loss_masks.append(torch.full((rows,), is_target, dtype=torch.bool))


def _cat_or_empty(parts: list[torch.Tensor], *, shape: tuple[int, ...], dtype: torch.dtype):
    if parts:
        return torch.cat(parts, dim=0)
    return torch.empty(shape, dtype=dtype)


def build_packed_layout(
    text_length: int,
    references: Sequence[ReferenceGeometry],
    target: TargetGeometry,
    *,
    patch_size: Sequence[int] = (1, 2, 2),
    visual_temporal_mapping: TemporalMapping | None = None,
    audio_temporal_mapping: TemporalMapping | None = None,
) -> PackedLayout:
    """Build H3-style positions and physical ``[text|visual|audio]`` indices."""

    if int(text_length) < 0:
        raise ValueError(f"text_length must be non-negative, got {text_length}")
    text_length = int(text_length)
    patch = _validate_patch_size(patch_size)
    visual_time = visual_temporal_mapping or StrideTemporalMapping(1)
    audio_time = audio_temporal_mapping or StrideTemporalMapping(1)
    visual = _StreamParts([], [], [], [], [])
    audio = _StreamParts([], [], [], [], [])
    cursor = 0.0
    image_index = 0
    video_index = 0
    audio_index = 0
    item_types: list[int] = []
    item_image_indices: list[int] = []
    item_video_indices: list[int] = []
    item_audio_indices: list[int] = []
    item_time_ranges: list[tuple[float, float]] = []

    for reference in references:
        start = cursor
        branch_ends: list[float] = []
        current_image_index = 0
        current_video_index = 0
        current_audio_index = 0
        if reference.type == "image":
            image_index += 1
            current_image_index = image_index
            positions, end = _visual_positions(
                reference.visual,  # type: ignore[arg-type]
                start=start,
                patch_size=patch,
                temporal_mapping=visual_time,
            )
            _append_identity(
                visual,
                positions,
                modality_id=IMAGE_REFERENCE,
                reference_index=image_index,
                is_target=False,
            )
            branch_ends.append(end)
        elif reference.type in {"video", "video_audio"}:
            video_index += 1
            current_video_index = video_index
            positions, end = _visual_positions(
                reference.visual,  # type: ignore[arg-type]
                start=start,
                patch_size=patch,
                temporal_mapping=visual_time,
            )
            _append_identity(
                visual,
                positions,
                modality_id=VIDEO_REFERENCE,
                reference_index=video_index,
                is_target=False,
            )
            branch_ends.append(end)
        if reference.type in {"audio", "video_audio"}:
            audio_index += 1
            current_audio_index = audio_index
            positions, end, plane_ids = _audio_positions(
                reference.audio,  # type: ignore[arg-type]
                start=start,
                temporal_mapping=audio_time,
            )
            audio.plane_ids.append(plane_ids)
            _append_identity(
                audio,
                positions,
                modality_id=AUDIO_REFERENCE,
                reference_index=audio_index,
                is_target=False,
            )
            branch_ends.append(end)
        item_end = max(branch_ends)
        item_types.append(REFERENCE_TYPE_IDS[reference.type])
        item_image_indices.append(current_image_index)
        item_video_indices.append(current_video_index)
        item_audio_indices.append(current_audio_index)
        item_time_ranges.append((start, item_end))
        cursor = float(math.ceil(item_end) + 1)

    num_reference_video_rows = sum(int(part.shape[0]) for part in visual.positions)
    num_reference_audio_rows = sum(int(part.shape[0]) for part in audio.positions)
    target_start = cursor
    target_ends: list[float] = []
    if target.visual is not None:
        positions, end = _visual_positions(
            target.visual,
            start=target_start,
            patch_size=patch,
            temporal_mapping=visual_time,
        )
        _append_identity(
            visual,
            positions,
            modality_id=NO_REFERENCE,
            reference_index=NO_REFERENCE,
            is_target=True,
        )
        target_ends.append(end)
    if target.audio is not None:
        positions, end, plane_ids = _audio_positions(
            target.audio,
            start=target_start,
            temporal_mapping=audio_time,
        )
        audio.plane_ids.append(plane_ids)
        _append_identity(
            audio,
            positions,
            modality_id=NO_REFERENCE,
            reference_index=NO_REFERENCE,
            is_target=True,
        )
        target_ends.append(end)
    target_end = max(target_ends)

    visual_positions = _cat_or_empty(
        visual.positions,
        shape=(0, 3),
        dtype=torch.float32,
    )
    audio_positions = _cat_or_empty(
        audio.positions,
        shape=(0, 3),
        dtype=torch.float32,
    )
    text_positions = torch.zeros((text_length, 3), dtype=torch.float32)
    position_ids = torch.cat((text_positions, visual_positions, audio_positions), dim=0)

    num_video_rows = int(visual_positions.shape[0])
    num_audio_rows = int(audio_positions.shape[0])
    text_indices = torch.arange(text_length, dtype=torch.long)
    video_indices = torch.arange(text_length, text_length + num_video_rows, dtype=torch.long)
    audio_indices = torch.arange(
        text_length + num_video_rows,
        text_length + num_video_rows + num_audio_rows,
        dtype=torch.long,
    )
    token_tags = torch.cat(
        (
            torch.full((text_length,), TEXT_TAG, dtype=torch.long),
            torch.full((num_video_rows,), VIDEO_TAG, dtype=torch.long),
            torch.full((num_audio_rows,), AUDIO_TAG, dtype=torch.long),
        )
    )

    visual_modality_ids = _cat_or_empty(
        visual.modality_ids,
        shape=(0,),
        dtype=torch.long,
    )
    audio_modality_ids = _cat_or_empty(
        audio.modality_ids,
        shape=(0,),
        dtype=torch.long,
    )
    visual_index_ids = _cat_or_empty(visual.index_ids, shape=(0,), dtype=torch.long)
    audio_index_ids = _cat_or_empty(audio.index_ids, shape=(0,), dtype=torch.long)
    reference_modality_ids = torch.cat(
        (
            torch.zeros(text_length, dtype=torch.long),
            visual_modality_ids,
            audio_modality_ids,
        )
    )
    reference_index_ids = torch.cat(
        (
            torch.zeros(text_length, dtype=torch.long),
            visual_index_ids,
            audio_index_ids,
        )
    )
    audio_stream_plane_ids = _cat_or_empty(
        audio.plane_ids,
        shape=(0,),
        dtype=torch.long,
    )
    audio_plane_ids = torch.zeros(position_ids.shape[0], dtype=torch.long)
    audio_plane_ids.index_copy_(0, audio_indices, audio_stream_plane_ids)
    video_loss_mask = _cat_or_empty(
        visual.loss_masks,
        shape=(0,),
        dtype=torch.bool,
    )
    audio_loss_mask = _cat_or_empty(
        audio.loss_masks,
        shape=(0,),
        dtype=torch.bool,
    )
    sequence_loss_mask = torch.cat(
        (
            torch.zeros(text_length, dtype=torch.bool),
            video_loss_mask,
            audio_loss_mask,
        )
    )

    target_visual = target.visual
    target_audio = target.audio
    return PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        reference_modality_ids=reference_modality_ids,
        reference_index_ids=reference_index_ids,
        audio_plane_ids=audio_plane_ids,
        reference_item_types=torch.tensor(item_types, dtype=torch.long),
        reference_item_image_indices=torch.tensor(item_image_indices, dtype=torch.long),
        reference_item_video_indices=torch.tensor(item_video_indices, dtype=torch.long),
        reference_item_audio_indices=torch.tensor(item_audio_indices, dtype=torch.long),
        reference_item_time_ranges=torch.tensor(
            item_time_ranges,
            dtype=torch.float32,
        ).reshape(-1, 2),
        video_loss_mask=video_loss_mask,
        audio_loss_mask=audio_loss_mask,
        sequence_loss_mask=sequence_loss_mask,
        num_reference_video_rows=num_reference_video_rows,
        num_reference_audio_rows=num_reference_audio_rows,
        num_target_video_rows=(target_visual.row_count(patch) if target_visual is not None else 0),
        num_target_audio_rows=(target_audio.row_count if target_audio is not None else 0),
        target_start=target_start,
        target_end=target_end,
        target_visual_shape=(target_visual.shape if target_visual is not None else None),
        target_audio_shape=(target_audio.shape if target_audio is not None else None),
        num_latent_frames=(target_visual.num_frames if target_visual is not None else 0),
        latent_height=(target_visual.height if target_visual is not None else 0),
        latent_width=(target_visual.width if target_visual is not None else 0),
        num_audio_channels=(target_audio.num_channels if target_audio is not None else 0),
        num_audio_latents=(target_audio.num_frames if target_audio is not None else 0),
    )


build_layout = build_packed_layout


__all__ = [
    "AUDIO_REFERENCE",
    "AUDIO_TAG",
    "AudioGeometry",
    "IMAGE_REFERENCE",
    "NO_REFERENCE",
    "PackedLayout",
    "REFERENCE_TYPE_IDS",
    "ReferenceGeometry",
    "TEXT_TAG",
    "TargetGeometry",
    "VIDEO_REFERENCE",
    "VIDEO_TAG",
    "VisualGeometry",
    "build_layout",
    "build_packed_layout",
]
