"""Runtime tensor conversion and packing for multimodal generation samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .codec import LatentCodecConfig
from .layout import (
    AudioGeometry,
    ReferenceGeometry,
    TargetGeometry,
    VisualGeometry,
    build_packed_layout,
)

PatchSize = tuple[int, int, int]


def _patch_size_tuple(patch_size: Sequence[int]) -> PatchSize:
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must have three entries, got {tuple(patch_size)}")
    result = tuple(int(value) for value in patch_size)
    if any(value <= 0 for value in result):
        raise ValueError(f"patch_size entries must be positive, got {result}")
    return result  # type: ignore[return-value]


def _cpu_tensor(value: Any, *, label: str, rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{label} must be on CPU, got device={value.device}")
    if value.ndim != rank:
        raise ValueError(f"{label} must have rank {rank}, got shape {tuple(value.shape)}")
    return value


def patchify_visual(
    latent: torch.Tensor,
    patch_size: Sequence[int] = (1, 2, 2),
) -> torch.Tensor:
    """Convert one ``[C,T,H,W]`` visual latent to frame-major patch rows."""

    latent = _cpu_tensor(latent, label="visual latent", rank=4)
    patch_t, patch_h, patch_w = _patch_size_tuple(patch_size)
    channels, frames, height, width = latent.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"visual latent shape {tuple(latent.shape)} is not divisible by patch "
            f"{(patch_t, patch_h, patch_w)}"
        )
    return (
        latent.reshape(
            channels,
            frames // patch_t,
            patch_t,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        .permute(1, 3, 5, 0, 2, 4, 6)
        .reshape(-1, channels * patch_t * patch_h * patch_w)
        .contiguous()
    )


def patchify_audio(latent: torch.Tensor) -> torch.Tensor:
    """Convert one ``[A,C,T]`` audio latent to A-major, time-major rows."""

    latent = _cpu_tensor(latent, label="audio latent", rank=3)
    audio_channels, row_width, frames = latent.shape
    return latent.permute(0, 2, 1).reshape(audio_channels * frames, row_width).contiguous()


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _media_mapping(container: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = container.get(name)
    return value if isinstance(value, Mapping) else None


def _latent_from(
    container: Mapping[str, Any],
    *,
    modality: str,
    nested_name: str,
) -> torch.Tensor | None:
    direct_names = (
        ("visual_latent", "video_latent", "image_latent", "visual")
        if modality == "visual"
        else ("audio_latent", "audio")
    )
    direct = _first_present(container, direct_names)
    if isinstance(direct, torch.Tensor):
        return direct
    if isinstance(direct, Mapping):
        nested_latent = _first_present(direct, ("latent", "latents", f"{modality}_latent"))
        if isinstance(nested_latent, torch.Tensor):
            return nested_latent
    nested = _media_mapping(container, nested_name)
    if nested is not None:
        nested_latent = _first_present(
            nested,
            ("latent", "latents", f"{nested_name}_latent", f"{modality}_latent"),
        )
        if isinstance(nested_latent, torch.Tensor):
            return nested_latent
    return None


def _effective_fps_from(
    container: Mapping[str, Any],
    *,
    modality: str,
    nested_name: str,
) -> float | None:
    direct_names = (
        ("visual_effective_fps", "video_effective_fps")
        if modality == "visual"
        else ("audio_effective_fps",)
    )
    value = _first_present(container, direct_names)
    if value is not None:
        return float(value)
    direct = container.get("visual" if modality == "visual" else "audio")
    if isinstance(direct, Mapping) and direct.get("effective_fps") is not None:
        return float(direct["effective_fps"])
    nested = _media_mapping(container, nested_name)
    if nested is not None and nested.get("effective_fps") is not None:
        return float(nested["effective_fps"])
    return None


def _shape_from(
    container: Mapping[str, Any],
    *,
    modality: str,
    nested_name: str,
) -> tuple[int, ...] | None:
    direct_names = (
        ("visual_latent_shape", "video_latent_shape", "visual_shape")
        if modality == "visual"
        else ("audio_latent_shape", "audio_shape")
    )
    value = _first_present(container, direct_names)
    if value is not None:
        return tuple(int(item) for item in value)
    direct = container.get("visual" if modality == "visual" else "audio")
    if isinstance(direct, Mapping):
        value = _first_present(direct, ("latent_shape", "shape"))
        if value is not None:
            return tuple(int(item) for item in value)
    nested = _media_mapping(container, nested_name)
    if nested is not None:
        value = _first_present(nested, ("latent_shape", "shape"))
        if value is not None:
            return tuple(int(item) for item in value)
    return None


def _reference_geometry(
    reference: Mapping[str, Any],
    *,
    latent_codec: LatentCodecConfig | None,
) -> tuple[ReferenceGeometry, torch.Tensor | None, torch.Tensor | None]:
    kind = str(reference["type"])
    visual_latent = None
    audio_latent = None
    visual = None
    audio = None
    if kind in {"image", "video", "video_audio"}:
        nested_name = "image" if kind == "image" else "video"
        visual_latent = _latent_from(
            reference,
            modality="visual",
            nested_name=nested_name,
        )
        if visual_latent is None:
            raise ValueError(f"{kind} reference is missing its visual latent")
        visual_latent = _cpu_tensor(
            visual_latent,
            label=f"{kind} reference visual latent",
            rank=4,
        )
        if latent_codec is not None:
            visual_latent = latent_codec.visual.to_model_space(visual_latent)
        visual = VisualGeometry(
            tuple(visual_latent.shape),
            _effective_fps_from(
                reference,
                modality="visual",
                nested_name=nested_name,
            ),
            is_image=kind == "image",
        )
    if kind in {"audio", "video_audio"}:
        audio_latent = _latent_from(reference, modality="audio", nested_name="audio")
        if audio_latent is None:
            raise ValueError(f"{kind} reference is missing its audio latent")
        audio_latent = _cpu_tensor(
            audio_latent,
            label=f"{kind} reference audio latent",
            rank=3,
        )
        if latent_codec is not None:
            audio_latent = latent_codec.audio.to_model_space(audio_latent)
        audio = AudioGeometry(
            tuple(audio_latent.shape),
            _effective_fps_from(
                reference,
                modality="audio",
                nested_name="audio",
            ),
        )
    return ReferenceGeometry(kind, visual=visual, audio=audio), visual_latent, audio_latent


def _geometry_from_mapping(value: Mapping[str, Any]) -> TargetGeometry:
    visual_value = value.get("visual")
    audio_value = value.get("audio")
    if isinstance(visual_value, VisualGeometry):
        visual = visual_value
    else:
        visual_shape = _shape_from(value, modality="visual", nested_name="video")
        visual = (
            VisualGeometry(
                visual_shape,  # type: ignore[arg-type]
                _effective_fps_from(value, modality="visual", nested_name="video"),
            )
            if visual_shape is not None
            else None
        )
    if isinstance(audio_value, AudioGeometry):
        audio = audio_value
    else:
        audio_shape = _shape_from(value, modality="audio", nested_name="audio")
        audio = (
            AudioGeometry(
                audio_shape,  # type: ignore[arg-type]
                _effective_fps_from(value, modality="audio", nested_name="audio"),
            )
            if audio_shape is not None
            else None
        )
    return TargetGeometry(visual=visual, audio=audio)


def target_geometry_from_spec(target_spec: Mapping[str, Any]) -> TargetGeometry:
    """Read normalized latent geometry from an inference ``target_spec``.

    Raw pixel geometry is intentionally not guessed: VAE temporal and spatial
    compression are model-specific. Callers may pass an explicit
    :class:`TargetGeometry` to :func:`pack_runtime_sample` instead.
    """

    try:
        return _geometry_from_mapping(target_spec)
    except (TypeError, ValueError) as exc:
        if not any(
            key in target_spec
            for key in (
                "visual_latent_shape",
                "video_latent_shape",
                "visual_shape",
                "audio_latent_shape",
                "audio_shape",
                "visual",
                "video",
                "audio",
            )
        ):
            raise ValueError(
                "target_spec has raw output geometry only; provide target_geometry "
                "after applying the model's VAE compression rules"
            ) from exc
        raise


def _target_payload(
    sample: Mapping[str, Any],
    explicit_geometry: TargetGeometry | Mapping[str, Any] | None,
    *,
    latent_codec: LatentCodecConfig | None,
) -> tuple[TargetGeometry, torch.Tensor | None, torch.Tensor | None, bool]:
    target_value = sample.get("target")
    target = target_value if isinstance(target_value, Mapping) else None
    target_spec_value = sample.get("target_spec")
    target_spec = target_spec_value if isinstance(target_spec_value, Mapping) else None
    if target is not None and target_spec is not None:
        raise ValueError("sample cannot contain both target and target_spec")
    if target is None and target_spec is None:
        raise ValueError("sample must contain target or target_spec")

    visual_latent = (
        _latent_from(target, modality="visual", nested_name="video") if target is not None else None
    )
    audio_latent = (
        _latent_from(target, modality="audio", nested_name="audio") if target is not None else None
    )
    if visual_latent is not None:
        visual_latent = _cpu_tensor(visual_latent, label="target visual latent", rank=4)
        if latent_codec is not None:
            visual_latent = latent_codec.visual.to_model_space(visual_latent)
    if audio_latent is not None:
        audio_latent = _cpu_tensor(audio_latent, label="target audio latent", rank=3)
        if latent_codec is not None:
            audio_latent = latent_codec.audio.to_model_space(audio_latent)

    if isinstance(explicit_geometry, TargetGeometry):
        geometry = explicit_geometry
    elif isinstance(explicit_geometry, Mapping):
        geometry = _geometry_from_mapping(explicit_geometry)
    elif target is not None:
        visual = (
            VisualGeometry(
                tuple(visual_latent.shape),
                _effective_fps_from(target, modality="visual", nested_name="video"),
            )
            if visual_latent is not None
            else None
        )
        audio = (
            AudioGeometry(
                tuple(audio_latent.shape),
                _effective_fps_from(target, modality="audio", nested_name="audio"),
            )
            if audio_latent is not None
            else None
        )
        geometry = TargetGeometry(visual=visual, audio=audio)
    else:
        geometry = target_geometry_from_spec(target_spec)  # type: ignore[arg-type]

    if visual_latent is not None:
        if geometry.visual is None:
            raise ValueError("target has a visual latent but target_geometry omits visual")
        if tuple(visual_latent.shape) != geometry.visual.shape:
            raise ValueError(
                f"target visual shape {tuple(visual_latent.shape)} disagrees with "
                f"geometry {geometry.visual.shape}"
            )
    if audio_latent is not None:
        if geometry.audio is None:
            raise ValueError("target has an audio latent but target_geometry omits audio")
        if tuple(audio_latent.shape) != geometry.audio.shape:
            raise ValueError(
                f"target audio shape {tuple(audio_latent.shape)} disagrees with "
                f"geometry {geometry.audio.shape}"
            )
    if target is not None:
        if geometry.visual is not None and visual_latent is None:
            raise ValueError(
                "training target geometry includes visual but its clean latent is absent"
            )
        if geometry.audio is not None and audio_latent is None:
            raise ValueError(
                "training target geometry includes audio but its clean latent is absent"
            )
    return geometry, visual_latent, audio_latent, target_spec is not None


def _stream_width(
    values: Sequence[int],
    *,
    label: str,
    expected: int | None,
) -> int:
    unique = set(values)
    if len(unique) > 1:
        raise ValueError(
            f"{label} row widths must match across references and target: {sorted(unique)}"
        )
    actual = next(iter(unique), None)
    if expected is not None:
        expected = int(expected)
        if expected <= 0:
            raise ValueError(f"{label}_row_width must be positive, got {expected}")
        if actual is not None and actual != expected:
            raise ValueError(f"{label} row width {actual} does not match model width {expected}")
        return expected
    return 0 if actual is None else actual


def _stream_dtype(values: Sequence[torch.Tensor]) -> torch.dtype:
    if not values:
        return torch.float32
    dtype = values[0].dtype
    if any(value.dtype != dtype for value in values[1:]):
        raise ValueError("all latent tensors in one stream must have the same dtype")
    return dtype


def _cat_rows(
    values: Sequence[torch.Tensor],
    *,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if values:
        return torch.cat(tuple(values), dim=0)
    return torch.empty((0, width), dtype=dtype)


def pack_runtime_sample(
    sample: Mapping[str, Any],
    *,
    patch_size: Sequence[int] = (1, 2, 2),
    target_geometry: TargetGeometry | Mapping[str, Any] | None = None,
    text_width: int | None = None,
    visual_row_width: int | None = None,
    audio_row_width: int | None = None,
    latent_codec: LatentCodecConfig | None = None,
) -> dict[str, Any]:
    """Convert VAE encoder outputs to model space and pack one runtime sample."""

    patch = _patch_size_tuple(patch_size)
    text = _cpu_tensor(sample.get("mme_embedding"), label="mme_embedding", rank=2)
    if text_width is not None and text.shape[1] != int(text_width):
        raise ValueError(
            f"mme_embedding width {text.shape[1]} does not match model text width {int(text_width)}"
        )
    reference_values = sample.get("references", ())
    if not isinstance(reference_values, Sequence) or isinstance(reference_values, (str, bytes)):
        raise TypeError("references must be a sequence of mappings")

    reference_geometries: list[ReferenceGeometry] = []
    reference_visual_latents: list[torch.Tensor] = []
    reference_audio_latents: list[torch.Tensor] = []
    for reference in reference_values:
        if not isinstance(reference, Mapping):
            raise TypeError("each reference must be a mapping")
        geometry, visual_latent, audio_latent = _reference_geometry(
            reference,
            latent_codec=latent_codec,
        )
        reference_geometries.append(geometry)
        if visual_latent is not None:
            reference_visual_latents.append(visual_latent)
        if audio_latent is not None:
            reference_audio_latents.append(audio_latent)

    target, target_visual_latent, target_audio_latent, is_inference = _target_payload(
        sample,
        target_geometry,
        latent_codec=latent_codec,
    )
    layout = build_packed_layout(
        int(text.shape[0]),
        reference_geometries,
        target,
        patch_size=patch,
        visual_temporal_mapping=(
            latent_codec.visual.temporal if latent_codec is not None else None
        ),
        audio_temporal_mapping=(latent_codec.audio.temporal if latent_codec is not None else None),
    )

    all_visual_geometries = [
        geometry.visual for geometry in reference_geometries if geometry.visual is not None
    ]
    if target.visual is not None:
        all_visual_geometries.append(target.visual)
    all_audio_geometries = [
        geometry.audio for geometry in reference_geometries if geometry.audio is not None
    ]
    if target.audio is not None:
        all_audio_geometries.append(target.audio)
    if latent_codec is not None:
        for geometry in all_visual_geometries:
            if geometry.channels != latent_codec.visual.latent_channels:
                raise ValueError(
                    "visual latent channel count does not match latent codec: "
                    f"{geometry.channels} != {latent_codec.visual.latent_channels}"
                )
        for geometry in all_audio_geometries:
            if geometry.row_width != latent_codec.audio.latent_channels:
                raise ValueError(
                    "audio latent channel count does not match latent codec: "
                    f"{geometry.row_width} != {latent_codec.audio.latent_channels}"
                )
            if geometry.num_channels != latent_codec.audio.planes:
                raise ValueError(
                    "audio plane count does not match latent codec: "
                    f"{geometry.num_channels} != {latent_codec.audio.planes}"
                )
    visual_width = _stream_width(
        [geometry.row_width(patch) for geometry in all_visual_geometries],
        label="visual",
        expected=visual_row_width,
    )
    audio_width = _stream_width(
        [geometry.row_width for geometry in all_audio_geometries],
        label="audio",
        expected=audio_row_width,
    )

    model_reference_visual = reference_visual_latents
    model_reference_audio = reference_audio_latents
    model_target_visual = target_visual_latent
    model_target_audio = target_audio_latent

    reference_video_parts = [patchify_visual(latent, patch) for latent in model_reference_visual]
    reference_audio_parts = [patchify_audio(latent) for latent in model_reference_audio]
    target_video_parts = (
        [patchify_visual(model_target_visual, patch)] if model_target_visual is not None else []
    )
    target_audio_parts = (
        [patchify_audio(model_target_audio)] if model_target_audio is not None else []
    )
    visual_tensors = model_reference_visual + (
        [model_target_visual] if model_target_visual is not None else []
    )
    audio_tensors = model_reference_audio + (
        [model_target_audio] if model_target_audio is not None else []
    )
    visual_dtype = _stream_dtype(visual_tensors)
    audio_dtype = _stream_dtype(audio_tensors)
    reference_video_rows = _cat_rows(
        reference_video_parts,
        width=visual_width,
        dtype=visual_dtype,
    )
    reference_audio_rows = _cat_rows(
        reference_audio_parts,
        width=audio_width,
        dtype=audio_dtype,
    )
    target_video_rows = _cat_rows(
        target_video_parts,
        width=visual_width,
        dtype=visual_dtype,
    )
    target_audio_rows = _cat_rows(
        target_audio_parts,
        width=audio_width,
        dtype=audio_dtype,
    )
    if not is_inference:
        if target_video_rows.shape[0] != layout.num_target_video_rows:
            raise ValueError("clean target visual rows do not match target layout geometry")
        if target_audio_rows.shape[0] != layout.num_target_audio_rows:
            raise ValueError("clean target audio rows do not match target layout geometry")

    packed = layout.as_dict()
    packed.update(
        {
            "layout": layout,
            "mme_embedding": text,
            "reference_video_rows": reference_video_rows,
            "target_video_rows": target_video_rows,
            "reference_audio_rows": reference_audio_rows,
            "target_audio_rows": target_audio_rows,
            "is_inference": is_inference,
        }
    )
    return packed


@dataclass(frozen=True)
class RuntimePacker:
    """Reusable model-specific patch configuration for runtime packing."""

    patch_size: PatchSize = (1, 2, 2)
    text_width: int | None = None
    visual_row_width: int | None = None
    audio_row_width: int | None = None
    latent_codec: LatentCodecConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_size", _patch_size_tuple(self.patch_size))
        for field in ("text_width", "visual_row_width", "audio_row_width"):
            value = getattr(self, field)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{field} must be positive, got {value}")
            if value is not None:
                object.__setattr__(self, field, int(value))

    def pack(
        self,
        sample: Mapping[str, Any],
        *,
        target_geometry: TargetGeometry | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return pack_runtime_sample(
            sample,
            patch_size=self.patch_size,
            target_geometry=target_geometry,
            text_width=self.text_width,
            visual_row_width=self.visual_row_width,
            audio_row_width=self.audio_row_width,
            latent_codec=self.latent_codec,
        )

    __call__ = pack


pack_sample = pack_runtime_sample
patchify_video = patchify_visual


__all__ = [
    "RuntimePacker",
    "pack_runtime_sample",
    "pack_sample",
    "patchify_audio",
    "patchify_video",
    "patchify_visual",
    "target_geometry_from_spec",
]
