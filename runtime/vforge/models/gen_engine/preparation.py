"""CPU preparation and a broadcastable contract for Gen Engine inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .geometry import _target_effective_fps, target_geometry_from_gen_config
from .layout import AudioGeometry, TargetGeometry, VisualGeometry
from .packing import RuntimePacker

PREPARED_SAMPLE_VERSION = 1


def prepare_inference_sample(
    sample: Mapping[str, Any],
    *,
    packer: RuntimePacker,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Pack on the batch leader's CPU before NCCL moves tensors to CUDA.

    Only mappings, tensors and Python scalars cross the loader boundary. Clean
    validation targets establish geometry but are never broadcast as conditions.
    """
    spec = sample.get("target_spec")
    geometry = (
        target_geometry_from_gen_config(spec, generation_config, packer.latent_codec)
        if isinstance(spec, Mapping)
        else None
    )
    packed = packer.pack(sample, target_geometry=geometry)
    layout = packed.pop("layout")
    if geometry is None:
        target = sample["target"]
        geometry = TargetGeometry(
            visual=(
                VisualGeometry(
                    layout.target_visual_shape, _target_effective_fps(target, stream="video")
                )
                if layout.target_visual_shape is not None
                else None
            ),
            audio=(
                AudioGeometry(
                    layout.target_audio_shape, _target_effective_fps(target, stream="audio")
                )
                if layout.target_audio_shape is not None
                else None
            ),
        )
    packed.pop("target_video_rows")
    packed.pop("target_audio_rows")
    return {
        "sample_id": str(sample["sample_id"]),
        "prepared_version": PREPARED_SAMPLE_VERSION,
        "packed": packed,
        "target_geometry": asdict(geometry),
        "target_spec": dict(spec) if isinstance(spec, Mapping) else None,
        "metadata": dict(sample.get("metadata") or {}),
        "__vforge_eval_padding__": bool(sample.get("__vforge_eval_padding__", False)),
    }


def unpack_inference_sample(sample: Mapping[str, Any]) -> tuple[dict[str, Any], TargetGeometry]:
    """Reconstruct scalar geometry; tensors may already reside on CUDA."""
    if sample.get("prepared_version") != PREPARED_SAMPLE_VERSION:
        raise ValueError("unsupported gen_engine prepared sample version")
    geometry = sample["target_geometry"]
    return dict(sample["packed"]), TargetGeometry(
        visual=VisualGeometry(**geometry["visual"]) if geometry["visual"] is not None else None,
        audio=AudioGeometry(**geometry["audio"]) if geometry["audio"] is not None else None,
    )
