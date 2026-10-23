"""Wan Taichu gridnormal output and shared-input artifacts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vforge.data.gameplay_dataset import GAMEPLAY_MEDIA_SOURCES_KEY
from vforge.data.video_io import resolve_reference_paths
from vforge.evaluation.multi_variant import MultiVariantOutputWriter
from vforge.evaluation.runner import EvalOutputWriter, sample_artifact_name
from vforge.tasks.wan_taichu.eval_task.gridnormal.variants import GridNormalVariant

GRIDNORMAL_SHARED_SOURCES_FIELD = "__wan_taichu_gridnormal_shared_sources__"

_SHARED_STREAM_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gt", ("video",)),
    ("normal", ("normal",)),
    ("normal_fg", ("normal_fg",)),
    ("normal_bg", ("normal_bg",)),
    ("grid", ("grid",)),
    ("ref", ("ref", "reference")),
)
_SHARED_STREAM_NAMES = frozenset(name for name, _fields in _SHARED_STREAM_FIELDS)
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
_CHECKPOINT_OUTPUT_RE = re.compile(r"^checkpoint-\d+(?:-[A-Za-z0-9._-]+)?$")


def gridnormal_sample_context(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Capture lightweight sources; shared media are decoded only by the output leader."""
    sources: dict[str, Any] = {}
    gameplay_sources = sample.get(GAMEPLAY_MEDIA_SOURCES_KEY)
    if isinstance(gameplay_sources, Mapping):
        sources.update(
            {
                str(stream): dict(source)
                for stream, source in gameplay_sources.items()
                if stream in _SHARED_STREAM_NAMES and isinstance(source, Mapping)
            }
        )
    for stream, fields in _SHARED_STREAM_FIELDS:
        if stream in sources:
            continue
        if stream == "ref":
            reference_paths = resolve_reference_paths(sample)
            if reference_paths:
                sources[stream] = (
                    reference_paths[0] if len(reference_paths) == 1 else list(reference_paths)
                )
            continue
        declared = [str(sample[field]) for field in fields if sample.get(field)]
        if len(set(declared)) > 1:
            raise ValueError(
                f"gridnormal sample declares conflicting sources for {stream!r}: {declared}"
            )
        if declared:
            sources[stream] = declared[0]
    return {GRIDNORMAL_SHARED_SOURCES_FIELD: sources}


class _GridNormalSharedOutput:
    """Write source media once above per-checkpoint output directories."""

    def __init__(
        self,
        *,
        fps: float,
        data_source: str,
        manifest: str,
        max_frames: int | None,
    ) -> None:
        self.fps = float(fps)
        self.data_source = str(data_source)
        self.manifest = os.path.abspath(os.path.expanduser(manifest)) if manifest else ""
        self.max_frames = int(max_frames) if max_frames is not None else None

    def is_complete(self, output_dir: str, *, sample_id: str) -> bool:
        marker = self._marker_path(output_dir, sample_id)
        try:
            with open(marker, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, TypeError, ValueError):
            return False
        if (
            payload.get("schema_version") != 2
            or payload.get("sample_id") != sample_id
            or payload.get("data_source") != self.data_source
            or payload.get("manifest") != self.manifest
        ):
            return False
        streams = payload.get("streams")
        if not isinstance(streams, Mapping):
            return False
        relative_paths = [
            str(relative_path)
            for value in streams.values()
            for relative_path in (
                value if isinstance(value, Sequence) and not isinstance(value, str) else (value,)
            )
        ]
        root = self._root(output_dir)
        return all(
            os.path.isfile(os.path.join(root, relative_path))
            and os.path.getsize(os.path.join(root, relative_path)) > 0
            for relative_path in relative_paths
        )

    def write(
        self,
        result: Mapping[str, Any],
        output_dir: str,
        *,
        sample_id: str,
    ) -> None:
        if self.is_complete(output_dir, sample_id=sample_id):
            return
        sources = result.get("shared_sources")
        if not isinstance(sources, Mapping):
            sources = {}
        metadata = result.get("metadata")
        ref_selection = metadata.get("ref_selection") if isinstance(metadata, Mapping) else None
        online_ref = (
            ref_selection
            if isinstance(ref_selection, Mapping) and ref_selection.get("source") == "target_video"
            else None
        )
        root = self._root(output_dir)
        safe_name = sample_artifact_name(sample_id, suffix="")
        written: dict[str, Any] = {}
        for stream, source in sources.items():
            if stream not in _SHARED_STREAM_NAMES:
                raise ValueError(f"unknown gridnormal shared stream {stream!r}")
            if stream == "ref":
                if online_ref is not None:
                    continue
                written[stream] = self._write_references(root, safe_name, source)
            else:
                written[stream] = self._write_video(root, stream, safe_name, source)

        if online_ref is not None and online_ref.get("indices"):
            target_video = sources.get("gt")
            if not target_video:
                raise ValueError("online reference artifact requires the target-video source")
            written["ref"] = self._write_online_references(
                root,
                safe_name,
                target_video,
                online_ref["indices"],
            )

        index = metadata.get("global_manifest_index") if isinstance(metadata, Mapping) else None
        marker_payload = {
            "schema_version": 2,
            "sample_id": sample_id,
            "global_manifest_index": index,
            "data_source": self.data_source,
            "manifest": self.manifest,
            "streams": written,
            "unavailable": [name for name, _fields in _SHARED_STREAM_FIELDS if name not in written],
            "ref_selection": ref_selection,
        }
        marker = self._marker_path(output_dir, sample_id)
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        temporary = marker + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(marker_payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, marker)

    def _write_references(
        self,
        root: str,
        safe_name: str,
        source: Any,
    ) -> str | list[str]:
        from vforge.data.video_io import read_image_tensor, read_video_tensor

        paths = (
            tuple(str(path) for path in source)
            if isinstance(source, Sequence) and not isinstance(source, str)
            else (str(source),)
        )
        relative_paths: list[str] = []
        for index, source_path in enumerate(paths):
            if Path(source_path).suffix.lower() in _IMAGE_SUFFIXES:
                image = read_image_tensor(source_path)
            else:
                frames = read_video_tensor(source_path, max_frames=1)
                if frames.shape[0] < 1:
                    raise ValueError(f"gridnormal reference video has no frames: {source_path}")
                image = frames[0]
            relative_paths.append(
                self._write_reference_image(
                    root,
                    safe_name,
                    image,
                    slot=index,
                    count=len(paths),
                )
            )
        return relative_paths[0] if len(relative_paths) == 1 else relative_paths

    def _write_online_references(
        self,
        root: str,
        safe_name: str,
        source: Any,
        indices: Sequence[Any],
    ) -> str | list[str]:
        frames = self._read_video_source(source)
        if self.max_frames is not None:
            frames = frames[: self.max_frames]
        resolved = [int(index) for index in indices]
        invalid = [index for index in resolved if index < 0 or index >= int(frames.shape[0])]
        if invalid:
            raise ValueError(
                "online reference indices exceed the exported target video: "
                f"indices={invalid} frames={int(frames.shape[0])}"
            )
        relative_paths = [
            self._write_reference_image(
                root,
                safe_name,
                frames[index],
                slot=slot,
                count=len(resolved),
            )
            for slot, index in enumerate(resolved)
        ]
        return relative_paths[0] if len(relative_paths) == 1 else relative_paths

    @staticmethod
    def _write_reference_image(
        root: str,
        safe_name: str,
        image: torch.Tensor,
        *,
        slot: int,
        count: int,
    ) -> str:
        from vforge.data.video_io import write_image_png

        filename = f"{safe_name}.png" if count == 1 else f"{safe_name}-{slot:03d}.png"
        relative = os.path.join("ref", filename)
        target = os.path.join(root, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = target + ".tmp"
        write_image_png(image, temporary)
        os.replace(temporary, target)
        return relative

    def _write_video(self, root: str, stream: str, safe_name: str, source: Any) -> str:
        from vforge.data.video_io import write_video_frames_mp4

        frames = self._read_video_source(source)
        if self.max_frames is not None:
            frames = frames[: self.max_frames]
        relative = os.path.join(stream, f"{safe_name}.mp4")
        target = os.path.join(root, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        temporary = target + ".tmp.mp4"
        write_video_frames_mp4(frames, temporary, fps=self.fps)
        os.replace(temporary, target)
        return relative

    @staticmethod
    def _read_video_source(source: Any) -> torch.Tensor:
        if isinstance(source, Mapping):
            from vforge.data.gameplay_dataset import decode_gameplay_media_source

            return decode_gameplay_media_source(source)
        from vforge.data.video_io import read_video_tensor

        return read_video_tensor(str(source))

    @staticmethod
    def _root(output_dir: str) -> str:
        normalized = os.path.normpath(output_dir)
        if _CHECKPOINT_OUTPUT_RE.fullmatch(os.path.basename(normalized)):
            return os.path.dirname(normalized)
        return normalized

    def _marker_path(self, output_dir: str, sample_id: str) -> str:
        return os.path.join(
            self._root(output_dir),
            ".gridnormal-inputs",
            sample_artifact_name(sample_id, suffix=".json"),
        )


def gridnormal_completion_artifact_name(sample_id: str) -> str:
    """Return the per-sample completion marker relative path."""
    return os.path.join(
        ".gridnormal",
        sample_artifact_name(sample_id, suffix=".complete"),
    )


def _write_variant_video(result: dict[str, Any], output_dir: str, *, fps: float) -> str:
    from vforge.data.video_io import write_video_mp4

    video = result.get("video")
    if not isinstance(video, torch.Tensor):
        raise TypeError("a gridnormal variant result must contain a video tensor")
    os.makedirs(output_dir, exist_ok=True)
    sample_id = str(result.get("sample_id") or "sample")
    path = os.path.join(
        output_dir,
        sample_artifact_name(sample_id, suffix=".mp4"),
    )
    temporary = path + ".tmp.mp4"
    write_video_mp4(video, temporary, fps=fps)
    os.replace(temporary, path)
    return path


def _write_completion(result: dict[str, Any], output_dir: str) -> str:
    sample_id = str(result.get("sample_id") or "sample")
    path = os.path.join(output_dir, gridnormal_completion_artifact_name(sample_id))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(temporary, path)
    return path


class GridNormalOutputWriter:
    """Bind Wan Taichu artifacts to framework multi-variant orchestration."""

    def __init__(
        self,
        *,
        fps: float,
        variants: Sequence[GridNormalVariant],
        data_source: str = "",
        manifest: str = "",
        max_frames: int | None = None,
    ) -> None:
        self._writer = MultiVariantOutputWriter(
            variant_names=tuple(variant.name for variant in variants),
            variant_writer=lambda _name: EvalOutputWriter(
                artifact_name=lambda sample_id: sample_artifact_name(
                    sample_id,
                    suffix=".mp4",
                ),
                write_output=lambda payload, directory: _write_variant_video(
                    payload,
                    directory,
                    fps=float(fps),
                ),
            ),
            completion_writer=EvalOutputWriter(
                artifact_name=gridnormal_completion_artifact_name,
                write_output=_write_completion,
            ),
            shared_output=_GridNormalSharedOutput(
                fps=fps,
                data_source=data_source,
                manifest=manifest,
                max_frames=max_frames,
            ),
            provenance_variant_key="gridnormal_variant",
        )

    def expected_path(self, output_dir: str, sample_id: str) -> str:
        return self._writer.expected_path(output_dir, sample_id)

    def is_complete(
        self,
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> bool:
        return self._writer.is_complete(
            output_dir,
            sample_id=sample_id,
            seed=seed,
            provenance=provenance,
        )

    def write(
        self,
        result: dict[str, Any],
        output_dir: str,
        *,
        sample_id: str,
        seed: int,
        provenance: Mapping[str, Any],
    ) -> str:
        return self._writer.write(
            result,
            output_dir,
            sample_id=sample_id,
            seed=seed,
            provenance=provenance,
        )


__all__ = [
    "GRIDNORMAL_SHARED_SOURCES_FIELD",
    "GridNormalOutputWriter",
    "gridnormal_completion_artifact_name",
    "gridnormal_sample_context",
]
