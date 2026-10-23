"""Export raw manifest media without building a model or distributed runtime."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from vforge.data.dataset import load_manifest_records
from vforge.data.gameplay_dataset import GameplayDataset
from vforge.data.video_io import (
    read_image_tensor,
    read_video_tensor,
    write_image_png,
    write_video_frames_mp4,
)

_VIDEO_FIELDS = (
    ("video", ("video", "rgb", "video_path")),
    ("normal", ("normal", "normal_path")),
    ("grid", ("grid",)),
    ("normal_fg", ("normal_fg",)),
    ("normal_bg", ("normal_bg",)),
)
_REFERENCE_FIELDS = ("ref", "reference")
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_PRESENCE_FIELDS = {
    "video": ("video", "rgb", "video_path"),
    "normal": ("normal", "normal_path"),
    "grid": ("grid",),
    "normal_fg": ("normal_fg",),
    "normal_bg": ("normal_bg",),
    "ref": _REFERENCE_FIELDS,
    "image": ("image",),
    "prompt": ("prompt", "text"),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export media and prompts from a manifest or evaluation preset.",
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", help="JSON or JSONL manifest path")
    inputs.add_argument(
        "--evaluation",
        help="Hydra evaluation preset; loads its PKL paths, sizes, and mount aliases",
    )
    parser.add_argument(
        "--partial-normal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="export foreground/background normal streams when a mask exists",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-index",
        "--sample_index",
        type=_non_negative_int,
        help="export exactly one zero-based sample",
    )
    selection.add_argument(
        "--num-samples",
        "--num_samples",
        type=_positive_int,
        default=3,
        help="export the first N samples (default: 3)",
    )
    parser.add_argument("--fps", type=_positive_float, default=16.0)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="outputs/visualize",
        help="artifact directory (default: outputs/visualize)",
    )
    return parser


def _selected_indices(
    size: int,
    *,
    sample_index: int | None,
    num_samples: int | None,
) -> list[int]:
    if size == 0:
        raise ValueError("manifest contains no samples")
    if sample_index is not None:
        if sample_index < 0 or sample_index >= size:
            raise ValueError(f"sample_index must be between 0 and {size - 1}, got {sample_index}")
        if num_samples is not None:
            raise ValueError("sample_index and num_samples cannot be used together")
        return [sample_index]
    if num_samples is None or num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    return list(range(min(num_samples, size)))


def _first_present(record: dict[str, object], fields: tuple[str, ...]) -> object | None:
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return value
    return None


def _resolve_media_path(value: object, *, manifest_dir: Path, field: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"manifest field {field!r} must be a path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path


def _stage_export(
    staging_dir: Path,
    destination: Path,
    writer: Callable[[Path], None],
) -> None:
    temporary = staging_dir / destination.name
    writer(temporary)
    shutil.copyfile(temporary, destination)


def _export_video(
    source: Path,
    destination: Path,
    *,
    staging_dir: Path,
    fps: float,
) -> None:
    frames = read_video_tensor(str(source))
    _stage_export(
        staging_dir,
        destination,
        lambda path: write_video_frames_mp4(frames, str(path), fps=fps),
    )


def _export_image(source: Path, destination: Path, *, staging_dir: Path) -> None:
    image = read_image_tensor(str(source))
    _stage_export(
        staging_dir,
        destination,
        lambda path: write_image_png(image, str(path)),
    )


def _export_record(
    record: dict[str, object],
    *,
    index: int,
    manifest_dir: Path,
    output_dir: Path,
    staging_dir: Path,
    fps: float,
) -> list[Path]:
    prefix = f"sample_{index:04d}"
    exported: list[Path] = []

    for output_name, fields in _VIDEO_FIELDS:
        value = _first_present(record, fields)
        if value is None:
            continue
        source = _resolve_media_path(value, manifest_dir=manifest_dir, field=fields[0])
        destination = output_dir / f"{prefix}_{output_name}.mp4"
        _export_video(source, destination, staging_dir=staging_dir, fps=fps)
        exported.append(destination)

    reference = _first_present(record, _REFERENCE_FIELDS)
    if reference is not None:
        source = _resolve_media_path(
            reference,
            manifest_dir=manifest_dir,
            field=_REFERENCE_FIELDS[0],
        )
        if source.suffix.lower() in _IMAGE_SUFFIXES:
            destination = output_dir / f"{prefix}_ref.png"
            _export_image(source, destination, staging_dir=staging_dir)
        else:
            destination = output_dir / f"{prefix}_ref.mp4"
            _export_video(source, destination, staging_dir=staging_dir, fps=fps)
        exported.append(destination)

    image = record.get("image")
    if image is not None and image != "":
        source = _resolve_media_path(image, manifest_dir=manifest_dir, field="image")
        destination = output_dir / f"{prefix}_image.png"
        _export_image(source, destination, staging_dir=staging_dir)
        exported.append(destination)

    prompt = record.get("prompt") if "prompt" in record else record.get("text")
    if prompt is not None:
        destination = output_dir / f"{prefix}_prompt.txt"
        _stage_export(
            staging_dir,
            destination,
            lambda path: path.write_text(str(prompt), encoding="utf-8"),
        )
        exported.append(destination)

    metadata = {
        "sample_index": index,
        "sample_id": record.get("sample_id"),
        "present": {
            name: _first_present(record, fields) is not None
            for name, fields in _PRESENCE_FIELDS.items()
        },
        "record": record,
    }
    destination = output_dir / f"{prefix}_metadata.json"
    _stage_export(
        staging_dir,
        destination,
        lambda path: path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    )
    exported.append(destination)

    return exported


def _jsonable_sample(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _jsonable_sample(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_sample(item) for item in value]
    return value


def _export_decoded_record(
    record: dict[str, Any],
    *,
    index: int,
    output_dir: Path,
    staging_dir: Path,
    fps: float,
) -> list[Path]:
    prefix = f"sample_{index:04d}"
    exported: list[Path] = []
    decoded_fields = (
        ("video", "video_frames"),
        ("normal", "normal_condition"),
        ("grid", "grid_condition"),
        ("normal_fg", "normal_fg_condition"),
        ("normal_bg", "normal_bg_condition"),
    )
    present: dict[str, bool] = {}
    for output_name, field in decoded_fields:
        frames = record.get(field)
        present[output_name] = isinstance(frames, torch.Tensor) and frames.ndim == 4
        if not present[output_name]:
            continue
        destination = output_dir / f"{prefix}_{output_name}.mp4"
        _stage_export(
            staging_dir,
            destination,
            lambda path, frames=frames: write_video_frames_mp4(frames, str(path), fps=fps),
        )
        exported.append(destination)

    image = record.get("image_condition")
    present["image"] = isinstance(image, torch.Tensor) and image.numel() > 0
    if present["image"]:
        destination = output_dir / f"{prefix}_image.png"
        frame = image[0] if image.ndim == 4 else image
        _stage_export(
            staging_dir,
            destination,
            lambda path: write_image_png(frame, str(path)),
        )
        exported.append(destination)

    prompt = record.get("prompt")
    present["prompt"] = prompt is not None
    if present["prompt"]:
        destination = output_dir / f"{prefix}_prompt.txt"
        _stage_export(
            staging_dir,
            destination,
            lambda path: path.write_text(str(prompt), encoding="utf-8"),
        )
        exported.append(destination)
    present["ref"] = False

    destination = output_dir / f"{prefix}_metadata.json"
    metadata = {
        "sample_index": index,
        "sample_id": record.get("sample_id"),
        "present": present,
        "record": _jsonable_sample(record),
    }
    _stage_export(
        staging_dir,
        destination,
        lambda path: path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    )
    exported.append(destination)
    return exported


def visualize_manifest(
    manifest: str | Path,
    *,
    sample_index: int | None = None,
    num_samples: int | None = None,
    fps: float = 16.0,
    output_dir: str | Path = "outputs/visualize",
) -> list[Path]:
    """Export selected manifest records and return their artifact paths."""
    manifest_path = Path(manifest).expanduser().resolve()
    records = load_manifest_records(manifest_path)
    if sample_index is None and num_samples is None:
        num_samples = 3
    indices = _selected_indices(
        len(records),
        sample_index=sample_index,
        num_samples=num_samples,
    )
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="vforge-visualize-") as temporary:
        staging_dir = Path(temporary)
        for index in indices:
            exported.extend(
                _export_record(
                    records[index],
                    index=index,
                    manifest_dir=manifest_path.parent,
                    output_dir=destination,
                    staging_dir=staging_dir,
                    fps=fps,
                )
            )
    return exported


def visualize_gameplay_index(
    index_files: Sequence[str | Path],
    *,
    frames: int,
    height: int,
    width: int,
    source: str = "gameplay",
    max_sequence_length: int = 512,
    camera_source_height: int = 720,
    camera_source_width: int = 1280,
    path_maps: Sequence[tuple[str, str]] = (),
    partial_normal: bool = True,
    sample_index: int | None = None,
    num_samples: int | None = None,
    fps: float = 16.0,
    output_dir: str | Path = "outputs/visualize",
) -> list[Path]:
    """Decode selected Taichu PKL records and write only final visualization artifacts."""
    dataset = GameplayDataset(
        index_files=[str(Path(path).expanduser()) for path in index_files],
        frames=int(frames),
        height=int(height),
        width=int(width),
        max_sequence_length=int(max_sequence_length),
        name=str(source),
        mode="val",
        extras={
            "use_partial_normal_aug": bool(partial_normal),
            "camera_source_height": int(camera_source_height),
            "camera_source_width": int(camera_source_width),
            "path_replacements": list(path_maps),
        },
    )
    if sample_index is None and num_samples is None:
        num_samples = 3
    indices = _selected_indices(
        len(dataset),
        sample_index=sample_index,
        num_samples=num_samples,
    )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="vforge-visualize-") as temporary:
        staging_dir = Path(temporary)
        for index in indices:
            exported.extend(
                _export_decoded_record(
                    dataset[index],
                    index=index,
                    output_dir=destination,
                    staging_dir=staging_dir,
                    fps=fps,
                )
            )
    return exported


def visualize_evaluation_preset(
    preset: str,
    *,
    partial_normal: bool | None = None,
    sample_index: int | None = None,
    num_samples: int | None = None,
    fps: float = 16.0,
    output_dir: str | Path = "outputs/visualize",
) -> list[Path]:
    """Visualize raw sources from one evaluation preset without model startup."""
    from vforge.tasks.common import compose_standalone_config

    cfg = compose_standalone_config("eval", [f"evaluation={preset}"])
    plain = OmegaConf.to_container(cfg.evaluation.sources, resolve=True)
    if not isinstance(plain, list) or not plain:
        raise ValueError(f"evaluation preset {preset!r} does not define sources")
    path_replacements = OmegaConf.to_container(
        cfg.evaluation.path_replacements,
        resolve=True,
    )

    datasets: list[GameplayDataset] = []
    for source in plain:
        if not isinstance(source, dict) or source.get("kind") != "gameplay":
            raise ValueError(f"evaluation preset {preset!r} contains a non-gameplay source")
        extras = dict(source.get("extras") or {})
        if path_replacements:
            extras["path_replacements"] = path_replacements
        if partial_normal is not None:
            extras["use_partial_normal_aug"] = bool(partial_normal)
        datasets.append(
            GameplayDataset(
                index_files=[str(path) for path in source.get("index_files") or ()],
                frames=int(source["frames"]),
                height=int(source["height"]),
                width=int(source["width"]),
                max_sequence_length=int(source.get("max_sequence_length", 512)),
                name=str(source.get("name") or "gameplay"),
                mode="val",
                extras=extras,
            )
        )

    total = sum(len(dataset) for dataset in datasets)
    if sample_index is None and num_samples is None:
        num_samples = 3
    indices = _selected_indices(
        total,
        sample_index=sample_index,
        num_samples=num_samples,
    )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="vforge-visualize-") as temporary:
        staging_dir = Path(temporary)
        for index in indices:
            offset = 0
            for dataset in datasets:
                if index < offset + len(dataset):
                    record = dataset[index - offset]
                    break
                offset += len(dataset)
            else:  # pragma: no cover - protected by _selected_indices
                raise IndexError(index)
            exported.extend(
                _export_decoded_record(
                    record,
                    index=index,
                    output_dir=destination,
                    staging_dir=staging_dir,
                    fps=fps,
                )
            )
    return exported


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    num_samples = None if args.sample_index is not None else args.num_samples
    if args.manifest:
        exported = visualize_manifest(
            args.manifest,
            sample_index=args.sample_index,
            num_samples=num_samples,
            fps=args.fps,
            output_dir=args.output_dir,
        )
    else:
        exported = visualize_evaluation_preset(
            args.evaluation,
            partial_normal=args.partial_normal,
            sample_index=args.sample_index,
            num_samples=num_samples,
            fps=args.fps,
            output_dir=args.output_dir,
        )
    print(f"exported {len(exported)} artifacts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
