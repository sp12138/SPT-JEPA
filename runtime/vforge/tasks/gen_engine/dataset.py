"""Precomputed ``mmgen.sample`` v1 manifests and local/S3 tensor delivery.

This module deliberately stops at the data/model boundary. It validates the
portable manifest contract and returns typed tensors plus their timing and
reference metadata; a model-owned packer decides how those values become model
tokens.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import operator
import os
import pickle
import re
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Literal

import torch
from torch.utils.data import Dataset

from vforge.data.artifact_io import ArtifactStore
from vforge.data.cache import EncodedBatchCodec, decode_latent_bundle

MMGEN_SCHEMA = "mmgen.sample"
MMGEN_SCHEMA_VERSION = 1

REFERENCE_TYPES = ("image", "video_audio", "video", "audio")

MMGEN_MME_BUNDLE_CODEC = EncodedBatchCodec(
    name="mmgen_mme",
    required_keys=("embedding",),
    schema_version=1,
    variable_shape_keys=frozenset({"embedding"}),
)
MMGEN_VISUAL_BUNDLE_CODEC = EncodedBatchCodec(
    name="mmgen_visual",
    required_keys=("latent",),
    schema_version=1,
    variable_shape_keys=frozenset({"latent"}),
)
MMGEN_AUDIO_BUNDLE_CODEC = EncodedBatchCodec(
    name="mmgen_audio",
    required_keys=("latent",),
    schema_version=1,
    variable_shape_keys=frozenset({"latent"}),
)

# Short aliases keep call sites readable while the longer names make the bundle
# role unambiguous to users discovering this module.
MMGEN_MME_CODEC = MMGEN_MME_BUNDLE_CODEC
MMGEN_VISUAL_CODEC = MMGEN_VISUAL_BUNDLE_CODEC
MMGEN_AUDIO_CODEC = MMGEN_AUDIO_BUNDLE_CODEC
MME_BUNDLE_CODEC = MMGEN_MME_BUNDLE_CODEC
VISUAL_BUNDLE_CODEC = MMGEN_VISUAL_BUNDLE_CODEC
AUDIO_BUNDLE_CODEC = MMGEN_AUDIO_BUNDLE_CODEC

_R2V_TASK = re.compile(r"^r2v(?:-[a-z][a-z0-9_]*)?$")
_SHA256 = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_MODALITIES = ("video", "audio")
SampleKind = Literal["any", "target", "target_spec"]


def _fail(context: str, message: str) -> None:
    raise ValueError(f"{context}: {message}")


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(context, "must be an object")
    return dict(value)


def _require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(context, "must be a non-empty string")
    return value


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(context, "must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        _fail(context, "must be a finite positive number")
    return result


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(context, "must be a positive integer")
    return value


def _resolve_path(value: Any, *, base: Path, context: str) -> str:
    path_text = _require_nonempty_string(value, context)
    if "://" in path_text:
        return path_text
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _declared_shape(
    descriptor: dict[str, Any],
    *,
    primary_key: str,
    alias_key: str | None,
    rank: int,
    context: str,
) -> list[int] | None:
    value = descriptor.get(primary_key)
    if alias_key and alias_key in descriptor:
        if value is not None and descriptor[alias_key] != value:
            _fail(context, f"{primary_key} and {alias_key} disagree")
        value = descriptor[alias_key]
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != rank:
        _fail(context, f"{primary_key} must contain {rank} dimensions")
    shape = [
        _positive_integer(size, f"{context}.{primary_key}[{index}]")
        for index, size in enumerate(value)
    ]
    return shape


def _declared_dtype(
    descriptor: dict[str, Any],
    *,
    primary_key: str,
    alias_key: str | None,
    context: str,
) -> str | None:
    value = descriptor.get(primary_key)
    if alias_key and alias_key in descriptor:
        if value is not None and descriptor[alias_key] != value:
            _fail(context, f"{primary_key} and {alias_key} disagree")
        value = descriptor[alias_key]
    if value is None:
        return None
    return _require_nonempty_string(value, f"{context}.{primary_key}")


def _normalize_digest(value: Any, context: str) -> str:
    text = _require_nonempty_string(value, context)
    match = _SHA256.fullmatch(text)
    if match is None:
        _fail(context, "must be sha256:<64 hexadecimal characters>")
    return f"sha256:{match.group(1).lower()}"


def _artifact_digest(
    descriptor: dict[str, Any],
    *,
    primary_key: str,
    context: str,
) -> str | None:
    primary = descriptor.get(primary_key)
    alias = descriptor.get("digest")
    if primary is not None and alias is not None and primary != alias:
        _fail(context, f"{primary_key} and digest disagree")
    value = primary if primary is not None else alias
    return None if value is None else _normalize_digest(value, f"{context}.{primary_key}")


def _normalize_media(
    value: Any,
    *,
    modality: str,
    base: Path,
    context: str,
    require_latent: bool,
    require_digest: bool,
) -> dict[str, Any]:
    media = _require_mapping(value, context)
    source_path = media.get("source_path")
    latent_path = media.get("latent_path")
    if source_path is None and latent_path is None:
        _fail(context, "requires source_path or latent_path")
    if source_path is not None:
        media["source_path"] = _resolve_path(
            source_path,
            base=base,
            context=f"{context}.source_path",
        )
    if latent_path is not None:
        media["latent_path"] = _resolve_path(
            latent_path,
            base=base,
            context=f"{context}.latent_path",
        )
    if require_latent and latent_path is None:
        _fail(context, "latent_path is required for runtime delivery")

    expected_rank = 3 if modality == "audio" else 4
    shape = _declared_shape(
        media,
        primary_key="latent_shape",
        alias_key="shape",
        rank=expected_rank,
        context=context,
    )
    if shape is not None:
        media["latent_shape"] = shape
        media.pop("shape", None)
        if modality == "image" and shape[1] != 1:
            _fail(f"{context}.latent_shape", "image temporal dimension must be 1")
        if modality == "audio" and shape[0] != 2:
            _fail(f"{context}.latent_shape", "audio leading A dimension must be 2")
    elif latent_path is not None:
        _fail(context, "latent_shape is required with latent_path")

    dtype = _declared_dtype(
        media,
        primary_key="latent_dtype",
        alias_key="dtype",
        context=context,
    )
    if dtype is not None:
        media["latent_dtype"] = dtype
        media.pop("dtype", None)

    digest = _artifact_digest(media, primary_key="latent_digest", context=context)
    if digest is not None:
        media["latent_digest"] = digest
        media.pop("digest", None)
    if require_digest and require_latent and digest is None:
        _fail(context, "latent_digest is required by the dataset digest contract")

    if modality in {"video", "audio"}:
        if latent_path is not None:
            if "effective_fps" not in media:
                _fail(context, "effective_fps is required with a temporal latent")
            media["effective_fps"] = _positive_number(
                media["effective_fps"],
                f"{context}.effective_fps",
            )
        elif "effective_fps" in media:
            media["effective_fps"] = _positive_number(
                media["effective_fps"],
                f"{context}.effective_fps",
            )
    elif "effective_fps" in media:
        _fail(context, "image media must not declare effective_fps")
    return media


def _normalize_mme(
    value: Any,
    *,
    base: Path,
    context: str,
    require_digest: bool,
) -> dict[str, Any]:
    mme = _require_mapping(value, context)
    if "embedding_path" not in mme:
        _fail(context, "embedding_path is required for runtime delivery")
    mme["embedding_path"] = _resolve_path(
        mme["embedding_path"],
        base=base,
        context=f"{context}.embedding_path",
    )
    shape = _declared_shape(
        mme,
        primary_key="shape",
        alias_key=None,
        rank=2,
        context=context,
    )
    if shape is not None:
        mme["shape"] = shape
    dtype = _declared_dtype(
        mme,
        primary_key="dtype",
        alias_key=None,
        context=context,
    )
    if dtype is not None:
        mme["dtype"] = dtype
    if "fingerprint" in mme:
        _require_nonempty_string(mme["fingerprint"], f"{context}.fingerprint")
    digest = _artifact_digest(mme, primary_key="embedding_digest", context=context)
    if digest is not None:
        mme["embedding_digest"] = digest
        mme.pop("digest", None)
    if require_digest and digest is None:
        _fail(context, "embedding_digest is required by the dataset digest contract")
    return mme


def _normalize_target_spec(value: Any, *, context: str) -> dict[str, Any]:
    spec = _require_mapping(value, context)
    modalities = spec.get("modalities")
    if not isinstance(modalities, list) or not modalities:
        _fail(f"{context}.modalities", "must be a non-empty list")
    if any(modality not in _MODALITIES for modality in modalities):
        _fail(f"{context}.modalities", "only video and audio are supported")
    if len(set(modalities)) != len(modalities):
        _fail(f"{context}.modalities", "must not contain duplicates")
    spec["modalities"] = list(modalities)

    duration = spec.get("duration_seconds")
    if duration is not None:
        spec["duration_seconds"] = _positive_number(duration, f"{context}.duration_seconds")

    video_spec = spec.get("video")
    if video_spec is not None:
        video_spec = _require_mapping(video_spec, f"{context}.video")
        spec["video"] = video_spec
    audio_spec = spec.get("audio")
    if audio_spec is not None:
        audio_spec = _require_mapping(audio_spec, f"{context}.audio")
        spec["audio"] = audio_spec

    if "video" in modalities:
        video = video_spec if video_spec is not None else spec
        video["height"] = _positive_integer(video.get("height"), f"{context}.video.height")
        video["width"] = _positive_integer(video.get("width"), f"{context}.video.width")
        video["fps"] = _positive_number(video.get("fps"), f"{context}.video.fps")
        if video.get("num_frames") is not None:
            video["num_frames"] = _positive_integer(
                video["num_frames"],
                f"{context}.video.num_frames",
            )
        if duration is None and video.get("num_frames") is None:
            _fail(context, "video output requires duration_seconds or num_frames")

    if "audio" in modalities:
        audio = audio_spec if audio_spec is not None else spec
        audio["sample_rate"] = _positive_integer(
            audio.get("sample_rate"),
            f"{context}.audio.sample_rate",
        )
        if audio.get("num_samples") is not None:
            audio["num_samples"] = _positive_integer(
                audio["num_samples"],
                f"{context}.audio.num_samples",
            )
        if duration is None and audio.get("num_samples") is None:
            _fail(context, "audio output requires duration_seconds or num_samples")
    return spec


def _selected_target_modalities(
    target: dict[str, Any],
    requested: tuple[str, ...] | None,
    *,
    context: str,
) -> tuple[str, ...]:
    available = tuple(modality for modality in _MODALITIES if modality in target)
    if not available:
        _fail(context, "must contain video, audio, or both")
    unknown = set(target) - set(_MODALITIES)
    if unknown:
        _fail(context, f"unsupported fields: {sorted(unknown)}")
    selected = available if requested is None else requested
    missing = [modality for modality in selected if modality not in target]
    if missing:
        _fail(context, f"selected target modalities are absent: {missing}")
    return selected


def _normalize_record(
    source: dict[str, Any],
    *,
    base: Path,
    line_number: int,
    target_modalities: tuple[str, ...] | None,
    require_digests: bool,
) -> dict[str, Any]:
    record = copy.deepcopy(source)
    context = f"manifest line {line_number}"
    if record.get("schema") != MMGEN_SCHEMA:
        _fail(context, f"schema must be {MMGEN_SCHEMA!r}")
    schema_version = record.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MMGEN_SCHEMA_VERSION
    ):
        _fail(context, f"schema_version must be {MMGEN_SCHEMA_VERSION}")
    sample_id = _require_nonempty_string(record.get("sample_id"), f"{context}.sample_id")
    context = f"sample {sample_id!r}"
    task = _require_nonempty_string(record.get("task"), f"{context}.task")
    if task != "t2v" and _R2V_TASK.fullmatch(task) is None:
        _fail(f"{context}.task", "must be t2v, r2v, or r2v-<snake_case>")
    record["caption"] = _require_nonempty_string(record.get("caption"), f"{context}.caption")

    references = record.get("references")
    if not isinstance(references, list):
        _fail(f"{context}.references", "must be a list")
    if task == "t2v" and references:
        _fail(f"{context}.references", "t2v samples must not have references")
    if task != "t2v" and not references:
        _fail(f"{context}.references", "r2v samples require at least one reference")

    normalized_references: list[dict[str, Any]] = []
    ref_ids: set[str] = set()
    counters = {"image": 0, "video": 0, "audio": 0}
    required_media = {
        "image": ("image",),
        "video_audio": ("video", "audio"),
        "video": ("video",),
        "audio": ("audio",),
    }
    for index, raw_reference in enumerate(references):
        ref_context = f"{context}.references[{index}]"
        reference = _require_mapping(raw_reference, ref_context)
        ref_id = _require_nonempty_string(reference.get("ref_id"), f"{ref_context}.ref_id")
        if ref_id in ref_ids:
            _fail(f"{ref_context}.ref_id", "must be unique within the sample")
        ref_ids.add(ref_id)
        ref_type = reference.get("type")
        if ref_type not in required_media:
            _fail(f"{ref_context}.type", f"must be one of {REFERENCE_TYPES}")
        expected = set(required_media[ref_type])
        actual = {modality for modality in ("image", "video", "audio") if modality in reference}
        if actual != expected:
            _fail(
                ref_context,
                f"type {ref_type!r} requires media {sorted(expected)}, got {sorted(actual)}",
            )
        numbers: dict[str, int] = {}
        for modality in required_media[ref_type]:
            counters[modality] += 1
            numbers[modality] = counters[modality]
            reference[modality] = _normalize_media(
                reference[modality],
                modality=modality,
                base=base,
                context=f"{ref_context}.{modality}",
                require_latent=True,
                require_digest=require_digests,
            )
        reference["reference_numbers"] = numbers
        normalized_references.append(reference)
    record["references"] = normalized_references

    if "mme" not in record:
        _fail(context, "mme.embedding_path is required for runtime delivery")
    record["mme"] = _normalize_mme(
        record["mme"],
        base=base,
        context=f"{context}.mme",
        require_digest=require_digests,
    )

    has_target = record.get("target") is not None
    has_target_spec = record.get("target_spec") is not None
    if has_target == has_target_spec:
        _fail(context, "exactly one of target and target_spec is required")
    if has_target:
        target = _require_mapping(record["target"], f"{context}.target")
        selected = _selected_target_modalities(
            target,
            target_modalities,
            context=f"{context}.target",
        )
        normalized_target: dict[str, Any] = {}
        for modality in _MODALITIES:
            if modality not in target:
                continue
            normalized_target[modality] = _normalize_media(
                target[modality],
                modality=modality,
                base=base,
                context=f"{context}.target.{modality}",
                require_latent=modality in selected,
                require_digest=require_digests,
            )
        record["target"] = normalized_target
        record["_selected_target_modalities"] = selected
    else:
        target_spec = _normalize_target_spec(
            record["target_spec"],
            context=f"{context}.target_spec",
        )
        if target_modalities is not None:
            missing = [
                modality
                for modality in target_modalities
                if modality not in target_spec["modalities"]
            ]
            if missing:
                _fail(
                    f"{context}.target_spec",
                    f"selected target modalities are absent: {missing}",
                )
            target_spec["modalities"] = list(target_modalities)
        record["target_spec"] = target_spec
    if "metadata" in record and not isinstance(record["metadata"], dict):
        _fail(f"{context}.metadata", "must be an object")
    return record


def _scan_jsonl(path: Path) -> tuple[array, array, str]:
    """Build a compact random-access line index without parsing JSON records."""
    offsets = array("Q")
    line_numbers = array("Q")
    digest = hashlib.sha256()
    first_content_checked = False
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            digest.update(line)
            if not line.strip():
                continue
            stripped = line.lstrip()
            if not first_content_checked:
                first_content_checked = True
                if stripped.startswith(b"["):
                    raise ValueError(f"manifest {path} must be JSONL, not a JSON array")
            offsets.append(offset)
            line_numbers.append(line_number)
    return offsets, line_numbers, digest.hexdigest()


def _decode_jsonl_record(line: bytes, *, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = getattr(exc, "msg", str(exc))
        raise ValueError(f"manifest line {line_number}: invalid JSON: {message}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest line {line_number}: sample must be an object")
    return value


def _canonical_dtype(value: str) -> str:
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }
    name = value.removeprefix("torch.").lower()
    return aliases.get(name, name)


def dataset_input_options(extras: Mapping[str, Any]) -> dict[str, Any]:
    """Validate model-owned file delivery options shared by train and eval."""
    options = {
        "tensor_format": extras.get("tensor_format", "bundle"),
        "audio_layout": extras.get("audio_layout", "ACT"),
        "artifact_cache_dir": extras.get("artifact_cache_dir"),
        "pin_artifacts": extras.get("pin_artifacts", False),
    }
    if options["tensor_format"] not in ("bundle", "tensor", "auto"):
        raise ValueError("tensor_format must be bundle, tensor, or auto")
    if options["audio_layout"] not in ("ACT", "ATC"):
        raise ValueError("audio_layout must be ACT or ATC")
    if not isinstance(options["pin_artifacts"], bool):
        raise ValueError("pin_artifacts must be true or false")
    cache_dir = options["artifact_cache_dir"]
    if cache_dir is not None and (
        not isinstance(cache_dir, (str, Path)) or not str(cache_dir).strip()
    ):
        raise ValueError("artifact_cache_dir must be a nonempty path or null")
    return options


def _load_tensor(
    descriptor: dict[str, Any],
    *,
    path_key: str,
    digest_key: str,
    shape_key: str,
    dtype_key: str,
    codec: EncodedBatchCodec,
    field: str,
    rank: int,
    context: str,
    store: ArtifactStore,
    tensor_format: str,
    audio: bool = False,
    image: bool = False,
) -> torch.Tensor:
    path = descriptor[path_key]
    digest = descriptor.get(digest_key)
    try:
        local_path = store.resolve(path, digest)
        payload = torch.load(local_path, map_location="cpu", weights_only=True)
        if tensor_format == "tensor" or (
            tensor_format == "auto" and isinstance(payload, torch.Tensor)
        ):
            value = payload
        else:
            value = codec.canonicalize(decode_latent_bundle(payload, path=path))[field]
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError) as exc:
        label = {
            "bundle": "bundle",
            "tensor": "tensor artifact",
            "auto": "artifact",
        }[tensor_format]
        detail = f"digest mismatch: {exc}" if "artifact SHA-256 mismatch" in str(exc) else str(exc)
        raise ValueError(f"{context}: failed to load {label} {path}: {detail}") from exc
    if not isinstance(value, torch.Tensor):
        _fail(context, f"bundle field {field!r} must be a tensor")
    if value.ndim != rank:
        _fail(context, f"bundle field {field!r} must have rank {rank}, got {value.ndim}")
    if any(size <= 0 for size in value.shape):
        _fail(context, f"bundle field {field!r} dimensions must be positive")
    if audio and value.shape[0] != 2:
        _fail(context, f"audio latent leading A dimension must be 2, got {value.shape[0]}")
    if image and value.shape[1] != 1:
        _fail(context, f"image latent temporal dimension must be 1, got {value.shape[1]}")
    expected_shape = descriptor.get(shape_key)
    if expected_shape is not None and list(value.shape) != expected_shape:
        _fail(
            context,
            f"declared {shape_key}={expected_shape} does not match tensor shape "
            f"{list(value.shape)}",
        )
    expected_dtype = descriptor.get(dtype_key)
    if expected_dtype is not None:
        actual_dtype = _canonical_dtype(str(value.dtype))
        if _canonical_dtype(expected_dtype) != actual_dtype:
            _fail(
                context,
                f"declared {dtype_key}={expected_dtype!r} does not match tensor dtype "
                f"{actual_dtype!r}",
            )
    return value


def _load_media(
    descriptor: dict[str, Any],
    *,
    modality: str,
    context: str,
    store: ArtifactStore,
    tensor_format: str,
    audio_layout: str,
) -> dict[str, Any]:
    result = copy.deepcopy(descriptor)
    codec = MMGEN_AUDIO_BUNDLE_CODEC if modality == "audio" else MMGEN_VISUAL_BUNDLE_CODEC
    result["latent"] = _load_tensor(
        descriptor,
        path_key="latent_path",
        digest_key="latent_digest",
        shape_key="latent_shape",
        dtype_key="latent_dtype",
        codec=codec,
        field="latent",
        rank=3 if modality == "audio" else 4,
        context=context,
        store=store,
        tensor_format=tensor_format,
        audio=modality == "audio",
        image=modality == "image",
    )
    if modality == "audio" and audio_layout == "ATC":
        # Validate the on-disk shape first; model packing always consumes ACT.
        result["latent"] = result["latent"].transpose(1, 2).contiguous()
        result["latent_shape"] = list(result["latent"].shape)
    return result


class MMGenDataset(Dataset):
    """Lazy precomputed dataset for JSONL ``mmgen.sample`` v1 manifests.

    ``target_modalities`` selects which supervised target streams are delivered.
    References and the MME embedding are always required. Declared artifact
    digests are checked on access; ``require_digests`` additionally makes them
    mandatory. By default construction scans only compact line offsets and JSON
    validation is lazy. ``pin_artifacts=True`` also validates all consumed
    descriptors, resolves their files and freezes content identities up front.
    Pinned manifests retain 32 bytes per record for per-access line verification.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        target_modalities: Sequence[str] | None = None,
        require_digests: bool = False,
        sample_kind: SampleKind = "any",
        max_reference_index: int | None = None,
        tensor_format: str = "bundle",
        audio_layout: str = "ACT",
        artifact_cache_dir: str | Path | None = None,
        pin_artifacts: bool = False,
    ) -> None:
        options = dataset_input_options(
            dict(
                tensor_format=tensor_format,
                audio_layout=audio_layout,
                artifact_cache_dir=artifact_cache_dir,
                pin_artifacts=pin_artifacts,
            )
        )
        self._tensor_format = options["tensor_format"]
        self._audio_layout = options["audio_layout"]
        self._artifact_store = ArtifactStore(options["artifact_cache_dir"])
        self._artifact_contract: dict[str, Any] | None = None
        self._pinned_manifest_stat: tuple[int, int, int, int] | None = None
        self._pinned_line_digests: bytes | None = None
        path = Path(manifest_path).expanduser().resolve()
        if sample_kind not in {"any", "target", "target_spec"}:
            raise ValueError("sample_kind must be any, target, or target_spec")
        if max_reference_index is not None and (
            isinstance(max_reference_index, bool) or int(max_reference_index) < 0
        ):
            raise ValueError("max_reference_index must be a non-negative integer")
        if target_modalities is None:
            selected = None
        else:
            selected = tuple(target_modalities)
            if not selected:
                raise ValueError("target_modalities must not be empty")
            if len(set(selected)) != len(selected) or any(
                modality not in _MODALITIES for modality in selected
            ):
                raise ValueError("target_modalities must be unique video/audio values")

        self._manifest_path = path
        self._target_modalities = selected
        self._require_digests = bool(require_digests)
        self._sample_kind = sample_kind
        self._max_reference_index = (
            None if max_reference_index is None else int(max_reference_index)
        )
        self._offsets, self._line_numbers, self._exact_content_digest = _scan_jsonl(path)
        self._manifest_handle: BinaryIO | None = None
        self._manifest_handle_pid: int | None = None
        self._sample_ids: list[str] | None = None
        if pin_artifacts:
            self.pin_artifacts()

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        target_modalities: Sequence[str] | None = None,
        require_digests: bool = False,
        sample_kind: SampleKind = "any",
        max_reference_index: int | None = None,
        tensor_format: str = "bundle",
        audio_layout: str = "ACT",
        artifact_cache_dir: str | Path | None = None,
        pin_artifacts: bool = False,
    ) -> MMGenDataset:
        return cls(
            path,
            target_modalities=target_modalities,
            require_digests=require_digests,
            sample_kind=sample_kind,
            max_reference_index=max_reference_index,
            tensor_format=tensor_format,
            audio_layout=audio_layout,
            artifact_cache_dir=artifact_cache_dir,
            pin_artifacts=pin_artifacts,
        )

    def __len__(self) -> int:
        return len(self._offsets)

    def _resolved_index(self, index: int) -> int:
        resolved = operator.index(index)
        if resolved < 0:
            resolved += len(self)
        if not 0 <= resolved < len(self):
            raise IndexError("MMGenDataset index out of range")
        return resolved

    def _open_manifest(self) -> BinaryIO:
        process_id = os.getpid()
        if self._manifest_handle is not None and self._manifest_handle_pid != process_id:
            self._manifest_handle.close()
            self._manifest_handle = None
        if self._manifest_handle is None:
            self._manifest_handle = self._manifest_path.open("rb")
            self._manifest_handle_pid = process_id
        return self._manifest_handle

    def _read_source_line(self, index: int) -> bytes:
        if (
            self._pinned_manifest_stat is not None
            and self._manifest_stat() != self._pinned_manifest_stat
        ):
            raise ValueError("manifest changed during pinned dataset session")
        resolved = self._resolved_index(index)
        handle = self._open_manifest()
        handle.seek(self._offsets[resolved])
        line = handle.readline()
        if self._pinned_line_digests is not None:
            expected = self._pinned_line_digests[resolved * 32 : (resolved + 1) * 32]
            if hashlib.sha256(line).digest() != expected:
                raise ValueError("manifest changed during pinned dataset session")
        return line

    def _read_source_record(self, index: int) -> dict[str, Any]:
        resolved = self._resolved_index(index)
        return _decode_jsonl_record(
            self._read_source_line(resolved),
            line_number=int(self._line_numbers[resolved]),
        )

    def _normalize_source_record(
        self,
        source: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        record = _normalize_record(
            source,
            base=self._manifest_path.parent,
            line_number=int(self._line_numbers[index]),
            target_modalities=self._target_modalities,
            require_digests=self._require_digests,
        )
        if self._sample_kind != "any" and self._sample_kind not in record:
            raise ValueError(
                f"sample {record['sample_id']!r}: runtime requires {self._sample_kind}"
            )
        if self._max_reference_index is not None:
            for reference in record["references"]:
                for modality, reference_index in reference["reference_numbers"].items():
                    if reference_index > self._max_reference_index:
                        raise ValueError(
                            f"sample {record['sample_id']!r}: {modality} reference index "
                            f"{reference_index} exceeds max_reference_index="
                            f"{self._max_reference_index}"
                        )
        return record

    def _record(self, index: int) -> dict[str, Any]:
        resolved = self._resolved_index(index)
        return self._normalize_source_record(
            self._read_source_record(resolved),
            index=resolved,
        )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        state["_manifest_handle_pid"] = None
        return state

    def __del__(self) -> None:
        handle = getattr(self, "_manifest_handle", None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    @property
    def sample_ids(self) -> list[str]:
        if self._sample_ids is None:
            sample_ids = [self._record(index)["sample_id"] for index in range(len(self))]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError("mmgen manifest contains duplicate sample_id values")
            self._sample_ids = sample_ids
        return list(self._sample_ids)

    @property
    def records(self) -> list[dict[str, Any]]:
        records = [copy.deepcopy(self._record(index)) for index in range(len(self))]
        for record in records:
            record.pop("_selected_target_modalities", None)
        return records

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def content_digest(self, *, exact: bool = False) -> str:
        """Return the exact streaming manifest digest without parsing JSON."""
        del exact
        return self._exact_content_digest

    def _manifest_stat(self) -> tuple[int, int, int, int]:
        stat = self._manifest_path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def pin_artifacts(self) -> dict[str, Any]:
        """Freeze every consumed artifact before constructing a resume fingerprint.

        Source media paths are provenance, not training inputs. Only embeddings,
        references and selected target latents are resolved. No tensors are decoded.
        """
        if self._artifact_contract is not None:
            return copy.deepcopy(self._artifact_contract)
        before = self._manifest_stat()
        manifest_digest = hashlib.sha256()
        line_digests = bytearray()
        with self._manifest_path.open("rb") as handle:
            for line in handle:
                manifest_digest.update(line)
                if line.strip():
                    line_digests.extend(hashlib.sha256(line).digest())
        if manifest_digest.hexdigest() != self._exact_content_digest:
            raise ValueError("manifest changed before artifact pinning")
        # Bind line identities to the same bytes that yielded the manifest SHA.
        # Check each line on access even when filesystem timestamps do not change.
        self._pinned_line_digests = bytes(line_digests)
        paths: set[str] = set()
        for index in range(len(self)):
            record = self._record(index)
            descriptors = [(record["mme"], "embedding_path", "embedding_digest")]
            for reference in record["references"]:
                descriptors.extend(
                    (reference[modality], "latent_path", "latent_digest")
                    for modality in ("image", "video", "audio")
                    if modality in reference
                )
            if "target" in record:
                descriptors.extend(
                    (record["target"][modality], "latent_path", "latent_digest")
                    for modality in record["_selected_target_modalities"]
                )
            for descriptor, path_key, digest_key in descriptors:
                path = descriptor[path_key]
                self._artifact_store.resolve(path, descriptor.get(digest_key))
                paths.add(path)
        contract = self._artifact_store.manifest_contract(paths)
        if self._manifest_stat() != before:
            raise ValueError("manifest changed during artifact pinning")
        with self._manifest_path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != self._exact_content_digest:
                raise ValueError("manifest changed during artifact pinning")
        self._pinned_manifest_stat = before
        self._artifact_contract = contract
        return copy.deepcopy(contract)

    def input_contract(self) -> dict[str, Any] | None:
        """File semantics and optional content identity, independent of cache location."""
        if (self._tensor_format, self._audio_layout, self._artifact_contract) == (
            "bundle",
            "ACT",
            None,
        ):
            return None
        result: dict[str, Any] = {
            "schema_version": 1,
            "tensor_format": self._tensor_format,
            "audio_layout": self._audio_layout,
        }
        if self._artifact_contract is not None:
            result["artifacts"] = copy.deepcopy(self._artifact_contract)
        return result

    def data_version(self) -> str:
        """Bind shuffle/resume identity to declared format and pinned file content."""
        contract = self.input_contract()
        if contract is None:
            return self.content_digest()
        value = {"manifest_digest": self.content_digest(), "input": contract}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record(index)
        sample_id = record["sample_id"]
        mme_descriptor = record["mme"]
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "task": record["task"],
            "caption": record["caption"],
            "mme_embedding": _load_tensor(
                mme_descriptor,
                path_key="embedding_path",
                digest_key="embedding_digest",
                shape_key="shape",
                dtype_key="dtype",
                codec=MMGEN_MME_BUNDLE_CODEC,
                field="embedding",
                rank=2,
                context=f"sample {sample_id!r}.mme",
                store=self._artifact_store,
                tensor_format=self._tensor_format,
            ),
            "references": [],
        }
        for index, reference in enumerate(record["references"]):
            loaded_reference = {
                key: copy.deepcopy(value)
                for key, value in reference.items()
                if key not in {"image", "video", "audio"}
            }
            for modality in ("image", "video", "audio"):
                if modality in reference:
                    loaded_reference[modality] = _load_media(
                        reference[modality],
                        modality=modality,
                        context=f"sample {sample_id!r}.references[{index}].{modality}",
                        store=self._artifact_store,
                        tensor_format=self._tensor_format,
                        audio_layout=self._audio_layout,
                    )
            result["references"].append(loaded_reference)

        if "target" in record:
            loaded_target: dict[str, Any] = {}
            for modality in record["_selected_target_modalities"]:
                loaded_target[modality] = _load_media(
                    record["target"][modality],
                    modality=modality,
                    context=f"sample {sample_id!r}.target.{modality}",
                    store=self._artifact_store,
                    tensor_format=self._tensor_format,
                    audio_layout=self._audio_layout,
                )
            result["target"] = loaded_target
        else:
            result["target_spec"] = copy.deepcopy(record["target_spec"])
        if "metadata" in record:
            result["metadata"] = copy.deepcopy(record["metadata"])
        return result


MMGenManifestDataset = MMGenDataset
MMGenPrecomputedDataset = MMGenDataset


__all__ = [
    "AUDIO_BUNDLE_CODEC",
    "MMGEN_AUDIO_BUNDLE_CODEC",
    "MMGEN_AUDIO_CODEC",
    "MMGEN_MME_BUNDLE_CODEC",
    "MMGEN_MME_CODEC",
    "MMGEN_SCHEMA",
    "MMGEN_SCHEMA_VERSION",
    "MMGEN_VISUAL_BUNDLE_CODEC",
    "MMGEN_VISUAL_CODEC",
    "MMGenDataset",
    "MMGenManifestDataset",
    "MMGenPrecomputedDataset",
    "MME_BUNDLE_CODEC",
    "REFERENCE_TYPES",
    "SampleKind",
    "VISUAL_BUNDLE_CODEC",
    "dataset_input_options",
]
