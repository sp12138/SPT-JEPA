"""Manifest readers and the datasets that transport samples to a loader.

Nothing here knows what a sample is made of or where a cache chooses to put a
bundle: the schema belongs to whoever wrote the manifest and the layout arrives
as a :data:`CachePathResolver`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from vforge.data.cache import EncodedBatchCodec, cache_path_for_sample

CachePathResolver = Callable[[str, str, int], str]
"""Locate one bundle from ``(cache_dir, sample_id, index)``.

Both coordinates are passed because the layouts disagree on which one names the
file: the native cache buckets by a hash of the sample ID, while an indexed
cache names bundles after their position.
"""

_MEDIA_FIELDS = (
    "video",
    "video_path",
    "rgb",
    "image",
    "normal",
    "normal_path",
    "normal_fg",
    "normal_bg",
    "grid",
    "ref",
    "reference",
)


def _resolve_relative_media_paths(
    records: list[dict[str, Any]],
    *,
    manifest_path: str | Path,
) -> list[dict[str, Any]]:
    base = Path(manifest_path).expanduser().resolve().parent
    resolved: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        for field in _MEDIA_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute() and "://" not in value:
                record[field] = str(base / path)
        resolved.append(record)
    return resolved


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    video: str
    prompt: str
    normal: str | None = None
    extras: dict[str, Any] | None = None


def _record_sample_id(record: dict[str, Any], *, index: int) -> str:
    explicit = record.get("sample_id")
    if explicit:
        return str(explicit)
    uuid = record.get("uuid")
    record_id = record.get("id")
    if uuid and record_id:
        return f"{uuid}:{record_id}"
    return str(record_id or uuid or f"sample-{index:06d}")


def _normalize_record(record: dict[str, Any], *, index: int) -> RawSample:
    sample_id = _record_sample_id(record, index=index)
    file_info = record.get("file")
    file_info = file_info if isinstance(file_info, dict) else {}
    video_info = file_info.get("video")
    video_info = video_info if isinstance(video_info, dict) else {}
    video = (
        record.get("video")
        or record.get("video_path")
        or record.get("rgb")
        or video_info.get("video_path")
    )
    if not video:
        raise ValueError(f"sample {sample_id!r} is missing a video path")
    prompt = record.get("prompt") or record.get("text")
    if not prompt:
        captions = file_info.get("text_caption")
        if isinstance(captions, list) and captions:
            first_caption = captions[0]
            prompt = (
                first_caption.get("caption") if isinstance(first_caption, dict) else first_caption
            )
    prompt = prompt or ""
    normal = record.get("normal") or record.get("normal_path") or video_info.get("normal_path")
    known = {
        "sample_id",
        "id",
        "video",
        "video_path",
        "rgb",
        "prompt",
        "text",
        "normal",
        "normal_path",
    }
    extras = {key: value for key, value in record.items() if key not in known}
    return RawSample(
        sample_id=sample_id,
        video=str(video),
        prompt=str(prompt),
        normal=str(normal) if normal is not None else None,
        extras=extras or None,
    )


def load_manifest_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON/JSONL records without imposing a raw-media schema."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError(f"manifest {path} must be a JSON list")
        return _resolve_relative_media_paths(
            [dict(item) for item in payload],
            manifest_path=path,
        )
    records: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(dict(json.loads(line)))
    return _resolve_relative_media_paths(records, manifest_path=path)


def load_manifest_sample_ids(path: str | Path) -> list[str]:
    """Read stable IDs from any ordered manifest, including chunk datasets."""
    sample_ids = [
        _record_sample_id(record, index=index)
        for index, record in enumerate(load_manifest_records(path))
    ]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample manifest contains duplicate sample_id values")
    return sample_ids


def manifest_file_digest(path: str | Path) -> str:
    """Hash the exact ordered manifest bytes for indexed-cache resume safety."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sample_manifest(path: str | Path) -> list[RawSample]:
    """Load a JSON list or JSONL manifest of flat raw-media samples."""
    return [
        _normalize_record(record, index=index)
        for index, record in enumerate(load_manifest_records(path))
    ]


def manifest_content_digest(samples: list[RawSample]) -> str:
    payload = [
        {
            "sample_id": sample.sample_id,
            "video": sample.video,
            "prompt": sample.prompt,
            "normal": sample.normal,
            "extras": sample.extras,
        }
        for sample in samples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RawSampleDataset(Dataset):
    """Dataset of raw video/prompt samples for online encode and prepare."""

    def __init__(self, samples: list[RawSample] | None = None) -> None:
        self._samples = list(samples or ())
        sample_ids = [sample.sample_id for sample in self._samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample manifest contains duplicate sample_id values")

    @classmethod
    def from_manifest(cls, path: str | Path) -> RawSampleDataset:
        return cls(load_sample_manifest(path))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        item: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "video": sample.video,
            "prompt": sample.prompt,
        }
        if sample.normal is not None:
            item["normal"] = sample.normal
        if sample.extras:
            item.update(sample.extras)
        return item

    @property
    def samples(self) -> list[RawSample]:
        return list(self._samples)

    def content_digest(self) -> str:
        return manifest_content_digest(self._samples)


class InferenceRequestDataset(Dataset):
    """Condition-only inference requests loaded from a JSON/JSONL manifest.

    Unlike :class:`RawSampleDataset`, this transport does not impose a media
    schema. Model eval adapters validate fields such as ``image``, ``normal``,
    and ``grid`` because those requirements depend on the selected workflow.
    """

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records: list[dict[str, Any]] = []
        for index, source in enumerate(records or ()):
            record = dict(source)
            record["sample_id"] = str(
                record.get("sample_id") or record.get("id") or f"sample-{index:06d}"
            )
            self._records.append(record)
        sample_ids = [str(record["sample_id"]) for record in self._records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("inference manifest contains duplicate sample_id values")

    @classmethod
    def from_manifest(cls, path: str | Path) -> InferenceRequestDataset:
        return cls(load_manifest_records(path))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._records[index])

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]


def _hashed_bucket_path(cache_dir: str, sample_id: str, index: int) -> str:
    """Locate a bundle in a vforge-native cache, which buckets by sample ID."""
    del index
    return cache_path_for_sample(cache_dir, sample_id)


class EncodedCacheDataset(Dataset):
    """Dataset yielding encoded bundles from a prepared cache.

    Where a bundle sits on disk is the cache's business, not this class's: the
    layout arrives as ``path_for`` so that a cache written under someone else's
    naming rule needs a resolver rather than a branch here.
    """

    def __init__(
        self,
        cache_dir: str,
        sample_ids: list[str],
        codec: EncodedBatchCodec,
        *,
        path_for: CachePathResolver | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._sample_ids = list(sample_ids)
        self._codec = codec
        self._path_for = path_for or _hashed_bucket_path

    def __len__(self) -> int:
        return len(self._sample_ids)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self._sample_ids[index]
        bundle = self._codec.verify(self._path_for(self._cache_dir, sample_id, index))
        bundle["sample_id"] = sample_id
        return bundle


__all__ = [
    "CachePathResolver",
    "EncodedCacheDataset",
    "InferenceRequestDataset",
    "RawSample",
    "RawSampleDataset",
    "load_manifest_records",
    "load_manifest_sample_ids",
    "load_sample_manifest",
    "manifest_file_digest",
    "manifest_content_digest",
]
