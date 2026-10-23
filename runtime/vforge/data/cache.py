"""Encoded latent bundle IO and cache manifest helpers.

Bundle format remains compatible with the existing ``latent_io`` v1 cache
layout. Cache root manifests are vforge-native.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from typing import Any

import torch

CACHE_SCHEMA_VERSION = 1
BUNDLE_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_ScalarT = (int, float, bool, str)
_RENAME_FALLBACK_ERRNOS = {errno.ENOSYS, errno.EXDEV, errno.EPERM}


@dataclass(frozen=True)
class EncodedBatchCodec:
    """Model-owned contract for projecting and validating encoded bundles."""

    name: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...] = ()
    schema_version: int = 1
    bundle_version: int = BUNDLE_VERSION
    variable_shape_keys: frozenset[str] = frozenset()
    accept_legacy_manifest: bool = False

    @property
    def keys(self) -> tuple[str, ...]:
        return self.required_keys + self.optional_keys

    def canonicalize(self, bundle: dict[str, Any]) -> dict[str, Any]:
        missing = [key for key in self.required_keys if key not in bundle]
        if missing:
            raise ValueError(f"{self.name} encoded bundle missing keys: {missing}")
        return {key: bundle.get(key) for key in self.keys}

    def verify(self, path: str) -> dict[str, Any]:
        return self.canonicalize(load_latent_bundle(path))

    def field_specs(self, bundle: dict[str, Any]) -> dict[str, Any]:
        canonical = self.canonicalize(bundle)
        specs: dict[str, Any] = {}
        for key, value in canonical.items():
            if value is None:
                specs[key] = None
            elif isinstance(value, torch.Tensor):
                spec: dict[str, Any] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                if key in self.variable_shape_keys:
                    spec["shape"] = ["*"] * value.ndim
                specs[key] = spec
            else:
                specs[key] = {"type": type(value).__name__}
        return specs


def _scratch_root() -> str:
    root = os.environ.get("VFORGE_SCRATCH_DIR") or os.environ.get(
        "TAICHU_SCRATCH_DIR",
        "/workspace/tmp",
    )
    try:
        os.makedirs(root, exist_ok=True)
        probe = os.path.join(root, f".probe_{os.getpid()}_{uuid.uuid4().hex}")
        with open(probe, "wb"):
            pass
        os.remove(probe)
        return root
    except OSError:
        root = tempfile.gettempdir()
        os.makedirs(root, exist_ok=True)
        return root


def _encode_field(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "tensor": value.detach().cpu().contiguous()}
    if isinstance(value, _ScalarT):
        return {"kind": "scalar", "value": value}
    raise TypeError(
        f"latent bundle unsupported field type {type(value).__name__}; "
        "expected Tensor / None / int / float / bool / str"
    )


def _decode_field(entry: dict[str, Any]) -> Any:
    kind = entry["kind"]
    if kind == "none":
        return None
    if kind == "tensor":
        return entry["tensor"]
    if kind == "scalar":
        return entry["value"]
    raise ValueError(f"latent bundle unknown field kind {kind!r}")


def save_latent_bundle(bundle: dict[str, Any], path: str) -> None:
    """Atomically write a v1-compatible latent bundle to ``path``."""
    payload = {
        "version": BUNDLE_VERSION,
        "fields": {key: _encode_field(value) for key, value in bundle.items()},
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    scratch_dir = _scratch_root()
    tmp = os.path.join(
        scratch_dir,
        f"{os.path.basename(path)}.tmp.{os.getpid()}.{uuid.uuid4().hex}",
    )
    try:
        torch.save(payload, tmp)
        try:
            os.replace(tmp, path)
        except OSError as exc:
            if exc.errno not in _RENAME_FALLBACK_ERRNOS:
                raise
            shutil.copyfile(tmp, path)
            try:
                os.remove(tmp)
            except OSError:
                pass
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def decode_latent_bundle(payload: Any, *, path: str = "<memory>") -> dict[str, Any]:
    """Decode the canonical bundle envelope after a caller has loaded it."""
    if not isinstance(payload, dict) or "fields" not in payload:
        raise ValueError(f"latent bundle is not a bundle file: {path}")
    if payload.get("version") != BUNDLE_VERSION:
        raise ValueError(
            f"latent bundle {path} has version {payload.get('version')!r}, "
            f"expected {BUNDLE_VERSION}"
        )
    fields = payload["fields"]
    if not isinstance(fields, dict) or any(
        not isinstance(value, dict) for value in fields.values()
    ):
        raise ValueError(f"latent bundle {path} has invalid fields")
    return {key: _decode_field(value) for key, value in fields.items()}


def load_latent_bundle(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return decode_latent_bundle(payload, path=path)


def cache_path_for_sample(cache_dir: str, sample_id: str) -> str:
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    bucket = digest[:2]
    return os.path.join(cache_dir, bucket, f"{digest}.pt")


@dataclass(frozen=True)
class CacheManifest:
    schema_version: int
    sample_ids: list[str]
    completed: int
    encoding_fingerprint: dict[str, Any]
    bundle_version: int = BUNDLE_VERSION
    sample_ids_digest: str = ""
    bucket_rule: str = "sha1-prefix-2"
    field_specs: dict[str, Any] | None = None
    codec_name: str = ""
    codec_schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_ids": list(self.sample_ids),
            "completed": int(self.completed),
            "encoding_fingerprint": dict(self.encoding_fingerprint),
            "bundle_version": self.bundle_version,
            "sample_ids_digest": self.sample_ids_digest or digest_sample_ids(self.sample_ids),
            "bucket_rule": self.bucket_rule,
            "field_specs": dict(self.field_specs or {}),
            "codec_name": self.codec_name,
            "codec_schema_version": self.codec_schema_version,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CacheManifest:
        return cls(
            schema_version=int(payload["schema_version"]),
            sample_ids=list(payload.get("sample_ids") or ()),
            completed=int(payload.get("completed") or 0),
            encoding_fingerprint=dict(payload.get("encoding_fingerprint") or {}),
            bundle_version=int(payload.get("bundle_version") or BUNDLE_VERSION),
            sample_ids_digest=str(payload.get("sample_ids_digest") or ""),
            bucket_rule=str(payload.get("bucket_rule") or "sha1-prefix-2"),
            field_specs=dict(payload.get("field_specs") or {}),
            codec_name=str(payload.get("codec_name") or ""),
            codec_schema_version=int(payload.get("codec_schema_version") or 1),
        )


def digest_sample_ids(sample_ids: list[str]) -> str:
    encoded = json.dumps(list(sample_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_cache_manifest(cache_dir: str, manifest: CacheManifest) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, _MANIFEST_NAME)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return path


def read_cache_manifest(
    cache_dir: str,
    *,
    codec: EncodedBatchCodec,
) -> CacheManifest | None:
    path = os.path.join(cache_dir, _MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = CacheManifest.from_dict(payload)
    if manifest.schema_version != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cache schema_version={manifest.schema_version}; "
            f"expected {CACHE_SCHEMA_VERSION}"
        )
    if manifest.bundle_version != codec.bundle_version:
        raise ValueError(
            f"unsupported bundle_version={manifest.bundle_version}; expected {codec.bundle_version}"
        )
    if not manifest.codec_name:
        if not codec.accept_legacy_manifest:
            raise ValueError(
                "cache manifest has no codec identity; "
                f"{codec.name!r} does not accept legacy manifests"
            )
        manifest = replace(
            manifest,
            codec_name=codec.name,
            codec_schema_version=codec.schema_version,
        )
    if manifest.codec_name != codec.name:
        raise ValueError(f"cache codec_name={manifest.codec_name!r}, expected {codec.name!r}")
    if manifest.codec_schema_version != codec.schema_version:
        raise ValueError(
            "cache codec_schema_version="
            f"{manifest.codec_schema_version}, expected {codec.schema_version}"
        )
    if len(set(manifest.sample_ids)) != len(manifest.sample_ids):
        raise ValueError("cache manifest contains duplicate sample IDs")
    actual_digest = digest_sample_ids(manifest.sample_ids)
    if manifest.sample_ids_digest and manifest.sample_ids_digest != actual_digest:
        raise ValueError("cache manifest sample_ids_digest does not match sample_ids")
    if not 0 <= manifest.completed <= len(manifest.sample_ids):
        raise ValueError("cache manifest completed count is invalid")
    return replace(manifest, sample_ids_digest=actual_digest)


__all__ = [
    "BUNDLE_VERSION",
    "CACHE_SCHEMA_VERSION",
    "CacheManifest",
    "EncodedBatchCodec",
    "cache_path_for_sample",
    "decode_latent_bundle",
    "digest_sample_ids",
    "load_latent_bundle",
    "read_cache_manifest",
    "save_latent_bundle",
    "write_cache_manifest",
]
