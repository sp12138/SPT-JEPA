"""Freeze local or S3 artifact bytes for deterministic dataset consumption.

An instance is a read session: the first resolution freezes each object's
identity, and subsequent reads verify the local bytes without contacting S3.
A new instance checks S3 metadata again, so URI reuse cannot silently select an
old cache entry. Dataset resume contracts should include ``manifest_contract``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from vforge.utils import s3_staging

_SHA256 = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    match = _SHA256.fullmatch(value)
    if match is None:
        raise ValueError("artifact digest must be sha256:<64 hexadecimal characters>")
    return "sha256:" + match.group(1).lower()


def _head_identity(client: Any, source: s3_staging.S3Source) -> dict[str, Any]:
    head = client.head_object(Bucket=source.bucket, Key=source.key_prefix)
    if not head.get("ETag") and not head.get("VersionId"):
        raise ValueError("S3 artifact HEAD must provide ETag or VersionId")
    identity: dict[str, Any] = {"size": int(head["ContentLength"])}
    for field in ("ETag", "VersionId", "ChecksumSHA256", "ChecksumCRC32C"):
        if head.get(field) is not None:
            identity[field] = str(head[field])
    if head.get("LastModified") is not None:
        identity["LastModified"] = str(head["LastModified"])
    return identity


class _VersionedClient:
    """Pin the existing transfer helper to the object version selected by HEAD."""

    def __init__(self, client: Any, version_id: str) -> None:
        self.client = client
        self.version_id = version_id

    def download_file(self, *args: Any, **kwargs: Any) -> None:
        kwargs["ExtraArgs"] = {"VersionId": self.version_id}
        self.client.download_file(*args, **kwargs)


@dataclass(frozen=True)
class _Artifact:
    path: Path
    sha256: str
    size: int
    remote: dict[str, Any] | None = None

    def identity(self, uri: str) -> dict[str, Any]:
        result: dict[str, Any] = {"uri": uri, "sha256": self.sha256, "size": self.size}
        if self.remote is not None:
            result["remote"] = dict(self.remote)
        return result


class ArtifactStore:
    """Resolve immutable dataset inputs with a process-safe node-local S3 cache.

    Local files are never copied. S3 downloads reuse vforge's transfer helper and
    the standard AWS credential chain. Versioned objects are fetched by version;
    unversioned objects must retain their HEAD identity throughout the download.
    An externally supplied SHA-256, when available, additionally pins the bytes.

    This class owns files only; tensor formats and layouts belong to the Dataset.
    Instances contain no live client or lock and can be copied into loader workers.
    """

    def __init__(self, cache_dir: str | os.PathLike[str] | None = None) -> None:
        cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else cache_base / "vforge/artifacts"
        )
        self._artifacts: dict[str, _Artifact] = {}

    def resolve(
        self,
        path: str | os.PathLike[str],
        expected_digest: str | None = None,
    ) -> Path:
        """Return a verified local path, freezing its identity on the first call."""
        uri = os.fspath(path)
        expected = _expected_digest(expected_digest)
        if uri in self._artifacts:
            artifact = self._artifacts[uri]
            actual = _digest(artifact.path)
            if actual != artifact.sha256 or artifact.path.stat().st_size != artifact.size:
                raise ValueError(f"artifact changed during dataset session: {uri}")
        elif uri.startswith("s3://"):
            artifact = self._resolve_s3(uri, expected)
        else:
            if "://" in uri:
                raise ValueError("artifact paths must be local files or s3:// URIs")
            local = Path(uri).expanduser().resolve(strict=True)
            artifact = _Artifact(local, _digest(local), local.stat().st_size)
        if expected is not None and artifact.sha256 != expected:
            raise ValueError(f"artifact SHA-256 mismatch: {uri}")
        self._artifacts[uri] = artifact
        return artifact.path

    def identity(
        self,
        path: str | os.PathLike[str],
        expected_digest: str | None = None,
    ) -> dict[str, Any]:
        """Return portable identity metadata; the machine's cache path is omitted."""
        self.resolve(path, expected_digest)
        return self._artifacts[os.fspath(path)].identity(os.fspath(path))

    def manifest_contract(self, paths: Iterable[str | os.PathLike[str]]) -> dict[str, Any]:
        """Fingerprint actual bytes and remote versions for a Dataset resume contract."""
        identities = [self.identity(path) for path in sorted({os.fspath(path) for path in paths})]
        return {
            "schema_version": 1,
            "artifact_count": len(identities),
            "sha256": "sha256:" + hashlib.sha256(_json_bytes(identities)).hexdigest(),
        }

    def _resolve_s3(self, uri: str, expected: str | None) -> _Artifact:
        parsed = urlsplit(uri)
        if not parsed.netloc or not parsed.path.lstrip("/") or parsed.query or parsed.fragment:
            raise ValueError("S3 artifact URI must be s3://bucket/key without query or fragment")
        if parsed.username is not None or parsed.port is not None:
            raise ValueError("S3 artifact URI must not contain credentials or a port")
        source = s3_staging.S3Source(
            bucket=parsed.netloc,
            key_prefix=unquote(parsed.path[1:]),
            region=os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1",
        )
        client = s3_staging._s3_client(source)
        remote = _head_identity(client, source)
        cache_identity = {"uri": uri, "remote": remote}
        cache_key = hashlib.sha256(_json_bytes(cache_identity)).hexdigest()
        entry = self.cache_dir / cache_key[:2] / cache_key
        entry.mkdir(parents=True, exist_ok=True)
        target, metadata = entry / "artifact.pt", entry / "identity.json"
        with (entry / "download.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if target.is_file() and metadata.is_file():
                    saved = json.loads(metadata.read_text(encoding="utf-8"))
                    actual = _digest(target)
                    if (
                        saved.get("uri") == uri
                        and saved.get("remote") == remote
                        and saved.get("sha256") == actual
                        and target.stat().st_size == remote["size"]
                        and (expected is None or actual == expected)
                    ):
                        return _Artifact(target, actual, remote["size"], remote)
            except (OSError, ValueError, TypeError, AttributeError):
                # Incomplete/corrupt cache entries are replaced under the same lock.
                pass
            version = remote.get("VersionId")
            download_client = (
                _VersionedClient(client, version) if version not in (None, "null", "") else client
            )
            with tempfile.TemporaryDirectory(prefix="download-", dir=entry) as temporary:
                staged = Path(temporary) / "artifact.pt"
                s3_staging._download_object(
                    staged,
                    s3=source,
                    key=source.key_prefix,
                    expected_size=remote["size"],
                    concurrency=4,
                    client=download_client,
                )
                if download_client is client and _head_identity(client, source) != remote:
                    raise ValueError(f"S3 artifact changed during download: {uri}")
                actual = _digest(staged)
                if expected is not None and actual != expected:
                    raise ValueError(f"artifact SHA-256 mismatch: {uri}")
                artifact = _Artifact(target, actual, staged.stat().st_size, remote)
                staged_metadata = Path(temporary) / "identity.json"
                staged_metadata.write_bytes(_json_bytes(artifact.identity(uri)))
                os.replace(staged, target)
                os.replace(staged_metadata, metadata)
            return artifact
