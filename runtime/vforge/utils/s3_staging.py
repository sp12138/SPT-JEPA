"""High-throughput S3 downloads for node-local checkpoint staging."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class S3Source:
    bucket: str
    key_prefix: str
    region: str
    endpoint_url: str | None = None
    addressing_style: str = "auto"


@dataclass(frozen=True)
class S3MountConfig:
    mount_path: str
    bucket: str
    region: str = "us-west-2"
    key_prefix: str = ""
    endpoint_url: str | None = None
    addressing_style: str = "auto"

    def __post_init__(self) -> None:
        if not os.path.isabs(os.path.expanduser(self.mount_path)):
            raise ValueError("s3 mount_path must be absolute")
        mount_path = os.path.realpath(os.path.expanduser(self.mount_path))
        bucket = self.bucket.strip()
        region = self.region.strip()
        key_prefix = self.key_prefix.strip("/")
        endpoint_url = self.endpoint_url.strip() if self.endpoint_url else None
        addressing_style = self.addressing_style.strip()
        if not bucket:
            raise ValueError("s3 mount bucket must not be empty")
        if not region:
            raise ValueError("s3 mount region must not be empty")
        if addressing_style not in {"auto", "path", "virtual"}:
            raise ValueError("s3 addressing_style must be 'auto', 'path', or 'virtual'")
        object.__setattr__(self, "mount_path", mount_path)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "region", region)
        object.__setattr__(self, "key_prefix", key_prefix)
        object.__setattr__(self, "endpoint_url", endpoint_url)
        object.__setattr__(self, "addressing_style", addressing_style)


@dataclass(frozen=True)
class StorageConfig:
    s3_mounts: tuple[S3MountConfig, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(self.s3_mounts)
        paths = [mount.mount_path for mount in normalized]
        if len(paths) != len(set(paths)):
            raise ValueError("storage.s3_mounts contains duplicate mount_path")
        object.__setattr__(
            self,
            "s3_mounts",
            tuple(
                sorted(
                    normalized,
                    key=lambda mount: len(Path(mount.mount_path).parts),
                    reverse=True,
                )
            ),
        )

    def resolve(self, source: str | os.PathLike[str]) -> S3Source | None:
        source_realpath = os.path.realpath(os.path.expanduser(os.fspath(source)))
        for mount in self.s3_mounts:
            try:
                if os.path.commonpath((mount.mount_path, source_realpath)) != (mount.mount_path):
                    continue
            except ValueError:
                continue
            relative = os.path.relpath(source_realpath, mount.mount_path)
            relative_key = "" if relative == "." else relative.replace(os.sep, "/")
            return S3Source(
                bucket=mount.bucket,
                key_prefix="/".join(part for part in (mount.key_prefix, relative_key) if part),
                region=mount.region,
                endpoint_url=mount.endpoint_url,
                addressing_style=mount.addressing_style,
            )
        return None


_STORAGE_CONFIG: ContextVar[StorageConfig | None] = ContextVar(
    "vforge_storage_config",
    default=None,
)


def build_storage_config(
    value: StorageConfig | Mapping[str, Any] | Iterable[Mapping[str, Any]] | None,
) -> StorageConfig:
    if value is None:
        return StorageConfig()
    if isinstance(value, StorageConfig):
        return value
    mounts_value: Any = value
    if isinstance(value, Mapping):
        mounts_value = value.get("s3_mounts", ())
    if isinstance(mounts_value, (str, bytes)) or not isinstance(
        mounts_value,
        Iterable,
    ):
        raise TypeError("storage.s3_mounts must be a sequence of mappings")
    mounts = []
    for item in mounts_value:
        if isinstance(item, S3MountConfig):
            mounts.append(item)
        elif isinstance(item, Mapping):
            mounts.append(S3MountConfig(**dict(item)))
        else:
            raise TypeError("each storage.s3_mounts item must be a mapping")
    return StorageConfig(s3_mounts=tuple(mounts))


def activate_storage_config(
    config: StorageConfig,
) -> Token[StorageConfig | None]:
    return _STORAGE_CONFIG.set(config)


def restore_storage_config(token: Token[StorageConfig | None]) -> None:
    _STORAGE_CONFIG.reset(token)


def current_storage_config() -> StorageConfig:
    return _STORAGE_CONFIG.get() or StorageConfig()


def resolve_s3_source(source: str) -> S3Source | None:
    """Resolve an explicitly declared S3 Mountpoint mapping."""
    return current_storage_config().resolve(source)


def _s3_client(s3: S3Source) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "S3 checkpoint transfer requires boto3 CRT support; run uv sync"
        ) from exc
    kwargs: dict[str, Any] = {"region_name": s3.region}
    if s3.endpoint_url:
        kwargs["endpoint_url"] = s3.endpoint_url
    if s3.addressing_style != "auto":
        from botocore.config import Config

        kwargs["config"] = Config(s3={"addressing_style": s3.addressing_style})
    return boto3.client("s3", **kwargs)


def _transfer_config(concurrency: int, *, prefer_crt: bool) -> Any:
    try:
        from boto3.s3.transfer import TransferConfig
    except ImportError as exc:
        raise RuntimeError(
            "S3 checkpoint transfer requires boto3 CRT support; run uv sync"
        ) from exc
    return TransferConfig(
        multipart_threshold=8 * 1024**2,
        multipart_chunksize=64 * 1024**2,
        max_concurrency=concurrency,
        preferred_transfer_client="crt" if prefer_crt else "classic",
    )


def _download_object(
    target: Path,
    *,
    s3: S3Source,
    key: str,
    expected_size: int | None,
    concurrency: int,
    client: Any | None = None,
) -> None:
    client = client or _s3_client(s3)
    if expected_size is None:
        expected_size = int(client.head_object(Bucket=s3.bucket, Key=key)["ContentLength"])
    client.download_file(
        s3.bucket,
        key,
        str(target),
        Config=_transfer_config(concurrency, prefer_crt=s3.endpoint_url is None),
    )
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise OSError(
            f"S3 download size mismatch for s3://{s3.bucket}/{key}: "
            f"{actual_size} != {expected_size}"
        )


def _upload_object(
    source: Path,
    *,
    s3: S3Source,
    key: str,
    concurrency: int,
) -> None:
    client = _s3_client(s3)
    client.upload_file(
        str(source),
        s3.bucket,
        key,
        Config=_transfer_config(concurrency, prefer_crt=s3.endpoint_url is None),
    )
    remote_size = int(client.head_object(Bucket=s3.bucket, Key=key)["ContentLength"])
    local_size = source.stat().st_size
    if remote_size != local_size:
        raise OSError(
            f"S3 upload size mismatch for s3://{s3.bucket}/{key}: {remote_size} != {local_size}"
        )


def download_file_crt(
    destination: str,
    *,
    s3: S3Source,
    transfer_concurrency: int = 32,
) -> None:
    """Download one S3 object atomically with the CRT transfer manager."""
    if not s3.key_prefix:
        raise ValueError("S3 file download requires a non-empty object key")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        _download_object(
            temporary,
            s3=s3,
            key=s3.key_prefix,
            expected_size=None,
            concurrency=transfer_concurrency,
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def upload_file_crt(
    source: str,
    *,
    s3: S3Source,
    transfer_concurrency: int = 32,
) -> None:
    """Upload one local file with the CRT transfer manager and verify size."""
    if not s3.key_prefix:
        raise ValueError("S3 file upload requires a non-empty object key")
    _upload_object(
        Path(source),
        s3=s3,
        key=s3.key_prefix,
        concurrency=transfer_concurrency,
    )


def download_tree_crt(
    source: str,
    destination: str,
    *,
    s3: S3Source,
    object_workers: int = 8,
    transfer_concurrency: int = 32,
    relative_paths: tuple[str, ...] | None = None,
) -> None:
    """Download one mounted S3 directory with a single coordinating process."""
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=False)
    discovery_started = time.perf_counter()
    if relative_paths is None:
        candidates = (
            path for path in source_path.rglob("*") if path.is_file() and not path.is_symlink()
        )
    else:
        candidates = []
        for relative in relative_paths:
            path = source_path / relative
            if (
                Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not path.is_file()
                or path.is_symlink()
            ):
                raise ValueError(f"invalid selective checkpoint file: {relative!r}")
            candidates.append(path)
    files = sorted(
        candidates,
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if os.environ.get("VFORGE_STAGE_TRACE") == "1":
        print(
            "VFORGE_STAGE_PHASE name=s3_discovery "
            f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
            f"elapsed_s={time.perf_counter() - discovery_started:.6f} "
            f"files={len(files)}",
            flush=True,
        )

    def download_one(item: tuple[object, Path]) -> Path:
        client, path = item
        relative = path.relative_to(source_path)
        target = destination_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        key = "/".join(part for part in (s3.key_prefix, relative.as_posix()) if part)
        _download_object(
            target,
            s3=s3,
            key=key,
            expected_size=path.stat().st_size,
            concurrency=transfer_concurrency,
            client=client,
        )
        return target

    def sync_one(target: Path) -> None:
        descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    download_started = time.perf_counter()
    targets = []
    for batch_index, offset in enumerate(range(0, len(files), object_workers)):
        batch_started = time.perf_counter()
        batch = files[offset : offset + object_workers]
        clients = [_s3_client(s3) for _ in batch]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            targets.extend(executor.map(download_one, zip(clients, batch, strict=True)))
        if os.environ.get("VFORGE_STAGE_TRACE") == "1":
            print(
                "VFORGE_STAGE_PHASE name=s3_download_batch "
                f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
                f"batch={batch_index} objects={len(batch)} "
                f"elapsed_s={time.perf_counter() - batch_started:.6f}",
                flush=True,
            )
    fsync_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=object_workers) as executor:
        list(executor.map(sync_one, targets))
    if os.environ.get("VFORGE_STAGE_TRACE") == "1":
        print(
            "VFORGE_STAGE_PHASE name=s3_fsync "
            f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
            f"elapsed_s={time.perf_counter() - fsync_started:.6f}",
            flush=True,
        )
        print(
            "VFORGE_STAGE_PHASE name=s3_download "
            f"local_rank={os.environ.get('LOCAL_RANK', '0')} "
            f"elapsed_s={time.perf_counter() - download_started:.6f}",
            flush=True,
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def upload_tree_crt(
    source: str,
    *,
    s3: S3Source,
    object_workers: int = 8,
    transfer_concurrency: int = 32,
) -> list[dict[str, Any]]:
    """Upload one node-local checkpoint tree with a single worker process."""
    if object_workers < 1:
        raise ValueError("object_workers must be >= 1")
    if transfer_concurrency < 1:
        raise ValueError("transfer_concurrency must be >= 1")

    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"S3 upload source must be a directory: {source}")
    files = sorted(
        (path for path in source_path.rglob("*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )

    def upload_one(path: Path) -> dict[str, Any]:
        relative = path.relative_to(source_path).as_posix()
        key = "/".join(part for part in (s3.key_prefix, relative) if part)
        size = path.stat().st_size
        sha256 = _sha256_file(path)
        _upload_object(
            path,
            s3=s3,
            key=key,
            concurrency=transfer_concurrency,
        )
        return {
            "path": relative,
            "size": size,
            "sha256": sha256,
        }

    uploaded = []
    for offset in range(0, len(files), object_workers):
        batch = files[offset : offset + object_workers]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            uploaded.extend(executor.map(upload_one, batch))
    uploaded.sort(key=lambda item: str(item["path"]))
    return uploaded


__all__ = [
    "S3MountConfig",
    "S3Source",
    "StorageConfig",
    "activate_storage_config",
    "build_storage_config",
    "current_storage_config",
    "download_file_crt",
    "download_tree_crt",
    "resolve_s3_source",
    "restore_storage_config",
    "upload_file_crt",
    "upload_tree_crt",
]
