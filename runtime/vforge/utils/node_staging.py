"""Node-local, single-writer staging for immutable checkpoint directories."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class NodeStagingConfig:
    root: str = "/local-ssd/vforge-checkpoints"
    timeout_seconds: float = 7200.0
    poll_seconds: float = 1.0
    minimum_free_bytes: int = 5 * 1024**3
    copy_workers: int = 8
    s3_transfer_concurrency: int = 32

    def __post_init__(self) -> None:
        if self.copy_workers < 1:
            raise ValueError("copy_workers must be >= 1")
        if self.s3_transfer_concurrency < 1:
            raise ValueError("s3_transfer_concurrency must be >= 1")


@dataclass(frozen=True)
class StagingResult:
    source: str
    destination: str
    copied: bool
    files: int
    bytes: int
    backend: str = "filesystem"


def default_node_staging_config() -> NodeStagingConfig:
    """Build one staging policy from performance-only environment settings."""
    return NodeStagingConfig(
        root=os.environ.get(
            "VFORGE_CKPT_LOCAL_ROOT",
            "/local-ssd/vforge-checkpoints",
        ),
        copy_workers=int(os.environ.get("VFORGE_STAGE_WORKERS", "8")),
        s3_transfer_concurrency=int(os.environ.get("VFORGE_S3_TRANSFER_CONCURRENCY", "32")),
    )


def _trace_phase(name: str, started: float, *, rank: int) -> None:
    if os.environ.get("VFORGE_STAGE_TRACE") != "1":
        return
    print(
        f"VFORGE_STAGE_PHASE name={name} local_rank={rank} "
        f"elapsed_s={time.perf_counter() - started:.6f}",
        flush=True,
    )


def _source_identity(source: str) -> tuple[str, int]:
    real_source = os.path.realpath(source)
    return real_source, os.stat(real_source).st_mtime_ns


def _destination(
    root: str,
    source: str,
    *,
    selection_digest: str | None = None,
) -> str:
    identity = source if selection_digest is None else f"{source}\0selection={selection_digest}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    basename = os.path.basename(source.rstrip(os.sep)) or "checkpoint"
    return os.path.join(root, f"{basename}-{digest}")


def _normalize_relative_paths(
    source: str,
    relative_paths: Sequence[str],
) -> tuple[str, ...]:
    normalized = []
    for value in relative_paths:
        relative = os.path.normpath(str(value))
        if (
            not relative
            or relative == "."
            or os.path.isabs(relative)
            or relative == ".."
            or relative.startswith(f"..{os.sep}")
        ):
            raise ValueError(f"invalid selective staging path: {value!r}")
        path = os.path.join(source, relative)
        if not os.path.isfile(path) or os.path.islink(path):
            raise ValueError(f"selective staging file is missing or unsupported: {relative}")
        normalized.append(relative.replace(os.sep, "/"))
    result = tuple(sorted(set(normalized)))
    if not result:
        raise ValueError("selective staging requires at least one file")
    return result


def _selection_digest(relative_paths: Sequence[str] | None) -> str | None:
    if relative_paths is None:
        return None
    encoded = json.dumps(list(relative_paths), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tree_stats(
    source: str,
    *,
    relative_paths: Sequence[str] | None = None,
) -> tuple[int, int]:
    if relative_paths is not None:
        return len(relative_paths), sum(
            os.path.getsize(os.path.join(source, relative)) for relative in relative_paths
        )
    files = 0
    total_bytes = 0
    for directory, _subdirs, names in os.walk(source):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                continue
            files += 1
            total_bytes += os.path.getsize(path)
    return files, total_bytes


def _copy_files_parallel(
    source: str,
    destination: str,
    *,
    relative_paths: Sequence[str],
    workers: int,
) -> None:
    """Copy an explicit immutable file set while preserving relative paths."""
    os.makedirs(destination, exist_ok=False)
    pairs = []
    for relative in relative_paths:
        source_file = os.path.join(source, relative)
        destination_file = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        pairs.append((source_file, destination_file))

    def copy_file(paths: tuple[str, str]) -> None:
        shutil.copy2(*paths)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(copy_file, pairs))


def _copytree_parallel(source: str, destination: str, *, workers: int) -> None:
    """Copy distinct files concurrently while preserving directory layout.

    The destination root directory must not already exist; a FileExistsError
    will be raised if it does.
    """
    os.makedirs(destination, exist_ok=False)
    file_pairs: list[tuple[str, str]] = []
    directories: list[tuple[str, str]] = []
    for source_dir, child_dirs, filenames in os.walk(source):
        relative_dir = os.path.relpath(source_dir, source)
        destination_dir = (
            destination if relative_dir == "." else os.path.join(destination, relative_dir)
        )
        directories.append((source_dir, destination_dir))
        retained_dirs = []
        for name in child_dirs:
            source_child = os.path.join(source_dir, name)
            destination_child = os.path.join(destination_dir, name)
            if os.path.islink(source_child):
                os.symlink(os.readlink(source_child), destination_child)
            else:
                os.makedirs(destination_child, exist_ok=True)
                retained_dirs.append(name)
        child_dirs[:] = retained_dirs
        for name in filenames:
            source_file = os.path.join(source_dir, name)
            destination_file = os.path.join(destination_dir, name)
            if os.path.islink(source_file):
                os.symlink(os.readlink(source_file), destination_file)
            else:
                file_pairs.append((source_file, destination_file))

    def copy_file(paths: tuple[str, str]) -> None:
        source_file, destination_file = paths
        try:
            shutil.copy2(source_file, destination_file)
        except OSError as error:
            # Must propagate: the caller writes READY.json as soon as this
            # returns, so a swallowed failure publishes a truncated checkpoint
            # that every later run on this node then trusts.
            raise OSError(f"failed to copy {source_file} -> {destination_file}: {error}") from error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(copy_file, file_pairs))
    for source_dir, destination_dir in reversed(directories):
        if source_dir != source:
            shutil.copystat(source_dir, destination_dir, follow_symlinks=False)


def _read_ready(path: str) -> dict[str, object] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _ready_matches(
    payload: dict[str, object] | None,
    *,
    source: str,
    source_mtime_ns: int,
    selection_digest: str | None,
) -> bool:
    return bool(
        payload
        and payload.get("source") == source
        and payload.get("source_mtime_ns") == source_mtime_ns
        and payload.get("selection_digest") == selection_digest
        and isinstance(payload.get("files"), int)
        and isinstance(payload.get("bytes"), int)
    )


def _result_from_ready(
    source: str,
    destination: str,
    payload: dict[str, object],
    *,
    copied: bool,
) -> StagingResult:
    return StagingResult(
        source=source,
        destination=destination,
        copied=copied,
        files=int(payload["files"]),
        bytes=int(payload["bytes"]),
        backend=str(payload.get("backend", "filesystem")),
    )


def stage_on_node(
    source: str,
    *,
    config: NodeStagingConfig | None = None,
    local_rank: int | None = None,
    overlap_callback: Callable[[], None] | None = None,
    relative_paths: Sequence[str] | None = None,
    require_checkpoint_metadata: bool = True,
) -> StagingResult:
    """Copy one immutable directory per node and let local peers await it.

    When supplied, every local rank invokes ``overlap_callback`` exactly once.
    A node leader starts an otherwise synchronous copy in a background thread
    first, allowing collective initialization to overlap remote transfer.
    """
    total_started = time.perf_counter()
    cfg = config or NodeStagingConfig()
    rank = int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else int(local_rank)
    if not os.path.isdir(source):
        raise ValueError(f"staging source must be a directory: {source}")
    validation_started = time.perf_counter()
    selected_paths = (
        _normalize_relative_paths(source, relative_paths) if relative_paths is not None else None
    )
    selection_digest = _selection_digest(selected_paths)
    if selected_paths is not None and require_checkpoint_metadata:
        has_dcp_metadata = any(
            relative == ".metadata" or relative.endswith(os.sep + ".metadata")
            for relative in selected_paths
        )
        if "manifest.json" not in selected_paths and not has_dcp_metadata:
            raise ValueError("selective DCP staging requires manifest.json or .metadata")
    else:
        contains_safetensors = any(
            name.endswith(".safetensors")
            for _directory, _subdirs, names in os.walk(source)
            for name in names
        )
        contains_dcp_manifest = os.path.isfile(os.path.join(source, "manifest.json"))
        if not contains_safetensors and not contains_dcp_manifest:
            raise ValueError(f"staging source is neither safetensors nor DCP: {source}")
    _trace_phase("validate_source", validation_started, rank=rank)

    identity_started = time.perf_counter()
    source_realpath, source_mtime_ns = _source_identity(source)
    from vforge.utils.s3_staging import resolve_s3_source

    s3_source = resolve_s3_source(source_realpath)
    backend = "s3-crt" if s3_source is not None else "filesystem"
    _trace_phase("resolve_source", identity_started, rank=rank)
    os.makedirs(cfg.root, exist_ok=True)
    if not os.access(cfg.root, os.W_OK):
        raise PermissionError(f"staging root is not writable: {cfg.root}")
    destination = _destination(
        cfg.root,
        source_realpath,
        selection_digest=selection_digest,
    )
    ready_path = os.path.join(destination, "READY.json")

    ready = _read_ready(ready_path)
    if _ready_matches(
        ready,
        source=source_realpath,
        source_mtime_ns=source_mtime_ns,
        selection_digest=selection_digest,
    ):
        if overlap_callback is not None:
            overlap_callback()
        return _result_from_ready(
            source_realpath,
            destination,
            ready,
            copied=False,
        )

    if rank != 0:
        if overlap_callback is not None:
            overlap_callback()
        wait_started = time.perf_counter()
        deadline = time.monotonic() + cfg.timeout_seconds
        while time.monotonic() < deadline:
            ready = _read_ready(ready_path)
            if _ready_matches(
                ready,
                source=source_realpath,
                source_mtime_ns=source_mtime_ns,
                selection_digest=selection_digest,
            ):
                _trace_phase("wait_for_leader", wait_started, rank=rank)
                _trace_phase("total", total_started, rank=rank)
                return _result_from_ready(
                    source_realpath,
                    destination,
                    ready,
                    copied=False,
                )
            time.sleep(cfg.poll_seconds)
        raise TimeoutError(f"timed out waiting for node-local checkpoint: {destination}")

    lock_path = destination + ".lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        ready = _read_ready(ready_path)
        if _ready_matches(
            ready,
            source=source_realpath,
            source_mtime_ns=source_mtime_ns,
            selection_digest=selection_digest,
        ):
            if overlap_callback is not None:
                overlap_callback()
            return _result_from_ready(
                source_realpath,
                destination,
                ready,
                copied=False,
            )

        stats_started = time.perf_counter()
        file_count, total_bytes = _tree_stats(
            source_realpath,
            relative_paths=selected_paths,
        )
        _trace_phase("tree_stats", stats_started, rank=rank)
        disk_started = time.perf_counter()
        free_bytes = shutil.disk_usage(cfg.root).free
        _trace_phase("disk_usage", disk_started, rank=rank)
        required_bytes = total_bytes + cfg.minimum_free_bytes
        if free_bytes < required_bytes:
            raise OSError(
                "insufficient node-local space: "
                f"required={required_bytes}, free={free_bytes}, root={cfg.root}"
            )

        work_path = destination + f".work.{os.getpid()}"
        shutil.rmtree(work_path, ignore_errors=True)
        try:
            copy_started = time.perf_counter()
            from vforge.utils.checkpoint_io_worker import (
                CheckpointIORequest,
                start_checkpoint_io_worker,
            )

            worker = start_checkpoint_io_worker(
                CheckpointIORequest(
                    operation="download",
                    source=source_realpath,
                    destination=work_path,
                    backend=backend,
                    object_workers=cfg.copy_workers,
                    transfer_concurrency=cfg.s3_transfer_concurrency,
                    s3=s3_source,
                    relative_paths=selected_paths,
                )
            )
            if overlap_callback is not None:
                try:
                    overlap_callback()
                except BaseException:
                    # The transfer result is still consumed so the child cannot
                    # outlive the failed collective callback.
                    try:
                        worker.wait(timeout=cfg.timeout_seconds)
                    except BaseException:
                        pass
                    raise
            worker.wait(timeout=cfg.timeout_seconds)
            _trace_phase("copy_tree", copy_started, rank=rank)
            commit_started = time.perf_counter()
            payload = {
                "source": source_realpath,
                "source_mtime_ns": source_mtime_ns,
                "selection_digest": selection_digest,
                "files": file_count,
                "bytes": total_bytes,
                "backend": backend,
            }
            ready_tmp = os.path.join(work_path, "READY.json.tmp")
            with open(ready_tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(ready_tmp, os.path.join(work_path, "READY.json"))
            if os.path.exists(destination):
                shutil.rmtree(destination)
            os.replace(work_path, destination)
            _trace_phase("commit", commit_started, rank=rank)
        except BaseException:
            shutil.rmtree(work_path, ignore_errors=True)
            raise

    _trace_phase("total", total_started, rank=rank)
    return StagingResult(
        source=source_realpath,
        destination=destination,
        copied=True,
        files=file_count,
        bytes=total_bytes,
        backend=backend,
    )


__all__ = ["NodeStagingConfig", "StagingResult", "stage_on_node"]
