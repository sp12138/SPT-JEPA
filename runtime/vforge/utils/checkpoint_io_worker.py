"""Isolated per-node checkpoint file transfer worker."""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from typing import Any

from vforge.utils.s3_staging import S3Source


@dataclass(frozen=True)
class CheckpointIORequest:
    operation: str
    source: str
    destination: str
    backend: str
    object_workers: int = 8
    transfer_concurrency: int = 32
    s3: S3Source | None = None
    relative_paths: tuple[str, ...] | None = None


@dataclass(frozen=True)
class CheckpointIOResult:
    operation: str
    source: str
    destination: str
    backend: str
    files: list[dict[str, Any]]
    elapsed_s: float
    worker_pid: int

    @property
    def total_bytes(self) -> int:
        return sum(int(item["size"]) for item in self.files)


class CheckpointIOWorkerError(RuntimeError):
    """Raised when the isolated node I/O worker fails."""


class CheckpointIOWorker:
    """Handle for one spawned checkpoint transfer process."""

    def __init__(
        self,
        process: multiprocessing.Process | None,
        connection: Connection | None,
        *,
        inline_payload: dict[str, Any] | None = None,
        inline_future: Future[dict[str, Any]] | None = None,
        inline_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._process = process
        self._connection = connection
        self._inline_payload = inline_payload
        self._inline_future = inline_future
        self._inline_executor = inline_executor

    def wait(self, timeout: float | None = None) -> CheckpointIOResult:
        if self._inline_future is not None:
            try:
                payload = self._inline_future.result(timeout=timeout)
            finally:
                if self._inline_executor is not None:
                    self._inline_executor.shutdown(wait=True)
                    self._inline_executor = None
        elif self._inline_payload is not None:
            payload = self._inline_payload
        else:
            if self._process is None or self._connection is None:
                raise RuntimeError("checkpoint I/O worker handle is incomplete")
            if not self._connection.poll(timeout):
                self._process.terminate()
                self._process.join(timeout=5.0)
                raise TimeoutError(f"checkpoint I/O worker timed out after {timeout} seconds")
            payload = self._connection.recv()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
                raise CheckpointIOWorkerError(
                    "checkpoint I/O worker did not exit after returning a result"
                )
            if self._process.exitcode not in {0, None} and payload.get("ok"):
                raise CheckpointIOWorkerError(
                    f"checkpoint I/O worker exited unexpectedly: exitcode={self._process.exitcode}"
                )
        if not payload.get("ok"):
            raise CheckpointIOWorkerError(
                "checkpoint I/O worker failed: "
                f"{payload.get('error', 'unknown error')}\n"
                f"{payload.get('traceback', '')}"
            )
        return CheckpointIOResult(**payload["result"])


def _local_files(root: str) -> list[dict[str, Any]]:
    files = []
    for directory, _subdirs, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(directory, name)
            if os.path.islink(path):
                continue
            files.append(
                {
                    "path": os.path.relpath(path, root).replace(os.sep, "/"),
                    "size": os.path.getsize(path),
                }
            )
    files.sort(key=lambda item: str(item["path"]))
    return files


def _run_request(request: CheckpointIORequest) -> CheckpointIOResult:
    started = time.perf_counter()
    if request.backend == "s3-crt" and request.s3 is None:
        raise ValueError("S3 checkpoint I/O requires a resolved S3 source")
    if request.operation == "download":
        if request.backend == "s3-crt":
            from vforge.utils.s3_staging import download_tree_crt

            download_kwargs = {}
            if request.relative_paths is not None:
                download_kwargs["relative_paths"] = request.relative_paths
            download_tree_crt(
                request.source,
                request.destination,
                s3=request.s3,
                object_workers=request.object_workers,
                transfer_concurrency=request.transfer_concurrency,
                **download_kwargs,
            )
            files = _local_files(request.destination)
        elif request.backend == "filesystem":
            from vforge.utils.node_staging import (
                _copy_files_parallel,
                _copytree_parallel,
            )

            if request.relative_paths is None:
                _copytree_parallel(
                    request.source,
                    request.destination,
                    workers=request.object_workers,
                )
            else:
                _copy_files_parallel(
                    request.source,
                    request.destination,
                    relative_paths=request.relative_paths,
                    workers=request.object_workers,
                )
            files = _local_files(request.destination)
        else:
            raise ValueError(f"unsupported checkpoint download backend: {request.backend}")
    elif request.operation == "upload":
        if request.backend != "s3-crt":
            raise ValueError(f"unsupported checkpoint upload backend: {request.backend}")
        from vforge.utils.s3_staging import upload_tree_crt

        files = upload_tree_crt(
            request.source,
            s3=request.s3,
            object_workers=request.object_workers,
            transfer_concurrency=request.transfer_concurrency,
        )
    elif request.operation == "upload_file":
        if request.backend != "s3-crt":
            raise ValueError(f"unsupported checkpoint file upload backend: {request.backend}")
        from vforge.utils.s3_staging import upload_file_crt

        upload_file_crt(
            request.source,
            s3=request.s3,
            transfer_concurrency=request.transfer_concurrency,
        )
        files = [
            {
                "path": os.path.basename(request.source),
                "size": os.path.getsize(request.source),
            }
        ]
    else:
        raise ValueError(f"unsupported checkpoint I/O operation: {request.operation}")
    return CheckpointIOResult(
        operation=request.operation,
        source=request.source,
        destination=request.destination,
        backend=request.backend,
        files=files,
        elapsed_s=time.perf_counter() - started,
        worker_pid=os.getpid(),
    )


def _worker_entry(request: CheckpointIORequest, connection: Connection) -> None:
    try:
        result = _run_request(request)
        connection.send({"ok": True, "result": asdict(result)})
    except BaseException as exc:
        connection.send(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        connection.close()


def start_checkpoint_io_worker(request: CheckpointIORequest) -> CheckpointIOWorker:
    """Start one isolated worker, or run inline only under an explicit test hook."""
    if os.environ.get("VFORGE_NODE_IO_INLINE") == "1":

        def run_inline() -> dict[str, Any]:
            try:
                result = _run_request(request)
                return {"ok": True, "result": asdict(result)}
            except BaseException as exc:
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_inline)
        return CheckpointIOWorker(
            None,
            None,
            inline_future=future,
            inline_executor=executor,
        )

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(request, child),
        name=f"vforge-checkpoint-{request.operation}",
        daemon=False,
    )
    process.start()
    child.close()
    return CheckpointIOWorker(process, parent)


__all__ = [
    "CheckpointIORequest",
    "CheckpointIOResult",
    "CheckpointIOWorker",
    "CheckpointIOWorkerError",
    "start_checkpoint_io_worker",
]
