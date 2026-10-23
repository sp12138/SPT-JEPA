"""Node-local DCP serialization and per-node S3 CRT upload."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from typing import Any

import torch.distributed as dist

from vforge.training.checkpoint.manifest import CheckpointManifest, json_digest
from vforge.utils.checkpoint_io_worker import (
    CheckpointIORequest,
    start_checkpoint_io_worker,
)
from vforge.utils.s3_staging import S3Source, resolve_s3_source


@dataclass(frozen=True)
class RemoteSaveContext:
    remote_path: str
    local_incomplete: str
    s3: S3Source
    node_rank: int
    local_rank: int


def _rank_layout() -> tuple[int, int]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % local_world_size)))
    node_rank = int(
        os.environ.get(
            "NODE_RANK",
            os.environ.get("GROUP_RANK", str(rank // local_world_size)),
        )
    )
    return node_rank, local_rank


def _raise_distributed_errors(
    local_error: str | None,
    *,
    phase: str,
    process_group: Any,
) -> None:
    errors = [local_error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size(group=process_group)
        dist.all_gather_object(errors, local_error, group=process_group)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"remote checkpoint {phase} failed: " + "; ".join(failures))


def _delete_prefix(client: Any, bucket: str, prefix: str) -> None:
    paginator = client.get_paginator("list_objects_v2")
    batch = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            batch.append({"Key": item["Key"]})
            if len(batch) == 1000:
                client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                batch = []
    if batch:
        client.delete_objects(Bucket=bucket, Delete={"Objects": batch})


def prepare_remote_save(
    remote_path: str,
    *,
    process_group: Any,
) -> RemoteSaveContext:
    """Prepare one node-isolated local directory and an invisible S3 prefix."""
    s3 = resolve_s3_source(remote_path)
    if s3 is None:
        raise ValueError(
            "remote checkpoint upload requires checkpoint.dirpath to match "
            "an explicit storage.s3_mounts entry"
        )
    if not remote_path.endswith(".dcp"):
        raise ValueError(f"remote DCP path must end with .dcp: {remote_path}")

    remote_error: str | None = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            import boto3
            from botocore.exceptions import ClientError

            client = boto3.client("s3", region_name=s3.region)
            manifest_key = "/".join(part for part in (s3.key_prefix, "manifest.json") if part)
            try:
                client.head_object(Bucket=s3.bucket, Key=manifest_key)
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                if code not in {"404", "NoSuchKey", "NotFound"}:
                    raise
            else:
                raise FileExistsError(f"refusing to overwrite complete checkpoint: {remote_path}")
            _delete_prefix(client, s3.bucket, s3.key_prefix)
        except BaseException as exc:
            remote_error = f"{type(exc).__name__}: {exc}"
    _raise_distributed_errors(
        remote_error,
        phase="prepare remote prefix",
        process_group=process_group,
    )

    node_rank, local_rank = _rank_layout()
    local_root = os.environ.get(
        "VFORGE_CKPT_LOCAL_ROOT",
        "/local-ssd/vforge-checkpoints",
    )
    identity = hashlib.sha256(f"{s3.bucket}/{s3.key_prefix}".encode()).hexdigest()[:16]
    basename = os.path.basename(remote_path.rstrip(os.sep))
    local_incomplete = os.path.join(
        local_root,
        "uploads",
        f"node-{node_rank}",
        f"{basename}-{identity}.incomplete",
    )
    local_error = None
    if local_rank == 0:
        try:
            shutil.rmtree(local_incomplete, ignore_errors=True)
            os.makedirs(local_incomplete, exist_ok=False)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _raise_distributed_errors(
        local_error,
        phase="prepare node-local directory",
        process_group=process_group,
    )
    return RemoteSaveContext(
        remote_path=remote_path,
        local_incomplete=local_incomplete,
        s3=s3,
        node_rank=node_rank,
        local_rank=local_rank,
    )


def _manifest_with_uploaded_integrity(
    manifest: CheckpointManifest,
    files: list[dict[str, Any]],
) -> CheckpointManifest:
    integrity = {}
    for component, present in manifest.components_present.items():
        if not present:
            continue
        component_root = str(manifest.components.get(component, component)).strip("/")
        prefix = component_root + "/"
        component_files = [
            {
                "path": str(item["path"])[len(prefix) :],
                "size": int(item["size"]),
                "sha256": str(item["sha256"]),
            }
            for item in files
            if str(item["path"]).startswith(prefix)
        ]
        component_files.sort(key=lambda item: item["path"])
        if not component_files:
            raise ValueError(f"uploaded checkpoint component is empty: {component}")
        metadata_files = [{"path": item["path"], "size": item["size"]} for item in component_files]
        integrity[component] = {
            "file_count": len(component_files),
            "total_bytes": sum(item["size"] for item in component_files),
            "digest": json_digest(component_files),
            "metadata_digest": json_digest(metadata_files),
        }
    fingerprints = dict(manifest.fingerprints)
    fingerprints["component_integrity"] = integrity
    return replace(manifest, fingerprints=fingerprints)


def _put_commit_objects(
    context: RemoteSaveContext,
    manifest: CheckpointManifest,
) -> None:
    import boto3

    client = boto3.client("s3", region_name=context.s3.region)
    manifest_key = "/".join(part for part in (context.s3.key_prefix, "manifest.json") if part)
    manifest_bytes = (json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n").encode()
    # manifest.json is the commit marker: payloads are invisible to automatic
    # resume until this final object exists.
    client.put_object(
        Bucket=context.s3.bucket,
        Key=manifest_key,
        Body=manifest_bytes,
        ContentType="application/json",
    )
    parent = context.s3.key_prefix.rsplit("/", 1)[0] if "/" in context.s3.key_prefix else ""
    latest_key = "/".join(part for part in (parent, "latest") if part)
    client.put_object(
        Bucket=context.s3.bucket,
        Key=latest_key,
        Body=(os.path.basename(context.remote_path) + "\n").encode(),
        ContentType="text/plain",
    )


def _upload_and_commit_remote_save(
    context: RemoteSaveContext,
    manifest: CheckpointManifest,
    *,
    process_group: Any,
    timeout_seconds: float = 7200.0,
) -> CheckpointManifest:
    """Upload one local tree per node and commit only after all nodes succeed."""
    local_result: dict[str, Any] | None = None
    local_error: str | None = None
    if context.local_rank == 0:
        try:
            worker = start_checkpoint_io_worker(
                CheckpointIORequest(
                    operation="upload",
                    source=context.local_incomplete,
                    destination=context.remote_path,
                    backend="s3-crt",
                    object_workers=int(os.environ.get("VFORGE_STAGE_WORKERS", "8")),
                    transfer_concurrency=int(
                        os.environ.get("VFORGE_S3_TRANSFER_CONCURRENCY", "32")
                    ),
                    s3=context.s3,
                )
            )
            result = worker.wait(timeout=timeout_seconds)
            local_result = {
                "node_rank": context.node_rank,
                "files": result.files,
                "elapsed_s": result.elapsed_s,
                "worker_pid": result.worker_pid,
            }
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    _raise_distributed_errors(
        local_error,
        phase="node upload",
        process_group=process_group,
    )

    gathered = [local_result]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size(group=process_group)
        dist.all_gather_object(gathered, local_result, group=process_group)
    leaders = [item for item in gathered if item is not None]
    by_node = {}
    for item in leaders:
        by_node.setdefault(int(item["node_rank"]), item)
    files = [file for node in sorted(by_node) for file in by_node[node]["files"]]
    paths = [str(item["path"]) for item in files]
    if len(paths) != len(set(paths)):
        raise RuntimeError("remote checkpoint upload produced duplicate object paths")

    committed_manifest = _manifest_with_uploaded_integrity(manifest, files)
    commit_error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            _put_commit_objects(context, committed_manifest)
        except BaseException as exc:
            commit_error = f"{type(exc).__name__}: {exc}"
    _raise_distributed_errors(
        commit_error,
        phase="manifest commit",
        process_group=process_group,
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            "VFORGE_REMOTE_CHECKPOINT "
            + json.dumps(
                {
                    "event": "committed",
                    "path": context.remote_path,
                    "nodes": len(by_node),
                    "files": len(files),
                    "bytes": sum(int(item["size"]) for item in files),
                    "node_upload_s": {
                        str(node): float(item["elapsed_s"])
                        for node, item in sorted(by_node.items())
                    },
                    "worker_pids": {
                        str(node): int(item["worker_pid"]) for node, item in sorted(by_node.items())
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return committed_manifest


def upload_and_commit_remote_save(
    context: RemoteSaveContext,
    manifest: CheckpointManifest,
    *,
    process_group: Any,
    timeout_seconds: float = 7200.0,
) -> CheckpointManifest:
    """Upload, commit, and always release node-local serialization data."""
    try:
        return _upload_and_commit_remote_save(
            context,
            manifest,
            process_group=process_group,
            timeout_seconds=timeout_seconds,
        )
    finally:
        if context.local_rank == 0:
            shutil.rmtree(context.local_incomplete, ignore_errors=True)


__all__ = [
    "RemoteSaveContext",
    "prepare_remote_save",
    "upload_and_commit_remote_save",
]
