from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from safetensors.torch import save_file

from vforge.utils.node_staging import NodeStagingConfig, stage_on_node
from vforge.utils.s3_staging import (
    S3MountConfig,
    StorageConfig,
    activate_storage_config,
    restore_storage_config,
)


@pytest.fixture(autouse=True)
def _run_node_io_worker_inline(monkeypatch):
    monkeypatch.setenv("VFORGE_NODE_IO_INLINE", "1")


@pytest.fixture(autouse=True)
def _run_node_io_worker_inline(monkeypatch):
    monkeypatch.setenv("VFORGE_NODE_IO_INLINE", "1")


def _source(tmp_path):
    source = tmp_path / "remote" / "model"
    source.mkdir(parents=True)
    save_file({}, source / "model.safetensors")
    (source / "config.json").write_text("{}\n")
    return source


def _config(tmp_path):
    return NodeStagingConfig(
        root=str(tmp_path / "local"),
        timeout_seconds=2,
        poll_seconds=0.01,
        minimum_free_bytes=0,
    )


def test_stage_on_node_copies_once_and_reuses_ready_cache(tmp_path) -> None:
    source = _source(tmp_path)
    config = _config(tmp_path)

    first = stage_on_node(str(source), config=config, local_rank=0)
    second = stage_on_node(str(source), config=config, local_rank=0)

    assert first.copied is True
    assert second.copied is False
    assert first.destination == second.destination
    assert (tmp_path / "local").is_dir()
    assert (source / "model.safetensors").read_bytes() == (
        tmp_path / "local" / first.destination.split("/")[-1] / "model.safetensors"
    ).read_bytes()


def test_local_peer_waits_for_leader_ready_marker(tmp_path) -> None:
    source = _source(tmp_path)
    config = _config(tmp_path)
    results = []
    failures = []

    def wait_for_stage():
        try:
            results.append(stage_on_node(str(source), config=config, local_rank=1))
        except BaseException as exc:
            failures.append(exc)

    waiter = threading.Thread(target=wait_for_stage)
    waiter.start()
    time.sleep(0.05)
    leader = stage_on_node(str(source), config=config, local_rank=0)
    waiter.join()

    assert failures == []
    assert leader.copied is True
    assert len(results) == 1
    assert results[0].destination == leader.destination
    assert results[0].copied is False


def test_stage_copies_distinct_files_concurrently(monkeypatch, tmp_path) -> None:
    import vforge.utils.node_staging as staging

    source = _source(tmp_path)
    for index in range(8):
        (source / f"shard-{index}.bin").write_bytes(bytes([index]) * 1024)
    config = NodeStagingConfig(
        root=str(tmp_path / "local"),
        minimum_free_bytes=0,
        copy_workers=4,
    )
    original_copy2 = staging.shutil.copy2
    lock = threading.Lock()
    active = 0
    max_active = 0

    def tracked_copy2(source_path, destination_path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            return original_copy2(source_path, destination_path)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(staging.shutil, "copy2", tracked_copy2)

    result = stage_on_node(str(source), config=config, local_rank=0)

    assert result.copied is True
    assert max_active >= 2
    for index in range(8):
        assert (
            tmp_path / "local" / result.destination.split("/")[-1] / f"shard-{index}.bin"
        ).read_bytes() == bytes([index]) * 1024


def test_stage_overlaps_callback_with_leader_copy(monkeypatch, tmp_path) -> None:
    import vforge.utils.node_staging as staging

    source = _source(tmp_path)
    config = _config(tmp_path)
    original_copytree = staging._copytree_parallel
    copy_started = threading.Event()
    allow_copy = threading.Event()
    copy_finished = threading.Event()
    callback_calls = 0

    def blocked_copytree(source_path, destination_path, *, workers):
        copy_started.set()
        assert allow_copy.wait(timeout=1)
        original_copytree(source_path, destination_path, workers=workers)
        copy_finished.set()

    def overlap_callback():
        nonlocal callback_calls
        callback_calls += 1
        assert copy_started.wait(timeout=1)
        assert not copy_finished.is_set()
        allow_copy.set()

    monkeypatch.setattr(staging, "_copytree_parallel", blocked_copytree)

    result = stage_on_node(
        str(source),
        config=config,
        local_rank=0,
        overlap_callback=overlap_callback,
    )

    assert result.copied is True
    assert copy_finished.is_set()
    assert callback_calls == 1


def test_stage_cached_path_still_invokes_overlap_callback(tmp_path) -> None:
    source = _source(tmp_path)
    config = _config(tmp_path)
    stage_on_node(str(source), config=config, local_rank=0)
    calls = []

    result = stage_on_node(
        str(source),
        config=config,
        local_rank=0,
        overlap_callback=lambda: calls.append("warmup"),
    )

    assert result.copied is False
    assert calls == ["warmup"]


def test_stage_rejects_non_positive_copy_workers() -> None:
    with pytest.raises(ValueError, match="copy_workers"):
        NodeStagingConfig(copy_workers=0)


def test_stage_rejects_pt_files(tmp_path) -> None:
    source = tmp_path / "weights.pt"
    source.write_bytes(b"pt")

    with pytest.raises(ValueError, match="directory"):
        stage_on_node(str(source), config=_config(tmp_path), local_rank=0)


def test_stage_rejects_directory_without_safetensors(tmp_path) -> None:
    source = tmp_path / "remote"
    source.mkdir()
    (source / "weights.pt").write_bytes(b"pt")

    with pytest.raises(ValueError, match="neither safetensors nor DCP"):
        stage_on_node(str(source), config=_config(tmp_path), local_rank=0)


def test_stage_accepts_dcp_and_uses_s3_crt_for_mountpoint(
    monkeypatch,
    tmp_path,
) -> None:
    import vforge.utils.s3_staging as s3_staging

    mount_root = tmp_path / "s3"
    source = mount_root / "run" / "00002.dcp"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n")
    (source / "weights").mkdir()
    (source / "weights" / "__0_0.distcp").write_bytes(b"weights")
    calls = []

    def fake_download(
        source_path,
        destination_path,
        *,
        s3,
        object_workers,
        transfer_concurrency,
    ):
        calls.append(
            (
                source_path,
                s3.bucket,
                s3.key_prefix,
                object_workers,
                transfer_concurrency,
            )
        )
        import shutil

        shutil.copytree(source_path, destination_path)

    monkeypatch.setattr(s3_staging, "download_tree_crt", fake_download)
    token = activate_storage_config(
        StorageConfig(
            s3_mounts=(
                S3MountConfig(
                    str(mount_root),
                    "checkpoint-bucket",
                    "us-west-2",
                ),
            )
        )
    )
    try:
        result = stage_on_node(
            str(source),
            config=NodeStagingConfig(
                root=str(tmp_path / "local"),
                minimum_free_bytes=0,
                copy_workers=8,
                s3_transfer_concurrency=32,
            ),
            local_rank=0,
        )
    finally:
        restore_storage_config(token)

    assert result.backend == "s3-crt"
    assert calls == [
        (
            str(source),
            "checkpoint-bucket",
            "run/00002.dcp",
            8,
            32,
        )
    ]
    assert (
        tmp_path / "local" / result.destination.split("/")[-1] / "weights" / "__0_0.distcp"
    ).read_bytes() == b"weights"


def test_selective_stage_copies_only_explicit_dcp_files(tmp_path) -> None:
    source = tmp_path / "remote" / "weights"
    source.mkdir(parents=True)
    (source / ".metadata").write_bytes(b"metadata")
    (source / "__0_0.distcp").write_bytes(b"selected")
    (source / "__1_0.distcp").write_bytes(b"omitted")
    config = _config(tmp_path)

    result = stage_on_node(
        str(source),
        config=config,
        local_rank=0,
        relative_paths=(".metadata", "__0_0.distcp"),
    )

    destination = Path(result.destination)
    assert result.files == 2
    assert result.bytes == len(b"metadata") + len(b"selected")
    assert (destination / ".metadata").read_bytes() == b"metadata"
    assert (destination / "__0_0.distcp").read_bytes() == b"selected"
    assert not (destination / "__1_0.distcp").exists()


def test_selective_stage_rejects_missing_metadata(tmp_path) -> None:
    source = tmp_path / "remote" / "weights"
    source.mkdir(parents=True)
    (source / "__0_0.distcp").write_bytes(b"selected")

    with pytest.raises(ValueError, match=r"\.metadata"):
        stage_on_node(
            str(source),
            config=_config(tmp_path),
            local_rank=0,
            relative_paths=("__0_0.distcp",),
        )
