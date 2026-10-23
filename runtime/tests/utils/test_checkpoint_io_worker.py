import os

from vforge.utils.checkpoint_io_worker import (
    CheckpointIORequest,
    start_checkpoint_io_worker,
)


def test_filesystem_transfer_runs_in_spawned_process(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")

    worker = start_checkpoint_io_worker(
        CheckpointIORequest(
            operation="download",
            source=str(source),
            destination=str(destination),
            backend="filesystem",
            object_workers=2,
        )
    )
    result = worker.wait(timeout=30)

    assert result.worker_pid != os.getpid()
    assert result.files == [{"path": "payload.bin", "size": 7}]
    assert (destination / "payload.bin").read_bytes() == b"payload"
