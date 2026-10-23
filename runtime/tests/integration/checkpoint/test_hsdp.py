"""Four-GPU integration smoke, not a full resume-equivalence contract."""

from pathlib import Path

import pytest


@pytest.mark.gpu(gpus=4)
def test_hsdp_checkpoint_roundtrip_and_reshard(tmp_path, torchrun):
    worker = Path(__file__).with_name("hsdp_worker.py")
    checkpoint = tmp_path / "checkpoint"
    torchrun(
        worker,
        "--dp-replicate-size=2",
        "--dp-shard-size=2",
        f"--checkpoint-dir={checkpoint}",
        "--checkpoint-mode=roundtrip",
    )
    torchrun(
        worker,
        "--dp-replicate-size=1",
        "--dp-shard-size=4",
        f"--checkpoint-dir={checkpoint}",
        "--checkpoint-mode=load",
    )
