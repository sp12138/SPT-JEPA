"""GPU integration coverage for vforge's MCore DDP adapter."""

from pathlib import Path

import pytest


@pytest.mark.gpu(gpus=2)
def test_mcore_ddp_and_distributed_optimizer_preserve_unused_parameters(torchrun) -> None:
    worker = Path(__file__).with_name("mcore_data_parallel_worker.py")
    torchrun(worker, "--dp-size=2", timeout=60)
