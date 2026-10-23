"""Native accumulation gates, intentionally separate from one-batch baselines."""

from pathlib import Path

import pytest


@pytest.mark.gpu(gpus=4)
@pytest.mark.parametrize("mode", ["ddp", "fsdp", "hsdp", "tp", "usp", "ep", "ep_usp"])
def test_native_accumulation_topologies(torchrun, mode, tmp_path):
    torchrun(
        Path(__file__).with_name("accumulation_worker.py"),
        mode,
        "--checkpoint-dir",
        tmp_path,
        timeout=900,
    )


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("mode", ["hsdp_tp", "hsdp_usp", "ep_hsdp", "ep_hsdp_usp"])
def test_native_accumulation_hsdp_combinations(torchrun, mode, tmp_path):
    torchrun(
        Path(__file__).with_name("accumulation_worker.py"),
        mode,
        "--checkpoint-dir",
        tmp_path,
        timeout=900,
    )
