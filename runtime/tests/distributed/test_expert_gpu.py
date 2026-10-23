"""GPU worker gates for EP and EP + USP, including exact full resume."""

from pathlib import Path

import pytest


@pytest.mark.gpu(gpus=4)
@pytest.mark.parametrize("ep,usp", [(2, 1), (2, 2), (4, 2), (2, 4)])
def test_ep_fsdp_four_gpus(torchrun, tmp_path, monkeypatch, ep, usp):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torchrun(
        Path(__file__).with_name("expert_parallel_gpu_worker.py"),
        tmp_path,
        "--ep",
        ep,
        "--usp",
        usp,
        timeout=900,
    )


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("ep,usp", [(4, 1), (4, 2), (4, 4)])
def test_ep_fsdp_eight_gpus(torchrun, tmp_path, monkeypatch, ep, usp):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torchrun(
        Path(__file__).with_name("expert_parallel_gpu_worker.py"),
        tmp_path,
        "--ep",
        ep,
        "--usp",
        usp,
        timeout=900,
    )
