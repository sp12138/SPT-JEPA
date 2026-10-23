"""Opt-in B300 numerical regression, without external model assets."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("stage,index", [("kernels", 0), *[("models", i) for i in range(8)]])
def test_fa4_numerics(tmp_path, stage, index):
    if "B300" not in torch.cuda.get_device_name(0):
        pytest.skip("FA4 RepWorld acceptance currently targets B300")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("fa4_worker.py")),
            "--stage",
            stage,
            "--index",
            str(index),
            "--output",
            str(tmp_path / "result.json"),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout
