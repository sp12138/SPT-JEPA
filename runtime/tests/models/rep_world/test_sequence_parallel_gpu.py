"""Real CUDA coverage for RepWorld Ulysses; no external codec assets required."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKER = Path(__file__).with_name("sequence_parallel_worker.py")


def run_oracle(tmp_path, world, usp, replicas):
    result = tmp_path / f"world{world}-usp{usp}-replicas{replicas}.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        str(WORKER),
        "--usp",
        str(usp),
        "--replicate",
        str(replicas),
        "--output",
        str(result),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout
    record = json.loads(result.read_text())
    assert record["passed"] and len(record["cases"]) == 20
    assert record["random_check"]["different_rank_rng"]


@pytest.mark.gpu(gpus=2)
def test_usp2_gradients_muon_recompute_and_kv(tmp_path):
    run_oracle(tmp_path, 2, 2, 1)


@pytest.mark.gpu(gpus=4)
@pytest.mark.parametrize("usp", [2, 4])
def test_dp_usp_and_four_way_sequence(tmp_path, usp):
    run_oracle(tmp_path, 4, usp, 1)


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("usp,replicas", [(8, 1), (2, 2)])
def test_usp8_and_hsdp_usp(tmp_path, usp, replicas):
    run_oracle(tmp_path, 8, usp, replicas)
