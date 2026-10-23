"""Public configuration and multi-GPU MMDiT EP/USP gates."""

from pathlib import Path

import pytest

from vforge.tasks.common import compose_standalone_config
from vforge.tasks.translate import build_runtime


@pytest.mark.parametrize(
    "world,ep,usp,replicate",
    [(4, 2, 2, 1), (8, 4, 2, 1), (8, 4, 4, 1), (16, 4, 4, 2), (128, 4, 4, 16)],
)
def test_parallel_overrides_compose(world, ep, usp, replicate):
    cfg = compose_standalone_config(
        "train",
        [
            "experiment=gen_engine",
            f"parallel.ep_size={ep}",
            f"parallel.usp_size={usp}",
            f"parallel.dp_replicate_size={replicate}",
            f"parallel.dp_shard_size={world // usp // replicate}",
            "checkpoint.enabled=false",
        ],
    )
    runtime = build_runtime(cfg, world_size=world)
    assert runtime.config.parallel.ep_size == ep
    assert runtime.config.parallel.usp_size == usp
    assert runtime.config.parallel.dp_replicate_size == replicate


@pytest.mark.gpu(gpus=4)
@pytest.mark.parametrize("case", ["balanced", "audio", "skew"])
@pytest.mark.parametrize("recompute", [False, True])
def test_mmdit_ep2_usp2(torchrun, tmp_path, monkeypatch, case, recompute):
    _run(torchrun, tmp_path, monkeypatch, 2, 2, case, recompute)


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("usp", [2, 4])
@pytest.mark.parametrize("case", ["balanced", "audio", "skew"])
@pytest.mark.parametrize("recompute", [False, True])
def test_mmdit_ep4_usp(torchrun, tmp_path, monkeypatch, usp, case, recompute):
    _run(torchrun, tmp_path, monkeypatch, 4, usp, case, recompute)


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("case", ["balanced", "audio", "skew"])
def test_mmdit_hsdp_ep_usp(torchrun, tmp_path, monkeypatch, case):
    _run(torchrun, tmp_path, monkeypatch, 2, 2, case, True, replicate=2)


def _run(torchrun, tmp_path, monkeypatch, ep, usp, case, recompute, replicate=1):
    del tmp_path
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = ["--ep", ep, "--usp", usp, "--replicate", replicate, "--case", case]
    if recompute:
        args.append("--recompute")
    torchrun(Path(__file__).with_name("mmdit_ep_usp_worker.py"), *args, timeout=900)
