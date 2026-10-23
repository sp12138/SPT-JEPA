"""B300 Native CD golden regression, eight ranks / SP8, serially on the existing Wan worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from training_regression import (
    CASES,
    SEED,
    STEPS,
    environment,
    prepare_assets,
    read_trace,
    train_config,
    verify_trace,
)


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("case", list(CASES))
def test_cd_native_training(case, tmp_path, torchrun, monkeypatch):
    golden = json.loads(Path(__file__).with_name("cd_b300_sp8.json").read_text())
    assert golden["environment"] == environment()
    assert "B300" in golden["environment"]["gpu"]
    assert (golden["seed"], golden["steps"], golden["world_size"], golden["usp_size"]) == (
        SEED,
        STEPS,
        8,
        8,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))
    assets, output = tmp_path / "assets", tmp_path / "train"
    assert prepare_assets(case, assets) == golden["cases"][case]["inputs"]
    torchrun(__file__, case, assets, output, timeout=2400)
    verify_trace(read_trace(case, output), golden["cases"][case]["losses"])


def test_cd_golden_rejects_one_ulp_drift():
    import math

    expected = [{"step": step, "loss": (1.0).hex()} for step in range(STEPS)]
    verify_trace(expected, expected)
    actual = [dict(row) for row in expected]
    actual[7]["loss"] = math.nextafter(1.0, math.inf).hex()
    with pytest.raises(AssertionError, match="step 7 bitwise mismatch"):
        verify_trace(actual, expected)


@pytest.mark.parametrize("case", list(CASES))
def test_cd_ci_precision_and_optimizer_contract(case, tmp_path):
    cfg = train_config(case, tmp_path / "assets", tmp_path / "train")
    assert cfg.runtime.max_steps == STEPS and cfg.runtime.seed == SEED
    assert cfg.runtime.deterministic and cfg.runtime.precision == "bf16"
    assert cfg.runtime.gradient_norm_dtype == "float32"
    assert cfg.parallel.mixed_precision.reduce_dtype == "float32"
    assert cfg.parallel.mixed_precision.contract_mode == "append"
    assert list(cfg.parallel.mixed_precision.preserve_forward_input_paths) == ["blocks.*"]
    assert cfg.parallel.usp_size == 8 and cfg.parallel.dp_size == 1
    assert cfg.parallel.include_usp_in_fsdp
    assert cfg.attention.backend == "flash_attention_4" and not cfg.attention.allow_fallback
    assert cfg.optimizers.student.eps == 1e-8
    assert cfg.optimizers.student.learning_rate == 5e-6
    assert list(cfg.optimizers.student.betas) == [0.9, 0.999]
    assert cfg.optimizers.student.weight_decay == 0.01
    assert cfg.runtime.gradient_clip_val == 1.0
    assert cfg.training.cd.ema.decay == 0.995
    assert cfg.scheduler.name == "none" and cfg.scheduler.warmup_steps == 0
    assert not cfg.checkpoint.enabled and not cfg.checkpoint.save_final


if __name__ == "__main__":
    import torch

    from vforge.tasks.train import run_train
    from vforge.utils.rng import seed_process

    case, assets, output = sys.argv[1:]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_process(SEED)
    run_train(train_config(case, Path(assets), Path(output)))
