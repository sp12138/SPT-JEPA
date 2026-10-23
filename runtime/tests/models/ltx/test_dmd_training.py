"""Full LTX DMD training with fixed, byte-identical original cache samples."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from tests.models.ltx.loss_regression import verify_loss_trace


def _golden(usp_size: int, *, long_tuning: bool = False) -> dict:
    name = "long_tuning_b300.json" if long_tuning else "dmd_b300.json"
    golden = json.loads(Path(__file__).with_name(name).read_text())
    if long_tuning:
        assert golden["steps"] == 18, "Long Tuning CI must cover two complete chains"
        golden["experiment"] = "LTX23_LONG_TUNING"
    return golden if usp_size == 1 else {**golden, **golden[f"sp{usp_size}"]}


def _check_training(path: Path, golden: dict) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [row for row in records if row.get("event") == "step"]
    steps = golden.get("steps", 10)
    assert [row["step"] for row in records] == list(range(steps))
    assert len(golden["losses"]) == steps
    trace = []
    for row, expected in zip(records, golden["losses"], strict=True):
        step = row["step"]
        assert expected["step"] == step
        assert row["world_size"] == row["dp_size"] * row["usp_size"] == 8
        assert f"DP{row['dp_size']}xUSP{row['usp_size']}" == golden["topology"], (
            "DMD golden topology does not match training"
        )
        phases = ["generator", "critic"] if step % 5 == 0 else ["critic"]
        assert row["step_metadata"]["phase_order"] == phases
        assert expected.keys() == {"step", *(f"{phase}/loss" for phase in phases)}, (
            f"DMD step {step}: golden loss fields do not match phases {phases}"
        )
        losses = {"step": step}
        for phase in phases:
            loss = row["metrics"][f"{phase}/loss"]
            norm = row["metrics"][f"{phase}/diag/grad_norm_post_reduce"]
            assert math.isfinite(loss), f"step {step}: non-finite {phase} loss"
            assert math.isfinite(norm) and norm > 0, f"step {step}: invalid {phase} gradient"
            rates = row["step_metadata"]["phase_learning_rates"][phase]
            assert rates == {"before": 1e-5, "after": 1e-5}
            losses[f"{phase}/loss"] = float(loss).hex()
        trace.append(losses)
    label = golden.get("experiment", "LTX23_DMD")
    print(f"{label}_{golden['topology']}_LOSS_TRACE=" + json.dumps(trace), flush=True)
    for actual, expected in zip(trace, golden["losses"], strict=True):
        verify_loss_trace(
            [actual], [expected], fields=tuple(field for field in actual if field != "step")
        )
    return trace


def test_dmd_loss_trace_rejects_one_ulp_drift(tmp_path):
    golden = _golden(1)
    records = []
    for losses in golden["losses"]:
        phases = ["generator", "critic"] if losses["step"] % 5 == 0 else ["critic"]
        records.append(
            {
                "event": "step",
                "step": losses["step"],
                "world_size": 8,
                "dp_size": 8,
                "usp_size": 1,
                "metrics": {
                    **{f"{phase}/loss": float.fromhex(losses[f"{phase}/loss"]) for phase in phases},
                    **{f"{phase}/diag/grad_norm_post_reduce": 1.0 for phase in phases},
                },
                "step_metadata": {
                    "phase_order": phases,
                    "phase_learning_rates": {
                        phase: {"before": 1e-5, "after": 1e-5} for phase in phases
                    },
                },
            }
        )
    path = tmp_path / "train_steps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert _check_training(path, golden) == golden["losses"]

    # A single changed generator loss must fail even when all critic losses match.
    field = "generator/loss"
    records[0]["metrics"][field] = math.nextafter(records[0]["metrics"][field], math.inf)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(AssertionError, match=f"step 0 {field} bitwise mismatch"):
        _check_training(path, golden)


def _run_training(torchrun, assets, output, usp_size, *overrides, long_tuning=False):
    golden = _golden(usp_size, long_tuning=long_tuning)
    experiment = "ltx23_long_tuning" if long_tuning else "ltx23_dmd"
    variant_overrides = (
        []
        if long_tuning
        else [
            f"training.models.real_score.init_weights={assets['score']}",
            "training.models.generator.config.max_train_frames=113",
        ]
    )
    torchrun(
        "--module",
        "vforge.tasks.train",
        f"experiment={experiment}",
        f"training.models.generator.pretrained_path={assets['base']}",
        f"training.models.generator.init_weights={assets['generator']}",
        f"training.models.generator.config.text_encoder_path={assets['gemma']}",
        f"data.sources.0.cache_dir={assets['cache']}",
        f"runtime.output_dir={output}",
        f"runtime.max_steps={golden.get('steps', 10)}",
        "runtime.seed=42",
        f"parallel.dp_size={8 // usp_size}",
        f"parallel.usp_size={usp_size}",
        *variant_overrides,
        "training.dmd.rollout.backward_mode=block_replay",
        "attention.backend=flash_attention_4",
        "runtime.deterministic=true",
        "checkpoint.enabled=false",
        "resume.path=none",
        *overrides,
        timeout=3600,
    )


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("usp_size", [1, 2], ids=["dp8_sp1", "dp4_sp2"])
def test_ltx23_dmd_b300_loss_is_bitwise_stable(torchrun, monkeypatch, dmd_assets, usp_size):
    golden = _golden(usp_size)
    assets = dmd_assets("base", "generator", "score", "gemma", "cache")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    with tempfile.TemporaryDirectory(prefix="vforge-dmd-training-", dir="/local-ssd") as directory:
        output = Path(directory) / "training"
        save = usp_size == 1
        _run_training(
            torchrun,
            assets,
            output,
            usp_size,
            f"checkpoint.enabled={str(save).lower()}",
            "checkpoint.every_n_steps=0",
            "checkpoint.save_final=true",
            "checkpoint.async_save=false",
        )
        _check_training(output / "train_steps.jsonl", golden)
        checkpoints = list((output / "checkpoints").glob("*.dcp"))
        if save:
            assert len(checkpoints) == 1
            # The SP1 loss run supplies the only full checkpoint; no extra training run.
            from tests.models.ltx.test_dmd_inference_gpu import run_dmd_eval

            run_dmd_eval(torchrun, output / "checkpoint-eval", assets, checkpoint=checkpoints[0])
            assert list((output / "checkpoints").glob("*.dcp")) == checkpoints
        else:
            assert not any(path.is_file() for path in (output / "checkpoints").rglob("*")), (
                "DMD SP2 loss CI must not save checkpoints"
            )


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("usp_size", [1, 2], ids=["dp8_sp1", "dp4_sp2"])
def test_ltx23_long_tuning_b300_loss_is_bitwise_stable(torchrun, monkeypatch, dmd_assets, usp_size):
    golden = _golden(usp_size, long_tuning=True)
    assets = dmd_assets("base", "generator", "gemma", "cache", long_tuning=True)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    with tempfile.TemporaryDirectory(prefix="vforge-long-tuning-", dir="/local-ssd") as directory:
        output = Path(directory) / "training"
        _run_training(torchrun, assets, output, usp_size, long_tuning=True)
        _check_training(output / "train_steps.jsonl", golden)
        assert not any(path.is_file() for path in (output / "checkpoints").rglob("*"))
