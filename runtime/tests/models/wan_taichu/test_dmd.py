"""Full A14B DMD: ten native SP8 steps with fixed inputs and bitwise loss checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

from vforge.data.cache import load_latent_bundle, save_latent_bundle


def _golden() -> dict:
    return json.loads(Path(__file__).with_name("dmd_b300.json").read_text())


def _asset(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(f"set {variable} to the existing DMD asset")
    return Path(value).expanduser().resolve(strict=True)


def _prepare_cache(source: Path, destination: Path, golden: dict) -> None:
    """Preserve the 30-sample pool used by the two independent DMD samplers."""
    assert len(golden["samples"]) == golden["sample_count"] == 30
    for index, sample in enumerate(golden["samples"]):
        number = int(sample["sample_id"])
        path = source / "train" / f"{number // 1000:04d}" / f"{number:08d}.pt"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == sample["sha256"], f"changed DMD input: {path}"
        raw = load_latent_bundle(str(path))
        bundle = {}
        for key in ("frame_latents", "condition_latents_rgb", "condition_latents_normal", "prompt"):
            value = raw[key]
            if key != "prompt":
                assert value.ndim == 4 and value.shape[1] >= golden["latent_frames"]
                value = value[:, : golden["latent_frames"]]
            bundle[key] = value.to(torch.bfloat16).contiguous()
        save_latent_bundle(bundle, str(destination / "train/0000" / f"{index:08d}.pt"))


def _check_training(path: Path, golden: dict) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if row.get("event") == "step"]
    assert [row["step"] for row in rows] == list(range(golden["steps"]))
    assert len(golden["losses"]) == golden["steps"] == 10
    trace = []
    for row in rows:
        assert (row["world_size"], row["dp_size"], row["usp_size"]) == (8, 1, 8)
        phases = ["generator", "critic"] if row["step"] % 5 == 0 else ["critic"]
        assert row["step_metadata"]["phase_order"] == phases
        losses = {"step": row["step"]}
        for phase in phases:
            loss = float(row["metrics"][f"{phase}/loss"])
            assert math.isfinite(loss), f"step {row['step']}: non-finite {phase} loss"
            losses[f"{phase}/loss"] = loss.hex()
        trace.append(losses)
    print("WAN_A14B_DMD_FP32_SP8_LOSS_TRACE=" + json.dumps(trace), flush=True)
    for actual, expected in zip(trace, golden["losses"], strict=True):
        assert actual.keys() == expected.keys() and actual["step"] == expected["step"]
        for field in actual.keys() - {"step"}:
            assert actual[field] == expected[field], (
                f"step {actual['step']} {field} bitwise mismatch: "
                f"actual={actual[field]}, expected={expected[field]}"
            )
    return trace


def test_dmd_loss_trace_rejects_one_ulp_drift(tmp_path):
    golden = _golden()
    rows = [
        {
            "event": "step",
            "step": losses["step"],
            "world_size": 8,
            "dp_size": 1,
            "usp_size": 8,
            "step_metadata": {
                "phase_order": ["generator", "critic"] if losses["step"] % 5 == 0 else ["critic"]
            },
            "metrics": {
                key: float.fromhex(value) for key, value in losses.items() if key != "step"
            },
        }
        for losses in golden["losses"]
    ]
    path = tmp_path / "train_steps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert _check_training(path, golden) == golden["losses"]
    rows[0]["metrics"]["generator/loss"] = math.nextafter(
        rows[0]["metrics"]["generator/loss"], math.inf
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError, match="step 0 generator/loss bitwise mismatch"):
        _check_training(path, golden)


@pytest.mark.gpu(gpus=8)
def test_wan_a14b_dmd_fp32_sp8(tmp_path, torchrun):
    golden = _golden()
    environment = golden["environment"]
    assert "B300" in torch.cuda.get_device_name(0)
    assert str(torch.__version__) == environment["torch"]
    assert torch.version.cuda == environment["cuda"]
    assert version("flash-attn-4") == environment["flash_attention_4"]
    model = _asset("WAN_DMD_A14B_MODEL")
    for expert in ("transformer", "transformer_2"):
        config = json.loads((model / expert / "config.json").read_text())
        assert config["num_layers"] == environment["layers"] == 40
    cache = tmp_path / "cache"
    _prepare_cache(_asset("WAN_DMD_A14B_CACHE"), cache, golden)
    output = tmp_path / "train"
    weights = {
        "generator": _asset("WAN_DMD_A14B_GENERATOR"),
        "real_score": _asset("WAN_DMD_A14B_SCORE"),
        "fake_score": _asset("WAN_DMD_A14B_SCORE"),
    }
    model_overrides = []
    for role, checkpoint in weights.items():
        model_overrides.extend(
            [
                f"training.models.{role}.pretrained_path={model}",
                f"training.models.{role}.init_weights={checkpoint}",
            ]
        )
    # RNG follows Taichu; shared-text computation keeps the native implementation.
    # Record and verify with this same entrypoint, without replay or model hooks.
    torchrun(
        "--module",
        "vforge.tasks.train",
        "experiment=wan22_a14b_dmd",
        *model_overrides,
        f"data.sources=[{{name:dmd_ci,cache_dir:{cache},weight:1.0}}]",
        "data.shuffle=true",
        f"data.sampler_seed={golden['sampler_seed']}",
        "data.batch_size=1",
        "data.num_workers=0",
        "latent_cache.mode=offline",
        f"runtime.output_dir={output}",
        f"runtime.max_steps={golden['steps']}",
        f"runtime.seed={environment['seed']}",
        "runtime.deterministic=true",
        "parallel.dp_size=1",
        "parallel.usp_size=8",
        "parallel.gradient_checkpointing=true",
        "attention.backend=flash_attention_4",
        "attention_offload.enable=false",
        "checkpoint.enabled=false",
        "resume.path=none",
        timeout=3600,
    )
    _check_training(output / "train_steps.jsonl", golden)
