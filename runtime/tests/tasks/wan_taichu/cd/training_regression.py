"""Small, fixed real-cache fixtures for Native CD; no production model patches."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from importlib.metadata import version
from pathlib import Path

import torch

STEPS = 20
SEED = 123
CASES = {"5b": ("wan22_ti2v_5b_cd", 30, 7), "a14b": ("wan22_a14b_cd", 2, 6)}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_assets(case: str, output: Path) -> dict:
    """Copy and verify the independently maintained EFS fixture package once."""
    value = os.environ.get("WAN_CD_CI_ROOT", "").strip()
    if not value:
        raise RuntimeError("set WAN_CD_CI_ROOT to the published taichu-cd asset directory")
    source = Path(value).expanduser().resolve(strict=True) / case
    manifest = json.loads((source / "manifest.json").read_text())
    _, layers, frames = CASES[case]
    assert (
        manifest["schema"],
        manifest["case"],
        manifest["layers_per_expert"],
        manifest["latent_frames"],
        manifest["spatial_size"],
    ) == (1, case, layers, frames, [16, 16])
    expected = {
        "weights.pt",
        "model/transformer/config.json",
        "cache/train/0000/00000000.pt",
        "cache/train/0000/00000001.pt",
    }
    if case == "a14b":
        expected.add("model/transformer_2/config.json")
    assert set(manifest["files"]) == expected, "incomplete or unexpected CD fixture files"
    if case == "5b":
        assert manifest["files"]["weights.pt"] == manifest["sources"]["checkpoint_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    for relative, checksum in manifest["files"].items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        assert digest(target) == checksum, f"changed CD input: {relative}"
    return manifest


def train_config(case: str, assets: Path, output: Path):
    from vforge.tasks.common import compose_standalone_config

    experiment, _, frames = CASES[case]
    cfg = compose_standalone_config(
        "train",
        [
            f"experiment={experiment}",
            f"runtime.output_dir={output}",
            f"runtime.max_steps={STEPS}",
            f"runtime.seed={SEED}",
            "runtime.deterministic=true",
            "parallel.usp_size=8",
            "parallel.dp_size=1",
            "checkpoint.enabled=false",
            "checkpoint.save_final=false",
            "resume.path=none",
            "scheduler.name=none",
            "scheduler.warmup_steps=0",
            "optimizers.student.eps=1e-8",
            f"data.sources.0.min_latents={frames}",
            f"data.sources.0.max_latents={frames}",
            f"data.sources.0.cache_dir={assets / 'cache'}",
        ],
    )
    for role in cfg.training.models.values():
        role.pretrained_path = str(assets / "model")
        role.init_weights = str(assets / "weights.pt")
    return cfg


def environment() -> dict:
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "flash_attention_4": version("flash-attn-4"),
    }


def read_trace(case: str, output: Path) -> list[dict]:
    records = [json.loads(line) for line in (output / "train_steps.jsonl").read_text().splitlines()]
    records = [row for row in records if row.get("event") == "step"]
    assert [row["step"] for row in records] == list(range(STEPS))
    trace = []
    for row in records:
        step = row["step"]
        assert (row["world_size"], row["usp_size"], row["dp_size"]) == (8, 8, 1)
        assert row["step_metadata"]["phase_order"] == ["student"]
        assert row["step_metadata"]["phase_learning_rates"]["student"] == {
            "before": 5e-6,
            "after": 5e-6,
        }
        metrics = row["metrics"]
        loss = metrics["student/loss"]
        norm = metrics["student/diag/grad_norm_post_reduce"]
        assert math.isfinite(loss) and math.isfinite(norm) and norm > 0
        if case == "a14b":
            assert metrics["student/high_noise_expert"] == (step % 2 == 0)
        trace.append({"step": step, "loss": float(loss).hex()})
    assert not (output / "checkpoints").exists()
    return trace


def verify_trace(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected) == STEPS
    for step, (observed, reference) in enumerate(zip(actual, expected, strict=True)):
        assert observed == reference, f"CD step {step} bitwise mismatch: {observed} != {reference}"
