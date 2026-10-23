"""Reduced Wan assets and deterministic loss checks shared by CT and TF tests."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

LAYERS = 2
SEED = 42


def check_loss(case: str, train_dir: Path) -> None:
    """Compare sample order and losses against the B300 reference run."""
    records = [
        json.loads(line) for line in (train_dir / "train_steps.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [0, 1, 2]
    assert records[1]["learning_rate_before_step"] > 0
    for previous, following in zip(records[:-1], records[1:], strict=True):
        assert following["learning_rate_before_step"] == previous["learning_rate_after_step"]
    golden = json.loads(Path(__file__).with_name("ct_tf_b300.json").read_text())
    environment = golden["environment"]
    assert "B300" in torch.cuda.get_device_name(0), "loss reference requires B300"
    assert str(torch.__version__) == environment["torch"]
    assert version("flash-attn-4") == environment["flash_attention_4"]
    assert torch.version.cuda == environment["cuda"]
    assert (environment["seed"], environment["layers"]) == (SEED, LAYERS)
    expected = golden["cases"][case]
    assert all(record["world_size"] == expected["world_size"] for record in records)
    assert all(record["usp_size"] == expected["usp_size"] for record in records)
    assert [record["loss"] for record in records] == pytest.approx(
        expected["loss"], rel=1e-5, abs=1e-6
    )
    assert [record["step_metadata"]["sample_ids"] for record in records] == expected["sample_ids"]


def reduced_assets(model: str, checkpoints: dict[str, str], output: Path) -> dict[str, str]:
    """Keep real model widths and encoders, with only the first two DiT blocks."""
    output.mkdir(parents=True)
    model_dir = output / "model"
    model_dir.mkdir()
    for child in Path(model).iterdir():
        if child.name in {"transformer", "transformer_2"}:
            target = model_dir / child.name
            target.mkdir()
            config = json.loads((child / "config.json").read_text())
            assert config["num_layers"] > LAYERS
            config["num_layers"] = LAYERS
            (target / "config.json").write_text(json.dumps(config))
        else:
            (model_dir / child.name).symlink_to(child, target_is_directory=child.is_dir())

    paths = {"model": str(model_dir)}
    block = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    for name, source in checkpoints.items():
        # mmap reads only selected tensor pages, avoiding full 5B/14B checkpoint copies.
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("state_dict", payload)
        selected = {}
        dropped = 0
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            match = block.search(key)
            if match and int(match[1]) >= LAYERS:
                dropped += 1
            else:
                selected[key] = tensor
        assert dropped and any(block.search(key) for key in selected)
        path = output / f"{name}.pt"
        torch.save(selected, path)
        paths[name] = str(path)
        del selected, state, payload
    return paths


def configure_training(cfg) -> None:
    cfg.runtime.max_steps = 3
    cfg.runtime.seed = SEED
    cfg.runtime.deterministic = True
    cfg.checkpoint.enabled = False
    cfg.resume.path = "none"


def run_training(cfg) -> None:
    """Run the normal entrypoint with deterministic component construction."""
    from vforge.tasks import train
    from vforge.utils.rng import seed_process

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Also seed component construction, which precedes Trainer.fit.
    seed_process(SEED)
    train.run_train(cfg)
