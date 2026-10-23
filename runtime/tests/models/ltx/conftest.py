"""Share byte-verified DMD assets across training and inference on the node SSD."""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def dmd_assets():
    import torch

    assert os.environ.get("CI_MACHINE") == "b300", "DMD references require B300"
    assert "B300" in torch.cuda.get_device_name(0)
    root = Path("/local-ssd")
    assert root.is_dir(), "LTX GPU tests require /local-ssd for weights and checkpoints"
    golden = json.loads(Path(__file__).with_name("dmd_b300.json").read_text())
    with tempfile.TemporaryDirectory(prefix="vforge-dmd-assets-", dir=root) as directory:
        staged = {}

        def load(*names, long_tuning=False):
            reference = golden
            if long_tuning:
                reference = json.loads(
                    Path(__file__).with_name("long_tuning_b300.json").read_text()
                )
                assert reference["checkpoints"]["base"] == golden["checkpoints"]["base"]
            selected = {}
            for name in names:
                variant = "long_tuning" if long_tuning and name in {"generator", "cache"} else "dmd"
                key = (variant, name)
                if key in staged:
                    selected[name] = staged[key]
                    continue
                variable = f"LTX23_{variant.upper()}_CI_{name.upper()}"
                value = os.environ.get(variable, "").strip()
                assert value, f"set {variable} to the existing DMD asset"
                source = Path(value).expanduser().resolve(strict=True)
                target = Path(directory) / f"{variant}-{name}"
                started = time.monotonic()
                print(f"Staging DMD {name}: {source} -> {target}", flush=True)
                if name == "cache":
                    for kind in ("latents", "conditions"):
                        (target / kind).mkdir(parents=True)
                        for index, sample in enumerate(reference["samples"]):
                            path = target / kind / f"{index:04d}.pt"
                            shutil.copyfile(source / kind / sample["path"], path)
                            digest = hashlib.sha256(path.read_bytes()).hexdigest()
                            assert digest == sample["sha256"][kind], f"changed DMD input: {path}"
                elif name == "gemma":
                    from vforge.utils.node_staging import NodeStagingConfig, stage_on_node

                    result = stage_on_node(
                        str(source),
                        config=NodeStagingConfig(root=str(Path(directory) / "gemma")),
                        local_rank=0,
                    )
                    target = Path(result.destination)
                else:
                    target = target.with_suffix(".safetensors")
                    shutil.copyfile(source, target)
                    assert target.stat().st_size == reference["checkpoints"][name]["bytes"]
                    with target.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    assert digest == reference["checkpoints"][name]["sha256"], f"changed DMD {name}"
                staged[key] = target
                print(
                    f"Staged DMD {name}: {target} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                selected[name] = target
            return selected

        yield load
