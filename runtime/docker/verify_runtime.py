"""Verify the selected locked runtime without requiring a GPU during image builds."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import platform
import tomllib
from pathlib import Path

import torch
from packaging.requirements import Requirement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--python-version", required=True)
    args = parser.parse_args()
    if platform.python_version() != args.python_version:
        raise RuntimeError(
            f"Expected Python {args.python_version}, got {platform.python_version()}"
        )
    config = tomllib.loads(Path("pyproject.toml").read_text())
    for entry in config["dependency-groups"][args.group]:
        requirement = Requirement(entry)
        if metadata.version(requirement.name) not in requirement.specifier:
            raise RuntimeError(f"Installed runtime does not satisfy {requirement}")
    if torch.version.cuda != "13.0" or not str(torch.__version__).endswith("+cu130"):
        raise RuntimeError(f"Expected a cu130 wheel, got torch {torch.__version__}")
    if importlib.util.find_spec("vforge") is not None:
        raise RuntimeError("Dependency images must not contain vforge source")
    try:
        metadata.distribution("vforge")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Dependency images must not install vforge")
    from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    for operation in (flash_attn_func, flash_attn_varlen_func, _flash_attn_fwd, _flash_attn_bwd):
        if not callable(operation):
            raise RuntimeError("FlashAttention-4 runtime is incomplete")
    runtime = {
        "image_kind": "dependencies",
        "vforge_installed": False,
        "group": args.group,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": metadata.version("torchvision"),
        "flash-attn-4": metadata.version("flash-attn-4"),
        "cuda": torch.version.cuda,
    }
    Path("/opt/vforge-image/runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime), flush=True)


if __name__ == "__main__":
    main()
