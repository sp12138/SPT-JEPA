"""Hand a preserved full Lightning state to the original frozen training entry."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    for key in ("parent", "config", "train-source", "train-pythonpath"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    parent = Path(args.parent)
    config = json.loads(Path(args.config).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    original, destination = parent / "last.ckpt", output / "last.ckpt"
    if not (output / "resume_manifest.json").exists():
        state = torch.load(original, map_location="cpu", weights_only=False)
        actual = int(state["global_step"])
        if actual != args.expected_step or "optimizer_states" not in state or "lr_schedulers" not in state:
            raise RuntimeError(f"Parent is not the expected full state: {actual}")
        del state
        if destination.exists():
            raise RuntimeError("Unidentified destination checkpoint already exists")
        destination.hardlink_to(original)
        manifest = dict(parent=str(parent), checkpoint=str(original), completed_steps=actual,
                        parent_checkpoint_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                        train_source=args.train_source, train_pythonpath=args.train_pythonpath,
                        scope="Full optimizer/scheduler/RNG/data cursor continuation; fresh output directory; original training source retained.")
        (output / "resume_manifest.json").write_text(json.dumps(manifest, indent=2))
    env = {**os.environ, "PYTHONPATH": args.train_pythonpath, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [os.sys.executable, "-B", "-u", "-m", "as_jepa_lewm.train", args.config, "--resume"]
    raise SystemExit(subprocess.run(command, cwd=args.train_source, env=env).returncode)


if __name__ == "__main__":
    main()
