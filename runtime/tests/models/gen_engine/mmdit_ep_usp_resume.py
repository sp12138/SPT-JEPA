"""Prepare inputs and compare native Trainer full checkpoints across a restart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from mmdit_ep_usp_worker import small_config
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from vforge.tasks.gen_engine.synthetic import generate_synthetic_manifest
from vforge.training.checkpoint import read_manifest


def prepare(directory, *, samples_per_kind=4):
    directory.mkdir(parents=True, exist_ok=False)
    config_path = directory / "transformer.json"
    config_path.write_text(json.dumps(small_config(), indent=2) + "\n")
    model_root = Path(__file__).resolve().parents[3] / "vforge/models/gen_engine"
    codecs = {
        kind: model_root / f"{kind}_vae/latent_codec_configs/minimax-h3.json"
        for kind in ("video", "audio")
    }
    manifest = generate_synthetic_manifest(
        directory / "data",
        transformer_config_path=config_path,
        video_latent_codec_config_path=codecs["video"],
        audio_latent_codec_config_path=codecs["audio"],
        samples_per_kind=samples_per_kind,
        seed=42,
    )
    # This stage exercises short video, audio, AV, and references. The 12,736
    # token full-size training gate belongs to the following review stage.
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    short = [row for row in rows if not row["sample_id"].startswith("synthetic-long_video-")]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in short))
    (directory / "data.sha256").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n")
    overrides = [
        "experiment=gen_engine",
        f"model.extras.transformer_config_path={config_path}",
        f"model.extras.video_latent_codec_config_path={codecs['video']}",
        f"model.extras.audio_latent_codec_config_path={codecs['audio']}",
        f"data.train_manifest={manifest}",
        "data.batch_size=1",
        "data.num_workers=0",
        "runtime.max_steps=10",
        "runtime.seed=42",
        "runtime.deterministic=true",
        "runtime.precision=bf16",
        "parallel.gradient_checkpointing=true",
        "attention.backend=sdpa_math",
        "attention.allow_fallback=false",
        "checkpoint.every_n_steps=5",
        "logging.wandb_enabled=false",
    ]
    (directory / "overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")


def assert_equal(a, b, path):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            assert_equal(left, right, f"{path}.{index}")
    else:
        assert a == b, (path, a, b)


def compare(directory):
    checkpoints = [directory / run / "checkpoints/00010.dcp" for run in ("continuous", "resumed")]
    manifests = [read_manifest(str(path)) for path in checkpoints]
    for manifest in manifests:
        assert manifest.completed_steps == 10
        assert all(manifest.components_present[key] for key in ("weights", "optimizer", "trainer"))
    assert manifests[0].topology == manifests[1].topology
    with tempfile.TemporaryDirectory() as temporary:
        for component in ("weights", "optimizer"):
            values = []
            for index, (path, manifest) in enumerate(zip(checkpoints, manifests, strict=True)):
                output = Path(temporary) / f"{index}.pt"
                dcp_to_torch_save(str(path / manifest.components[component]), output)
                values.append(torch.load(output, map_location="cpu", weights_only=False))
            assert_equal(values[0], values[1], component)
            if component == "optimizer":
                expert_states = {
                    name: state
                    for name, state in values[0]["optimizer"]["state"].items()
                    if ".main_bank." in name
                }
                assert expert_states, "checkpoint has no main-expert optimizer state"
                active = {}
                for name, state in expert_states.items():
                    moment = state["exp_avg"]
                    assert torch.isfinite(moment).all(), name
                    active[name] = int(moment.flatten(1).ne(0).any(dim=1).sum())
                    assert active[name] > 0, f"main experts received no gradient: {name}"
                print("MMDIT_ACTIVE_EXPERTS " + json.dumps(active, sort_keys=True), flush=True)
            del values
    for rank in range(manifests[0].world_size):
        values = [
            torch.load(
                path / manifest.components["trainer"] / f"rank-{rank:05d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for path, manifest in zip(checkpoints, manifests, strict=True)
        ]
        assert_equal(values[0], values[1], f"trainer.rank{rank}")
    traces = [
        [
            json.loads(line)
            for line in (directory / run / "train_steps.jsonl").read_text().splitlines()
        ]
        for run in ("continuous", "resumed")
    ]
    assert len(traces[0]) == 10 and len(traces[1]) == 5
    for expected, actual in zip(traces[0][5:], traces[1], strict=True):
        for key in ("step", "loss", "loss_backward", "metrics"):
            assert_equal(expected[key], actual[key], f"step{actual['step']}.{key}")
    print(
        "MMDIT_RESUME: model, optimizer, routing bias, scheduler, loader and RNG match exactly",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "compare"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--usp", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--samples-per-kind", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory, samples_per_kind=args.samples_per_kind)
    elif args.action == "compare":
        compare(args.directory)
    else:
        from vforge.tasks.common import compose_standalone_config
        from vforge.tasks.train import run_train

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        overrides = json.loads((args.directory / "overrides.json").read_text())
        run = "resumed" if args.resume else "continuous"
        overrides += [
            f"parallel.ep_size={args.ep}",
            f"parallel.usp_size={args.usp}",
            f"runtime.output_dir={args.directory / run}",
        ]
        if args.resume:
            overrides += [f"resume.path={args.directory / 'continuous/checkpoints/00005.dcp'}"]
        run_train(compose_standalone_config("train", overrides))


if __name__ == "__main__":
    main()
