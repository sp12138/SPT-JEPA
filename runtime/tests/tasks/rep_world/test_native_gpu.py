"""Opt-in real Native CLI gate, with no source-checkout runtime dependency.

Set REP_WORLD_ASSET_ROOT and REP_WORLD_SMOKE_MANIFEST (videos >=160 frames).
Run with pytest --gpu; each case owns one visible GPU and a temporary output.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vforge.data.dataset import load_manifest_records

ROOT = Path(__file__).resolve().parents[3]
ASSETS = os.environ.get("REP_WORLD_ASSET_ROOT")
MANIFEST = os.environ.get("REP_WORLD_SMOKE_MANIFEST")


def launch(verb, arguments):
    completed = subprocess.run(
        [sys.executable, "-m", f"vforge.tasks.{verb}", *arguments],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout
    return completed.stdout


@pytest.mark.gpu(gpus=1)
@pytest.mark.skipif(
    not ASSETS or not MANIFEST, reason="Requires explicit published assets and video manifest"
)
@pytest.mark.parametrize(
    "experiment,checkpoint,statistics",
    [
        ("rep_world", "VideoRAE/vjepa21_3d.pth", "VideoRAE/solarwm_latent_stats_32clips.pt"),
        ("rep_world_1d", "VideoRAE/vjepa_rae.pth", "VideoRAE/solarwm_1d_latent_stats_32clips.pt"),
        ("rep_world_eupe", "V-RAE/vrae/vrae_eupe.pt", ""),
        ("rep_world_vjepa", "V-RAE/vrae/vrae_vjepa2.1.pt", ""),
    ],
)
@pytest.mark.parametrize("domain,patch", [("rgb", 2), ("rgb", 4), ("latent", 2)])
def test_real_native_train_dcp_and_rollout(
    tmp_path, experiment, checkpoint, statistics, domain, patch, mode="no_zprev", states=None
):
    import av

    # Materialize one explicit sample so each arm has a stable source and ID.
    original = load_manifest_records(MANIFEST)[0]
    media = original.get("rgb_path") or original.get("video") or original.get("video_path")
    assert media
    media = Path(media)
    if not media.is_absolute():
        media = Path(MANIFEST).resolve().parent / media
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(sample_id="gate", video=str(media.resolve()))))
    train_dir = tmp_path / "train"
    common = [
        f"experiment={experiment}",
        "attention.backend=sdpa",
        f"model.extras.codec.checkpoint={Path(ASSETS) / checkpoint}",
        f"model.extras.conditioner.input={domain}",
        f"model.extras.conditioner.temporal_patch={patch}",
        f"model.extras.flow.mode={mode}",
        f"evaluation.guidance_scale={1 if mode == 'pure_tf' else 2}",
        "logging.wandb_enabled=false",
    ]
    if states is not None:
        common.append(f"model.extras.states={states}")
    if statistics:
        common.append(f"model.extras.codec.statistics={Path(ASSETS) / statistics}")
    output = launch(
        "train",
        [
            *common,
            f"data.train_manifest={manifest}",
            "data.batch_size=1",
            "data.shuffle=false",
            "model.extras.random_clip=false",
            "runtime.max_steps=2",
            f"runtime.output_dir={train_dir}",
            "checkpoint.async_save=false",
        ],
    )
    rows = [
        json.loads(line.split("VFORGE_TRAIN ", 1)[1])
        for line in output.splitlines()
        if line.startswith("VFORGE_TRAIN ")
    ]
    assert [row["step"] for row in rows] == [0, 1]
    assert all(math.isfinite(row["loss"]) for row in rows)
    assert rows[-1]["metrics"]["diag/grad_norm_post_reduce"] > 0
    checkpoint_dir = train_dir / "checkpoints/00002.dcp"
    assert checkpoint_dir.is_dir()
    eval_dir = tmp_path / "eval"
    launch(
        "eval",
        [
            *common,
            f"evaluation.checkpoint_path={checkpoint_dir}",
            "evaluation.data_source=inference",
            f"evaluation.manifest={manifest}",
            "evaluation.num_inference_steps=2",
            "model.extras.rollout.total_states=3",
            f"runtime.output_dir={eval_dir}",
        ],
    )
    videos = list((eval_dir / "samples").glob("*.mp4"))
    assert len(videos) == 1
    with av.open(str(videos[0])) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == (48 if experiment in {"rep_world", "rep_world_1d"} else 12)
    assert all(frame.width == 256 and frame.height == 256 for frame in frames)
    sidecar = json.loads(Path(str(videos[0]) + ".json").read_text())
    assert sidecar["metadata"]["condition_input"] == (None if mode == "pure_tf" else domain)
    assert sidecar["metadata"]["cfg_scale"] == (1 if mode == "pure_tf" else 2)


@pytest.mark.gpu(gpus=1)
@pytest.mark.skipif(
    not ASSETS or not MANIFEST, reason="Requires explicit published assets and video manifest"
)
@pytest.mark.parametrize("mode", ["zprev", "pure_tf"])
@pytest.mark.parametrize(
    "experiment,checkpoint,statistics",
    [
        ("rep_world", "VideoRAE/vjepa21_3d.pth", "VideoRAE/solarwm_latent_stats_32clips.pt"),
        ("rep_world_1d", "VideoRAE/vjepa_rae.pth", "VideoRAE/solarwm_1d_latent_stats_32clips.pt"),
        ("rep_world_eupe", "V-RAE/vrae/vrae_eupe.pt", ""),
        ("rep_world_vjepa", "V-RAE/vrae/vrae_vjepa2.1.pt", ""),
    ],
)
def test_context_modes_train_dcp_and_rollout(tmp_path, experiment, checkpoint, statistics, mode):
    # Full-size trainable towers/codecs with a short sequence exercise the
    # previous-state mask and the pure-TF training/KV inference paths.
    test_real_native_train_dcp_and_rollout(
        tmp_path, experiment, checkpoint, statistics, "rgb", 2, mode=mode, states=3
    )
