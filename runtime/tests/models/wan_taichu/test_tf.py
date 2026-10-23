"""A14B 99-latent/SWA TF training on eight GPUs and evaluation from existing weights."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from ct_tf_helpers import check_loss, configure_training, reduced_assets, run_training
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, open_dict

from vforge.tasks.common import compose_standalone_config

ROOT = Path(__file__).resolve().parents[3]


def _asset(variable: str) -> str:
    # Asset lookup happens during execution, never during CPU test collection.
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(f"set {variable} to the existing model, checkpoint or cache path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{variable}={path}")
    return str(path)


def _train_config(output: Path) -> DictConfig:
    # Reuse the 99-latent/SWA recipe, including its model, optimizer and data settings.
    ConfigStore.instance().store(
        name="wan_tf_train_test",
        node={
            "defaults": [
                "/train",
                {"/user": "wan-taichu-a14b-causal-teacher-l99-swa-usp16-dp8"},
            ],
        },
    )
    cfg = compose_standalone_config(
        "wan_tf_train_test",
        overrides=[f"hydra.searchpath=[file://{ROOT / 'examples/configs'}]"],
    )

    # Run three deterministic steps without checkpoint I/O.
    # The recipe's original USP16 x DP8 topology is not exercised here.
    with open_dict(cfg):
        cfg.runtime.output_dir = str(output)
        cfg.runtime.seed = 42
        cfg.parallel.usp_size = 8
        cfg.attention.backend = "flash_attention_4"
        cfg.attention.allow_fallback = False
        configure_training(cfg)
        cfg.runtime.weights_strict = True
        cfg.logging.wandb_enabled = False

    cfg.model.pretrained_path = str(output.parent / "assets/train.pt")
    cfg.model.extras.condition_pretrained_path = str(output.parent / "assets/model")
    cfg.data.sources[0].cache_dir = _asset("WAN_TF_A14B_CACHE")
    return cfg


def _eval_config(output: Path) -> DictConfig:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_a14b_causal_teacher",
            "evaluation=wan_taichu_a14b_causal_teacher_99lat_swa_train",
        ],
    )

    # Keep the preset's 99-latent/SWA rollout; evaluate one sample in two steps.
    with open_dict(cfg):
        cfg.runtime.output_dir = str(output)
        cfg.runtime.seed = 42
        cfg.runtime.weights_strict = True
        cfg.runtime.deterministic = True
    cfg.parallel.usp_size = 2
    cfg.attention.backend = "flash_attention_4"
    cfg.attention.allow_fallback = False
    cfg.evaluation.output_dir = str(output)
    cfg.evaluation.num_inference_steps = 2
    cfg.evaluation.seed = 42
    cfg.evaluation.skip_existing = False
    cfg.evaluation.num_samples = 1
    cfg.evaluation.sample_selection = "first"

    cfg.model.pretrained_path = str(output.parent / "assets/model")
    cfg.evaluation.sources[0].cache_dir = _asset("WAN_TF_A14B_CACHE")
    cfg.evaluation.checkpoint_format = "pt"
    cfg.evaluation.checkpoint_path = str(output.parent / "assets/eval.pt")
    return cfg


def _check_video(path: str) -> None:
    import av

    count, low, high = 0, 255, 0
    with av.open(path) as container:
        for frame in container.decode(video=0):
            assert (frame.height, frame.width) == (720, 1280)
            pixels = frame.to_ndarray(format="rgb24")
            low, high = min(low, int(pixels.min())), max(high, int(pixels.max()))
            count += 1
    # 99 latent frames with temporal VAE compression 4 -> 393 video frames.
    assert count == 393 and high > low, f"wrong frame count or constant video: {path}"


def _check_eval(cfg: DictConfig, eval_dir: Path) -> None:
    manifest = json.loads((eval_dir / "eval_manifest.json").read_text())
    assert manifest["usp_size"] == 2 and manifest["dp_size"] == 1
    assert manifest["runtime"]["resolved_backend"] == "flash_attention_4"
    (loaded,) = manifest["checkpoint"]["sources"]
    assert loaded["path"] == cfg.evaluation.checkpoint_path and loaded["weights_strict"]
    assert loaded["format"] == cfg.evaluation.checkpoint_format
    assert len(manifest["written"]) == 1
    (video,) = manifest["written"]
    sidecar = json.loads(Path(video + ".json").read_text())
    assert sidecar["provenance"]["checkpoint"] == manifest["checkpoint"]
    _check_video(video)


@pytest.mark.gpu(gpus=8)
def test_tf_train_and_eval(tmp_path, torchrun, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    with tempfile.TemporaryDirectory(prefix="tf-a14b-", dir=tmp_path) as scratch:
        root = Path(scratch)
        train_dir, eval_dir = root / "train", root / "eval"
        reduced_assets(
            _asset("WAN_TF_A14B_MODEL"),
            {
                "train": _asset("WAN_TF_A14B_TRAIN_CHECKPOINT"),
                "eval": _asset("WAN_TF_A14B_EVAL_CHECKPOINT"),
            },
            root / "assets",
        )
        _train_config(train_dir)
        cfg = _eval_config(eval_dir)
        torchrun(__file__, "train", train_dir, timeout=2400)
        check_loss("tf_a14b", train_dir)
        torchrun(__file__, "eval", eval_dir, nproc=2, timeout=2700)
        _check_eval(cfg, eval_dir)


if __name__ == "__main__":
    mode, output = sys.argv[1:]
    if mode == "eval":
        from vforge.tasks.eval import run_eval

        run_eval(_eval_config(Path(output)))
    else:
        run_training(_train_config(Path(output)))
