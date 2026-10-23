"""CT 5B Ref and A14B GPU regressions; asset environment variables are listed below."""

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
from vforge.tasks.translate import build_runtime, eval_request
from vforge.tasks.wan_taichu.eval_task.gridnormal.variants import resolve_gridnormal_variants

ROOT = Path(__file__).resolve().parents[3]
CASES = {
    "5b_ref": {
        "user": "wan-taichu-5b-ref-prod",
        "experiment": "wan_taichu_wan22_ti2v_5b_ref_prod",
        "evaluation": "wan_taichu_5b_ref_prod",
        "model_env": "WAN_CT_5B_MODEL",
        "train_checkpoint_env": "WAN_CT_5B_TRAIN_CHECKPOINT",
        "eval_checkpoint_env": "WAN_CT_5B_EVAL_CHECKPOINT",
        "cache_envs": ["WAN_CT_5B_CACHE_5S", "WAN_CT_5B_CACHE_20S", "WAN_CT_5B_CACHE_60S"],
        "index_envs": ["WAN_CT_5B_EVAL_5S_PKL", "WAN_CT_5B_EVAL_20S_PKL", "WAN_CT_5B_EVAL_60S_PKL"],
        "usp": 4,
    },
    "a14b": {
        "user": "wan-taichu-a14b-prod",
        "experiment": "wan_taichu_wan22_a14b_prod",
        "evaluation": "wan_taichu_a14b_20s_prod",
        "model_env": "WAN_CT_A14B_MODEL",
        "train_checkpoint_env": "WAN_CT_A14B_TRAIN_CHECKPOINT",
        "eval_checkpoint_env": "WAN_CT_A14B_EVAL_CHECKPOINT",
        "cache_envs": ["WAN_CT_A14B_CACHE_5S", "WAN_CT_A14B_CACHE_20S"],
        "index_envs": ["WAN_CT_A14B_EVAL_20S_PKL"],
        "usp": 8,
    },
}


def _asset(variable: str) -> str:
    # Resolve only when running a case, so CPU collection needs no asset environment.
    value = os.environ.get(variable, "").strip()
    if not value:
        raise RuntimeError(f"set {variable} to the existing model, checkpoint, cache or PKL path")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{variable}={path}")
    return str(path)


def _train_config(name: str, output: Path) -> DictConfig:
    case = CASES[name]
    # Load the standard train entry, then the existing examples/configs/user recipe.
    ConfigStore.instance().store(
        name="wan_ct_train_test",
        node={"defaults": ["/train", {"/user": case["user"]}]},
    )
    cfg = compose_standalone_config(
        "wan_ct_train_test",
        overrides=[f"hydra.searchpath=[file://{ROOT / 'examples/configs'}]"],
    )

    # GPU-test settings; model, optimizer and data-mixture settings come from the recipe.
    with open_dict(cfg):
        cfg.runtime.output_dir = str(output)
        cfg.runtime.seed = 42
        cfg.parallel.usp_size = case["usp"]
        cfg.attention.backend = "flash_attention_4"
        cfg.attention.allow_fallback = False
        configure_training(cfg)
        # Only the 5B warm-start permits its missing Ref projection to be initialized.
        cfg.runtime.weights_strict = name == "a14b"
        cfg.logging = {"wandb_enabled": False}

    cfg.model.pretrained_path = str(output.parent / "assets/train.pt")
    cfg.model.extras.condition_pretrained_path = str(output.parent / "assets/model")
    for source, variable in zip(cfg.data.sources, case["cache_envs"], strict=True):
        source.cache_dir = _asset(variable)
    return cfg


def _eval_config(name: str, output: Path) -> DictConfig:
    case = CASES[name]
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            f"experiment={case['experiment']}",
            f"evaluation={case['evaluation']}",
        ],
    )

    # Keep every source, spatial resolution and CFG variant; shorten video export.
    # The training recipe still exercises its long latent sequences.
    with open_dict(cfg):
        cfg.runtime.output_dir = str(output)
        cfg.runtime.seed = 42
        cfg.runtime.weights_strict = True
        cfg.runtime.deterministic = True
    cfg.parallel.usp_size = 4
    cfg.attention.backend = "flash_attention_4"
    cfg.attention.allow_fallback = False
    cfg.latent_cache.mode = "online"
    cfg.evaluation.data_source = "validation"
    cfg.evaluation.output_dir = str(output)
    cfg.evaluation.num_inference_steps = 2
    cfg.evaluation.seed = 42
    cfg.evaluation.skip_existing = False
    cfg.evaluation.num_samples = len(case["index_envs"])
    cfg.evaluation.sample_selection = "evenly_spaced"

    cfg.model.pretrained_path = str(output.parent / "assets/model")
    cfg.evaluation.checkpoint_format = "pt"
    cfg.evaluation.checkpoint_path = str(output.parent / "assets/eval.pt")
    for source, variable in zip(cfg.evaluation.sources, case["index_envs"], strict=True):
        source.index_files = [_asset(variable)]
        source.frames = 17
    # Optional mount remapping for media paths stored inside the PKLs.
    cfg.evaluation.path_replacements = [
        ["/efs", os.environ.get("WAN_EFS_ROOT", "/efs")],
        ["/threed-code", os.environ.get("WAN_CODE_ROOT", "/threed-code")],
    ]
    return cfg


def _video_shape(path: Path) -> tuple[int, int, int]:
    import av

    count, low, high = 0, 255, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        height, width = stream.height, stream.width
        for frame in container.decode(stream):
            assert (frame.height, frame.width) == (height, width)
            pixels = frame.to_ndarray(format="rgb24")
            low, high = min(low, int(pixels.min())), max(high, int(pixels.max()))
            count += 1
    assert count > 0 and high > low, f"empty or constant video: {path}"
    return count, height, width


def _check_eval(cfg: DictConfig, eval_dir: Path) -> None:
    request = eval_request(build_runtime(cfg, world_size=4).config)
    manifest = json.loads((eval_dir / "eval_manifest.json").read_text())
    assert manifest["usp_size"] == 4 and manifest["dp_size"] == 1
    assert manifest["runtime"]["resolved_backend"] == "flash_attention_4"
    assert len(manifest["written"]) == request.num_samples
    (source,) = manifest["checkpoint"]["sources"]
    assert source["path"] == request.checkpoint_path and source["weights_strict"]
    assert source["format"] == "pt"
    for written in manifest["written"]:
        assert Path(written).is_file()
        sidecar = json.loads(Path(written + ".json").read_text())
        assert sidecar["provenance"]["checkpoint"] == manifest["checkpoint"]
    expected = sorted((source.frames, source.height, source.width) for source in request.sources)
    for variant in resolve_gridnormal_variants(request.extras, request.gridnormal_variants):
        videos = list((eval_dir / variant.name).glob("*.mp4"))
        assert len(videos) == request.num_samples
        assert sorted(_video_shape(path) for path in videos) == expected


@pytest.mark.gpu(gpus=8)
@pytest.mark.parametrize("name", CASES)
def test_ct_train_and_eval(name, tmp_path, torchrun, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    with tempfile.TemporaryDirectory(prefix=f"ct-{name}-", dir=tmp_path) as scratch:
        root = Path(scratch)
        train_dir, eval_dir = root / "train", root / "eval"
        case = CASES[name]
        reduced_assets(
            _asset(case["model_env"]),
            {
                "train": _asset(case["train_checkpoint_env"]),
                "eval": _asset(case["eval_checkpoint_env"]),
            },
            root / "assets",
        )
        _train_config(name, train_dir)  # Fail on missing assets before launching ranks.
        cfg = _eval_config(name, eval_dir)
        torchrun(__file__, name, "train", train_dir, timeout=2400)
        check_loss(f"ct_{name}", train_dir)
        torchrun(__file__, name, "eval", eval_dir, nproc=4, timeout=2700)
        _check_eval(cfg, eval_dir)


if __name__ == "__main__":
    name, mode, output = sys.argv[1:]
    if mode == "eval":
        from vforge.tasks.eval import run_eval

        run_eval(_eval_config(name, Path(output)))
    else:
        run_training(_train_config(name, Path(output)))
