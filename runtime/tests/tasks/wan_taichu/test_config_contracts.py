from __future__ import annotations

from pathlib import Path

import pytest

from vforge.tasks.common import compose_standalone_config
from vforge.tasks.translate import build_runtime, eval_request, train_request
from vforge.tasks.wan_taichu.train import _resolve_request

ROOT = Path(__file__).resolve().parents[3]
USER_CONFIG_DIR = ROOT / "examples" / "configs"


@pytest.mark.parametrize(
    ("experiment", "variant", "shift"),
    (
        ("wan_taichu_wan21_i2v_14b", "wan21_i2v_14b", 1.0),
        ("wan_taichu_wan21_i2v_14b_audio", "wan21_i2v_14b_audio", 5.0),
        ("wan_taichu_wan22_a14b", "wan22_a14b", 3.0),
        (
            "wan_taichu_wan22_a14b_causal_teacher",
            "wan22_a14b_causal_teacher",
            3.0,
        ),
        ("wan_taichu_wan22_ti2v_5b", "wan22_ti2v_5b", 5.0),
        ("wan_taichu_wan22_ti2v_5b_ref", "wan22_ti2v_5b_ref", 5.0),
        (
            "wan_taichu_wan22_ti2v_5b_causal_teacher",
            "wan22_ti2v_5b_causal_teacher",
            5.0,
        ),
    ),
)
def test_train_experiments_select_one_public_model(
    experiment: str,
    variant: str,
    shift: float,
) -> None:
    cfg = compose_standalone_config(
        "train",
        overrides=[f"experiment={experiment}", "checkpoint.enabled=false"],
    )
    runtime = build_runtime(cfg, world_size=1)
    request = train_request(runtime.config)

    assert runtime.config.model.name == "wan_taichu"
    assert request.extras["variant"] == variant
    # Defaults live in variants.py rather than being repeated in the preset.
    assert "noise_shift" not in request.extras
    _, _, resolved_extras = _resolve_request(request)
    assert resolved_extras["noise_shift"] == shift
    assert runtime.trainer.checkpoint.enabled is False


def test_train_request_carries_tensor_parallel_size_to_preflight() -> None:
    cfg = compose_standalone_config(
        "train",
        overrides=[
            "experiment=wan_taichu_wan21_i2v_14b",
            "parallel.tp_size=2",
            "checkpoint.enabled=false",
        ],
    )
    request = train_request(build_runtime(cfg, world_size=2).config)

    assert request.tp_size == 2


def test_ref_experiment_does_not_apply_pixel_augmentation_to_offline_cache() -> None:
    cfg = compose_standalone_config(
        "train",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref",
            "checkpoint.enabled=false",
        ],
    )
    request = train_request(build_runtime(cfg, world_size=1).config)

    assert request.extras["condition_streams"] == ["grid", "normal"]
    assert "grid_random_rot_deg" not in request.extras


def test_ref_and_ref_prod_experiments_compose_gridnormal_eval() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref",
            "evaluation.gridnormal.enabled=true",
            "latent_cache.mode=online",
            "parallel.usp_size=1",
            "attention.backend=sdpa",
            "attention.allow_fallback=false",
            "evaluation.data_source=validation",
            "evaluation.manifest=/data/validation.jsonl",
            "evaluation.num_inference_steps=40",
            "evaluation.guidance_scale=1.0",
        ],
    )
    runtime = build_runtime(cfg, world_size=1)
    request = eval_request(runtime.config)

    assert runtime.config.model.name == "wan_taichu"
    assert request.extras["variant"] == "wan22_ti2v_5b_ref"
    assert request.gridnormal_enabled is True
    assert request.gridnormal_variants is None
    assert request.num_inference_steps == 40
    assert request.guidance_scale == 1.0

    prod_cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref_prod",
            "evaluation.data_source=validation",
            "evaluation.manifest=/data/validation.jsonl",
        ],
    )
    prod_request = eval_request(build_runtime(prod_cfg, world_size=1).config)

    assert prod_request.extras["variant"] == request.extras["variant"]
    assert prod_request.gridnormal_enabled is request.gridnormal_enabled
    assert prod_request.gridnormal_variants is request.gridnormal_variants
    assert prod_request.num_inference_steps == request.num_inference_steps
    assert prod_request.guidance_scale == request.guidance_scale
    assert prod_request.num_samples == 16
    assert prod_request.sample_selection == "evenly_spaced"
    assert prod_request.seed == 42


def test_a14b_prod_experiment_composes_gridnormal_eval() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_a14b_prod",
            "evaluation.data_source=validation",
            "evaluation.manifest=/data/a14b.jsonl",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.extras["variant"] == "wan22_a14b"
    assert request.extras["condition_streams"] == ["rgb", "grid", "normal"]
    assert request.gridnormal_enabled is True
    assert request.gridnormal_variants is None
    assert request.num_samples == 16
    assert request.sample_selection == "evenly_spaced"
    assert request.num_inference_steps == 40
    assert request.guidance_scale == 3.5
    assert request.seed == 42


def test_a14b_eval_translates_native_gameplay_source_without_manifest() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_a14b_prod",
            "evaluation=wan_taichu_a14b_20s_prod",
            "evaluation.data_source=validation",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.manifest == ""
    assert len(request.sources) == 1
    source = request.sources[0]
    assert source.name == "gameplay-20s"
    assert source.kind == "gameplay"
    assert source.index_files == (
        "/efs/shared/dataconfig/production-v1/gameplay-camera/items_20s_ref_gemini.pkl",
    )
    assert (source.frames, source.height, source.width) == (597, 720, 1280)
    assert source.extras["use_partial_normal_aug"] is True
    assert "path_replacements" not in source.extras


def test_a14b_causal_eval_translates_indexed_training_cache(monkeypatch) -> None:
    monkeypatch.setenv("A14B_LATENT_CACHE", "/cache/gameplay-a14b-20s")
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_a14b_causal_teacher",
            "evaluation=wan_taichu_a14b_causal_teacher_99lat_swa_train",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.data_source == "cache"
    assert request.manifest == ""
    assert request.rollout_mode == "incremental_kv"
    assert request.num_samples == 30
    assert request.extras["use_vae_channels_last_3d"] is False
    assert len(request.sources) == 1
    source = request.sources[0]
    assert source.cache_dir == "/cache/gameplay-a14b-20s"
    assert source.kind == ""
    assert (source.min_latents, source.max_latents) == (99, 99)


def test_eval_cli_path_replacements_apply_to_every_source() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_a14b_prod",
            "evaluation=wan_taichu_a14b_20s_prod",
            "evaluation.data_source=validation",
            "evaluation.path_replacements=[[/efs,/gwm-efs],[/threed-code,/gwm-code]]",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.sources[0].extras["path_replacements"] == [
        ["/efs", "/gwm-efs"],
        ["/threed-code", "/gwm-code"],
    ]


def test_5b_eval_preset_translates_three_native_gameplay_sources() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref_prod",
            "evaluation=wan_taichu_5b_ref_prod",
            "evaluation.data_source=validation",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.manifest == ""
    assert [source.name for source in request.sources] == [
        "seedance-5s",
        "gameplay-20s",
        "gameplay-60s",
    ]
    assert [source.frames for source in request.sources] == [149, 597, 1797]
    assert all(source.kind == "gameplay" for source in request.sources)


def test_ref_eval_translates_explicit_online_reference_indices() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref",
            "evaluation.data_source=validation",
            "evaluation.manifest=/data/validation.jsonl",
            "evaluation.online_ref.frame_indices=[3,9]",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.online_ref_num_frames is None
    assert request.online_ref_frame_indices == (3, 9)


def test_causal_experiment_declares_incremental_eval_overrides() -> None:
    cfg = compose_standalone_config(
        "eval",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_causal_teacher",
            "evaluation.rollout_mode=incremental_kv",
            "latent_cache.mode=online",
            "parallel.usp_size=1",
            "evaluation.data_source=inference",
            "evaluation.manifest=/data/inference.jsonl",
        ],
    )
    request = eval_request(build_runtime(cfg, world_size=1).config)

    assert request.rollout_mode == "incremental_kv"
    assert request.gridnormal_enabled is False
    assert request.extras["variant"] == "wan22_ti2v_5b_causal_teacher"


def test_train_preflight_rejects_negative_grid_rotation() -> None:
    cfg = compose_standalone_config(
        "train",
        overrides=[
            "experiment=wan_taichu_wan22_ti2v_5b_ref",
            "+model.extras.grid_random_rot_deg=-1",
            "checkpoint.enabled=false",
        ],
    )
    request = train_request(build_runtime(cfg, world_size=1).config)

    with pytest.raises(ValueError, match="grid_random_rot_deg must be non-negative"):
        _resolve_request(request)


def _production_request(name: str, *, world_size: int = 8):
    cfg = compose_standalone_config(
        "train",
        overrides=[
            f'hydra.searchpath=["file://{USER_CONFIG_DIR}"]',
            f"+user={name}",
        ],
    )
    runtime = build_runtime(cfg, world_size=world_size)
    return runtime, train_request(runtime.config)


def test_5b_ref_production_user_config_composes_the_taichu_mixture() -> None:
    runtime, request = _production_request("wan-taichu-5b-ref-prod")
    spec, streams, extras = _resolve_request(request)

    assert spec.name == "wan22_ti2v_5b_ref"
    assert streams == ("grid", "normal")
    assert request.latent_cache_mode == "offline"
    assert runtime.parallel.usp_size == 4
    assert runtime.parallel.dp_size == 2
    assert [source.name for source in request.sources] == [
        "seedance-5s",
        "gameplay-20s",
        "gameplay-60s",
    ]
    assert [source.weight for source in request.sources] == [0.5, 0.3, 0.2]
    assert [(source.min_latents, source.max_latents) for source in request.sources] == [
        (21, 38),
        (21, 150),
        (150, 450),
    ]
    assert extras["ref_num_frames_online"] == 6
    assert extras["grid_drop_prob"] == 0.1
    assert extras["normal_drop_prob"] == 0.2
    assert extras["load_vae_for_dropout_fallback"] is True
    assert "wan_use_fsdp_precision_contract" not in extras
    assert "grid_random_rot_deg" not in extras
    assert request.learning_rate == pytest.approx(1e-5)
    assert request.weight_decay == pytest.approx(0.01)
    assert request.eps == pytest.approx(1e-15)
    assert request.scheduler_name == "constant_with_warmup"
    assert request.warmup_steps == 100
    assert runtime.trainer.checkpoint.every_n_steps == 500


def test_a14b_production_user_config_composes_schedule_and_optimizer() -> None:
    runtime, request = _production_request("wan-taichu-a14b-prod")
    spec, streams, extras = _resolve_request(request)

    assert spec.name == "wan22_a14b"
    assert streams == ("rgb", "grid", "normal")
    assert request.latent_cache_mode == "offline"
    assert runtime.parallel.usp_size == 8
    assert runtime.parallel.dp_size == 1
    assert runtime.trainer.attention_offload.enable is True
    assert runtime.trainer.attention_offload.config.max_cpu_bytes == 64 * 1024**3
    assert runtime.trainer.attention_offload.config.async_transfer is True
    assert [source.weight for source in request.sources] == [0.2, 0.8]
    assert [(source.min_latents, source.max_latents) for source in request.sources] == [
        (21, 38),
        (38, 150),
    ]
    assert extras["grid_drop_prob"] == 0.1
    assert extras["normal_drop_prob"] == 1.0
    assert extras["train_expert"] == "high"
    assert extras["weighting_scheme"] == "bsmntw"
    assert "wan_use_fsdp_precision_contract" not in extras
    assert extras["max_latents_schedule"] == {
        "start_step": 0,
        "end_step": 4000,
        "start_value": 38,
        "end_value": 150,
    }
    assert "grid_random_rot_deg" not in extras
    assert request.learning_rate == pytest.approx(5e-6)
    assert request.weight_decay == pytest.approx(0.01)
    assert request.eps == pytest.approx(1e-15)
    assert request.scheduler_name == "constant_with_warmup"
    assert request.warmup_steps == 100
    assert runtime.trainer.checkpoint.every_n_steps == 500


def test_a14b_causal_teacher_user_config_composes_production_swa() -> None:
    runtime, request = _production_request("wan-taichu-a14b-causal-teacher-prod")
    spec, streams, extras = _resolve_request(request)

    assert spec.name == "wan22_a14b_causal_teacher"
    assert streams == ("rgb", "grid", "normal")
    assert runtime.parallel.usp_size == 8
    assert runtime.parallel.dp_size == 1
    assert extras["frame_per_block"] == 3
    assert extras["first_frame_own_block"] is False
    assert extras["local_attn_size"] == 30
    assert extras["sink_size"] == 9
    assert "grid_random_rot_deg" not in extras
    assert [(source.min_latents, source.max_latents) for source in request.sources] == [(99, 99)]
    assert request.learning_rate == pytest.approx(1e-5)
    assert request.eps == pytest.approx(1e-8)
    assert runtime.trainer.checkpoint.every_n_steps == 100


def test_a14b_causal_teacher_l99_swa_128_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A14B_CHECKPOINT", raising=False)
    monkeypatch.delenv("A14B_CONDITION_MODEL", raising=False)
    monkeypatch.setenv("A14B_LATENT_CACHE", "/read-only/latent-cache")
    monkeypatch.setenv("A14B_OUTPUT_DIR", "/efs/user/a14b-l99-swa/run")

    runtime, request = _production_request(
        "wan-taichu-a14b-causal-teacher-l99-swa-usp16-dp8",
        world_size=128,
    )
    _, streams, extras = _resolve_request(request)

    assert streams == ("rgb", "grid", "normal")
    assert runtime.config.model.pretrained_path.endswith("/checkpoints/01700.pt")
    assert extras["condition_pretrained_path"] == (
        "/gwm-efs/shared/models/Wan2.2-I2V-A14B-Diffusers"
    )
    assert runtime.parallel.usp_size == 16
    assert runtime.parallel.dp_size == 8
    assert extras["train_expert"] == "both"
    assert extras["grid_random_rot_deg"] == pytest.approx(30.0)
    assert extras["frame_per_block"] == 3
    assert extras["local_attn_size"] == 30
    assert extras["sink_size"] == 9
    assert [(source.min_latents, source.max_latents) for source in request.sources] == [(99, 99)]
    assert runtime.config.runtime.max_steps == 1_000_000
    assert runtime.trainer.checkpoint.every_n_steps == 100
    assert runtime.resume.path == "auto"
    assert runtime.trainer.checkpoint.dirpath == "/efs/user/a14b-l99-swa/run/checkpoints"
    assert runtime.trainer.storage.s3_mounts == ()
    assert runtime.config.logging.wandb_group == "a14b-causal-teacher-l99-swa-usp16-dp8"
