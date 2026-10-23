"""CPU/Hydra compose and builder resolution for standalone entries.

These tests intentionally avoid Trainer.fit / CUDA / flash-attn. Full torch
distributed regression belongs on the cloud acceptance job.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_standalone_packages_import_without_taichu() -> None:
    assert "taichu" not in sys.modules
    importlib.import_module("vforge.conf")
    importlib.import_module("vforge.tasks.train")
    importlib.import_module("vforge.tasks.prepare")
    importlib.import_module("vforge.tasks.eval")
    importlib.import_module("vforge.tasks.convert_checkpoint")
    importlib.import_module("vforge.tasks.wan.v21.train")
    importlib.import_module("vforge.tasks.wan.v21.eval")
    importlib.import_module("vforge.tasks.wan.v21.encode")
    importlib.import_module("vforge.data.dataset")
    importlib.import_module("vforge.evaluation.runner")
    assert "taichu" not in sys.modules


def test_prepare_hash_separates_qwen_patch_embedding_modes() -> None:
    from vforge.hydra_config import resolved_config_hash
    from vforge.tasks.prepare import _prepare_config_hashes

    default = {
        "model": {
            "name": "minimax_h3",
            "extras": {
                "num_frames": 124,
                "qwen_patch_embed_mode": "default",
                "video_vae_tile_batch_size": 1,
                "prepare_cpu_prefetch": True,
            },
        },
        "runtime": {"precision": "bf16"},
    }
    fp32 = {
        **default,
        "model": {
            **default["model"],
            "extras": {
                **default["model"]["extras"],
                "qwen_patch_embed_mode": "fp32",
            },
        },
    }
    last_channel = {
        **default,
        "model": {
            **default["model"],
            "extras": {
                **default["model"]["extras"],
                "qwen_patch_embed_mode": "last_channel",
            },
        },
    }
    no_prefetch = {
        **default,
        "model": {
            **default["model"],
            "extras": {
                **default["model"]["extras"],
                "prepare_cpu_prefetch": False,
            },
        },
    }

    default_hash, default_compatible = _prepare_config_hashes(default)
    fp32_hash, _ = _prepare_config_hashes(fp32)
    last_channel_hash, _ = _prepare_config_hashes(last_channel)
    no_prefetch_hash, no_prefetch_compatible = _prepare_config_hashes(no_prefetch)

    assert len({default_hash, fp32_hash, last_channel_hash}) == 3
    assert no_prefetch_hash == default_hash
    assert resolved_config_hash(default) in default_compatible
    assert resolved_config_hash(no_prefetch) in no_prefetch_compatible


def test_native_hydra_maps_explicit_s3_mounts() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    config = config_from_dict(
        {
            "model": {"name": "wan21"},
            "runtime": {"weights_strict": True},
            "diagnostics": {"log_local_parameter_stats": True},
            "storage": {
                "s3_mounts": [
                    {
                        "mount_path": "/asset",
                        "bucket": "asset-bucket",
                        "region": "region",
                        "key_prefix": "models",
                        "endpoint_url": "https://oss.example.com",
                        "addressing_style": "virtual",
                    }
                ]
            },
        }
    )
    runtime = build_runtime(config, world_size=1)

    mount = runtime.trainer.storage.s3_mounts[0]
    assert mount.bucket == "asset-bucket"
    assert mount.region == "region"
    assert mount.mount_path == "/asset"
    assert mount.key_prefix == "models"
    assert mount.endpoint_url == "https://oss.example.com"
    assert mount.addressing_style == "virtual"
    assert runtime.trainer.weights_strict
    assert runtime.trainer.diagnostics.log_local_parameter_stats


def test_runtime_seed_composes_and_reaches_trainer_and_task() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_runtime, train_request

    composed = compose_standalone_config(
        "train", overrides=["experiment=wan21_i2v_14b", "runtime.seed=17"]
    )
    config = config_from_dict(composed)
    assert "execution" not in composed.runtime
    assert config.runtime.deterministic is False
    assert build_runtime(config, world_size=1).trainer.seed == 17
    assert train_request(config).seed == 17


@pytest.mark.parametrize("mode", ["performance", "fixed-rng", "strict"])
def test_removed_execution_mode_is_rejected(mode) -> None:
    from hydra.errors import ConfigCompositionException

    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config

    with pytest.raises(ConfigCompositionException, match="runtime.execution"):
        compose_standalone_config("train", overrides=[f"runtime.execution={mode}"])
    with pytest.raises(TypeError, match="execution"):
        config_from_dict({"runtime": {"execution": mode}})


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("backend", ["auto", "sdpa", "sdpa_math", "flash_attention_4"])
def test_runtime_deterministic_composes_and_reaches_ops(enabled, backend) -> None:
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_runtime

    composed = compose_standalone_config(
        "train",
        overrides=[
            f"runtime.deterministic={str(enabled).lower()}",
            f"attention.backend={backend}",
            f"attention.allow_fallback={str(backend == 'auto').lower()}",
        ],
    )
    runtime = build_runtime(composed, world_size=1)
    assert runtime.config.runtime.deterministic is enabled
    assert runtime.ops.deterministic is enabled
    assert runtime.trainer.runtime_ops == runtime.ops
    assert "deterministic" not in composed.attention


def test_attention_deterministic_is_rejected_after_move_to_runtime() -> None:
    from hydra.errors import ConfigCompositionException

    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config

    with pytest.raises(ConfigCompositionException, match="attention.deterministic"):
        compose_standalone_config("train", overrides=["attention.deterministic=true"])
    with pytest.raises(TypeError, match="deterministic"):
        config_from_dict({"attention": {"deterministic": True}})


@pytest.mark.parametrize("value", [1, "true", None])
def test_runtime_deterministic_requires_a_boolean(value) -> None:
    from vforge.tasks.translate import build_runtime

    with pytest.raises(TypeError, match="runtime.deterministic must be a boolean"):
        build_runtime({"runtime": {"deterministic": value}}, world_size=1)


def test_runtime_deterministic_rejects_flash_attention_3() -> None:
    from vforge.tasks.translate import build_runtime

    with pytest.raises(ValueError, match="runtime.deterministic=true.*flash_attention_3"):
        build_runtime(
            {
                "runtime": {"deterministic": True},
                "attention": {"backend": "flash_attention_3", "allow_fallback": False},
            },
            world_size=1,
        )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ([1, 3], (1, 3)),
        (None, None),
    ],
)
def test_attention_offload_fa_call_selection_translates(configured, expected) -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    config = config_from_dict(
        {
            "attention_offload": {
                "enable": True,
                "preserve_fa_call_indices": configured,
            }
        }
    )
    runtime = build_runtime(config, world_size=1)

    assert runtime.trainer.attention_offload.preserve_fa_call_indices == expected


def test_attention_offload_rejects_negative_fa_call_index() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    config = config_from_dict({"attention_offload": {"preserve_fa_call_indices": [-1]}})
    with pytest.raises(ValueError, match="non-negative"):
        build_runtime(config, world_size=1)


def test_attention_offload_transfer_policy_translates() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    config = config_from_dict(
        {
            "attention_offload": {
                "min_tensor_bytes": 4096,
                "pin_memory": True,
                "max_cpu_bytes": 1 << 30,
                "async_transfer": True,
            }
        }
    )
    policy = build_runtime(config, world_size=1).trainer.attention_offload.config

    assert policy.min_tensor_bytes == 4096
    assert policy.pin_memory is True
    assert policy.max_cpu_bytes == 1 << 30
    assert policy.async_transfer is True


def test_attention_offload_is_not_a_strict_resume_fingerprint() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    runtime = build_runtime(
        config_from_dict(
            {
                "attention_offload": {
                    "enable": True,
                    "preserve_fa_call_indices": [1, 3],
                }
            }
        ),
        world_size=1,
    )

    assert "attention_offload" not in runtime.trainer.checkpoint_fingerprints


def test_schema_imports_nothing_from_vforge() -> None:
    """A module of field declarations must not drag a runtime layer along.

    Checked on the source rather than on ``sys.modules`` so the guarantee holds
    no matter what an earlier test already imported.
    """
    import ast

    source = ROOT / "vforge" / "hydra_config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not {name for name in imported if name.split(".")[0] == "vforge"}


def test_schema_engine_names_match_the_engine_registry() -> None:
    from vforge.hydra_config import ENGINE_NAMES as SCHEMA_ENGINE_NAMES
    from vforge.training.engine import ENGINE_NAMES

    assert SCHEMA_ENGINE_NAMES == ENGINE_NAMES


def test_unified_checkpoint_and_resume_schema_defaults() -> None:
    from vforge.hydra_config import CheckpointHydraConfig, ResumeHydraConfig

    checkpoint = CheckpointHydraConfig()
    assert checkpoint.enabled is True
    assert checkpoint.every_n_steps == 0
    assert checkpoint.dirpath == ""
    assert checkpoint.state == "full"
    assert checkpoint.format == "dcp"
    assert checkpoint.save_final is True
    assert checkpoint.keep_last == 2
    assert checkpoint.async_save is True
    assert ResumeHydraConfig().path == "none"
    assert ResumeHydraConfig().strict is False


def test_native_hydra_user_group_inherits_train_and_cli_override() -> None:
    config_dir = ROOT / "examples" / "configs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vforge.tasks.train",
            "--config-dir",
            str(config_dir),
            "--cfg",
            "job",
            "+user=wan21-local",
            "runtime.max_steps=1",
            "parallel.mixed_precision.param_dtype=float32",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "name: wan21" in result.stdout
    assert "max_steps: 1" in result.stdout
    assert "every_n_steps: 500" in result.stdout
    assert "path: none" in result.stdout
    assert "param_dtype: float32" in result.stdout


@pytest.mark.parametrize(
    ("task_name", "expected"),
    (
        ("train", ("latent_cache.mode", "offline")),
        ("prepare", ("latent_cache.mode", "online")),
        ("eval", ("evaluation.output_dir", "/runs/wan21/local/samples")),
        ("convert_checkpoint", ("model.name", "wan21")),
    ),
)
def test_native_hydra_user_group_reuses_across_task_lifecycles(
    task_name: str,
    expected: tuple[str, str],
) -> None:
    from omegaconf import OmegaConf

    from vforge.tasks.common import compose_standalone_config

    config_dir = ROOT / "examples" / "configs"
    cfg = compose_standalone_config(
        task_name,
        overrides=[
            f'hydra.searchpath=["file://{config_dir}"]',
            "+user=wan21-local",
        ],
    )

    assert OmegaConf.select(cfg, expected[0]) == expected[1]
    assert cfg.checkpoint.every_n_steps == 500
    assert "user" not in cfg


@pytest.mark.parametrize(
    "experiment",
    (
        "wan21_i2v_14b",
        "wan22_a14b",
        "wan22_ti2v_5b",
        "wan22_ti2v_5b_tf",
        "minimax_h3_fl2va",
        "gen_engine",
    ),
)
def test_model_experiments_inherit_unified_output_paths(
    experiment: str,
    tmp_path: Path,
) -> None:
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_runtime

    train = compose_standalone_config(
        "train",
        overrides=[
            f"experiment={experiment}",
            f"runtime.output_dir={tmp_path}",
            "checkpoint.enabled=false",
        ],
    )
    train_runtime = build_runtime(train, world_size=1)
    assert train_runtime.trainer.checkpoint.dirpath == str(tmp_path / "checkpoints")
    assert train_runtime.resume.path == "none"

    evaluation = compose_standalone_config(
        "eval",
        overrides=[
            f"experiment={experiment}",
            f"runtime.output_dir={tmp_path}",
        ],
    )
    assert evaluation.evaluation.output_dir == f"{tmp_path}/samples"
    assert evaluation.evaluation.data_source == ""
    assert evaluation.evaluation.manifest == ""


def test_data_cache_dir_is_the_single_cache_shorthand() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_runtime

    cfg = config_from_dict({"data": {"cache_dir": "/tmp/one-cache"}})
    assert len(cfg.data.sources) == 1
    assert cfg.data.sources[0].cache_dir == "/tmp/one-cache"

    composed = compose_standalone_config(
        "train",
        overrides=["experiment=wan21_i2v_14b", "data.cache_dir=/tmp/cli-cache"],
    )
    runtime = build_runtime(composed, world_size=1)
    assert len(runtime.config.data.sources) == 1
    assert runtime.config.data.sources[0].cache_dir == "/tmp/cli-cache"


def test_data_sources_takes_paths_or_mappings() -> None:
    from vforge.hydra_config import config_from_dict

    cfg = config_from_dict(
        {
            "data": {
                "sources": [
                    "/tmp/a",
                    {"name": "b", "cache_dir": "/tmp/b", "weight": 0.5},
                ]
            }
        }
    )
    assert [source.cache_dir for source in cfg.data.sources] == ["/tmp/a", "/tmp/b"]
    assert cfg.data.sources[1].name == "b"
    assert cfg.data.sources[1].weight == 0.5


def test_data_cache_dir_rejects_a_list() -> None:
    from vforge.hydra_config import config_from_dict

    with pytest.raises(ValueError, match="single-cache shorthand"):
        config_from_dict({"data": {"cache_dir": [{"cache_dir": "/tmp/a"}]}})


def test_data_cache_dir_and_sources_cannot_both_name_data() -> None:
    """Overriding with sources must clear an inherited shorthand, not stack on it."""
    from vforge.hydra_config import config_from_dict

    with pytest.raises(ValueError, match="both name training data"):
        config_from_dict(
            {"data": {"cache_dir": "/tmp/inherited", "sources": [{"cache_dir": "/tmp/a"}]}}
        )

    # The layering that experiments actually use: the base leaves cache_dir empty.
    cleared = config_from_dict({"data": {"cache_dir": "", "sources": [{"cache_dir": "/tmp/a"}]}})
    assert [source.cache_dir for source in cleared.data.sources] == ["/tmp/a"]


def _raw_source(**overrides) -> dict:
    source = {
        "name": "gameplay20s",
        "kind": "gameplay",
        "index_files": ["/tmp/items.pkl"],
        "frames": 149,
        "height": 704,
        "width": 1280,
    }
    source.update(overrides)
    return source


def test_a_raw_data_source_reaches_the_train_request() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import train_request

    cfg = config_from_dict(
        {"data": {"sources": [_raw_source(extras={"use_partial_normal_aug": True})]}}
    )
    source = cfg.data.sources[0]

    assert source.kind == "gameplay"
    assert source.index_files == ("/tmp/items.pkl",)
    assert source.extras == {"use_partial_normal_aug": True}

    spec = train_request(cfg).sources[0]
    assert (spec.kind, spec.frames, spec.height, spec.width) == ("gameplay", 149, 704, 1280)
    assert spec.extras == {"use_partial_normal_aug": True}


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"index_files": ["/tmp/items.pkl"]}, "without a kind"),
        (_raw_source(manifest="/tmp/m.json"), "together with manifest"),
        (_raw_source(index_files=[]), "requires index_files"),
        (_raw_source(frames=0), "requires frames"),
    ],
)
def test_a_malformed_data_source_is_rejected_at_compose_time(source, message) -> None:
    from vforge.hydra_config import config_from_dict

    with pytest.raises(ValueError, match=message):
        config_from_dict({"data": {"sources": [source]}})


def test_a_raw_source_and_an_encoded_cache_cannot_share_a_run() -> None:
    from vforge.tasks.requests import DataSourceSpec
    from vforge.tasks.wan.shared import _split_sources

    with pytest.raises(ValueError, match="mixes raw sources"):
        _split_sources(
            (DataSourceSpec(kind="gameplay"), DataSourceSpec(cache_dir="/tmp/cache")),
            latent_cache_mode="online",
        )


def test_raw_sources_require_online_latent_cache_mode() -> None:
    from vforge.tasks.requests import DataSourceSpec
    from vforge.tasks.wan.shared import _split_sources

    with pytest.raises(ValueError, match="latent_cache.mode=online"):
        _split_sources((DataSourceSpec(kind="gameplay"),), latent_cache_mode="offline")


def test_a_raw_source_with_a_cache_dir_is_ambiguous_for_training() -> None:
    """That pair is the prepare spelling: encode this raw source into there."""
    from vforge.tasks.requests import DataSourceSpec
    from vforge.tasks.wan.shared import _split_sources

    with pytest.raises(ValueError, match="both kind and cache_dir"):
        _split_sources(
            (DataSourceSpec(kind="gameplay", cache_dir="/tmp/out"),),
            latent_cache_mode="online",
        )


def test_empty_data_cache_dir_is_ignored() -> None:
    from vforge.hydra_config import config_from_dict

    cfg = config_from_dict({"data": {"cache_dir": ""}})
    assert cfg.data.sources == ()


@pytest.mark.parametrize("data_source", ("validation", "inference", "cache"))
def test_eval_request_translates_evaluation_source(
    data_source: str,
) -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config, omega_to_container
    from vforge.tasks.translate import eval_request

    value = f"/data/{data_source}/manifest.json"
    cfg = config_from_dict(
        omega_to_container(
            compose_standalone_config(
                "eval",
                overrides=[
                    f"evaluation.data_source={data_source}",
                    f"evaluation.manifest={value}",
                    "evaluation.num_samples=7",
                ],
            )
        )
    )

    request = eval_request(cfg)
    assert request.data_source == data_source
    assert request.manifest == value
    assert request.num_samples == 7


@pytest.mark.parametrize(
    "evaluation",
    (
        {},
        {"data_source": "inference"},
        {"manifest": "/data/infer.jsonl"},
    ),
)
def test_eval_request_requires_data_source_and_one_input(evaluation: dict[str, str]) -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import eval_request

    with pytest.raises(ValueError, match="data_source and exactly one input"):
        eval_request(config_from_dict({"evaluation": evaluation}))


def test_eval_request_rejects_unknown_data_source() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import eval_request

    config = config_from_dict(
        {
            "evaluation": {
                "data_source": "future",
                "manifest": "/data/future.jsonl",
            }
        }
    )
    with pytest.raises(ValueError, match="unsupported evaluation.data_source='future'"):
        eval_request(config)


@pytest.mark.parametrize(
    ("legacy_field", "replacement"),
    (
        ("val_manifest", "evaluation.data_source=validation"),
        ("cache_dir", "evaluation.data_source=cache"),
    ),
)
def test_eval_request_rejects_legacy_data_source(
    legacy_field: str,
    replacement: str,
) -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config, omega_to_container
    from vforge.tasks.translate import eval_request

    cfg = config_from_dict(
        omega_to_container(
            compose_standalone_config(
                "eval",
                overrides=[f"data.{legacy_field}=/legacy/input"],
            )
        )
    )

    with pytest.raises(ValueError, match=replacement):
        eval_request(cfg)


def test_eval_request_accepts_mcore_checkpoint() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import eval_request

    config = config_from_dict(
        {
            "model": {"name": "wan22_a14b"},
            "parallel": {
                "engine": "megatron",
                "dp_mode": "ddp",
            },
            "evaluation": {
                "checkpoint_format": "mcore",
                "data_source": "inference",
                "manifest": "/manifests/eval.jsonl",
                "checkpoint_path": {
                    "high": "/checkpoints/high.mcore",
                    "low": "/checkpoints/low.mcore",
                },
            },
        }
    )

    request = eval_request(config)

    assert request.checkpoint_format == "mcore"
    assert request.checkpoint_path == {
        "high": "/checkpoints/high.mcore",
        "low": "/checkpoints/low.mcore",
    }


@pytest.mark.parametrize(
    "checkpoint_path",
    [42, ["high", "low"], {"high": 42}],
)
def test_eval_checkpoint_path_rejects_values_outside_the_public_contract(
    checkpoint_path,
) -> None:
    from vforge.hydra_config import config_from_dict

    with pytest.raises(TypeError, match="checkpoint_path"):
        config_from_dict({"evaluation": {"checkpoint_path": checkpoint_path}})


def test_wan22_mcore_cp_tp_topology_resolves() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_runtime

    config = config_from_dict(
        {
            "model": {"name": "wan22_a14b"},
            "parallel": {
                "engine": "megatron",
                "dp_mode": "ddp",
                "dp_size": 2,
                "cp_size": 2,
                "tp_size": 2,
            },
            "checkpoint": {"format": "mcore"},
        }
    )

    runtime = build_runtime(config, world_size=8)

    assert runtime.trainer.engine == "megatron"
    assert runtime.parallel.dp_size == 2
    assert runtime.parallel.cp_size == 2
    assert runtime.parallel.tp_size == 2
    assert runtime.trainer.checkpoint.format == "mcore"


def test_wan21_keeps_fp32_optimizer_facing_parameters() -> None:
    from vforge.tasks.wan.shared import parameter_storage_dtype

    assert parameter_storage_dtype() is torch.float32


def test_hydra_train_compose_and_resolved_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch.nn as nn

    import vforge.tasks.wan.v21.train as wan_task
    from vforge.tasks.common import build_and_snapshot, compose_standalone_config
    from vforge.tasks.registry import resolve_builder
    from vforge.tasks.translate import train_request

    def build_train_components(runtime):
        build = resolve_builder(runtime.config.model.name, "train")
        return build(train_request(runtime.config))

    class DummyWan(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(1, 1)

    monkeypatch.setattr(
        wan_task,
        "_build_wan21_model",
        lambda **_kwargs: DummyWan(),
    )
    monkeypatch.setattr(
        wan_task,
        "_wan_pretrained_source",
        lambda **_kwargs: None,
    )

    cfg = compose_standalone_config(
        "train",
        overrides=[
            f"runtime.output_dir={tmp_path}",
            "runtime.max_steps=3",
            "runtime.run_backward=false",
            "parallel.usp_size=1",
            "latent_cache.mode=offline",
            "optimizer.learning_rate=2.0e-5",
            "checkpoint.enabled=false",
        ],
    )
    runtime = build_and_snapshot(cfg, world_size=1)
    assert runtime.trainer.max_steps == 3
    assert runtime.trainer.run_backward is False
    assert runtime.parallel.usp_size == 1
    assert runtime.parallel.dp_size == 1
    assert runtime.config.model.name == "wan21"
    assert (tmp_path / "resolved_config.yaml").is_file()
    resolved = (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "padded_cross_attention_requires_varlen: true" in resolved
    assert (tmp_path / "resolved_config.hash").read_text(encoding="utf-8").strip() == (
        runtime.resolved_config_hash
    )
    assert runtime.trainer.checkpoint_fingerprints["semantic_config"]["limitations"] == {
        "expert_parallel": False,
        "ring_attention": False,
        "train_condition_encoders": False,
        "cross_attention_ulysses": False,
        "padded_cross_attention_requires_varlen": True,
    }
    with pytest.raises(ValueError, match="offline training"):
        build_train_components(runtime)

    compose_only = compose_standalone_config(
        "train",
        overrides=[
            f"runtime.output_dir={tmp_path}",
            "runtime.max_steps=0",
            "parallel.usp_size=1",
            "latent_cache.mode=offline",
            "checkpoint.enabled=false",
        ],
    )
    components = build_train_components(build_and_snapshot(compose_only, world_size=1))
    from vforge.training.recipe import TrainComponents

    assert isinstance(components, TrainComponents)
    assert runtime.config.latent_cache.mode == "offline"
    assert runtime.config.optimizer.learning_rate == 2.0e-5
    assert components.model is not None
    assert components.step_fn is not None


def test_hydra_prepare_requires_online_mode(tmp_path: Path) -> None:
    from vforge.tasks.common import build_and_snapshot, compose_standalone_config
    from vforge.tasks.registry import resolve_builder

    cfg = compose_standalone_config(
        "prepare",
        overrides=[
            f"runtime.output_dir={tmp_path}",
            "latent_cache.mode=offline",
        ],
    )
    runtime = build_and_snapshot(cfg, world_size=1)
    with pytest.raises(ValueError, match="latent_cache.mode=online"):
        resolve_builder(runtime.config.model.name, "encode")(runtime.config)


def test_resume_and_ops_config_validation() -> None:
    from vforge.ops.config import OpsConfig
    from vforge.training.checkpoint.config import ResumeConfig, build_resume_config

    with pytest.raises(ValueError, match="allow_fallback"):
        OpsConfig(backend="sdpa", allow_fallback=True)
    with pytest.raises(ValueError, match="resume.weights"):
        ResumeConfig(weights=False, optimizer=True)
    with pytest.raises(ValueError, match="resume.training_state"):
        ResumeConfig(training_state=True, optimizer=False)
    with pytest.raises(ValueError, match="merged into training_state"):
        build_resume_config({"progress": False})


def test_schema_blocks_are_validated_by_their_runtime_twin() -> None:
    """The schema takes the value; the runtime type is what rejects it."""
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_ops_config, build_resume_plan

    bad_ops = config_from_dict({"attention": {"backend": "sdpa", "allow_fallback": True}})
    assert bad_ops.attention.backend == "sdpa"
    with pytest.raises(ValueError, match="allow_fallback"):
        build_ops_config(bad_ops.attention, deterministic=bad_ops.runtime.deterministic)

    bad_resume = config_from_dict({"resume": {"weights": False, "optimizer": True}})
    with pytest.raises(ValueError, match="resume.weights"):
        build_resume_plan(bad_resume.resume)

    with pytest.raises(ValueError, match="merged into training_state"):
        config_from_dict({"resume": {"progress": False}})

    legacy_multi_node = config_from_dict({"limitations": {"multi_node": False}})
    assert "multi_node" not in asdict(legacy_multi_node.limitations)
    with pytest.raises(ValueError, match="no longer a feature toggle"):
        config_from_dict({"limitations": {"multi_node": True}})


def test_dp_size_auto_from_world_size() -> None:
    from vforge.hydra_config import ParallelHydraConfig
    from vforge.tasks.translate import resolve_dp_size

    parallel = ParallelHydraConfig(usp_size=8, dp_size="auto")
    assert resolve_dp_size(parallel, world_size=8) == 1
    with pytest.raises(ValueError, match="divisible"):
        resolve_dp_size(parallel, world_size=7)


def test_dp_size_accounts_for_tensor_parallel() -> None:
    from vforge.hydra_config import ParallelHydraConfig
    from vforge.tasks.translate import build_parallel_config, resolve_dp_size

    # tp replicates the batch exactly like ulysses does, so it divides out of
    # dp too. Leaving it out would hand DP the whole world and silently build a
    # mesh whose dimensions no longer multiply to world_size.
    assert resolve_dp_size(ParallelHydraConfig(tp_size=2), world_size=8) == 4
    assert resolve_dp_size(ParallelHydraConfig(cp_size=2, tp_size=2), world_size=8) == 2
    assert resolve_dp_size(ParallelHydraConfig(tp_size=2, usp_size=2), world_size=8) == 2

    with pytest.raises(ValueError, match="divisible"):
        resolve_dp_size(ParallelHydraConfig(tp_size=3), world_size=8)
    with pytest.raises(ValueError, match=r"usp_size \* cp_size \* tp_size"):
        resolve_dp_size(ParallelHydraConfig(tp_size=2, dp_size=8), world_size=8)

    parallel = build_parallel_config(ParallelHydraConfig(tp_size=2), world_size=8)
    assert parallel.tp_size == 2
    assert parallel.dp_size == 4
    assert parallel.dp_shard_size == 4


def test_explicit_hsdp_topology_translates_from_hydra() -> None:
    from vforge.hydra_config import ParallelHydraConfig, config_from_dict
    from vforge.tasks.common import compose_standalone_config, omega_to_container
    from vforge.tasks.translate import build_parallel_config

    parallel = build_parallel_config(
        ParallelHydraConfig(dp_replicate_size=2, dp_shard_size=4),
        world_size=8,
    )

    assert parallel.dp_size == 8
    assert parallel.dp_replicate_size == 2
    assert parallel.dp_shard_size == 4

    composed = compose_standalone_config(
        "train",
        overrides=[
            "parallel.dp_replicate_size=2",
            "parallel.dp_shard_size=2",
        ],
    )
    composed_config = config_from_dict(omega_to_container(composed))
    composed_parallel = build_parallel_config(composed_config.parallel, world_size=4)
    assert composed_parallel.dp_replicate_size == 2
    assert composed_parallel.dp_shard_size == 2

    with pytest.raises(ValueError, match="must equal dp_size"):
        build_parallel_config(
            ParallelHydraConfig(dp_replicate_size=2, dp_shard_size=2),
            world_size=8,
        )


def test_hsdp_requires_a_real_shard_axis_and_fsdp2() -> None:
    from vforge.hydra_config import ParallelHydraConfig

    with pytest.raises(ValueError, match="dp_shard_size > 1"):
        ParallelHydraConfig(dp_replicate_size=2)
    with pytest.raises(ValueError, match="engine=fsdp2"):
        ParallelHydraConfig(
            engine="megatron",
            dp_mode="ddp",
            dp_replicate_size=2,
            dp_shard_size=2,
        )


def test_parallel_schema_rejects_non_positive_sizes() -> None:
    from vforge.hydra_config import ParallelHydraConfig

    with pytest.raises(ValueError, match="tp_size must be >= 1"):
        ParallelHydraConfig(tp_size=0)
    with pytest.raises(ValueError, match="usp_size must be >= 1"):
        ParallelHydraConfig(usp_size=0)
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        ParallelHydraConfig(cp_size=0)
    with pytest.raises(ValueError, match="dp_replicate_size must be >= 1"):
        ParallelHydraConfig(dp_replicate_size=0)
    with pytest.raises(ValueError, match="dp_shard_size must be >= 1"):
        ParallelHydraConfig(dp_shard_size=0)


def test_parallel_config_uses_torchrun_world_size_by_default(monkeypatch) -> None:
    from vforge.hydra_config import ParallelHydraConfig
    from vforge.tasks.translate import build_parallel_config

    monkeypatch.setenv("WORLD_SIZE", "8")
    parallel = build_parallel_config(ParallelHydraConfig(usp_size=8))
    assert parallel.dp_size == 1
    assert parallel.usp_size == 8


def test_hydra_mixed_precision_keeps_glob_paths() -> None:
    from omegaconf import OmegaConf

    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_parallelize_config

    yaml_cfg = OmegaConf.create(
        """
parallel:
  mixed_precision:
    preserve_module_paths:
      - blocks.*.norm1
      - patch_embedding_ref
    preserve_forward_input_paths:
      - .
      - blocks.*
    contract_mode: replace
"""
    )
    config = config_from_dict(yaml_cfg)
    mixed = config.parallel.mixed_precision
    assert mixed.preserve_module_paths == ("blocks.*.norm1", "patch_embedding_ref")
    assert mixed.preserve_forward_input_paths == (".", "blocks.*")
    assert mixed.contract_mode == "replace"

    runtime = build_parallelize_config(config, world_size=1)
    assert runtime.fsdp2.mixed_precision.preserve_module_paths == (
        "blocks.*.norm1",
        "patch_embedding_ref",
    )
    assert runtime.fsdp2.mixed_precision.preserve_forward_input_paths == (
        ".",
        "blocks.*",
    )


def test_mcore_ddp_overlap_policy_translates_from_hydra() -> None:
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.translate import build_parallelize_config

    config = config_from_dict(
        {
            "parallel": {
                "engine": "megatron",
                "dp_mode": "ddp",
                "dp_size": 1,
                "mcore_ddp": {
                    "overlap_grad_reduce": False,
                    "overlap_param_gather": False,
                },
            }
        }
    )

    runtime = build_parallelize_config(config, world_size=1)

    assert runtime.mcore_ddp.overlap_grad_reduce is False
    assert runtime.mcore_ddp.overlap_param_gather is False


@pytest.mark.parametrize(
    "experiment, overrides, dtype",
    [
        ("wan21_i2v_14b", [], "float32"),
        ("ltx23_dmd", [], "float32"),
        ("ltx23_dmd", ["runtime.gradient_norm_dtype=float64"], "float64"),
    ],
)
def test_gradient_clipping_compose_and_translation(experiment, overrides, dtype):
    from hydra import compose, initialize_config_module

    from vforge.tasks.translate import build_trainer_config

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={experiment}", *overrides])
    assert "gradient_clip_impl" not in cfg.runtime
    assert cfg.runtime.gradient_norm_dtype == dtype
    from vforge.hydra_config import config_from_dict

    runtime = build_trainer_config(config_from_dict(cfg))
    assert not hasattr(runtime, "gradient_clip_impl")
    assert runtime.gradient_norm_dtype == dtype


@pytest.mark.parametrize("entry", ["train", "prepare", "eval", "convert_checkpoint"])
def test_native_entries_expose_schema_defaults_before_cli_overrides(entry):
    from hydra.errors import ConfigCompositionException

    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config

    cfg = compose_standalone_config(entry, ["experiment=ltx23_dmd"])
    assert cfg.parallel.mixed_precision.param_dtype == "bfloat16"
    assert cfg.parallel.mixed_precision.reduce_dtype == "float32"
    # Experiment differences must still override the shared defaults.
    assert cfg.parallel.mixed_precision.cast_forward_inputs is False

    cfg = compose_standalone_config(
        entry,
        [
            "experiment=ltx23_dmd",
            "parallel.mixed_precision.param_dtype=float32",
            "parallel.mixed_precision.reduce_dtype=float64",
            "parallel.mixed_precision.cast_forward_inputs=true",
            "runtime.gradient_norm_dtype=float64",
        ],
    )
    parsed = config_from_dict(cfg)
    assert parsed.parallel.mixed_precision.param_dtype == "float32"
    assert parsed.parallel.mixed_precision.reduce_dtype == "float64"
    assert parsed.parallel.mixed_precision.cast_forward_inputs is True
    assert parsed.runtime.gradient_norm_dtype == "float64"
    with pytest.raises(ConfigCompositionException, match="Could not override"):
        compose_standalone_config(entry, ["parallel.mixed_precision.param_dtyp=float32"])


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("verb", ["train", "eval", "prepare", "convert_checkpoint"])
def test_accumulation_override_is_declared_and_translated(verb, count):
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_runtime

    cfg = compose_standalone_config(
        verb, overrides=[f"runtime.gradient_accumulation_steps={count}"]
    )
    runtime = build_runtime(cfg, world_size=1)
    assert runtime.trainer.gradient_accumulation_steps == count
    assert runtime.config.runtime.gradient_accumulation_steps == count
    assert "gradient_accumulation_steps" not in asdict(runtime.config.limitations)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_accumulation_config_rejects_invalid_count(count):
    from vforge.hydra_config import config_from_dict

    with pytest.raises(ValueError, match="positive integer"):
        config_from_dict({"runtime": {"gradient_accumulation_steps": count}})


@pytest.mark.parametrize("count", [1, 4])
def test_removed_accumulation_field_is_not_accepted(count):
    from hydra.errors import ConfigCompositionException

    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config

    with pytest.raises(TypeError, match="gradient_accumulation_steps"):
        config_from_dict({"limitations": {"gradient_accumulation_steps": count}})
    with pytest.raises(ConfigCompositionException):
        compose_standalone_config(
            "train", overrides=[f"limitations.gradient_accumulation_steps={count}"]
        )


@pytest.mark.parametrize(
    "experiment",
    sorted(path.stem for path in (ROOT / "vforge/conf/experiment").glob("*.yaml")),
)
@pytest.mark.parametrize("strict", [False, True])
def test_resume_strict_compose_and_translation(experiment, strict):
    from vforge.hydra_config import config_from_dict
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.translate import build_resume_plan

    overrides = [f"experiment={experiment}"]
    if strict:
        overrides.append("resume.strict=true")
    cfg = compose_standalone_config("train", overrides)
    assert cfg.resume.strict is strict
    plan = build_resume_plan(config_from_dict(cfg).resume)
    assert plan.effective_strict is strict
    assert plan.optimizer is True
    assert plan.training_state is True
