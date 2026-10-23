"""The shared evaluation assembly, exercised without real Wan weights.

Every version now runs the same body, so the wiring it does on a version's
behalf -- which codec canonicalizes a raw sample, which collate reads a cache,
whether a state-dict mapper reaches the loader, and the latent-to-pixel
arithmetic in ``prepare_sample`` -- is worth pinning down once here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import EvalRequest, validate_eval_source
from vforge.tasks.wan.eval import WanEvalSpec, assemble_eval_components, eval_artifact_name

SPATIAL = 8
TEMPORAL = 4
LATENT_FRAMES = 21
LATENT_H = 60
LATENT_W = 104


def _frame_latents() -> torch.Tensor:
    return torch.zeros(1, 16, LATENT_FRAMES, LATENT_H, LATENT_W)


class _Codec:
    def __init__(self) -> None:
        self.canonicalized = 0

    def canonicalize(self, bundle: dict[str, Any]) -> dict[str, Any]:
        self.canonicalized += 1
        return {**bundle, "canonicalized": True}


class _Tokenizer:
    model_max_length = 2

    def __call__(self, texts, **_kwargs):
        batch_size = len(texts)
        return SimpleNamespace(
            input_ids=torch.zeros(batch_size, 2, dtype=torch.long),
            attention_mask=torch.ones(batch_size, 2, dtype=torch.long),
        )


class _TextEncoder:
    def __call__(self, input_ids, _mask):
        return SimpleNamespace(
            last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], 4)
        )


class _Encoder:
    tokenizer = _Tokenizer()
    text_encoder = _TextEncoder()
    vae = SimpleNamespace(
        config=SimpleNamespace(
            scale_factor_spatial=SPATIAL,
            scale_factor_temporal=TEMPORAL,
        )
    )

    def eval(self) -> None:
        return None

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {"frame_latents": _frame_latents(), "prompt": torch.zeros(1, 3, 4)}

    def encode_inference_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {"prompt": torch.zeros(1, 3, 4)}


class _Pipeline:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.guidance_scale = float(kwargs["guidance_scale"])
        self.negative_prompt = kwargs["negative_prompt"]

    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        del seed
        return sample


class _Dataset:
    def __init__(self) -> None:
        self.items = [{"sample_id": "s0"}]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class _Runtime:
    device = torch.device("cpu")


def _spec(**overrides: Any) -> tuple[WanEvalSpec, dict[str, Any]]:
    calls: dict[str, Any] = {"collated": 0, "pipelines": []}

    def collate_encoded(batches: list[dict[str, Any]]) -> dict[str, Any]:
        calls["collated"] += 1
        return {"frame_latents": _frame_latents(), "prompt": torch.zeros(1, 3, 4)}

    def build_pipeline(**kwargs: Any) -> _Pipeline:
        calls["pipelines"].append(kwargs)
        return _Pipeline(**kwargs)

    defaults: dict[str, Any] = {
        "model": nn.Linear(1, 1),
        "codec": _Codec(),
        "build_condition_encoder": lambda **_kwargs: _Encoder(),
        "build_pipeline": build_pipeline,
        "default_negative_prompt": "spec-negative",
        "collate_encoded": collate_encoded,
        "pretrained_weights": PretrainedWeightSource(path="/published"),
        "noise_shift": 5.0,
        "fps": 24.0,
    }
    defaults.update(overrides)
    return WanEvalSpec(**defaults), calls


def _assemble(
    monkeypatch: pytest.MonkeyPatch,
    *,
    request_kwargs: dict[str, Any],
    spec_overrides: dict[str, Any] | None = None,
) -> tuple[Any, WanEvalSpec, dict[str, Any]]:
    spec, calls = _spec(**(spec_overrides or {}))

    def open_dataset(request: EvalRequest, **_kwargs: Any) -> _Dataset:
        validate_eval_source(request)
        return _Dataset()

    monkeypatch.setattr(
        "vforge.tasks.wan.eval.open_eval_dataset",
        open_dataset,
        raising=True,
    )
    monkeypatch.setattr(
        "vforge.tasks.wan.eval.build_inference_collate_fn",
        lambda **_kwargs: (
            lambda samples: {
                "image_condition": torch.zeros(1),
                "height": LATENT_H * SPATIAL,
                "width": LATENT_W * SPATIAL,
                "num_frames": (LATENT_FRAMES - 1) * TEMPORAL + 1,
            }
        ),
        raising=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers.schedulers",
        SimpleNamespace(FlowMatchEulerDiscreteScheduler=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr("vforge.distributed.runtime.get_runtime", lambda: _Runtime())
    components = assemble_eval_components(
        EvalRequest(**request_kwargs), extras=request_kwargs.get("extras", {}), spec=spec
    )
    components.build_inference(components.model, lambda *args, **kwargs: (args, kwargs))
    return components, spec, calls


def test_cached_evaluation_reads_through_the_version_collate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, spec, calls = _assemble(
        monkeypatch,
        request_kwargs={"data_source": "cache", "manifest": "/cache/manifest.json"},
    )

    next(iter(components.dataloader))

    assert calls["collated"] == 1
    assert spec.codec.canonicalized == 0


def test_validation_manifest_uses_condition_only_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, spec, calls = _assemble(
        monkeypatch,
        request_kwargs={"data_source": "validation", "manifest": "/m.json"},
    )

    prepared = next(iter(components.dataloader))

    assert spec.codec.canonicalized == 0
    assert calls["collated"] == 0
    assert "frame_latents" not in prepared["__encoded__"]


def test_inference_source_uses_condition_only_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, spec, _calls = _assemble(
        monkeypatch,
        request_kwargs={"data_source": "inference", "manifest": "/infer.json"},
    )

    prepared = next(iter(components.dataloader))

    assert spec.codec.canonicalized == 0
    assert prepared["__height__"] == LATENT_H * SPATIAL


def test_evaluation_requires_a_manifest_or_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="data_source and exactly one input"):
        _assemble(monkeypatch, request_kwargs={})


def test_evaluation_rejects_unknown_data_source(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unsupported evaluation.data_source"):
        _assemble(
            monkeypatch,
            request_kwargs={
                "data_source": "unknown",
                "manifest": "/source.json",
            },
        )


def test_eval_cache_requires_internal_manifest(tmp_path: Path) -> None:
    from vforge.tasks.eval_source import open_eval_dataset

    with pytest.raises(ValueError, match="manifest.json"):
        open_eval_dataset(
            EvalRequest(data_source="cache", manifest=str(tmp_path / "manifest.json")),
            codec=_Codec(),
        )


def test_eval_cache_requires_manifest_file_path(tmp_path: Path) -> None:
    from vforge.tasks.eval_source import open_eval_dataset

    with pytest.raises(ValueError, match="point directly"):
        open_eval_dataset(
            EvalRequest(data_source="cache", manifest=str(tmp_path)),
            codec=_Codec(),
        )


def test_eval_cache_rejects_incomplete_manifest(tmp_path: Path) -> None:
    from vforge.data.cache import (
        CACHE_SCHEMA_VERSION,
        CacheManifest,
        EncodedBatchCodec,
        write_cache_manifest,
    )
    from vforge.tasks.eval_source import open_eval_dataset

    codec = EncodedBatchCodec(name="test-eval", required_keys=("prompt",))
    write_cache_manifest(
        str(tmp_path),
        CacheManifest(
            schema_version=CACHE_SCHEMA_VERSION,
            sample_ids=["sample"],
            completed=0,
            encoding_fingerprint={},
            codec_name=codec.name,
            codec_schema_version=codec.schema_version,
        ),
    )

    with pytest.raises(ValueError, match="incomplete"):
        open_eval_dataset(
            EvalRequest(data_source="cache", manifest=str(tmp_path / "manifest.json")),
            codec=codec,
        )


def test_prepare_sample_converts_latent_extents_into_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, _spec, _calls = _assemble(
        monkeypatch,
        request_kwargs={"data_source": "cache", "manifest": "/cache/manifest.json"},
    )

    prepared = next(iter(components.dataloader))

    assert prepared["__height__"] == LATENT_H * SPATIAL
    assert prepared["__width__"] == LATENT_W * SPATIAL
    assert prepared["__num_frames__"] == (LATENT_FRAMES - 1) * TEMPORAL + 1
    # frame_latents is the sampler's target, not one of its inputs.
    assert "frame_latents" not in prepared["__encoded__"]
    assert prepared["__negative_prompt_embeds__"] is not None


def test_state_dict_mapper_reaches_the_weight_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    from vforge.training.checkpoint import StateDictMapper, TrainingCheckpointWeightSource

    mapper = StateDictMapper.strip_prefix("transformer.")
    components, _spec, _calls = _assemble(
        monkeypatch,
        request_kwargs={
            "data_source": "cache",
            "manifest": "/cache/manifest.json",
            "checkpoint_path": "/ckpt",
        },
        spec_overrides={"state_dict_mapper": mapper},
    )

    assert isinstance(components.initial_weights, TrainingCheckpointWeightSource)
    assert components.initial_weights.mapper is mapper


def test_evaluation_uses_published_weights_without_a_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components, _spec, _calls = _assemble(
        monkeypatch,
        request_kwargs={"data_source": "cache", "manifest": "/cache/manifest.json"},
    )

    assert isinstance(components.initial_weights, PretrainedWeightSource)
    assert components.initial_weights.path == "/published"


def test_shared_wan_eval_rejects_checkpoint_mapping_without_version_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="expects checkpoint_path to be a string"):
        _assemble(
            monkeypatch,
            request_kwargs={
                "data_source": "cache",
                "manifest": "/cache/manifest.json",
                "checkpoint_path": {"high": "/high", "low": "/low"},
            },
        )


def test_save_output_writes_an_mp4_and_a_json_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: dict[str, Any] = {}

    def fake_write(video, path, *, fps):
        written["fps"] = fps
        Path(path).write_bytes(b"mp4")

    monkeypatch.setattr("vforge.data.video_io.write_video_mp4", fake_write, raising=True)
    components, _spec, _calls = _assemble(
        monkeypatch,
        request_kwargs={
            "data_source": "cache",
            "manifest": "/cache/manifest.json",
            "extras": {"fps": 30.0},
        },
    )

    path = components.output_writer.write(
        {"sample_id": "s0", "video": torch.zeros(1, 3, 2, 2), "seed": 11},
        str(tmp_path),
        sample_id="s0",
        seed=11,
        provenance={"resolved_config_hash": "hash"},
    )

    assert Path(path).name == eval_artifact_name("s0")
    assert written["fps"] == 30.0
    sidecar = json.loads(Path(path + ".json").read_text())
    assert sidecar["sample_id"] == "s0"
    assert sidecar["seed"] == 11
    assert sidecar["provenance"]["resolved_config_hash"] == "hash"
