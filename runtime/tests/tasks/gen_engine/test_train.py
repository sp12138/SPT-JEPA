"""CPU contracts for the precomputed gen-engine training task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from vforge.data.cache import save_latent_bundle
from vforge.distributed.parallel_state import ParallelConfig, ParallelState
from vforge.models.gen_engine.packing import RuntimePacker
from vforge.tasks.gen_engine.shared import (
    LAYOUT_VERSION,
    PACKING_VERSION,
    _build_gen_engine_model,
    _transformer_config_digest,
)
from vforge.tasks.gen_engine.train import (
    _build_train_dataloader,
    _collate_runtime_samples,
    _gen_engine_train_step,
    build_row_timesteps,
    build_train_components,
)
from vforge.tasks.requests import DataSourceSpec, TrainRequest
from vforge.training.recipe import StepContext


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(path: Path, field: str, tensor: torch.Tensor) -> Path:
    save_latent_bundle({field: tensor}, str(path))
    return path


def _transformer_config(tmp_path: Path) -> Path:
    path = tmp_path / "transformer.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vforge.gen_engine.dit",
                "schema_version": 1,
                "architecture": "mmdit_dense",
                "num_attention_heads": 1,
                "attention_head_dim": 8,
                "hidden_size": 8,
                "num_layers": 1,
                "num_refiner_layers": 1,
                "ffn_dim": 16,
                "in_channels": 1,
                "audio_in_channels": 3,
                "patch_size": [1, 1, 1],
                "text_dim": 2,
                "freq_dim": 4,
                "time_embed_hidden_dim": 8,
                "time_embed_dim": 4,
                "rope_freq_dim": 1,
                "max_reference_index": 4,
            }
        ),
        encoding="utf-8",
    )
    return path


def _video_latent_codec_config(tmp_path: Path) -> Path:
    path = tmp_path / "video_latent_codec.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vforge.video_latent_codec",
                "schema_version": 1,
                "latent_channels": 1,
                "spatial_compression": 1,
                "normalization": {
                    "type": "channelwise_mean_std",
                    "mean": [0.0],
                    "std": [1.0],
                },
                "temporal": {
                    "input_rate_unit": "frames_per_second",
                    "mapping": {"type": "stride", "input_stride": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _audio_latent_codec_config(tmp_path: Path) -> Path:
    path = tmp_path / "audio_latent_codec.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vforge.audio_latent_codec",
                "schema_version": 1,
                "latent_channels": 3,
                "planes": 2,
                "normalization": {
                    "type": "channelwise_mean_std",
                    "mean": [0.0, 0.0, 0.0],
                    "std": [1.0, 1.0, 1.0],
                },
                "temporal": {
                    "input_rate_unit": "samples_per_second",
                    "mapping": {"type": "stride", "input_stride": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _manifest(
    tmp_path: Path,
    *,
    modalities: tuple[str, ...] = ("video", "audio"),
    digests: bool = True,
) -> Path:
    mme = _bundle(tmp_path / "mme.pt", "embedding", torch.randn(2, 2))
    target: dict[str, Any] = {}
    if "video" in modalities:
        video = _bundle(
            tmp_path / "video.pt",
            "latent",
            torch.randn(1, 2, 1, 1),
        )
        target["video"] = {
            "latent_path": str(video),
            "latent_shape": [1, 2, 1, 1],
            "latent_dtype": "float32",
            "effective_fps": 1.0,
        }
        if digests:
            target["video"]["latent_digest"] = _digest(video)
    if "audio" in modalities:
        audio = _bundle(
            tmp_path / "audio.pt",
            "latent",
            torch.randn(2, 3, 2),
        )
        target["audio"] = {
            "latent_path": str(audio),
            "latent_shape": [2, 3, 2],
            "latent_dtype": "float32",
            "effective_fps": 1.0,
        }
        if digests:
            target["audio"]["latent_digest"] = _digest(audio)
    mme_descriptor = {
        "embedding_path": str(mme),
        "shape": [2, 2],
        "dtype": "float32",
        "fingerprint": "mme-test-v1",
    }
    if digests:
        mme_descriptor["embedding_digest"] = _digest(mme)
    record = {
        "schema": "mmgen.sample",
        "schema_version": 1,
        "sample_id": "sample-1",
        "task": "t2v",
        "caption": "A test sample.",
        "mme": mme_descriptor,
        "references": [],
        "target": target,
    }
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _request(
    tmp_path: Path,
    *,
    manifest: Path | None = None,
    **overrides: Any,
) -> TrainRequest:
    values: dict[str, Any] = {
        "extras": {
            "transformer_config_path": str(_transformer_config(tmp_path)),
            "video_latent_codec_config_path": str(_video_latent_codec_config(tmp_path)),
            "audio_latent_codec_config_path": str(_audio_latent_codec_config(tmp_path)),
            "patch_size": [1, 1, 1],
        },
        "train_manifest": str(manifest or _manifest(tmp_path)),
        "pin_memory": False,
    }
    values.update(overrides)
    return TrainRequest(**values)


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.inputs: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs: torch.Tensor):
        self.inputs = kwargs
        return SimpleNamespace(
            sample=kwargs["hidden_states"] * self.scale,
            audio_sample=kwargs["audio_hidden_states"] * self.scale,
        )


@pytest.mark.parametrize("modalities", [("video",), ("audio",), ("video", "audio")])
def test_all_target_combinations_have_finite_losses(modalities: tuple[str, ...]) -> None:
    sample: dict[str, Any] = {
        "sample_id": "sample",
        "mme_embedding": torch.ones(2, 2),
        "references": [],
        "target": {},
    }
    if "video" in modalities:
        sample["target"]["video"] = {
            "latent": torch.ones(1, 2, 1, 1),
            "effective_fps": 1.0,
        }
    if "audio" in modalities:
        sample["target"]["audio"] = {
            "latent": torch.ones(2, 3, 2),
            "effective_fps": 1.0,
        }
    batch = _collate_runtime_samples(
        [sample],
        packer=RuntimePacker((1, 1, 1)),
        video_row_width=1,
        audio_row_width=3,
    )
    model = _FakeModel()

    output = _gen_engine_train_step(
        model,
        batch,
        3,
        StepContext(device=torch.device("cpu"), parallel_state=ParallelState(ParallelConfig())),
        seed=17,
        extras={"video_flow_shift": 12.0, "audio_flow_shift": 3.0},
    )

    assert torch.isfinite(output.loss)
    assert all(torch.isfinite(value) for value in output.metrics.values())
    assert bool(output.metrics["video_loss"] == 0) is ("video" not in modalities)
    assert bool(output.metrics["audio_loss"] == 0) is ("audio" not in modalities)
    assert bool(output.metrics["video_output_rms"] == 0) is ("video" not in modalities)
    assert bool(output.metrics["audio_output_rms"] == 0) is ("audio" not in modalities)
    output.loss.backward()
    assert torch.isfinite(model.scale.grad)

    sigma_video = output.metrics["sigma_video"]
    sigma_audio = output.metrics["sigma_audio"]
    base_from_video = sigma_video / (12.0 - 11.0 * sigma_video)
    base_from_audio = sigma_audio / (3.0 - 2.0 * sigma_audio)
    torch.testing.assert_close(base_from_video, base_from_audio)


def test_task_noise_advances_independently_of_step_and_restores_exactly() -> None:
    batch = _collate_runtime_samples(
        [
            {
                "sample_id": "rng",
                "mme_embedding": torch.ones(2, 2),
                "references": [],
                "target": {
                    "video": {"latent": torch.ones(1, 2, 1, 1), "effective_fps": 1.0},
                    "audio": {"latent": torch.ones(2, 3, 2), "effective_fps": 1.0},
                },
            }
        ],
        packer=RuntimePacker((1, 1, 1)),
        video_row_width=1,
        audio_row_width=3,
    )
    context = StepContext(torch.device("cpu"), ParallelState(ParallelConfig()))
    reference_context = StepContext(torch.device("cpu"), ParallelState(ParallelConfig()))

    def noise_inputs(step, step_context):
        model = _FakeModel()
        _gen_engine_train_step(model, batch, step, step_context, seed=42, extras={})
        return tuple(
            model.inputs[key].detach().clone() for key in ("hidden_states", "audio_hidden_states")
        )

    previous = []
    for index in range(4):
        saved = context.random_streams.state_dict()
        actual = noise_inputs(3, context)
        context.random_streams.load_state_dict(saved)
        replay = noise_inputs(300, context)
        expected = noise_inputs(index, reference_context)
        for first, second, baseline in zip(actual, replay, expected, strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first, baseline, rtol=0, atol=0)
        assert all(not torch.equal(actual[0], earlier) for earlier in previous)
        previous.append(actual[0])


def test_references_stay_before_targets_and_identity_ids_reach_model() -> None:
    reference_video = torch.tensor([[[[2.0]]]])
    reference_audio = torch.full((2, 3, 1), 3.0)
    sample = {
        "sample_id": "reference",
        "mme_embedding": torch.ones(2, 2),
        "references": [
            {"type": "image", "image": {"latent": reference_video}},
            {
                "type": "audio",
                "audio": {"latent": reference_audio, "effective_fps": 1.0},
            },
        ],
        "target": {
            "video": {
                "latent": torch.ones(1, 2, 1, 1),
                "effective_fps": 1.0,
            },
            "audio": {
                "latent": torch.ones(2, 3, 2),
                "effective_fps": 1.0,
            },
        },
    }
    batch = _collate_runtime_samples(
        [sample],
        packer=RuntimePacker((1, 1, 1)),
        video_row_width=1,
        audio_row_width=3,
    )
    model = _FakeModel()

    _gen_engine_train_step(
        model,
        batch,
        0,
        StepContext(device=torch.device("cpu"), parallel_state=ParallelState(ParallelConfig())),
        seed=1,
        extras={"reference_timestep": 1.0},
    )

    torch.testing.assert_close(
        model.inputs["hidden_states"][:, :1],
        batch["reference_video_rows"],
    )
    torch.testing.assert_close(
        model.inputs["audio_hidden_states"][:, :2],
        batch["reference_audio_rows"],
    )
    torch.testing.assert_close(
        model.inputs["encoder_hidden_states"],
        batch["mme_embedding"],
    )
    assert torch.equal(
        model.inputs["reference_modality_ids"],
        batch["reference_modality_ids"],
    )
    assert torch.equal(model.inputs["audio_plane_ids"], batch["audio_plane_ids"])
    assert torch.equal(
        model.inputs["reference_index_ids"],
        batch["reference_index_ids"],
    )


def test_row_timesteps_are_task_owned_for_each_stream() -> None:
    values, indices = build_row_timesteps(
        text_indices=torch.tensor([0, 1]),
        video_indices=torch.tensor([2, 3, 4]),
        audio_indices=torch.tensor([5, 6]),
        num_reference_video_rows=1,
        num_reference_audio_rows=1,
        video_timestep=0.2,
        audio_timestep=0.4,
        reference_timestep=0.9,
        text_timestep=1.0,
    )

    torch.testing.assert_close(values[indices], torch.tensor([1.0, 1.0, 0.9, 0.2, 0.2, 0.9, 0.4]))


def test_loader_uses_mmgen_manifest_and_records_resume_fingerprint(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    loader, fingerprint = _build_train_dataloader(request)
    batch = next(iter(loader))

    assert batch["mme_embedding"].shape == (1, 2, 2)
    assert fingerprint["source_kind"] == "mmgen_precomputed"
    assert fingerprint["manifest_digest"]
    assert fingerprint["packing"] == {
        "version": PACKING_VERSION,
        "patch_size": [1, 1, 1],
    }
    assert fingerprint["layout"]["version"] == LAYOUT_VERSION
    assert fingerprint["latent_codecs"]["video"]["digest"]
    assert fingerprint["latent_codecs"]["video"]["config"]["schema"] == "vforge.video_latent_codec"
    assert fingerprint["latent_codecs"]["audio"]["digest"]
    assert fingerprint["latent_codecs"]["audio"]["config"]["schema"] == "vforge.audio_latent_codec"
    assert fingerprint["model_dimensions"]["architecture"] == "mmdit_dense"
    assert fingerprint["model_dimensions"]["scratch_init_scheme"] == "mmdit_stable_v1"
    assert fingerprint["model_dimensions"]["in_channels"] == 1
    assert fingerprint["model_dimensions"]["audio_in_channels"] == 3
    assert fingerprint["model_dimensions"]["moe"] is None


def test_transformer_digest_distinguishes_dense_and_moe_architectures(
    tmp_path: Path,
) -> None:
    dense_path = _transformer_config(tmp_path)
    moe_path = tmp_path / "transformer-moe.json"
    moe_config = json.loads(dense_path.read_text(encoding="utf-8"))
    moe_config.update(
        {
            "architecture": "mmdit_moe",
            "num_layers": 2,
            "moe": {
                "start_layer": 1,
                "expert_intermediate_size": 4,
                "main_experts": 2,
                "main_top_k": 1,
                "block_experts": 1,
                "block_top_k": 1,
                "shared_multiplier": 1,
                "shared_gate": "sigmoid",
            },
        }
    )
    moe_path.write_text(json.dumps(moe_config), encoding="utf-8")

    dense_digest = _transformer_config_digest({"transformer_config_path": str(dense_path)})
    moe_digest = _transformer_config_digest({"transformer_config_path": str(moe_path)})

    assert dense_digest != moe_digest


def test_task_requires_digests_by_default_but_can_disable_them(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, digests=False)
    loader, _fingerprint = _build_train_dataloader(_request(tmp_path, manifest=manifest))
    with pytest.raises(ValueError, match="digest is required"):
        next(iter(loader))

    request = _request(tmp_path, manifest=manifest)
    request = TrainRequest(
        **{
            **request.__dict__,
            "extras": {**request.extras, "require_digests": False},
        }
    )
    _loader, fingerprint = _build_train_dataloader(request)
    assert fingerprint["require_digests"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"latent_cache_mode": "online"}, "latent_cache.mode=offline"),
        ({"batch_size": 2}, "data.batch_size=1"),
        ({"pretrained_path": "/models/published"}, "pretrained_path must be empty"),
    ],
)
def test_task_rejects_incompatible_training_modes(
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_train_components(_request(tmp_path, **overrides))


def test_task_rejects_cache_sources(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match="data.train_manifest and data.sources are mutually exclusive"
    ):
        _build_train_dataloader(
            _request(
                tmp_path,
                sources=(DataSourceSpec(cache_dir="/unused/cache"),),
                max_steps=1,
            )
        )


def test_linear_scheduler_uses_public_request_parameters(tmp_path):
    from dataclasses import replace

    request = replace(
        _request(tmp_path),
        scheduler_name="linear",
        learning_rate=1.0,
        scheduler_start_factor=0.8,
        scheduler_end_factor=0.2,
        scheduler_total_iters=2,
    )
    components = build_train_components(request)
    optimizer = components.optimizer_factory(torch.nn.Linear(1, 1))
    scheduler = components.scheduler_factory(optimizer)
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(2):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates == pytest.approx([0.8, 0.5, 0.2])


@pytest.mark.usefixtures("gen_engine_dit")
def test_native_train_dcp_resume_and_eval_round_trip(tmp_path, monkeypatch):
    from vforge.tasks.common import compose_standalone_config
    from vforge.tasks.eval import run_eval
    from vforge.tasks.gen_engine.shared import _gen_engine_state_dict_mapper
    from vforge.tasks.train import run_train
    from vforge.training.checkpoint import load_model_weights, read_manifest

    request = _request(tmp_path)
    common = [
        "experiment=gen_engine",
        f"model.extras.transformer_config_path={request.extras['transformer_config_path']}",
        (
            "model.extras.video_latent_codec_config_path="
            f"{request.extras['video_latent_codec_config_path']}"
        ),
        (
            "model.extras.audio_latent_codec_config_path="
            f"{request.extras['audio_latent_codec_config_path']}"
        ),
        "parallel.dp_mode=ddp",
        "attention.backend=sdpa",
        "attention.allow_fallback=false",
        "runtime.precision=fp32",
        "logging.wandb_enabled=false",
        "data.pin_memory=false",
        f"data.train_manifest={request.train_manifest}",
    ]
    resumed = tmp_path / "resumed"
    baseline = tmp_path / "baseline"
    run_train(
        compose_standalone_config(
            "train",
            common
            + [
                f"runtime.output_dir={resumed}",
                "runtime.max_steps=1",
                "checkpoint.async_save=false",
            ],
        )
    )
    first = resumed / "checkpoints/00001.dcp"
    assert read_manifest(str(first)).completed_steps == 1

    def reject_reinitialization(*_args, **_kwargs):
        pytest.fail("strict resume must not initialize scratch weights")

    with monkeypatch.context() as patch:
        patch.setattr(
            "vforge.tasks.gen_engine.train._initialize_gen_engine_model", reject_reinitialization
        )
        run_train(
            compose_standalone_config(
                "train",
                common
                + [
                    f"runtime.output_dir={resumed}",
                    "runtime.max_steps=2",
                    "checkpoint.async_save=false",
                    "resume.path=auto",
                ],
            )
        )
    run_train(
        compose_standalone_config(
            "train",
            common
            + [
                f"runtime.output_dir={baseline}",
                "runtime.max_steps=2",
                "checkpoint.async_save=false",
            ],
        )
    )
    weights = []
    for directory in (resumed, baseline):
        model = _build_gen_engine_model(
            extras=request.extras,
            init_device="meta",
            pretrained_path="",
        ).to_empty(device="cpu")
        load_model_weights(
            model,
            str(directory / "checkpoints/00002.dcp"),
            format="dcp",
            strict=True,
            state_dict_mapper=_gen_engine_state_dict_mapper(),
        )
        weights.append(model.state_dict())
    for name in weights[0]:
        torch.testing.assert_close(weights[0][name], weights[1][name], rtol=0, atol=0)

    output = tmp_path / "eval"
    run_eval(
        compose_standalone_config(
            "eval",
            common
            + [
                "checkpoint.enabled=false",
                f"runtime.output_dir={output}",
                "evaluation.data_source=validation",
                f"evaluation.manifest={request.train_manifest}",
                f"evaluation.checkpoint_path={resumed}/checkpoints/00002.dcp",
                "evaluation.num_inference_steps=2",
            ],
        )
    )
    result = torch.load(output / "samples/sample-1.pt", weights_only=True)
    assert result["video_latent"].shape == (1, 2, 1, 1)
    assert result["audio_latent"].shape == (2, 3, 2)
    assert torch.isfinite(result["video_latent"]).all()
    assert torch.isfinite(result["audio_latent"]).all()
