from __future__ import annotations

import json
import os

import pytest
import torch

from vforge.data.cache import save_latent_bundle
from vforge.tasks.gen_engine.eval import (
    _open_mmgen_dataset,
    build_eval_components,
    build_output_writer,
)
from vforge.tasks.requests import EvalRequest


def _inference_manifest(tmp_path) -> str:
    mme_path = tmp_path / "mme.pt"
    save_latent_bundle({"embedding": torch.ones(2, 3)}, str(mme_path))
    manifest = tmp_path / "samples.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema": "mmgen.sample",
                "schema_version": 1,
                "sample_id": "sample/one",
                "task": "t2v",
                "caption": "A test clip.",
                "mme": {
                    "embedding_path": str(mme_path),
                    "shape": [2, 3],
                    "dtype": "float32",
                    "fingerprint": "test",
                },
                "references": [],
                "target_spec": {
                    "modalities": ["video"],
                    "duration_seconds": 1,
                    "video": {
                        "height": 4,
                        "width": 4,
                        "fps": 4,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return str(manifest)


def _transformer_config(tmp_path) -> str:
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
                "text_dim": 3,
                "freq_dim": 4,
                "time_embed_hidden_dim": 8,
                "time_embed_dim": 4,
                "rope_freq_dim": 1,
                "max_reference_index": 4,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _video_latent_codec_config(tmp_path) -> str:
    path = tmp_path / "video_latent_codec.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vforge.video_latent_codec",
                "schema_version": 1,
                "latent_channels": 1,
                "spatial_compression": 2,
                "normalization": {
                    "type": "channelwise_mean_std",
                    "mean": [0.0],
                    "std": [1.0],
                },
                "temporal": {
                    "input_rate_unit": "frames_per_second",
                    "mapping": {"type": "stride", "input_stride": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _audio_latent_codec_config(tmp_path) -> str:
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
                    "mapping": {"type": "stride", "input_stride": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_eval_opens_precomputed_mmgen_inference_manifest(tmp_path) -> None:
    dataset = _open_mmgen_dataset(
        EvalRequest(
            checkpoint_path="/checkpoint",
            data_source="inference",
            manifest=_inference_manifest(tmp_path),
            extras={"require_digests": False},
        )
    )

    sample = dataset[0]
    assert sample["sample_id"] == "sample/one"
    assert sample["mme_embedding"].shape == (2, 3)
    assert sample["target_spec"]["modalities"] == ["video"]
    assert sample["__vforge_eval_index__"] == 0


@pytest.mark.parametrize("topology", [{"tp_size": 2}, {"cp_size": 2}])
def test_eval_builder_rejects_unsupported_runtime_topology(monkeypatch, topology) -> None:
    from vforge.distributed.parallel_state import ParallelConfig, ParallelState

    state = ParallelState(config=ParallelConfig(**topology))
    monkeypatch.setattr("vforge.tasks.gen_engine.eval.get_parallel_state", lambda: state)
    with pytest.raises(ValueError, match="tensor/context parallelism is not implemented"):
        build_eval_components(EvalRequest())


def test_eval_packs_before_broadcast_and_accepts_device_tensors(tmp_path, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    from vforge.data.loader import _rebuild_batch, _split_batch
    from vforge.models.gen_engine.packing import RuntimePacker

    components = build_eval_components(
        EvalRequest(
            checkpoint_path="/checkpoint",
            data_source="inference",
            manifest=_inference_manifest(tmp_path),
            extras={
                "transformer_config_path": _transformer_config(tmp_path),
                "video_latent_codec_config_path": _video_latent_codec_config(tmp_path),
                "audio_latent_codec_config_path": _audio_latent_codec_config(tmp_path),
                "require_digests": False,
            },
        )
    )
    sample = next(iter(components.dataloader))
    assert "layout" not in sample["packed"]
    assert "target_video_rows" not in sample["packed"]
    assert "target_audio_rows" not in sample["packed"]
    metadata, tensors = _split_batch(sample)
    assert tensors and all(tensor.device.type == "cpu" for tensor in tensors)
    inference = components.build_inference(torch.nn.Linear(1, 1), lambda **_: None)
    monkeypatch.setattr(
        RuntimePacker, "pack", lambda *_a, **_k: pytest.fail("packed after broadcast")
    )
    mode = FakeTensorMode()
    device_tensors = [
        FakeTensor(mode, torch.empty_like(tensor, device="meta"), torch.device("cuda:0"))
        for tensor in tensors
    ]
    with mode:
        received = _rebuild_batch(metadata, device_tensors)
        packed, geometry = inference._pack(received)
        assert packed["mme_embedding"].device.type == "cuda"
        assert geometry.visual.shape == (1, 2, 2, 2)


def test_validation_source_rejects_inference_only_records(tmp_path) -> None:
    dataset = _open_mmgen_dataset(
        EvalRequest(
            checkpoint_path="/checkpoint",
            data_source="validation",
            manifest=_inference_manifest(tmp_path),
            extras={"require_digests": False},
        )
    )

    with pytest.raises(ValueError, match="runtime requires target"):
        dataset[0]


def test_eval_requires_an_explicit_checkpoint_before_model_setup(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        build_eval_components(
            EvalRequest(
                data_source="inference",
                manifest=str(tmp_path / "samples.jsonl"),
            )
        )


def test_eval_rejects_cache_sources_before_model_setup(tmp_path) -> None:
    with pytest.raises(ValueError, match="inference or validation"):
        build_eval_components(
            EvalRequest(
                checkpoint_path="/checkpoint",
                data_source="cache",
                manifest=str(tmp_path / "manifest.json"),
            )
        )


def test_output_writer_atomically_saves_latents_and_geometry(
    monkeypatch,
    tmp_path,
) -> None:
    replacements: list[tuple[str, str]] = []
    real_replace = os.replace

    def replace(source: str, destination: str) -> None:
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr("vforge.tasks.gen_engine.eval.os.replace", replace)
    writer = build_output_writer()
    path = writer.write(
        {
            "sample_id": "sample/one",
            "seed": 17,
            "video_latent": torch.ones(1, 2, 2, 2),
            "audio_latent": torch.ones(2, 3, 4),
            "geometry": {
                "video": {"shape": [1, 2, 2, 2], "effective_fps": 2.0},
                "audio": {"shape": [2, 3, 4], "effective_fps": 4.0},
            },
            "metadata": {"split": "test"},
        },
        str(tmp_path),
        sample_id="sample/one",
        seed=17,
        provenance={"resolved_config_hash": "hash", "checkpoint": {"id": "checkpoint"}},
    )

    assert path == str(tmp_path / "sample_one.pt")
    assert (path + ".tmp", path) in replacements
    assert not os.path.exists(path + ".tmp")
    payload = torch.load(path, weights_only=True)
    assert payload["video_latent"].shape == (1, 2, 2, 2)
    assert payload["audio_latent"].shape == (2, 3, 4)
    assert payload["geometry"]["audio"]["effective_fps"] == 4.0
    assert payload["metadata"] == {"split": "test"}
