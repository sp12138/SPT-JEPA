"""DMD eval numerics and generator checkpoint isolation."""

from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module

from vforge.algorithms.dmd.model import RoleModelBundle
from vforge.evaluation.runner import _load_initial_weights
from vforge.hydra_config import config_from_dict
from vforge.models.ltx.kv_cache import CacheMode, KVCache
from vforge.tasks.ltx.dmd.eval import (
    GeneratorEvalModel,
    generator_weight_source,
    latent_frames_for_seconds,
)
from vforge.tasks.ltx.dmd.generator import generate
from vforge.tasks.ltx.dmd.inputs import prepare_batch
from vforge.tasks.requests import EvalRequest
from vforge.tasks.translate import eval_request


def config():
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        return compose(config_name="eval", overrides=["experiment=ltx23_dmd_eval"])


def test_eval_overrides_and_registry():
    from vforge.tasks.registry import resolve_builder

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name="eval",
            overrides=[
                "experiment=ltx23_dmd_eval",
                "model.extras.seconds=5",
                "model.extras.kv_cache_blocks=4",
                "model.extras.num_sink_blocks=1",
                "model.extras.prompt=hello",
                "parallel.usp_size=2",
                "runtime.deterministic=true",
                "evaluation.checkpoint_format=safetensors",
            ],
        )
    request = eval_request(config_from_dict(cfg))
    assert cfg.runtime.autocast is False
    assert cfg.parallel.mixed_precision.param_dtype == "bfloat16"
    assert request.extras["seconds"] == 5
    assert request.extras["num_sink_blocks"] == 1
    assert request.usp_size == 2
    assert callable(resolve_builder("ltx23", "eval"))


@pytest.mark.parametrize("seconds,frames", [(30, 91), (5, 16), (1, 4), (4.1, 13)])
def test_reference_duration_rounding(seconds, frames):
    assert latent_frames_for_seconds(seconds, 24) == frames


@pytest.mark.parametrize("window,sink", [(5, 0), (5, 4), (40, 4)])
def test_ring_matches_independent_token_oracle_and_preserves_snapshots(window, sink):
    cache = KVCache(window_tokens=window, sink_tokens=sink)
    cache.write(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), CacheMode.PREFILL_REF)
    pinned, ring, cursor, total = [], [], 0, 0
    snapshots = []
    for count in [4, 3, 3, 7, 2]:
        pair = torch.arange(total, total + count).float().reshape(1, count, 1).expand(-1, -1, 2)
        for item in range(total, total + count):
            if len(pinned) < sink:
                pinned.append(item)
            elif len(ring) < window:
                ring.append(item)
                cursor = len(ring) % window
            else:
                ring[cursor] = item
                cursor = (cursor + 1) % window
        total += count
        cache.write(pair, pair + 1, CacheMode.PREFILL_CLEAN)
        keys = torch.cat([p[0] for p in cache.clean], dim=1)
        expected = torch.tensor(pinned + ring).float().reshape(1, -1, 1).expand(-1, -1, 2)
        assert torch.equal(keys, expected)
        query = torch.tensor([[[0.25, 0.5]]])
        actual = torch.nn.functional.scaled_dot_product_attention(query, keys, keys + 1)
        oracle = torch.softmax(query @ expected.transpose(-1, -2) / 2**0.5, dim=-1) @ (expected + 1)
        torch.testing.assert_close(actual, oracle)
        for old, saved in snapshots:
            assert torch.equal(torch.cat([p[0] for p in old.clean], 1), saved)
        snapshots.append((cache.snapshot(), keys.clone()))
    with pytest.raises(RuntimeError, match="snapshot"):
        snapshots[0][0].write(pair, pair, CacheMode.PREFILL_CLEAN)


@pytest.mark.parametrize("format", ["pt", "dcp", "safetensors"])
def test_load_only_generator_from_training_or_reference_checkpoint(tmp_path, monkeypatch, format):
    from safetensors.torch import save_file

    from vforge.tasks.ltx.dmd import shared
    from vforge.training.checkpoint import CheckpointManifest, save_dcp

    source = RoleModelBundle(
        {"generator": torch.nn.Linear(2, 2), "fake_score": torch.nn.Linear(2, 2)}
    )
    target = GeneratorEvalModel(torch.nn.Linear(2, 2))
    path = str(tmp_path / ("00001.dcp" if format == "dcp" else f"model.{format}"))
    if format == "dcp":
        save_dcp(
            source,
            None,
            path,
            manifest=CheckpointManifest.create(state="weights", completed_steps=1),
        )
    elif format == "pt":
        torch.save(source.state_dict(), path)
    else:
        save_file(source.generator.state_dict(), path)
        monkeypatch.setattr(shared, "build_model", lambda **_: torch.nn.Linear(2, 2, device="meta"))
    request = EvalRequest(checkpoint_path=path, checkpoint_format=format, pretrained_path="unused")
    _load_initial_weights(target, generator_weight_source(request), weights_strict=True)
    for key, value in target.generator.state_dict().items():
        expected = source.generator.state_dict()[key]
        if format == "safetensors":
            expected = expected.bfloat16().float()
        assert torch.equal(value, expected)
    assert target.roles == ("generator",)


class TinyGenerator(torch.nn.Module):
    transformer_blocks = [None]

    def compute_block_cond_pe(self, *args):
        return None, None

    def forward(self, *, video, audio, cache, cache_mode):
        return video.latent * 0.25, audio.latent * 0.25


def prepared_batch(frames=7):
    options = dict(config().model.extras)
    options["max_train_frames"] = 113
    connectors = SimpleNamespace(create_embeddings=lambda video, audio, mask: (video, audio, mask))
    raw = {
        "video_latents": {"latents": torch.ones(128, 13, 1, 1), "fps": 24.0},
        "conditions": {
            "video_prompt_embeds": torch.ones(2, 4),
            "prompt_attention_mask": torch.ones(2),
        },
    }
    return prepare_batch(
        raw,
        torch.device("cpu"),
        config=options,
        connectors=connectors,
        negative={},
        latent_frames=frames,
    )


def test_rollout_matches_reference_per_block_rng_and_does_not_consume_global_rng():
    from vforge.models.ltx.scheduler import to_denoised
    from vforge.tasks.ltx.dmd.inputs import resolve_rollout_noise

    batch = prepared_batch()
    seed, steps = 17, [999, 757, 522, 0]
    state = torch.get_rng_state().clone()
    generated = generate(TinyGenerator(), batch, seed=seed, steps=steps)
    assert torch.equal(state, torch.get_rng_state())
    random = torch.Generator().manual_seed(seed)
    outputs = {"video": [], "audio": []}
    for video, audio in zip(
        batch.model_inputs.layout.video_ranges, batch.model_inputs.layout.audio_ranges, strict=True
    ):
        values = {
            key: torch.randn((1, b - a, 128), dtype=torch.bfloat16, generator=random)
            for key, (a, b) in (("video", video), ("audio", audio))
        }
        for index, step in enumerate(steps[:-1]):
            level = resolve_rollout_noise(None, torch.tensor(step / 1000), batch)
            values = {
                key: to_denoised(value, value * 0.25, level.model_timestep)
                for key, value in values.items()
            }
            if index < len(steps) - 2:
                sigma = resolve_rollout_noise(
                    None, torch.tensor(steps[index + 1] / 1000), batch
                ).model_timestep
                values = {
                    key: (
                        (1 - sigma.float().reshape(1, 1, 1)) * value
                        + sigma.float().reshape(1, 1, 1)
                        * torch.randn(value.shape, dtype=value.dtype, generator=random)
                    ).to(value.dtype)
                    for key, value in values.items()
                }
        for key in outputs:
            outputs[key].append(values[key])
    for key in outputs:
        assert torch.equal(generated[key], torch.cat(outputs[key], 1))


def test_long_tuning_loads_score_from_base_without_decoder_weights(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from vforge.tasks.ltx.dmd import shared
    from vforge.tasks.ltx.dmd.integration import build_weight_sources

    original = torch.nn.Linear(2, 2)
    path = str(tmp_path / "base.safetensors")
    save_file(
        {
            **{f"model.diffusion_model.{k}": v for k, v in original.state_dict().items()},
            "vae.encoder.weight": torch.ones(3),
        },
        path,
    )
    monkeypatch.setattr(shared, "build_model", lambda **_: torch.nn.Linear(2, 2, device="meta"))
    request = SimpleNamespace(
        training_models={
            "real_score": SimpleNamespace(
                pretrained_path=path, init_weights="", config={}, weight_loading="auto"
            )
        }
    )
    target = RoleModelBundle({"real_score": torch.nn.Linear(2, 2).bfloat16()})
    _load_initial_weights(
        target, build_weight_sources(request, ("real_score",)), weights_strict=True
    )
    for key, value in target.real_score.state_dict().items():
        assert torch.equal(value, original.state_dict()[key].bfloat16())
