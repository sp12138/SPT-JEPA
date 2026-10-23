"""Opt-in end-to-end parity against trained research checkpoints.

REP_WORLD_LEGACY_CASES is a JSON array of {name, config, checkpoint, input,
extras} records. Paths refer to local, trusted assets; extras are the explicit
Native model configuration. REP_WORLD_REFERENCE_ROOT supplies the original
checkout only for this test, never for production. Both samplers consume the
same persisted RGB tensor, seed, 50-step Euler schedule and CFG interval.
The reference's optional FlashAttention fast path is disabled to compare both
implementations with Native's configured SDPA backend.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.shared import CodecProvider, parse_config
from vforge.tasks.rep_world.weights import load_legacy_model

CASES_PATH = os.environ.get("REP_WORLD_LEGACY_CASES")
REFERENCE = os.environ.get("REP_WORLD_REFERENCE_ROOT")
CASES = json.loads(Path(CASES_PATH).read_text()) if CASES_PATH else []


@pytest.mark.gpu(gpus=1)
@pytest.mark.skipif(not REFERENCE, reason="Requires explicit original reference checkout")
@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_trained_full_rollout_parity(case, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(REFERENCE)
    from rep_miniworld import miniworld
    from rep_miniworld.sample import (
        generate_full_history_latent_ar_rollout,
        generate_next_latent_semantic_renderer_rollout,
    )
    from rep_miniworld.train import (
        build_denoiser,
        build_runtime_semantic_encoder,
        parse_args,
        resolve_conditioning,
    )
    from rep_miniworld.vae.codec import load_latent_codec

    monkeypatch.setattr(miniworld, "flash_attn_func", None)

    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cuda"),
        parallel_state=None,
    )
    args = parse_args(["--config", case["config"]])
    resolve_conditioning(args, None)
    config = parse_config(case["extras"])
    steps, seed = 50, 1234
    guidance = 1 if config.flow.mode == "pure_tf" else 2
    rollout = RolloutConfig(
        total_states=config.states,
        trained_states=config.states,
        history_window_states=config.states,
        cfg_scale=guidance,
        steps=steps,
    )
    video = torch.load(case["input"], map_location="cpu", weights_only=True)["video"].cuda()
    assert video.shape[2] == config.states * config.geometry.rgb_frames
    payload = torch.load(case["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
    # The source loader resolves codec geometry onto args before the trainable
    # towers are built (notably EUPE's 768 rather than the generic 1024).
    old_codec = load_latent_codec(args)
    encoder = None
    if config.conditioner is not None:
        encoder = build_runtime_semantic_encoder(args, torch.device("cuda"))
        args.semantic_cond_dim = encoder.output_dim
        encoder.load_state_dict(payload["causal_semantic_encoder"], strict=True)
        encoder.eval().requires_grad_(False)
    original = build_denoiser(args).cuda().eval().requires_grad_(False)
    original.load_state_dict(payload["model"], strict=True)
    original.steps, original.cfg_scale = steps, guidance
    original.cfg_interval_min = rollout.cfg_interval_min
    original.cfg_interval_max = rollout.cfg_interval_max
    native = load_legacy_model(case["checkpoint"], config).cuda().eval().requires_grad_(False)
    assert original.timestep_shift == native.timestep_shift
    provider = CodecProvider(config, "bf16")
    codec = provider.get("cuda:0")
    inference = RepWorldInference(
        native,
        codec,
        config.geometry,
        config.flow,
        config.conditioner,
        native.timestep_shift,
        rollout,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefix = video[:, :, : config.geometry.rgb_frames].contiguous()
        observed = old_codec.encode(prefix)
        torch.testing.assert_close(observed, codec.encode(prefix), atol=0, rtol=0)
        if config.flow.mode == "pure_tf":
            geo = config.geometry
            explicit = observed.reshape(1, geo.planes, geo.channels, 1, geo.grid, geo.grid)
            explicit = explicit.permute(0, 2, 3, 1, 4, 5).flatten(2, 3).contiguous()
            zero = explicit.new_zeros(1, geo.planes)
            old_preview, old_cache = original.net.forward_with_cache(
                explicit, zero, None, chunk_size=geo.planes, return_kv=True
            )
            new_preview, new_cache = native(
                operation="world",
                noisy=explicit,
                timestep=zero,
                causal_state_planes=geo.planes,
                return_key_values=True,
            )
            for old_pair, new_pair in zip(old_cache, new_cache, strict=True):
                for old_value, new_value in zip(old_pair, new_pair, strict=True):
                    torch.testing.assert_close(new_value, old_value, atol=0, rtol=0)
            torch.testing.assert_close(new_preview, old_preview, atol=0, rtol=0)
        if encoder is not None:
            source = observed if config.conditioner.input == "latent" else prefix
            old_condition = encoder.predict_next_dense(source, initial_state_is_full=True)
            new_condition = native(operation="condition", source=source)
            torch.testing.assert_close(new_condition, old_condition, atol=0, rtol=0)
            if config.flow.mode == "no_zprev":
                geo = config.geometry
                noisy = torch.randn(
                    (1, geo.channels, geo.planes, geo.grid, geo.grid),
                    device="cuda",
                    dtype=old_condition.dtype,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
                time = noisy.new_full((1, geo.planes), 0.5)
                old_velocity = original.net(
                    noisy,
                    time,
                    noisy.new_empty(1, 1, 0),
                    semantic_query_cond=old_condition,
                )
                new_velocity = native(
                    operation="velocity", noisy=noisy, timestep=time, condition=new_condition
                )
                torch.testing.assert_close(new_velocity, old_velocity, atol=0, rtol=0)
        torch.cuda.manual_seed(seed)
        if config.flow.mode == "pure_tf":
            old_latents, old_rgb = generate_full_history_latent_ar_rollout(
                original, old_codec, observed, prefix, total_len=config.states, history_len=1
            )
        else:
            old_latents, old_rgb = generate_next_latent_semantic_renderer_rollout(
                denoiser=original,
                semantic_encoder=encoder,
                vae=old_codec,
                observed_latents=observed,
                cond_seq=observed.new_empty(1, config.states, 0),
                initial_rgb_history=prefix,
                total_len=config.states,
                history_len=1,
                semantic_history_window_frames=config.states * config.geometry.rgb_frames,
                initial_state_is_full=True,
            )
        rng = torch.cuda.get_rng_state()
        result = inference.infer_sample({"sample_id": case["name"], "rgb": video}, seed=seed)
        assert torch.equal(rng, torch.cuda.get_rng_state())
    report = {
        "case": case["name"],
        "seed": seed,
        "steps": steps,
        "cfg": guidance,
        "states": config.states,
        "frames": video.shape[2],
        "latent_max_error": (result["latents"] - old_latents).abs().max().item(),
        "rgb_max_error": (result["prediction"] - old_rgb).abs().max().item(),
    }
    (tmp_path / "parity.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    torch.testing.assert_close(result["latents"], old_latents, atol=0, rtol=0)
    torch.testing.assert_close(result["prediction"], old_rgb, atol=0, rtol=0)
    reset_ops()
