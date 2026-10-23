from __future__ import annotations

from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.models.wan_taichu.inference.causal import (
    build_active_block_tf_per_token_timestep,
)
from vforge.models.wan_taichu.inference.incremental_kv import (
    build_incremental_kv_executor,
    forward_incremental_kv_block,
)
from vforge.models.wan_taichu.model import WanTaichuTransformer3DModel

PATCH_SIZE = (1, 2, 2)
Z_DIM = 2
LATENT_FRAMES = 3
LATENT_HEIGHT = LATENT_WIDTH = 4
TOKENS_PER_FRAME = 4


@pytest.fixture
def cpu_ops_runtime():
    from vforge.distributed.parallel_state import ParallelConfig
    from vforge.distributed.runtime import destroy_runtime, initialize_runtime
    from vforge.ops.config import OpsConfig

    initialize_runtime(
        ParallelConfig(),
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
    )
    try:
        yield
    finally:
        destroy_runtime()


def _tiny_transformer(num_layers: int = 1) -> WanTaichuTransformer3DModel:
    return WanTaichuTransformer3DModel(
        architecture="wan22_ti2v_5b",
        attention="block_causal_teacher",
        patch_size=PATCH_SIZE,
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=2 * Z_DIM,
        out_channels=Z_DIM,
        text_dim=4,
        freq_dim=16,
        ffn_dim=16,
        num_layers=num_layers,
        rope_max_seq_len=32,
        frame_per_block=1,
    ).eval()


def test_incremental_block_matches_full_recompute_without_processor_state(
    cpu_ops_runtime,
) -> None:
    torch.manual_seed(17)
    model = _tiny_transformer()
    executor = build_incremental_kv_executor(model, usp_size=1)
    module_calls = 0

    def count_module_call(_module, _args, _kwargs):
        nonlocal module_calls
        module_calls += 1

    model.register_forward_pre_hook(count_module_call, with_kwargs=True)
    clean = torch.randn(1, Z_DIM, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    condition = torch.randn_like(clean)
    prompt = torch.randn(1, 5, 4)
    active = torch.randn(1, Z_DIM, 1, LATENT_HEIGHT, LATENT_WIDTH)
    full_rope = executor.full_clip_rotary_emb(torch.cat([clean, condition], dim=1))
    cache = executor.init_kv_cache(
        batch_size=1,
        num_cache_tokens=LATENT_FRAMES * TOKENS_PER_FRAME,
        device="cpu",
        dtype=torch.float32,
    )

    executor.forward_kv_block(
        hidden_states=torch.cat([clean[:, :, :1], condition[:, :, :1]], dim=1),
        timestep=torch.zeros(1, TOKENS_PER_FRAME),
        encoder_hidden_states=prompt,
        kv_cache=cache,
        full_rotary_emb=full_rope,
        kv_write_start=0,
        commit_kv=True,
    )
    incremental = executor.forward_kv_block(
        hidden_states=torch.cat([active, condition[:, :, 1:2]], dim=1),
        timestep=torch.full((1, TOKENS_PER_FRAME), 500.0),
        encoder_hidden_states=prompt,
        kv_cache=cache,
        full_rotary_emb=full_rope,
        kv_write_start=TOKENS_PER_FRAME,
        commit_kv=False,
    )

    noisy = clean.clone()
    noisy[:, :, 1:2] = active
    doubled = torch.cat(
        [
            torch.cat([clean, condition], dim=1),
            torch.cat([noisy, condition], dim=1),
        ],
        dim=2,
    )
    timestep = build_active_block_tf_per_token_timestep(
        clean,
        torch.tensor(500.0),
        PATCH_SIZE,
        active_block=(1, 2),
    )
    full = model(
        hidden_states=doubled,
        timestep=timestep,
        encoder_hidden_states=prompt,
    )

    torch.testing.assert_close(incremental, full[:, :, 1:2], rtol=1e-5, atol=1e-6)
    assert module_calls == 3
    assert cache.committed_tokens == TOKENS_PER_FRAME
    processor = model.blocks[0].attn1.processor
    assert not hasattr(processor, "_kv_layer")
    assert not hasattr(processor, "_commit_kv")


def _rollout_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(23)
    clean = torch.randn(1, Z_DIM, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    return (
        clean,
        torch.randn_like(clean),
        torch.randn(1, 5, 4),
        torch.randn(LATENT_FRAMES, 1, Z_DIM, 1, LATENT_HEIGHT, LATENT_WIDTH),
    )


def _rollout_loss(model: WanTaichuTransformer3DModel, inputs) -> torch.Tensor:
    """Mirror the DMD rollout: no_grad commits around one grad forward per block."""
    clean, condition, prompt, block_noise = inputs
    executor = build_incremental_kv_executor(model, usp_size=1)
    full_rotary_emb = executor.full_clip_rotary_emb(torch.cat([clean, condition], dim=1))
    cache = executor.init_kv_cache(
        batch_size=1,
        num_cache_tokens=LATENT_FRAMES * TOKENS_PER_FRAME,
        device="cpu",
        dtype=clean.dtype,
    )

    def forward_block(value: torch.Tensor, start: int, timestep: float, commit: bool):
        return executor.forward_kv_block(
            torch.cat([value, condition[:, :, start : start + 1]], dim=1),
            torch.full((1, TOKENS_PER_FRAME), timestep),
            prompt,
            kv_cache=cache,
            full_rotary_emb=full_rotary_emb,
            kv_write_start=cache.committed_tokens,
            commit_kv=commit,
        )

    outputs = []
    for start in range(LATENT_FRAMES):
        if start == 0:
            accepted = clean[:, :, :1]
        else:
            noisy = block_noise[start]
            accepted = noisy - 0.3 * forward_block(noisy, start, 300.0, commit=False)
        outputs.append(accepted)
        with torch.no_grad():
            forward_block(accepted.detach(), start, 0.0, commit=True)
    return torch.cat(outputs, dim=2).square().mean()


@pytest.mark.parametrize("recompute", [False, True])
def test_fp32_weights_support_bf16_kv_rollout_and_optimizer_update(cpu_ops_runtime, recompute):
    model = _tiny_transformer().train()
    if recompute:
        model.enable_gradient_checkpointing(
            gradient_checkpointing_func=partial(checkpoint, use_reentrant=False)
        )
    inputs = tuple(value.bfloat16() for value in _rollout_inputs())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.patch_embedding.weight.detach().clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = _rollout_loss(model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(model.patch_embedding.weight, before)
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


def test_activation_recompute_reproduces_uncheckpointed_rollout_gradients(
    cpu_ops_runtime,
) -> None:
    """Recompute must replay each block against the history it first attended.

    The rollout commits more frames after a block runs, so a recomputed forward
    that reads the cache's live extent would attend future frames.
    """
    inputs = _rollout_inputs()
    results = {}
    for enabled in (False, True):
        torch.manual_seed(17)
        model = _tiny_transformer(num_layers=2)
        if enabled:
            model.enable_gradient_checkpointing(
                gradient_checkpointing_func=partial(checkpoint, use_reentrant=False)
            )
        assert bool(model.gradient_checkpointing) is enabled
        loss = _rollout_loss(model, inputs)
        loss.backward()
        results[enabled] = (
            loss.detach(),
            {name: param.grad.clone() for name, param in model.named_parameters()},
        )

    reference_loss, reference_grads = results[False]
    checkpoint_loss, checkpoint_grads = results[True]
    torch.testing.assert_close(checkpoint_loss, reference_loss, rtol=0, atol=0)
    assert set(checkpoint_grads) == set(reference_grads)
    for name, reference in reference_grads.items():
        torch.testing.assert_close(
            checkpoint_grads[name],
            reference,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda built, name=name: f"gradient mismatch for {name}\n{built}",
        )


def test_sliding_kv_window_rejects_activation_recompute(cpu_ops_runtime) -> None:
    """A sliding window compacts cache slots, so recompute cannot replay it."""
    torch.manual_seed(17)
    model = _tiny_transformer()
    model.enable_gradient_checkpointing(
        gradient_checkpointing_func=partial(checkpoint, use_reentrant=False)
    )
    clean, condition, prompt, _ = _rollout_inputs()
    executor = build_incremental_kv_executor(model, usp_size=1)
    cache = executor.init_kv_cache(
        batch_size=1,
        num_cache_tokens=LATENT_FRAMES * TOKENS_PER_FRAME,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(NotImplementedError, match="sliding KV"):
        forward_incremental_kv_block(
            model,
            torch.cat([clean[:, :, :1], condition[:, :, :1]], dim=1),
            torch.zeros(1, TOKENS_PER_FRAME),
            prompt,
            kv_cache=cache,
            full_rotary_emb=executor.full_clip_rotary_emb(torch.cat([clean, condition], dim=1)),
            kv_write_start=0,
            commit_kv=False,
            local_attn_size=1,
        )


def test_executor_passes_its_usp_size_to_the_block_forward(cpu_ops_runtime) -> None:
    """The cache is head-sharded by ``usp_size``; the forward must know that too.

    ``init_kv_cache`` stores ``num_heads // usp_size`` heads per rank. A forward
    that never receives the size builds full-width keys against that cache, and
    the mismatch only surfaces on a real multi-rank run.
    """
    from vforge.models.causal.executor import IncrementalKVExecutor

    seen: dict[str, object] = {}

    def record_block_forward(transformer, hidden_states, timestep, encoder_hidden_states, **kwargs):
        seen.update(kwargs)
        return hidden_states

    model = _tiny_transformer()
    executor = IncrementalKVExecutor(model, block_forward=record_block_forward, usp_size=2)
    clean, condition, prompt, _ = _rollout_inputs()
    block = torch.cat([clean[:, :, :1], condition[:, :, :1]], dim=1)
    executor.forward_kv_block(
        block,
        build_active_block_tf_per_token_timestep(
            block,
            torch.tensor(500.0),
            PATCH_SIZE,
            active_block=(0, 1),
        ),
        prompt,
        kv_cache=executor.init_kv_cache(
            batch_size=1,
            num_cache_tokens=LATENT_FRAMES * TOKENS_PER_FRAME,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        full_rotary_emb=executor.full_clip_rotary_emb(torch.cat([clean, condition], dim=1)),
        kv_write_start=0,
        commit_kv=True,
    )
    assert seen["usp_size"] == 2


@pytest.mark.parametrize("rank", [0, 1])
def test_kv_prologue_backpropagates_only_local_token_gradients(
    cpu_ops_runtime, monkeypatch, rank
) -> None:
    from types import SimpleNamespace

    from vforge.models.wan_taichu.inference import incremental_kv

    group = object()
    monkeypatch.setattr(
        incremental_kv, "get_parallel_state", lambda: SimpleNamespace(usp_group=group)
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: rank)
    real_slice = incremental_kv.slice_ulysses_sequence_locally
    slices = []

    def record_slice(tensor, *, dim, group):
        local = real_slice(tensor, dim=dim, group=group)
        if tensor.requires_grad:
            slices.append((tensor, local))
        return local

    class PrologueComplete(Exception):
        pass

    def stop_before_attention(*args, **kwargs):
        raise PrologueComplete

    monkeypatch.setattr(incremental_kv, "slice_ulysses_sequence_locally", record_slice)
    monkeypatch.setattr(incremental_kv, "run_transformer_blocks", stop_before_attention)
    torch.manual_seed(17)
    model = _tiny_transformer()
    clean, condition, prompt, _ = _rollout_inputs()
    executor = build_incremental_kv_executor(model, usp_size=2)
    cache = executor.init_kv_cache(
        batch_size=1,
        num_cache_tokens=LATENT_FRAMES * TOKENS_PER_FRAME,
        device="cpu",
        dtype=torch.float32,
    )
    with pytest.raises(PrologueComplete):
        forward_incremental_kv_block(
            model,
            torch.cat([clean[:, :, :1], condition[:, :, :1]], dim=1),
            torch.zeros(1, TOKENS_PER_FRAME),
            prompt,
            kv_cache=cache,
            full_rotary_emb=executor.full_clip_rotary_emb(torch.cat([clean, condition], dim=1)),
            kv_write_start=0,
            commit_kv=False,
            usp_size=2,
        )

    assert len(slices) == 3  # patch tokens, timestep projection, output timestep embedding
    for full, local in slices:
        gradient = torch.autograd.grad(local.sum(), full, retain_graph=True)[0]
        expected = torch.zeros_like(full)
        expected.chunk(2, dim=1)[rank].fill_(1)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
