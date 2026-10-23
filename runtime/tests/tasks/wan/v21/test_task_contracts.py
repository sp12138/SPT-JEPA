"""Small-model contracts for Wan21 encode/process/step/sampling boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def _encoded_batch() -> dict[str, torch.Tensor | None]:
    return {
        "prompt": torch.ones(1, 3, 4),
        "frame_latents": torch.arange(
            8,
            dtype=torch.float32,
        ).reshape(1, 2, 1, 2, 2),
        "image_condition_latent_vae": None,
        "image_condition_latent_clip": None,
        "image_condition_latent_dino": None,
        "condition_latents_normal": torch.ones(1, 1, 1, 2, 2),
    }


_EXTRAS = {
    "noise_shift": 1.0,
    "text_drop_prob": 0.0,
    "rgb_cond_drop_prob": 0.0,
    "normal_drop_prob": 0.0,
}


def _assert_tensor_tree_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tensor_tree_equal(left[key], right[key])
    else:
        assert left == right


def test_constant_with_warmup_matches_taichu_learning_rates() -> None:
    from vforge.utils.training import build_scheduler_factory

    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-5)
    scheduler = build_scheduler_factory(
        name="constant_with_warmup",
        factor=1.0,
        warmup_steps=100,
    )(optimizer)

    assert optimizer.param_groups[0]["lr"] == 0.0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-7)


def test_checkpoint_key_mapping_is_shared_and_roundtrips():
    from vforge.training.checkpoint import StateDictMapper

    external = {"blocks.0.weight": torch.ones(1)}
    mapper = StateDictMapper.strip_prefix("transformer.")
    wrapped = mapper.import_(external)

    assert set(wrapped) == {"transformer.blocks.0.weight"}
    restored = mapper.export(wrapped)
    assert restored.keys() == external.keys()
    assert torch.equal(restored["blocks.0.weight"], external["blocks.0.weight"])


def test_wan_eval_artifact_name_uses_mp4_suffix() -> None:
    from vforge.tasks.wan.eval import eval_artifact_name

    assert eval_artifact_name("sample/one").endswith(".mp4")


def test_parallelizer_targets_wan_transformer_gradient_checkpointing() -> None:
    from vforge.distributed.parallelize import (
        FSDP2Config,
        ParallelizeConfig,
        parallelize_model,
    )
    from vforge.models.wan.v21.model import Wan21Model

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.checkpoint_callable = None

        def enable_gradient_checkpointing(
            self,
            *,
            gradient_checkpointing_func=None,
        ) -> None:
            self.checkpoint_callable = gradient_checkpointing_func

    transformer = Transformer()
    model = Wan21Model(transformer)

    parallelize_model(
        model,
        ParallelizeConfig(
            fsdp2=FSDP2Config(enable=False),
            gradient_checkpointing=True,
        ),
    )

    assert callable(transformer.checkpoint_callable)
    assert transformer.checkpoint_callable.keywords == {"use_reentrant": False}


def test_parallelizer_injects_manager_aware_checkpoint_callable() -> None:
    from vforge.distributed.parallelize import (
        FSDP2Config,
        ParallelizeConfig,
        parallelize_model,
    )
    from vforge.models.wan.v21.model import Wan21Model

    class Transformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.checkpoint_callable = None

        def enable_gradient_checkpointing(
            self,
            *,
            gradient_checkpointing_func=None,
        ) -> None:
            self.checkpoint_callable = gradient_checkpointing_func

    context_fn = object()
    transformer = Transformer()
    model = Wan21Model(transformer)
    parallelize_model(
        model,
        ParallelizeConfig(
            fsdp2=FSDP2Config(enable=False),
            gradient_checkpointing=True,
        ),
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
            "context_fn": context_fn,
        },
    )

    assert transformer.checkpoint_callable.keywords == {
        "use_reentrant": False,
        "context_fn": context_fn,
    }


def test_online_and_offline_condition_processing_are_identical() -> None:
    from vforge.tasks.wan.v21.train import _ConditionProcessor

    encoded = _encoded_batch()
    torch.manual_seed(123)
    offline = _ConditionProcessor(extras=_EXTRAS, precision="fp32").process(
        encoded,
        step=0,
        device=torch.device("cpu"),
        apply_dropouts=False,
    )
    torch.manual_seed(123)
    online = _ConditionProcessor(extras=_EXTRAS, precision="fp32").process(
        encoded,
        step=0,
        device=torch.device("cpu"),
        apply_dropouts=False,
    )
    _assert_tensor_tree_equal(offline, online)


def test_online_and_offline_first_step_loss_are_identical() -> None:
    from vforge.tasks.wan.v21.train import _ConditionProcessor, _wan_train_step
    from vforge.training.recipe import StepContext

    class Encoder:
        def encode_batch(self, _batch):
            return _encoded_batch()

        def apply_training_dropouts(self, bundle):
            return bundle

        def eval(self):
            return self

    class Model(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return (torch.zeros_like(hidden_states[:, :2]),)

    context = StepContext(
        device=torch.device("cpu"),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(usp_size=1),
            global_rank=0,
            batch_group=None,
            is_batch_leader=True,
        ),
    )
    common = {
        "model": Model(),
        "step": 0,
        "context": context,
        "probe_encoder": [None],
        "probe_collate": None,
        "probe_samples": {},
        "seed": 42,
    }
    torch.manual_seed(456)
    saved_streams = context.random_streams.state_dict()
    offline = _wan_train_step(
        batch={**_encoded_batch(), "sample_id": ["sample"]},
        latent_cache_mode="offline",
        processor=_ConditionProcessor(extras=_EXTRAS, precision="fp32"),
        encoder=None,
        **common,
    )
    torch.manual_seed(456)
    context.random_streams.load_state_dict(saved_streams)
    online = _wan_train_step(
        batch={"raw": True, "sample_id": ["sample"]},
        latent_cache_mode="online",
        processor=_ConditionProcessor(extras=_EXTRAS, precision="fp32"),
        encoder=Encoder(),
        **common,
    )
    assert torch.equal(offline.loss, online.loss)
    assert offline.metadata == online.metadata


def test_weighted_mse_uses_explicit_fp32_loss_boundary() -> None:
    from vforge.tasks.wan.v21.train import _weighted_mse_loss

    output = torch.tensor([1.5, -0.5], dtype=torch.bfloat16)
    targets = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    weights = torch.tensor([1.0, 0.5], dtype=torch.bfloat16)

    loss = _weighted_mse_loss(
        output,
        targets,
        weights,
    )

    assert loss.dtype is torch.float32
    expected = weights.float() * (output - targets).float().square()
    assert torch.equal(loss, expected.mean())


def test_condition_dropout_splits_rgbnormal_blocks() -> None:
    from vforge.models.wan.v21.encoder import apply_encoded_training_dropouts

    bundle = _encoded_batch()
    bundle["condition_latents_normal"] = torch.ones(1, 40, 1, 1, 1)
    result = apply_encoded_training_dropouts(
        bundle,
        rgb_first_frame_drop_prob=1.0,
        rgb_cond_drop_prob=0.0,
        normal_drop_prob=0.0,
        text_drop_prob=0.0,
        streams=("rgb", "normal"),
    )
    condition = result["condition_latents_normal"]
    assert torch.count_nonzero(condition[:, :20]) == 0
    assert torch.count_nonzero(condition[:, 20:]) == 20


def test_sampling_is_fixed_seed_without_decode() -> None:
    from vforge.models.wan.v21.inference import Wan21Pipeline

    class Transformer(torch.nn.Module):
        config = SimpleNamespace(in_channels=2)

        @property
        def dtype(self):
            return torch.float32

        def cache_context(self, _name):
            return nullcontext()

        def forward(self, hidden_states, **_kwargs):
            return (torch.zeros_like(hidden_states),)

    class Scheduler:
        def set_timesteps(self, _steps, device):
            self.timesteps = [torch.tensor(1.0, device=device)]

        def step(self, _prediction, _timestep, latents, return_dict):
            assert return_dict is False
            return (latents,)

    transformer = Transformer()
    pipeline = Wan21Pipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        engine_forward=transformer,
        scheduler=Scheduler(),
    )
    prompt = torch.ones(1, 3, 4)
    first = pipeline.call_with_latent(
        prompt,
        y=None,
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(99),
        decode_output=False,
    )
    second = pipeline.call_with_latent(
        prompt,
        y=None,
        height=16,
        width=16,
        num_frames=5,
        num_inference_steps=1,
        guidance_scale=1.0,
        generator=torch.Generator().manual_seed(99),
        decode_output=False,
    )
    assert torch.equal(first, second)


def test_native_wan_processor_explicitly_requests_distributed_attention(
    monkeypatch,
) -> None:
    import vforge.models.wan.attention as attention_module
    from vforge.models.wan.v21.transformer import (
        WanUlyssesSelfAttentionProcessor,
    )

    captured = {}

    def attention(query, key, value, **kwargs):
        captured.update(kwargs)
        return query

    monkeypatch.setattr(attention_module, "vforge_attention", attention)
    identity = torch.nn.Identity()
    attn = SimpleNamespace(
        fused_projections=False,
        is_cross_attention=False,
        norm_q=identity,
        norm_k=identity,
        to_q=identity,
        to_k=identity,
        to_v=identity,
        heads=2,
        to_out=(identity, identity),
    )
    hidden_states = torch.randn(1, 3, 4)

    output = WanUlyssesSelfAttentionProcessor()(attn, hidden_states)

    assert output.shape == hidden_states.shape
    # The ops layer selects Ulysses or MCore CP from the active topology.
    assert captured["distributed"] is True


def test_native_wan_processor_matches_non_sp_attention(monkeypatch) -> None:
    import vforge.models.wan.attention as attention_module
    import vforge.models.wan.v21.transformer as transformer_module
    from vforge.models.wan.v21.transformer import (
        WanAttnProcessor,
        WanVforgeSelfAttentionProcessor,
    )

    def attention(query, key, value, **_kwargs):
        return query + key + value

    monkeypatch.setattr(transformer_module, "vforge_attention", attention)
    monkeypatch.setattr(attention_module, "vforge_attention", attention)
    identity = torch.nn.Identity()
    attn = SimpleNamespace(
        fused_projections=False,
        is_cross_attention=False,
        add_k_proj=None,
        norm_q=identity,
        norm_k=identity,
        to_q=identity,
        to_k=identity,
        to_v=identity,
        heads=2,
        to_out=(identity, identity),
    )
    hidden_states = torch.randn(1, 3, 4)

    expected = WanAttnProcessor()(attn, hidden_states)
    actual = WanVforgeSelfAttentionProcessor()(attn, hidden_states)

    assert torch.equal(actual, expected)
