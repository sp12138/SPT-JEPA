"""A tiny two-expert A14B end to end: build, widen, load, step, alternate.

Everything here runs on CPU against a miniature checkpoint. It is the only
place the plugin is exercised as a whole -- meta construction, per-expert weight
loading through the narrowed sources, and a real training step landing on the
expert the step parity picked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

Z_DIM = 4
HEADS = 2
HEAD_DIM = 16

_EXTRAS = {
    "noise_shift": 3.0,
    "boundary_ratio": 0.9,
    "text_drop_prob": 0.0,
    "rgb_cond_drop_prob": 0.0,
    "rgb_first_frame_drop_prob": 0.0,
    "normal_drop_prob": 0.0,
}


def _tiny_config(in_channels: int) -> dict[str, object]:
    return {
        "_class_name": "WanTransformer3DModel",
        "in_channels": in_channels,
        "out_channels": Z_DIM,
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "num_layers": 1,
        "ffn_dim": 32,
        "text_dim": 16,
        "freq_dim": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "cross_attn_norm": True,
        "eps": 1e-6,
        "image_dim": None,
        "added_kv_proj_dim": None,
        "rope_max_seq_len": 32,
        "use_vae_condition": False,
    }


@pytest.fixture
def cpu_ops_runtime():
    """vforge attention refuses to dispatch until a runtime exists."""
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


@pytest.fixture
def checkpoint(tmp_path):
    """A config-plus-weights export shaped like the published A14B directory."""
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    published_in = Z_DIM + (4 + Z_DIM)  # the single rgb condition stream
    root = tmp_path / "a14b"
    torch.manual_seed(0)
    for index, subfolder in enumerate(("transformer", "transformer_2")):
        directory = root / subfolder
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(
            json.dumps(_tiny_config(published_in)),
            encoding="utf-8",
        )
        expert = WanTransformer3DModel(**_tiny_config(published_in))
        with torch.no_grad():
            # Give the two towers different weights so routing is observable.
            for parameter in expert.parameters():
                parameter.add_(float(index))
        torch.save(expert.state_dict(), directory / "weights.pt")
    return root


def _expert_factory(checkpoint, subfolder: str):
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    def build() -> nn.Module:
        config = json.loads((checkpoint / subfolder / "config.json").read_text(encoding="utf-8"))
        expert = WanTransformer3DModel(**config)
        expert.load_state_dict(torch.load(checkpoint / subfolder / "weights.pt"))
        return expert

    return build


def _load_experts(model, checkpoint, *, layouts, expert_names=("wan", "wan_2")) -> None:
    """Populate selected towers the way the trainer does, one source at a time."""
    from vforge.models.loading import load_pretrained_weights
    from vforge.models.wan.v22.a14b.model import EXPERT_SUBFOLDERS
    from vforge.tasks.wan.v22.a14b.train import _a14b_pretrained_sources

    sources = _a14b_pretrained_sources(
        layouts=layouts,
        pretrained_path=str(checkpoint),
        weight_loading="rank0_broadcast",
        expert_names=expert_names,
    )
    for source, name in zip(sources, expert_names, strict=True):
        load_pretrained_weights(
            model,
            SimpleNamespace(
                **{
                    **source.__dict__,
                    "fallback_model_factory": _expert_factory(
                        checkpoint,
                        EXPERT_SUBFOLDERS[name],
                    ),
                }
            ),
        )


def _build_model(checkpoint, *, streams: list[str], expert_names=("wan", "wan_2")):
    from vforge.distributed.parallelize import materialize_meta_module
    from vforge.models.wan.v22.a14b.model import Wan22A14bModel
    from vforge.tasks.wan.v22.a14b.shared import _resolve_expert_configs

    configs, layouts = _resolve_expert_configs(
        {"condition_streams": streams},
        str(checkpoint),
        expert_names=expert_names,
    )
    model = Wan22A14bModel.build_empty(
        configs,
        boundary_ratio=0.9,
        expert_names=expert_names,
    )
    materialize_meta_module(model, torch.device("cpu"))
    _load_experts(model, checkpoint, layouts=layouts, expert_names=expert_names)
    return model


def _encoded_batch(streams: list[str]) -> dict[str, object]:
    return {
        "prompt": torch.ones(1, 3, 16),
        "frame_latents": torch.randn(1, Z_DIM, 1, 4, 4),
        "image_condition_latent_vae": None,
        "image_condition_latent_clip": None,
        "image_condition_latent_dino": None,
        "condition_latents_normal": torch.randn(1, len(streams) * (4 + Z_DIM), 1, 4, 4),
        "sample_id": ["sample-0"],
    }


def _step_context():
    from vforge.training.recipe import StepContext

    return StepContext(
        device=torch.device("cpu"),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(usp_size=1),
            usp_rank=0,
            global_rank=0,
            ulysses_group=None,
            usp_group=None,
        ),
    )


def test_both_towers_load_their_own_published_weights(checkpoint) -> None:
    from vforge.models.wan.v22.a14b.model import EXPERT_SUBFOLDERS

    model = _build_model(checkpoint, streams=["rgb"])

    high = model.wan.blocks[0].ffn.net[0].proj.weight
    low = model.wan_2.blocks[0].ffn.net[0].proj.weight
    assert not torch.equal(high, low)

    # Each tower has to carry its own subfolder's weights, not a copy of the
    # other's: one source loading over both would pass a "they differ" check
    # only until the second load overwrote the first.
    for name, subfolder in EXPERT_SUBFOLDERS.items():
        published = torch.load(checkpoint / subfolder / "weights.pt")
        loaded = getattr(model, name).state_dict()
        for key, expected in published.items():
            assert torch.equal(loaded[key], expected), f"{name}.{key}"


@pytest.mark.parametrize(
    ("expert_names", "present", "absent"),
    [(("wan",), "wan", "wan_2"), (("wan_2",), "wan_2", "wan")],
)
def test_single_expert_build_loads_only_its_tower(
    checkpoint, expert_names, present, absent
) -> None:
    model = _build_model(checkpoint, streams=["rgb"], expert_names=expert_names)

    assert getattr(model, present) is not None
    assert getattr(model, absent) is None
    assert all(key.startswith(present + ".") for key in model.state_dict())


def test_widened_tower_keeps_published_channels_and_starts_neutral(checkpoint) -> None:
    model = _build_model(checkpoint, streams=["rgb", "normal"])
    published_in = Z_DIM + (4 + Z_DIM)

    for expert in model.fsdp_execution_roots():
        weight = expert.patch_embedding.weight
        assert weight.shape[1] == Z_DIM + 2 * (4 + Z_DIM)
        assert torch.count_nonzero(weight[:, published_in:]) == 0
        assert torch.count_nonzero(weight[:, :published_in]) > 0


def test_a_training_step_reaches_the_expert_its_parity_selected(
    checkpoint,
    cpu_ops_runtime,
) -> None:
    from vforge.tasks.wan.v22.a14b.shared import WAN22_A14B_ENCODED_BATCH_CODEC
    from vforge.tasks.wan.v22.a14b.train import _a14b_train_step, _A14bConditionProcessor

    streams = ["rgb"]
    model = _build_model(checkpoint, streams=streams)
    processor = _A14bConditionProcessor(
        extras={**_EXTRAS, "condition_streams": streams},
        precision="fp32",
    )
    torch.manual_seed(0)
    batch = _encoded_batch(streams)

    losses = {}
    for step in (0, 1):
        output = _a14b_train_step(
            model,
            batch,
            step,
            _step_context(),
            latent_cache_mode="offline",
            seed=42,
            codec=WAN22_A14B_ENCODED_BATCH_CODEC,
            processor=processor,
            encoder=None,
        )
        output.loss.backward()
        losses[step] = float(output.metrics["loss_scalar"])
        assert output.workload["local_tokens"] == 1 * 1 * 2 * 2

    assert all(value > 0 for value in losses.values())
    # Step 0 trains high noise, step 1 low noise, so each tower took gradient
    # exactly once and neither was left out.
    assert model.wan.patch_embedding.weight.grad is not None
    assert model.wan_2.patch_embedding.weight.grad is not None


def test_only_the_selected_expert_takes_gradient_in_one_step(checkpoint, cpu_ops_runtime) -> None:
    from vforge.tasks.wan.v22.a14b.shared import WAN22_A14B_ENCODED_BATCH_CODEC
    from vforge.tasks.wan.v22.a14b.train import _a14b_train_step, _A14bConditionProcessor

    streams = ["rgb"]
    model = _build_model(checkpoint, streams=streams)
    processor = _A14bConditionProcessor(
        extras={**_EXTRAS, "condition_streams": streams},
        precision="fp32",
    )
    torch.manual_seed(0)

    output = _a14b_train_step(
        model,
        _encoded_batch(streams),
        0,
        _step_context(),
        latent_cache_mode="offline",
        seed=42,
        codec=WAN22_A14B_ENCODED_BATCH_CODEC,
        processor=processor,
        encoder=None,
    )
    output.loss.backward()

    assert model.wan.patch_embedding.weight.grad is not None
    assert model.wan_2.patch_embedding.weight.grad is None
