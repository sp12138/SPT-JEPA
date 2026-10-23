"""End-to-end CPU contracts for native Dense and MoE transformer variants."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import vforge.models.gen_engine.dit as native_dit
from vforge.distributed.parallel_state import ParallelState
from vforge.models.gen_engine.dit import GenEngineModel, GenEngineTransformer3DModel
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig


@pytest.fixture(autouse=True)
def _initialize_sdpa_ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=ParallelState(),
    )
    yield
    reset_ops()


def _config(*, architecture: str, num_refiner_layers: int = 1) -> dict:
    config = {
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": architecture,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "hidden_size": 8,
        "num_layers": 2,
        "num_refiner_layers": num_refiner_layers,
        "ffn_dim": 16,
        "in_channels": 1,
        "audio_in_channels": 3,
        "patch_size": (1, 1, 1),
        "text_dim": 4,
        "freq_dim": 4,
        "time_embed_hidden_dim": 8,
        "time_embed_dim": 4,
        "rope_freq_dim": 1,
        "max_reference_index": 4,
    }
    if architecture == "mmdit_moe":
        config["moe"] = {
            "start_layer": 1,
            "expert_intermediate_size": 4,
            "main_experts": 3,
            "main_top_k": 2,
            "block_experts": 2,
            "block_top_k": 1,
            "shared_multiplier": 2,
            "shared_gate": "sigmoid",
        }
    return config


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "hidden_states": torch.randn(1, 3, 1),
        "audio_hidden_states": torch.randn(1, 2, 3),
        "encoder_hidden_states": torch.randn(1, 2, 4),
        "timestep": torch.tensor([0.25, 0.75]),
        "timestep_indices": torch.tensor([0, 0, 0, 1, 1, 0, 1]),
        "token_tags": torch.tensor([1, 1, 0, 0, 0, 2, 2]),
        "position_ids": torch.randn(7, 3),
        "text_indices": torch.tensor([0, 1]),
        "video_indices": torch.tensor([2, 3, 4]),
        "audio_indices": torch.tensor([5, 6]),
        "reference_modality_ids": torch.tensor([0, 0, 1, 2, 0, 3, 0]),
        "reference_index_ids": torch.tensor([0, 0, 1, 2, 0, 3, 0]),
        "audio_plane_ids": torch.tensor([0, 0, 0, 0, 0, 1, 2]),
    }


@pytest.mark.parametrize("architecture", ["mmdit_dense", "mmdit_moe"])
def test_native_variants_run_training_backward(architecture: str) -> None:
    model = GenEngineTransformer3DModel(**_config(architecture=architecture)).train()

    output = model(**_inputs())
    loss = output.sample.square().mean() + output.audio_sample.square().mean()
    loss.backward()

    assert output.sample.shape == (1, 3, 1)
    assert output.audio_sample.shape == (1, 2, 3)
    assert model.proj_in.weight.grad is not None
    assert model.transformer_blocks[0].text_ffn.down.weight.grad is not None
    if architecture == "mmdit_moe":
        visual_moe = model.transformer_blocks[1].visual_moe
        assert visual_moe.router.main_router.weight.grad is not None
        assert any(expert.down.weight.grad is not None for expert in visual_moe.main_bank.experts)


def test_checkpoint_recomputation_emits_one_set_of_routing_statistics() -> None:
    model = GenEngineTransformer3DModel(**_config(architecture="mmdit_moe")).train()
    model.gradient_checkpointing_enable()

    output = model(**_inputs())
    (output.sample.sum() + output.audio_sample.sum()).backward()

    assert len(model._pending_moe_statistics) == 1
    statistics = model._pending_moe_statistics[1]
    assert statistics.main.counts.sum().item() == 6
    assert statistics.block.counts.sum().item() == 3


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("num_layers", [2, 3])
def test_microbatches_accumulate_moe_statistics_until_one_bias_update(
    recompute, num_layers
) -> None:
    torch.manual_seed(42)
    config = _config(architecture="mmdit_moe")
    config["num_layers"] = num_layers
    model = GenEngineTransformer3DModel(**config).train()
    if recompute:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad(set_to_none=True)
    router = model.transformer_blocks[1].visual_moe.router
    bias_before = router.main_routing_bias.clone()
    for index in range(3):
        output = model(**_inputs())
        ((output.sample.sum() + output.audio_sample.sum()) / 3).backward()
        assert set(model._pending_moe_statistics) == set(range(1, num_layers))
        for statistics in model._pending_moe_statistics.values():
            assert statistics.main.counts.sum().item() == 6 * (index + 1)
            assert statistics.block.counts.sum().item() == 3 * (index + 1)
            assert all(value.grad_fn is None for value in statistics.tensors())
        torch.testing.assert_close(router.main_routing_bias, bias_before, rtol=0, atol=0)
    # A no-grad forward must neither overwrite nor double-count pending training stats.
    with torch.no_grad():
        model(**_inputs())
    assert model._pending_moe_statistics[1].main.counts.sum().item() == 18
    optimizer.step()
    model.post_optimizer_step(completed_step=1, total_steps=10)
    assert model._pending_moe_statistics == {}
    assert model._last_moe_statistics[0].main.counts.sum().item() == 18
    after = router.main_routing_bias.clone()
    assert model.post_optimizer_step(completed_step=1, total_steps=10) == {}
    torch.testing.assert_close(router.main_routing_bias, after, rtol=0, atol=0)
    model(**_inputs())
    for statistics in model._pending_moe_statistics.values():
        assert statistics.main.counts.sum().item() == 6


def test_post_optimizer_step_consumes_statistics_updates_bias_and_returns_metrics() -> None:
    model = GenEngineTransformer3DModel(**_config(architecture="mmdit_moe")).train()
    output = model(**_inputs())
    (output.sample.sum() + output.audio_sample.sum()).backward()
    router = model.transformer_blocks[1].visual_moe.router
    before = router.block_routing_bias.clone()

    metrics = model.post_optimizer_step(
        completed_step=1,
        total_steps=10,
    )

    assert not torch.equal(router.block_routing_bias, before)
    assert "moe/layer_1/block/load_cv" in metrics
    assert "moe/layer_1/shared/gate_mean" in metrics
    assert "moe/layer_1/main/output_rms" in metrics
    assert "moe/layer_1/block/output_rms" in metrics
    assert "moe/layer_1/shared/output_rms" in metrics
    assert model.post_optimizer_step(completed_step=1, total_steps=10) == {}


def test_routing_bias_is_persistent_in_state_dict_roundtrip() -> None:
    model = GenEngineTransformer3DModel(**_config(architecture="mmdit_moe"))
    router = model.transformer_blocks[1].visual_moe.router
    with torch.no_grad():
        router.main_routing_bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        router.block_routing_bias.copy_(torch.tensor([4.0, 5.0]))
    state = model.state_dict()
    restored = GenEngineTransformer3DModel(**_config(architecture="mmdit_moe"))

    restored.load_state_dict(state)

    restored_router = restored.transformer_blocks[1].visual_moe.router
    torch.testing.assert_close(restored_router.main_routing_bias, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(restored_router.block_routing_bias, torch.tensor([4.0, 5.0]))
    inputs = _inputs()
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(**inputs)
        actual = restored(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample)


def test_linear_only_text_connector_and_inference_do_not_queue_statistics() -> None:
    model = GenEngineTransformer3DModel(
        **_config(architecture="mmdit_moe", num_refiner_layers=0)
    ).eval()
    assert isinstance(model.token_refiner.final_norm, torch.nn.Identity)

    with torch.no_grad():
        output = model(**_inputs())

    assert torch.isfinite(output.sample).all()
    assert model._pending_moe_statistics == {}


def test_reference_identity_modes_cover_none_modality_and_index() -> None:
    packed = torch.zeros(1, 2, 8)
    modality_ids = torch.tensor([0, 1])
    index_ids = torch.tensor([0, 1])
    outputs = {}
    for mode in ("none", "modality", "modality_index"):
        model = GenEngineTransformer3DModel(
            **{
                **_config(architecture="mmdit_dense"),
                "reference_identity_mode": mode,
            }
        )
        with torch.no_grad():
            model.reference_modality_embedding.weight[1].fill_(1.0)
            model.reference_index_embedding.weight[1].fill_(2.0)
        outputs[mode] = model._add_reference_identity(
            packed,
            modality_ids,
            index_ids,
        )

    torch.testing.assert_close(outputs["none"], packed)
    torch.testing.assert_close(outputs["modality"][0, 1], torch.ones(8))
    torch.testing.assert_close(
        outputs["modality_index"][0, 1],
        torch.full((8,), 3.0),
    )


def test_native_dit_has_no_h3_wan_diffusers_or_mcore_runtime_imports() -> None:
    source_root = Path(native_dit.__file__).parent
    imported_modules: list[str] = []
    for path in source_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
    forbidden = ("vforge.models.minimax", "vforge.models.wan", "diffusers", "megatron", "mcore")
    assert not any(
        module == name or module.startswith(f"{name}.")
        for module in imported_modules
        for name in forbidden
    )
    state_keys = GenEngineTransformer3DModel(**_config(architecture="mmdit_dense")).state_dict()
    assert all("h3" not in key.lower() and "wan" not in key.lower() for key in state_keys)
    wrapped_keys = GenEngineModel(
        GenEngineTransformer3DModel(**_config(architecture="mmdit_dense"))
    ).state_dict()
    assert wrapped_keys
    assert all(key.startswith("transformer.") for key in wrapped_keys)
