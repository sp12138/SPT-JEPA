"""Packed MMDiT layout, initialization and single-GPU numerical contracts."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from vforge.distributed.expert_parallel import Expert
from vforge.distributed.parallel_state import ParallelState
from vforge.distributed.parallelize import materialize_meta_module
from vforge.models.gen_engine.dit import GenEngineModel
from vforge.models.gen_engine.dit.moe import (
    ExpertBank,
    TorchReferenceDispatcher,
    VisualMoE,
    _dispatch_experts,
)
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "vforge/models/gen_engine/dit/configs"


def _config(dispatcher="vforge_expert", *, block_experts=0):
    return {
        "schema": "vforge.gen_engine.dit",
        "schema_version": 1,
        "architecture": "mmdit_moe",
        "hidden_size": 32,
        "num_attention_heads": 2,
        "attention_head_dim": 16,
        "num_layers": 2,
        "num_refiner_layers": 1,
        "ffn_dim": 64,
        "in_channels": 1,
        "audio_in_channels": 3,
        "patch_size": [1, 1, 1],
        "text_dim": 4,
        "freq_dim": 4,
        "time_embed_hidden_dim": 32,
        "time_embed_dim": 8,
        "rope_freq_dim": 1,
        "moe": {
            "start_layer": 1,
            "expert_intermediate_size": 64,
            "main_experts": 8,
            "main_top_k": 2,
            "block_experts": block_experts,
            # Top-1 normalization is identically one: its router gradient is
            # mathematically zero. Top-2 exercises a meaningful router gradient.
            "block_top_k": 2 if block_experts else 0,
            "shared_multiplier": 1,
            "shared_gate": "sigmoid",
            "dispatcher": dispatcher,
        },
    }


def _build(dispatcher, *, device="cpu", dtype=torch.float32, block_experts=0):
    model = GenEngineModel.build_empty(
        config=_config(dispatcher, block_experts=block_experts), parameter_dtype=dtype
    )
    materialize_meta_module(model, torch.device(device))
    model.initialize_parameters(seed=42)
    return model


def _packed_reference_state(reference, candidate, *, gradients=False):
    """Stack the independent oracle's weights/gradients without swapping halves."""
    source = dict(reference.named_parameters()) if gradients else reference.state_dict()
    expected = {}
    for name, value in source.items():
        if ".experts." not in name:
            expected[name] = value.grad if gradients else value
    for path, module in candidate.named_modules():
        if not isinstance(module, Expert):
            continue
        bank = reference.get_submodule(path)
        assert isinstance(bank, ExpertBank)
        for parameter in ("gate_up", "down"):
            tensors = [getattr(expert, parameter).weight for expert in bank.experts]
            if gradients:
                tensors = [value.grad for value in tensors]
            assert all(value is not None for value in tensors)
            expected[f"{path}.{parameter}"] = torch.stack(tensors)
    return expected


@pytest.mark.parametrize(
    "name,total",
    [
        ("mmdit-dense-isoflop-4.1b", 4_217_594_752),
        ("mmdit-dense-leverage-10.4b", 10_514_060_800),
        ("mmdit-moe-32b-a4.1b", 32_188_759_936),
    ],
)
def test_full_catalog_meta_parameter_counts(name, total):
    config = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    for dispatcher in ("torch_reference", "vforge_expert"):
        if "moe" in config:
            config["moe"]["dispatcher"] = dispatcher
        model = GenEngineModel.build_empty(config=config, parameter_dtype=torch.float32)
        assert all(parameter.is_meta for parameter in model.parameters())
        assert sum(parameter.numel() for parameter in model.parameters()) == total
        banks = [module for module in model.modules() if isinstance(module, Expert)]
        if "moe" in config and dispatcher == "vforge_expert":
            assert len(banks) == 26
            assert all(bank.gate_up.shape == (160, 3072, 1536) for bank in banks)
        else:
            assert not banks


def test_packed_initialization_preserves_fan_in_and_stable_roles():
    model = _build("vforge_expert", block_experts=2)
    torch.testing.assert_close(model.transformer.rope.inv_freq, torch.ones(1))
    original = {name: value.clone() for name, value in model.state_dict().items()}
    banks = [module for module in model.modules() if isinstance(module, Expert)]
    assert len(banks) == 2
    for bank in banks:
        assert bank.projection_order == "up_gate"
        for value, fan_in in (
            (bank.gate_up, bank.hidden_size),
            (bank.down, bank.intermediate_size),
        ):
            assert value.abs().max() <= math.sqrt(3 / fan_in)
            assert value.std().item() == pytest.approx(1 / math.sqrt(fan_in), rel=0.05)
            assert not torch.equal(value[0], value[1])
    for block in model.blocks:
        assert torch.count_nonzero(block.adaln_modulation.linear.weight) == 0
    moe = model.blocks[1].visual_moe
    assert torch.count_nonzero(moe.router.main_routing_bias) == 0
    assert torch.count_nonzero(moe.shared.shared_router_gate.weight) == 0
    assert torch.count_nonzero(model.transformer.proj_out.weight) == 0
    model.initialize_parameters(seed=42)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_packed_geometry_is_validated_before_materialization():
    config = _config()
    config["moe"]["expert_intermediate_size"] = 12
    with pytest.raises(ValueError, match="divisible by 8"):
        GenEngineModel.build_empty(config=config)


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("empty", [False, True])
def test_mixing_matches_reference_output_dtype(output_dtype, empty):
    """Exercise real dispatch with CPU output stubs, not the CUDA GEMM kernel.

    FP32 parameter storage deliberately differs from the low-precision output,
    as can happen across a mixed-precision FSDP forward boundary.
    """

    class ReferenceOutput(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = torch.nn.Parameter(scale.clone())

        def forward(self, states):
            return torch.tanh(states * self.scale).to(output_dtype)

    class PackedOutput(Expert):
        def forward(self, states, ids):
            return torch.tanh(states * self.gate_up[ids, 0]).to(output_dtype)

    torch.manual_seed(42)
    bank = PackedOutput(2, 8, 8, dtype=torch.float32)
    reference = torch.nn.ModuleList([ReferenceOutput(value[0]) for value in bank.gate_up])
    inputs = torch.randn(0 if empty else 3, 8, requires_grad=True)
    weights = torch.randn(len(inputs), 2, requires_grad=True)
    ids = torch.tensor([[1, 0], [0, 1], [1, 0]])[: len(inputs)]
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weights = weights.detach().clone().requires_grad_()
    actual = _dispatch_experts(bank, inputs, ids, weights)
    expected = TorchReferenceDispatcher()(reference_inputs, reference, ids, reference_weights)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    if not empty:
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(weights.grad, reference_weights.grad, rtol=0, atol=0)
        assert weights.grad.dtype == torch.float32
        assert weights.grad.count_nonzero()
    expected_grad = torch.stack([expert.scale.grad for expert in reference])
    torch.testing.assert_close(bank.gate_up.grad[:, 0], expected_grad, rtol=1e-6, atol=1e-7)


def _inputs():
    return {
        "hidden_states": torch.randn(1, 5, 1, device="cuda", requires_grad=True),
        "audio_hidden_states": torch.randn(1, 2, 3, device="cuda", requires_grad=True),
        "encoder_hidden_states": torch.randn(1, 2, 4, device="cuda", requires_grad=True),
        "timestep": torch.tensor([0.25, 0.75], device="cuda"),
        "timestep_indices": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], device="cuda"),
        "token_tags": torch.tensor([1, 1, 0, 0, 0, 0, 0, 2, 2], device="cuda"),
        "position_ids": torch.randn(9, 3, device="cuda"),
        "text_indices": torch.tensor([0, 1], device="cuda"),
        "video_indices": torch.tensor([2, 3, 4, 5, 6], device="cuda"),
        "audio_indices": torch.tensor([7, 8], device="cuda"),
        "reference_modality_ids": torch.tensor([0, 0, 1, 2, 0, 0, 0, 0, 0], device="cuda"),
        "reference_index_ids": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0, 0], device="cuda"),
        "audio_plane_ids": torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2], device="cuda"),
    }


def _assert_bf16_error(actual, expected, name):
    """Bound whole-tensor error without a loose absolute floor on small grads.

    BF16 GEMM reductions differ at cancellation points; elementwise relative
    error there is unbounded. Use <=2% L2 error and <=10% of reference RMS for
    any element. A missing gradient or incorrect scaling fails these bounds.
    """
    assert actual is not None and expected is not None, name
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    if not expected.count_nonzero():
        assert not actual.count_nonzero(), name
        return
    difference = actual - expected
    l2 = (difference.norm() / expected.norm()).item()
    peak = (difference.abs().max() / expected.square().mean().sqrt()).item()
    print(f"BF16 error {name}: relative_l2={l2:.6g} max_over_rms={peak:.6g}")
    assert l2 <= 0.02 and peak <= 0.1, (name, l2, peak)


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("block_experts", [0, 2])
def test_visual_moe_bf16_expert_and_router_gradients(block_experts):
    """Probe MoE directly so zero-initialized trunk gates cannot mask errors."""
    torch.manual_seed(42)
    reference = (
        _build("torch_reference", device="cuda", dtype=torch.bfloat16, block_experts=block_experts)
        .blocks[1]
        .visual_moe
    )
    candidate = (
        _build("vforge_expert", device="cuda", dtype=torch.bfloat16, block_experts=block_experts)
        .blocks[1]
        .visual_moe
    )
    candidate.load_state_dict(_packed_reference_state(reference, candidate), strict=True)
    fp32 = copy.deepcopy(reference).float()
    inputs = torch.randn(1, 64, 32, device="cuda", requires_grad=True)
    candidate_inputs = inputs.detach().clone().requires_grad_()
    fp32_inputs = inputs.detach().clone().requires_grad_()
    categories = torch.arange(64, device="cuda") % 3
    expected, expected_statistics = reference(inputs, categories)
    actual, statistics = candidate(candidate_inputs, categories)
    full_precision, _ = fp32(fp32_inputs, categories)
    _assert_bf16_error(actual, expected, "packed/reference output")
    _assert_bf16_error(actual, full_precision, "packed/FP32 output")
    _assert_bf16_error(expected, full_precision, "reference/FP32 output")
    assert torch.equal(statistics.main.counts, expected_statistics.main.counts)
    assert statistics.main.counts.sum() == 128
    probe = torch.randn_like(expected) / 64
    expected.backward(probe)
    actual.backward(probe)
    full_precision.backward(probe)
    _assert_bf16_error(candidate_inputs.grad, inputs.grad, "packed/reference input grad")
    _assert_bf16_error(candidate_inputs.grad, fp32_inputs.grad, "packed/FP32 input grad")
    expected_gradients = _packed_reference_state(reference, candidate, gradients=True)
    fp32_gradients = _packed_reference_state(fp32, candidate, gradients=True)
    for name, parameter in candidate.named_parameters():
        _assert_bf16_error(parameter.grad, expected_gradients[name], f"{name}/reference grad")
        _assert_bf16_error(parameter.grad, fp32_gradients[name], f"{name}/FP32 grad")
    assert candidate.main_bank.gate_up.grad.count_nonzero()
    assert candidate.main_bank.down.grad.count_nonzero()
    assert candidate.router.main_router.weight.grad.count_nonzero()
    if block_experts:
        assert candidate.router.block_router.weight.grad.count_nonzero()


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("block_experts", [0, 2])
def test_packed_mmdit_bf16_multistep_matches_reference(block_experts):
    assert torch.__version__.split("+")[0].startswith("2.13."), torch.__version__
    torch.manual_seed(42)
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cuda"),
        parallel_state=ParallelState(),
    )
    try:
        reference = _build(
            "torch_reference", device="cuda", dtype=torch.bfloat16, block_experts=block_experts
        )
        candidate = _build(
            "vforge_expert", device="cuda", dtype=torch.bfloat16, block_experts=block_experts
        )
        candidate.load_state_dict(_packed_reference_state(reference, candidate), strict=True)
        initial = {name: value.clone() for name, value in candidate.named_parameters()}
        optimizers = [
            torch.optim.AdamW(
                model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0, foreach=False
            )
            for model in (reference, candidate)
        ]
        expert_gradient_seen = router_gradient_seen = False
        for step in range(5):
            inputs = _inputs()
            candidate_inputs = {
                name: value.detach().clone().requires_grad_(value.requires_grad)
                for name, value in inputs.items()
            }
            outputs = []
            for model, batch, optimizer in zip(
                (reference, candidate), (inputs, candidate_inputs), optimizers, strict=True
            ):
                optimizer.zero_grad(set_to_none=True)
                result = model(**batch)
                loss = (result.sample.float() - 0.25).square().mean()
                loss = loss + (result.audio_sample.float() + 0.5).square().mean()
                loss.backward()
                outputs.append(result)
            for field in ("sample", "audio_sample"):
                torch.testing.assert_close(
                    getattr(outputs[1], field), getattr(outputs[0], field), rtol=5e-2, atol=5e-3
                )
            for name, value in inputs.items():
                if value.requires_grad:
                    torch.testing.assert_close(
                        candidate_inputs[name].grad, value.grad, rtol=5e-2, atol=5e-3
                    )
            expected_grads = _packed_reference_state(reference, candidate, gradients=True)
            for name, value in candidate.named_parameters():
                expected = expected_grads[name]
                if expected is None:
                    assert value.grad is None, name
                    continue
                torch.testing.assert_close(value.grad, expected, rtol=5e-2, atol=5e-3, msg=name)
                if ".main_bank." in name:
                    expert_gradient_seen |= bool(value.grad.count_nonzero())
                if ".main_router." in name:
                    router_gradient_seen |= bool(value.grad.count_nonzero())
            for model, optimizer in zip((reference, candidate), optimizers, strict=True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.post_optimizer_step(completed_step=step + 1, total_steps=5)
            expected_state = _packed_reference_state(reference, candidate)
            for name, value in candidate.state_dict().items():
                torch.testing.assert_close(
                    value, expected_state[name], rtol=5e-2, atol=5e-3, msg=name
                )
        assert expert_gradient_seen and router_gradient_seen
        for path, module in candidate.named_modules():
            if isinstance(module, VisualMoE):
                assert module.router.main_router.weight.dtype == torch.float32
                for parameter in ("gate_up", "down"):
                    name = f"{path}.main_bank.{parameter}"
                    assert not torch.equal(dict(candidate.named_parameters())[name], initial[name])
        print(
            f"MMDiT packed parity: torch={torch.__version__} GPU={torch.cuda.get_device_name()} "
            f"EP=1 USP=1 seed=42 BF16 steps=5 block_experts={block_experts}"
        )
    finally:
        reset_ops()


def test_parallel_plan_shards_only_main_banks():
    model = GenEngineModel.build_empty(config=_config(block_experts=2))
    plan = model.get_parallel_plan()
    assert len(plan.banks) == 1
    bank = plan.banks[0]
    assert bank.module_path == "transformer.transformer_blocks.1.visual_moe.main_bank"
    assert bank.parameter_names == ("gate_up", "down")
    assert bank.fsdp_shard_dim == 1
    with pytest.raises(ValueError, match="moe.dispatcher=vforge_expert"):
        GenEngineModel.build_empty(config=_config("torch_reference")).get_parallel_plan()


def test_recompute_replays_all_collectives():
    model = GenEngineModel.build_empty(config=_config())
    model.gradient_checkpointing_enable()
    assert model._gradient_checkpointing_func.keywords["early_stop"] is False
    model.gradient_checkpointing_enable({"use_reentrant": False, "early_stop": True})
    assert model._gradient_checkpointing_func.keywords["early_stop"] is False
