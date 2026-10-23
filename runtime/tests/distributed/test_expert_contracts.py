"""CPU-only declarations, meta initialization, and configuration checks."""

from types import SimpleNamespace

import pytest
import torch
from expert_parallel_models import MoE
from hydra import compose, initialize_config_module

from vforge.distributed.expert_parallel import (
    Expert,
    ExpertBankSpec,
    ExpertLayout,
    ExpertParallelPlan,
    dispatch_to_local_experts,
)
from vforge.distributed.parallel_state import ParallelConfig
from vforge.hydra_config import LimitationsConfig, ParallelHydraConfig, config_from_dict
from vforge.tasks.translate import build_parallel_config


def test_meta_experts_and_explicit_bank_declaration():
    experts = Expert(400, 16, 1152, device="meta")
    assert experts.gate_up.is_meta and experts.down.is_meta
    assert experts.gate_up.shape == (400, 2304, 16)
    assert experts.down.shape == (400, 16, 1152)
    spec = experts.parallel_spec("layers.*.main")
    assert spec.module_path == "layers.*.main"
    assert spec.parameter_names == ("gate_up", "down")
    assert not hasattr(experts, "_vforge_expert_layout")
    assert experts.projection_order == "gate_up"


def test_projection_order_validation_and_compatible_parameter_layout():
    original = Expert(3, 8, 16, device="meta")
    reversed_order = Expert(3, 8, 16, device="meta", projection_order="up_gate")
    assert {name: value.shape for name, value in original.state_dict().items()} == {
        name: value.shape for name, value in reversed_order.state_dict().items()
    }
    with pytest.raises(ValueError, match="projection_order"):
        Expert(3, 8, 16, device="meta", projection_order="invalid")


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cast_weights", [False, True])
def test_dispatch_weight_precision_uses_actual_output_dtype(output_dtype, cast_weights):
    class OutputBank(torch.nn.Module):
        def forward(self, states, ids):
            return states.to(output_dtype)

    inputs = torch.tensor([[1.234567, -0.765432]], requires_grad=True)
    weights = torch.tensor([[0.234567]], requires_grad=True)
    output = dispatch_to_local_experts(
        inputs,
        torch.zeros(1, 1, dtype=torch.long),
        weights,
        experts=OutputBank(),
        cast_routing_weights_to_output=cast_weights,
    )
    expected_inputs = inputs.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()
    # The default promotes with FP32 weights; the opt-in rounds both the
    # coefficient and product to the output dtype, then accumulates in FP32.
    product_dtype = output_dtype if cast_weights else torch.float32
    expected = (
        expected_inputs.to(output_dtype).to(product_dtype) * expected_weights.to(product_dtype)
    ).float()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.dtype == torch.float32
    output.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, expected_weights.grad, rtol=0, atol=0)


@pytest.mark.parametrize("verb", ["train", "eval", "prepare", "convert_checkpoint"])
@pytest.mark.parametrize("ep,world", [(2, 4), (20, 40), (40, 40)])
def test_ep_override_is_declared_in_entry_yaml(verb, ep, world):
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(config_name=verb, overrides=[f"parallel.ep_size={ep}"])
    assert cfg.parallel.ep_size == ep
    runtime = build_parallel_config(ParallelHydraConfig(ep_size=ep), world_size=world)
    assert runtime.ep_size == ep
    assert runtime.dp_size == world


@pytest.mark.parametrize(
    "values",
    [
        {"tp_size": 2},
        {"cp_size": 2},
        {"pp_size": 2},
        {"dp_mode": "ddp"},
    ],
)
def test_ep_rejects_unsupported_combinations(values):
    with pytest.raises(NotImplementedError, match="EP requires FSDP2"):
        ParallelConfig(dp_size=4, extra_parallel_sizes=(2,), **values).normalized(4)


def test_ep_size_validation():
    with pytest.raises(ValueError, match="ep_size must divide"):
        ParallelConfig(dp_size=4, extra_parallel_sizes=(3,)).normalized(4)
    with pytest.raises(ValueError, match="fsdp2"):
        ParallelHydraConfig(ep_size=2, engine="megatron")


@pytest.mark.parametrize("verb", ["train", "eval", "prepare", "convert_checkpoint"])
@pytest.mark.parametrize("world,ep,usp", [(4, 2, 2), (4, 4, 2), (4, 2, 4), (8, 4, 2), (12, 4, 3)])
def test_ep_usp_compose_and_translation(verb, world, ep, usp):
    with initialize_config_module(config_module="vforge.conf", version_base=None):
        cfg = compose(
            config_name=verb,
            overrides=[f"parallel.ep_size={ep}", f"parallel.usp_size={usp}"],
        )
    runtime = build_parallel_config(config_from_dict(cfg).parallel, world_size=world)
    assert runtime.ep_size == ep
    assert runtime.usp_size == usp
    assert runtime.dp_size == world // usp
    runtime.normalized(world)


def test_ep_usp_rejects_invalid_domain_and_decoupled_fsdp():
    with pytest.raises(ValueError, match="ep_size must divide dp_shard_size \\* usp_size"):
        ParallelConfig(dp_size=2, usp_size=2, extra_parallel_sizes=(3,)).normalized(4)
    with pytest.raises(NotImplementedError, match="Decoupled Ulysses/FSDP"):
        ParallelConfig(
            dp_size=2, usp_size=2, extra_parallel_sizes=(2,), include_usp_in_fsdp=False
        ).normalized(4)


def test_legacy_ep_limitation_does_not_block_current_capability_checks():
    LimitationsConfig()
    LimitationsConfig(expert_parallel=True)
    with pytest.raises(ValueError, match="immutable"):
        LimitationsConfig(ring_attention=True)


def _state(rank=0, ep=2):
    return SimpleNamespace(
        ep_size=ep,
        ep_rank=rank,
        world_size=ep,
        ep_fsdp_mesh=SimpleNamespace(size=lambda dim=None: 1),
    )


@pytest.mark.parametrize("ep", [20, 40])
def test_visual_moe_39_layer_meta_layout(ep):
    with torch.device("meta"):
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList(
            [MoE(main=400, block=16, intermediate=1152, top_k=20) for _ in range(39)]
        )
    plan = ExpertParallelPlan((ExpertBankSpec("layers.*.main", ("gate_up", "down")),))
    banks = plan._resolve_banks(model, _state(rank=ep // 2, ep=ep))
    assert len(banks) == 39
    for layer in model.layers:
        assert layer.main.gate_up.shape == (400, 2304, 8)
        assert layer.block.gate_up.shape == (16, 2304, 8)
        assert not hasattr(layer.block, "_vforge_expert_layout")
        assert layer.shared.down.in_features == 2304
        assert layer.shared.gate_up.out_features == 4608
        assert ExpertLayout(400, ep, ep // 2).start == 200
        assert layer.top_k == 20
    assert all(parameter.is_meta for parameter in model.parameters())


@pytest.mark.parametrize("ep", [1, 2, 4, 20, 40])
def test_undeclared_modules_keep_their_model_structure(ep):
    model = MoE(block=16)
    # An ordinary Block can use nested modules and arbitrary parameter shapes.
    model.block.extra = torch.nn.Linear(3, 7)
    parameters = dict(model.block.named_parameters())
    for rank in range(ep):
        current = MoE(main=ep * 2, block=16)
        current.block = model.block
        banks = current.get_parallel_plan()._resolve_banks(current, _state(rank=rank, ep=ep))
        assert set(banks) == {"main"}
        assert not hasattr(current.block, "_vforge_expert_layout")
        assert not current.block._forward_hooks
        for name, parameter in current.block.named_parameters():
            assert parameter is parameters[name]
    assert model.block.gate_up.shape[0] == 16


def test_invalid_bank_plan_does_not_mutate_model():
    model = MoE()
    original = model.main.gate_up
    plan = ExpertParallelPlan(
        (ExpertBankSpec("main", ("gate_up", "down")), ExpertBankSpec("missing", ("weight",)))
    )
    with pytest.raises(ValueError, match="matched no modules"):
        plan.apply(model, _state())
    assert model.main.gate_up is original


def test_non_divisible_bank_is_rejected_before_mutation():
    model = MoE()
    model.extra = Expert(5, 8, 12)
    original = model.main.gate_up
    plan = ExpertParallelPlan(
        (ExpertBankSpec("main", ("gate_up", "down")), ExpertBankSpec("extra", ("gate_up", "down")))
    )
    with pytest.raises(ValueError, match="expert count must be divisible"):
        plan.apply(model, _state())
    assert model.main.gate_up is original
    assert not hasattr(model.main, "_vforge_expert_layout")


def test_parameter_alias_is_rejected():
    aliased = MoE()
    aliased.alias = aliased.main.gate_up
    with pytest.raises(ValueError, match="cannot have aliases"):
        aliased.get_parallel_plan().apply(aliased, _state())


def test_model_capability_checked_before_fsdp_wrapping(monkeypatch):
    import vforge.distributed.parallelize as implementation

    monkeypatch.setattr(
        implementation,
        "init_parallel_state",
        lambda config: SimpleNamespace(ep_enabled=True, fsdp_enabled=True),
    )
    with pytest.raises(ValueError, match="get_parallel_plan"):
        implementation.parallelize_model(torch.nn.Linear(2, 2))


def test_ep_engine_accepts_gradient_accumulation():
    from vforge.distributed.parallelize import ParallelizeConfig
    from vforge.training.engine.base import EnginePolicy
    from vforge.training.engine.fsdp import FsdpEngine

    engine = FsdpEngine(
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(dp_size=4, extra_parallel_sizes=(2,))
            ),
            gradient_accumulation_steps=2,
        )
    )
    assert engine.policy.gradient_accumulation_steps == 2


def test_expert_dimension_divisibility_ignores_hsdp_replication():
    model = MoE(main=8, intermediate=12)
    state = _state()
    state.ep_fsdp_mesh = SimpleNamespace(size=lambda dim=None: 2 if dim == -1 else 6)
    assert model.get_parallel_plan()._resolve_banks(model, state)
