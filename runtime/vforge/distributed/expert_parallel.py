"""Packed GLU experts, expert-parallel layouts, and differentiable dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from vforge.distributed.parallel_state import ParallelState, get_parallel_state

if TYPE_CHECKING:
    from vforge.distributed.parallelize import FSDPModulePlan


@dataclass(frozen=True)
class ExpertBankSpec:
    """Packed parameters to shard over EP, with experts on dimension zero.

    The bank forward accepts assignment states and local expert IDs and returns
    one output per assignment. All direct parameters must be declared. A bank
    cannot contain child parameters or overlap another bank. Undeclared modules
    remain ordinary FSDP modules; local expert pools need no EP declaration.

    EP parameters are DTensors: local compute uses their differentiable
    to_local() views. The bank owns zero gradients and input/parameter autograd
    connections even when it receives no assignments.
    """

    module_path: str
    parameter_names: tuple[str, ...]
    fsdp_shard_dim: int = 1


class Expert(nn.Module):
    """Bias-free packed GLU experts with a model-selected gate activation.

    Compute is ``down(activation_fn(gate) * up)``. The default SiLU gives
    SwiGLU; GELU gives GEGLU and sigmoid gives GLU. ``activation_fn`` must be
    stateless and elementwise, preserving the input shape, dtype and device.
    ``projection_order`` describes the two halves of ``gate_up``: the default
    ``"gate_up"`` stores [gate, up], while ``"up_gate"`` stores [up, gate].
    Parameter names and shapes are identical; callers must retain this setting
    in their model configuration when loading weights.

    ``dispatch(hidden, ids, weights)`` accepts token states ``[T, H]`` and
    model-selected expert IDs and weights ``[T, K]``. It returns ``[T, H]`` in
    source-token order. Routing weights retain their dtype by default; set
    ``cast_routing_weights_to_output=True`` to multiply in the expert output
    dtype before accumulating in the source input dtype. Declare
    ``parallel_spec(module_path)`` in the model's ``ExpertParallelPlan`` to
    shard this bank; an undeclared bank executes every assignment locally.

    ``forward(states, local_ids)`` is the local compute boundary called by the
    dispatcher through FSDP. CUDA FP32 residual inputs are cast to BF16 here;
    dispatch combines outputs in the source input dtype. Compute requires
    CUDA BF16 weights and hidden/intermediate
    sizes divisible by eight, using PyTorch 2.13 grouped MM. Parameters have logical shapes
    ``gate_up[E, 2*I, H]`` and ``down[E, H, I]`` before parallelization.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.silu,
        projection_order: str = "gate_up",
        init_std: float = 0.02,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if any(
            type(size) is not int or size < 1
            for size in (num_experts, hidden_size, intermediate_size)
        ):
            raise ValueError(
                "expert count, hidden size and intermediate size must be positive integers"
            )
        if not 0 < init_std < float("inf"):
            raise ValueError("init_std must be finite and positive")
        if projection_order not in ("gate_up", "up_gate"):
            raise ValueError("projection_order must be 'gate_up' or 'up_gate'")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_fn = activation_fn
        self.projection_order = projection_order
        self.init_std = init_std
        self.gate_up = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        self.down = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize complete logical weights before EP/FSDP is applied."""
        nn.init.normal_(self.gate_up, std=self.init_std)
        nn.init.normal_(self.down, std=self.init_std)

    def parallel_spec(self, module_path: str, *, fsdp_shard_dim: int = 1) -> ExpertBankSpec:
        """Explicitly opt this module path into the model's EP plan."""
        return ExpertBankSpec(module_path, ("gate_up", "down"), fsdp_shard_dim)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        *,
        cast_routing_weights_to_output: bool = False,
    ) -> torch.Tensor:
        # Keep dispatch outside this bank's FSDP forward boundary; the helper
        # calls self(incoming, local_ids) exactly once for local expert compute.
        return dispatch_to_local_experts(
            hidden_states,
            selected_experts,
            routing_weights,
            experts=self,
            cast_routing_weights_to_output=cast_routing_weights_to_output,
        )

    def forward(self, states: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        # FSDP all-gathers its axis before forward; the weights remain sharded
        # over EP. to_local() preserves autograd back to those DTensor weights.
        gate_up_weight = (
            self.gate_up.to_local() if isinstance(self.gate_up, DTensor) else self.gate_up
        )
        down_weight = self.down.to_local() if isinstance(self.down, DTensor) else self.down
        if states.ndim != 2 or states.shape[1] != self.hidden_size:
            raise ValueError("expert states must have shape [assignments, hidden_size]")
        if expert_ids.shape != (states.shape[0],) or expert_ids.dtype != torch.long:
            raise ValueError("local expert IDs must be int64 with shape [assignments]")
        if expert_ids.device != states.device:
            raise ValueError("expert states and IDs must be on the same device")
        if (
            not states.is_cuda
            or states.dtype not in (torch.float32, torch.bfloat16)
            or any(
                value.device != states.device or value.dtype != torch.bfloat16
                for value in (gate_up_weight, down_weight)
            )
        ):
            raise ValueError("Expert requires CUDA BF16 weights and BF16/FP32 inputs")
        if self.hidden_size % 8 or self.intermediate_size % 8:
            raise ValueError("Expert requires hidden/intermediate sizes divisible by 8")
        states = states.to(torch.bfloat16)
        if not states.shape[0]:
            # Keep both input and packed-parameter gradients when a whole bank
            # receives no assignments, including ordinary local Block banks.
            zero = (gate_up_weight.reshape(-1)[:1].sum() + down_weight.reshape(-1)[:1].sum()) * 0
            return states + zero.to(states.dtype)
        if ((expert_ids < 0) | (expert_ids >= gate_up_weight.shape[0])).any():
            raise ValueError("local expert ID is outside this bank")
        order = expert_ids.argsort(stable=True)
        counts = torch.bincount(expert_ids, minlength=gate_up_weight.shape[0])
        active = counts.nonzero().flatten()
        offsets = counts[active].cumsum(0, dtype=torch.int32)
        # Indexing only active experts avoids zero-K weight-gradient GEMMs;
        # indexing backward writes zeros into every unselected packed expert.
        gate_up = torch.nn.functional.grouped_mm(
            states[order], gate_up_weight[active].transpose(-2, -1), offs=offsets
        )
        gate, up = gate_up.chunk(2, dim=-1)
        if self.projection_order == "up_gate":
            up, gate = gate, up
        outputs = torch.nn.functional.grouped_mm(
            self.activation_fn(gate) * up,
            down_weight[active].transpose(-2, -1),
            offs=offsets,
        )
        return outputs[order.argsort()]


@dataclass(frozen=True)
class ExpertLayout:
    num_experts: int
    ep_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        if self.num_experts < 1 or self.ep_size < 1 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError("invalid expert count or EP coordinate")
        if self.num_experts % self.ep_size:
            raise ValueError("expert count must be divisible by ep_size")

    @property
    def start(self) -> int:
        return self.ep_rank * self.count

    @property
    def count(self) -> int:
        return self.num_experts // self.ep_size


@dataclass(frozen=True)
class ExpertParallelPlan:
    """Explicitly selected banks to shard when expert parallelism is enabled.

    Returned by ``model.get_parallel_plan()`` and applied before FSDP wrapping
    and materialization. Banks outside this declaration remain ordinary model
    computation; declaring a bank does not enable EP by itself.
    """

    banks: tuple[ExpertBankSpec, ...]

    def apply(self, model: nn.Module, state: ParallelState) -> dict[nn.Module, FSDPModulePlan]:
        """Shard declared banks on EP and return their resolved FSDP settings."""
        from vforge.distributed.parallelize import FSDPModulePlan

        banks = self._resolve_banks(model, state)
        ep_mesh = state.extra_parallel_mesh("ep")["ep"]
        resolved = {}
        for path, bank in banks.items():
            module = model.get_submodule(path)
            count = module.get_parameter(bank.parameter_names[0]).shape[0]
            for name, parameter in module.named_parameters(recurse=False):
                # Each rank supplies the same logical weights (or meta tensors).
                # DTensor owns both the global shape and the local expert slice.
                sharded = distribute_tensor(parameter, ep_mesh, [Shard(0)], src_data_rank=None)
                module.register_parameter(name, nn.Parameter(sharded, parameter.requires_grad))
            module._vforge_expert_layout = ExpertLayout(count, state.ep_size, state.ep_rank)
            # Dispatch sums contributions from P source ranks. Expert FSDP
            # reduces the shard and HSDP replica dimensions, but divides by W
            # to match ordinary FSDP, including the USP loss scaling.
            resolved[module] = FSDPModulePlan(
                mesh=state.ep_fsdp_mesh,
                shard_placement=Shard(bank.fsdp_shard_dim),
                gradient_divide_factor=state.world_size,
            )
        return resolved

    def _resolve_banks(self, model: nn.Module, state: ParallelState) -> dict[str, ExpertBankSpec]:
        """Validate all declarations before mutating any parameters."""
        resolved = {}
        modules = dict(model.named_modules())
        aliases: dict[int, list[str]] = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for bank in self.banks:
            if bank.fsdp_shard_dim < 1:
                raise ValueError("EP banks require fsdp_shard_dim>=1")
            matches = [name for name in modules if name and fnmatchcase(name, bank.module_path)]
            if not matches:
                raise ValueError(f"expert bank pattern {bank.module_path!r} matched no modules")
            for path in matches:
                if path in resolved:
                    raise ValueError(f"duplicate expert bank {path!r}")
                parameters = dict(modules[path].named_parameters())
                if (
                    not bank.parameter_names
                    or set(parameters) != set(bank.parameter_names)
                    or any("." in key for key in parameters)
                ):
                    raise ValueError(
                        f"expert bank {path!r} must declare all direct parameters "
                        "and no child parameters"
                    )
                first = next(iter(parameters.values()))
                if first.ndim < 2:
                    raise ValueError(f"invalid expert parameter shape for {path}")
                count = first.shape[0]
                ExpertLayout(count, state.ep_size, state.ep_rank)
                for name, parameter in parameters.items():
                    if len(aliases[id(parameter)]) != 1:
                        raise ValueError(f"expert parameter {path}.{name} cannot have aliases")
                    if isinstance(parameter, DTensor):
                        raise ValueError(f"expert parameter {path}.{name} is already distributed")
                    if parameter.ndim <= bank.fsdp_shard_dim or parameter.shape[0] != count:
                        raise ValueError(f"invalid expert parameter shape for {path}.{name}")
                    if not parameter.is_contiguous():
                        raise ValueError(f"expert parameter {path}.{name} must be contiguous")
                    if parameter.shape[bank.fsdp_shard_dim] % state.ep_fsdp_mesh.size(-1):
                        raise ValueError(
                            "expert FSDP parameter dimension must divide evenly across its mesh"
                        )
                resolved[path] = bank
        if not resolved:
            raise ValueError("EP requires at least one expert bank")
        return resolved


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, receive_counts, send_counts, group):
        ctx.receive_counts, ctx.send_counts, ctx.group = receive_counts, send_counts, group
        outputs = inputs.new_empty((sum(receive_counts), *inputs.shape[1:]))
        dist.all_to_all_single(
            outputs, inputs.contiguous(), list(receive_counts), list(send_counts), group=group
        )
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = grad_outputs.new_empty((sum(ctx.send_counts), *grad_outputs.shape[1:]))
        dist.all_to_all_single(
            grad_inputs,
            grad_outputs.contiguous(),
            list(ctx.send_counts),
            list(ctx.receive_counts),
            group=ctx.group,
        )
        return grad_inputs, None, None, None


def dispatch_to_local_experts(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    experts: nn.Module,
    cast_routing_weights_to_output: bool = False,
) -> torch.Tensor:
    """Execute each assignment once, then combine on its source rank.

    ``experts(states, local_ids)`` returns one state per assignment, including
    an empty tensor for empty input. IDs are zero-based within the packed bank.
    Routers retain ownership of Top-K, normalization, and gate precision.
    By default, multiplication uses PyTorch dtype promotion. Models can opt
    into casting routing weights to the actual expert output dtype, after
    local compute and any return communication, without inspecting parameters
    that FSDP may restore to another dtype outside forward.
    Without an EP layout (including EP=1), execution stays local.
    """
    if (
        hidden_states.ndim != 2
        or selected_experts.ndim != 2
        or selected_experts.shape != routing_weights.shape
        or selected_experts.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("expected hidden [tokens, hidden] and IDs/weights [tokens, top_k]")
    if selected_experts.dtype != torch.long:
        raise ValueError("expert IDs must be torch.long")
    layout = getattr(experts, "_vforge_expert_layout", None)
    tokens = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(
        selected_experts.shape[1]
    )
    ids = selected_experts.reshape(-1)
    states = hidden_states[tokens]
    distributed = layout is not None and layout.ep_size > 1
    if distributed:
        group = get_parallel_state().ep_group
        invalid = ((ids < 0) | (ids >= layout.num_experts)).any().to(torch.int32)
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=group)
        if invalid.item():
            raise ValueError("expert ID is outside the logical bank")
        owners = ids // layout.count
        order = owners.argsort(stable=True)
        local_ids = ids[order] % layout.count
        states = states[order]
        sends = torch.bincount(owners, minlength=layout.ep_size)
        receives = torch.empty_like(sends)
        dist.all_to_all_single(receives, sends, group=group)
        send_counts, receive_counts = tuple(sends.tolist()), tuple(receives.tolist())
        # Every rank needs an autograd edge even when the caller froze the input
        # path; otherwise only some ranks would run the backward collectives.
        if torch.is_grad_enabled() and not states.requires_grad:
            states = states.detach().requires_grad_()
        incoming = _AllToAll.apply(states, receive_counts, send_counts, group)
        incoming_ids = local_ids.new_empty(sum(receive_counts))
        dist.all_to_all_single(
            incoming_ids,
            local_ids.contiguous(),
            list(receive_counts),
            list(send_counts),
            group=group,
        )
        outputs = experts(incoming, incoming_ids)
        returned = _AllToAll.apply(outputs, send_counts, receive_counts, group)
        tokens = tokens[order]
        weights = routing_weights.reshape(-1)[order]
    else:
        returned = experts(states, ids)
        weights = routing_weights.reshape(-1)
    if cast_routing_weights_to_output:
        weights = weights.to(returned.dtype)
    result = hidden_states.new_zeros(hidden_states.shape)
    result.index_add_(0, tokens, (returned * weights.unsqueeze(-1)).to(result.dtype))
    return result + returned.sum().to(result.dtype) * 0
