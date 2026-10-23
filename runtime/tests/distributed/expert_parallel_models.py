"""Small single- and dual-bank models for the generic EP contract gates."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from vforge.distributed.expert_parallel import (
    Expert,
    ExpertParallelPlan,
)
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.ulysses import gather_sequence, slice_sequence
from vforge.ops import attention


class ReferenceExperts(Expert):
    """Dense all-expert oracle independent of the runtime's token grouping."""

    def forward(self, states, expert_ids):
        gate_up = self.gate_up.to_local() if isinstance(self.gate_up, DTensor) else self.gate_up
        down = self.down.to_local() if isinstance(self.down, DTensor) else self.down
        outputs = []
        for expert in range(gate_up.shape[0]):
            gate, up = torch.nn.functional.linear(states, gate_up[expert]).chunk(2, -1)
            if self.projection_order == "up_gate":
                up, gate = gate, up
            outputs.append(torch.nn.functional.linear(self.activation_fn(gate) * up, down[expert]))
        all_experts = torch.stack(outputs, dim=1)
        return all_experts[torch.arange(states.shape[0], device=states.device), expert_ids]


class SharedBranch(nn.Module):
    """Two expert-equivalents combined into one gated shared MLP."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_up = nn.Linear(hidden, intermediate * 4)
        self.down = nn.Linear(intermediate * 2, hidden)

    def forward(self, hidden):
        gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, main=8, block=2, hidden=8, intermediate=12, top_k=2, *, experts_cls=Expert):
        super().__init__()
        self.main = experts_cls(main, hidden, intermediate, init_std=0.1)
        self.main_router = nn.Linear(hidden, main, bias=False)
        self.block = experts_cls(block, hidden, intermediate, init_std=0.1) if block else None
        self.block_router = nn.Linear(hidden, block, bias=False) if block else None
        self.shared = SharedBranch(hidden, intermediate)
        self.shared_gate = nn.Linear(hidden, 1)
        self.main_count = main
        self.block_count = block
        self.top_k = top_k
        self.recompute = False
        self.register_buffer("runtime_scale", torch.tensor(0.5), persistent=False)

    def reset_non_persistent_buffers(self):
        self.runtime_scale.fill_(0.5)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.recompute = True

    def get_parallel_plan(self):
        return ExpertParallelPlan((self.main.parallel_spec("main"),))

    def forward(self, hidden, route="normal"):
        if self.recompute:
            # Recompute every collective, even if this rank needs fewer tensors.
            return checkpoint(self._forward, hidden, route, use_reentrant=False, early_stop=False)
        return self._forward(hidden, route)

    def _forward(self, hidden, route):
        scores = self.main_router(hidden).softmax(-1)
        weights, ids = scores.topk(min(self.top_k, self.main_count), -1)
        if route == "zero_assignments":
            weights, ids = weights[:, :0], ids[:, :0]
        if route == "skew":
            ids = torch.zeros_like(ids[:, :1])
            weights = scores[:, :1]
        weights = weights / weights.sum(-1, keepdim=True)
        output = self.main.dispatch(hidden, ids, weights)
        if self.block is not None:
            block_weights, block_ids = (
                self.block_router(hidden).softmax(-1).topk(min(2, self.block_count), -1)
            )
            if route == "zero_assignments":
                block_weights, block_ids = block_weights[:, :0], block_ids[:, :0]
            block_weights = block_weights / block_weights.sum(-1, keepdim=True)
            # No EP declaration: the same public module dispatches locally.
            output = output + self.block.dispatch(hidden, block_ids, block_weights)
        return (
            output + self.shared(hidden) * self.shared_gate(hidden).sigmoid() * self.runtime_scale
        )


class SequenceMoE(MoE):
    """Full-sequence boundaries around Ulysses attention and token-local MoE."""

    def __init__(self, *args, sequence_parallel=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_parallel = sequence_parallel
        hidden = self.main.hidden_size
        self.embed = nn.Linear(hidden, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def _forward(self, hidden, route):
        group = get_parallel_state().usp_group if self.sequence_parallel else None
        hidden = self.embed(hidden)
        if group is not None:
            # Cancel the gather's duplicate-loss factor before the replicated
            # embedding boundary, as in the Native model USP contract.
            hidden = slice_sequence(hidden, dim=1, group=group, scale_grad=True)
        q, k, v = self.qkv(hidden).unflatten(-1, (3, 4, hidden.shape[-1] // 4)).unbind(-3)
        attended = attention(q, k, v, distributed=self.sequence_parallel)
        hidden = hidden + self.proj(attended.flatten(-2))
        output = super()._forward(hidden.flatten(0, 1), route).reshape(hidden.shape)
        if group is not None:
            output = gather_sequence(output, dim=1, group=group)
        return self.output(output)
