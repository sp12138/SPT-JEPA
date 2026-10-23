"""Fixed-input accumulation oracle across native parallel topologies."""

import argparse
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from expert_parallel_models import MoE, ReferenceExperts, SequenceMoE
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.utils.checkpoint import checkpoint

from vforge.distributed.expert_parallel import Expert
from vforge.distributed.parallel_state import (
    ParallelConfig,
    destroy_parallel_state,
    init_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
    unwrap_replicated_data_parallel,
)
from vforge.distributed.tensor_parallel import build_linear
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig
from vforge.training.diagnostics import LogicalReduction, clip_grad_norm_logical_
from vforge.training.engine import EnginePolicy, EngineRunContext
from vforge.training.engine.fsdp import FsdpEngine
from vforge.training.recipe import StepContext, StepOutput


class Projection(torch.nn.Module):
    def __init__(self, tp=False):
        super().__init__()
        self.up = build_linear(8, 16, style=ColwiseParallel()) if tp else torch.nn.Linear(8, 16)
        self.down = build_linear(16, 8, style=RowwiseParallel()) if tp else torch.nn.Linear(16, 8)

    def forward(self, hidden):
        return self.down(self.up(hidden).tanh())


class AlternatingModel(torch.nn.Module):
    """Earlier micro-batches use a tower absent from the final backward."""

    _no_split_modules = ["Projection"]

    def __init__(self, tp=False):
        super().__init__()
        self.shared = Projection(tp)
        self.towers = torch.nn.ModuleList([Projection(tp), Projection(tp)])
        self.unused = torch.nn.Parameter(torch.ones(8))
        self.checkpoint_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpoint_kwargs = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}

    def forward(self, hidden, tower):
        if self.checkpoint_kwargs is not None:
            return checkpoint(self._forward, hidden, tower, **self.checkpoint_kwargs)
        return self._forward(hidden, tower)

    def _forward(self, hidden, tower):
        return self.towers[tower](self.shared(hidden))


def _full(value):
    return value.full_tensor() if hasattr(value, "full_tensor") else value


def run_case(device, mode, *, count, amp=False, recompute=False):
    destroy_parallel_state()
    reset_ops()
    rank, world = dist.get_rank(), dist.get_world_size()
    usp = 2 if "usp" in mode else 1
    tp = 2 if "tp" in mode else 1
    ep = 2 if "ep" in mode else 1
    dp = world // (usp * tp)
    config = ParallelizeConfig(
        parallel=ParallelConfig(
            dp_size=dp,
            usp_size=usp,
            tp_size=tp,
            dp_mode="ddp" if mode == "ddp" else "fsdp2",
            dp_replicate_size=2 if "hsdp" in mode else 1,
            dp_shard_size=dp // 2 if "hsdp" in mode else dp,
            extra_parallel_sizes=(ep,),
            device_type=device.type,
        ),
        fsdp2=FSDP2Config(
            enable=mode != "ddp",
            mixed_precision=MixedPrecisionConfig(enable=amp),
        ),
        gradient_checkpointing=recompute,
    )
    torch.manual_seed(42)
    is_moe = usp > 1 or ep > 1
    if is_moe:
        model_class = SequenceMoE if usp > 1 else MoE
        reference = model_class(intermediate=16, experts_cls=ReferenceExperts).to(device)
        if usp > 1:
            reference.sequence_parallel = False
        candidate = model_class(intermediate=16, experts_cls=Expert if amp else ReferenceExperts)
    else:
        reference = AlternatingModel().to(device)
        candidate = AlternatingModel(tp=tp > 1)
    candidate = candidate.to(device)
    candidate.load_state_dict(reference.state_dict())
    state = init_parallel_state(config.parallel)
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False, deterministic=True),
        device=device,
        parallel_state=state,
    )
    engine = FsdpEngine(
        EnginePolicy(
            parallelize=config,
            gradient_accumulation_steps=count,
            gradient_clip_val=0.1,
        )
    )
    model = engine.prepare_data_parallel(engine.prepare_model(candidate))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
    oracle_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=0.001, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    oracle_scheduler = torch.optim.lr_scheduler.StepLR(oracle_optimizer, 1, gamma=0.8)
    precision = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if amp else nullcontext
    rtol, atol = (5e-2, 5e-3) if amp else (1e-4, 1e-5)
    updates = []
    optimizer.register_step_post_hook(lambda *_: updates.append(True))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device=device).manual_seed(42)
    context = StepContext(device, state)
    for step in range(2):
        shape = (dp, count, 1, 4 * usp, 8) if usp > 1 else (dp, count, 3, 8)
        full_inputs = torch.randn(*shape, device=device, generator=generator)
        routes = ["normal" if i % 2 == 0 else "zero_assignments" for i in range(count)]
        routes = routes if is_moe else [i % 2 for i in range(count)]
        oracle_optimizer.zero_grad(set_to_none=True)
        for index in range(count):
            for source in range(dp):
                with precision():
                    expected = reference(full_inputs[source, index], routes[index])
                    expected_loss = expected.float().square().sum() / (count * dp)
                expected_loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in reference.named_parameters()
        }
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1).item()

        def check_and_clip(target, optimizer, max_norm, reduction, expected_gradients=gradients):
            logical = unwrap_replicated_data_parallel(target)
            for name, parameter in logical.named_parameters():
                expected = expected_gradients[name]
                if expected is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(
                        _full(parameter.grad), expected, rtol=rtol, atol=atol, msg=name
                    )
            return clip_grad_norm_logical_(target, max_norm, reduction)

        def microbatches(inputs=full_inputs, routing=routes):
            for index in range(count):
                yield inputs[state.dp_rank, index], routing[index]

        result = engine.run_update(
            model=model,
            microbatches=microbatches(),
            context=context,
            step=step,
            compute_step=lambda root, batch, *_: StepOutput(root(*batch).float().square().sum()),
            optimizer=optimizer,
            scheduler=scheduler,
            run=EngineRunContext(
                precision_context=precision,
                gradient_clipper=check_and_clip,
                logical_reduction=LogicalReduction.for_model(model, state),
            ),
        )
        assert abs(result.diagnostics["diag/grad_norm_post_reduce"] - expected_norm) <= (
            atol + rtol * abs(expected_norm)
        )
        oracle_optimizer.step()
        oracle_scheduler.step()
        assert len(updates) == step + 1
        assert scheduler.state_dict() == oracle_scheduler.state_dict()
        actual = unwrap_replicated_data_parallel(model)
        for name, parameter in actual.named_parameters():
            torch.testing.assert_close(
                _full(parameter),
                dict(reference.named_parameters())[name],
                rtol=rtol,
                atol=atol,
                msg=name,
            )
    if rank == 0:
        print(
            "VFORGE_ACCUMULATION "
            + json.dumps(
                {
                    "mode": mode,
                    "world": world,
                    "dp": dp,
                    "tp": tp,
                    "usp": usp,
                    "ep": ep,
                    "dp_replicate": config.parallel.dp_replicate_size,
                    "dp_shard": dp // config.parallel.dp_replicate_size,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "seed": 42,
                    "dtype": "bf16" if amp else "fp32",
                    "count": count,
                    "recompute": recompute,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                    ),
                }
            ),
            flush=True,
        )
    reset_ops()
    destroy_parallel_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "ddp",
            "fsdp",
            "hsdp",
            "tp",
            "usp",
            "ep",
            "ep_usp",
            "hsdp_tp",
            "hsdp_usp",
            "ep_hsdp",
            "ep_hsdp_usp",
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    try:
        for amp in (False, True):
            for count in (1, 2, 4):
                for recompute in (False, True):
                    run_case(device, args.mode, count=count, amp=amp, recompute=recompute)
            if args.mode in {"ep", "ep_usp"}:
                from test_expert_checkpoint import run_checkpoint_case

                directory = args.checkpoint_dir / ("bf16" if amp else "fp32")
                directory.mkdir(parents=True, exist_ok=True)
                run_checkpoint_case(
                    dist.get_rank(),
                    dist.get_world_size(),
                    str(directory),
                    device=device,
                    ep=2,
                    usp=2 if args.mode == "ep_usp" else 1,
                    dtype=torch.float32,
                    amp=amp,
                    recompute=True,
                    accumulation=4,
                )
                reset_ops()
                destroy_parallel_state()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
