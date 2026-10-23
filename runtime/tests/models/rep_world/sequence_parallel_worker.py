"""Distributed RepWorld oracle: full-sequence loss, every gradient and Muon update."""

from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from vforge.distributed.parallel_state import (
    ParallelConfig,
    destroy_parallel_state,
    init_parallel_state,
)
from vforge.distributed.parallelize import FSDP2Config, MixedPrecisionConfig, ParallelizeConfig
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.tasks.rep_world.optimizer import MuonWithAuxAdam
from vforge.training.engine import EnginePolicy
from vforge.training.engine.fsdp import FsdpEngine


def full(x):
    return x.full_tensor() if hasattr(x, "full_tensor") else x


def build(mode, domain, patch):
    geo = CodecGeometry("test", "test", 4, 4, 2, grid=4)
    vit = ConditionerConfig(
        input=domain,
        input_size=32,
        width=48,
        depth=2,
        heads=3,
        temporal_patch=patch,
        feature_layers=(0, 1),
    )
    world = WorldConfig(
        4,
        2,
        grid=4,
        width=48,
        heads=3,
        depth=2,
        condition_width=0 if mode == "pure_tf" else 48,
        condition_dropout=0.25,
        adaln_lora_dim=16,
    )
    model = RepWorld(geo, world, None if mode == "pure_tf" else vit, FlowConfig(mode=mode))
    with torch.no_grad():
        model.net.final_layer.linear.weight.normal_(std=0.03)
        model.net.shared_mod[-1].weight.normal_(std=0.02)
    return model


def disable(model):
    for m in model.modules():
        if hasattr(m, "sequence_parallel"):
            m.sequence_parallel = False


def inputs(mode, source, device):
    g = torch.Generator(device=device).manual_seed(771 + source)
    lat = torch.randn(1, 8, 4, 4, 4, device=device, generator=g)
    rgb = torch.randn(1, 3, 16, 32, 32, device=device, generator=g)
    shape = (1, 4, 3, 2, 4, 4) if mode == "pure_tf" else (3, 4, 2, 4, 4)
    noise = torch.randn(shape, device=device, generator=g)
    u = torch.rand((1, 3) if mode == "pure_tf" else (3, 1), device=device, generator=g) * 0.8 + 0.1
    return dict(
        latents=lat,
        rgb=rgb,
        noise=noise,
        uniform=u,
        condition_drop=None
        if mode == "pure_tf"
        else torch.tensor([False, True, False], device=device),
    )


def run_case(state, device, mode, domain, patch, amp, recompute):
    torch.manual_seed(420)
    reference = build(mode, domain, patch).to(device)
    disable(reference)
    candidate = build(mode, domain, patch).to(device)
    candidate.load_state_dict(reference.state_dict())
    cfg = ParallelizeConfig(
        parallel=state.config,
        fsdp2=FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=False)),
        gradient_checkpointing=recompute,
    )
    candidate = FsdpEngine(EnginePolicy(parallelize=cfg)).prepare_model(candidate)
    opt = MuonWithAuxAdam(candidate, lr=1e-4, weight_decay=0)
    ropt = MuonWithAuxAdam(reference, lr=1e-4, weight_decay=0)
    precision = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    rows = []
    for step in range(2):
        candidate.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        expected_loss = 0.0
        rank_loss = None
        for source in range(state.config.dp_size):
            data = inputs(mode, source + step * 13, device)
            with precision():
                loss = reference(**data)
            if source == state.dp_rank:
                rank_loss = loss.detach()
            expected_loss += float(loss.detach()) / state.config.dp_size
            (loss / state.config.dp_size).backward()
        with precision():
            actual = candidate(**inputs(mode, state.dp_rank + step * 13, device))
        torch.testing.assert_close(
            actual, rank_loss, atol=2e-3 if amp else 2e-6, rtol=2e-3 if amp else 2e-6
        )
        actual.backward()
        maxima = []
        error_sq = 0.0
        reference_sq = 0.0
        for name, p in candidate.named_parameters():
            q = dict(reference.named_parameters())[name]
            if q.grad is None:
                assert p.grad is None, name
                continue
            grad = full(p.grad)
            diff = (grad - q.grad).float()
            maxima.append(float(diff.abs().max()))
            error_sq += float(diff.square().sum())
            reference_sq += float(q.grad.float().square().sum())
            torch.testing.assert_close(
                grad, q.grad, atol=4e-3 if amp else 3e-6, rtol=0.06 if amp else 2e-4, msg=name
            )
        rel = (error_sq / max(reference_sq, 1e-30)) ** 0.5
        assert rel < (0.035 if amp else 3e-5), (rel, max(maxima), mode, domain, patch, step)
        # First verify every distributed gradient against the unsharded oracle.
        # Muon's BF16 Newton-Schulz can amplify tiny reduction roundoff. Test the
        # distributed optimizer itself with exactly those already-validated full
        # gradients, so this assertion isolates sharding rather than quantization.
        for name, p in candidate.named_parameters():
            q = dict(reference.named_parameters())[name]
            if p.grad is not None:
                q.grad.copy_(full(p.grad))
        opt.step()
        ropt.step()
        update_max = 0.0
        for name, p in candidate.named_parameters():
            q = dict(reference.named_parameters())[name]
            diff = float((full(p).detach() - q.detach()).abs().max())
            update_max = max(update_max, diff)
            torch.testing.assert_close(full(p), q, atol=0, rtol=0, msg=name)
        rows.append(
            dict(
                step=step,
                loss=float(actual.detach()),
                reference_loss=float(rank_loss),
                max_gradient_error=max(maxima),
                relative_gradient_error=rel,
                max_parameter_error=update_max,
            )
        )
    record = dict(mode=mode, domain=domain, patch=patch, amp=amp, recompute=recompute, steps=rows)
    if state.global_rank == 0:
        print("PARITY " + json.dumps(record), flush=True)
    del candidate, reference, opt, ropt
    torch.cuda.empty_cache()
    return record


@torch.no_grad()
def cache_case(state, device, amp):
    torch.manual_seed(72)
    ref = build("pure_tf", "rgb", 2).to(device).eval()
    disable(ref)
    model = build("pure_tf", "rgb", 2).to(device).eval()
    model.load_state_dict(ref.state_dict())
    precision = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    past = None
    r_past = None
    rows = []
    for i in range(3):
        gen = torch.Generator(device=device).manual_seed(21 + i)
        x = torch.randn(1, 4, 2, 4, 4, device=device, generator=gen)
        kwargs = dict(
            operation="world",
            noisy=x,
            timestep=torch.zeros(1, 2, device=device),
            frame_offset=i * 2,
            causal_state_planes=2,
            return_key_values=True,
        )
        with precision():
            actual, new = model(**kwargs, past_key_values=past)
            expected, r_new = ref(**kwargs, past_key_values=r_past)
        torch.testing.assert_close(
            actual, expected, atol=2e-3 if amp else 2e-6, rtol=2e-3 if amp else 2e-6
        )
        for pair, rpair in zip(new, r_new, strict=True):
            for a, b in zip(pair, rpair, strict=True):
                torch.testing.assert_close(
                    a, b, atol=2e-3 if amp else 2e-6, rtol=2e-3 if amp else 2e-6
                )
        rows.append(float((actual - expected).abs().max()))
        past = (
            new
            if past is None
            else [
                (torch.cat([a[0], b[0]], -2), torch.cat([a[1], b[1]], -2))
                for a, b in zip(past, new, strict=True)
            ]
        )
        r_past = (
            r_new
            if r_past is None
            else [
                (torch.cat([a[0], b[0]], -2), torch.cat([a[1], b[1]], -2))
                for a, b in zip(r_past, r_new, strict=True)
            ]
        )
    if state.global_rank == 0:
        print("CACHE " + json.dumps(dict(amp=amp, errors=rows)), flush=True)
    return dict(amp=amp, errors=rows)


@torch.no_grad()
def random_case(state, device):
    torch.manual_seed(91)
    reference = build("no_zprev", "latent", 2).to(device)
    disable(reference)
    candidate = build("no_zprev", "latent", 2).to(device)
    candidate.load_state_dict(reference.state_dict())
    data = inputs("no_zprev", state.dp_rank, device)
    data = {k: v for k, v in data.items() if k in {"latents", "rgb"}}
    source_rank = dist.get_global_rank(state.usp_group, 0)
    torch.manual_seed(300 + source_rank)
    expected = reference(**data)
    torch.manual_seed(300 + state.global_rank)
    actual = candidate(**data)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    return {
        "rank": state.global_rank,
        "leader": source_rank,
        "loss_error": float((actual - expected).abs()),
        "different_rank_rng": True,
    }


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--usp", type=int, default=2)
    a.add_argument("--replicate", type=int, default=1)
    a.add_argument("--min-hosts", type=int, default=1)
    a.add_argument("--output", type=Path, required=True)
    n = a.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    world = dist.get_world_size()
    dp = world // n.usp
    state = init_parallel_state(
        ParallelConfig(
            dp_size=dp,
            usp_size=n.usp,
            dp_mode="fsdp2",
            dp_replicate_size=n.replicate,
            dp_shard_size=dp // n.replicate,
            device_type="cuda",
        )
    )
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False), device=device, parallel_state=state
    )
    topology = [None] * world
    dist.all_gather_object(
        topology,
        dict(
            rank=state.global_rank,
            hostname=socket.gethostname(),
            gpu=torch.cuda.get_device_name(device),
            usp_ranks=dist.get_process_group_ranks(state.usp_group),
        ),
    )
    if len({entry["hostname"] for entry in topology}) < n.min_hosts:
        raise AssertionError(f"Expected at least {n.min_hosts} distinct worker hosts")
    rows = []
    caches = []
    try:
        random_check = random_case(state, device)
        for amp in [False, True]:
            for recompute in [False, True]:
                for mode, domain, patch in [
                    ("no_zprev", "rgb", 2),
                    ("no_zprev", "rgb", 4),
                    ("no_zprev", "latent", 2),
                    ("zprev", "rgb", 2),
                    ("pure_tf", "rgb", 2),
                ]:
                    rows.append(run_case(state, device, mode, domain, patch, amp, recompute))
            caches.append(cache_case(state, device, amp))
        if state.global_rank == 0:
            n.output.write_text(
                json.dumps(
                    dict(
                        world=world,
                        usp=n.usp,
                        dp=dp,
                        replicate=n.replicate,
                        torch=torch.__version__,
                        topology=topology,
                        tf32=False,
                        cases=rows,
                        caches=caches,
                        random_check=random_check,
                        passed=True,
                    ),
                    indent=2,
                )
            )
    finally:
        reset_ops()
        destroy_parallel_state()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
