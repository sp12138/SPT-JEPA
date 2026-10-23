"""MMDiT main-bank EP + USP numerical gate, launched by torchrun."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from vforge.distributed.expert_parallel import Expert
from vforge.distributed.parallel_state import (
    ParallelConfig,
    ParallelState,
    destroy_parallel_state,
    get_parallel_state,
)
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
    _apply_fsdp2,
    materialize_meta_module,
    parallelize_model,
)
from vforge.models.gen_engine.dit import GenEngineModel
from vforge.ops import initialize_ops, reset_ops
from vforge.ops.config import OpsConfig
from vforge.training.diagnostics import LogicalReduction, clip_grad_norm_logical_


class UnshardedExpert(Expert):
    """Independent per-assignment matmul with FP32 optimizer storage."""

    def forward(self, states, expert_ids):
        up, gate = (
            torch.bmm(self.gate_up[expert_ids], states.to(self.gate_up.dtype).unsqueeze(-1))
            .squeeze(-1)
            .chunk(2, dim=-1)
        )
        return torch.bmm(
            self.down[expert_ids], (up * torch.nn.functional.silu(gate)).unsqueeze(-1)
        ).squeeze(-1)


def small_config():
    path = Path(__file__).resolve().parents[3] / (
        "vforge/models/gen_engine/dit/configs/mmdit-moe-32b-a4.1b.json"
    )
    config = json.loads(path.read_text())
    config.update(
        num_layers=4,
        hidden_size=384,
        num_attention_heads=12,
        attention_head_dim=32,
        ffn_dim=1024,
        refiner_ffn_dim=1024,
        time_embed_hidden_dim=384,
        time_embed_dim=192,
        rope_freq_dim=4,
    )
    config["moe"].update(expert_intermediate_size=384, dispatcher="vforge_expert")
    return config


def inputs(device, *, source, case):
    # Seven rows require padding at USP2/USP4; some shards have no visual rows.
    video, audio, text = (0, 5, 2) if case == "audio" else (3, 2, 2)
    generator = torch.Generator(device=device).manual_seed(42 + source)
    total = video + audio + text
    tags = torch.tensor([1] * text + [0] * video + [2] * audio, device=device)
    return {
        "hidden_states": torch.randn(
            1, video, 96, generator=generator, device=device
        ).requires_grad_(),
        "audio_hidden_states": torch.randn(
            1, audio, 32, generator=generator, device=device
        ).requires_grad_(),
        "encoder_hidden_states": torch.randn(
            1, text, 5120, generator=generator, device=device
        ).requires_grad_(),
        "timestep": torch.tensor([0.25, 0.75], device=device),
        "timestep_indices": (tags == 2).long(),
        "token_tags": tags,
        "position_ids": torch.randn(total, 3, generator=generator, device=device),
        "text_indices": torch.arange(text, device=device),
        "video_indices": torch.arange(text, text + video, device=device),
        "audio_indices": torch.arange(text + video, total, device=device),
        "reference_modality_ids": torch.zeros(total, dtype=torch.long, device=device),
        "reference_index_ids": torch.zeros(total, dtype=torch.long, device=device),
        "audio_plane_ids": (tags == 2).long(),
    }


def loss_of(output):
    # A sum also supports an empty video stream without NaN from mean([]).
    return (output.sample.float() - 0.25).square().sum() / 288 + (
        output.audio_sample.float() + 0.5
    ).square().sum() / 64


def full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def setup_ops(device, state):
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False, deterministic=True),
        device=device,
        parallel_state=state,
    )


def check_close(actual, expected, label):
    if expected is None:
        assert actual is None, label
        return
    assert actual is not None, label
    actual = torch.as_tensor(actual, device=expected.device)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.05,
        atol=0.005,
        msg=lambda details: f"{label}: {details}",
    )


def check_gradient(actual, expected, label):
    if expected is None:
        assert actual is None, label
        return
    assert actual is not None, label
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    if not expected.count_nonzero():
        assert not actual.count_nonzero(), label
        return
    relative_error = ((actual - expected).norm() / expected.norm()).item()
    assert relative_error <= 0.05, (
        label,
        relative_error,
        "actual_norm",
        actual.norm().item(),
        "expected_norm",
        expected.norm().item(),
        "max_error",
        (actual - expected).abs().max().item(),
    )


def run_case(device, args, case, recompute):
    world = dist.get_world_size()
    config = small_config()
    print(f"rank={dist.get_rank()} build {case} recompute={recompute}", flush=True)
    reference = GenEngineModel.build_empty(config=config, parameter_dtype=torch.float32)
    materialize_meta_module(reference, device)
    reference.initialize_parameters(seed=42)
    for block in reference.blocks:
        if block.is_moe:
            bank = block.visual_moe.main_bank
            oracle = UnshardedExpert(160, 384, 384, projection_order="up_gate").to(device)
            oracle.load_state_dict(bank.state_dict())
            block.visual_moe.main_bank = oracle
    # Scratch output projections and residual gates are zero. Open them in the
    # numerical probe so the first backward actually measures expert gradients.
    with torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(43)
        for name, parameter in reference.named_parameters():
            if "proj_out.weight" in name or "adaln_modulation.linear.weight" in name:
                parameter.normal_(0, 0.01, generator=generator)
        if case in {"balanced", "skew"}:
            # Hard Top-K is discontinuous at a tie. Keep a clear selection
            # margin for numerical parity, while scores and router gradients
            # remain trainable. Native Trainer tests use the natural router.
            preferred = torch.arange(0, 160, 20) if case == "balanced" else torch.arange(8)
            for block in reference.blocks:
                if block.is_moe:
                    block.visual_moe.router.main_routing_bias[preferred] = 10
    candidate = GenEngineModel.build_empty(config=config, parameter_dtype=torch.float32)
    materialize_meta_module(candidate, device)
    candidate.load_state_dict(reference.state_dict())
    candidate = parallelize_model(
        candidate,
        ParallelizeConfig(
            parallel=ParallelConfig(
                dp_size=world // args.usp,
                dp_replicate_size=args.replicate,
                dp_shard_size=world // args.usp // args.replicate,
                usp_size=args.usp,
                extra_parallel_sizes=(args.ep,),
            ),
            fsdp2=FSDP2Config(
                mixed_precision=MixedPrecisionConfig(enable=True),
                leaf_module_paths=("context_embedder", "token_refiner.final_norm"),
            ),
            gradient_checkpointing=recompute,
        ),
    )
    state = get_parallel_state()
    # A singleton FSDP mesh gives the oracle the same module precision
    # boundaries as Native training, without sharding or expert dispatch.
    for rank in range(world):
        group = dist.new_group([rank])
        if rank == dist.get_rank():
            oracle_mesh = DeviceMesh.from_group(group, "cuda", mesh=[rank])
    with patch(
        "vforge.distributed.parallelize.get_parallel_state",
        return_value=SimpleNamespace(fsdp_mesh=oracle_mesh),
    ):
        reference = _apply_fsdp2(
            reference,
            FSDP2Config(
                mixed_precision=MixedPrecisionConfig(enable=True),
                leaf_module_paths=("context_embedder", "token_refiner.final_norm"),
            ),
        )
    setup_ops(device, state)
    candidate.train()
    reference.train()

    def require_dtype(dtype):
        def check(module, arguments, output):
            assert output.dtype == dtype, (type(module).__name__, output.dtype, dtype)

        return check

    for model in (candidate, reference):
        for block in model.blocks:
            if block.is_moe:
                block.visual_moe.router.main_router.register_forward_hook(
                    require_dtype(torch.float32)
                )
                block.visual_moe.main_bank.register_forward_hook(require_dtype(torch.bfloat16))
    optimizers = [
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0,
            foreach=False,
            fused=False,
        )
        for model in (candidate, reference)
    ]
    for block in candidate.blocks:
        if block.is_moe:
            parameter = block.visual_moe.main_bank.gate_up
            assert parameter.to_local().shape[0] == 160 // args.ep
            assert parameter.device_mesh["ep_fsdp"].size() == world // args.ep // args.replicate
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        setup_ops(device, state)
        batch = inputs(device, source=state.dp_rank, case=case)
        print(f"rank={dist.get_rank()} candidate forward step={step}", flush=True)
        output = candidate(**batch)
        print(f"rank={dist.get_rank()} candidate backward step={step}", flush=True)
        loss_of(output).backward()
        print(f"rank={dist.get_rank()} oracle step={step}", flush=True)
        # The oracle executes full sequences independently for every DP sample.
        oracle_state = ParallelState()
        setup_ops(device, oracle_state)
        with patch(
            "vforge.models.gen_engine.dit.transformer.get_parallel_state", return_value=oracle_state
        ):
            for source in range(state.config.dp_size):
                source_batch = inputs(device, source=source, case=case)
                expected = reference(**source_batch)
                (loss_of(expected) / state.config.dp_size).backward()
                if source == state.dp_rank:
                    for field in ("sample", "audio_sample"):
                        check_close(getattr(output, field), getattr(expected, field), field)
                    for name, value in batch.items():
                        if value.requires_grad:
                            target = source_batch[name].grad
                            check_gradient(
                                value.grad,
                                None if target is None else target * state.config.dp_size,
                                name,
                            )
        # The model already summed all DP samples; consume the totals once.
        oracle_counts = {
            index: statistics.main.counts
            for index, statistics in reference.transformer._pending_moe_statistics.items()
        }
        reference.transformer._pending_moe_statistics.clear()
        for statistics in candidate.transformer._pending_moe_statistics.values():
            counts = statistics.main.counts.clone()
            dist.all_reduce(counts, group=state.fsdp_group)
            expected_counts = oracle_counts[statistics.layer_index]
            assert torch.equal(counts, expected_counts), (
                "forward routing counts",
                statistics.layer_index,
                "candidate",
                counts.tolist(),
                "oracle",
                expected_counts.tolist(),
            )
        # Finish model-owned streams before the test starts its own DTensor
        # all-gathers on the same ranks.
        torch.cuda.synchronize()
        schema = [
            (
                name,
                type(parameter.grad).__name__,
                tuple(parameter.grad.shape) if parameter.grad is not None else None,
                str(getattr(parameter.grad, "placements", None)),
            )
            for name, parameter in candidate.named_parameters()
        ]
        schemas = [None] * world
        dist.all_gather_object(schemas, schema)
        assert all(item == schema for item in schemas), "gradient layout differs across ranks"
        expert_nonzero = False
        update_diagnostics = {}
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in candidate.named_parameters():
            actual = full(parameter.grad)
            check_gradient(actual, full(reference_parameters[name].grad), f"gradient {name}")
            if actual is not None and actual.numel():
                expected = full(reference_parameters[name].grad)
                signs = actual * expected < 0
                if signs.any():
                    update_diagnostics[name] = {
                        "changed_sign": int(signs.sum()),
                        "gradient_rms": expected.square().mean().sqrt().item(),
                        "largest_expected_changed_sign": expected[signs].abs().max().item(),
                        "largest_actual_changed_sign": actual[signs].abs().max().item(),
                        "gradient_relative_error": (
                            (actual - expected).norm() / expected.norm()
                        ).item(),
                    }
            if ".main_bank." in name and actual is not None:
                expert_nonzero |= bool(actual.count_nonzero())
        assert expert_nonzero == (case != "audio"), (case, "expert gradients")
        actual_norm = clip_grad_norm_logical_(
            candidate, 1.0, LogicalReduction.for_model(candidate, state)
        )
        expected_norm = full(torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0))
        check_close(actual_norm, expected_norm, "clip norm")
        # Independent forward/backward comparisons above bound BF16 error.
        # Tiny gradients can change sign across GEMM reductions; AdamW then
        # amplifies that rounding into a full learning-rate-sized update.
        # Check the optimizer/sharding boundary with the same logical clipped
        # gradients, so rounding cannot hide a wrong shard update.
        for name, parameter in candidate.named_parameters():
            logical_gradient = full(parameter.grad)
            check_gradient(
                logical_gradient, full(reference_parameters[name].grad), f"clipped {name}"
            )
            reference_parameters[name].grad = (
                None
                if logical_gradient is None
                else DTensor.from_local(
                    logical_gradient.detach().clone(),
                    reference_parameters[name].device_mesh,
                    reference_parameters[name].placements,
                )
            )
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in candidate.named_parameters():
            expected_parameter = reference_parameters[name]
            torch.testing.assert_close(
                full(parameter), full(expected_parameter), rtol=0, atol=0, msg=name
            )
            for key, value in optimizers[0].state[parameter].items():
                torch.testing.assert_close(
                    full(value),
                    full(optimizers[1].state[expected_parameter][key]),
                    rtol=0,
                    atol=0,
                    msg=f"optimizer {name}.{key}",
                )
        if dist.get_rank() == 0 and update_diagnostics:
            summary = {
                "parameters_with_sign_changes": len(update_diagnostics),
                "max_relative_gradient_error": max(
                    item["gradient_relative_error"] for item in update_diagnostics.values()
                ),
                "context_bias": update_diagnostics.get("transformer.context_embedder.bias"),
            }
            print("MMDIT_BF16_ROUNDING " + json.dumps(summary, sort_keys=True), flush=True)
        # Reset stats each iteration. Compare counts separately from BF16 scores.
        candidate.post_optimizer_step(
            completed_step=step + 1, total_steps=10, process_group=state.fsdp_group
        )
        assert (
            candidate.post_optimizer_step(
                completed_step=step + 1, total_steps=10, process_group=state.fsdp_group
            )
            == {}
        )
        for statistics in candidate.transformer._last_moe_statistics:
            assert torch.equal(statistics.main.counts, oracle_counts[statistics.layer_index]), (
                "routing counts",
                statistics.layer_index,
            )
            assert statistics.main.token_count.item() == (
                0 if case == "audio" else 3 * state.config.dp_size
            )
        for block in candidate.blocks:
            if block.is_moe:
                bias = block.visual_moe.router.main_routing_bias
                replicas = [torch.empty_like(bias) for _ in range(world)]
                dist.all_gather(replicas, bias)
                assert all(torch.equal(bias, replica) for replica in replicas)
                # Bias is replicated model state; synchronize the oracle after
                # checking it to keep the next step's routing decision identical.
                reference.blocks[block.layer_index].visual_moe.router.main_routing_bias.copy_(bias)
        if dist.get_rank() == 0:
            print(
                f"MMDIT_EP_USP case={case} recompute={recompute} step={step + 1} "
                f"loss={loss_of(output).item()} norm={actual_norm}",
                flush=True,
            )
    del candidate, reference, optimizers
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--usp", type=int, required=True)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--case", default="balanced")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    faulthandler.dump_traceback_later(90, repeat=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    try:
        cases = ("balanced", "audio", "skew") if args.suite else (args.case,)
        modes = (False, True) if args.suite else (args.recompute,)
        for recompute in modes:
            for case in cases:
                run_case(device, args, case, recompute)
    finally:
        reset_ops()
        destroy_parallel_state()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
