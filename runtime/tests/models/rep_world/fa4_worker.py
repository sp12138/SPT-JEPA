"""B300 FA4 gate: exact visibility, independent BF16 gradients and Muon updates.

Thresholds below are fixed before the remote run. SDPA remains an independent
execution: no gradient copying, backend fallback or parameter synchronization
between paired optimizer trajectories.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.models.rep_world.sparsity import (
    prefix_sparsity,
    state_causal_sparsity,
    teacher_forcing_sparsity,
)
from vforge.ops import OpsConfig, attention, initialize_ops, reset_ops
from vforge.ops._attention import dispatch
from vforge.ops._attention.segmented import segmented_dense_mask
from vforge.tasks.rep_world.optimizer import MuonWithAuxAdam

LIMITS = dict(
    kernel_output_rel=0.005,
    kernel_gradient_rel=0.01,
    model_output_rel=0.01,
    loss_rel=0.005,
    gradient_rel=0.025,
    tensor_gradient_rel=0.08,
    tensor_gradient_rms_atol=2e-7,
    update_rel=0.05,
)
DEVICE = None
DIAGNOSTIC = False
DETERMINISTIC = False
VIOLATIONS = []
BACKENDS = ("sdpa", "flash_attention_4")


def select(backend):
    reset_ops()
    initialize_ops(
        OpsConfig(backend=backend, allow_fallback=False, deterministic=DETERMINISTIC),
        device=DEVICE,
        parallel_state=None,
    )


@contextmanager
def count_kernels():
    current = dict(fa_dense=0, fa_segmented=0, sdpa=0)

    def wrap(name, fn):
        def call(*args, **kwargs):
            current[name] += 1
            return fn(*args, **kwargs)

        return call

    with (
        patch.object(
            dispatch, "_run_flash_attention", wrap("fa_dense", dispatch._run_flash_attention)
        ),
        patch.object(
            dispatch,
            "_run_segmented_attention",
            wrap("fa_segmented", dispatch._run_segmented_attention),
        ),
        patch.object(dispatch, "_run_sdpa", wrap("sdpa", dispatch._run_sdpa)),
    ):
        yield current


def metric(actual, expected):
    a, b = actual.detach().double(), expected.detach().double()
    assert a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    delta = a - b
    return dict(
        max_abs=float(delta.abs().max()),
        rms=float(delta.square().mean().sqrt()),
        reference_rms=float(b.square().mean().sqrt()),
        rel_l2=float(delta.norm() / b.norm().clamp_min(1e-30)),
        error_sq=float(delta.square().sum()),
        reference_sq=float(b.square().sum()),
    )


def aggregate(rows):
    return (
        sum(r["error_sq"] for r in rows) / max(sum(r["reference_sq"] for r in rows), 1e-30)
    ) ** 0.5


def require(value, maximum, label):
    if not value <= maximum:
        message = f"{label}: {value} > {maximum}"
        if DIAGNOSTIC:
            VIOLATIONS.append(message)
            print("NUMERIC_VIOLATION", message, flush=True)
        else:
            raise AssertionError(message)


def kernel_gate():
    torch.manual_seed(419)
    rows = []
    cases = [
        ("dense", 513, 513, None),
        ("short_state", 96, 96, state_causal_sparsity(96, 32)),
        ("cross", 257, 204, None),
        ("state_unaligned", 411, 411, state_causal_sparsity(411, 137)),
        ("mask80", 612, 612, state_causal_sparsity(612, 204)),
        ("prefix", 333, 333, prefix_sparsity(333, 137)),
        ("pure_tf", 438, 438, teacher_forcing_sparsity(438, 73)),
        ("kv_cache", 137, 411, state_causal_sparsity(137, 137, 274)),
    ]
    for name, nq, nk, spec in cases:
        q = torch.randn(2, nq, 3, 64, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(2, nk, 3, 64, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        grad = torch.randn_like(q)
        dense = None if spec is None else segmented_dense_mask(spec, device=DEVICE)
        oracle_inputs = [x.double().detach().requires_grad_() for x in (q, k, v)]
        # FP64 math attention is independent of both the sparse kernel and its tile metadata.
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            oracle = F.scaled_dot_product_attention(
                *(x.transpose(1, 2) for x in oracle_inputs), attn_mask=dense
            ).transpose(1, 2)
        oracle_grads = torch.autograd.grad(oracle, oracle_inputs, grad.double())
        result = {}
        for backend in ("sdpa", "flash_attention_4"):
            select(backend)
            inputs = [x.detach().clone().requires_grad_() for x in (q, k, v)]
            with count_kernels() as counts:
                out = attention(*inputs, sparsity=spec)
                grads = torch.autograd.grad(out, inputs, grad)
            metrics = [metric(out, oracle)] + [
                metric(a, b) for a, b in zip(grads, oracle_grads, strict=True)
            ]
            require(metrics[0]["rel_l2"], LIMITS["kernel_output_rel"], name + "/output")
            for index, entry in enumerate(metrics[1:]):
                require(entry["rel_l2"], LIMITS["kernel_gradient_rel"], name + f"/grad{index}")
            if backend == "flash_attention_4":
                assert counts["sdpa"] == 0 and counts["fa_dense"] + counts["fa_segmented"] > 0
            result[backend] = dict(metrics=metrics, calls=counts)
        rows.append(dict(case=name, shape=[2, nq, nk, 3, 64], results=result))
        print("KERNEL_PASS", name, flush=True)
    # Strict causality: changing future keys/values must leave early outputs
    # identical and the reverse gradient to future tokens exactly zero.
    select("flash_attention_4")
    spec = state_causal_sparsity(411, 137)
    q = torch.randn(1, 411, 3, 64, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q).requires_grad_()
    v = torch.randn_like(q).requires_grad_()
    out = attention(q, k, v, sparsity=spec)
    changed_k, changed_v = k.detach().clone(), v.detach().clone()
    changed_k[:, 274:] += 50
    changed_v[:, 274:] -= 50
    changed = attention(q, changed_k, changed_v, sparsity=spec)
    torch.testing.assert_close(out[:, :274], changed[:, :274], rtol=0, atol=0)
    gk, gv = torch.autograd.grad(out[:, 137:274].float().square().sum(), (k, v))
    assert gk[:, 274:].count_nonzero() == 0 and gv[:, 274:].count_nonzero() == 0
    return dict(cases=rows, future_output_bitwise_equal=True, future_gradient_exact_zero=True)


def small_model(domain, mode, ratio, temporal):
    geo = CodecGeometry("test", "test", 16, 8, 4, grid=8)
    conditioner = (
        None
        if mode == "pure_tf"
        else ConditionerConfig(
            input=domain,
            input_size=32,
            temporal_patch=temporal,
            width=192,
            heads=3,
            depth=2,
            feature_layers=(0, 1),
            context_mask_ratio=ratio,
        )
    )
    world = WorldConfig(
        8,
        4,
        grid=8,
        width=192,
        heads=3,
        depth=2,
        condition_width=0 if conditioner is None else 192,
        condition_dropout=0.1,
        adaln_lora_dim=64,
    )
    model = RepWorld(geo, world, conditioner, FlowConfig(mode=mode))
    with torch.no_grad():
        model.net.final_layer.linear.weight.normal_(std=0.03)
        model.net.shared_mod[-1].weight.normal_(std=0.02)
    return model


def model_inputs(model, states, seed, codec_latents=None, rgb=None):
    geo, mode = model.geometry, model.flow.mode
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    lat = (
        codec_latents
        if codec_latents is not None
        else torch.randn(
            1, geo.packed_channels, states, geo.grid, geo.grid, device=DEVICE, generator=gen
        )
    )
    pixels = (
        rgb
        if rgb is not None
        else torch.randn(1, 3, geo.rgb_frames * states, 32, 32, device=DEVICE, generator=gen)
    )
    noise_shape = (
        (1, geo.channels, states - 1, geo.planes, geo.grid, geo.grid)
        if mode == "pure_tf"
        else (states - 1, geo.channels, geo.planes, geo.grid, geo.grid)
    )
    uniform_shape = (1, states - 1) if mode == "pure_tf" else (states - 1, 1)
    return dict(
        latents=lat,
        rgb=pixels,
        noise=torch.randn(noise_shape, device=DEVICE, generator=gen),
        uniform=torch.rand(uniform_shape, device=DEVICE, generator=gen) * 0.8 + 0.1,
        condition_drop=None
        if mode == "pure_tf"
        else (torch.arange(states - 1, device=DEVICE) % 3 == 1),
    )


def trajectory(reference, candidate, data, steps, recompute=False):
    if recompute:
        reference.gradient_checkpointing_enable()
        candidate.gradient_checkpointing_enable()
    opts = [MuonWithAuxAdam(m, lr=1e-4, weight_decay=0) for m in (reference, candidate)]
    input_hashes = {
        name: hashlib.sha256(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for name, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    rows = []
    for step in range(steps):
        before = [
            {n: p.detach().clone() for n, p in m.named_parameters()} for m in (reference, candidate)
        ]
        outputs, losses, counts_all = [], [], []
        for backend, model in zip(BACKENDS, (reference, candidate), strict=True):
            select(backend)
            torch.manual_seed(720 + step)
            model.zero_grad(set_to_none=True)
            captured = {}
            handles = [
                model.net.register_forward_hook(
                    lambda m, a, output, captured=captured: captured.update(
                        velocity=output.detach().clone()
                    )
                )
            ]
            if model.causal_semantic_encoder is not None:
                handles.append(
                    model.causal_semantic_encoder.register_forward_hook(
                        lambda m, a, output, captured=captured: captured.update(
                            condition=output.detach().clone()
                        )
                    )
                )
            with count_kernels() as counts:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**data)
                loss.backward()
            for handle in handles:
                handle.remove()
            losses.append(loss.detach())
            outputs.append(captured)
            counts_all.append(counts.copy())
        for backend, counts in zip(BACKENDS, counts_all, strict=True):
            if backend == "flash_attention_4":
                assert counts["sdpa"] == 0
                assert counts["fa_segmented"] > 0
            else:
                assert counts["sdpa"] > 0
        loss_metric = metric(losses[1], losses[0])
        require(loss_metric["rel_l2"], LIMITS["loss_rel"], "loss")
        output_metrics = {
            name: metric(outputs[1][name], expected) for name, expected in outputs[0].items()
        }
        for name, item in output_metrics.items():
            require(item["rel_l2"], LIMITS["model_output_rel"], name)
        gradients = {}
        for (name, a), (_, b) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            if b.grad is None:
                assert a.grad is None, name
                continue
            item = metric(a.grad, b.grad)
            allowed = (
                LIMITS["tensor_gradient_rel"] * item["reference_rms"]
                + LIMITS["tensor_gradient_rms_atol"]
            )
            require(item["rms"], allowed, name)
            gradients[name] = item
        grad_rel = aggregate(list(gradients.values()))
        require(grad_rel, LIMITS["gradient_rel"], "all gradients")
        for model, opt in zip((reference, candidate), opts, strict=True):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        updates = {}
        for (name, a), (_, b) in zip(
            candidate.named_parameters(), reference.named_parameters(), strict=True
        ):
            updates[name] = metric(a - before[1][name], b - before[0][name])
        update_rel = aggregate(list(updates.values()))
        require(update_rel, LIMITS["update_rel"], "independent Muon updates")
        rows.append(
            dict(
                step=step + 1,
                input_sha256=input_hashes,
                losses=[float(x) for x in losses],
                loss=loss_metric,
                outputs=output_metrics,
                gradient_rel=grad_rel,
                update_rel=update_rel,
                gradients=gradients,
                updates=updates,
                calls=counts_all,
            )
        )
        print("MODEL_STEP_METRICS", step + 1, "grad", grad_rel, "update", update_rel, flush=True)
    return rows


def cache_gate(reference):
    """Verify full-prefix and incremental KV semantics under both backends."""
    from copy import deepcopy

    model = deepcopy(reference).eval()
    geo = model.geometry
    generator = torch.Generator(device=DEVICE).manual_seed(803)
    states = torch.randn(
        1,
        geo.channels,
        3 * geo.planes,
        geo.grid,
        geo.grid,
        device=DEVICE,
        generator=generator,
    )
    results = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for backend in ("sdpa", "flash_attention_4"):
            select(backend)
            cache = None
            chunks = []
            for index in range(3):
                chunk = states[:, :, index * geo.planes : (index + 1) * geo.planes]
                output, current = model.net(
                    chunk,
                    torch.zeros(1, device=DEVICE),
                    causal_state_planes=geo.planes,
                    frame_offset=index * geo.planes,
                    past_key_values=cache,
                    return_key_values=True,
                )
                # WorldDiT returns only this chunk's K/V, as used by the
                # production sampler. The caller appends the retained history.
                cache = (
                    current
                    if cache is None
                    else [
                        (torch.cat((old_k, new_k), -2), torch.cat((old_v, new_v), -2))
                        for (old_k, old_v), (new_k, new_v) in zip(cache, current, strict=True)
                    ]
                )
                chunks.append(output)
            complete = model.net(
                states,
                torch.zeros(1, device=DEVICE),
                causal_state_planes=geo.planes,
            )
            incremental = torch.cat(chunks, dim=2)
            entry = metric(incremental, complete)
            require(entry["rel_l2"], LIMITS["model_output_rel"], backend + "/KV-vs-full")
            results[backend] = (complete, incremental, entry)
    cross = metric(results["flash_attention_4"][1], results["sdpa"][1])
    require(cross["rel_l2"], LIMITS["model_output_rel"], "FA4-vs-SDPA KV")
    return dict(
        same_backend={name: value[2] for name, value in results.items()},
        cross_backend=cross,
    )


def models_gate(index):
    variants = [
        ("latent", "no_zprev", 0.0, 2, False),
        ("latent", "no_zprev", 0.8, 2, False),
        ("rgb", "no_zprev", 0.0, 2, False),
        ("rgb", "no_zprev", 0.0, 4, False),
        ("rgb", "zprev", 0.0, 2, False),
        ("rgb", "pure_tf", 0.0, 2, False),
        ("latent", "no_zprev", 0.0, 2, True),
        ("rgb", "pure_tf", 0.0, 2, True),
    ]
    domain, mode, ratio, temporal, recompute = variants[index]
    torch.manual_seed(420)
    reference = small_model(domain, mode, ratio, temporal).to(DEVICE)
    candidate = small_model(domain, mode, ratio, temporal).to(DEVICE)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    data = model_inputs(reference, 4, 1984)
    rows = trajectory(reference, candidate, data, 3, recompute)
    result = dict(variant=variants[index], steps=rows)
    if mode == "pure_tf":
        result["cache"] = cache_gate(reference)
    return result


def full_gate(checkpoint, fixed_input, codec_checkpoint, statistics):
    from vforge.tasks.rep_world.shared import CodecProvider, parse_config
    from vforge.tasks.rep_world.weights import load_legacy_model

    config = parse_config(
        dict(
            codec=dict(
                family="videorae",
                variant="3d",
                checkpoint=codec_checkpoint,
                statistics=statistics,
                encode_batch_size=40,
            ),
            conditioner=dict(input="latent"),
            initial_weights_format="legacy",
            states=10,
        )
    )
    select("sdpa")
    rgb = torch.load(fixed_input, map_location="cpu", weights_only=True)["video"][:1].float()
    content_hash = hashlib.sha256(rgb.contiguous().numpy().tobytes()).hexdigest()
    provider = CodecProvider(config, "bf16")
    codec = provider.get(DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latent = codec.encode(rgb.to(DEVICE))
    del codec, provider
    gc.collect()
    torch.cuda.empty_cache()
    reference = load_legacy_model(checkpoint, config).to(DEVICE)
    candidate = load_legacy_model(checkpoint, config).to(DEVICE)
    for (name, a), (_, b) in zip(
        reference.named_parameters(), candidate.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)
    data = model_inputs(reference, 10, 1984, latent.detach(), rgb.to(DEVICE))
    result = trajectory(reference, candidate, data, 3)
    return dict(
        input_sha256=content_hash,
        checkpoint=checkpoint,
        parameters=sum(p.numel() for p in reference.parameters()),
        steps=result,
    )


def profile_gate():
    from vforge.training.profiling import _warmup_cuda_profiler

    select("flash_attention_4")
    q = torch.randn(1, 411, 3, 64, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    spec = state_causal_sparsity(411, 137)
    attention(q, q, q, sparsity=spec).float().sum().backward()
    _warmup_cuda_profiler()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        attention(q, q, q, sparsity=spec).float().sum().backward()
        torch.cuda.synchronize()
    events = [
        event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert events, "Invalid profiler result: no CUDA activity"
    selected = sorted(
        {name for name in events if any(x in name.lower() for x in ("flash", "fmha", "cute"))}
    )
    assert selected, sorted(set(events))
    return dict(cuda_events=len(events), attention_kernels=selected)


def main():
    global DEVICE, DIAGNOSTIC, BACKENDS, DETERMINISTIC
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["kernels", "models", "full", "profile"], required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--input", default="")
    p.add_argument("--codec-checkpoint", default="")
    p.add_argument("--statistics", default="")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="Collect all violations; still exit nonzero on any failure",
    )
    p.add_argument("--reference-backend", choices=["sdpa", "flash_attention_4"], default="sdpa")
    p.add_argument(
        "--candidate-backend", choices=["sdpa", "flash_attention_4"], default="flash_attention_4"
    )
    args = p.parse_args()
    DIAGNOSTIC = args.diagnostic
    DETERMINISTIC = args.deterministic
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    BACKENDS = (args.reference_backend, args.candidate_backend)
    if args.stage == "full" and not all(
        (args.checkpoint, args.input, args.codec_checkpoint, args.statistics)
    ):
        p.error("full stage requires checkpoint, input, codec-checkpoint and statistics")
    DEVICE = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(DEVICE)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert "B300" in torch.cuda.get_device_name(DEVICE)
    result = dict(
        stage=args.stage,
        index=args.index,
        limits=LIMITS,
        gpu=torch.cuda.get_device_name(DEVICE),
        host=socket.gethostname(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        backends=BACKENDS,
        diagnostic=DIAGNOSTIC,
        deterministic=DETERMINISTIC,
    )
    try:
        result["data"] = {
            "kernels": kernel_gate,
            "models": lambda: models_gate(args.index),
            "full": lambda: full_gate(
                args.checkpoint, args.input, args.codec_checkpoint, args.statistics
            ),
            "profile": profile_gate,
        }[args.stage]()
        result["passed"] = not VIOLATIONS
        if VIOLATIONS:
            raise AssertionError(f"{len(VIOLATIONS)} numerical checks failed; see violations")
    except BaseException as exc:
        result.update(passed=False, error=repr(exc))
        raise
    finally:
        result["violations"] = list(VIOLATIONS)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
    print("FA4_GATE_PASS", args.stage, args.index, flush=True)


if __name__ == "__main__":
    main()
