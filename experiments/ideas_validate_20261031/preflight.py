"""Inspect an existing native short run; never implements or runs a trainer.

Run the repository's native train_stage entry to two updates first. This script
checks that checkpoint, then computes one frozen evaluation-mode backward pass
without an optimizer step. Expected frozen groups are explicit model/config data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.ideas_validate_20261031.audit import (
    expected_gradients, file_hash, make_manifest, move_batch, verify_manifest,
)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def in_group(name, prefix):
    return name == prefix or name.startswith(prefix.rstrip(".") + ".")


def gradient_report(model, expectations):
    rows = {}
    for prefix, expectation in expectations.items():
        named = [(name, p) for name, p in model.named_parameters() if in_group(name, prefix)]
        if not named:
            rows[prefix] = dict(expectation=expectation, parameters=0, parameter_tensors=0,
                                passed=False, reason="Declared group has no parameters")
            continue
        trainable = sum(p.numel() for _, p in named if p.requires_grad)
        grads = [p.grad for _, p in named if p.grad is not None]
        norm2 = sum(float(g.detach().double().square().sum()) for g in grads)
        norm = math.sqrt(norm2)
        gradients_finite = all(finite(g) for g in grads)
        if expectation == "frozen":
            passed = trainable == 0 and len(grads) == 0
        elif expectation == "active":
            passed = trainable > 0 and gradients_finite and norm > 0
        elif expectation == "optional":
            passed = trainable > 0 and gradients_finite
        else:
            raise ValueError(expectation)
        rows[prefix] = dict(expectation=expectation, parameters=sum(p.numel() for _, p in named),
            parameter_tensors=len(named), trainable_parameters=trainable,
            gradient_tensors=len(grads), gradient_norm=norm,
            gradients_finite=gradients_finite, passed=passed)
    return rows


def compare_tensors(left, right, prefixes):
    names = sorted(name for name in left if any(in_group(name, prefix) for prefix in prefixes))
    missing = [name for name in names if name not in right]
    different = [name for name in names if name in right and not torch.equal(left[name], right[name])]
    return dict(prefixes=list(prefixes), compared_tensors=len(names), missing=missing,
                different=different, equal=bool(names) and not missing and not different)


def checkpoint_report(checkpoint, portable, cfg, expected_step):
    step = int(checkpoint["global_step"])
    native = checkpoint["state_dict"]
    missing = [name for name in portable if "model." + name not in native]
    different = [name for name in portable if "model." + name in native
                 and not torch.equal(portable[name], native["model." + name])]
    rng = checkpoint.get("campaign_rng", {})
    return dict(step=step, expected_step=expected_step,
        checks=dict(exact_step=step == expected_step,
            model_finite=finite(native), portable_finite=finite(portable),
            optimizer_present=bool(checkpoint.get("optimizer_states")),
            optimizer_finite=finite(checkpoint.get("optimizer_states")),
            scheduler_present=bool(checkpoint.get("lr_schedulers")),
            scheduler_finite=finite(checkpoint.get("lr_schedulers")),
            rng_present=all(name in rng for name in ["python", "numpy", "torch", "cuda"]),
            sample_cursor=checkpoint.get("campaign_samples") == step * cfg["batch_size"],
            portable_equal=not missing and not different),
        portable_missing=missing, portable_different=different,
        campaign_samples=checkpoint.get("campaign_samples"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-step", type=int, default=2)
    p.add_argument("--initial-weights")
    p.add_argument("--reference-initial")
    p.add_argument("--reference-prefix", action="append")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument("--gradient-cases", type=int, default=2)
    args = p.parse_args()
    root, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "config.json"
    cfg = json.loads(cfg_path.read_text())
    checkpoint_path = root / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["global_step"])
    weights_path = root / f"model-step{step:07d}" / "weights.pt"
    portable = torch.load(weights_path, map_location="cpu", weights_only=True)
    report = dict(run=str(root), config_sha256=file_hash(cfg_path),
        checkpoint_sha256=file_hash(checkpoint_path), weights_sha256=file_hash(weights_path),
        checkpoint=checkpoint_report(checkpoint, portable, cfg, args.expected_step))
    optimizer_parameter_ids = [pid for state in checkpoint.get("optimizer_states", [])
                               for group in state["param_groups"] for pid in group["params"]]
    del checkpoint
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from vforge.ops import OpsConfig, initialize_ops
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    initialize_ops(OpsConfig(backend="sdpa_math" if device.type == "cpu" else "sdpa", allow_fallback=False),
                   device=device, parallel_state=None)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(portable, strict=True)
    expectations = expected_gradients(model, cfg)
    report["gradient_expectations"] = expectations
    trainable_tensors = sum(p.requires_grad for p in model.parameters())
    report["optimizer_parameter_count"] = dict(saved_tensors=len(optimizer_parameter_ids),
        model_trainable_tensors=trainable_tensors,
        unique_saved_ids=len(set(optimizer_parameter_ids)),
        passed=len(optimizer_parameter_ids) == trainable_tensors
            and len(set(optimizer_parameter_ids)) == len(optimizer_parameter_ids))
    initial_path = Path(args.initial_weights) if args.initial_weights else root / "initial-model" / "weights.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    report["initial_weights"] = dict(path=str(initial_path), sha256=file_hash(initial_path))
    frozen = [prefix for prefix, value in expectations.items() if value == "frozen"]
    frozen.extend(cfg.get("audit_expectations", {}).get("initial_unchanged_prefixes", []))
    # Calibration/projection are immutable observed-data inputs in every F arm,
    # even when its chart learns. They are registered buffers, not parameters.
    if cfg.get("model_class") == "carrier":
        frozen.extend(["carrier_projection", "carrier_mean", "carrier_scale"])
    report["frozen_tensors_unchanged"] = (compare_tensors(initial, portable, frozen)
        if frozen else dict(prefixes=[], compared_tensors=0, equal=True, scope="No groups declared frozen"))
    if args.reference_initial:
        reference = torch.load(args.reference_initial, map_location="cpu", weights_only=True)
        prefixes = args.reference_prefix or ["encoder", "pool", "predictor", "action_encoder", "renderer"]
        report["paired_initial_equal"] = compare_tensors(initial, reference, prefixes)
        report["paired_initial_equal"]["reference_sha256"] = file_hash(args.reference_initial)
        del reference
    del initial, portable
    dataset = BenchmarkWindows(cfg["data"], window=model.window, split="validation")
    manifest = make_manifest(dataset, args.gradient_cases, args.seed)
    verify_manifest(dataset, manifest)
    batch = next(iter(DataLoader(Subset(dataset, [x["index"] for x in manifest["windows"]]),
                                  batch_size=args.gradient_cases, num_workers=0)))
    before_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    model.zero_grad(set_to_none=True)
    torch.manual_seed(args.seed)
    loss, metrics = model.loss(move_batch(batch, device), step=step, ramp_steps=cfg.get("ramp_steps", 0))
    report["loss"] = float(loss.detach())
    report["loss_finite"] = finite(loss)
    report["metrics"] = {key: float(value.detach()) if torch.is_tensor(value) else value
                         for key, value in metrics.items()}
    report["metrics_finite"] = finite(report["metrics"])
    loss.backward()
    report["gradients"] = gradient_report(model, expectations)
    after_buffers = {name: value.detach().cpu() for name, value in model.named_buffers()}
    report["eval_buffers_unchanged"] = all(name in after_buffers and torch.equal(value, after_buffers[name])
                                            for name, value in before_buffers.items())
    report["all_parameter_gradients_finite"] = all(finite(p.grad) for p in model.parameters() if p.grad is not None)
    report["gradient_windows"] = manifest
    report["scope"] = ("Reads an existing native short-stage checkpoint; eval-mode FP32 actual objective backward only. "
        "No optimizer or update, no new trainer, no probe labels. Positive group norm means some trainable parameters "
        "in that declared group participate; it does not require every tensor to have a nonzero gradient.")
    checks = list(report["checkpoint"]["checks"].values()) + [
        report["optimizer_parameter_count"]["passed"], report["frozen_tensors_unchanged"]["equal"],
        report["loss_finite"], report["metrics_finite"], report["eval_buffers_unchanged"],
        report["all_parameter_gradients_finite"], *[row["passed"] for row in report["gradients"].values()]]
    if "paired_initial_equal" in report:
        checks.append(report["paired_initial_equal"]["equal"])
    report["passed"] = all(checks)
    (out / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "step": step, "gradients": report["gradients"],
                      "output": str(out / "preflight.json")}, allow_nan=False), flush=True)
    if not report["passed"]:
        raise AssertionError("Preflight failed; inspect the written preflight.json")


if __name__ == "__main__":
    main()
