"""Frozen clean-state/forecast audit, independent of the proposed objective.

Uses the existing model factory and BenchmarkWindows. No optimizer, renderer
sampling, physical labels, or new trainer. A shifted observed carry baseline
explicitly receives a fresh real observation at each step; it is not a rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return None
    value = float(numerator / denominator)
    return value if math.isfinite(value) else None


def expected_gradients(model, cfg):
    """Factory-owned defaults with an explicit, recorded config override."""
    spec = getattr(model, "audit_expected_gradients", None)
    spec = spec() if callable(spec) else spec
    if spec is None:
        spec = {"encoder": "active", "pool": "active", "predictor": "active",
                "renderer": "active", "action_encoder": "optional"}
    spec = dict(spec)
    override = cfg.get("audit_expectations", {}).get("gradients", {})
    spec.update(override)
    aliases = {"trainable_active": "active", "trainable_optional": "optional"}
    spec = {key.rstrip("."): aliases.get(value, value) for key, value in spec.items()}
    if not set(spec.values()).issubset({"frozen", "active", "optional"}):
        raise ValueError("Unknown gradient expectation: " + repr(spec))
    return spec


def make_manifest(dataset, count, seed):
    # Identical selection rule to benchmark_information_audit.episode_indices:
    # one randomly located window per distinct held-out episode.
    if count < 2 or count > len(dataset.episodes):
        raise ValueError(f"Need 2..{len(dataset.episodes)} distinct episodes, got {count}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset.episodes), count, replace=False)
    windows = []
    for position in chosen:
        begin = int(dataset.ends[position - 1]) if position else 0
        index = begin + int(rng.integers(dataset.counts[position]))
        ep, start = dataset.locate(index)
        rows = (int(dataset.offsets[ep]) + start
                + np.arange(dataset.window) * dataset.stride).tolist()
        windows.append(dict(index=index, episode=int(dataset.episode_ids[ep]),
                            episode_array_index=ep, start=start, rows=rows))
    if len({row["episode"] for row in windows}) != count:
        raise AssertionError("Episode IDs are not unique")
    return dict(version=1, split="validation", seed=seed, count=count,
                window=dataset.window, frameskip=dataset.stride,
                dataset_card_sha256=file_hash(dataset.root / "dataset_card.json"),
                metadata_sha256={name: file_hash(dataset.root / name)
                    for name in ["offsets.npy", "lengths.npy", "episode_ids.npy",
                                 "validation_episodes.npy", "train_episodes.npy"]},
                windows=windows)


def verify_manifest(dataset, manifest):
    if manifest["split"] != "validation":
        raise ValueError("The diagnostic requires validation windows")
    if manifest["window"] != dataset.window or manifest["frameskip"] != dataset.stride:
        raise ValueError("Window/stride mismatch")
    if manifest["dataset_card_sha256"] != file_hash(dataset.root / "dataset_card.json"):
        raise ValueError("Dataset card mismatch")
    for name, digest in manifest["metadata_sha256"].items():
        if file_hash(dataset.root / name) != digest:
            raise ValueError("Dataset metadata changed: " + name)
    train = set(np.load(dataset.root / "train_episodes.npy").tolist())
    validation = set(np.load(dataset.root / "validation_episodes.npy").tolist())
    if train.intersection(validation):
        raise AssertionError("Training/validation episode overlap")
    seen = set()
    for row in manifest["windows"]:
        ep, start = dataset.locate(row["index"])
        actual_rows = (int(dataset.offsets[ep]) + start
                       + np.arange(dataset.window) * dataset.stride).tolist()
        if (int(dataset.episode_ids[ep]) != row["episode"] or start != row["start"]
                or actual_rows != row["rows"] or ep != row["episode_array_index"]):
            raise AssertionError("Window does not match authoritative metadata")
        if row["episode"] not in validation or row["episode"] in train or row["episode"] in seen:
            raise AssertionError("Window is not a unique held-out episode")
        seen.add(row["episode"])
    if len(seen) != manifest["count"]:
        raise AssertionError("Manifest count mismatch")


def prefix_prediction(model, observed, embedded_actions):
    """One-step predictions, each allowed only its actual observed prefix."""
    horizon = observed.shape[1] - 1
    values = []
    for t in range(horizon):
        result = model.predict(observed[:, :t + 1], embedded_actions[:, :t + 1])
        if result.shape != observed[:, :t + 1].shape:
            raise AssertionError("predict must return one state per input position")
        values.append(result[:, -1])
    return torch.stack(values, 1)


def clean_rollout(model, initial, embedded_actions, horizon):
    """Returns future only: output[:,0] predicts observed frame 1, never frame 0."""
    if embedded_actions.shape[1] < horizon:
        raise ValueError("Not enough action blocks")
    history = [initial]
    for t in range(horizon):
        prefix = torch.stack(history, 1)
        result = model.predict(prefix, embedded_actions[:, :t + 1])
        if result.shape != prefix.shape:
            raise AssertionError("predict shape disagrees with its causal state prefix")
        history.append(result[:, -1])
    return torch.stack(history[1:], 1)


def intervention_actions(actions):
    if actions.ndim != 3 or actions.shape[1] < 2 or actions.shape[0] < 2:
        raise ValueError("Expected B>=2, W>=2, A block tensor")
    order = actions.clone()
    order[:, :-1] = actions[:, :-1].roll(1, 1)
    other = actions.clone()
    other[:, :-1] = actions.roll(1, 0)[:, :-1]
    # The final Dataset action is padding. Never shift it into a valid step.
    assert torch.equal(order[:, -1], actions[:, -1])
    assert torch.equal(other[:, -1], actions[:, -1])
    return {"wrong_action_order": order, "other_episode_actions": other}


def squared_error(prediction, target):
    if prediction.shape != target.shape:
        raise AssertionError(f"State alignment mismatch: {prediction.shape}, {target.shape}")
    return (prediction.double() - target.double()).square().mean(-1)


def summarize_condition(error, initial_variation, local_variation, effect=None):
    # Ratios of summed errors to summed true variation, not a mean of unstable
    # per-window ratios. Exact zero denominators remain undefined, never floored.
    error, initial_variation, local_variation = [x.cpu() for x in
                                               (error, initial_variation, local_variation)]
    raw = float(error.mean())
    initial = float(initial_variation.mean())
    local = float(local_variation.mean())
    out = dict(mse=raw, normalized_by_initial_variation=safe_ratio(raw, initial),
               normalized_by_local_variation=safe_ratio(raw, local),
               per_horizon_mse=error.mean(0).tolist(),
               per_horizon_normalized_by_initial_variation=[
                   safe_ratio(float(a), float(b)) for a, b in
                   zip(error.mean(0), initial_variation.mean(0))],
               per_window_mse=error.mean(1).tolist())
    if effect is not None:
        value = float(effect.mean())
        out.update(intervention_effect_mse=value,
                   intervention_effect_over_initial_variation=safe_ratio(value, initial),
                   per_horizon_intervention_effect_mse=effect.mean(0).cpu().tolist())
    return out


@torch.no_grad()
def evaluate_latents(model, observed, actions, batch_size=8):
    """Inputs follow the manifest order; other-initial mapping is global +1."""
    if observed.ndim != 3 or observed.shape[:2] != actions.shape[:2]:
        raise ValueError("Expected aligned B,W,D and B,W,A tensors")
    n, window, _ = observed.shape
    if n < 2:
        raise ValueError("Other-episode intervention requires distinct episodes")
    target = observed[:, 1:]
    initial_variation = squared_error(target, observed[:, :1].expand_as(target))
    local_variation = squared_error(target, observed[:, :-1])
    other_initial = observed.roll(1, 0)[:, 0]
    alternate_actions = intervention_actions(actions)
    outputs = {key: [] for key in ["clean_rollout", "teacher_forced_prefix",
        "other_initial", "zero_initial", *alternate_actions]}
    causal_errors = []
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        z, a = observed[start:stop], actions[start:stop]
        act = model.action_encoder(a)
        horizon = window - 1
        reference = clean_rollout(model, z[:, 0], act, horizon)
        teacher = prefix_prediction(model, z, act)
        outputs["clean_rollout"].append(reference)
        outputs["teacher_forced_prefix"].append(teacher)
        outputs["other_initial"].append(clean_rollout(model, other_initial[start:stop], act, horizon))
        outputs["zero_initial"].append(clean_rollout(model, torch.zeros_like(z[:, 0]), act, horizon))
        for name, alternative in alternate_actions.items():
            outputs[name].append(clean_rollout(model, z[:, 0],
                model.action_encoder(alternative[start:stop]), horizon))
        # Compare to the common vectorized teacher call to disclose any future
        # state leakage. Prefix-only results above are authoritative regardless.
        vectorized = model.predict(z[:, :-1], act[:, :-1])
        causal_errors.append(squared_error(vectorized, teacher))
    states = {key: torch.cat(value) for key, value in outputs.items()}
    states.update(observed_future=target,
                  initial_carry=observed[:, :1].expand_as(target),
                  shifted_observed_carry=observed[:, :-1],
                  zero_prediction=torch.zeros_like(target))
    summaries = {}
    for name, prediction in states.items():
        effect = squared_error(prediction, states["clean_rollout"]) if name in {
            "other_initial", "zero_initial", *alternate_actions} else None
        summaries[name] = summarize_condition(squared_error(prediction, target),
                                             initial_variation, local_variation, effect)
    for name in ["other_initial", "zero_initial", *alternate_actions]:
        summaries[name]["mse_change_vs_clean_rollout"] = summaries[name]["mse"] - summaries["clean_rollout"]["mse"]
    variation = dict(initial=initial_variation.mean(0).tolist(),
                     local=local_variation.mean(0).tolist(),
                     initial_mean=float(initial_variation.mean()),
                     local_mean=float(local_variation.mean()),
                     exact_zero_initial_windows=int((initial_variation.sum(1) == 0).sum()),
                     exact_zero_local_windows=int((local_variation.sum(1) == 0).sum()))
    action_changes = {name: float((value[:, :-1].double() - actions[:, :-1].double()).square().mean())
                      for name, value in alternate_actions.items()}
    causal = torch.cat(causal_errors)
    return dict(conditions=summaries, actual_temporal_variation=variation,
                action_intervention_input_mse=action_changes,
                vectorized_teacher_vs_prefix_mse=float(causal.mean()),
                vectorized_teacher_vs_prefix_max_mse=float(causal.max()),
                observed_state_std_mean=float(observed.double().std((0, 1)).mean()),
                observed_state_rms=float(observed.double().square().mean().sqrt()),
                other_initial_case_indices=np.roll(np.arange(n), 1).tolist()), states


@torch.no_grad()
def path_query_diagnostic(model, observed, future_states, render, query_step=0, batch_size=4):
    """Use the model-owned head assembly with one fixed set of real queries."""
    from torch.nn import functional as F
    targets = model.path_query_targets(render, step=query_step)
    n, window, _ = observed.shape
    complete = {}
    for name, future in future_states.items():
        first = observed[:, 0]
        if name == "other_initial":
            first = observed.roll(1, 0)[:, 0]
        elif name in {"zero_initial", "zero_prediction"}:
            first = torch.zeros_like(first)
        complete[name] = torch.cat([first[:, None], future], 1)
    complete.update(other_episode_observed=observed.roll(1, 0),
                    reversed_observed=observed.flip(1))
    batch_target_keys = {"features", "prototypes", "symbols", "transforms", "completion"}
    values = {}
    for name, state in complete.items():
        logits = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            local = {key: value[start:stop] if key in batch_target_keys else value
                     for key, value in targets.items()}
            logits.append(model.path_query_logits_from_states(state[start:stop], local).float())
        logits = torch.cat(logits)
        selections = {
            "symbol": (logits[:, :, :window, :3], targets["symbols"]),
            "symbol_future_only": (logits[:, :, 1:window, :3], targets["symbols"][:, :, 1:]),
            "completion": (logits[:, :, window:, :2], targets["completion"]),
            "transform": (logits[:, :, window:, :5], targets["transforms"]),
        }
        row = {}
        for kind, (prediction, target) in selections.items():
            count = prediction.shape[-1]
            ce = F.cross_entropy(prediction.reshape(-1, count), target.reshape(-1),
                                 reduction="none").reshape_as(target)
            accuracy = (prediction.argmax(-1) == target).float()
            row[kind] = dict(ce=float(ce.mean()), relative_to_uniform_ce=float(ce.mean()) / math.log(count),
                accuracy=float(accuracy.mean()), per_window_ce=ce.mean((1, 2)).cpu().tolist(),
                per_position_ce=ce.mean((0, 1)).cpu().tolist())
        values[name] = row
    labels = {}
    for name, count in [("symbols", 3), ("completion", 2), ("transforms", 5)]:
        counts = torch.bincount(targets[name].flatten(), minlength=count)
        labels[name] = dict(counts=counts.cpu().tolist(),
                            observed_majority_fraction=float(counts.max() / counts.sum()))
    return dict(trained_mode=model.path_query_mode, query_step=query_step,
        query_indices=targets["query_indices"].cpu().tolist(), label_support=labels,
        conditions=values,
        scope="Same model-owned query prototypes, masks and targets under every intervention. zero_prediction is a prototype/mask-only shortcut control. observed/reversed/other_episode branches access real future encodings; shifted_observed_carry gets a new real observation each step. Only the declared trained mode is an objective-matched score; other heads are descriptive. This is not a task-success metric.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cases", type=int, default=64)
    p.add_argument("--seed", type=int, default=1042)
    p.add_argument("--manifest")
    p.add_argument("--encode-batch-size", type=int, default=4)
    p.add_argument("--predict-batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--step", type=int)
    p.add_argument("--query-step", type=int, default=0)
    p.add_argument("--manifest-only", action="store_true")
    args = p.parse_args()
    root, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    dataset = BenchmarkWindows(cfg["data"], window=cfg["model"]["window"], split="validation")
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else make_manifest(dataset, args.cases, args.seed)
    verify_manifest(dataset, manifest)
    if len(manifest["windows"]) != args.cases:
        raise ValueError("--cases must equal the supplied manifest count")
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (out / "windows.json").write_bytes(manifest_bytes)
    if args.manifest_only:
        print(json.dumps({"manifest": str(out / "windows.json"), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}))
        return
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    from vforge.ops import OpsConfig, initialize_ops
    from as_jepa_lewm.factory import build_model
    device = torch.device(args.device)
    initialize_ops(OpsConfig(backend="sdpa_math" if device.type == "cpu" else "sdpa", allow_fallback=False),
                   device=device, parallel_state=None)
    if args.step is None:
        checkpoint = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
        step = int(checkpoint["global_step"])
        del checkpoint
    else:
        step = args.step
    weights = root / f"model-step{step:07d}" / "weights.pt"
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    if model.training:
        raise AssertionError("Frozen diagnostic must run in evaluation mode")
    started = time.time()
    observed, actions, query_renders = [], [], []
    has_path_query = getattr(model, "path_query_mode", "none") != "none"
    if has_path_query and not hasattr(model, "path_query_logits_from_states"):
        raise RuntimeError("Path-query audit requires the model-owned logits-from-states hook")
    loader = DataLoader(Subset(dataset, [r["index"] for r in manifest["windows"]]),
                        batch_size=args.encode_batch_size, num_workers=0, shuffle=False)
    offset = 0
    with torch.no_grad():
        for raw in loader:
            for i in range(len(raw["episode"])):
                expected = manifest["windows"][offset + i]
                if (int(raw["episode"][i]) != expected["episode"]
                        or raw["rows"][i].tolist() != expected["rows"]):
                    raise AssertionError("Loaded bytes do not follow the manifest")
            offset += len(raw["episode"])
            batch = move_batch(raw, device)
            pixels, render, _ = model.prepare(batch)
            observed.append(model.encode(dict(pixels=pixels))["emb"].float())
            actions.append(batch["action"].float())
            if has_path_query:
                query_renders.append(render.float())
        observed, actions = torch.cat(observed), torch.cat(actions)
        results, states = evaluate_latents(model, observed, actions, args.predict_batch_size)
        if has_path_query:
            results["path_queries"] = path_query_diagnostic(model, observed, states,
                torch.cat(query_renders), args.query_step, args.encode_batch_size)
    report = dict(run=str(root), step=step, model_class=cfg.get("model_class", "legacy"),
        config_sha256=file_hash(root / "config.json"), weights_sha256=file_hash(weights),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), cases=len(observed),
        window=observed.shape[1], state_dim=observed.shape[2], elapsed_seconds=time.time()-started,
        gradient_expectations=expected_gradients(model, cfg), **results,
        access_contract={
            "observed_future": "Actual future encoding: tautological zero-error reference, not prediction.",
            "clean_rollout": "Only real frame 0 plus executed actions; all subsequent states predicted.",
            "teacher_forced_prefix": "Each step receives all real past frames through that step; no future frames.",
            "shifted_observed_carry": "Copies the latest real encoding at every step; extra observations, no learned dynamics.",
            "initial_carry": "Only initial encoding, repeated over the horizon; no dynamics.",
            "interventions": "Same clean target; global different-episode initial, zero initial, valid-step action-order shift and other-episode actions.",
        }, scope="Frozen clean planner-path diagnostics. No posterior noise, labels, update or task-success claim. Latent MSE scales are model-dependent; compare normalized quantities together with their raw denominators. Existing benchmark_information_audit provides separate probes.")
    chart = getattr(model, "carrier_chart_module", None)
    if chart is not None and hasattr(chart, "norm_certificate"):
        with torch.no_grad():
            certificate = dict(chart.norm_certificate())
            raw_input = float(torch.linalg.matrix_norm(chart.input.weight.detach().double(), ord=2))
            raw_output = float(torch.linalg.matrix_norm(chart.output.weight.detach().double(), ord=2))
        certificate.update(raw_input_operator_norm=raw_input, raw_output_operator_norm=raw_output,
            input_normalization_active=certificate["input_operator_norm"] < raw_input - 1e-7,
            output_normalization_active=certificate["output_operator_norm"] < raw_output - 1e-7)
        report["carrier_chart_certificate"] = certificate
    (out / "state-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    # Small arrays permit later same-case checks without re-encoding images.
    np.savez_compressed(out / "state-audit.npz", observed=observed.cpu().numpy(),
                        actions=actions.cpu().numpy(),
                        **{name: value.cpu().numpy() for name, value in states.items()})
    print(json.dumps({"output": str(out), "step": step, "cases": len(observed),
        "manifest_sha256": report["manifest_sha256"],
        "conditions": {key: {name: value for name, value in row.items() if not name.startswith("per_")}
                       for key, row in results["conditions"].items()}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
