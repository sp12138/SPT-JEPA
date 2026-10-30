"""Frozen-model information audit; oracle labels are used only by offline probes.

No optimizer step or model mutation is performed. State permutations and batch
centering below are diagnostic interventions, not proposed training objectives.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.model import ASJepaLeWM
from as_jepa_lewm.robust import RobustLeWM
from as_jepa_tools.benchmark_data import BenchmarkWindows
from vforge.models.as_jepa.pixels import pack, unpack
from vforge.ops import OpsConfig, initialize_ops


def device_batch(batch):
    return {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}


def posterior_noise(mu, rho):
    eps = torch.randn_like(mu)
    for t in range(1, mu.shape[1]):
        eps[:, t] = rho * eps[:, t - 1] + math.sqrt(1 - rho * rho) * eps[:, t]
    return eps


def forecast(model, initial, actions):
    embedded = model.action_encoder(actions)
    history = [initial]
    for t in range(actions.shape[1] - 1):
        history.append(model.predict(torch.stack(history, 1), embedded[:, :t + 1])[:, -1])
    return torch.stack(history[1:], 1)


def region_stats(error, moving):
    """Return means and energies, with masks shared by all interventions."""
    per_pixel = error.square().mean(1)
    moving = moving.to(per_pixel.dtype)
    static = 1 - moving
    total = per_pixel.numel()
    moving_count = moving.sum()
    static_count = static.sum()
    return {
        "all_mse": float(per_pixel.mean()),
        "motion_mse": float((per_pixel * moving).sum() / moving_count.clamp_min(1)),
        "static_mse": float((per_pixel * static).sum() / static_count.clamp_min(1)),
        "motion_fraction": float(moving_count / total),
        "motion_error_fraction": float((per_pixel * moving).sum() / per_pixel.sum().clamp_min(1e-30)),
    }


def gradient_stats(flow, rate, mu):
    f, r, z = flow.float(), rate.float(), mu.detach().float()
    centered = z - z.mean((0, 1), keepdim=True)
    temporal = z - z[:, :1]
    fn, rn = f.norm(), r.norm()
    return {
        "flow_norm": float(fn), "priced_rate_norm": float(rn),
        "rate_to_flow": float(rn / fn.clamp_min(1e-30)),
        "cosine": float((f * r).sum() / (fn * rn).clamp_min(1e-30)),
        # Positive dL/dlog(scale) means reducing signal scale locally lowers L.
        "flow_centered_scale_derivative": float((f * centered).sum()),
        "rate_centered_scale_derivative": float((r * centered).sum()),
        "flow_temporal_scale_derivative": float((f * temporal).sum()),
        "rate_temporal_scale_derivative": float((r * temporal).sum()),
        "state_std": float(z.std((0, 1)).mean()),
    }


def objective_gradients(model, batch, step, ramp_steps, seed):
    captured = []
    original = model.encode

    def capture(info):
        result = original(info)
        captured.append(result["emb"])
        return result

    model.encode = capture
    # Evaluation mode isolates the frozen objective from dropout randomness.
    torch.manual_seed(seed)
    try:
        loss, metrics = model.loss(batch, step=step, ramp_steps=ramp_steps)
        mu = captured[0]
        beta = model.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else model.beta
        f = torch.autograd.grad(metrics["fm"], mu, retain_graph=True)[0]
        r = torch.autograd.grad(beta * metrics["rate"], mu)[0]
        return dict(gradient_stats(f, r, mu), fm=float(metrics["fm"].detach()),
                    rate=float(metrics["rate"].detach()), beta=beta,
                    scope="Original loss, evaluation mode, FP32, no update")
    finally:
        model.encode = original


@torch.no_grad()
def intervention_batch(model, batch, seed):
    torch.manual_seed(seed)
    pixels, render, anchor = model.prepare(batch)
    mu = model.encode(dict(pixels=pixels))["emb"]
    b, w, d = mu.shape
    noise = posterior_noise(mu, getattr(model, "rho", 0.))
    sampled = mu + model.sigma0 * noise
    actions = batch["action"]
    wrong_actions = actions.clone()
    wrong_actions[:, :-1] = actions[:, :-1].roll(1, 1)
    pred = forecast(model, sampled[:, 0], actions)
    conditions = {
        "posterior": sampled[:, 1:],
        "posterior_other_episode": mu.roll(1, 0)[:, 1:] + model.sigma0 * noise[:, 1:],
        "posterior_time_reversed": mu.flip(1)[:, 1:] + model.sigma0 * noise[:, 1:],
        "forecast": pred,
        "forecast_other_initial": forecast(model, mu.roll(1, 0)[:, 0] + model.sigma0 * noise[:, 0], actions),
        "forecast_wrong_actions": forecast(model, sampled[:, 0], wrong_actions),
    }
    center = mu.mean((0, 1), keepdim=True)
    # Fixed renderer tests whether tiny encoder signals are usable at another
    # scale. These are off-support interventions, not an improved model claim.
    for scale in [0., 1., 10., 100.]:
        signal = center + scale * (mu - center)
        for sigma in [0., model.sigma0]:
            conditions[f"scaled_posterior_scale{scale:g}_sigma{sigma:g}"] = (signal + sigma * noise)[:, 1:]
    y = pack(render[:, 1:].flatten(0, 1))
    eps = torch.randn_like(y)
    true_anchor = pack(anchor).repeat_interleave(w - 1, 0)
    wrong_anchor = pack(anchor.roll(1, 0)).repeat_interleave(w - 1, 0)
    # Motion support is obtained solely from observed frames. Threshold is two
    # uint8 levels; both erased and newly occupied locations are represented.
    temporal_range = render.amax(1) - render.amin(1)
    mask = (temporal_range.amax(1) > 2 / 127.5)[:, None].expand(-1, w - 1, -1, -1).flatten(0, 1)
    rows = []
    for level in [.2, .5, .8, .95, 1.]:
        t = torch.full((len(y), 1), level, device=y.device)
        for name, z in conditions.items():
            # Scale curves only at the decision-relevant high-corruption end.
            if name.startswith("scaled_") and level not in [.95, 1.]:
                continue
            z = z.reshape(-1, model.state_slots, model.state_width)
            with model.renderer.anchored(true_anchor):
                velocity = model.renderer((1 - level) * y + level * eps, t, condition=z)
            rows.append(dict(condition=name, t=level,
                             **region_stats(unpack(y - eps - velocity), mask)))
        for name, anc in [("forecast_wrong_anchor", wrong_anchor), ("forecast_zero_anchor", torch.zeros_like(true_anchor))]:
            with model.renderer.anchored(anc):
                velocity = model.renderer((1 - level) * y + level * eps, t,
                                          condition=pred.reshape(-1, model.state_slots, model.state_width))
            rows.append(dict(condition=name, t=level,
                             **region_stats(unpack(y - eps - velocity), mask)))
    return rows, region_stats(render[:, 1:].flatten(0, 1) - render[:, :1].expand(-1, w - 1, -1, -1, -1).flatten(0, 1), mask)


def episode_indices(dataset, count, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(dataset.episodes), min(count, len(dataset.episodes)), replace=False)
    indices = []
    for i in chosen:
        begin = int(dataset.ends[i - 1]) if i else 0
        indices.append(begin + int(rng.integers(dataset.counts[i])))
    return indices


@torch.no_grad()
def probe_features(model, cfg, split, count, seed):
    ds = BenchmarkWindows(cfg["data"], window=model.window, split=split)
    loader = DataLoader(Subset(ds, episode_indices(ds, count, seed)), batch_size=4, num_workers=0)
    xs, ys, episodes = [], [], []
    label_keys = None
    for batch in loader:
        ds.open()
        if label_keys is None:
            label_keys = [key for key in ["pos_agent", "pos_target", "privileged_block_0_pos", "proprio_effector_pos"] if key in ds.file]
            if not label_keys:
                raise ValueError("No supported diagnostic physical labels")
        rows = batch["rows"].numpy()
        labels = np.concatenate([np.stack([np.asarray(ds.file[key][row]) for row in rows]) for key in label_keys], -1)
        prepared = device_batch(batch)
        pixels, _, _ = model.prepare(prepared)
        xs.append(model.encode(dict(pixels=pixels))["emb"].float().cpu().numpy())
        ys.append(labels)
        episodes.extend(batch["episode"].tolist())
    return np.concatenate(xs), np.concatenate(ys), episodes, label_keys


def ridge_probe(train_x, train_y, test_x, test_y):
    x, y = train_x.reshape(-1, train_x.shape[-1]).astype(np.float64), train_y.reshape(-1, train_y.shape[-1]).astype(np.float64)
    tx, ty = test_x.reshape(-1, test_x.shape[-1]).astype(np.float64), test_y.reshape(-1, test_y.shape[-1]).astype(np.float64)
    mean, std = x.mean(0), x.std(0)
    std = np.maximum(std, 1e-12)
    x, tx = (x - mean) / std, (tx - mean) / std
    ym = y.mean(0)
    weight = np.linalg.solve(x.T @ x / len(x) + .001 * np.eye(x.shape[1]), x.T @ (y - ym) / len(x))
    err = ((tx @ weight + ym - ty) ** 2).mean(0)
    reference = ((ty - ym) ** 2).mean(0)
    valid = reference > 1e-12
    r2 = 1 - err / np.maximum(reference, 1e-20)
    return dict(mse=err.tolist(), r2_against_train_mean=[float(v) if ok else None for v, ok in zip(r2, valid)],
                mean_r2=float(r2[valid].mean()) if valid.any() else None,
                informative_target_dimensions=int(valid.sum()),
                train_std_mean=float(std.mean()), ridge=.001)


def run_probes(model, cfg, count, seed):
    x, y, train_ep, keys = probe_features(model, cfg, "train", count, seed)
    tx, ty, test_ep, test_keys = probe_features(model, cfg, "validation", count, seed + 1)
    assert keys == test_keys and not set(train_ep).intersection(test_ep)
    rng = np.random.default_rng(seed + 2)
    # Deterministic cyclic shifts preserve each episode's feature distribution.
    shifted = np.roll(x, 3, axis=1)
    test_shifted = np.roll(tx, 3, axis=1)
    out = {"clean": ridge_probe(x, y, tx, ty),
           "time_shifted": ridge_probe(shifted, y, test_shifted, ty),
           "initial_carry": ridge_probe(np.repeat(x[:, :1], x.shape[1], 1), y, np.repeat(tx[:, :1], tx.shape[1], 1), ty)}
    for sigma in [model.sigma0]:
        out[f"noisy_sigma{sigma:g}"] = ridge_probe(x + sigma * rng.standard_normal(x.shape), y,
                                                  tx + sigma * rng.standard_normal(tx.shape), ty)
    # Within-trajectory displacement removes static episode/layout information.
    out["temporal_displacement"] = ridge_probe(x[:, 1:] - x[:, :1], y[:, 1:] - y[:, :1],
                                               tx[:, 1:] - tx[:, :1], ty[:, 1:] - ty[:, :1])
    return dict(label_keys=keys, train_episodes=train_ep, validation_episodes=test_ep,
                scores=out, scope="Diagnostic privileged labels only; frozen encoder; fixed ridge; training-only standardization")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cases", type=int, default=16)
    p.add_argument("--probe-episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=1042)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    root, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "config.json").read_text())
    checkpoint = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
    step = int(checkpoint["global_step"])
    del checkpoint
    weights_path = root / f"model-step{step:07d}" / "weights.pt"
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False), device=torch.device("cuda"), parallel_state=None)
    model = (RobustLeWM if cfg.get("robust_variant") else ASJepaLeWM)(**cfg["model"]).cuda().eval()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    if model.mode != "L2":
        raise ValueError("This audit currently requires the AS-JEPA rate/flow model")
    started = time.time()
    ds = BenchmarkWindows(cfg["data"], window=model.window, split="validation")
    loader = DataLoader(Subset(ds, episode_indices(ds, args.cases, args.seed)), batch_size=2, num_workers=0, drop_last=True)
    batches = []
    for i, raw in enumerate(loader):
        batch = device_batch(raw)
        gradient = objective_gradients(model, batch, step, cfg.get("ramp_steps", 0), args.seed + i)
        interventions, motion = intervention_batch(model, batch, args.seed + i)
        batches.append(dict(episodes=raw["episode"].tolist(), starts=raw["start"].tolist(),
                            gradient=gradient, anchor_copy=motion, interventions=interventions))
        (out / "interventions.partial.json").write_text(json.dumps(batches, indent=2) + "\n")
    probes = run_probes(model, cfg, args.probe_episodes, args.seed)
    report = dict(run=str(root), weights=str(weights_path), step=step, config=cfg["model"], batches=batches,
                  probes=probes, seconds=time.time() - started,
                  scope="Frozen diagnostics only. Small amplitude is not proof of absent information. Off-support scale/anchor interventions are not policy improvements.")
    (out / "information-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(run=str(root), step=step, seconds=report["seconds"], probe_scores=probes["scores"])), flush=True)


if __name__ == "__main__":
    main()
