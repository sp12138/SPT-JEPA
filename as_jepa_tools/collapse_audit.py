"""Evaluate the specified noisy-posterior collapse rule on all target frames."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_tools.data import PushTWindows
from as_jepa_tools.evaluate import load_model, pose_net, select_windows
from vforge.ops import OpsConfig, initialize_ops


def variable_dimension_rate(means,surprise,transition_variance_constant=0.):
    return .5*means[:,0].square()+(surprise-transition_variance_constant).sum(1)


def summarize_errors(posterior, anchor, moving, dimension_rate, ratio_limit=.9):
    posterior, anchor = np.asarray(posterior), np.asarray(anchor)
    moving, dimension_rate = np.asarray(moving, dtype=bool), np.asarray(dimension_rate)
    if not moving.any():
        raise ValueError("No target frames satisfy the anchor movement condition")
    if not all(np.isfinite(x).all() for x in (posterior, anchor, dimension_rate)):
        raise ValueError("Nonfinite collapse evidence")
    denominator = float(anchor[moving].mean())
    if denominator <= 0:
        raise ValueError("Zero anchor baseline error")
    ratio = float(posterior[moving].mean() / denominator)
    active = int((dimension_rate > .1).sum())
    return dict(
        collapsed=bool(ratio >= ratio_limit or active < 2),
        posterior_agent_error=float(posterior[moving].mean()),
        anchor_agent_error=denominator,
        posterior_over_anchor=ratio,
        active_dims=active,
        state_dim=len(dimension_rate),
        eligible_target_frames=int(moving.sum()),
        ratio_limit=ratio_limit,
        active_nats_per_window=.1,
        per_dimension_variable_rate_nats=dimension_rate.tolist(),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    for key in ("checkpoint", "data", "pose", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--cases", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        raise RuntimeError("Completed audit exists; use another output directory")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    initialize_ops(OpsConfig(backend="sdpa", allow_fallback=False),
                   device=torch.device("cuda", 0), parallel_state=None)
    model, pose = load_model(args.checkpoint), pose_net(args.pose)
    if model.mode not in {"L2", "S2"}:
        raise ValueError("The formal collapse rule applies to noisy AS-JEPA states")
    data = PushTWindows(args.data, window=16, gap=8, split="test")
    data.open()
    indices = select_windows(data, args.cases)
    width = model.window
    if width > 16:
        raise ValueError("Audit requires a common window bank at least as long as training")
    collected = {k: [] for k in ("posterior", "anchor", "moving", "rate", "entry", "surprise")}
    interventions, records = [], []
    started = time.time()
    generator = torch.Generator(device="cuda").manual_seed(20261028)
    for batch_id, batch in enumerate(DataLoader(Subset(data, indices), batch_size=args.batch,
                                               num_workers=2, pin_memory=True)):
        batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        past = [int(data.offsets[int(ep)]) + int(start) - model.anchor_gap
                for ep, start in zip(batch["episode"].cpu(), batch["start"].cpu())]
        batch["anchor224"] = torch.from_numpy(np.array(data.p224[past])).permute(0, 3, 1, 2).cuda()
        batch["anchor"] = torch.from_numpy(np.array(data.p64[past])).permute(0, 3, 1, 2).cuda()
        batch["anchor_row"] = torch.tensor(past, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pixels, frames, anchor = model.prepare(batch)
            means = model.encode(dict(pixels=pixels[:, :width]))["emb"].float()
            noise = torch.randn(means.shape, device="cuda", generator=generator)
            rho = getattr(model, "rho", 0.) if getattr(model, "method", "") == "correlated_noise" else 0.
            for t in range(1, width):
                noise[:, t] = rho * noise[:, t - 1] + (1 - rho ** 2) ** .5 * noise[:, t]
            diagonal = getattr(model, "noise_diagonal", torch.ones(means.shape[-1], device="cuda"))
            noisy = means + model.sigma0 * noise * diagonal.sqrt()
            predicted = model.predict(noisy[:, :-1], model.action_encoder(batch["action"][:, :width - 1])).float()
            target_mean = means[:, 1:] + rho * (noisy[:, :-1] - means[:, :-1])
            surprise = (target_mean - predicted).square() / (2 * model.sigma0 ** 2 * diagonal)
            markov_constant = .5 * (-rho ** 2 - np.log1p(-rho ** 2))
            surprise = surprise + markov_constant
            entry = .5 * (means[:, 0].square() + model.sigma0 ** 2 * diagonal - 1
                          - torch.log(model.sigma0 ** 2 * diagonal))
            variable_rate = variable_dimension_rate(means, surprise, markov_constant)
            states = noisy[:, 1:].flatten(0, 1)
            anchors = anchor.repeat_interleave(width - 1, 0)
            generated = torch.cat([
                model.render_states(states[lo:lo + 64], anchors[lo:lo + 64],
                                    steps=args.steps, seed=51000 + batch_id * 100 + lo)
                for lo in range(0, len(states), 64)
            ]).float().clamp(-1, 1)
            estimated = pose(generated).float().reshape(len(means), width - 1, 6)
            baseline = pose(anchor).float()
            own = model.render_states(noisy[:, 1], anchor, steps=args.steps, seed=73000 + batch_id).float()
            zero = model.render_states(noisy[:, 1] * 0, anchor, steps=args.steps, seed=73000 + batch_id).float()
            swap = model.render_states(noisy[:, 1].flip(0), anchor, steps=args.steps, seed=73000 + batch_id).float()
        truth = batch["state"][:, 1:width, :2].float()
        anchor_truth = torch.as_tensor(np.array(data.states[past, :2]), device="cuda")
        moving = (truth - anchor_truth[:, None]).norm(dim=-1) > 10
        posterior_error = (estimated[:, :, :2] * 512 - truth).norm(dim=-1)
        anchor_error = (baseline[:, None, :2] * 512 - truth).norm(dim=-1)
        for key, value in dict(posterior=posterior_error, anchor=anchor_error, moving=moving,
                               rate=variable_rate, entry=entry, surprise=surprise).items():
            collected[key].append(value.cpu().numpy())
        interventions.append(dict(
            zero_mse=float((own - zero).square().mean()),
            permutation_mse=float((own - swap).square().mean()),
        ))
        records.extend(dict(episode=int(ep), start=int(start), anchor_row=row)
                       for ep, start, row in zip(batch["episode"].cpu(), batch["start"].cpu(), past))
        (out / "status.json").write_text(json.dumps(dict(
            state="running", completed=len(records), total=len(indices), updated_at=time.time())))
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    result = summarize_errors(arrays["posterior"], arrays["anchor"], arrays["moving"],
                              arrays["rate"].mean(0))
    result.update(
        checkpoint=str(Path(args.checkpoint).resolve()), cases=len(records),
        target_frames_per_window=width - 1, sampler_steps=args.steps, sigma0=model.sigma0,
        actual_renderer_sampler=("finiteGaussianDiffusion"+str(model.density_steps)+"transitions+8bitGaussianEndpoint" if getattr(model,"method","")=="vdm_nats" else "Euler"+str(args.steps)),
        rate_nats_per_frame=float((arrays["entry"].sum(-1) + arrays["surprise"].sum((1, 2))).mean() / width),
        zero_state_intervention_mse=float(np.mean([x["zero_mse"] for x in interventions])),
        state_permutation_intervention_mse=float(np.mean([x["permutation_mse"] for x in interventions])),
        seconds=time.time() - started,
        protocol="All training-window targets t>=1; own posterior sample; common fixed heldout windows; primary past anchor; agent moved>10px from anchor; collapsed if ratio>=.9 or active_dims<2.",
        active_scope="Per-window variable entry .5*mu0^2 plus summed transition KL per dimension; shared fixed-variance entry and transition constants excluded from activity.",
        forward_precision="BF16 autocast; Gaussian rate and pose errors FP32; highest FP32 matmul",
    )
    if hasattr(model, "frontier_method"):
        result["training_objective_variant"] = model.frontier_method
        result["rate_diagnostic_note"] = "This audit uses the original noisy one-step prior as a reference; surrogate training rate may differ. Posterior reconstruction and interventions evaluate the declared marginal posterior."
    np.savez(out / "per-case.npz", **arrays)
    (out / "cases.json").write_text(json.dumps(records, indent=2))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / ("COLLAPSED.json" if result["collapsed"] else "PASS.json")).write_text(json.dumps(result, indent=2))
    (out / "status.json").write_text(json.dumps(dict(state="finished", updated_at=time.time())))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
