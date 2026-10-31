"""Does the prior contract or slow down its rollouts? Rollout diagnostics on clean encoder states (round-2 spec, Section 5.3).

Mechanism (spec Appendix A.8). With independent posterior noise, the rate charges the prior for carrying
its noisy input forward, so for a state component whose spread is r times the code noise the rate-optimal
one-step carry is a = r^2 / (1 + r^2) < 1: the prior pulls the state toward a centre, as a regression on a
noisy input does. At planning time the prior is fed clean encoder means but keeps the pull, so open-loop
rollouts contract toward the centre and forget where they started.

This tool measures that on held-out windows, without the renderer:
- the planner's open-loop rollout (`predict_states`: `--context` frames, noiseless means), at every horizon:
  spread ratio, carry ratio, motion gain and error relative to copying, overall and within quartiles of
  dimensions ranked by spread (the pull is strongest where the spread is smallest);
- the same metrics for teacher-forced one-step predictions from clean encoder means;
- for L2 models, the one-step gain on posterior samples drawn with the model's own noise (sigma0, and
  AR(1) when rho > 0): the calibration slope of the training condition, against which SLOPE_MIN applies;
- the state's spread over sigma0 and the noisy-state information bound, sum of 0.5 ln(1 + Var mu / sigma0^2).

How to read the metrics (spec Section 5.3): spread ratio and carry ratio below 1 mean contraction; a motion
gain below 1 means predicted motion lags the true motion, above 1 that it is exaggerated.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from as_jepa_lewm.forecast_scored import ar1


def _slope(y, x):
    """Least-squares slope of y on x through the origin, pooled over every element."""
    x, y = x.double(), y.double()
    return float((x * y).sum() / x.square().sum().clamp_min(1e-30))


def displacement_gain(target, predicted, reference):
    """Slope of (target - reference) on (predicted - reference): revision 2's calibration slope. It is 1 for a
    prior whose mean is the conditional expectation of its training inputs; below 1 the prior exaggerates."""
    return _slope(target - reference, predicted - reference)


def _centred_summary(truth, predicted, last_centred):
    """Carry of the rollout and of the truth on the centred last context state, and the spread ratio."""
    tc, pc = truth - truth.mean(0), predicted - predicted.mean(0)
    den = last_centred.square().sum().clamp_min(1e-30)
    carry, truth_carry = float((pc * last_centred).sum() / den), float((tc * last_centred).sum() / den)
    spread = float(pc.square().sum() / tc.square().sum().clamp_min(1e-30))
    return carry, truth_carry, spread


def _ratio(a, b):
    return a / b if abs(b) > 1e-12 else float("nan")


def rollout_metrics(truth, predicted, last):
    """truth, predicted: [B, H, D] states at horizons 1..H; last: [B, D], the last context state.

    motion_gain: slope of the predicted displacement on the true displacement, both from `last`. A pull
      toward a centre that is unrelated to the motion does not bias it. 1 = motion at the right size.
    carry, truth_carry, carry_ratio: slope of the centred state at horizon k on the centred `last`, for the
      rollout and for the encoded truth, and their ratio. Below 1 = the rollout forgets where it started.
    spread_ratio: across-window variance of the rollout over that of the truth. Below 1 = contraction.
    error_vs_copy: squared error of the rollout over that of copying `last`.
    by_spread_quartile: spread and carry ratios within quartiles of dimensions ranked by the spread of
      `last` (q1 = lowest spread). dimension_std: that spread, per dimension, in state units.
    """
    truth, predicted, last = truth.double(), predicted.double(), last.double()
    centred = last - last.mean(0)
    std = last.std(0, unbiased=False)
    quartiles = torch.argsort(std).chunk(4)
    out = dict(motion_gain=[], carry=[], truth_carry=[], carry_ratio=[], spread_ratio=[], error_vs_copy=[],
               by_spread_quartile={f"q{i + 1}": dict(spread_ratio=[], carry_ratio=[]) for i in range(len(quartiles))},
               dimension_std=std.tolist())
    for k in range(truth.shape[1]):
        t, p = truth[:, k], predicted[:, k]
        out["motion_gain"].append(_slope(p - last, t - last))
        carry, truth_carry, spread = _centred_summary(t, p, centred)
        out["carry"].append(carry); out["truth_carry"].append(truth_carry)
        out["carry_ratio"].append(_ratio(carry, truth_carry)); out["spread_ratio"].append(spread)
        out["error_vs_copy"].append(float((p - t).square().sum() / (last - t).square().sum().clamp_min(1e-30)))
        for i, dims in enumerate(quartiles):
            c, tc, s = _centred_summary(t[:, dims], p[:, dims], centred[:, dims])
            q = out["by_spread_quartile"][f"q{i + 1}"]
            q["spread_ratio"].append(s); q["carry_ratio"].append(_ratio(c, tc))
    return out


@torch.no_grad()
def one_step_sums(model, mu, act, noise=None):
    """Numerator and denominator of the teacher-forced one-step gain, so that batches pool exactly.
    Inputs are encoder means mu [B, W, D] (noise=None) or posterior samples mu + noise; with AR(1) noise the
    target is the posterior's conditional mean mu_t + rho * noise_{t-1}."""
    w = min(mu.shape[1], model.window)
    inputs = mu[:, :w - 1] if noise is None else (mu + noise)[:, :w - 1]
    predicted = model.predict(inputs, act[:, :w - 1])
    target = mu[:, 1:w]
    rho = getattr(model, "rho", 0.)
    if noise is not None and rho:
        target = target + rho * noise[:, :w - 1]
    d_true, d_pred = (target - inputs).double(), (predicted - inputs).double()
    return float((d_true * d_pred).sum()), float(d_pred.square().sum())


def one_step_gain(model, mu, act, noise=None):
    """Teacher-forced one-step gain (slope of the true change on the predicted change) for one batch."""
    num, den = one_step_sums(model, mu, act, noise)
    return num / max(den, 1e-30)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="PushT evaluation data (phase0) or a benchmark metadata directory")
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=["pusht", "benchmark"], default="pusht")
    p.add_argument("--cases", type=int, default=128)
    p.add_argument("--window", type=int, default=16, help="frames per evaluation window")
    p.add_argument("--context", type=int, default=3, help="context frames of the open-loop rollout, as the planner")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from as_jepa_tools.evaluate import load_model, predict_states, select_windows
    from vforge.ops import OpsConfig, initialize_ops
    device = torch.device(a.device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest" if a.fp32 else "high")
    initialize_ops(OpsConfig(backend="sdpa" if device.type == "cuda" else "sdpa_math", allow_fallback=False),
                   device=device, parallel_state=None)
    model = load_model(a.checkpoint, device=a.device)
    if a.dataset == "pusht":
        from as_jepa_tools.data import PushTWindows
        data = PushTWindows(a.data, window=a.window, gap=8, split="test")
        indices = select_windows(data, a.cases)                      # the same cases as decode_decomposition
    else:
        from as_jepa_tools.benchmark_data import BenchmarkWindows
        data = BenchmarkWindows(a.data, window=a.window, split="validation")
        indices = np.random.default_rng(0).permutation(len(data))[:a.cases].tolist()
    loader = DataLoader(Subset(data, indices), batch_size=8, num_workers=0)
    horizon = a.window - a.context
    w1 = min(a.window, model.window)
    noisy = model.mode in {"L2", "S2"}
    generator = torch.Generator(device=device).manual_seed(20261031)
    kept = dict(mu=[], predicted=[], one_step=[])
    noisy_num = noisy_den = 0.
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and not a.fp32):
            pixels, _, _ = model.prepare(batch)
            mu = model.encode(dict(pixels=pixels))["emb"].float()
            act = model.action_encoder(batch["action"]).float()
            predicted = predict_states(model, pixels[:, :a.context], batch["action"], horizon).float()
        with torch.autocast(device.type, enabled=False):
            kept["one_step"].append(model.predict(mu[:, :w1 - 1], act[:, :w1 - 1]).float().cpu())
            if noisy:
                noise = model.sigma0 * ar1(torch.randn(mu.shape, device=device, generator=generator), getattr(model, "rho", 0.))
                num, den = one_step_sums(model, mu, act, noise)
                noisy_num += num; noisy_den += den
        kept["mu"].append(mu.cpu()); kept["predicted"].append(predicted.cpu())
    mu, predicted, one_step = torch.cat(kept["mu"]), torch.cat(kept["predicted"]), torch.cat(kept["one_step"])
    d = mu.shape[-1]
    report = dict(
        checkpoint=a.checkpoint, cases=len(indices), window=a.window, context=a.context, fp32=a.fp32,
        one_step_noisy_gain=noisy_num / max(noisy_den, 1e-30) if noisy else None,
        one_step_clean=rollout_metrics(mu[:, 1:w1].reshape(-1, 1, d), one_step.reshape(-1, 1, d), mu[:, :w1 - 1].reshape(-1, d)),
        open_loop=rollout_metrics(mu[:, a.context:], predicted, mu[:, a.context - 1]),
        state_spread_over_sigma0=float(mu.std((0, 1)).mean() / model.sigma0) if noisy else None,
        noisy_state_information_upper_bound_nats=(float(.5 * torch.log1p(mu.flatten(0, 1).var(0) / model.sigma0 ** 2).sum())
                                                  if noisy else None))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / "rollout_gain.json").write_text(json.dumps(report, indent=2) + "\n")
    ol = report["open_loop"]
    print(json.dumps(dict(one_step_noisy_gain=report["one_step_noisy_gain"], spread_ratio=ol["spread_ratio"],
                          carry_ratio=ol["carry_ratio"], motion_gain=ol["motion_gain"],
                          state_spread_over_sigma0=report["state_spread_over_sigma0"]), indent=2))


if __name__ == "__main__":
    main()
