"""CPU-only review of the TwoRoom posterior/target-noise 2x2 gate."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .information_audit_summary import summarize
except ImportError:
    from information_audit_summary import summarize


ARMS = ["base", "clean-state", "pure-target-noise", "both"]
LABELS = ["Base", "Clean state", "Pure target noise", "Both"]


def read(path):
    return json.loads(path.read_text())


def agent_r2(row, kind="clean"):
    return float(np.mean(row["probes"][kind]["r2_against_train_mean"][:2]))


def collect(root, out):
    shared = root.parent
    rate_training = shared / "mechanism_rate600_20261030"
    rate_evaluation = shared / "mechanism_rateeval_20261030"
    exits = [read(p) for p in sorted((root / "exits").glob("*.json"))]
    assert len(exits) == 26 and all(x["returncode"] == 0 for x in exits)
    initial = read(out / "initial-tensor-audit.json")
    assert len(initial) == 6 and all(x["tensors"] == 550 and x["key_sets_match"] and not x["different_tensors"] for x in initial)
    rows, reference = [], None
    for seed in [1, 2]:
        base_id = f"tworoom-fresh-rate-on-seed{seed}"
        base_config = read(rate_training / base_id / "config.json")
        base_planning = read(root / f"planning-control-seed{seed}" / "metrics.json")
        for arm in ARMS:
            rid = f"tworoom-{arm}-seed{seed}"
            if arm == "base":
                info_path = rate_evaluation / ("info-" + base_id) / "information-audit.json"
                planning = base_planning
                health_path = rate_training / ("health-" + base_id) / "health.json"
                cfg = base_config
            else:
                info_path = root / ("info-" + rid) / "information-audit.json"
                planning = read(root / ("planning-" + rid) / "metrics.json")
                health_path = root / ("health-" + rid) / "health.json"
                cfg = read(root / rid / "config.json")
            model_config = dict(cfg["model"])
            noise_scale = model_config.pop("posterior_noise_scale", 1.)
            force_t_one = model_config.pop("flow_force_t_one", False)
            assert model_config == base_config["model"]
            for key in ["seed", "batch_size", "learning_rate", "lr", "weight_decay", "warmup_steps", "scheduler_steps", "ramp_steps", "data"]:
                assert cfg.get(key) == base_config.get(key), (rid, key)
            row = summarize(info_path)
            row.update(id=rid, arm=arm, seed=seed, task="tworoom", posterior_noise_scale=noise_scale,
                       flow_force_t_one=force_t_one, config=cfg, planning=planning, health=read(health_path))
            assert row["step"] == 600 and row["health"]["audit"]["step"] == 600
            source = read(info_path)
            signature = {k: source["probes"][k] for k in ["train_episodes", "validation_episodes", "label_keys"]}
            if reference is None:
                reference = signature
            assert signature == reference
            assert planning["pairs"] == base_planning["pairs"]
            a = np.asarray(base_planning["metrics"]["episode_successes"], dtype=bool)
            b = np.asarray(planning["metrics"]["episode_successes"], dtype=bool)
            row["paired_against_base"] = dict(n=len(a), success_percent=float(100 * b.mean()),
                difference_pp=float(100 * (b.astype(float) - a).mean()), gains=int((b & ~a).sum()), losses=int((a & ~b).sum()))
            v = row["interventions"]["1.0"]
            row["clean_constant_state_mse_change_percent"] = 100 * (v["scaled_posterior_scale0_sigma0"]["all_mse"] / v["scaled_posterior_scale1_sigma0"]["all_mse"] - 1)
            rows.append(row)
    interaction = []
    for seed in [1, 2]:
        rr = {r["arm"]: r for r in rows if r["seed"] == seed}
        interaction.append(dict(seed=seed, success_interaction_pp=rr["both"]["paired_against_base"]["success_percent"]
            - rr["clean-state"]["paired_against_base"]["success_percent"] - rr["pure-target-noise"]["paired_against_base"]["success_percent"]
            + rr["base"]["paired_against_base"]["success_percent"]))
    return dict(root=str(root), protocol=read(root / "protocol.json"), preflight=read(root / "PREFLIGHT-REVIEW.json"),
                exits=exits, initial_tensor_audit=initial, reused_baselines=2, rows=rows, interaction=interaction)


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    x = np.arange(4)
    measures = [("Planning on 50 matched cases", "Success %", lambda r: r["paired_against_base"]["success_percent"]),
                ("Clean agent position probe", "Held-out R² (agent xy)", lambda r: agent_r2(r)),
                ("Encoded-state amplitude", "Mean feature std", lambda r: r["gradient"]["state_std"]["mean"]),
                ("Clean agent displacement probe", "Held-out R² (agent xy)", lambda r: agent_r2(r, "temporal_displacement"))]
    for ax, (title, ylabel, measure) in zip(axes.flat, measures):
        for seed, shift, color in [(1, -.18, "#2563eb"), (2, .18, "#f97316")]:
            rr = {r["arm"]: r for r in result["rows"] if r["seed"] == seed}
            ax.bar(x + shift, [measure(rr[arm]) for arm in ARMS], .35, color=color, label=f"Training seed {seed}")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, LABELS, rotation=15, ha="right")
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylim(0, 20)
    axes[0, 1].set_ylim(0, 1)
    axes[1, 1].set_ylim(-.1, 1)
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
    axes[0, 0].legend()
    fig.suptitle("TwoRoom information-channel 2×2: all runs stop at 600 updates", fontsize=15)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"channel-controls.{ext}", dpi=170)
    plt.close(fig)


def report(result, out):
    rows = result["rows"]
    lines = ["# TwoRoom information-channel 2×2 diagnostic", "",
        "All 26 new endpoints completed with exit 0: six training, six health, six frozen-information audits and eight planning evaluations (including two reused baseline weights). All six candidates match all 550 initial tensors of their corresponding fresh baseline. The model config differs only in the two declared channel controls; selected training/data settings also match. All eight planning runs use the same 50 validation cases.", "",
        "The 2×2 factors are: posterior sampling noise retained/removed, and standard target corruption/pure target noise t=1. Beta=.25 and sigma0=.1 inside the rate expression are unchanged. The controls are diagnostic surrogates, not an exact ELBO claim. Base training and information audits are reused from the rate-on fresh-600 cohort; its planning was evaluated in this campaign.", "",
        "## Results", "", "| Arm | Seed | P0 % | Versus base pp | Gained/lost | Clean agent xy R² | State std | Clean constant-state MSE change % |", "|---|---:|---:|---:|---:|---|---:|---:|"]
    for arm in ARMS:
        for r in [x for x in rows if x["arm"] == arm]:
            p = r["paired_against_base"]
            xy = r["probes"]["clean"]["r2_against_train_mean"][:2]
            lines.append(f"| {arm} | {r['seed']} | {p['success_percent']:.0f} | {p['difference_pp']:+.0f} | {p['gains']}/{p['losses']} | {xy[0]:.4f}, {xy[1]:.4f} | {r['gradient']['state_std']['mean']:.6g} | {r['clean_constant_state_mse_change_percent']:.6f} |")
    lines += ["", "**Neither channel intervention produces a clear representation recovery at this gate.** Clean position probes and state amplitude remain close to the baseline across both seeds. Planning is base 6/8%, clean-state 4/4%, pure-target-noise 6/14%, and both 8/8%. The pure-target-noise gain appears only in seed 2; no reliable cross-seed advantage or consistent positive interaction is established.", "",
        "**This is not merely an artifact of adding noise during the audit.** Under the explicitly clean-state sigma=0, t=1 diagnostic, replacing the tiny varying posterior means by their batch mean changes renderer MSE by less than .001% in every candidate. This intervention is a fixed-model readout check, not a proposed batch-dependent training loss. The decoder currently makes little use of these clean state differences.", "",
        "**The result does not imply permanent impossibility.** All runs stop at 600, inside the inherited 1249-update learning-rate warmup. These controls reject an immediate rescue at this measured stage; they cannot establish that more learning would never help. The spatial probe result independently shows position is accessible before the learned pooling path, motivating a more specific representation intervention rather than treating these two channel changes as a validated fix.", "",
        "## Correct interpretation of noise and loss", "",
        "For clean-state and both, the sigma=.1 noisy probes and noisy forecast interventions deliberately test robustness to the original channel; they are counterfactual to those arms' noise-free training. They must not be called measurements of native noisy-state use. The clean probes and clean sigma=0 intervention above avoid that ambiguity.", "",
        "Pure-target-noise changes the training corruption distribution. Its scalar training FM loss is therefore not directly comparable with the baseline's sampled-t loss as evidence of quality. Fixed-t diagnostic errors also remain denoising measurements, not generated-image quality or task success. No policy receives diagnostic physical labels.", "",
        "## Per-coordinate probes", "", "Coordinate order: agent x/y, target x/y. The fixed sigma=.1 column is a counterfactual noise-tolerance test for noise-off arms.", "", "| Arm | Seed | Clean R² vector | Original-noise R² vector |", "|---|---:|---|---|"]
    for arm in ARMS:
        for r in [x for x in rows if x["arm"] == arm]:
            vals = [", ".join(f"{v:.4f}" for v in r["probes"][k]["r2_against_train_mean"]) for k in ["clean", "noisy_sigma0.1"]]
            lines.append(f"| {arm} | {r['seed']} | {vals[0]} | {vals[1]} |")
    lines += ["", "## Decision", "", "Hold these six candidates at the measured gate. Preserve all seeds, full state and source snapshots. There is no evidence here for promoting a channel-only change to a long run. Continue the already specified, separately gated test of spatial locality versus matched shuffled pooling. This report does not alter or launch that campaign.", "", "![Channel controls](channel-controls.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = collect(Path(args.root), out)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report(result, out)
    plot(result, out)
    print(json.dumps(dict(output=str(out), rows=len(result["rows"]), new_endpoints=len(result["exits"]))))


if __name__ == "__main__":
    main()
