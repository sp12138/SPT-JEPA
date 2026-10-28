"""Make scientific figures from the checked main-conclusions JSON only."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def finish(fig, root, name):
    fig.savefig(root / (name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(root / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def clean(ax):
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data = json.loads(Path(a.report).read_text())
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    arms = data["arms"]
    keys = ["base001", "geometry", "permuted", "correlated", "forecast", "fork_null", "fork",
            "base1", "snr", "vdm", "gradient"]
    labels = {
        "base001": "Baseline beta=.01", "geometry": "I02 diagonal Q",
        "permuted": "I02 permuted Q", "correlated": "I03 correlated noise",
        "forecast": "I05 forecast rendering", "fork_null": "I09 zero-loss control",
        "fork": "I09 inverse-action loss", "base1": "Baseline beta=1",
        "snr": "I01 weighted FM", "vdm": "I01 finite VDM bound",
        "gradient": "I06 gradient constraint",
    }
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(keys))
    for cohort, offset, color, label in [
        ("diagnostic", -.16, "#4c78a8", "Fixed diagnostic cases"),
        ("confirmation", .16, "#f2a541", "New windows / new evaluation seeds"),
    ]:
        values = [arms[k].get(cohort, {}).get("p0_mean", np.nan) for k in keys]
        bars = ax.barh(y + offset, values, .29, color=color, label=label)
        for value, bar in zip(values, bars):
            if np.isfinite(value):
                ax.text(value + .8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    ax.set_yticks(y, [labels[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("PushT P0 closed-loop success (%)")
    ax.set_title("Matched candidate screen: several positive changes, several decisive failures", pad=28)
    ax.text(.5, 1.02, "One training seed per arm; three evaluation seeds per bar. Not a final cross-training-seed ranking.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#444444")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    clean(ax)
    finish(fig, out, "main-candidate-planning")

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for seed, color in zip(["0", "1", "2"], ["#4c78a8", "#f28e2b", "#59a14f"]):
        pair = data["semantic_feedback"].get(seed, {})
        if "semantic" not in pair or "same_weights_pixel" not in pair:
            continue
        for variant, style, name in [
            ("semantic", "-", "semantic feedback"), ("same_weights_pixel", "--", "pixel feedback"),
        ]:
            item = pair[variant]
            curve = item["quality"]["mean_by_horizon"]
            axs[0].plot(np.arange(1, 14), curve["mean_agent"], style, color=color,
                        label=f"seed {seed}: {name}", lw=1.7)
            axs[1].plot(np.arange(1, 14), curve["mean_block"], style, color=color, lw=1.7)
    for ax, label in zip(axs, ["Agent position error (px)", "Block position error (px)"]):
        ax.set_xlabel("Autoregressive prediction horizon")
        ax.set_ylabel(label)
        ax.set_xticks([1, 4, 7, 10, 13])
        ax.axvline(7, color="gray", ls=":", lw=1)
        clean(ax)
    axs[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("I08: identical weights, different feedback loop; long-horizon drift remains")
    fig.tight_layout(rect=[0, 0, 1, .94])
    finish(fig, out, "main-I08-same-weights")

    full = data["full_three_epoch"]
    names, summaries = [], []
    for key, item in full.items():
        names.append(key.replace("-beta", " beta="))
        summaries.append(item["summary"])
    if names:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.6))
        values = [
            [s["h7_agent_copy"] for s in summaries],
            [s["fvd"] for s in summaries],
            [s["p0_mean"] if s["p0_mean"] is not None else np.nan for s in summaries],
        ]
        for ax, vals, title in zip(axs, values, ["h7 agent / copy (lower)", "FVD-256 (lower)", "P0 success % (higher)"]):
            bars = ax.bar(np.arange(len(names)), vals, color="#4c78a8")
            ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            for i, (bar, value) in enumerate(zip(bars, vals)):
                if np.isfinite(value):
                    ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
                else:
                    ax.text(i, .05, "not tested", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, rotation=90)
            if title.startswith("P0"):
                ax.set_ylim(0, 100)
            clean(ax)
        fig.suptitle("Full-data checkpoints at three epochs: passing the collapse gate does not ensure useful control")
        fig.text(.5, -.01, "Training seed 0; all shown checkpoints pass the noisy-posterior collapse gate. Native S2 planning was not tested.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=[0, .04, 1, .92])
        finish(fig, out, "main-full-data-checkpoints")

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.6))
    for i, (keys_here, title) in enumerate([
        (["base001", "geometry", "permuted"], "I02: orientation-specific advantage is not supported"),
        (["fork_null", "fork"], "I09: inverse-action loss harms closed-loop control"),
    ]):
        ax = axs[i]
        x = np.arange(len(keys_here))
        for cohort, off, color in [("diagnostic", -.17, "#4c78a8"), ("confirmation", .17, "#f2a541")]:
            vals = [arms[k][cohort]["p0_mean"] for k in keys_here]
            ax.bar(x + off, vals, .3, color=color, label=cohort)
        ax.set_xticks(x, [labels[k] for k in keys_here], rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("P0 success (%)")
        ax.set_title(title, fontsize=10)
        clean(ax)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Useful negative controls change the mechanistic interpretation")
    fig.text(.5, -.01, "Core initial tensors match across I02 arms. All 444 initial tensors match for fork / zero-loss control.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, .91])
    finish(fig, out, "main-mechanism-controls")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    methods = ["goal_mse", "covariance", "gramian-contextfix"]
    labels_i10 = ["Goal MSE", "Covariance cost", "Corrected Gramian cost"]
    for i, (method, label) in enumerate(zip(methods, labels_i10)):
        vals = [x["result"]["metrics"]["success_rate"] for x in data["geometric_planning"][method]]
        ax.bar(i, np.mean(vals), color="#4c78a8" if i == 0 else "#c44e52", alpha=.8)
        ax.scatter(np.full(3, i) + np.linspace(-.09, .09, 3), vals, color="#222222", s=25, zorder=3)
        ax.text(i, max(vals) + 3, f"mean {np.mean(vals):.1f}%", ha="center", fontsize=9)
    ax.set_xticks(np.arange(3), labels_i10, rotation=12, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("P0 closed-loop success (%)")
    ax.set_title("I10: coordinate-aware costs do not ensure useful control", pad=25)
    ax.text(.5, 1.02, "Same frozen weights and CEM search budget; three paired evaluation seeds.", transform=ax.transAxes, ha="center", fontsize=8)
    clean(ax)
    finish(fig, out, "main-I10-planning-costs")
    for suffix in ["png", "svg"]:
        stale = out / ("main-probe-vs-planning." + suffix)
        if stale.exists():
            stale.unlink()
    print(json.dumps({"figures": len(list(out.glob("main-*.png"))), "output": str(out)}))


if __name__ == "__main__":
    main()

