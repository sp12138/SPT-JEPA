"""Plot synthetic D7/D8 checks; CPU only, no model or benchmark data."""
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
data = json.loads((base / "third-cpu.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4), constrained_layout=True)
fig.suptitle("Synthetic rule checks — no Cube / TwoRoom performance claims", fontsize=14)

ax = axes[0]
x = np.arange(2)
ax.bar(x - .17, data["D7"]["euclidean_costs"], .34, label="Euclidean / Gramian", color="#506a8b")
ax.bar(x + .17, data["D7"]["deep_output_costs"], .34, label="Deepest output", color="#d27540")
ax.set(xticks=x, xticklabels=["(actuator, object)\n(0, 0)", "(actuator, object)\n(2, 1)"],
       ylabel="Goal discrepancy", title="D7: different preferred endpoint", ylim=(0, 5.2))
ax.legend(frameon=False, fontsize=9)

ax = axes[1]
sigma = data["D8"]["sigma"]
distance = np.linspace(0, .5, 301)
n = NormalDist()
tau = np.array([2 * n.cdf(d / (2 * sigma)) - 1 for d in distance])
for delta, color in [(0.8, "#a64838"), (0.1, "#316c8c")]:
    error = np.maximum(0, delta - tau) / 2
    ax.plot(distance, error, color=color, label=f"True future gap {delta:g}")
ax.axhline(.02, color="#777777", ls="--", lw=1, label="Uniform tolerance 0.02")
ax.set(xlabel="Gaussian mean distance (noise σ = 0.1)", ylabel="Necessary worst conditional TV error",
       title="D8: tight lower bound in binary example", ylim=(-.012, .42))
ax.legend(frameon=False, fontsize=9)

ax = axes[2]
rows = data["D8"]["adaptive_pair_constraints"][:2]
adaptive = [r["mean_dependent_kl_min_nats"] for r in rows]
fixed = [adaptive[0]] * 2
ax.bar(x - .17, fixed, .34, color="#506a8b", label="Same strong margin for both")
ax.bar(x + .17, adaptive, .34, color="#d27540", label="Observed future requirement")
ax.set(xticks=x, xticklabels=["Future gap 0.8", "Future gap 0.1"],
       ylabel="Minimum mean-dependent KL (nats)", title="D8: different minimum code allocation", ylim=(0, .0095))
ax.legend(frameon=False, fontsize=8.8, loc="upper right")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
for suffix in ["png", "svg"]:
    fig.savefig(base / f"third-rule-checks.{suffix}", dpi=170)
print(base / "third-rule-checks.png")
