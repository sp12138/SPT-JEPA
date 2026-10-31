"""Plot the complete frozen-model NGC comparison; no model or GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    assert data["passed"] and data["all48exit0"] and len(data["rows"]) == 48
    rows = data["rows"]
    for row in rows:
        assert len(row["successes"]) == row["cases"] == 50
        assert abs(100 * sum(row["successes"]) / 50 - row["percent"]) < 1e-9
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
    families = [("local1024", "Local 1024"), ("native-raw", "Native raw"),
                ("TF64", "Temporal forecast")]
    colors = {"l2": "#87919b", "ngc": "#007d83"}
    for ti, task in enumerate(["tworoom", "cube"]):
        for ci, case_set in enumerate(["primary", "fresh"]):
            ax = axes[ti, ci]
            for j, (family, label) in enumerate(families):
                item, = [x for x in data["combined"] if
                         (x["task"], x["family"], x["case_set"]) == (task, family, case_set)]
                y = 2 - j
                for mode in ["l2", "ngc"]:
                    subset = [r for r in rows if (r["task"], r["family"], r["case_set"], r["mode"])
                              == (task, family, case_set, mode)]
                    assert {r["seed"] for r in subset} == {1, 2}
                    assert abs(sum(r["percent"] for r in subset) / 2 - item[mode]) < 1e-9
                ax.plot([item["l2"], item["ngc"]], [y, y], color="#b4bec5", lw=3, zorder=1)
                for mode in ["l2", "ngc"]:
                    ax.scatter(item[mode], y, s=65, color=colors[mode], edgecolors="white", zorder=2)
                ax.text((item["l2"] + item["ngc"]) / 2, y + .22,
                        f'{item["l2"]:g} → {item["ngc"]:g}', ha="center", fontsize=10)
                lo, hi = item["ci95_pp"]
                ax.text(108, y, f'{item["delta_pp"]:+g} [{lo:+g}, {hi:+g}]',
                        va="center", fontsize=9, color="#263542")
            ax.set_title(f'{"TwoRoom" if task == "tworoom" else "Cube"} · {case_set} 50 cases', loc="left")
            ax.set_yticks([2, 1, 0], [x[1] for x in families])
            ax.set_ylim(-.45, 2.65)
            ax.set_xlim(-5, 155)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", alpha=.15)
            ax.axvline(104, color="#dae0e4", lw=.8)
            ax.text(108, 2.5, "Δ pp [95% case CI]", fontsize=9, color="#596570")
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_bounds(0, 100)
            ax.tick_params(axis="y", length=0)
            if ti == 1:
                ax.set_xlabel("Strict task success (%) · mean of two training seeds", loc="left")
    fig.suptitle("A public graph cost unlocks TwoRoom local-model planning; Cube remains unresolved",
                 x=.02, ha="left", fontsize=15, fontweight="bold")
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[m], label=l)
               for m, l in [("l2", "Contemporary L2 replay"), ("ngc", "Public NGC · stable attachment")]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.015, .95), frameon=False, ncol=2)
    fig.text(.02, .035,
             "Same checkpoint, cases and CEM budget within each pair. All 48 runs exited 0. "
             "CI: paired cases, conditional on these two trained seeds.\n"
             "Families differ in capacity; Cube local: 6k updates, native/TF: 12k. "
             "5,000 train-only graph nodes/model; no new network training or action labels.",
             fontsize=9, color="#4d5963", va="bottom")
    fig.subplots_adjust(left=.14, right=.98, bottom=.17, top=.84, hspace=.42, wspace=.34)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / f"ngc-comparison.{ext}", dpi=180, facecolor="white")
    (args.output / "provenance.json").write_text(json.dumps({
        "input_sha256": hashlib.sha256(raw).hexdigest(), "rows": 48,
        "success_vectors_checked": True, "all_mean_points_checked": True,
        "model_or_gpu_calls": 0,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
