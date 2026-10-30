"""Aggregate frozen information audits without pooling away training seeds."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def aggregate_interventions(batches):
    keys = sorted({(z["condition"], z["t"]) for b in batches for z in b["interventions"]})
    result = {}
    for condition, level in keys:
        rows = [z for b in batches for z in b["interventions"] if z["condition"] == condition and z["t"] == level]
        area = np.array([z["motion_fraction"] for z in rows])
        total = np.array([z["all_mse"] for z in rows])
        motion = np.array([z["motion_mse"] for z in rows]) * area
        static = np.array([z["static_mse"] for z in rows]) * (1 - area)
        result.setdefault(str(level), {})[condition] = dict(
            all_mse=float(total.mean()), motion_mse=float(motion.sum() / area.sum()),
            static_mse=float(static.sum() / (1 - area).sum()),
            motion_fraction=float(area.mean()), motion_error_fraction=float(motion.sum() / total.sum()),
            batch_all_mse=total.tolist())
    for level, conditions in result.items():
        for name, row in conditions.items():
            reference = "posterior" if name.startswith("posterior_") else "forecast"
            if name.startswith("scaled_"):
                sigma = name.split("_sigma")[-1]
                reference = f"scaled_posterior_scale1_sigma{sigma}"
            base = conditions[reference]
            for measure in ["all_mse", "motion_mse", "static_mse"]:
                row[measure + "_increase_percent"] = 100 * (row[measure] / base[measure] - 1)
    return result


def summarize(path):
    audit = json.loads(path.read_text())
    gradients = [b["gradient"] for b in audit["batches"]]
    gradient = {}
    for key, value in gradients[0].items():
        if isinstance(value, (int, float)):
            vals = [g[key] for g in gradients]
            gradient[key] = dict(mean=float(np.mean(vals)), minimum=float(np.min(vals)), maximum=float(np.max(vals)))
    labels = []
    for key in audit["probes"]["label_keys"]:
        labels += [key + "_" + axis for axis in ("xy" if key.startswith("pos_") else "xyz")]
    assert len(labels) == len(audit["probes"]["scores"]["clean"]["mse"])
    return dict(id=path.parent.name, source=str(path), task="cube" if path.parent.name.startswith("cube") else "tworoom",
                seed=int(re.search(r"seed(\d+)", path.parent.name).group(1)), step=audit["step"],
                labels=labels, probes=audit["probes"]["scores"],
                probe_train_episodes=len(audit["probes"]["train_episodes"]),
                probe_validation_episodes=len(audit["probes"]["validation_episodes"]),
                gradient=gradient, interventions=aggregate_interventions(audit["batches"]))


def target_score(row, probe):
    width = 3 if row["task"] == "cube" else 2
    return float(np.mean(row["probes"][probe]["r2_against_train_mean"][:width]))


def label(row):
    if row["task"] == "cube":
        return f"Cube TF s{row['seed']}"
    method = "no-anchor" if "constant" in row["id"] else ("TF" if "temporal" in row["id"] else "original")
    return f"TwoRoom {method} s{row['seed']}"


def figures(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [label(r) for r in rows]
    idx = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    for shift, key, title, color in [(-.24, "clean", "Clean state", "#2563eb"),
                                    (0, "noisy_sigma0.1", "State + noise 0.1", "#dc2626"),
                                    (.24, "time_shifted", "Time-shifted state", "#9ca3af")]:
        ax.bar(idx + shift, [target_score(r, key) for r in rows], .23, label=title, color=color)
    ax.set_ylabel("Held-out linear probe R²")
    ax.set_title("Position information: TwoRoom agent xy / Cube block xyz")
    ax.set_ylim(-.15, 1.05)
    ax.axhline(0, color="black", linewidth=.6)
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for shift, key, title, color in [(-.24, "forecast_other_initial", "Wrong initial state", "#2563eb"),
                                    (0, "forecast_wrong_actions", "Wrong actions", "#f59e0b"),
                                    (.24, "forecast_wrong_anchor", "Wrong anchor", "#16a34a")]:
        vals = [r["interventions"]["0.95"][key]["motion_mse_increase_percent"] for r in rows]
        ax.bar(idx + shift, vals, .23, label=title, color=color)
    ax.set_yscale("symlog", linthresh=.01)
    ax.set_ylabel("Motion-region MSE increase % (symlog)")
    ax.set_title("Functional use at corruption t = 0.95")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.bar(idx - .18, [100 * r["interventions"]["0.95"]["forecast"]["motion_fraction"] for r in rows], .35, label="Motion-area fraction", color="#94a3b8")
    ax.bar(idx + .18, [100 * r["interventions"]["0.95"]["forecast"]["motion_error_fraction"] for r in rows], .35, label="Fraction of error in motion area", color="#7c3aed")
    ax.set_ylabel("Percent")
    ax.set_title("Small motion area already contains most residual error")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(idx, [r["gradient"]["rate_to_flow"]["mean"] for r in rows], color="#0d9488")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Priced-rate / flow gradient norm")
    ax.set_title("Frozen objective gradients with respect to encoded state")
    for ax in axes.flat:
        ax.set_xticks(idx, names, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen information audit: each training seed retained", fontsize=16)
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"information-audit.{suffix}", dpi=170)
    plt.close(fig)


def write_report(rows, out):
    text = ["# Frozen information audit: TwoRoom and Cube", "",
            f"All {len(rows)} completed checkpoints are retained separately. Each audit uses 16 validation windows for interventions, and 128 training / 128 validation episodes for fixed-ridge probes. The model is frozen. Physical labels train only the diagnostic linear readout; they do not update the world model or policy.", "",
            "## Main observations", "",
            "TwoRoom retains weak clean, linearly recoverable position information, but the fixed posterior noise overwhelms it. Frozen-renderer state substitution has negligible effect even when error is measured only on moving pixels. Removing posterior noise and amplifying the tiny signal up to 100 times does not substantially improve this renderer. Small state amplitude is therefore not the entire failure, and linear-probe weakness is not a proof that all nonlinear information is absent.", "",
            "The moving region occupies about 9.4% of TwoRoom pixels but contains most of the current residual denoising error. The simple explanation that the current loss is numerically dominated by static-background error is contradicted by this audit. Earlier optimization may still have prioritized easily predictable pixels. Both flow and rate share FRAME_ELEMS normalization; image resolution alone does not create an asymmetric weighting.", "",
            "At the frozen temporal-forecast endpoints, the priced-rate gradient norm is smaller than the flow gradient norm, and the two are nearly orthogonal in encoded-state coordinates. This does not support a claim that the current total state gradient is simply overwhelmed by rate. It does not reconstruct the earlier optimization trajectory or encoder-parameter gradients. Fresh-versus-resumed rate controls are needed to distinguish objective preference from an optimization trap.", "",
            "Cube is qualitatively different: the state supports strong position probes and the renderer strongly responds to state and action substitutions. Effector position is more accurately linearly recoverable than cube height, and cube displacement probes are weaker than effector displacement probes. This suggests a possible imbalance toward robot information; it does not prove the cause of planning failure or rule out nonlinear decoding. The companion action-candidate/goal-metric audit is needed for causal attribution.", "",
            "## Position probes and residual error", "",
            "Position R² below averages agent x/y for TwoRoom and block x/y/z for Cube. Full per-coordinate values are preserved below and in results.json. The motion mask is derived from observed within-window pixel variation, not semantic object labels.", "",
            "| Checkpoint | Update | Clean R² | Noisy R² | Time-shift R² | Motion pixels % | Error in motion % | Rate/flow grad |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        v = r["interventions"]["0.95"]["forecast"]
        text.append(f"| {label(r)} | {r['step']} | {target_score(r,'clean'):.4f} | {target_score(r,'noisy_sigma0.1'):.4f} | {target_score(r,'time_shifted'):.4f} | {100*v['motion_fraction']:.2f} | {100*v['motion_error_fraction']:.2f} | {r['gradient']['rate_to_flow']['mean']:.4f} |")
    text += ["", "## Functional substitutions", "", "Values are percent change of aggregate MSE relative to the correct forecast, at t=.95. Motion-region values weight each batch by its actual motion area. Very small signed changes should be treated as negligible, not as reliable improvements.", "", "| Checkpoint | Wrong state: all / motion % | Wrong actions: all / motion % | Wrong anchor: all / motion % |", "|---|---:|---:|---:|"]
    for r in rows:
        v = r["interventions"]["0.95"]
        vals = [f"{v[k]['all_mse_increase_percent']:.5f} / {v[k]['motion_mse_increase_percent']:.5f}" for k in ["forecast_other_initial", "forecast_wrong_actions", "forecast_wrong_anchor"]]
        text.append("| " + label(r) + " | " + " | ".join(vals) + " |")
    text += ["", "## Per-coordinate probes", "", "Clean and noisy probes use only training-split feature normalization. Ridge coefficient .001 is fixed. Target dimensions with zero displacement variance have null R² and are excluded from means.", ""]
    for r in rows:
        text += ["### " + label(r), "", "| Coordinate | Clean | Noisy .1 | Time-shifted | Initial carry | Displacement |", "|---|---:|---:|---:|---:|---:|"]
        for i, key in enumerate(r["labels"]):
            values = [r["probes"][p]["r2_against_train_mean"][i] for p in ["clean", "noisy_sigma0.1", "time_shifted", "initial_carry", "temporal_displacement"]]
            text.append("| " + key + " | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    text += ["", "## Scope", "", "These are frozen endpoint diagnostics, not training or control improvements. Scaling states or removing anchors moves the renderer off its training support. Cross-task checkpoints differ in training budget (TwoRoom 2000/2600 versus Cube 11671); they demonstrate current mechanisms, not equal-budget algorithm rankings. Existing TwoRoom adaptation results from update 2600 used different checkpoints/cases and are not replaced by update-2000 intervention values. All seeds and full gradient ranges, scale curves and intervention levels are retained in results.json.", "", "![Frozen audit](information-audit.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(Path(args.root).glob("*/information-audit.json"))
    if len(paths) != args.expected:
        raise ValueError(f"Expected {args.expected} audits; found {len(paths)}")
    rows = [summarize(p) for p in paths]
    rows.sort(key=lambda r: (r["task"] == "cube", r["id"]))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(dict(count=len(rows), checkpoints=rows), indent=2, allow_nan=False) + "\n")
    write_report(rows, out)
    figures(rows, out)
    print(json.dumps(dict(output=str(out), checkpoints=len(rows))))


if __name__ == "__main__":
    main()
