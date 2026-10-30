"""CPU aggregation of frozen pre-/post-pooling position probes."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


DIMENSIONS = {"query_pool": 64, "cls": 192, "patch_mean": 192, "spatial4x4": 3072,
              "cls_rp64": 64, "patch_mean_rp64": 64, "spatial4x4_rp64": 64}
MATCHED = ["query_pool", "cls_rp64", "patch_mean_rp64", "spatial4x4_rp64"]
NAMES = {"query_pool": "QueryPool 64", "cls_rp64": "CLS RP64", "patch_mean_rp64": "Mean RP64", "spatial4x4_rp64": "Spatial RP64"}


def read(path):
    return json.loads(path.read_text())


def position_score(row, feature, kind="clean", split="validation"):
    width = 2 if row["task"] == "tworoom" else 3
    return float(np.mean(row["features"][feature][kind][split]["r2_against_train_mean"][:width]))


def collect(root):
    paths = sorted(root.glob("*/spatial-probe.json"))
    exits = [read(p) for p in sorted((root / "exits").glob("*.json"))]
    assert len(paths) == len(exits) == 14 and all(x["returncode"] == 0 for x in exits)
    assert {x["id"] for x in exits} == {p.parent.name for p in paths}
    rows, reference, projections = [], {}, None
    for path in paths:
        row = read(path)
        name = path.parent.name
        match = re.fullmatch(r"(tworoom|cube)-seed([12])-step(\d+)", name)
        assert match and row["step"] == int(match[3])
        row.update(id=name, task=match[1], seed=int(match[2]), source=str(path))
        assert len(row["train"]["episodes"]) == len(row["validation"]["episodes"]) == 128
        assert not set(row["train"]["episodes"]).intersection(row["validation"]["episodes"])
        assert row["projection_seed"] == 20261030
        if projections is None:
            projections = row["projections"]
        assert row["projections"] == projections
        signature = dict(train=row["train"], validation=row["validation"])
        reference.setdefault(row["task"], signature)
        assert reference[row["task"]] == signature
        assert set(row["features"]) == set(DIMENSIONS)
        for feature, dim in DIMENSIONS.items():
            for kind, score in row["features"][feature].items():
                n = 896 if kind == "temporal_displacement" else 1024
                assert score["feature_dim"] == dim and score["ridge"] == .001
                assert score["train_samples"] == score["validation_samples"] == n
                assert score["solver"] == ("dual" if dim > n else "primal")
        rows.append(row)
    rows.sort(key=lambda r: (r["task"] == "cube", r["seed"], r["step"]))
    return dict(root=str(root), endpoints=len(exits), all_exit_zero=True,
                all_projection_hashes_identical=True, taskwise_probe_cases_identical=True,
                train_validation_episode_disjoint=True, dimensions=DIMENSIONS,
                protocol=read(root / "protocol.json"), exits=exits, checkpoints=rows)


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ["#2563eb", "#9333ea", "#f59e0b", "#16a34a"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, task in enumerate(["tworoom", "cube"]):
        for j, kind in enumerate(["clean", "temporal_displacement"]):
            ax = axes[i, j]
            steps = [0, 600, 2000] + ([11671] if task == "cube" else [])
            for feature, color in zip(MATCHED, colors):
                for seed, style in [(1, "-"), (2, "--")]:
                    rows = sorted([r for r in result["checkpoints"] if r["task"] == task and r["seed"] == seed], key=lambda r: r["step"])
                    ax.plot(range(len(steps)), [position_score(r, feature, kind) for r in rows], style,
                            color=color, marker="o", markersize=4, linewidth=1.7)
            title = "TwoRoom agent xy" if task == "tworoom" else "Cube block xyz"
            ax.set_title(title + (": position" if kind == "clean" else ": displacement"))
            ax.set_xticks(range(len(steps)), [str(s) for s in steps])
            ax.set_xlabel("Checkpoint update (stages equally spaced)")
            ax.set_ylabel("Held-out linear probe R²")
            ax.set_ylim(-.15, 1.03)
            ax.axhline(0, color="black", linewidth=.6)
            ax.grid(alpha=.2)
    handles = [Line2D([0], [0], color=c, label=NAMES[f], linewidth=2) for f, c in zip(MATCHED, colors)]
    handles += [Line2D([0], [0], color="black", linestyle=s, label=f"Training seed {seed}") for seed, s in [(1, "-"), (2, "--")]]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("Same 64-dimensional readout budget: where is position linearly accessible?", fontsize=15)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"spatial-probes.{ext}", dpi=170)
    plt.close(fig)


def report(result, out):
    rows = result["checkpoints"]
    lines = ["# Frozen spatial readout audit", "",
        "All 14 endpoints exit 0. Each checkpoint uses exactly the same task-specific 128 training and 128 validation episodes and window starts. Train and validation episodes are disjoint. All projection matrices have identical hashes across tasks, seeds, stages and splits; the projection seed was fixed at 20261030. Every probe uses training-only feature normalization and ridge=.001. Primal/dual solver choice and feature dimensions were checked.", "",
        "## What changed in our understanding", "",
        "**TwoRoom position is already linearly accessible in encoder patch features, while the current learned QueryPool exposes much less of it.** At update 2000, spatial4x4 followed by a fixed 64-dimensional random projection gives agent x/y R²=.958/.938 and .951/.893, versus QueryPool64=.141/.255 and .113/.291. The matched output dimension matters: this gap is not solely an advantage from fitting a 3072-dimensional readout.", "",
        "**Much of this spatial accessibility predates learning.** At initialization, spatialRP64 already reaches .944/.943 and .928/.892, while QueryPool gives .130/.282 and .122/.297. The initialization-to-2000 path does not create a substantially stronger QueryPool position code in TwoRoom. Spatial displacement readout is also strong, so the result is not explained solely by static episode identity: at update 2000, spatialRP64 displacement x/y R²=.872/.801 and .850/.648, versus QueryPool −.007/.123 and −.070/.162.", "",
        "**Cube has a different trajectory.** QueryPool position readout initially weakens at 600, then improves at 2000 and 11671. By 11671, QueryPool and other matched-64 readouts are broadly competitive. Cube height is .546/.720 from QueryPool versus .577/.697 from spatialRP64. QueryPool block-displacement probes can outperform spatialRP64 at this endpoint. A universal claim that the pooling architecture must be replaced is not supported.", "",
        "**The unrestricted high-dimensional probe is an important failure control.** On TwoRoom, the full 3072-dimensional spatial ridge nearly fits the training positions but generalizes poorly, sometimes with very negative R². It is not evidence that spatial features lack information. The fixed high-dimensional probe is poorly regularized/conditioned for this finite dataset. Random projection changes capacity and conditioning, and occurs before coordinate-wise standardization, so it is not a pure information-preservation intervention.", "",
        "## Matched 64-dimensional position probes", "",
        "Values average agent x/y for TwoRoom and block x/y/z for Cube. Individual training seeds are never pooled away.", "",
        "| Task | Seed | Update | QueryPool64 | CLS RP64 | Mean RP64 | Spatial RP64 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['seed']} | {r['step']} | " + " | ".join(f"{position_score(r, f):.4f}" for f in MATCHED) + " |")
    lines += ["", "## Capacity and overfitting control", "", "| Task | Seed | Update | Full spatial train R² | Full spatial validation R² | Spatial RP64 validation R² |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['seed']} | {r['step']} | {position_score(r,'spatial4x4',split='train'):.4f} | {position_score(r,'spatial4x4'):.4f} | {position_score(r,'spatial4x4_rp64'):.4f} |")
    lines += ["", "## Per-coordinate position probes", "", "Coordinate order is agent x/y, target x/y for TwoRoom; block xyz, effector xyz for Cube. Full vectors for all seven feature families and all temporal controls are in results.json.", "", "| Checkpoint | Feature | Validation R² vector |", "|---|---|---|"]
    for r in rows:
        for f in MATCHED:
            vals = r["features"][f]["clean"]["validation"]["r2_against_train_mean"]
            lines.append(f"| {r['id']} | {NAMES[f]} | " + ", ".join("null" if x is None else f"{x:.4f}" for x in vals) + " |")
    lines += ["", "## Provenance and limits", "",
        "Initial and 600-step weights come from the new fresh rate-on gate; 2000-step weights come from the earlier temporal-forecast campaign, and Cube 11671 comes from its first-window-epoch continuation. A separate CPU tensor comparison confirms all 550 initial tensors exactly match the historical initialization for each of the four task/seed pairs. The 600 point is a numerical repeat rather than a byte-identical intermediate checkpoint on the historical 2000 training path; the lines describe comparable stages, not a single uninterrupted optimizer trace.", "",
        "These are diagnostic readouts trained with physical labels on frozen features, not usable policies or demonstrations of learned dynamics. Good random-feature probes at initialization do not imply the world model predicts actions' effects. The evaluated planner still consumes the original QueryPool state. The projected spatial features have not been trained with a predictor, priced noisy channel or matched planner, and their raw scale differs strongly from QueryPool. This audit neither proves robustness under posterior noise nor promises a control gain from replacing pooling.", "",
        "The supported conclusion is narrower: the TwoRoom image encoder has spatial information accessible to a fixed 64-dimensional spatial readout, but the current learned pooling path does not expose it nearly as well. Coupled with the frozen-information and rate-removal audits, this localizes a concrete representation bottleneck worth testing. It does not establish an information-theoretic absence of all nonlinear signal, nor algorithmic novelty.", "", "![Matched dimensional probes](spatial-probes.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = collect(Path(args.root))
    identity_path = out / "initial-tensor-identity.json"
    identities = read(identity_path)
    assert len(identities) == 4 and all(r["key_sets_match"] and not r["different_tensors"] for r in identities)
    result["initial_tensor_identity"] = identities
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report(result, out)
    plot(result, out)
    print(json.dumps(dict(output=str(out), checkpoints=len(result["checkpoints"]), matched_dimensions=True)))


if __name__ == "__main__":
    main()
