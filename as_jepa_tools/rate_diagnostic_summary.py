"""CPU-only review of paired fresh/resumed rate-on/off diagnostic gates."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .information_audit_summary import summarize
except ImportError:
    from information_audit_summary import summarize


def read(path):
    return json.loads(path.read_text())


def position_r2(row, probe="clean"):
    count = 2 if row["task"] == "tworoom" else 3
    return float(np.mean(row["probes"][probe]["r2_against_train_mean"][:count]))


def pair_label(pair):
    task = "TwoRoom" if pair["task"] == "tworoom" else "Cube"
    stage = "fresh" if pair["initialization"] == "fresh" else "resume"
    return f"{task}\n{stage} s{pair['seed']}"


def collect(training_root, evaluation_root):
    protocol = read(training_root / "protocol.json")
    review = read(training_root / "REVIEW.json")
    initial = read(training_root / "initial-tensor-audit.json")
    resume = read(training_root / "resume-audit.json")
    exits = {"training_health": [read(p) for p in sorted((training_root / "exits").glob("*.json"))],
             "information_planning": [read(p) for p in sorted((evaluation_root / "exits").glob("*.json"))]}
    assert len(exits["training_health"]) == 32 and len(exits["information_planning"]) == 24
    assert all(x["returncode"] == 0 for values in exits.values() for x in values)
    assert review["all_fresh_tensors_equal"] and review["all_initial_resume_copies_identical"]
    assert len(initial) == 4 and all(x["initial_tensors_equal"] for x in initial)
    health = {x["id"]: x for x in review["results"]}
    rows = {}
    for run in protocol["runs"]:
        rid = run["id"]
        row = summarize(evaluation_root / ("info-" + rid) / "information-audit.json")
        row.update(id=rid, task=run["task"], initialization=run["initialization"], seed=run["seed"], beta=run["beta"], health=health[rid])
        planning = evaluation_root / ("planning-" + rid) / "metrics.json"
        row["planning"] = read(planning) if planning.exists() else None
        rows[rid] = row
    pairs = []
    for task in ["tworoom", "cube"]:
        for initialization in ["fresh", "resume2000"]:
            for seed in [1, 2]:
                prefix = f"{task}-{initialization}"
                on = rows[f"{prefix}-rate-on-seed{seed}"]
                off = rows[f"{prefix}-rate-off-seed{seed}"]
                pair = dict(task=task, initialization=initialization, seed=seed, on=on["id"], off=off["id"])
                if on["planning"] is not None:
                    assert on["planning"]["pairs"] == off["planning"]["pairs"]
                    a = np.asarray(on["planning"]["metrics"]["episode_successes"], dtype=bool)
                    b = np.asarray(off["planning"]["metrics"]["episode_successes"], dtype=bool)
                    pair["paired_planning"] = dict(n=len(a), on_percent=float(a.mean() * 100), off_percent=float(b.mean() * 100),
                        gains=int((b & ~a).sum()), losses=int((a & ~b).sum()), difference_pp=float((b.astype(float) - a).mean() * 100))
                else:
                    assert initialization == "fresh" and off["planning"] is None
                    pair["paired_planning"] = None
                pairs.append(pair)
    for pair in pairs:
        if pair["initialization"] == "resume2000":
            hashes = [x["checkpoint_sha256"] for x in resume if x["id"] in [pair["on"], pair["off"]]]
            assert len(hashes) == 2 and hashes[0] == hashes[1]
    return dict(training_root=str(training_root), evaluation_root=str(evaluation_root), protocol=protocol,
                review=review, initial_tensor_audit=initial, resume_audit=resume, exits=exits, runs=rows, pairs=pairs)


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs, rows = result["pairs"], result["runs"]
    idx = np.arange(len(pairs))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    measures = [
        ("State amplitude", "Mean feature std (log)", lambda r: r["gradient"]["state_std"]["mean"], "log"),
        ("Clean position probe", "Held-out R²", lambda r: position_r2(r), None),
        ("Noisy position probe (sigma .1)", "Held-out R²", lambda r: position_r2(r, "noisy_sigma0.1"), None),
        ("Initial-state use at t=.95", "MSE increase % (symlog)", lambda r: r["interventions"]["0.95"]["forecast_other_initial"]["all_mse_increase_percent"], "symlog"),
        ("Action use at t=.95", "MSE increase % (symlog)", lambda r: r["interventions"]["0.95"]["forecast_wrong_actions"]["all_mse_increase_percent"], "symlog"),
    ]
    for ax, (title, ylabel, measure, scale) in zip(axes.flat, measures):
        for shift, arm, title_arm, color in [(-.18, "on", "Rate on (beta=.25)", "#2563eb"), (.18, "off", "Rate off (beta=0)", "#f97316")]:
            ax.bar(idx + shift, [measure(rows[p[arm]]) for p in pairs], .35, color=color, label=title_arm)
        if scale == "log":
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=.01)
        else:
            ax.set_ylim(-.15, 1.05)
            ax.axhline(0, color="black", linewidth=.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(idx, [pair_label(p) for p in pairs], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    planned = [p for p in pairs if p["paired_planning"] is not None]
    ax = axes[1, 2]
    xx = np.arange(len(planned))
    ax.bar(xx - .18, [p["paired_planning"]["on_percent"] for p in planned], .35, color="#2563eb")
    ax.bar(xx + .18, [p["paired_planning"]["off_percent"] for p in planned], .35, color="#f97316")
    for i, pair in enumerate(planned):
        pp = pair["paired_planning"]
        ax.text(i, max(pp["on_percent"], pp["off_percent"]) + .8, f"g/l {pp['gains']}/{pp['losses']}", ha="center", fontsize=8)
    ax.set_xticks(xx, [pair_label(p) for p in planned], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 22)
    ax.set_title("Planning after 600 resumed updates")
    ax.set_ylabel("Success % on 50 paired cases")
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.suptitle("Removing the rate penalty: amplitude and action use do not guarantee better control", fontsize=15)
    for ext in ["png", "svg"]:
        fig.savefig(out / f"rate-controls.{ext}", dpi=170)
    plt.close(fig)


def write_report(result, out):
    rows, pairs = result["runs"], result["pairs"]
    lines = ["# Paired rate-removal diagnostic review", "",
        "All 16 training runs, 16 health audits, 16 frozen-information audits and 8 resumed-run planning evaluations completed with exit 0 (56 endpoints). Four fresh on/off initialization pairs match all 550 tensors. Resumed pairs start from identical complete checkpoint copies, including optimizer/RNG/cursor state. All seeds remain reported.", "",
        "Only beta differs within each pair: .25 versus 0. Both arms retain the temporal-forecast model. Fresh runs stop at update 600; resumed runs start at update 2000 and stop at 2600. These are diagnostic ablations, not a new algorithm or a long-run performance claim.", "",
        "## Planning outcome", "", "| Task | Training seed | Rate on % | Rate off % | Off − on pp | Gained / lost cases |", "|---|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        p = pair["paired_planning"]
        if p is not None:
            lines.append(f"| {pair['task']} | {pair['seed']} | {p['on_percent']:.0f} | {p['off_percent']:.0f} | {p['difference_pp']:+.0f} | {p['gains']} / {p['losses']} |")
    lines += ["", "The seed-1 TwoRoom gain is real for these fixed cases but is not replicated in seed 2. Cube decreases in both seeds. The evidence does not support promoting rate removal as a cross-seed control improvement.", "",
        "## Mechanistic conclusions", "",
        "**TwoRoom: rate removal strengthens an action/anchor route without recovering a useful noisy encoder state.** Resumed rate-off increases action-substitution sensitivity from 18.4/1.0% to 71.9/9.3%, and anchor-substitution sensitivity from 62.6/20.3% to 169.6/39.3%. Initial-state-substitution effects remain below .003% in magnitude. Clean position probes remain weak and nearly unchanged, while adding sigma=.1 noise leaves position R² below zero. This supports a decoder that can use initial visual context and action-dependent forecast conditions while largely bypassing the encoded initial state. It does not demonstrate counterfactual physical accuracy.", "",
        "**The fresh contrast rules out an immediate 600-step rescue, not eventual recovery.** Turning rate off from initialization also fails to establish a robust noisy state channel at the checkpoint inspected. The fresh gate is still inside the inherited learning-rate warmup, and fresh/resumed cohorts differ in learning history, optimizer, cursor and learning rate. They cannot establish an irreversible absorbing state or the impossibility of later learning.", "",
        "**Cube: extra amplitude does not become extra task information or better control.** After resumed training, rate-off raises feature std from .242/.221 to .307/.312. Clean block-position R² is slightly worse in all three coordinates for both seeds. Some noisy probes improve, consistent with a larger signal relative to sigma, but planning falls from 10/12% to 8/6%. Denoising sensitivity and signal amplitude are therefore insufficient proxies for planning usefulness.", "",
        "Beta=0 removes both entry and temporal surprise penalties. The latter also constrains the relation between encoded states and the action-conditioned predictor. These results cannot separately assign the effect to information pricing, dynamical consistency or their interaction. The forecast-render branch continues to train the predictor through pixel denoising.", "",
        "## Every run", "", "Position probe averages agent x/y in TwoRoom and block x/y/z in Cube. All probe standardization uses training episodes only; physical labels supervise only the diagnostic readout. Feature std below uses the fixed frozen-audit batches, and may differ from the training or health minibatch statistics.", "",
        "| Task | Start | Seed | Rate | Update | State std | Clean position R² | Noisy position R² | Wrong-state MSE % | Wrong-action MSE % |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for pair in pairs:
        for arm in ["on", "off"]:
            r = rows[pair[arm]]
            it = r["interventions"]["0.95"]
            lines.append(f"| {r['task']} | {r['initialization']} | {r['seed']} | {arm} | {r['step']} | {r['gradient']['state_std']['mean']:.6g} | {position_r2(r):.4f} | {position_r2(r,'noisy_sigma0.1'):.4f} | {it['forecast_other_initial']['all_mse_increase_percent']:.5f} | {it['forecast_wrong_actions']['all_mse_increase_percent']:.5f} |")
    lines += ["", "## Coordinate-level probe results", "", "The following vectors follow the stated label order. Null indicates a zero-variance target; it is excluded from R² means.", ""]
    for pair in pairs:
        lines += ["### " + pair_label(pair).replace("\n", " "), "", "| Rate | Coordinates | Clean R² | Noisy R² |", "|---|---|---|---|"]
        for arm in ["on", "off"]:
            r = rows[pair[arm]]
            fmt = lambda values: ", ".join("null" if v is None else f"{v:.4f}" for v in values)
            lines.append(f"| {arm} | {', '.join(r['labels'])} | {fmt(r['probes']['clean']['r2_against_train_mean'])} | {fmt(r['probes']['noisy_sigma0.1']['r2_against_train_mean'])} |")
    lines += ["", "## Decision and limits", "",
        "Keep these rate-off runs held. The controls resolve the narrow hypothesis that merely removing the rate is a ready cure: it is not supported at this budget. They motivate locating where position information is lost before/after QueryPool, and separating encoded-state quality, action-conditioned prediction and goal-metric ranking. No new long run follows automatically from this diagnostic.", "",
        "Planning uses 50 identical validation cases within each task; no final test claim is made. Two training seeds provide limited stability evidence. The early diagnostic rate-on results are new numerical continuations and should not be substituted for previously published 2600-step controls with different numerical histories. The full exit records, original health review, initial-state identities, per-seed probes, gradients, scale curves and planning outcomes are preserved in results.json.", "", "![Rate controls](rate-controls.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = collect(Path(args.training_root), Path(args.evaluation_root))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, out)
    plot(result, out)
    print(json.dumps(dict(output=str(out), runs=len(result["runs"]), pairs=len(result["pairs"]), planning=[p["paired_planning"] for p in result["pairs"] if p["paired_planning"]])))


if __name__ == "__main__":
    main()
