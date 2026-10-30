"""Aggregate frozen Cube oracle audits with paired case bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


GROUPS = ("nonoracle", "oracle_neighborhood", "all")
STAGES = ("physical_oracle", "encoded_oracle", "model")
RANKS = ("physical_encoded_rank", "encoded_predicted_rank", "physical_predicted_rank")


def mean_ci(values, indices):
    values = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    keep = np.isfinite(values)
    if not keep.any():
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    boot = values[indices]
    counts = np.isfinite(boot).sum(1)
    estimates = np.nansum(boot, axis=1)[counts > 0] / counts[counts > 0]
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return dict(mean=float(values[keep].mean()), lo=float(lo), hi=float(hi), n=int(keep.sum()))


def sample_metrics(record):
    value = {"coverage_success": float(record["candidate_any_success"])}
    for stage in STAGES:
        selected = record["selections"][stage]
        value[stage + "_success"] = float(selected["sustained_success"])
        value[stage + "_cost"] = selected["normalized_pose_cost"]
        value[stage + "_position_m"] = selected["position_error_m"]
        value[stage + "_orientation_deg"] = selected["orientation_error_deg"]
    for key in (*RANKS, "metric_regret", "signed_dynamics_regret", "action_effect_cosine", "centered_action_effect_mse"):
        value[key] = record.get(key)
    value["rollout_endpoint_mse"] = record["rollout_mse_by_block"][-1]
    value["candidate_endpoint_latent_variance"] = record["candidate_latent_variance_by_block"][-1]
    denom = value["candidate_endpoint_latent_variance"]
    value["centered_effect_error_ratio"] = value["centered_action_effect_mse"] / denom if denom > 1e-12 else None
    return value


def load_scores(directory):
    files = sorted(Path(directory).glob("scores-shard*.json"))
    if not files:
        raise FileNotFoundError(f"No score shards: {directory}")
    records, metadata = {}, None
    for path in files:
        doc = json.loads(path.read_text())
        identity = {k: doc[k] for k in ["checkpoint", "checkpoint_weights_sha256", "manifest_sha256", "script_sha256"]}
        identity["pool"] = doc["args"]["pool"]
        if metadata is None:
            metadata = identity
        assert identity == metadata, f"Inconsistent shard metadata: {path}"
        for record in doc["records"]:
            key = (int(record["case"]), record["group"])
            assert key not in records, f"Duplicate case/group in {directory}: {key}"
            records[key] = record
    return records, metadata


def readable(stat, scale=1, digits=2):
    if stat["mean"] is None:
        return "undefined"
    return f'{stat["mean"]*scale:.{digits}f} [{stat["lo"]*scale:.{digits}f}, {stat["hi"]*scale:.{digits}f}]'


def make_plots(out, summary, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#8797a7", "#26845f", "#e4a13b", "#346cbb"]
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(labels)*1.75), 12), sharex=True, constrained_layout=True)
    for ax, group in zip(axes, GROUPS):
        x = np.arange(len(labels))
        for j, (metric, name) in enumerate([
            ("coverage_success", "Any candidate succeeds"),
            ("physical_oracle_success", "Physical endpoint selector"),
            ("encoded_oracle_success", "Actual-embedding selector"),
            ("model_success", "Predicted-embedding selector"),
        ]):
            stats = [summary[label][group][metric] for label in labels]
            means = np.array([s["mean"] for s in stats]) * 100
            low = np.array([s["lo"] for s in stats]) * 100
            high = np.array([s["hi"] for s in stats]) * 100
            ax.bar(x+(j-1.5)*.19, means, width=.18, color=colors[j], label=name,
                   yerr=np.array([means-low, high-means]), capsize=2)
        ax.set_title(group.replace("_", " "))
        ax.set_ylabel("25-step sustained success (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper right", ncol=2, fontsize=9)
    axes[-1].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    fig.suptitle("Fixed candidate library: coverage, goal metric, rollout ranking\n95% case bootstrap; frozen models; not the closed-loop benchmark", fontsize=14)
    for ext in ["png", "svg"]:
        fig.savefig(out/f"oracle-selection.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(labels)*2), 5.5), sharey=True, constrained_layout=True)
    rank_names = ["Physical vs encoded", "Encoded vs predicted", "Physical vs predicted"]
    for ax, group in zip(axes, GROUPS):
        x = np.arange(len(labels))
        for j, key in enumerate(RANKS):
            stats = [summary[label][group][key] for label in labels]
            means = np.array([np.nan if s["mean"] is None else s["mean"] for s in stats])
            low = np.array([np.nan if s["lo"] is None else s["lo"] for s in stats])
            high = np.array([np.nan if s["hi"] is None else s["hi"] for s in stats])
            ax.errorbar(x+(j-1)*.17, means, yerr=np.array([means-low, high-means]),
                        fmt="o", color=colors[j+1], label=rank_names[j], capsize=3)
        ax.set_title(group.replace("_", " "))
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="#666666", linewidth=.8)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Mean per-case Spearman correlation")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Which action ordering fails? 95% case bootstrap intervals", fontsize=14)
    for ext in ["png", "svg"]:
        fig.savefig(out/f"oracle-ranks.{ext}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", action="append", required=True, help="LABEL=DIRECTORY; repeat for each checkpoint")
    parser.add_argument("--reference", required=True, help="Label used for paired differences")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261030)
    a = parser.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    data, meta = {}, {}
    for item in a.scores:
        label, directory = item.split("=", 1)
        assert label not in data, label
        data[label], meta[label] = load_scores(directory)
    assert a.reference in data
    labels = list(data)
    keys = set(data[a.reference])
    identity = (meta[a.reference]["manifest_sha256"], meta[a.reference]["pool"])
    for label in labels:
        assert set(data[label]) == keys, f"Unequal evaluated case sets: {label}"
        assert (meta[label]["manifest_sha256"], meta[label]["pool"]) == identity, f"Different candidate pool: {label}"
    cases = sorted({case for case, _ in keys})
    assert all((case, group) in keys for case in cases for group in GROUPS)
    boot = np.random.default_rng(a.seed).integers(0, len(cases), size=(a.bootstrap, len(cases)))
    summary, paired, raw = {}, {}, {}
    for label in labels:
        summary[label], raw[label] = {}, {}
        for group in GROUPS:
            metrics = [sample_metrics(data[label][(case, group)]) for case in cases]
            raw[label][group] = {key: [m[key] for m in metrics] for key in metrics[0]}
            summary[label][group] = {key: mean_ci(values, boot) for key, values in raw[label][group].items()}
    for label in labels:
        paired[label] = {}
        for group in GROUPS:
            paired[label][group] = {}
            for key, values in raw[label][group].items():
                reference = raw[a.reference][group][key]
                delta = [None if x is None or y is None else x-y for x, y in zip(values, reference)]
                paired[label][group][key] = mean_ci(delta, boot)
    result = dict(summary=summary, paired_difference_from=a.reference, paired_differences=paired,
                  cases=cases, metadata=meta, bootstrap_replicates=a.bootstrap, bootstrap_seed=a.seed,
                  scope="Conditional on fixed trained weights and fixed candidate library. Paired case resampling; no inference over training seeds. Recorded-action oracle families are not deployable policies. 25-step first-plan diagnostic, not 50-step MPC benchmark.")
    (out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Cube fixed-candidate mechanism audit", "", result["scope"], "",
             f"Matched cases: {len(cases)}. Intervals: 95% percentile case bootstrap ({a.bootstrap} draws).", "",
             "Success is the strict position + 24-symmetry orientation + hold criterion. The physical selector minimizes endpoint normalized pose error, so its success need not equal the library's any-success coverage.", ""]
    for group in GROUPS:
        lines += ["## "+group.replace("_", " "), "", "| Model | Any success % | Physical selector % | Encoded selector % | Model selector % |", "|---|---:|---:|---:|---:|"]
        for label in labels:
            s = summary[label][group]
            lines.append("| "+label+" | "+" | ".join(readable(s[k], 100, 1) for k in ["coverage_success", "physical_oracle_success", "encoded_oracle_success", "model_success"])+" |")
        lines += ["", "| Model | Physical / encoded rank | Encoded / predicted rank | Metric regret | Signed dynamics regret | Model success delta vs reference (pp) |", "|---|---:|---:|---:|---:|---:|"]
        for label in labels:
            s = summary[label][group]
            cells = [readable(s[k]) for k in ["physical_encoded_rank", "encoded_predicted_rank", "metric_regret", "signed_dynamics_regret"]]
            cells.append(readable(paired[label][group]["model_success"], 100, 1))
            lines.append("| "+label+" | "+" | ".join(cells)+" |")
        lines += [""]
    lines += ["![Selection decomposition](oracle-selection.png)", "", "![Action ranking](oracle-ranks.png)", "",
              "Interpretation requires the collection report's duplicate-state and recorded-action replay audits. Inspect all per-case failures before assigning a mechanism. Candidate coverage, metric alignment and rollout fidelity are separate requirements; this analysis does not establish a new algorithm or its novelty.", ""]
    (out/"RESULTS.md").write_text("\n".join(lines))
    make_plots(out, summary, labels)
    print(json.dumps(dict(models=labels, cases=len(cases), output=str(out))), flush=True)


if __name__ == "__main__":
    main()
