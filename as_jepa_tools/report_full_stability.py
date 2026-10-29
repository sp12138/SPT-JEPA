"""Report independent full-data replications without treating eval seeds as train seeds."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def stats(values):
    return dict(values=values, mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None,
                minimum=min(values), maximum=max(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    root, out = Path(a.campaign), Path(a.output)
    protocol = read(root / "protocol.json")
    reference = read(protocol["reference_config"])
    byseed = {row["seed"]: row for row in protocol["replications"]}
    primary = protocol["primary_planning_eval_seeds"]
    confirmation = protocol["confirmation_planning_eval_seeds"]
    seeds = protocol["training_seeds"]
    rows, signatures, pending, all_curves = {}, {}, [], {}
    expected_size = None
    for seed in seeds:
        train = Path(protocol["reference_config"]).parent if seed == 0 else Path(byseed[seed]["config"]).parent
        cfg = read(train / "config.json")
        expected = dict(reference, output=str(train), seed=seed)
        assert cfg == expected, (seed, "Training configuration mismatch")
        status = read(train / "status.json")
        if status.get("state") != "finished" or status.get("step") != protocol["training_steps"]:
            pending.append("training-seed" + str(seed))
            continue
        if seed:
            exit_record = read(root / "exits" / (byseed[seed]["id"] + ".json"))
            assert exit_record["returncode"] == 0, (seed, exit_record)
        else:
            exit_record = read(train.parent / "exits" / (train.name + ".json"))
            assert exit_record["returncode"] == 0
        size = read(train / "model_size.json")
        if expected_size is None:
            expected_size = size
        assert size == expected_size, (seed, size, expected_size)
        checkpoint = train / f"model-step{protocol['training_steps']:07d}"
        cfg_export = read(checkpoint / "config.json")
        assert cfg_export == dict(_target_="as_jepa_lewm.model.ASJepaLeWM", **reference["model"])
        weight_path = checkpoint / "weights.pt"
        values = {}
        for evalseed in primary + confirmation:
            ident = f"p0-trainseed{seed}-evalseed{evalseed}"
            path = root / ident
            if not (root / "exits" / (ident + ".json")).exists():
                pending.append(ident)
                continue
            exit_eval = read(root / "exits" / (ident + ".json"))
            execution, value = read(path / "execution.json"), read(path / "metrics.json")
            assert exit_eval["returncode"] == execution["returncode"] == 0, ident
            assert (path / "policy/weights.pt").samefile(weight_path), ident
            assert read(path / "policy/config.json") == cfg_export
            assert execution["episodes"] == 50 and execution["seed"] == evalseed and not execution["bf16"]
            assert value["split"] == "heldout" and value["solver_rng_seed"] == evalseed
            assert value["float32_matmul_precision"] == "highest"
            outcome = value["metrics"]["episode_successes"]
            assert len(outcome) == len(value["episodes"]) == len(value["start_steps"]) == 50
            calculated = 100 * sum(outcome) / 50
            assert math.isfinite(calculated) and abs(calculated - value["metrics"]["success_rate"]) < 1e-9
            sig = list(zip(value["episodes"], value["start_steps"]))
            if evalseed not in signatures:
                signatures[evalseed] = sig
            assert sig == signatures[evalseed], (ident, "Unpaired reset identities")
            values[str(evalseed)] = dict(success_rate=calculated, metrics=value, execution=execution, exit=exit_eval)
        curves = []
        for line in (train / "metrics.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            selected = {k: value[k] for k in ["step", "loss", "fm", "rate", "feature_std", "active_dims", "learning_rates"] if k in value}
            assert all(math.isfinite(float(selected[k])) for k in ["loss", "fm", "rate"])
            curves.append(selected)
        all_curves[str(seed)] = curves
        rows[str(seed)] = dict(training_config=cfg, source=protocol["reference_training_source"], status=status,
                               exit=exit_record, model_size=size, checkpoint=str(checkpoint),
                               weights_sha256=sha(weight_path), planning=values)
        if len(values) == len(primary + confirmation):
            for cohort, evals in [("primary", primary), ("confirmation", confirmation), ("combined", primary + confirmation)]:
                rows[str(seed)][cohort] = stats([values[str(s)]["success_rate"] for s in evals])
    if pending:
        raise RuntimeError("Required replications not complete: " + ", ".join(pending))
    across = {}
    for label, subset in [("new_training_seeds", seeds[1:]), ("all_training_seeds", seeds)]:
        across[label] = {cohort: stats([rows[str(s)][cohort]["mean"] for s in subset])
                         for cohort in ["primary", "confirmation", "combined"]}
    limits = protocol["stability_decision"]["clear_stable_high_performance"]
    new_primary = across["new_training_seeds"]["primary"]
    criteria = {
        "new_seed_primary_mean_ge85": new_primary["mean"] >= limits["new_seeds_primary_mean_at_least_percent"],
        "every_new_seed_primary_mean_ge80": new_primary["minimum"] >= limits["each_new_seed_primary_mean_at_least_percent"],
        "new_seed_primary_sample_std_le5": new_primary["sample_std"] <= limits["new_seeds_sample_std_at_most_percentage_points"],
        "all_training_finished_without_numerical_failure": True,
    }
    quality = {}
    for seed in seeds:
        folder = Path(protocol["reference_results"]) if seed == 0 else root / f"endpoint-trainseed{seed}" / "results"
        quality[str(seed)] = {"root": str(folder)}
        for key, rel in [("status", "status.json"), ("collapse", "collapse/metrics.json"),
                         ("quality", "quality/metrics.json"), ("counterfactual", "counterfactual/counterfactual.json")]:
            path = folder / rel
            if path.exists():
                quality[str(seed)][key] = read(path)
    collapse_complete = all("collapse" in q for q in quality.values())
    primary_clear = all(criteria.values())
    record = dict(
        complete_planning_replication=True, independent_training_seeds=len(seeds),
        original_selected_seed0_reported_separately=True, source_commit=a.source_commit,
        report_script_sha256=sha(__file__), protocol=protocol, model_size=expected_size,
        replications=rows, statistics_across_training_seeds=across,
        practical_primary_stability_checks=criteria,
        primary_practical_stability_pass=primary_clear,
        noisy_posterior_audits_all_available=collapse_complete,
        stability_interpretation="Primary practical checks pass; inspect confirmation and noisy-posterior audits before final interpretation." if primary_clear else "The predeclared primary practical stability criteria are not all met; retain every seed and diagnose before method changes.",
        reset_pairing_verified=True, same_exact_weight_files_used_by_planner=True,
        training_curves=all_curves, available_quality=quality,
        limits=["A practical stability screen over four new training seeds is not a formal population guarantee.",
                "Seed0 was selected using historical results; new seeds1-4 are reported separately.",
                "Evaluation seeds are paired test conditions, not independent training replications.",
                "Confirmation uses the same heldout episode split, not a new dataset.",
                "This reproduces three-epoch intermediate checkpoints under a ten-epoch LR schedule, not the full original benchmark.",
                "No human gallery signoff or equal-budget LeWM superiority is inferred."])
    assert not out.exists(), "Preserve frozen previous report"
    out.mkdir(parents=True)
    (out / "full-data-stability.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seeds))
    colors = ["#4b769b", "#649268"]
    for k, (cohort, evalseeds) in enumerate([("primary", primary), ("confirmation", confirmation)]):
        offset = (-.19 if k == 0 else .19)
        vals = [rows[str(s)][cohort]["mean"] for s in seeds]
        bars = ax.bar(x + offset, vals, .34, color=colors[k], label=cohort.title() + " cases")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        for j, seed in enumerate(seeds):
            pts = [rows[str(seed)]["planning"][str(e)]["success_rate"] for e in evalseeds]
            ax.scatter([j + offset] * len(pts), pts, s=18, color="#222222", alpha=.7, zorder=3)
    ax.set_xticks(x, ["0 (selected reference)", "1", "2", "3", "4"])
    ax.set_xlabel("Independent training seed")
    ax.set_ylabel("P0 success (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Full-data AS-JEPA: identical beta=.25 configuration at three epochs")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left")
    fig.text(.5, .01, "Dots: three evaluation seeds per model. Bars: mean per training seed; bad seeds are retained.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    figures = [("full-data-stability-planning", fig)]
    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for seed in seeds:
        vals = all_curves[str(seed)]
        steps = [v["step"] for v in vals]
        axes[0].plot(steps, [v["fm"] for v in vals], label=f"seed{seed}", linewidth=1)
        axes[1].plot(steps, [v["feature_std"] for v in vals], label=f"seed{seed}", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Flow-matching training loss")
    axes[1].set_ylabel("Feature standard deviation (diagnostic)")
    for ax2 in axes:
        ax2.set_xlabel("Optimizer updates")
        ax2.spines[["top", "right"]].set_visible(False)
    axes[1].legend(fontsize=8)
    fig2.suptitle("Training trajectories; loss and feature variance do not establish planning quality")
    fig2.tight_layout()
    figures.append(("full-data-stability-training", fig2))
    for stem, fig in figures:
        for ext in ["png", "svg"]:
            path = out / (stem + "." + ext)
            fig.savefig(path, dpi=180, bbox_inches="tight")
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
        plt.close(fig)
    table = ["# 全数据 beta=.25：独立训练种子复现", "",
             "固定模型、数据、batch128、sigma=.1和优化器；每个种子训练37473步，保留原124910步学习率日程。只改变训练随机种子和输出目录。", "",
             "seed0是历史选出的参考；新增seed1–4单独统计，不能把评测种子当作训练种子。", "",
             "| 训练种子 | 原评测42/43/44 % | 均值 % | 新评测1042/1043/1044 % | 均值 % |",
             "| --- | --- | ---: | --- | ---: |"]
    for seed in seeds:
        row = rows[str(seed)]
        show = lambda es: " / ".join(f"{row['planning'][str(e)]['success_rate']:.0f}" for e in es)
        table.append(f"| {seed} | {show(primary)} | {row['primary']['mean']:.2f} | {show(confirmation)} | {row['confirmation']['mean']:.2f} |")
    table += ["", "| 统计对象 | 原评测：跨训练种子均值 ± SD % | 最差训练种子 % | 新评测：均值 ± SD % | 最差训练种子 % |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for label, name in [("new_training_seeds", "新增seed1–4"), ("all_training_seeds", "全部seed0–4")]:
        x, y = across[label]["primary"], across[label]["confirmation"]
        table.append(f"| {name} | {x['mean']:.2f} ± {x['sample_std']:.2f} | {x['minimum']:.2f} | {y['mean']:.2f} ± {y['sample_std']:.2f} | {y['minimum']:.2f} |")
    table += ["", "预先声明的主要实用稳定性检查：" + ("全部通过。" if primary_clear else "未全部通过。"),
              "需结合新评测及含噪后验检查解释；有限种子样本不构成形式化总体保证。", "",
              "![Planning across training seeds](full-data-stability-planning.png)", "",
              "![Training trajectories](full-data-stability-training.png)", "",
              "[完整逐例规划、训练曲线、身份与统计](full-data-stability.json)。这仍是三epoch中期结果，未声称完成原始完整benchmark。", ""]
    (out / "FULL-DATA-STABILITY.md").write_text("\n".join(table).rstrip() + "\n")
    print(json.dumps({"complete_planning": True, "training_seeds": len(seeds), "statistics": across,
                      "primary_practical_stability_pass": primary_clear, "noisy_posterior_audits_all_available": collapse_complete}))


if __name__ == "__main__":
    main()
