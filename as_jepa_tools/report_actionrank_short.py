"""Summarize short objective adaptations against paired original continuations."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    root, out = Path(a.campaign), Path(a.output)
    protocol, plan = read(root / "protocol.json"), read(root / "plan.json")
    seeds = [str(s) for s in protocol["training_seeds"]]
    expected = [j for j in plan["jobs"] if j["phase"] in {"adaptation600", "planning", "health", "quality", "counterfactual"}]
    missing = [j["id"] for j in expected if not (root / "exits" / (j["id"] + ".json")).exists()]
    if missing:
        raise RuntimeError("Pending short trials: " + ", ".join(missing))
    rows, arrays, identity = {}, {}, {}
    for j in expected:
        e = read(root / "exits" / (j["id"] + ".json"))
        if e["returncode"]:
            raise RuntimeError("Trial failure requires a separate failure record: " + j["id"])
        method, seed, phase = j["method"], str(j["training_seed"]), j["phase"]
        row = rows.setdefault(method, {}).setdefault(seed, {"planning": {}})
        folder = Path(j["output"])
        if phase == "adaptation600":
            stage, size, status = read(folder / "stage-complete.json"), read(folder / "model_size.json"), read(folder / "status.json")
            assert stage["initial_step"] == 24982 and stage["stop_step"] == 25582
            assert stage["scheduler_total_steps"] == 124910 and size["total"] == 51097134
            row.update(stage=stage, training_status=status, model_size=size, training_exit=e,
                       weights_sha256=sha(folder / "model-step0025582/weights.pt"))
        elif phase == "planning":
            value, execution = read(folder / "metrics.json"), read(folder / "execution.json")
            assert execution["returncode"] == 0 and execution["episodes"] == 50 and not execution["bf16"]
            assert value["float32_matmul_precision"] == "highest" and value["split"] == "heldout"
            es = str(j["evaluation_seed"])
            sig = (value["episodes"], value["start_steps"])
            key = ("planning", es)
            if key not in identity:
                identity[key] = sig
            assert identity[key] == sig, j["id"]
            assert len(value["metrics"]["episode_successes"]) == 50
            row["planning"][es] = dict(metrics=value, execution=execution, exit=e)
        elif phase == "health":
            row["health"] = read(folder / "metrics.json")
        elif phase == "quality":
            value, cases = read(folder / "metrics.json"), read(folder / "cases.json")
            assert value["cases"] == 64 and value["samples"] == 1 and value["sampler_steps"] == 20
            sig = [(x["episode"], x["start"]) for x in cases]
            if "quality" not in identity:
                identity["quality"] = sig
            assert identity["quality"] == sig
            row["quality"] = value
            row["quality_cases"] = cases
            with np.load(folder / "per-case-metrics.npz", allow_pickle=False) as z:
                row["numeric_quality"] = {k: z[k].copy() for k in z.files}
                for key, value in row["numeric_quality"].items():
                    assert value.dtype.kind in "fibu" and np.isfinite(value).all() and value.ndim <= 3
                    arrays[f"{method}/seed{seed}/" + key] = value
        elif phase == "counterfactual":
            value = read(folder / "counterfactual.json")
            assert value["cases"] == 16
            sig = [(x["episode"], x["start"], x["donor_raw_rows"]) for x in value["details"]]
            if "cf" not in identity:
                identity["cf"] = sig
            assert identity["cf"] == sig
            row["counterfactual"] = value
    summaries = {}
    effects = {}
    for method, byseed in rows.items():
        summaries[method] = {}
        for seed, row in byseed.items():
            scores = [x["metrics"]["metrics"]["success_rate"] for _, x in sorted(row["planning"].items())]
            assert len(scores) == 2
            cf, q = row["counterfactual"], row["quality"]
            summaries[method][seed] = dict(p0_mean=statistics.mean(scores), p0_values=scores,
                h7_agent_copy=q["h7_agent_ratio"], h7_moving_block_copy=q["h7_moving_block_ratio"],
                cf_percent=100 * cf["action_identification_accuracy"],
                block_cf_percent=100 * cf["block_action_identification_accuracy"],
                collapsed=row["health"]["collapsed"],
                posterior_over_anchor=row["health"]["posterior_over_anchor"],
                training_seconds=row["training_exit"]["finished_at"] - row["training_exit"]["started_at"],
                last_seconds_per_step=row["training_status"]["seconds_per_step"])
            if method == "original":
                continue
            base = rows["original"][seed]
            deltas, episodes = [], []
            for es in sorted(row["planning"]):
                left, right = row["planning"][es]["metrics"], base["planning"][es]["metrics"]
                deltas.extend((np.asarray(left["metrics"]["episode_successes"], float) - np.asarray(right["metrics"]["episode_successes"], float)) * 100)
                episodes.extend(left["episodes"])
            quality_groups = [x["episode"] for x in row["quality_cases"]]
            effect = dict(p0_delta_pp=cluster_ci(deltas, episodes),
                h7_agent_px_delta=cluster_ci(row["numeric_quality"]["mean_agent"][:,6] - base["numeric_quality"]["mean_agent"][:,6], quality_groups))
            cf_delta = [(np.mean(x["correct"]) - np.mean(y["correct"])) * 100
                        for x, y in zip(cf["details"], base["counterfactual"]["details"])]
            effect["cf_delta_pp"] = cluster_ci(cf_delta, [x["episode"] for x in cf["details"]])
            effects.setdefault(method, {})[seed] = effect
    if "differential-flow" in rows:
        effects["differential-vs-shared-noise"] = {}
        for seed in seeds:
            left, right = summaries["differential-flow"][seed], summaries["shared-flow-noise"][seed]
            effects["differential-vs-shared-noise"][seed] = dict(p0_delta_pp=left["p0_mean"] - right["p0_mean"],
                h7_agent_copy_delta=left["h7_agent_copy"] - right["h7_agent_copy"], cf_delta_pp=left["cf_percent"] - right["cf_percent"])
    for byseed in rows.values():
        for row in byseed.values():
            row.pop("numeric_quality")
    result = dict(short_adaptation_complete=True, additional_updates=600, initial_updates=24982,
        all_model_parameters=51097134, all_objectives_sample_level=True,
        source_commit=a.source_commit, reporter_sha256=sha(__file__), protocol=protocol,
        summaries=summaries, paired_effects=effects, complete_endpoints=rows,
        automatic_long_continuation=False,
        limitations=["600update adaptation from an existing2epochcheckpoint; not from-scratch training or a stability result.",
            "Four existing mature training seeds; final claims need independent from-scratch training and unused evaluation cases.",
            "P0twoevalseeds,16CFcases and64generationclips are a screen, not the full original benchmark.",
            "Bootstrap intervals condition on these fixed weights and episode clusters; not uncertainty across training seeds.",
            "Candidate/control comparisons are exploratory with no multiple-testing correction; selected wins require independent confirmation.",
            "Forward values match the original rate/FM; semiaction ranking is not full-gradient original KL optimization.",
            "Same parameter/update count is not equal compute; report per-arm training time and extra model calls.",
            "Future states only enter training targets within the same video; no futureRGB is passed to inference."])
    assert not out.exists(), "Preserve existing frozen report"
    out.mkdir(parents=True)
    (out / "actionrank-short-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out / "actionrank-short-per-case.npz", **arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(protocol["methods"])
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, seed in enumerate(seeds):
        vals = [summaries[m][seed]["p0_mean"] for m in methods]
        bars = ax.bar(x + (i - (len(seeds) - 1) / 2) * .2, vals, .18, label="training seed" + seed)
        ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    ax.set_xticks(x, methods, rotation=35, ha="right")
    ax.set_ylabel("P0 success (%)")
    ax.set_ylim(0, 108)
    ax.set_title("600-update action ranking from the same two-epoch models")
    ax.legend(loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5,.01,"51.1M parameters; five settings including controls; four existing training seeds. Adaptation only.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png", "svg"]:
        path = out / ("actionrank-short-planning." + ext)
        fig.savefig(path,dpi=180,bbox_inches="tight")
        if ext == "svg":
            path.write_text("\n".join(x.rstrip() for x in path.read_text().splitlines()).rstrip()+"\n")
    lines = ["# Sample-level动作排序：600更新配对适应筛选", "",
        "全部方法从同一训练种子对应的原算法2epoch完整状态出发，只适应600更新。模型参数固定51,097,134，batch128，beta=.25，sigma=.1，学习率日程124910步。", "",
        "这不是从头训练稳定性，也不是划时代或新颖性的证明。包含原算法续训及必要对照，未修正多重比较；后续胜者必须独立确认。", "",
        "| 方法 | 训练种子 | P0均值 % | 相对原算法续训百分点 | h7 agent/copy ↓ | h7 moving block/copy ↓ | 动作识别 % | 训练秒数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        for seed in seeds:
            s = summaries[method][seed]
            delta = s["p0_mean"] - summaries["original"][seed]["p0_mean"]
            lines.append(f"| {method} | {seed} | {s['p0_mean']:.1f} | {delta:+.1f} | {s['h7_agent_copy']:.3f} | {s['h7_moving_block_copy']:.3f} | {s['cf_percent']:.2f} | {s['training_seconds']:.1f} |")
    lines += ["", "![Paired short adaptations](actionrank-short-planning.png)", "",
        "[完整逐例结果、配对episodebootstrap、检查点身份与成本](actionrank-short-results.json)、[生成逐例数值](actionrank-short-per-case.npz)。", "",
        "所有新运行均已停在600更新上限。根据两种子的规划/生成/动作结果及成本作继续决定，不自动延长整组。", ""]
    (out / "ACTIONRANK-SHORT.md").write_text("\n".join(lines).rstrip()+"\n")
    print(json.dumps({"short_adaptation_complete":True,"summaries":summaries,"automatic_long_continuation":False}))


if __name__ == "__main__":
    main()
