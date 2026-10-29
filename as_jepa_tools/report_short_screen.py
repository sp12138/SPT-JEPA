"""Matched first-epoch diagnostic screen; no automatic long-budget continuation."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    root, out = Path(a.campaign), Path(a.output)
    protocol, plan = read(root / "protocol.json"), read(root / "plan.json")
    evidence, rows, arrays, identities = {}, {}, {}, {}
    pending = []
    for job in plan["jobs"]:
        ex = root / "exits" / (job["id"] + ".json")
        if not ex.exists():
            pending.append(job["id"])
            continue
        exit_record = read(ex)
        assert exit_record["returncode"] == 0, (job["id"], exit_record)
        evidence[job["id"]] = dict(job=job, exit=exit_record)
    if pending:
        raise RuntimeError("Incomplete first-stage jobs: " + ", ".join(pending))
    methods = ["original"] + protocol["methods"]
    for method in methods:
        rows[method] = {}
        for seed in protocol["screen_training_seeds"]:
            values = {"planning": {}}
            for job in plan["jobs"]:
                if job["method"] != method or job["training_seed"] != seed:
                    continue
                folder = Path(job["output"])
                phase = job["phase"]
                if phase == "stage-training":
                    complete = read(folder / "stage-complete.json")
                    assert complete["stop_step"] == protocol["stage_steps"] == 12491
                    assert complete["scheduler_total_steps"] == 124910
                    values["stage"] = complete
                    values["stage_execution"] = evidence[job["id"]]["exit"]
                    size = read(folder / "model_size.json")
                    assert size["total"] == protocol["same_model_parameters"] == 51097134
                    values["model_size"] = size
                elif phase == "screen-planning":
                    x = read(folder / "metrics.json")
                    execution = read(folder / "execution.json")
                    assert execution["returncode"] == 0 and execution["episodes"] == 50 and not execution["bf16"]
                    assert len(x["metrics"]["episode_successes"]) == 50
                    score = 100 * sum(x["metrics"]["episode_successes"]) / 50
                    assert abs(score - x["metrics"]["success_rate"]) < 1e-9
                    assert x["split"] == "heldout" and x["float32_matmul_precision"] == "highest"
                    es = job["evaluation_seed"]
                    sig = (x["episodes"], x["start_steps"])
                    key = ("planning", es)
                    if key not in identities:
                        identities[key] = sig
                    assert identities[key] == sig, (job["id"], "Reset identity differs")
                    values["planning"][str(es)] = dict(metrics=x, execution=execution)
                elif phase == "screen-quality":
                    x = read(folder / "metrics.json")
                    assert x["cases"] == 64 and x["samples"] == 1 and x["sampler_steps"] == 50
                    values["quality"] = x
                    cases = read(folder / "cases.json")
                    sig = [(v["episode"], v["start"]) for v in cases]
                    if "quality" not in identities:
                        identities["quality"] = sig
                    assert identities["quality"] == sig
                    with np.load(folder / "per-case-metrics.npz", allow_pickle=False) as z:
                        for k in z.files:
                            v = z[k]
                            assert v.dtype.kind in "fibu" and np.isfinite(v).all() and v.ndim <= 3
                            arrays[f"{method}/seed{seed}/" + k] = v
                elif phase == "screen-counterfactual":
                    x = read(folder / "counterfactual.json")
                    assert x["cases"] == 16 and len(x["details"]) == 16
                    sig = [(v["episode"], v["start"], v["donor_raw_rows"]) for v in x["details"]]
                    if "cf" not in identities:
                        identities["cf"] = sig
                    assert identities["cf"] == sig
                    values["counterfactual"] = x
                elif phase == "screen-collapse":
                    values["collapse"] = read(folder / "metrics.json")
            assert len(values["planning"]) == 2
            scores = [values["planning"][str(es)]["metrics"]["metrics"]["success_rate"] for es in protocol["planning_eval_seeds"]]
            values["summary"] = dict(
                p0_values=scores, p0_mean=statistics.mean(scores),
                h7_agent_copy=values["quality"]["h7_agent_ratio"],
                h7_moving_block_copy=values["quality"]["h7_moving_block_ratio"],
                fvd64=values["quality"]["fvd"]["fvd"],
                cf_percent=100 * values["counterfactual"]["action_identification_accuracy"],
                block_cf_percent=100 * values["counterfactual"]["block_action_identification_accuracy"],
                posterior_over_anchor=values["collapse"]["posterior_over_anchor"],
                collapsed=values["collapse"]["collapsed"], active_dims=values["collapse"]["active_dims"])
            rows[method][str(seed)] = values
    effects = {}
    for method in methods[1:]:
        effects[method] = {}
        for seed in ["1", "2"]:
            x, y = rows[method][seed]["summary"], rows["original"][seed]["summary"]
            effects[method][seed] = {"p0_delta_pp": x["p0_mean"] - y["p0_mean"],
                                     "cf_delta_pp": x["cf_percent"] - y["cf_percent"],
                                     "h7_agent_ratio_delta": x["h7_agent_copy"] - y["h7_agent_copy"],
                                     "h7_moving_block_ratio_delta": x["h7_moving_block_copy"] - y["h7_moving_block_copy"]}
    record = dict(first_stage_complete=True, automatic_long_continuation=False,
                  source_commit=a.source_commit, reporter_sha256=sha(__file__),
                  protocol=protocol, endpoints=evidence, methods=rows, paired_effects=effects,
                  all_planning_quality_counterfactual_case_identities_match=True,
                  limits=["Training seeds1/2 were selected for diagnosis; this is not final cross-seed confirmation.",
                          "All checkpoints are exactly12491updates under the original124910step LR schedule.",
                          "Two evaluation seeds are test conditions, not two extra training seeds.",
                          "FVD64 and16counterfactual cases are diagnostic; no full generation gate or benchmark claim.",
                          "Weak early results can be inconclusive; stop/continue must use task evidence, trajectories and cost.",
                          "No human gallery signoff or improvement in model-size-matched final performance is inferred."])
    assert not out.exists(), "Preserve previous frozen report"
    out.mkdir(parents=True)
    (out / "short-screen-results.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    np.savez_compressed(out / "short-screen-per-case.npz", **arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Original", "Lower beta", "Temporal noise", "Forecast render", "Combined"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(len(methods))
    for idx, seed in enumerate(["1", "2"]):
        offset = -.18 if idx == 0 else .18
        color = ["#4e79a7", "#e09b48"][idx]
        for ax, key, title in zip(axes, ["p0_mean", "h7_agent_copy", "cf_percent"],
                                  ["Paired P0 success (%)", "h7 agent/copy (lower)", "Action identification (%)"]):
            vals = [rows[m][seed]["summary"][key] for m in methods]
            bars = ax.bar(x + offset, vals, .34, color=color, label="training seed" + seed)
            ax.bar_label(bars, fmt="%.2f" if key == "h7_agent_copy" else "%.0f", padding=3, fontsize=8)
            ax.set_xticks(x, labels, rotation=28, ha="right")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            if key != "h7_agent_copy":
                ax.set_ylim(0, 108)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("One-epoch diagnostic screen: matched models, data and update budget")
    fig.text(.5, .01, "Selected training seeds1/2;100P0 episodes/model,64generation clips,16counterfactual cases. No automatic long continuation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for ext in ["png", "svg"]:
        p = out / ("short-screen-comparison." + ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        if ext == "svg":
            p.write_text("\n".join(v.rstrip() for v in p.read_text().splitlines()).rstrip() + "\n")
    lines = ["# 一epoch短线筛选", "",
             "全部模型固定为51,097,134参数、全数据、batch128；从可恢复状态补到12491更新，保留124910步学习率日程。原算法直接使用已保存的同预算checkpoint。", "",
             "seed1/2是有意选择的诊断种子，本表不是正式多种子benchmark。旧16运行均已暂停，短线结果不会自动触发长跑。", "",
             "| 方法 | 训练种子 | P0原评测42/43 % | 均值 % | h7 agent/copy ↓ | h7 moving block/copy ↓ | 动作识别 % | block动作识别 % |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in methods:
        for seed in ["1", "2"]:
            s = rows[method][seed]["summary"]
            scores = " / ".join(f"{v:.0f}" for v in s["p0_values"])
            lines.append(f"| {method} | {seed} | {scores} | {s['p0_mean']:.1f} | {s['h7_agent_copy']:.3f} | {s['h7_moving_block_copy']:.3f} | {s['cf_percent']:.2f} | {s['block_cf_percent']:.2f} |")
    lines += ["", "![Matched short-budget comparison](short-screen-comparison.png)", "",
              "规划起点、目标及动作反事实案例已逐例匹配。完整逐例规划/反事实/生成数值及运行身份见[JSON](short-screen-results.json)和[数值NPZ](short-screen-per-case.npz)。FVD64只作为诊断，不宣称完整生成质量。", "",
              "继续/暂缓决定须结合两种子的任务差异、生成和动作信号另行记录。早期较弱不构成方法普遍失败的证明。", ""]
    (out / "SHORT-SCREEN.md").write_text("\n".join(lines).rstrip() + "\n")
    print(json.dumps({"screen_complete": True, "model_checkpoints": 10, "paired_effects": effects, "automatic_long_continuation": False}))


if __name__ == "__main__":
    main()
