#!/usr/bin/env python3
"""Publish a small summary of completed Carrier600/2000 reports; no model loads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASKS = ("tworoom", "cube")
ARMS = ("identity", "unconstrained", "bounded")
NAMES = {"tworoom": "TwoRoom", "cube": "Cube", "identity": "F identity",
         "unconstrained": "F free", "bounded": "F bounded"}


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text())


def collect(base):
    learn_path = base / "reports/carrier-learn2000-20261031/results.json"
    diag_path = base / "reports/carrier-diagnostics-20261031/results.json"
    learn, diag = read(learn_path), read(diag_path)
    assert learn["completed_endpoint_count"] == 96
    assert diag["completed_endpoint_count"] == 34
    initial_path = Path(learn["initial600_results"]["path"])
    assert artifact(initial_path)["sha256"] == learn["initial600_results"]["sha256"]
    trajectories = []
    for x in learn["trajectories"]:
        assert x["steps"] == [600, 2000] and x["n_cases"] == 50
        trajectories.append(dict(task=x["task"], arm=x["arm"], seed=x["seed"],
            steps=x["steps"], success_rates=x["success_rates"], n_cases=x["n_cases"],
            certificate2000=x["chart_certificates"][-1]))
    assert {(x["task"], x["arm"], x["seed"]) for x in trajectories} == {
        (t, a, s) for t in TASKS for a in ARMS for s in [1, 2]}
    tf = [{k: x[k] for k in ["task", "seed", "method", "step", "success_rate"]}
          for x in learn["baselines"] if x["method"] == "temporal-forecast"]
    assert len(tf) == 4
    released = []
    primary = {}
    for task in TASKS:
        path = base / ("runs/hardtasks_gate600_20261030/"
                       f"reference-{task}-released-lewm-strict-validation/metrics.json")
        ref = read(path)
        sig = next(x["planning_signature"] for x in learn["rows"] if x["task"] == task)
        assert ref["manifest_sha256"] == sig["planning_manifest_sha256"]
        assert ref["pairs"] == sig["planning_pairs"]
        for key in ["solver", "plan", "protocol"]:
            assert ref[key] == sig[key]
        assert len(ref["metrics"]["episode_successes"]) == 50
        released.append(dict(task=task, success_rate=ref["metrics"]["success_rate"],
            source=artifact(path), normalization=ref["normalization"],
            same_ordered_cases_and_planner=True,
            scope="Released checkpoint; unmatched training budget, data exposure and capacity."))
        primary[task] = sig["planning_manifest_sha256"]
    fresh = []
    for x in diag["rows"]:
        if x["case_set"] == "fresh" and x["metric"] == "identity":
            fresh.append({k: x.get(k) for k in
                          ["task", "family", "arm", "seed", "success_rate"]})
    assert len(fresh) == 14
    metric = []
    for case in ["primary", "fresh"]:
        for arm in ["identity", "bounded"]:
            for seed in [1, 2]:
                rows = [x for x in diag["rows"] if x["task"] == "cube"
                        and x["case_set"] == case and x["family"] == "carrier"
                        and x["arm"] == arm and x["seed"] == seed]
                rates = {x["metric"]: x["success_rate"] for x in rows}
                assert set(rates) == {"identity", "observation", "increment"}
                metric.append(dict(case_set=case, arm=arm, seed=seed,
                                   success_rates=rates))
    comparisons = [x for x in learn["comparison_summaries"]
                   if x["reference"] in ["temporal-forecast", "unconstrained"]]
    case_sets = {k: {q: v[q] for q in ["manifest_sha256", "policy_seed", "scope"]}
                 for k, v in diag["case_sets"].items()}
    for key, val in diag["case_sets"].items():
        if val.get("fresh_selection"):
            select = val["fresh_selection"]
            assert select["exact_pair_overlap_count"] == 0
            case_sets[key]["overlap"] = {q: select[q] for q in
                ["episode_overlap_count", "exact_pair_overlap_count"]}
    return dict(source_commit=learn["source_commit"],
        inputs=dict(learn2000=artifact(learn_path), diagnostics600=artifact(diag_path),
                    initial600=artifact(initial_path)),
        full_arrays_location="Original results.json files remain on EFS; omitted here.",
        completed_endpoints=dict(learn2000=96, diagnostics600=34),
        primary_manifest_sha256=primary, trajectories=trajectories,
        temporal_forecast2000=tf, released_primary_reference=released,
        comparisons2000=comparisons, fresh600=fresh, cube_metric600=metric,
        case_sets=case_sets, limitations=learn["limitations"],
        next_action="Eight F and four TF continuations to6000 are authorized; this report does not assert launch or completion.",
        report_script=artifact(__file__))


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = {"identity": "#5B6674", "unconstrained": "#D17600", "bounded": "#0072B2"}
    markers = {"identity": "o", "unconstrained": "^", "bounded": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8))
    for row, task in enumerate(TASKS):
        ref = next(x for x in summary["released_primary_reference"] if x["task"] == task)
        for col, seed in enumerate([1, 2]):
            ax = axes[row, col]
            ax.set_title(f"{NAMES[task]} · training seed {seed}", loc="left", fontsize=12)
            ax.axhline(ref["success_rate"], color="#8B8793", ls=(0, (5, 4)), lw=1.4, zorder=1)
            ax.text(620, ref["success_rate"] + (2.2 if task == "tworoom" else .7),
                    f"Released LeWM: {ref['success_rate']:g}%", color="#746E7B", fontsize=9)
            for arm in ARMS:
                x = next(x for x in summary["trajectories"]
                         if (x["task"], x["arm"], x["seed"]) == (task, arm, seed))
                ax.plot(x["steps"], x["success_rates"], color=colors[arm], marker=markers[arm],
                        markersize=7, lw=2, markeredgecolor="white", markeredgewidth=.6)
            tf = next(x for x in summary["temporal_forecast2000"]
                      if (x["task"], x["seed"]) == (task, seed))
            ax.scatter([2000], [tf["success_rate"]], marker="D", s=43, color="#B34366",
                       edgecolor="white", linewidth=.6, zorder=5)
            ax.set_xlim(480, 2120)
            ax.set_xticks([600, 2000], ["600", "2,000"])
            ax.set_ylim(-2 if task == "tworoom" else -1, 100 if task == "tworoom" else 36)
            ax.set_yticks(range(0, 101, 20) if task == "tworoom" else range(0, 36, 5))
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Planning success (%)")
            ax.grid(axis="y", color="#E3E5E8", lw=.7)
            ax.set_axisbelow(True)
    handles = [Line2D([], [], color=colors[a], marker=markers[a], lw=2, label=NAMES[a]) for a in ARMS]
    handles += [Line2D([], [], color="#B34366", marker="D", ls="", label="TF at 2,000 updates"),
                Line2D([], [], color="#8B8793", ls="--", label="Released LeWM (different budget)")]
    fig.suptitle("Carrier F: learning progress on two tasks", x=.07, ha="left", y=.985, fontsize=16)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3,
               frameon=False, fontsize=10)
    fig.subplots_adjust(top=.82, bottom=.22, left=.08, right=.97, hspace=.4, wspace=.22)
    fig.text(.08, .04,
        "50 fixed validation cases per task; identical planner. Each training seed is shown separately. Rows use different y-scales.\n"
        "Released LeWM: unmatched training exposure, capacity and normalization. TF: same updates; F changes frozen/trainable capacity.\n"
        "Segments connect two evaluated checkpoints only; intermediate training performance was not measured.",
        fontsize=9, color="#4F5660", linespacing=1.6)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output / f"progress2000.{ext}", dpi=190, facecolor="white")
    plt.close(fig)


def pair(rows, predicate):
    selected = sorted([x for x in rows if predicate(x)], key=lambda x: x["seed"] or 0)
    assert selected
    return " / ".join(f"{x['success_rate']:g}" for x in selected)


def report(s):
    lines = ["# Carrier 600→2000：完成结果", "",
        "TwoRoom 的F家族随训练明显提高：三种F在2000步均达到70–82%，明显超过同更新数TF的2/6%。Cube只有8–16%，尚无跨任务性能突破。以下保留两seed，不以平均数掩盖差异。", "",
        "![所有seed与同更新数TF](progress2000.png)", "",
        "| 任务 | 方法 | seed1：600→2000 | seed2：600→2000 |", "|---|---|---:|---:|"]
    for task in TASKS:
        for arm in ARMS:
            row = []
            for seed in [1, 2]:
                x = next(x for x in s["trajectories"] if (x["task"], x["arm"], x["seed"]) == (task, arm, seed))
                row.append("→".join(f"{v:g}" for v in x["success_rates"]))
            lines.append(f"| {NAMES[task]} | {NAMES[arm]} | {row[0]} | {row[1]} |")
        rates = [next(x["success_rate"] for x in s["temporal_forecast2000"]
                      if (x["task"], x["seed"]) == (task, seed)) for seed in [1, 2]]
        lines.append(f"| {NAMES[task]} | TF，2000步 | {rates[0]:g} | {rates[1]:g} |")
    lines += ["", "数值均为成功率%。每任务同50个固定validation cases、同规划器；公开LeWM在同病例上为TwoRoom72%、Cube30%，但训练预算、数据暴露、容量与归一化不同，虚线仅作参考，不是匹配预算复现。F/TF更新数匹配，但F额外8320参数、冻结encoder/pool、使用128个训练帧校准，训练计算与优化器状态不同。", "",
        "TwoRoom bound−free分别为+2/−2个百分点，均值0，配对case bootstrap 95%区间[−11,+11]；Cube为−2/+4，均值+1，区间[−5,+7]。约束真实存在，但尚无性能优势证据。区间仅反映固定两训练seed下的病例不确定性，未涵盖seed不确定性或多重比较。", "",
        "TwoRoom free的全局残差Lipschitz上界为1.2711/1.3238，已不满足小于1的充分可逆性证书；这不证明映射不可逆。bound为0.499999/0.499883，仍有严格全局双Lipschitz保证（下界约0.5000，上界约1.5000）。Cube free为0.9459/0.9225，bound为0.4966/0.4783。", "",
        "## 600步fresh病例确认", "",
        "| 任务 | F identity | F bounded | TF | 公开LeWM单checkpoint |",
        "|---|---:|---:|---:|---:|"]
    for task in TASKS:
        values = [pair(s["fresh600"], lambda x, arm=arm: x["task"] == task and x["family"] == "carrier" and x["arm"] == arm)
                  for arm in ["identity", "bounded"]]
        values += [pair(s["fresh600"], lambda x, family=family: x["task"] == task and x["family"] == family)
                   for family in ["temporal-forecast", "released-lewm"]]
        lines.append(f"| {NAMES[task]} | " + " | ".join(values) + " |")
    lines += ["", "斜线两侧为seed1/2；这里只评估600步模型。新50例与primary精确(episode,start,goal)重合为0，但仍来自相同validation pool，episode重合TwoRoom34、Cube1；不是独立test set。病例抽样seed与policy seed均预先固定为2026103101，不可将primary/fresh差异只归因为病例变化。", "",
        "## Cube 600步规划度量控制", "",
        "| 病例 | 方法 | native | observation | increment |", "|---|---|---:|---:|---:|"]
    for case in ["primary", "fresh"]:
        for arm in ["identity", "bounded"]:
            values = []
            for metric in ["identity", "observation", "increment"]:
                values.append(" / ".join(f"{next(x['success_rates'][metric] for x in s['cube_metric600'] if (x['case_set'], x['arm'], x['seed']) == (case, arm, seed)):g}" for seed in [1, 2]))
            lines.append(f"| {case} | {NAMES[arm]} | " + " | ".join(values) + " |")
    lines += ["", "改度量未跨两seed、两病例集一致提高成功率；这里仅干预controller cost，模型权重与sample-level训练目标不变。", "",
        "已完成96个2000阶段端点与34个diagnostics端点均exit0。下一阶段8个F与4个TF到6000步已获授权；本报告不声明已启动或完成。", "",
        "完整数组保留EFS原报告；小型[summary.json](summary.json)记录原results SHA256、全部两seed数值及证书。", "",
        "来源：", ""]
    for name, src in s["inputs"].items():
        lines.append(f"- {name}: `{src['path']}`，SHA256 `{src['sha256']}`")
    lines += ["", f"冻结训练源码：`{s['source_commit']}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/efs/prescottsun/SPT-JEPA"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.base / "reports/carrier-progress2000-20261031"
    out.mkdir(parents=True, exist_ok=True)
    summary = collect(args.base)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (out / "RESULTS.md").write_text(report(summary))
    plot(summary, out)
    print(json.dumps({"output": str(out), "artifacts": [artifact(p) for p in sorted(out.iterdir()) if p.is_file()]}, indent=2))


if __name__ == "__main__":
    main()
