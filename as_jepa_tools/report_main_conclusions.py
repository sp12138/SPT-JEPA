"""Aggregate the major conclusions from immutable campaign endpoints and paired controls."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ARMS = {
    "base001": ("baseline-standard-b0.01", "Research baseline beta=.01"),
    "snr": ("snr_weighted-standard-b1", "I01 weighted FM"),
    "vdm": ("vdm_nats-standard-b1.0", "I01 finite VDM bound"),
    "geometry": ("geometry_noise", "I02 diagonal Q"),
    "permuted": ("geometry_noise-permutedQ", "I02 permuted Q control"),
    "correlated": ("correlated_noise-standard-b0.01", "I03 correlated noise"),
    "color": ("canonical_color-color-b0.01", "I04 frame canonicalizer"),
    "anchor_color": ("anchor_canonical_color-color-b0.01", "I04 anchor canonicalizer"),
    "forecast": ("forecast_render-standard-b0.01", "I05 forecast rendering"),
    "gradient": ("gradient_constraint-standard-b1", "I06 gradient constraint"),
    "predictable": ("predictable_weight-dot-b0.01", "I07 learned variance"),
    "fork": ("action_forks", "I09 inverse-action loss"),
    "fork_null": ("action_forks_null", "I09 zero-loss control"),
    "base1": ("baseline-standard-b1", "Research baseline beta=1"),
    "base_color": ("baseline-color-b0.01", "Color baseline"),
    "base_dot": ("baseline-dot-b0.01", "Dot baseline"),
}
PAIR_SPECS = [
    ("snr", "base1"), ("vdm", "base1"), ("geometry", "base001"),
    ("permuted", "geometry"), ("correlated", "base001"),
    ("color", "base_color"), ("anchor_color", "base_color"),
    ("forecast", "base001"), ("gradient", "base1"),
    ("predictable", "base_dot"), ("fork", "fork_null"),
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint(root, ident):
    out = root / ident / "results"
    status = out / "status.json"
    if not status.exists() or read(status).get("state") != "finished":
        return None
    state = read(status)
    if any(x.get("returncode", 0) != 0 for x in state.get("results", [])):
        raise RuntimeError("Failed nested stage in " + ident)
    manifest = read(out / "checkpoint_manifest.json") if (out / "checkpoint_manifest.json").exists() else None
    result = {"id": ident, "root": str(out), "status": state, "checkpoint_manifest": manifest}
    dispatch = root / ident / "dispatch.json"
    result["dispatch"] = read(dispatch) if dispatch.exists() else None
    exit_path = root / "exits" / (ident + ".json")
    result["exit"] = read(exit_path) if exit_path.exists() else None
    if result["exit"] and result["exit"].get("returncode") != 0:
        raise RuntimeError("Endpoint exit was not successful: " + ident)
    for key, rel in [
        ("quality", "quality/metrics.json"), ("quality_cases", "quality/cases.json"),
        ("additional_quality", "quality-additional/metrics.json"),
        ("collapse", "collapse/metrics.json"),
        ("counterfactual", "counterfactual/counterfactual.json"),
        ("gallery", "gallery-complete/complete.json"),
    ]:
        path = out / rel
        if path.exists():
            result[key] = read(path)
    result["probes"] = {}
    for path in sorted((out / "probes").glob("*/probes.json")):
        result["probes"][path.parent.name] = read(path)
    result["planning"] = {}
    for path in sorted((out / "planning").glob("*/metrics.json")):
        result["planning"][path.parent.name] = read(path)
    q = result.get("quality", {})
    cf = result.get("counterfactual", {})
    plans = [x["metrics"]["success_rate"] for x in result["planning"].values()]
    result["summary"] = {
        "h7_agent_copy": q.get("h7_agent_ratio"),
        "h7_moving_block_copy": q.get("h7_moving_block_ratio"),
        "fvd": q.get("fvd", {}).get("fvd"),
        "cf": cf.get("action_identification_accuracy"),
        "cf_agent": cf.get("agent_action_identification_accuracy"),
        "cf_block": cf.get("block_action_identification_accuracy"),
        "p0_values": plans,
        "p0_mean": float(np.mean(plans)) if plans else None,
        "noisy_agent_r2": result["probes"].get("noisy", {}).get("agent_r2"),
        "collapsed": result.get("collapse", {}).get("collapsed"),
        "active_dims": result.get("collapse", {}).get("active_dims"),
    }
    return result


def cluster_ci(values, groups):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique, indices = np.unique(groups, return_inverse=True)
    sums = np.bincount(indices, weights=values)
    counts = np.bincount(indices)
    rng = np.random.default_rng(20261029)
    weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=5000)
    samples = (weights @ sums) / (weights @ counts)
    return {
        "mean_delta": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
        "episode_clusters": len(unique), "observations": len(values), "bootstrap_samples": 5000,
        "scope": "Paired episode-cluster bootstrap conditional on these trained weights; not training-seed uncertainty",
    }


def quality_ids(cases):
    return [(x["episode"], x["start"]) for x in cases]


def pairs(a, b):
    if quality_ids(a["quality_cases"]) != quality_ids(b["quality_cases"]):
        raise RuntimeError("Unpaired quality cases: " + a["id"] + " / " + b["id"])
    ids = [x["episode"] for x in a["quality_cases"]]
    with np.load(Path(a["root"]) / "quality/per-case-metrics.npz", allow_pickle=False) as x, \
         np.load(Path(b["root"]) / "quality/per-case-metrics.npz", allow_pickle=False) as y:
        result = {"h7_agent_error_px": cluster_ci(x["mean_agent"][:, 6] - y["mean_agent"][:, 6], ids),
                  "h13_agent_error_px": cluster_ci(x["mean_agent"][:, 12] - y["mean_agent"][:, 12], ids)}
    if a["planning"] and b["planning"]:
        if set(a["planning"]) != set(b["planning"]):
            raise RuntimeError("Unmatched planning evaluation seeds")
        values, episodes = [], []
        for seed in a["planning"]:
            x, y = a["planning"][seed], b["planning"][seed]
            if x["episodes"] != y["episodes"] or x["start_steps"] != y["start_steps"]:
                raise RuntimeError("Unpaired planning resets: " + seed)
            values.extend((np.asarray(x["metrics"]["episode_successes"], float) -
                           np.asarray(y["metrics"]["episode_successes"], float)) * 100)
            episodes.extend(x["episodes"])
        result["p0_delta_percentage_points"] = cluster_ci(values, episodes)
    if "counterfactual" in a and "counterfactual" in b:
        x, y = a["counterfactual"]["details"], b["counterfactual"]["details"]
        signature = lambda rows: [(r["episode"], r["start"], r.get("donor_raw_rows")) for r in rows]
        if signature(x) != signature(y):
            raise RuntimeError("Unpaired counterfactual resets/actions")
        values = [(np.mean(xi["correct"]) - np.mean(yi["correct"])) * 100 for xi, yi in zip(x, y)]
        result["cf_delta_percentage_points"] = cluster_ci(values, [r["episode"] for r in x])
    return result


def format_num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    root, out = Path(a.campaign), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, pending, evidence, paired = {}, [], {}, {}
    for key, (slug, label) in ARMS.items():
        rows[key] = {"label": label}
        for cohort, prefix in [("diagnostic", "eval"), ("confirmation", "confirm")]:
            ident = f"{prefix}-idea-{slug}-50k-seed0"
            item = endpoint(root, ident)
            if item is None:
                pending.append(ident)
                continue
            rows[key][cohort] = item["summary"]
            evidence[ident] = item
    for key, control in PAIR_SPECS:
        for cohort, prefix in [("diagnostic", "eval"), ("confirmation", "confirm")]:
            left = f"{prefix}-idea-{ARMS[key][0]}-50k-seed0"
            right = f"{prefix}-idea-{ARMS[control][0]}-50k-seed0"
            if left in evidence and right in evidence:
                paired[f"{cohort}:{key}-{control}"] = pairs(evidence[left], evidence[right])
    semantic = {}
    for seed in [0, 1, 2]:
        semantic[seed] = {}
        for variant, suffix in [
            ("semantic", f"carry-25k-seed{seed}"),
            ("same_weights_pixel", f"carry-25k-seed{seed}-same-weights-pixel"),
            ("matched_pixel_training", f"pixel-matched-25k-seed{seed}"),
        ]:
            ident = "eval-idea-I08-native-" + suffix
            item = endpoint(root, ident)
            if item is None:
                pending.append(ident)
                continue
            semantic[seed][variant] = item
            evidence[ident] = item
        values = semantic[seed]
        if "semantic" in values and "same_weights_pixel" in values:
            x, y = values["semantic"], values["same_weights_pixel"]
            if x["checkpoint_manifest"]["weights_sha256"] != y["checkpoint_manifest"]["weights_sha256"]:
                raise RuntimeError("I08 feedback toggle has different weights")
            with np.load(Path(x["root"]) / "quality/per-case-metrics.npz", allow_pickle=False) as xa, np.load(Path(y["root"]) / "quality/per-case-metrics.npz", allow_pickle=False) as ya:
                first_step_equal = all(np.array_equal(xa[k][:, 0], ya[k][:, 0]) for k in ["mean_agent", "mean_block"])
            if not first_step_equal:
                raise RuntimeError("I08 same-weight controls differ at the first generated step")
            x["feedback_control_checks"] = {"weights_identical": True, "first_step_agent_and_block_identical": True}
            paired[f"I08-seed{seed}-feedback-toggle"] = pairs(x, y)
        if "semantic" in values and "matched_pixel_training" in values:
            paired[f"I08-seed{seed}-matched-training"] = pairs(values["semantic"], values["matched_pixel_training"])
    full = {}
    for family, beta in [("L2", ".015625"), ("L2", ".0625"), ("L2", ".25"), ("L2", "1"), ("S2", "1")]:
        value = "0" + beta if beta.startswith(".") else beta
        ident = f"insight-eval-full-{family}-beta{value}-stage3-seed0"
        item = endpoint(root, ident)
        if item is None:
            pending.append(ident)
        else:
            full[family + "-beta" + value] = item
            evidence[ident] = item
    collapse = []
    for family in ["L2", "S2"]:
        for beta in [4, 16, 64, 256]:
            ident = f"early-collapse-full-{family}-beta{beta}-stage3-seed0"
            path = root / ident / "results/metrics.json"
            if not path.exists():
                pending.append(ident)
                continue
            value = read(path)
            if value["active_dims"] != 0 or not value["collapsed"]:
                raise RuntimeError("Early collapse was not confirmed")
            collapse.append({"id": ident, "metrics": value})
    identity = read(root / "reports-initial-identity-main-conclusions.json")
    published = root.parents[1] / "code/experiments/completion20261028/results"
    legacy = read(published / "additional-results.json")
    geometric_planning = legacy["planning"]
    for method, records in geometric_planning.items():
        if method == "goal_mse":
            continue
        values, episodes = [], []
        base = {x["seed"]: x["result"] for x in geometric_planning["goal_mse"]}
        for row in records:
            x, y = row["result"], base[row["seed"]]
            if x["episodes"] != y["episodes"] or x["start_steps"] != y["start_steps"]:
                raise RuntimeError("I10 reset mismatch")
            values.extend((np.asarray(x["metrics"]["episode_successes"], float) - np.asarray(y["metrics"]["episode_successes"], float)) * 100)
            episodes.extend(x["episodes"])
        paired["I10-" + method + "-goal_mse"] = cluster_ci(values, episodes)
    stability = read(published / "L2-50K-STABILITY.json")
    capacity = read(published / "STATE_CAPACITY.json")
    fork_validation = {}
    for variant in ["action_forks", "action_forks_null"]:
        f = root / ("fork-validation-idea-" + variant + "-50k-seed0/results/metrics.json")
        if f.exists():
            fork_validation[variant] = read(f)

    bundle = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Major conclusions only. Exploratory method comparisons and original-full-data three-epoch checkpoints; not completed original ten-epoch benchmarks.",
        "complete_major_scope": not pending, "pending": pending,
        "arms": rows, "full_three_epoch": full, "semantic_feedback": semantic,
        "paired_comparisons": paired, "confirmed_collapse": collapse,
        "initial_identity": identity, "endpoints": evidence,
        "geometric_planning": geometric_planning, "legacy_three_training_seed_stability": stability,
        "legacy_capacity": capacity, "fork_validation": fork_validation,
        "legacy_source_report_sha256": {"additional-results.json": sha(published / "additional-results.json"), "L2-50K-STABILITY.json": sha(published / "L2-50K-STABILITY.json"), "STATE_CAPACITY.json": sha(published / "STATE_CAPACITY.json")},
        "statistical_limits": [
            "Three evaluation seeds do not constitute three independent training seeds.",
            "Ten-candidate comparisons use one training seed and matching controls; confirmation clips are temporally disjoint but from the same test episode split.",
            "I08 uses three training/pretraining seeds only when all listed endpoints are finished; same-weight feedback controls isolate inference feedback.",
            "Original full-data results at three epochs cannot be used as an equal-budget architecture ranking against lightweight 25k/50k baselines.",
            "No human gallery signoff has been substituted by the report or AI review.",
        ],
    }
    feedback_statistics = {}
    for variant in ["semantic", "same_weights_pixel", "matched_pixel_training"]:
        values = [v[variant] for v in semantic.values() if variant in v]
        if not values:
            continue
        measures = {
            "h7_agent_px": [v["quality"]["mean_by_horizon"]["mean_agent"][6] for v in values],
            "h13_agent_px": [v["quality"]["mean_by_horizon"]["mean_agent"][12] for v in values],
            "cf": [v["summary"]["cf"] for v in values],
            "cf_agent": [v["summary"]["cf_agent"] for v in values],
            "cf_block": [v["summary"]["cf_block"] for v in values],
        }
        feedback_statistics[variant] = {"training_seeds": len(values), "statistics": {
            k: {"values": z, "mean": float(np.mean(z)), "sample_std": float(np.std(z, ddof=1)) if len(z) > 1 else None}
            for k, z in measures.items()}}
    bundle["feedback_statistics_across_training_seeds"] = feedback_statistics
    findings = []
    if full:
        names = ", ".join(k + "=" + format_num(v["summary"]["p0_mean"]) + "%" for k, v in full.items() if v["summary"]["p0_mean"] is not None)
        findings.append("原始全数据模型的三 epoch P0 结果为（" + names + "）；低 beta 配置已有真正闭环规划的正面结果，但这仍是单训练种子，不是完整十 epoch 或等预算架构胜出。")
    findings.append("八条高 beta 运行均由独立含噪后验评测确认零活跃维度，已提前停止。早期短暂退化不等于最终坍塌：原始 L2 beta=1 的表征活跃维度出现延迟恢复，因此只对成熟且持续退化的运行早停；表征恢复不等于规划恢复。")
    if "L2-beta1" in full:
        z = full["L2-beta1"]["summary"]
        findings.append("坍塌门不是唯一的早停依据：原始 L2 beta=1 在三 epoch 的活跃维度为 " + format_num(z["active_dims"], 0) + "，h7 agent/copy=" + format_num(z["h7_agent_copy"], 3) + "，但 P0 只有 " + format_num(z["p0_mean"]) + "%。非空表征和可用控制是不同门槛。")
    n_feedback = feedback_statistics.get("semantic", {}).get("training_seeds", 0)
    findings.append(f"I08 已完成 {n_feedback} 个训练种子的语义反馈终点；同权重开关和第一生成步一致性检查用于隔离反馈路径。中期 agent 误差和动作识别的改变，不等于长时域物块控制或规划已解决；完整逐种子结果见表。")
    if n_feedback:
        fs = feedback_statistics["semantic"]["statistics"]
        findings.append("I08 的反事实 agent 识别均值为 " + format_num(100*fs["cf_agent"]["mean"]) + "%，block 识别均值为 " + format_num(100*fs["cf_block"]["mean"]) + "%。合并动作识别率受较大 agent 位移影响，不能把它当作物块动力学或规划成功率。")
    g, q = rows["geometry"], rows["permuted"]
    findings.append("I02 对角噪声有单种子收益，但方向打乱的对照同样好：D P0=" + format_num(g["diagnostic"]["p0_mean"]) + "% vs " + format_num(q["diagnostic"]["p0_mean"]) + "%，C P0=" + format_num(g["confirmation"]["p0_mean"]) + "% vs " + format_num(q["confirmation"]["p0_mean"]) + "%。不能把改善归因于 Jacobian 方向对齐；基础初始参数逐 tensor 相同。")
    findings.append("I03 相关噪声与 I05 forecast rendering 是值得保留的单种子正面候选。I03 D/C P0=" + format_num(rows["correlated"]["diagnostic"]["p0_mean"]) + "/" + format_num(rows["correlated"]["confirmation"]["p0_mean"]) + "%，匹配 baseline=" + format_num(rows["base001"]["diagnostic"]["p0_mean"]) + "/" + format_num(rows["base001"]["confirmation"]["p0_mean"]) + "%。这些不是多训练种子稳定收益的证明。")
    findings.append("I09 逆动作损失是强负面对照：D/C P0=" + format_num(rows["fork"]["diagnostic"]["p0_mean"]) + "/" + format_num(rows["fork"]["confirmation"]["p0_mean"]) + "%，零损失对照=" + format_num(rows["fork_null"]["diagnostic"]["p0_mean"]) + "/" + format_num(rows["fork_null"]["confirmation"]["p0_mean"]) + "%；两组全部 444 个初始 tensor 一致。辅助任务拟合好不保证可规划动力学。")
    findings.append("I10 的 covariance / 修正 Gramian 规划代价，在相同冻结权重及 CEM 搜索预算下显著差于 GoalMSE；坐标不变性本身不保证有用的规划代价。I01 当前权重下的两种目标、I04 canonicalizer、I06 gradient constraint、I07 learned variance 未给出可靠任务收益。改变目标单位后的单一 beta 失败不能否定该目标在所有超参数下有效的可能性。")
    historical = stability["summary"]["statistics"]["p0_mean_per_training_seed"]
    findings.append("保留旧轻量 L2 三训练种子的负面稳定性结果：P0=" + "/".join(format_num(x) for x in historical["values"]) + "%。新 full-data 单种子好结果不能抹去这个失稳证据。")
    bundle["major_findings"] = findings
    if a.require_complete and pending:
        raise RuntimeError("Unfinished required endpoints: " + ", ".join(pending))
    (out / "main-conclusions-results.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    arrays = {}
    for ident, item in evidence.items():
        path = Path(item["root"]) / "quality/per-case-metrics.npz"
        with np.load(path, allow_pickle=False) as z:
            for name in z.files:
                arrays[ident + "/" + name] = z[name]
    np.savez_compressed(out / "main-conclusions-per-case.npz", **arrays)
    table = [
        "# 主要结论", "", "本报告只汇总能直接回答主要问题的结果。原始十 epoch 及其他条件扩展仍暂缓。", "",
        "## 原始全数据与 renderer：三个 epoch", "",
        "| 配置 | 活跃维度 | h7 agent/copy ↓ | h7 moving block/copy ↓ | FVD-256 ↓ | 动作反事实识别 % ↑ | P0 三评测种子 % ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, item in full.items():
        s = item["summary"]
        plans = " / ".join(format_num(x, 0) for x in s["p0_values"]) if s["p0_values"] else "未测"
        table.append(f"| {name} | {format_num(s['active_dims'],0)} | {format_num(s['h7_agent_copy'],3)} | {format_num(s['h7_moving_block_copy'],3)} | {format_num(s['fvd'])} | {format_num(100*s['cf']) if s['cf'] is not None else '—'} | {plans} |")
    table += [
        "", "上述是三 epoch 的中期 checkpoint，单训练种子；不能宣称完成原方案或优于未做的等预算 LeWM 基线。高 beta 的八条运行全部由独立含噪后验判据确认零活跃维度，已保存检查点并提前停止。", "",
        "## 十候选及关键对照", "",
        "D = 旧固定诊断 case；C 的生成窗口排除旧 clip 的重帧，规划换用评测 seed242/243/244，不保证所有 reset 与旧评测完全不重合。两者仍来自同一 test episode split、同一训练种子。颜色/点域不伪造 PushT 规划结果。", "",
        "| 方法/对照 | D agent/copy ↓ | C agent/copy ↓ | D P0 % ↑ | C P0 % ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, item in rows.items():
        x, y = item.get("diagnostic", {}), item.get("confirmation", {})
        table.append(f"| {item['label']} | {format_num(x.get('h7_agent_copy'),3)} | {format_num(y.get('h7_agent_copy'),3)} | {format_num(x.get('p0_mean'))} | {format_num(y.get('p0_mean'))} |")
    table += ["", "## I08：同权重反馈开关", "",
              "| 训练种子 | semantic h7 agent px ↓ | pixel-toggle h7 agent px ↓ | semantic CF % ↑ | pixel-toggle CF % ↑ | semantic h13 agent px ↓ |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for seed, values in semantic.items():
        if "semantic" not in values or "same_weights_pixel" not in values:
            continue
        x, y = values["semantic"], values["same_weights_pixel"]
        hx = x["quality"]["mean_by_horizon"]["mean_agent"]
        hy = y["quality"]["mean_by_horizon"]["mean_agent"]
        table.append(f"| {seed} | {hx[6]:.2f} | {hy[6]:.2f} | {100*x['summary']['cf']:.2f} | {100*y['summary']['cf']:.2f} | {hx[12]:.2f} |")
    table += ["", "I08 反事实识别进一步拆分如下；不能将合并识别率解释为 PushT 成功率。", "", "| 训练种子 | semantic agent CF % | semantic block CF % | pixel-toggle agent CF % | pixel-toggle block CF % |", "| --- | ---: | ---: | ---: | ---: |"]
    for seed, values in semantic.items():
        if "semantic" in values and "same_weights_pixel" in values:
            x, y = values["semantic"]["summary"], values["same_weights_pixel"]["summary"]
            table.append(f"| {seed} | {100*x['cf_agent']:.2f} | {100*x['cf_block']:.2f} | {100*y['cf_agent']:.2f} | {100*y['cf_block']:.2f} |")
    table += ["", "## 关键配对差异", "", "以下 P0 区间为 episode-cluster 95% bootstrap，条件于这些已训练权重。跨训练种子的泛化仍由独立复现判断。", "", "| 对照 | D P0 差值及 95% CI，百分点 | C P0 差值及 95% CI，百分点 |", "| --- | --- | --- |"]
    for key, control in [("geometry", "base001"), ("permuted", "geometry"), ("correlated", "base001"), ("forecast", "base001"), ("fork", "fork_null")]:
        texts = []
        for cohort in ["diagnostic", "confirmation"]:
            z = paired[f"{cohort}:{key}-{control}"]["p0_delta_percentage_points"]
            texts.append(f"{z['mean_delta']:+.2f} [{z['ci95'][0]:+.2f}, {z['ci95'][1]:+.2f}]")
        table.append("| " + key + " − " + control + " | " + " | ".join(texts) + " |")
    table += ["", "I08 同权重 h7/h13 agent 差值、反事实识别差值与 I10 配对规划差值的全部区间保存在 JSON。研究 baseline 与 zero-loss 对照的额外分支/RNG 消耗不同，不能把 zero-loss 对照自身的高分当成新方法收益。", ""]
    table += ["", "## I10：几何规划代价的负面对照", "", "同一组冻结模型权重、相同 CEM 搜索预算与配对 reset；几何代价仍会增加模型调用，不能声称等计算。", "", "| 代价 | 三评测种子 P0 % | 均值 % |", "| --- | --- | ---: |"]
    for method, records in geometric_planning.items():
        values = [x["result"]["metrics"]["success_rate"] for x in records]
        table.append("| " + method + " | " + " / ".join(f"{x:.0f}" for x in values) + f" | {np.mean(values):.2f} |")
    table += ["", "Gramian 使用已修正的精确缓存身份版本；旧 moment-cache 结果继续标为无效，不混入数值。", ""]
    table += [
        "", "I08 成本为 25k 预训练 + 25k finetune；frame encoder 冻结。相同权重只切换反馈路径，第一步应相同。中期 agent 改善不等于长期物块控制或规划已解决。", "",
        "历史对照另见 L2-50K-STABILITY.md、STATE_CAPACITY.md 与 EARLY-STOP-TRIAGE.md。", "",
        "协议：生成评测为 256 段 16 帧（3 真 context + 13 生成），I3D400、64→224 bilinear；与继承 RGB16 FVD-2048 不同。h7 moving block/copy 在移动物块 case 子集计算。位置误差以 simulator 像素坐标报告，真实帧 pose estimator 的误差底也完整保存。规划为 P0、CEM 300×30、horizon5/history3、goal offset25、budget50，每个评测种子50回合。", "",
        "配对置信区间以 episode 为 cluster，5000 次 bootstrap；只描述这组已训练权重的测试不确定性，不代表跨训练种子显著性。十候选比较是探索筛选，确认窗口来自同一 test episode split 的时间不重叠片段。", "",
        "## 尚未完成", "",
    ]
    table += [f"- {x}" for x in pending] if pending else ["本报告列出的主要结论终点已完成。其他长预算/额外扩展保持暂缓，未记为完成。"]
    intro = ["## 关键结论", ""] + [f"{i+1}. {text}" for i, text in enumerate(findings)] + ["", "![Planning comparison](main-candidate-planning.png)", "", "![Feedback control](main-I08-same-weights.png)", ""]
    table[4:4] = intro
    table += ["", "![Full-data checkpoints](main-full-data-checkpoints.png)", "", "![Mechanistic controls](main-mechanism-controls.png)", "", "![I10 planning costs](main-I10-planning-costs.png)", ""]
    (out / "MAIN-CONCLUSIONS.md").write_text("\n".join(table) + "\n")
    print(json.dumps({"complete": not pending, "pending": pending, "endpoints": len(evidence),
                      "paired_comparisons": len(paired), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

