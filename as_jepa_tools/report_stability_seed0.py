"""Audit one frozen model's P0 replay and predeclared fresh evaluation cases."""
import argparse
import hashlib
import json
import math
from pathlib import Path


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
    campaign, out = Path(a.campaign), Path(a.output)
    protocol = read(campaign / "protocol.json")
    checkpoint = Path(protocol["reference_training_seed0_checkpoint"])
    golden_weights = checkpoint / "weights.pt"
    config = read(checkpoint / "config.json")
    rows, old_rows = {}, {}
    for seed in protocol["primary_planning_eval_seeds"] + protocol["confirmation_planning_eval_seeds"]:
        job_id = f"p0-trainseed0-evalseed{seed}"
        root = campaign / job_id
        exit_record = read(campaign / "exits" / (job_id + ".json"))
        execution, metrics = read(root / "execution.json"), read(root / "metrics.json")
        assert exit_record["returncode"] == execution["returncode"] == 0, job_id
        assert (root / "policy/weights.pt").samefile(golden_weights), job_id
        assert read(root / "policy/config.json") == config, job_id
        assert execution["episodes"] == 50 and execution["seed"] == seed and not execution["bf16"]
        assert metrics["split"] == "heldout" and metrics["seed"] == metrics["solver_rng_seed"] == seed
        assert metrics["float32_matmul_precision"] == "highest"
        successes = metrics["metrics"]["episode_successes"]
        assert len(successes) == len(metrics["episodes"]) == len(metrics["start_steps"]) == 50
        score = 100 * sum(successes) / len(successes)
        assert math.isfinite(score) and abs(score - metrics["metrics"]["success_rate"]) < 1e-9
        rows[str(seed)] = {"metrics": metrics, "execution": execution, "exit": exit_record}
        if seed in protocol["primary_planning_eval_seeds"]:
            old = read(Path(protocol["reference_results"]) / "planning" / f"seed{seed}" / "metrics.json")
            for key in ["episodes", "start_steps"]:
                assert old[key] == metrics[key], (seed, key)
            old_rows[str(seed)] = old
    scores = lambda seeds: [rows[str(seed)]["metrics"]["metrics"]["success_rate"] for seed in seeds]
    old_values = [old_rows[str(seed)]["metrics"]["success_rate"] for seed in protocol["primary_planning_eval_seeds"]]
    primary, confirmation = scores(protocol["primary_planning_eval_seeds"]), scores(protocol["confirmation_planning_eval_seeds"])
    def cases(seeds):
        return [(ep, start) for seed in seeds for ep, start in zip(rows[str(seed)]["metrics"]["episodes"], rows[str(seed)]["metrics"]["start_steps"])]
    old_cases, fresh_cases = cases(protocol["primary_planning_eval_seeds"]), cases(protocol["confirmation_planning_eval_seeds"])
    identical = all(old_rows[str(seed)]["metrics"]["episode_successes"] == rows[str(seed)]["metrics"]["metrics"]["episode_successes"] for seed in protocol["primary_planning_eval_seeds"])
    result = {
        "complete_seed0_evaluation_check": True, "independent_training_seed_count": 1,
        "multi_training_seed_stability_established": False,
        "checkpoint": str(checkpoint), "weights_sha256": sha(golden_weights), "model_config": config,
        "source_commit": a.source_commit, "report_script_sha256": sha(__file__),
        "evaluation_source": protocol["evaluation_source"],
        "old_primary_values": old_values, "replay_primary_values": primary,
        "old_primary_mean": sum(old_values) / len(old_values), "replay_primary_mean": sum(primary) / len(primary),
        "confirmation_values": confirmation, "confirmation_mean": sum(confirmation) / len(confirmation),
        "paired_case_identity_verified": True, "all_150_replay_outcomes_identical": identical,
        "exact_reset_overlap_between_primary_and_confirmation": len(set(old_cases) & set(fresh_cases)),
        "unique_primary_resets": len(set(old_cases)), "unique_confirmation_resets": len(set(fresh_cases)),
        "scope": "One fixed trained model. Fresh evaluation seeds use the same heldout episode split; they are not independent training seeds or a new dataset. New seed1-4 training remains pending.",
        "planning_results": rows, "original_planning_results": old_rows,
    }
    assert not out.exists(), "Preserve existing frozen output"
    out.mkdir(parents=True)
    (out / "seed0-reevaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    x = np.arange(3)
    axes[0].bar(x - .18, old_values, .36, label="Published reference", color="#97b8cf")
    axes[0].bar(x + .18, primary, .36, label="Same-weight replay", color="#3d6d97")
    axes[0].set_xticks(x, ["42", "43", "44"])
    axes[0].set_title("Paired original cases")
    axes[0].legend(fontsize=9, loc="lower left")
    bars = axes[1].bar(x, confirmation, .6, color="#56845d")
    axes[1].bar_label(bars, fmt="%.0f%%", padding=3)
    axes[1].set_xticks(x, ["1042", "1043", "1044"])
    axes[1].set_title("Predeclared fresh cases")
    for ax in axes:
        ax.axhline(sum(old_values) / 3, color="#666666", linestyle="--", linewidth=1)
        ax.set_ylim(0, 103)
        ax.set_xlabel("Evaluation seed")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("P0 success (%)")
    fig.suptitle("One fixed trained model: planning replay and fresh evaluation cases")
    fig.text(.5, .01, "Same weights and heldout episode split; this does not establish stability across training seeds.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    for ext in ["png", "svg"]:
        path = out / ("seed0-reevaluation." + ext)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + "\n")
    text = [
        "# 固定 seed0 checkpoint：规划复跑与新案例", "",
        "本记录不是多训练种子复现结果。seed1–4 的独立训练仍在进行。", "",
        f"同一份权重，在原 P0 评测种子42/43/44上的已发布结果为 {old_values}%，本次复跑为 {primary}%，均值 {sum(primary)/3:.2f}%。",
        f"150个起点/目标案例身份完全一致，逐例成功/失败结果全部相同：{identical}。",
        f"预先声明的新评测种子1042/1043/1044为 {confirmation}%，均值 **{sum(confirmation)/3:.2f}%**。",
        f"新旧各150个案例的完全相同reset数量为 {result['exact_reset_overlap_between_primary_and_confirmation']}；它们仍来自同一heldout episode split，不能称为新数据集。", "",
        "这支持原checkpoint的规划结果可以复跑、且在新案例上仍有较高成功率；它不能回答换训练初始化后是否稳定。", "",
        "![Planning replay and fresh cases](seed0-reevaluation.png)", "",
        "[完整逐例数值与验证身份](seed0-reevaluation.json)。所有六次新评测均exit0，权重与原checkpoint为同一物理文件。", "",
    ]
    (out / "SEED0-REEVALUATION.md").write_text("\n".join(text).rstrip() + "\n")
    print(json.dumps({k: result[k] for k in ["replay_primary_mean", "confirmation_mean", "all_150_replay_outcomes_identical", "exact_reset_overlap_between_primary_and_confirmation", "multi_training_seed_stability_established"]}))


if __name__ == "__main__":
    main()
