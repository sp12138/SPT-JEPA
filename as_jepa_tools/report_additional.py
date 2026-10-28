"""Publish additional numbers and scientific figures from immutable EFS results."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign, out = Path(args.campaign), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((campaign / "plan.json").read_text())
    jobs = {j["id"]: j for j in plan["jobs"]}
    evidence = {}

    def read(path):
        evidence[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    quality_root = campaign / "quality-50k-seed0-full256/results"
    quality = read(quality_root / "metrics.json")
    additional = read(campaign / "quality-50k-seed0-postprocess/results/metrics.json")
    cases = read(quality_root / "cases.json")
    arrays = np.load(quality_root / "per-case-metrics.npz")
    modes = ["goal_mse", "covariance", "gramian-contextfix"]
    names = ["GoalMSE", "Train covariance", "Local Gramian (fixed cache)"]
    planning = {}
    for mode in modes:
        rows = []
        for seed in [142, 143, 144]:
            ident = f"idea-I10-{mode}-seed{seed}"
            entry = read(campaign / ident / "results/metrics.json")
            exit_record = read(campaign / "exits" / (ident + ".json"))
            if exit_record["returncode"] != 0:
                raise RuntimeError("Incomplete planning result: " + ident)
            row = dict(seed=seed, result=entry, exit=exit_record, job=jobs[ident])
            stats = campaign / ident / "results/geometry-statistics.json"
            if stats.exists():
                row["geometry_statistics"] = read(stats)
            rows.append(row)
        planning[mode] = rows
    for seed_index in range(3):
        reference = planning["goal_mse"][seed_index]["result"]
        for mode in modes[1:]:
            other = planning[mode][seed_index]["result"]
            for key in ["episodes", "start_steps", "seed", "solver_rng_seed"]:
                assert reference[key] == other[key], key
    invalid = [read(campaign / f"idea-I10-gramian-seed{s}/results/metrics.json") for s in [142, 143, 144]]
    covariance = np.load(campaign / "idea-I10-covariance/covariance.npy")
    eigen = np.linalg.eigvalsh(covariance)
    result = dict(
        scope="Additional exploratory evidence; one frozen training seed. Three evaluation seeds are not three training seeds. Original full-budget campaign is still active.",
        quality=quality, quality_additional=additional, quality_cases=cases,
        quality_job=jobs["quality-50k-seed0-full256"],
        planning=planning,
        invalidated_gramian=dict(results=invalid, reason="Legacy cache compared only pixel moments; translations can collide. Retained for provenance and excluded from scientific comparison."),
        covariance_spectrum=dict(eigenvalues=eigen.tolist(), condition_number=float(eigen[-1] / eigen[0]),
                                 scope="Training covariance including the declared ridge; descriptive diagnosis, not proof of why a planner fails."),
        evidence_sha256=evidence,
    )
    (out / "additional-results.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "quality-per-case-metrics.npz", **{key: arrays[key] for key in arrays.files})
    extra_arrays = np.load(campaign / "quality-50k-seed0-postprocess/results/per-case-quality.npz")
    np.savez_compressed(out / "quality-additional-per-case.npz", **{key: extra_arrays[key] for key in extra_arrays.files})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "svg.fonttype": "none"})
    colors = ["#1976b9", "#db7b25", "#60519b"]
    horizon = np.arange(1, 14)
    moving = np.array([x["moving_block"] for x in cases])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    specifications = [
        ("agent", "Agent position error (px)", False),
        ("block", "Moving-block position error (px)", True),
        ("angle_deg", "Block angle error (degrees)", False),
        ("lpips", "LPIPS (AlexNet)", False),
    ]
    for ax, (key, label, moving_only) in zip(axes.flat, specifications):
        select = moving if moving_only else np.ones(len(cases), dtype=bool)
        ax.plot(horizon, arrays["mean_" + key][select].mean(0), label="Mean-state rollout", color=colors[0])
        if "sample_" + key in arrays:
            ax.plot(horizon, arrays["sample_" + key][select].mean((0, 1)),
                    label="Four latent draws (average)", color=colors[1])
        if "copy_" + key in arrays:
            ax.plot(horizon, arrays["copy_" + key][select].mean(0), "--", label="Copy last context", color="#505050")
        ax.set(xlabel="Future kept-frame horizon", ylabel=label, xticks=[1, 3, 7, 10, 13])
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Successful tiny L2, seed0 / 50k: generation improves on copying\n256 fixed cases; three observed context frames; single training seed", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out / ("quality-horizons." + ext), dpi=180)
    plt.close(fig)

    values = [quality["copy_fvd"]["fvd"], quality["fvd"]["fvd"]] + [x["fvd"] for x in additional["sample_fvd"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(["Copy", "Mean state", "Draw1", "Draw2", "Draw3", "Draw4"],
                  values, color=["#777777", colors[0]] + [colors[1]] * 4)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(ylabel="FVD (lower is better)", ylim=(0, max(values) * 1.13))
    ax.set_title("Matched256-case FVD: 3 true context +13 generated frames\n64px to224px; I3D400; each draw separately256real/256fake")
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out / ("fvd-comparison." + ext), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scores = np.array([[x["result"]["metrics"]["success_rate"] for x in planning[m]] for m in modes])
    times = np.array([[x["result"]["seconds"] for x in planning[m]] for m in modes])
    for i, name in enumerate(names):
        axes[0].bar(i, scores[i].mean(), color=colors[i], alpha=.65)
        axes[0].scatter(i + np.linspace(-.09, .09, 3), scores[i], color=colors[i], edgecolors="black", s=30)
        axes[0].text(i, scores[i].mean() + 3, f"{scores[i].mean():.1f}%", ha="center")
        ratios = times[i] / times[0]
        axes[1].bar(i, ratios.mean(), color=colors[i], alpha=.65)
        axes[1].scatter(i + np.linspace(-.09, .09, 3), ratios, color=colors[i], edgecolors="black", s=30)
    for ax in axes:
        ax.set_xticks(range(3), names, rotation=14)
        ax.grid(axis="y", alpha=.2)
    axes[0].set(ylabel="Planning success (%)", ylim=(0, 100))
    axes[1].set(ylabel="Evaluation wall time / paired GoalMSE")
    axes[1].axhline(1, color="#555555", linestyle="--", linewidth=1)
    fig.suptitle("I10: paired planner comparison on eval seeds142/143/144\nOne frozen training model,50episodes/seed; dots are evaluation seeds", fontsize=12)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(out / ("planning-geometry." + ext), dpi=180)
    plt.close(fig)

    truth = np.load(quality_root / "truth-rgb.npy", mmap_mode="r")
    mean = np.load(quality_root / "mean-rgb.npy", mmap_mode="r")
    canvas = Image.new("RGB", (54 + 6 * 64, 26 + 32 * 64), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((56, 5), "h1 GT/pred       h7 GT/pred       h13 GT/pred", fill="black")
    for i in range(32):
        draw.text((3, 31 + i * 64), str(i).zfill(2), fill="black")
        for column, h in enumerate([1, 7, 13]):
            canvas.paste(Image.fromarray(np.array(truth[i, h + 2])), (54 + column * 128, 26 + i * 64))
            canvas.paste(Image.fromarray(np.array(mean[i, h - 1])), (118 + column * 128, 26 + i * 64))
    canvas.save(out / "fixed32-overview.png")
    table = "\n".join(f"| {name} | " + " / ".join(f"{x:.0f}" for x in row) + f" | {row.mean():.2f} |"
                      for name, row in zip(names, scores))
    h7 = {k: np.asarray(v)[6] for k, v in quality["mean_by_horizon"].items()}
    text = f"""# 追加结果：生成指标与规划度量

原始全量实验和候选长跑仍在继续。这里仅报告已完成、可复核的额外结果；一个训练种子的三组评估seed不能作为三训练种子统计。

轻量L2 beta=.01 /50k /seed0的FVD为{values[1]:.2f}，复制末帧为{values[0]:.2f}；四次latent采样FVD为{", ".join(f"{x:.2f}" for x in values[2:])}。每个分数独立使用256real/256fake、16帧视频，其中3帧真实context加13帧预测，64→224双线性缩放、I3D400特征。这不是继承RGB16的FVD2048协议，也不能跨协议比绝对值。

h7 agent/copy={quality["h7_agent_ratio"]:.4f}，moving-block/copy={quality["h7_moving_block_ratio"]:.4f}；mean LPIPS={h7["mean_lpips"]:.4f}，SSIM={h7["mean_ssim"]:.4f}，角度误差={h7["mean_angle_deg"]:.2f}°（copy={h7["copy_angle_deg"]:.2f}°）。长horizon仍有明显误差增长。32例缩略图按固定顺序全部保留，不能当作正式人工gallery签收。

![全部horizon的生成误差](quality-horizons.png)

![FVD均值与四次采样](fvd-comparison.png)

I10在同一个冻结模型上比较原GoalMSE、训练协方差距离与局部Gramian。每个eval seed的episode、start和solver RNG均配对，CEM预算相同。局部Gramian已修正缓存，按完整pixels和action-history确认context，并记录额外模型调用和准备耗时。

| 规划成本 | seed142 /143 /144成功率(%) | 均值(%) |
| --- | --- | --- |
{table}

![规划成功率与评估耗时](planning-geometry.png)

这个实现没有显示规划收益。坐标变换下的不变性仍有数学意义，但不是任务有效性的充分条件；训练协方差的条件数为{eigen[-1]/eigen[0]:.2e}，低方差方向被逆矩阵放大只是可能原因，尚未确认为因果解释。没有依据这批test结果继续调alpha。旧Gramian试验的像素矩缓存可能混淆平移后的context，已标记无效并完整保留在JSON，未纳入比较。

![固定32例，GT与mean对照](fixed32-overview.png)

逐例、逐horizon及四个sample的全部质量数值另见\n[quality-per-case-metrics.npz](quality-per-case-metrics.npz)和[quality-additional-per-case.npz](quality-additional-per-case.npz)。这些是小型数值结果，没有像素或权重。完整数值、每例规划成功/失败、case身份、计时、源版本、参数和证据哈希见[additional-results.json](additional-results.json)；图提供PNG及SVG。原始RGB数组、视频、训练日志和权重均在EFS，Git中不含这些大文件。生成结果仍须匹配其他模型和训练种子，完整方案的完成状态见[执行进度](../README.md)。
"""
    (out / "README.md").write_text(text)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
             if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(dict(files=files, source="as_jepa_tools/report_additional.py",
                                                      evidence_sha256=evidence), indent=2))
    print(json.dumps(dict(files=len(files),planning_means=scores.mean(1).tolist(),fvd=values)))


if __name__ == "__main__":
    main()
