"""Publish completed new-window quality and available confirmation stages."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--confirmation",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root,out=Path(args.confirmation),Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    quality=json.loads((root/"quality/metrics.json").read_text())
    extra=json.loads((root/"quality-additional/metrics.json").read_text())
    cases=json.loads((root/"quality/cases.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())
    arrays=np.load(root/"quality/per-case-metrics.npz")
    more=np.load(root/"quality-additional/per-case-quality.npz")
    cf_path=root/"counterfactual/counterfactual.json"
    cf=json.loads(cf_path.read_text()) if cf_path.exists() else None
    planning=[]
    for seed in [242,243,244]:
        path=root/"planning"/("seed"+str(seed))/"metrics.json"
        if path.exists():planning.append(json.loads(path.read_text()))
    payload=dict(protocol=protocol,quality=quality,additional=extra,cases=cases,counterfactual=cf,planning=planning,
                 scope="Complete quality stage on sealed nonoverlapping windows; other stages included only when their numerical artifact exists. Same training seed0 and original heldout episodes; no claim of a new model or new dataset.",
                 stage_status=json.loads((root/"status.json").read_text()),
                 source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [root/"quality/metrics.json",root/"quality-additional/metrics.json",root/"protocol.json"]})
    (out/"confirmation-results.json").write_text(json.dumps(payload,indent=2))
    np.savez_compressed(out/"confirmation-quality-metrics.npz",**{k:arrays[k] for k in arrays.files})
    np.savez_compressed(out/"confirmation-additional-metrics.npz",**{k:more[k] for k in more.files})
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","savefig.bbox":"tight"})
    moving=np.array([x["moving_block"] for x in cases])
    horizon=np.arange(1,14)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,key,label,motion in [(axes[0,0],"agent","Agent position error (px)",False),(axes[0,1],"block","Moving-block position error (px)",True),(axes[1,0],"angle_deg","Block angle error (degrees)",False)]:
        subset=moving if motion else np.ones(len(cases),bool)
        ax.plot(horizon,arrays["mean_"+key][subset].mean(0),color="#1976b9",label="Mean-state rollout")
        ax.plot(horizon,arrays["sample_"+key][subset].mean((0,1)),color="#db7b25",label="Four latent draws (average)")
        ax.plot(horizon,arrays["copy_"+key][subset].mean(0),color="#555555",linestyle="--",label="Copy last context")
        ax.set(xlabel="Future kept-frame horizon",ylabel=label,xticks=[1,3,7,10,13]);ax.grid(alpha=.2)
    ax=axes[1,1]
    ax.plot(horizon,more["mean_uint8_lpips"].mean(0),color="#1976b9",label="Mean-state rollout")
    ax.plot(horizon,np.mean([more[f"sample{s}_lpips"].mean(0) for s in range(4)],0),color="#db7b25",label="Four latent draws (average)")
    ax.plot(horizon,more["copy_lpips"].mean(0),color="#555555",linestyle="--",label="Copy last context")
    ax.set(xlabel="Future kept-frame horizon",ylabel="LPIPS (same saved uint8 protocol)",xticks=[1,3,7,10,13]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Confirmation: 256 new windows, same trained L2 seed 0 / 50k\nNo shared frames with diagnostic clips; same test episode split",fontsize=12)
    fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(out/("confirmation-horizons."+ext),dpi=180)
    plt.close(fig)
    h7=quality["mean_by_horizon"]
    cf_sentence=(f'反事实动作识别={cf["action_identification_accuracy"]*100:.2f}%（新case seed142，排除旧case时间段）。' if cf else "反事实及规划阶段仍在运行，不能由本生成阶段代替。")
    p0_sentence=(f'P0已有{len(planning)}/3个确认评估种子的结果，当前值为'+"/".join(f'{x["metrics"]["success_rate"]:.1f}%' for x in planning)+"。") if planning else ""
    text=f"""# 新窗口确认：同一训练模型的正面生成结果

模型保持原L2 beta=.01 /50k /training seed0。新256例在看到候选最终结果前固定，移除了与旧诊断16帧clip共享帧的2785个窗口起点。它仍来自同一test episode split，不是新训练种子或独立数据集；不应该把与旧case数值的差别叫训练提升。

- h7 agent/copy = **{quality["h7_agent_ratio"]:.4f}**，moving-block/copy = **{quality["h7_moving_block_ratio"]:.4f}**。
- FVD = **{quality["fvd"]["fvd"]:.2f}**，copy = **{quality["copy_fvd"]["fvd"]:.2f}**。四个latent draw分别为{", ".join(f'{x["fvd"]:.2f}' for x in extra["sample_fvd"])}。
- h7 mean LPIPS={h7["mean_lpips"][6]:.4f}，SSIM={h7["mean_ssim"][6]:.4f}，block angle={h7["mean_angle_deg"][6]:.2f}°；mean LPIPS原始浮点与图中统一uint8版本分别保留。

![新窗口全部horizon](confirmation-horizons.png)

{cf_sentence}{p0_sentence}

FVD仍是256real/256fake的16帧协议，3帧真实context+13帧生成、64→224双线性、I3D400；每个draw独立计算，未把四个draw合并成1024个real样本。这个确认支持已有单种子模型的动力学收益；最终跨训练种子稳定性和原始全量方案仍待完成。

[完整JSON与case身份](confirmation-results.json)，[逐例质量数值](confirmation-quality-metrics.npz)，[逐例copy/sample完整质量数值](confirmation-additional-metrics.npz)。原始像素和视频保存在EFS，未放Git。
"""
    (out/"CONFIRMATION.md").write_text(text)
    for artifact in [out/"CONFIRMATION.md",out/"confirmation-horizons.svg"]:
        artifact.write_text("\n".join(x.rstrip() for x in artifact.read_text().splitlines()).rstrip()+"\n")
    manifest=json.loads((out/"manifest.json").read_text())
    manifest["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
    manifest["confirmation_report_scope"]=payload["scope"]
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(cases=quality["cases"],agent_ratio=quality["h7_agent_ratio"],block_ratio=quality["h7_moving_block_ratio"],fvd=quality["fvd"]["fvd"],cf=cf["action_identification_accuracy"] if cf else None,planning_completed=len(planning))))


if __name__=="__main__":
    main()
