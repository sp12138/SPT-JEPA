"""Publish the semantic-memory candidate with exact same-weight feedback control."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True);parser.add_argument("--output",required=True);args=parser.parse_args()
    campaign,out=Path(args.campaign),Path(args.output);names=[
        ("semantic","eval-idea-I08-native-carry-25k-seed0"),
        ("pixel_same_weights","eval-idea-I08-native-carry-25k-seed0-same-weights-pixel"),
        ("pixel_matched_training","eval-idea-I08-native-pixel-matched-25k-seed0")]
    rows=[]
    for label,name in names:
        folder=campaign/name/"results"
        quality=json.loads((folder/"quality/metrics.json").read_text())
        additional=json.loads((folder/"quality-additional/metrics.json").read_text())
        cf=folder/"counterfactual/counterfactual.json"
        row=dict(label=label,job=name,checkpoint=json.loads((folder/"checkpoint_manifest.json").read_text()),
                 quality=quality,additional_quality=additional,
                 cases=json.loads((folder/"quality/cases.json").read_text()),
                 counterfactual=json.loads(cf.read_text()) if cf.exists() else None,
                 stage_status=json.loads((folder/"status.json").read_text()))
        arrays=np.load(folder/"quality/per-case-metrics.npz")
        np.savez_compressed(out/("I08-"+label+"-quality.npz"),**{k:arrays[k] for k in arrays.files})
        rows.append(row)
    assert rows[0]["checkpoint"]["weights_sha256"]==rows[1]["checkpoint"]["weights_sha256"]
    true_cfg=dict(rows[0]["checkpoint"]["model_config"]);false_cfg=dict(rows[1]["checkpoint"]["model_config"])
    assert true_cfg.pop("use_semantic_feedback") is True and false_cfg.pop("use_semantic_feedback") is False
    assert true_cfg==false_cfg
    identity=[[(x["episode"],x["start"]) for x in row["cases"]] for row in rows]
    assert identity[0]==identity[1]==identity[2]
    mean0=rows[0]["quality"]["mean_by_horizon"];mean1=rows[1]["quality"]["mean_by_horizon"]
    for metric in ["mean_agent","mean_block","mean_psnr"]:
        assert mean0[metric][0]==mean1[metric][0],metric
    payload=dict(models=rows,same_weights_verified=True,first_step_metrics_identical=True,
                 scope="One pretrainedS1seed0 exact25k,then25kmatchedfinetune. Frozenframeencoder. Semantic/pixel-same-weightsdifferonlyinfeedbackflag;matchedpixeltraininghaslatentauxiliaryweight0. Noindependenttrainingseedreplicationyet;longhorizondriftremains.")
    (out/"I08-SEMANTIC-FEEDBACK.json").write_text(json.dumps(payload,indent=2))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","savefig.bbox":"tight"})
    fig,axes=plt.subplots(2,2,figsize=(10,7));horizon=np.arange(1,14)
    colors=["#1976b9","#c65346","#8a8a8a"];labels=["Semantic feedback","Pixel feedback, same weights","Pixel feedback, matched training"]
    for ax,metric,title in [(axes[0,0],"mean_agent","Agent position error (px)"),(axes[0,1],"mean_block","Block position error (px)"),(axes[1,0],"mean_lpips","LPIPS")]:
        for row,color,label in zip(rows,colors,labels):
            ax.plot(horizon,row["quality"]["mean_by_horizon"][metric],color=color,label=label,
                    linestyle=":" if row["label"]=="pixel_matched_training" else "-",linewidth=2)
        ax.set(xlabel="Future kept-frame horizon",ylabel=title,xticks=[1,3,7,10,13]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    ax=axes[1,1];values=[row["quality"]["fvd"]["fvd"] for row in rows]
    bars=ax.bar(range(3),values,color=colors,alpha=.75);ax.bar_label(bars,fmt="%.1f",padding=3)
    ax.set_xticks(range(3),["Semantic","Pixel, same\nweights","Pixel, matched\ntraining"])
    ax.set(ylabel="FVD (256 matched clips)",ylim=(0,max(values)*1.15));ax.grid(axis="y",alpha=.2)
    fig.suptitle("I08: same first prediction, different feedback accumulation\nOne pretrained seed; horizon-7 benefit; horizon-13 agent drift remains",fontsize=12);fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(out/("I08-feedback-comparison."+ext),dpi=180)
    plt.close(fig)
    table=[]
    for row,label in zip(rows,["语义反馈","同权重切回像素反馈","相同finetune预算的像素反馈"]):
        q=row["quality"];cf=row["counterfactual"]
        table.append(f'| {label} | {q["h7_agent_ratio"]:.3f} / {q["h7_moving_block_ratio"]:.3f} | {q["fvd"]["fvd"]:.1f} | '+(f'{cf["action_identification_accuracy"]*100:.2f}%' if cf else "仍在评估")+" |")
    text="""# I08正面机制结果：中期误差主要受反馈路径影响

从同一份NativeS1精确25k检查点出发，冻结frame encoder，再训练25k。一个分支学习semantic token转移，另一个有相同模块和计算但辅助项权重为0。最关键的推理对照使用**同一套权重/hash**，只切换是否重新编码生成图像；不是换模型或换训练预算。

| 反馈方式 | h7 agent /moving-block误差比 | FVD | 反事实动作识别 |
| --- | --- | --- | --- |
"""+ "\n".join(table)+"""

![同权重反馈机制对照](I08-feedback-comparison.png)

首步agent/block误差两种反馈完全相同，为6.12/3.98px；h7时语义反馈agent24.15px，切回pixel反馈125.20px。语义反馈h7移动物块也优于copy。这个控制把中期改善定位到生成图像重新编码所引入的反馈误差；匹配训练的pixel控制进一步检查额外finetune预算。

**长时问题没有解决。** h13时semantic agent误差137.58px，pixel为129.80px，语义记忆自身仍会漂移。不能凭h7/FVD改善就叫稳定世界模型，更不能用单个训练种子宣称最终benchmark通过。当前是预训练seed0的一次机制实验；其他训练种子的确认、真观测校正以及长时稳定性仍待验证。

语义分支的动作识别83.59%，匹配训练pixel控制31.64%；同权重pixel的反事实项若尚未结束会明确保留pending，不把generation结果替代它。新窗口确认也已排入队列，使用与旧诊断clip无共享帧的case。

[完整数值、checkpoint身份、每horizon指标、反事实逐case结果和各阶段状态](I08-SEMANTIC-FEEDBACK.json)。I08-*-quality.npz保留逐例指标，视频在EFS。训练成本应计25kS1预训练+25kfinetune，不能只报新增25k。
"""
    (out/"I08-SEMANTIC-FEEDBACK.md").write_text(text)
    for p in [out/"I08-SEMANTIC-FEEDBACK.md",out/"I08-feedback-comparison.svg"]:
        p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()).rstrip()+"\n")
    manifest=json.loads((out/"manifest.json").read_text());manifest["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!="manifest.json"};(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(same_weights=True,first_step_identical=True,arms=len(rows),counterfactual_complete=[row["counterfactual"] is not None for row in rows])))


if __name__=="__main__":
    main()
