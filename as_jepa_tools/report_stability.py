"""Summarize complete three-training-seed lightweight endpoints without hiding failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    campaign,out=Path(args.campaign),Path(args.output)
    records=[]
    names=["eval-endpoint-source-L2-50k-seed0","eval-L2-beta001-50k-seed1","eval-L2-beta001-50k-seed2","eval-L2-beta001-g0-50k-seed0"]
    for seed,name in enumerate(names):
        folder=campaign/name/"results"
        exit_record=json.loads((campaign/"exits"/(name+".json")).read_text())
        status=json.loads((folder/"status.json").read_text())
        assert exit_record["returncode"]==0 and status["state"]=="finished"
        quality=json.loads((folder/"quality/metrics.json").read_text())
        case_rows=json.loads((folder/"quality/cases.json").read_text())
        probe={key:json.loads((folder/"probes"/key/"probes.json").read_text()) for key in ["mean","noisy","mean-fp32","noisy-fp32"]}
        cf=json.loads((folder/"counterfactual/counterfactual.json").read_text())
        plans=[json.loads((folder/"planning"/("seed"+str(value))/"metrics.json").read_text()) for value in [42,43,44]]
        collapse=json.loads((folder/"collapse/metrics.json").read_text())
        record=dict(id=name,seed=seed if seed<3 else 0,gap=8 if seed<3 else 0,steps=50000,
                    checkpoint=json.loads((folder/"checkpoint_manifest.json").read_text()),quality=quality,
                    additional_quality=json.loads((folder/"quality-additional/metrics.json").read_text()),
                    cases=case_rows,probes=probe,counterfactual=cf,planning=plans,collapse=collapse,exit=exit_record)
        records.append(record)
        arrays=np.load(folder/"quality/per-case-metrics.npz")
        np.savez_compressed(out/("L2-50k-"+("seed"+str(seed) if seed<3 else "g0-seed0")+"-quality.npz"),
                            **{key:arrays[key] for key in arrays.files})
    identities=[[(x["episode"],x["start"]) for x in row["cases"]] for row in records]
    assert all(x==identities[0] for x in identities)
    main_records=records[:3]
    p0=np.array([[x["metrics"]["success_rate"] for x in row["planning"]] for row in main_records])
    per_seed=p0.mean(1)
    agent=np.array([row["quality"]["h7_agent_ratio"] for row in main_records])
    block=np.array([row["quality"]["h7_moving_block_ratio"] for row in main_records])
    accuracy=np.array([row["counterfactual"]["action_identification_accuracy"] for row in main_records])
    gen_pass=bool(agent.mean()<=.5 and block.mean()<1.)
    action_pass=bool(accuracy.mean()>=.75)
    stats={key:dict(mean=float(values.mean()),sample_std=float(values.std(ddof=1)),values=values.tolist())
           for key,values in dict(h7_agent_ratio=agent,h7_moving_block_ratio=block,
                                  counterfactual_accuracy=accuracy,p0_mean_per_training_seed=per_seed).items()}
    summary=dict(statistics=stats,planning_grid=p0.tolist(),three_training_seed_generation_ratio_gate=gen_pass,
                 three_training_seed_action_identification_gate=action_pass,
                 human_gallery_review_complete=False,formal_gate_passed=False,
                 rule="Generation agent mean<=.5,movingblockmean<1;actionidentificationmean>=.75;atleast24/32humanreviewpasses.",
                 scope="Lightweight phase0/50ksettingwithone-timeinitialstatestd=.1,notoriginalfivephase10epochbenchmark. Same256qualitycases and64counterfactualcaseprotocol across trainingseeds.")
    (out/"L2-50K-STABILITY.json").write_text(json.dumps(dict(summary=summary,records=records),indent=2))
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","savefig.bbox":"tight"})
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    x=np.arange(3)
    axes[0].bar(x,agent,color="#1976b9",alpha=.7,label="Agent")
    axes[0].scatter(x,block,color="#db7b25",marker="D",label="Moving block")
    axes[0].axhline(.5,color="#1976b9",linestyle="--",linewidth=1,label="Agent threshold")
    axes[0].axhline(1,color="#555555",linestyle=":",linewidth=1,label="Copy")
    axes[0].set(ylabel="h7 error / copy",ylim=(0,1.1));axes[0].legend(fontsize=7)
    axes[1].bar(x,accuracy*100,color="#60519b",alpha=.7)
    axes[1].axhline(75,color="#555555",linestyle="--",label="Gate75%")
    axes[1].set(ylabel="Counterfactual action ID (%)",ylim=(0,100));axes[1].legend(fontsize=8)
    for eval_index,color in enumerate(["#1976b9","#db7b25","#60519b"]):
        axes[2].scatter(x+(-.1+eval_index*.1),p0[:,eval_index],color=color,s=35,label="Eval seed "+str(42+eval_index))
    axes[2].plot(x,per_seed,color="#222222",marker="_",markersize=16,linestyle="none",label="Train-seed mean")
    axes[2].set(ylabel="P0 planning success (%)",ylim=(0,100));axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x,["0","1","2"]);ax.set_xlabel("Training seed");ax.grid(axis="y",alpha=.2)
    fig.suptitle("Lightweight L2,50k: seed1 exposes a stability problem\nSame256qualitycases and64counterfactualcases;P0seeds42/43/44",fontsize=12)
    fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(out/("L2-50k-seed-stability."+ext),dpi=180)
    plt.close(fig)
    lines=[]
    for i,row in enumerate(records):
        q=row["quality"];probe=row["probes"]["noisy"];rates=[x["metrics"]["success_rate"] for x in row["planning"]]
        lines.append("| "+("g0seed0" if i==3 else "seed"+str(i))+f' | {probe["agent_r2"]:.3f} / {probe["block_position_r2"]:.3f} | {q["h7_agent_ratio"]:.3f} / {q["h7_moving_block_ratio"]:.3f} | {row["counterfactual"]["action_identification_accuracy"]*100:.2f} | '+"/".join(f"{v:.0f}" for v in rates)+f' | {np.mean(rates):.2f} |')
    text=f"""# 轻量L2的50k三训练种子：正面结果存在，稳定性尚未解决

三个训练种子和g0对照的全部终点评估均结束，包括50k帧train/5k帧testprobe、含噪后验坍塌检测、256例mean/sample生成、64例动作反事实和P042/43/44。以下全部case和测量协议相同，失败种子未被筛掉。

| 训练配置 | noisy agent/block位置R2 | h7 agent /moving-block误差比 | 动作识别(%) | P042/43/44(%) | P0均值(%) |
| --- | --- | --- | --- | --- | --- |
"""+ "\n".join(lines)+f"""

![三训练种子稳定性](L2-50k-seed-stability.png)

- 三种子h7 agent/copy平均{agent.mean():.3f}±{agent.std(ddof=1):.3f}，移动物块{block.mean():.3f}±{block.std(ddof=1):.3f}，按平均生成误差的数值条件通过。
- 动作识别平均{accuracy.mean()*100:.2f}±{accuracy.std(ddof=1)*100:.2f}%，**未达到75%**。seed1的48.44%同时伴随规划均值12.67%，不能把这个配置称作稳定解决任务。
- P0先平均每个训练种子的三个eval seed，再计算训练种子间统计：{per_seed.mean():.2f}±{per_seed.std(ddof=1):.2f}%。9次评估没有被当成9个训练种子。
- g0seed0仍有较好的生成/动作跟随，P0三个eval seed均为70%。这是单训练种子的anchor消融，不能推导所有种子都无需distantanchor。
- 所有三个主种子都通过含噪后验重建坍塌检测，seed1仍未获得强动作控制。只排除“空状态”不足以保证工作；可预测的视觉状态与任务控制之间仍存在差距。

正式generation gate仍未通过：动作识别平均低于阈值，人工32例检查也未签收。新窗口seed0确认的[正面结果](CONFIRMATION.md)不能替代这里的三训练种子稳定性结论。

同一个seed0检查点在新终点评估中的P0为70/78/58%，历史冻结报告为74/76/60%。两组原始值均保留，没有选择更好的重测值；这提示应显式记录GPU规划实现的重复运行波动，而不能声称跨机器bitwise可复现。

这是小renderer、单frameskip相位、50k且一次初始化缩放的探索结果。原始全数据/10epoch网格、最终选择和十候选验证仍在继续；不能与25k基线表直接当作相同预算的最终排名。

[完整逐种子数值、逐例反事实/规划、配置和hash](L2-50K-STABILITY.json)。L2-50k-*-quality.npz保留逐例每horizon的误差；GPU媒体位于EFS。
"""
    (out/"L2-50K-STABILITY.md").write_text(text)
    for p in [out/"L2-50K-STABILITY.md",out/"L2-50k-seed-stability.svg"]:
        p.write_text("\n".join(x.rstrip() for x in p.read_text().splitlines()).rstrip()+"\n")
    manifest=json.loads((out/"manifest.json").read_text());manifest["files"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!="manifest.json"};(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
