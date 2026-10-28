"""Publish the doubled-state control without confusing active dimensions with usefulness."""
import argparse
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    campaign,out=Path(a.campaign),Path(a.output);records=[]
    for dimension,name in [(64,"eval-endpoint-source-L2-50k-seed0"),(128,"eval-L2-beta001-state128-50k-seed0")]:
        folder=campaign/name/"results"
        assert json.loads((campaign/"exits"/(name+".json")).read_text())["returncode"]==0
        assert json.loads((folder/"status.json").read_text())["state"]=="finished"
        q=json.loads((folder/"quality/metrics.json").read_text())
        cf=json.loads((folder/"counterfactual/counterfactual.json").read_text())
        collapse=json.loads((folder/"collapse/metrics.json").read_text())
        plans=[json.loads((folder/"planning"/("seed"+str(seed))/"metrics.json").read_text()) for seed in [42,43,44]]
        records.append(dict(state_dim=dimension,quality=q,counterfactual=cf,collapse=collapse,planning=plans,
                            checkpoint=json.loads((folder/"checkpoint_manifest.json").read_text())))
    (out/"STATE_CAPACITY.json").write_text(json.dumps(dict(records=records,scope="Seed0only,same50kstepsandtargetdata;statecapacitydoubledwithsameinitialstdandbeta,entry/surprisecostandparametercountthereforechange. Notaniso-rateoriso-computecomparison."),indent=2))
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","savefig.bbox":"tight"})
    fig,axes=plt.subplots(1,3,figsize=(11,3.8));x=np.arange(2)
    for ax,values,label in [
        (axes[0],[r["quality"]["h7_agent_ratio"] for r in records],"h7 agent error / copy"),
        (axes[1],[r["counterfactual"]["action_identification_accuracy"]*100 for r in records],"Counterfactual action ID (%)"),
        (axes[2],[np.mean([p["metrics"]["success_rate"] for p in r["planning"]]) for r in records],"P0 success (%)")]:
        bars=ax.bar(x,values,color=["#1976b9","#db7b25"],alpha=.75)
        ax.bar_label(bars,fmt="%.2f",padding=3)
        ax.set_xticks(x,["64D","128D"]);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
        ax.set_ylim(0,max(1 if ax==axes[0] else 100,max(values)*1.15))
    fig.suptitle("More active dimensions did not improve this seed-0 control\nSame50kupdatesandbeta;bothpassnoisy-posteriorcollapsecheck",fontsize=12)
    fig.tight_layout()
    for ext in ["png","svg"]:fig.savefig(out/("state-capacity."+ext),dpi=180)
    plt.close(fig)
    first,second=records
    text=f"""# 64→128维容量检查：该单种子对照退化

原成功seed0的64个state维度都active，因此按计划加倍token宽度到128维，同样训练50k。128维仍全部active，并通过含噪后验重建坍塌检测；但生成与控制退化：

| state维度 | h7 agent /moving-block误差比 | 动作识别 | P0均值 |
| --- | --- | --- | --- |
"""
    for r in records:
        text+=f'| {r["state_dim"]} | {r["quality"]["h7_agent_ratio"]:.3f} / {r["quality"]["h7_moving_block_ratio"]:.3f} | {r["counterfactual"]["action_identification_accuracy"]*100:.2f}% | {np.mean([p["metrics"]["success_rate"] for p in r["planning"]]):.2f}% |\n'
    text+="""
![状态容量](state-capacity.png)

active_dims衡量的是本定义下每维支付的可变rate，不是有效任务信息。128维“活跃”与非空后验并未换来更强动作控制。这个对照不足以证明64维最优：加倍容量同时增加了总rate、模型参数和计算，只有一个训练种子，未做等rate比较。应把它作为“盲目加维度未解决当前问题”的负面证据。

[完整各horizon、逐维rate与规划记录](STATE_CAPACITY.json)。原始全数据10epoch cohort的选定配置还会按相同条件做自己的容量检查。
"""
    (out/"STATE_CAPACITY.md").write_text(text)
    for f in [out/"STATE_CAPACITY.md",out/"state-capacity.svg"]:
        f.write_text("\n".join(line.rstrip() for line in f.read_text().splitlines()).rstrip()+"\n")
    manifest=json.loads((out/"manifest.json").read_text());manifest["files"]={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!="manifest.json"};(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("CAPACITY_REPORT_COMPLETE")


if __name__=="__main__":
    main()
