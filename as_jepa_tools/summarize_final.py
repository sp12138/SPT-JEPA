"""Summarize completed finalizers with episode-clustered uncertainty and review plots."""
import argparse,csv,datetime,json,math
from pathlib import Path
import numpy as np


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None


def interval_ratio(numerator,denominator,episodes):
    unique,inverse=np.unique(np.asarray(episodes),return_inverse=True)
    a=np.bincount(inverse,weights=np.asarray(numerator,dtype=float));b=np.bincount(inverse,weights=np.asarray(denominator,dtype=float))
    weights=np.random.default_rng(79).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    values=(weights@a)/np.maximum(weights@b,1e-12)
    return dict(value=float(a.sum()/max(b.sum(),1e-12)),lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)),
                clusters=len(unique),method="95% episode-cluster bootstrap, conditional on the trained model and fixed test windows")


def generation(root):
    metric=read(root/"metrics.json")
    if not metric:return None
    records=read(root/"cases.json");arrays=np.load(root/"per_case.npz")
    result={k:metric.get(k) for k in ["cases","sampler_steps","steps","samples","h7_agent_ratio","h7_block_ratio","h7_moving_block_ratio","pose_weights","feature_forward_precision","feature_precision"]}
    episodes=np.array([r["episode"] for r in records]);moving=np.array([r.get("moving_block",False) for r in records])
    for key in ["agent","block"]:
        values=np.asarray(arrays[key]);copy=np.asarray(arrays["copy_"+key])
        if values.ndim==3:values=values.mean(1);copy=copy.mean(1)
        result[key+"_h7_absolute"]=float(values[:,6].mean())
        result["copy_"+key+"_h7_absolute"]=float(copy[:,6].mean())
        result[key+"_h7_ratio_ci"]=interval_ratio(values[:,6],copy[:,6],episodes)
        if key=="block" and moving.any():
            result["moving_block_h7_ratio_ci"]=interval_ratio(values[moving,6],copy[moving,6],episodes[moving])
    result["metrics_path"]=str(root/"metrics.json")
    return result


def r2_pair(anchor,noanchor,data):
    aa=np.load(anchor/"probe_predictions.npz");bb=np.load(noanchor/"probe_predictions.npz")
    y=aa["truth"][:,:8].astype(float);pa=aa["prediction"][:,:8].astype(float);pb=bb["prediction"][:,:8].astype(float)
    if not np.allclose(y,bb["truth"][:,:8]):raise AssertionError("Probe test labels do not match across the color pair")
    lengths=np.load(data/"lengths.npy");test=np.load(data/"test_episodes.npy")
    episodes=np.asarray([i for i,n in enumerate(lengths) if n>=1 and i in set(test.tolist())])
    ends=np.cumsum(lengths[episodes]);indices=np.random.default_rng(18).choice(int(ends[-1]),size=len(y),replace=False)
    identity=episodes[np.searchsorted(ends,indices,side="right")]
    unique,inverse=np.unique(identity,return_inverse=True);n=np.bincount(inverse)
    sums=np.stack([np.bincount(inverse,weights=y[:,k]) for k in range(8)],1)
    squares=np.stack([np.bincount(inverse,weights=y[:,k]**2) for k in range(8)],1)
    sse_a=np.stack([np.bincount(inverse,weights=(y[:,k]-pa[:,k])**2) for k in range(8)],1)
    sse_b=np.stack([np.bincount(inverse,weights=(y[:,k]-pb[:,k])**2) for k in range(8)],1)
    weights=np.random.default_rng(81).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    total_n=weights@n;total_sums=weights@sums;sst=weights@squares-total_sums**2/total_n[:,None]
    ra=1-(weights@sse_a)/np.maximum(sst,1e-12);rb=1-(weights@sse_b)/np.maximum(sst,1e-12)
    result=dict(test_frames=len(y),test_episodes=len(unique),scope="Paired 95% episode-cluster bootstrap; conditional on these trained weights")
    for name,sl in [("agent",slice(0,2)),("block",slice(2,4))]:
        delta=(ra[:,sl]-rb[:,sl]).mean(1)
        result[name+"_r2_difference_ci"]=[float(np.quantile(delta,q)) for q in [.025,.5,.975]]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    campaign=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[];pending=[]
    for root in sorted(campaign.glob("final-*")):
        status=read(root/"status.json");manifest=read(root/"checkpoint_manifest.json")
        if not status or status.get("state")!="finished":
            pending.append(dict(model=root.name,status=status.get("state","not_started") if status else "not_started"))
            continue
        cfg=manifest["model_config"];parent=Path(manifest["parent"]);train_cfg=read(parent/"config.json")
        row=dict(model=parent.name,step=manifest["checkpoint_step"],mode=cfg["mode"],domain=cfg.get("domain","standard"),
                 seed=train_cfg["seed"],beta=cfg.get("beta"),renderer_width=cfg.get("renderer_width",192),renderer_depth=cfg.get("renderer_depth",6),
                 calibrate_state_std=train_cfg.get("calibrate_state_std"),precision=train_cfg.get("precision","bf16-mixed"),
                 reached_max_steps=manifest["reached_max_steps"],output=str(root),parameter_counts=read(parent/"model_size.json"))
        training_source=read(parent/"actual_source.json") or read(parent/"source_manifest.json") or read(parent/"execution.json") or {}
        row["training_source"]=training_source
        row["resume_from"]=read(parent/"resume_from.json")
        source_path=Path(training_source.get("source","/nonexistent"))/"as_jepa_lewm/train.py"
        source_text=source_path.read_text() if source_path.exists() else ""
        row["sampler_rule"]="legacy iterator-advance cursor" if "self.start=start+self.length" in source_text else ("completed-sample cursor" if source_text else "unknown")
        row["probes"]={name:read(root/"probes"/name/"probes.json") for name in ["mean","noisy","mean-fp32"]}
        row["generation_mean"]=generation(root/"generation/mean")
        row["generation_sample"]=generation(root/"generation/samples")
        plans=[]
        for pp in sorted((root/"planning").glob("seed*/metrics.json")):
            metric=read(pp);success=metric["metrics"]["episode_successes"]
            plans.append(dict(seed=metric["seed"],successes=int(sum(success)),episodes=len(success),success_rate=float(np.mean(success)*100),source=str(pp)))
        if plans:
            values=np.array([r["success_rate"] for r in plans])
            row["planning"]=dict(trials=plans,mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else None,
                                 scope="Mean and sample SD over evaluation seeds; training seeds remain separate")
        cf=read(root/"counterfactual/counterfactual.json")
        if cf:
            row["counterfactual"]={k:v for k,v in cf.items() if k!="details"}
            details=cf["details"];row["counterfactual"]["accuracy_ci"]=interval_ratio([np.mean(x["correct"]) for x in details],np.ones(len(details)),[x["episode"] for x in details])
        row["color_swaps"]=read(root/"colour-swaps/swaps.json")
        rows.append(row)
    lookup={r["model"]:r for r in rows};color_comparisons=[]
    for seed in [0,1]:
        an=lookup.get(f"L2-color-calibrated1-b025-seed{seed}");na=lookup.get(f"L2-color-noanchor-calibrated1-b025-seed{seed}")
        if an and na:
            pair=r2_pair(Path(an["output"])/"probes/noisy",Path(na["output"])/"probes/noisy",campaign.parent.parent/"data/pusht")
            color_comparisons.append(dict(seed=seed,**pair))
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=dict(collected_at=stamp,models=rows,pending=pending,color_probe_comparisons=color_comparisons,
                source_campaign=str(campaign),scope="Exploratory campaign; no claim of the original three-training-seed generation gate")
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    fields=["model","step","seed","mode","domain","beta","precision","sampler_rule","renderer_width","renderer_depth","plan_mean","plan_sd","h7_agent_ratio","h7_moving_block_ratio","cf_accuracy","agent_r2","block_r2","nuisance_r2","output"]
    with (out/"summary.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            pp=row["probes"].get("noisy") or row["probes"].get("mean") or {};gg=row.get("generation_mean") or {};plan=row.get("planning") or {}
            writer.writerow({**{k:row.get(k) for k in fields if k in row},"plan_mean":plan.get("mean"),"plan_sd":plan.get("std"),
                             "h7_agent_ratio":gg.get("h7_agent_ratio"),"h7_moving_block_ratio":gg.get("h7_moving_block_ratio"),
                             "cf_accuracy":row.get("counterfactual",{}).get("action_identification_accuracy"),
                             "agent_r2":pp.get("agent_r2"),"block_r2":pp.get("block_position_r2"),"nuisance_r2":pp.get("nuisance_r2")})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    names=[
        ("L0-standard-cursorfix-seed0","L0, matched seed 0"),("L0-standard-seed1","L0, matched seed 1"),("L0-standard-seed0","L0, legacy order seed 0"),
        ("L1-standard-cursorfix-seed0","L1, matched seed 0"),("L2-standard-b4-cursorfix-seed0","L2 beta4, matched seed 0"),("L1-standard-bn-dev2","L1 BN64, seed 0"),("L1-standard-bn-seed1","L1 BN64, seed 1"),("L1-standard-mlpbn-seed0","L1 MLP-BN"),
        ("L2-standard-calibrated-b001-seed0","L2 beta .01, seed 0"),("L2-standard-calibrated-b001-seed1","L2 beta .01, seed 1"),
        ("L2-standard-calibrated-b001-50k-seed0","L2 beta .01, 50k"),
        ("L2-standard-staged-b025-seed0","L2 staged beta .25"),
        ("L2-standard-b1-extended-dev13","L2 beta1, resumed"),("L2-standard-b1-seed1-continuous","L2 beta1, seed 1"),
        ("L2-standard-original-renderer-b001-budget","L2, original-size renderer")]
    chosen=[(lookup[n],label) for n,label in names if n in lookup]
    if chosen:
        fig,axes=plt.subplots(1,3,figsize=(15,max(4.5,len(chosen)*.48)),sharey=True,layout="constrained")
        y=np.arange(len(chosen));colors=["#375c93" if r["mode"]=="L0" else "#4e9a89" if r["mode"]=="L1" else "#cf8a43" for r,_ in chosen]
        labels=[f"{label} @{r['step']/1000:g}k" for r,label in chosen]
        for i,(row,label) in enumerate(chosen):
            pp=row.get("planning")
            if pp:axes[0].errorbar(pp["mean"],i,xerr=pp["std"] or 0,fmt="o",color=colors[i],capsize=3)
            gg=row.get("generation_mean")
            if gg:
                for ax,key in [(axes[1],"agent_h7_ratio_ci"),(axes[2],"moving_block_h7_ratio_ci")]:
                    ci=gg.get(key)
                    if ci:ax.errorbar(ci["value"],i,xerr=[[ci["value"]-ci["lower"]],[ci["upper"]-ci["value"]]],fmt="o",color=colors[i],capsize=3)
        axes[0].set_yticks(y,labels);axes[0].set(xlim=(-2,102),xlabel="P0 success (%), mean +/- SD",title="Three evaluation seeds")
        axes[0].invert_yaxis()
        axes[1].axvline(.5,color="#b64d4d",linestyle="--",linewidth=1);axes[1].axvline(1,color="#888888",linestyle=":",linewidth=1)
        axes[1].set(xlabel="h7 agent error / copy error",title="256 fixed windows")
        axes[2].axvline(1,color="#888888",linestyle=":",linewidth=1);axes[2].set(xlabel="h7 moving-block error / copy error",title="Moving-block subset")
        fig.supxlabel("Generation intervals: 95% episode-cluster bootstrap, conditional on each trained model. Training seeds are shown separately.",fontsize=9)
        fig.savefig(out/"final-comparison.png",bbox_inches="tight");fig.savefig(out/"final-comparison.svg",bbox_inches="tight");plt.close(fig)
    color=[r for r in rows if r["domain"]=="color" and r["model"].startswith(("L2-color-calibrated1-b025","L2-color-noanchor-calibrated1-b025","L1-color-bn"))]
    if color:
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
        labels=[];agent=[];block=[];nuisance=[]
        for row in color:
            pp=row["probes"].get("noisy") or row["probes"]["mean"]
            label=("L1 BN" if row["mode"]=="L1" else "No anchor" if "noanchor" in row["model"] else "Anchor")+f", s{row['seed']}"
            labels.append(label);agent.append(pp["agent_r2"]);block.append(pp["block_position_r2"]);nuisance.append(pp.get("nuisance_r2",0))
        x=np.arange(len(labels));w=.25
        for dx,value,label,color_value in [(-w,agent,"Agent","#375c93"),(0,block,"Block","#4e9a89"),(w,nuisance,"Palette","#cf8a43")]:axes[0].bar(x+dx,value,w,label=label,color=color_value)
        axes[0].set_xticks(x,labels,rotation=25,ha="right");axes[0].set(ylim=(-.05,1),ylabel="Held-out linear probe R2",title="Dynamics and nuisance decodability");axes[0].legend(frameon=False)
        labels=[];anchor=[];state=[]
        for row in color:
            swap=row.get("color_swaps")
            if not swap:continue
            labels.append(("L1 BN" if row["mode"]=="L1" else "No anchor" if "noanchor" in row["model"] else "Anchor")+f", s{row['seed']}")
            anchor.append(swap["anchor_style_transfer_projection"]);state.append(swap["state_style_transfer_projection"])
        x=np.arange(len(labels));axes[1].bar(x-.18,anchor,.35,label="Change anchor",color="#375c93");axes[1].bar(x+.18,state,.35,label="Change state",color="#4e9a89")
        axes[1].set_xticks(x,labels,rotation=25,ha="right");axes[1].set(ylabel="Fraction of donor background palette transferred",title="Common-noise interventions");axes[1].legend(frameon=False)
        fig.savefig(out/"color-mechanism.png",bbox_inches="tight");fig.savefig(out/"color-mechanism.svg",bbox_inches="tight");plt.close(fig)
    text=["# SPT-JEPA：统一评估结果",f"核查时间：{stamp}","",
          "这是单次探索性 campaign 的结果。不同训练种子分别列出；规划为 3 个评估种子的均值与标准差，生成比值的 95% 区间按测试 episode 重采样。未完成项目不会算作通过。",
          "","| 模型 | 步数 | P0 均值±SD (%) | h7 agent / copy | h7 moving block / copy | 反事实动作识别 |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def num(value,digits=3):return "—" if value is None else f"{value:.{digits}f}"
    for row in rows:
        pp=row.get("planning",{});gg=row.get("generation_mean") or {};cf=row.get("counterfactual",{})
        plan="—" if not pp else num(pp["mean"],1)+"±"+num(pp.get("std"),1)
        text.append(f"| {row['model']} | {row['step']} | {plan} | {num(gg.get('h7_agent_ratio'))} | {num(gg.get('h7_moving_block_ratio'))} | {num(cf.get('action_identification_accuracy'))} |")
    text+=["","背景颜色交换与噪声状态探针见 color-mechanism.png。完整逐配置协议与区间在 summary.json，简表在 summary.csv。","",
           "旧 L0 seed0 与旧 L1 无 BN 使用修复前的迭代器游标。主比较优先使用 cursorfix seed0 与修复后的 seed1；源版本和恢复来源在 summary.json 中明确记录。","","原方案要求三训练种子、h7 agent 不超过复制基线的 0.5 倍、移动 block 优于复制，以及反事实和人工画面检查。本轮只用于关键 insight 筛选，不能把单项表征、规划、方差或低 loss 当作通过这个门槛。","",
           "LPIPS、SSIM 和 FVD 未测。权威检查点、视频、逐 case 数据与日志留在 EFS，未进入 Git。","",
           "尚未完成："+(", ".join(r["model"] for r in pending) if pending else "无")]
    (out/"RESULTS.md").write_text("\n".join(text)+"\n")
    print(json.dumps(dict(completed=len(rows),pending=len(pending),color_pairs=len(color_comparisons),output=str(out)),indent=2))


if __name__=="__main__":main()
