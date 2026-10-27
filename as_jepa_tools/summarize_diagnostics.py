"""Collect final diagnostic evidence without modifying experiment outputs."""
import argparse,csv,datetime,json
from pathlib import Path
import numpy as np
from as_jepa_tools.summarize_final import read,interval_ratio,generation


def paired_generation(anchor,noanchor,moving_only=False,entity="block"):
    ca=read(anchor/"cases.json");cb=read(noanchor/"cases.json")
    assert [(r["episode"],r["start"]) for r in ca]==[(r["episode"],r["start"]) for r in cb]
    a=np.load(anchor/"per_case.npz");b=np.load(noanchor/"per_case.npz")
    assert np.allclose(a["copy_"+entity],b["copy_"+entity])
    keep=np.array([r.get("moving_block",False) for r in ca]) if moving_only else np.ones(len(ca),bool)
    episodes=np.array([r["episode"] for r in ca])[keep]
    aa=a[entity][keep,6].astype(float);bb=b[entity][keep,6].astype(float);den=a["copy_"+entity][keep,6].astype(float)
    unique,inverse=np.unique(episodes,return_inverse=True)
    sums=np.bincount(inverse,weights=aa-bb);base=np.bincount(inverse,weights=den)
    weights=np.random.default_rng(83).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    values=(weights@sums)/np.maximum(weights@base,1e-12)
    return dict(value=float(sums.sum()/base.sum()),lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)),
                cases=len(aa),episodes=len(unique),meaning="anchor minus no-anchor error/copy; negative favors anchor",
                method="Paired 95% episode-cluster bootstrap, conditional on trained weights")


def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    c=Path(a.campaign);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);pending=[];probes=[]
    for name in ["late-probes-standard-dev2","late-probes-native-dev2"]:
        root=c/name;status=read(root/"results/status.json") or {}
        if status.get("state")!="finished" or any(r.get("error") for r in status.get("results",[])):
            pending.append(dict(job=name,state=status.get("state"),errors=[r.get("error") for r in status.get("results",[]) if r.get("error")]))
        manifests=read(root/"checkpoint_manifests.json") or []
        for manifest in manifests:
            variants={}
            for file in sorted((root/"results"/manifest["model"]).glob("*/probes.json")):
                metric=read(file)
                variants[file.parent.name]={k:v for k,v in metric.items() if k in ["agent_r2","block_position_r2","block_angle_r2","velocity_r2","feature_std","feature_precision","feature_type","nuisance_r2","r2"]}
            probes.append(dict(model=manifest["model"],step=manifest["step"],weights_sha256=manifest["weights_sha256"],checkpoint=manifest["checkpoint"],origin=manifest["origin"],variants=variants))
    reference=read(c/"final-L2-standard-calibrated-b001-seed0/generation/mean/cases.json")
    native=[]
    for job in read(c/"late_native_quality_registry.json") or []:
        root=Path(job["output"]);status=read(root/"status.json") or {};metric=read(root/"generation/metrics.json")
        if status.get("state")!="finished" or not metric:
            pending.append(dict(job=job["id"],state=status.get("state")));continue
        cases=read(root/"generation/cases.json")
        assert [(r["episode"],r["start"]) for r in cases]==[(r["episode"],r["start"]) for r in reference]
        arr=np.load(root/"generation/per_case.npz");episodes=np.array([r["episode"] for r in cases]);moving=np.array([r["moving_block"] for r in reference])
        result=dict(model=job["id"].removeprefix("quality-late-"),source=read(root/"actual_source.json"),checkpoint=read(root/"checkpoint_manifest.json"),
                    cases=len(cases),moving_cases=int(moving.sum()),common_case_ids_verified=True,
                    metrics={k:v for k,v in metric.items() if "ratio" in k or "by_horizon" in k or k in ["sampler_steps","feature_forward_precision","pose_weights"]})
        result["horizons"]={}
        for h in [1,3,7,13]:
            result["horizons"][str(h)]={}
            for entity in ["agent","block"]:
                ci=interval_ratio(arr[entity][:,h-1],arr["copy_"+entity][:,h-1],episodes)
                result["horizons"][str(h)][entity]=dict(error_px=float(arr[entity][:,h-1].mean()),copy_px=float(arr["copy_"+entity][:,h-1].mean()),ratio_ci=ci)
                if h==7:result[entity+"_h7_ci"]=ci
        result["moving_block_h7_ci"]=interval_ratio(arr["block"][moving,6],arr["copy_block"][moving,6],episodes[moving])
        native.append(result)
    teacher_forcing=[]
    for job in read(c/"native_context_registry.json") or []:
        root=Path(job["output"]);status=read(root/"status.json") or {};score=read(root/"results/teacher_forcing.json")
        if status.get("state")!="finished" or not score or not score.get("control",{}).get("passed"):
            pending.append(dict(job=job["id"],state=status.get("state")));continue
        teacher_forcing.append(dict(job=job["id"],source=read(root/"actual_source.json"),**score))
    feedback=[];geometry=[]
    for registry,destination,filename in [("native_feedback_registry.json",feedback,"feedback_clipping.json"),("color_geometry_registry.json",geometry,"swaps.json")]:
        for job in read(c/registry) or []:
            root=Path(job["output"]);status=read(root/"status.json") or {};score=read(root/"results"/filename)
            if status.get("state")!="finished" or not score:
                pending.append(dict(job=job["id"],state=status.get("state")));continue
            destination.append(dict(job=job["id"],parent=job.get("model"),source=read(root/"actual_source.json"),matched_control=read(root/"control.json"),**score))
    decomposition=[]
    for file in sorted(c.glob("decode-decomposition-*/results/*-n128/decomposition.json")):
        decomposition.append(dict(model=file.parent.name.removesuffix("-n128"),**read(file)))
    color=[]
    for seed in [0,1]:
        anchor=c/f"final-L2-color-calibrated1-b025-seed{seed}"/"generation/mean";noanchor=c/f"final-L2-color-noanchor-calibrated1-b025-seed{seed}"/"generation/mean"
        if (anchor/"metrics.json").exists() and (noanchor/"metrics.json").exists():
            color.append(dict(seed=seed,agent=paired_generation(anchor,noanchor,entity="agent"),
                              moving_block=paired_generation(anchor,noanchor,moving_only=True,entity="block")))
    dot=[]
    for name in ["L2-dot-flow-calibrated-b001-seed0","L2-dot-mse-calibrated-b001-seed0","L1-dot-bn-seed0"]:
        root=c/("final-"+name);pp=read(root/"probes/noisy/probes.json") or read(root/"probes/mean/probes.json")
        if pp:dot.append(dict(model=name,probe=pp,metrics=read(root/"generation/mean/metrics.json")))
    budget=[]
    def budget_row(label,step,training_budget,probe_root,gen_root,plan_files,cf_root=None):
        probe=read(probe_root/"noisy/probes.json");gen=generation(gen_root);plans=[]
        if not probe or not gen:return None
        for file in plan_files:
            score=read(file)
            if not score:return None
            plans.append(dict(seed=score["seed"],success_rate=score["metrics"]["success_rate"],episodes=len(score["metrics"]["episode_successes"]),path=str(file)))
        assert len(plans)==3
        row=dict(label=label,step=step,training_budget=training_budget,noisy_agent_r2=probe["agent_r2"],noisy_block_r2=probe["block_position_r2"],generation=gen,
                 planning=dict(trials=plans,mean=float(np.mean([r["success_rate"] for r in plans])),std=float(np.std([r["success_rate"] for r in plans],ddof=1))),generation_root=str(gen_root))
        cf=read(cf_root/"counterfactual.json") if cf_root else None
        if cf:
            details=cf["details"];row["counterfactual"]=dict(value=cf["action_identification_accuracy"],oracle=cf["oracle_pose_action_identification_accuracy"],interval=interval_ratio([np.mean(r["correct"]) for r in details],np.ones(len(details)),[r["episode"] for r in details]))
        return row
    short=c/"final-L2-standard-calibrated-b001-seed0";long=c/"final-L2-standard-calibrated-b001-50k-seed0";mid=c/"budget24k-main-dev2"
    for label,step,training_budget,probe_root,gen_root,plan_files,cf_root in [
        ("25k budget @25k",25000,25000,short/"probes",short/"generation/mean",[short/f"planning/seed{seed}/metrics.json" for seed in [42,43,44]],short/"counterfactual"),
        ("50k budget @24k",24000,50000,mid/"probes",mid/"generation",[c/f"budget24k-planning-seed{seed}-dev2/results/metrics.json" for seed in [42,43,44]],None),
        ("50k budget @50k",50000,50000,long/"probes",long/"generation/mean",[long/f"planning/seed{seed}/metrics.json" for seed in [42,43,44]],long/"counterfactual")]:
        row=budget_row(label,step,training_budget,probe_root,gen_root,plan_files,cf_root)
        if row:budget.append(row)
        else:pending.append(dict(job=label,state="budget evidence incomplete"))
    if len(budget)==3:
        identities=[[(x["episode"],x["start"]) for x in read(Path(row["generation_root"])/"cases.json")] for row in budget]
        assert identities[0]==identities[1]==identities[2],"Budget comparison case identities differ"
    result=dict(collected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),probes=probes,native=native,
                decomposition=decomposition,native_context=teacher_forcing,native_feedback=feedback,color_geometry=geometry,budget_comparison=budget,color_generation_comparisons=color,dot=dot,pending=pending,
                scope="Final pinned diagnostics; single training seeds unless listed; uncertainty is conditional on trained weights")
    (out/"diagnostics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/"late-probes.csv").open("w") as f:
        fields=["model","step","variant","agent_r2","block_position_r2","feature_std","weights_sha256"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in probes:
            for variant,metric in row["variants"].items():w.writerow(dict(model=row["model"],step=row["step"],variant=variant,weights_sha256=row["weights_sha256"],**{k:metric[k] for k in ["agent_r2","block_position_r2","feature_std"]}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    decomp_order=["L2-standard-calibrated-b001-seed0","L2-standard-calibrated-b001-50k-seed0","L1-standard-bn-dev2"]
    chosen=sorted([r for r in decomposition if r["model"] in decomp_order],key=lambda r:decomp_order.index(r["model"]))
    if chosen:
        fig,axes=plt.subplots(2,len(chosen),figsize=(6*len(chosen),7.5),squeeze=False,layout="constrained")
        styles=[("posterior_mean","True-future state (oracle)","#375c93"),("forecast_mean","Forecast state","#cf8a43"),("copy","Copy context","#777777"),("real_floor","Real-frame pose floor","#4e9a89")]
        for col,row in enumerate(chosen):
            for axis,entity in zip(axes[:,col],["agent","block"]):
                for key,label,color_value in styles:axis.plot(row["horizons"],row["mean_error"][key+"_"+entity],marker="o",label=label,color=color_value)
                axis.set(xlabel="Forecast horizon",ylabel=entity.capitalize()+" position error (px)",xticks=row["horizons"],ylim=(0,None))
                axis.set_title(("L2 beta .01 @50k" if "50k" in row["model"] else "L2 beta .01 @25k" if row["model"].startswith("L2") else "L1 BN64 @25k")+" - "+entity)
        axes[0,0].legend(frameon=False,fontsize=9);fig.supxlabel("128 fixed held-out cases. Oracle states encode actual future images; this is a diagnostic, not deployable prediction.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("posterior-forecast."+suffix),bbox_inches="tight")
        plt.close(fig)
    if dot:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained");labels=["Flow beta .01","MSE beta .01","L1 BN64"];x=np.arange(len(dot));width=.2
        for offset,key,label,color_value in [(-width,"agent_r2","Agent","#375c93"),(0,"block_position_r2","Block","#4e9a89"),(width,"nuisance_r2","Dot (all 5 properties)","#cf8a43")]:
            axes[0].bar(x+offset,[r["probe"].get(key,0) for r in dot],width,label=label,color=color_value)
        axes[0].set(xticks=x,xticklabels=labels,ylabel="Held-out probe R2",ylim=(-.05,1),title="Dynamics versus random-dot information");axes[0].legend(frameon=False,fontsize=8)
        keys=["dot_x","dot_y","dot_r","dot_g","dot_b"]
        for i,row in enumerate(dot):axes[1].bar(np.arange(5)+(i-1)*.25,[row["probe"]["r2"][k] for k in keys],.25,label=labels[i])
        axes[1].set(xticks=np.arange(5),xticklabels=["x","y","red","green","blue"],ylabel="Held-out probe R2",ylim=(-.05,1),title="The flow model retains dot RGB");axes[1].legend(frameon=False,fontsize=8)
        fig.supxlabel("15k updates, seed 0, same data and state size. Low-beta result does not establish behavior at beta >= 1.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("dot-information."+suffix),bbox_inches="tight")
        plt.close(fig)
    pixel=[r for r in native if r["model"].startswith(("S0-","S1-"))]
    if pixel:
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout="constrained");horizons=[1,3,7,13]
        for axis,entity in zip(axes,["agent","block"]):
            for row in pixel:axis.plot(horizons,[row["horizons"][str(h)][entity]["error_px"] for h in horizons],marker="o",label=row["model"][:2]+", 24k")
            axis.plot(horizons,[pixel[0]["horizons"][str(h)][entity]["copy_px"] for h in horizons],linestyle="--",marker="o",color="#777777",label="Copy context")
            axis.set(xlabel="Forecast horizon",ylabel=entity.capitalize()+" position error (px)",xticks=horizons,ylim=(0,None));axis.legend(frameon=False)
            axis.set_title("Native pixel autoregression - "+entity)
        fig.supxlabel("256 fixed held-out windows, seed 0, 50 Euler steps. Good one-step agent prediction does not persist through rollout.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("native-rollout-drift."+suffix),bbox_inches="tight")
        plt.close(fig)
    if teacher_forcing:
        fig,axes=plt.subplots(2,len(teacher_forcing),figsize=(6*len(teacher_forcing),7.5),squeeze=False,layout="constrained");horizons=[1,3,7,13]
        for col,row in enumerate(teacher_forcing):
            for axis,entity in zip(axes[:,col],["agent","block"]):
                for key,label,color_value in [("teacher_forced_ratio_ci","True past frames (oracle)","#375c93"),("autoregressive_ratio_ci","Generated past frames","#cf8a43")]:
                    ci=[row["horizons"][str(h)][entity][key] for h in horizons];values=np.array([x["value"] for x in ci]);lower=np.array([x["lower"] for x in ci]);upper=np.array([x["upper"] for x in ci])
                    axis.errorbar(horizons,values,yerr=np.stack([values-lower,upper-values]),marker="o",label=label,color=color_value,capsize=3)
                axis.axhline(1,color="#777777",linestyle="--",label="Copy context")
                axis.set(xlabel="Forecast horizon",ylabel=entity.capitalize()+" error / copy error",xticks=horizons,ylim=(0,None),title=row["model"]+" - "+entity)
        axes[0,0].legend(frameon=False,fontsize=9)
        fig.supxlabel("256 fixed cases; same checkpoints, actions, anchor and sampler noise. Privileged true history is diagnostic only. 95% episode-cluster intervals.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("native-context-intervention."+suffix),bbox_inches="tight")
        plt.close(fig)
    if feedback:
        row=feedback[0];oracle=next((x for x in teacher_forcing if x["model"]==row["model"]),None);horizons=[1,3,7,13]
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
        for axis,entity in zip(axes,["agent","block"]):
            curves=[(row,"autoregressive_ratio_ci","Canonical generated history","#cf8a43"),(row,"clipped_feedback_ratio_ci","Clip history to training pixel range","#4e9a89")]
            if oracle:curves.append((oracle,"teacher_forced_ratio_ci","True past frames (oracle)","#375c93"))
            for source,key,label,color_value in curves:
                ci=[source["horizons"][str(h)][entity][key] for h in horizons];value=np.array([x["value"] for x in ci]);lower=np.array([x["lower"] for x in ci]);upper=np.array([x["upper"] for x in ci])
                axis.errorbar(horizons,value,yerr=np.stack([value-lower,upper-value]),marker="o",color=color_value,label=label,capsize=3)
            axis.axhline(1,color="#777777",linestyle="--");axis.set(xticks=horizons,xlabel="Forecast horizon",ylabel=entity.capitalize()+" error / copy error",ylim=(0,None),title="Native "+row["model"]+" - "+entity)
        axes[0].legend(frameon=False,fontsize=8);fig.supxlabel("256 same cases, weights, actions, anchor and sampling noise; clipping changes only generated encoder history. Oracle uses privileged past frames.\n95% episode-cluster intervals, conditional on trained weights.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("native-feedback-clipping."+suffix),bbox_inches="tight")
        plt.close(fig)
    anchor_geometry=[r for r in geometry if r.get("parent","").startswith("L2-color-calibrated1-")]
    if anchor_geometry:
        fig,axes=plt.subplots(1,len(anchor_geometry),figsize=(6*len(anchor_geometry),4.7),squeeze=False,layout="constrained");x=np.arange(3);keys=["own","anchor_swap","state_swap"]
        for axis,row in zip(axes[0],anchor_geometry):
            axis.bar(x-.18,[row["mean"][k+"_block_error"] for k in keys],.36,label="Error to own block",color="#375c93")
            axis.bar(x+.18,[row["mean"][k+"_block_donor_error"] for k in keys],.36,label="Error to donor block",color="#cf8a43")
            axis.set_xticks(x,["Own state / own anchor","Own state / donor anchor","Donor state / own anchor"],rotation=18,ha="right")
            axis.set(ylabel="Mean block position error (px)",title="Anchor model, "+row["parent"].split("-")[-1],ylim=(0,1.3*max(row["mean"][k+suffix] for k in keys for suffix in ["_block_error","_block_donor_error"])));axis.legend(frameon=False,fontsize=9)
        fig.supxlabel("128 matched posterior-state interventions per seed; same donor pairing and rendering noise. Means are conditional on each trained model.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("color-geometry-interventions."+suffix),bbox_inches="tight")
        plt.close(fig)
    if len(budget)==3:
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained");x=np.arange(3);labels=[r["label"] for r in budget]
        for offset,key,label,color_value in [(-.18,"noisy_agent_r2","Agent","#375c93"),(.18,"noisy_block_r2","Block","#4e9a89")]:axes[0,0].bar(x+offset,[r[key] for r in budget],.36,label=label,color=color_value)
        axes[0,0].set(ylabel="Held-out noisy-code probe R2",ylim=(0,1),title="State information");axes[0,0].legend(frameon=False)
        for offset,key,label,color_value in [(-.08,"agent_h7_ratio_ci","Agent","#375c93"),(.08,"moving_block_h7_ratio_ci","Moving block","#4e9a89")]:
            ci=[r["generation"][key] for r in budget];value=np.array([z["value"] for z in ci]);low=np.array([z["lower"] for z in ci]);high=np.array([z["upper"] for z in ci]);axes[0,1].errorbar(x+offset,value,yerr=np.stack([value-low,high-value]),fmt="o",color=color_value,capsize=3,label=label)
        axes[0,1].axhline(1,color="#777777",linestyle="--");axes[0,1].axhline(.5,color="#b64d4d",linestyle=":");axes[0,1].set(ylabel="h7 error / copy error",ylim=(0,None),title="256 fixed windows, mean states");axes[0,1].legend(frameon=False)
        axes[1,0].bar(x,[r["planning"]["mean"] for r in budget],yerr=[r["planning"]["std"] for r in budget],capsize=3,color=["#9ba8b8","#6e8eb7","#375c93"]);axes[1,0].set(ylabel="P0 success (%)",ylim=(0,100),title="Three evaluation seeds: mean +/- SD")
        cf_rows=[r for r in budget if r.get("counterfactual")];cx=np.arange(len(cf_rows));value=np.array([r["counterfactual"]["value"]*100 for r in cf_rows]);low=np.array([r["counterfactual"]["interval"]["lower"]*100 for r in cf_rows]);high=np.array([r["counterfactual"]["interval"]["upper"]*100 for r in cf_rows])
        axes[1,1].bar(cx,value,yerr=np.stack([value-low,high-value]),capsize=3,color=["#9ba8b8","#375c93"]);axes[1,1].axhline(25,color="#777777",linestyle="--",label="Chance");axes[1,1].axhline(cf_rows[-1]["counterfactual"]["oracle"]*100,color="#4e9a89",linestyle=":",label="Real-frame pose reference")
        axes[1,1].set_xticks(cx,[r["label"] for r in cf_rows],rotation=12,ha="right");axes[1,1].set(ylabel="Counterfactual action ID (%)",ylim=(0,100),title="64 reset states, four action sequences");axes[1,1].legend(frameon=False,fontsize=8)
        for axis in [axes[0,0],axes[0,1],axes[1,0]]:axis.set_xticks(x,labels,rotation=12,ha="right")
        fig.supxlabel("Seed 0. The last two checkpoints are one continuous 50k run; the short run has a different cosine budget. Generation/CF intervals condition on trained weights.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("training-budget-evidence."+suffix),bbox_inches="tight")
        plt.close(fig)
        with (out/"budget-comparison.csv").open("w") as f:
            fields=["label","step","training_budget","noisy_agent_r2","noisy_block_r2","h7_agent_ratio","h7_moving_block_ratio","planning_mean","planning_sd"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for row in budget:w.writerow({**{k:row[k] for k in fields if k in row},"h7_agent_ratio":row["generation"]["h7_agent_ratio"],"h7_moving_block_ratio":row["generation"]["h7_moving_block_ratio"],"planning_mean":row["planning"]["mean"],"planning_sd":row["planning"]["std"]})
    names=[("L2-standard-calibrated-b025-resume-dev15","std .1, beta .25"),("L2-standard-calibrated1-b025-seed0","std 1, beta .25"),("L2-standard-calibrated1-b4-seed0","std 1, beta 4"),("L2-standard-direct-calibrated1-b4-seed0","Direct prior"),("L2-standard-w16g4-calibrated1-b4-seed0","W16 / gap4"),("L2-standard-warmencoder-b4-seed0","Warm encoder, beta 4")]
    lookup={r["model"]:r for r in probes};chosen=[(lookup[n],label) for n,label in names if n in lookup]
    if chosen:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained");x=np.arange(len(chosen))
        for offset,key,label in [(-.18,"mean","BF16"),(.18,"mean-fp32","FP32")]:axes[0].bar(x+offset,[r["variants"][key]["feature_std"] for r,_ in chosen],.36,label=label)
        axes[0].set(yscale="log",ylabel="Feature standard deviation",title="Same pinned weights, different forward precision");axes[0].legend(frameon=False)
        for offset,key,label in [(-.18,"agent_r2","Agent"),(.18,"block_position_r2","Block")]:axes[1].bar(x+offset,[r["variants"]["noisy-fp32"][key] for r,_ in chosen],.36,label=label)
        axes[1].set(ylabel="Noisy FP32 probe R2",ylim=(-.02,1),title="Task information after fixed code noise");axes[1].legend(frameon=False)
        for axis in axes:axis.set_xticks(x,[label for _,label in chosen],rotation=30,ha="right")
        fig.supxlabel("25k final checkpoints. A large BF16 feature variance alone is insufficient evidence of useful state information.",fontsize=9)
        for suffix in ["png","svg"]:fig.savefig(out/("precision-diagnostics."+suffix),bbox_inches="tight")
        plt.close(fig)
    gallery_path=long/"generation/mean/gallery.png"
    if gallery_path.exists():
        from PIL import Image,ImageDraw
        original=Image.open(gallery_path).convert("RGB");tile=original.width//8;cases=read(long/"generation/mean/cases.json")[:16]
        assert len(cases)==16 and all(r.get("moving_block") for r in cases),"First16 gallery must match moving-block metadata"
        canvas=Image.new("RGB",(original.width+94,original.height+48),"white");canvas.paste(original,(94,48));draw=ImageDraw.Draw(canvas)
        draw.text((6,3),"beta=0.01, 50k, seed 0 | first 16 fixed moving-block cases; no error-based selection",fill="black")
        for j,label in enumerate(["GT h1","50k h1","GT h3","50k h3","GT h7","50k h7","GT h13","50k h13"]):draw.text((94+j*tile+3,28),label,fill="black")
        for i,record in enumerate(cases):draw.text((4,48+i*tile+tile//3),"ep"+str(record["episode"])+"\nstart "+str(record["start"]),fill="black")
        canvas.save(out/"rollout-50k-first16.png");(out/"rollout-50k-first16-cases.json").write_text(json.dumps(dict(cases=cases,selection="First16 of fixed256 evaluation order; all are in the moving-block subset; no error-based selection",horizons=[1,3,7,13],checkpoint=str(long/"checkpoint"),scope="Qualitative examples; aggregate performance is measured on all256 cases"),indent=2))
    print(json.dumps(dict(output=str(out),probe_models=len(probes),native_completed=len(native),color_pairs=len(color),decompositions=len(decomposition),native_context=len(teacher_forcing),native_feedback=len(feedback),color_geometry=len(geometry),budget_points=len(budget),pending=pending),indent=2))


if __name__=="__main__":main()
