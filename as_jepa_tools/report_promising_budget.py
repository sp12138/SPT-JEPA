"""Report selected budget endpoints with exact original and retuning controls."""
import argparse,json,hashlib,statistics,copy
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);parser.add_argument("--scope",choices=["epoch3","components"],required=True);parser.add_argument("--output",required=True);a=parser.parse_args()
    b=Path(a.root);r=b/"runs/promising_20261029";out=Path(a.output);plan=read(r/"plan.json")
    phase="selected-epoch3" if a.scope=="epoch3" else "component-epoch2";budget=37473 if a.scope=="epoch3" else 24982
    jobs=[j for j in plan["jobs"] if j["phase"]==phase or j.get("endpoint_phase")==phase]
    assert len(jobs)==40
    for j in jobs:assert read(r/"exits"/(j["id"]+".json"))["returncode"]==0,j["id"]
    rows={};arrays={};identities={}
    if a.scope=="components":
        base=read(b/"code/experiments/shortscreen2_20261029/results/stage2-results.json")
        rows=copy.deepcopy(base["complete_numeric_endpoints"])
        for byseed in rows.values():
            for row in byseed.values():
                row["planning"]={**row.pop("primary"),**row.pop("fresh")}
        primary=[42,43]
    else:
        primary=[42,43,44];sp=b/"runs/stability_20261029";bp=read(sp/"plan.json")
        rows["original"]={}
        for j in bp["jobs"]:
            if j["training_seed"] not in [1,2] or j["phase"] not in ["planning-primary","planning-confirmation"]:continue
            assert read(sp/"exits"/(j["id"]+".json"))["returncode"]==0
            folder=Path(j["output"]);v,e=read(folder/"metrics.json"),read(folder/"execution.json")
            rows["original"].setdefault(str(j["training_seed"]),{"planning":{}})["planning"][str(j["evaluation_seed"])]=dict(metrics=v,execution=e)
    for j in jobs:
        m,s,ph=j["method"],str(j["training_seed"]),j["phase"];folder=Path(j["output"]);row=rows.setdefault(m,{}).setdefault(s,{"planning":{}})
        if ph==phase:
            status,stage,size=read(folder/"status.json"),read(folder/"stage-complete.json"),read(folder/"model_size.json")
            assert stage["stop_step"]==budget and stage["scheduler_total_steps"]==124910 and size["total"]==51097134 and status["step"]==budget
            row["training"]=dict(status=status,stage=stage,size=size,config=read(folder/"config.json"),trace=[json.loads(x) for x in (folder/"metrics.jsonl").read_text().splitlines()],weights_sha256=sha(folder/f"model-step{budget:07d}/weights.pt"),exit=read(r/"exits"/(j["id"]+".json")))
        elif ph=="planning":
            v,e=read(folder/"metrics.json"),read(folder/"execution.json");assert e["returncode"]==0 and e["episodes"]==50 and not e["bf16"]
            assert v["split"]=="heldout" and v["float32_matmul_precision"]=="highest" and len(v["metrics"]["episode_successes"])==50
            row["planning"][str(j["evaluation_seed"])]=dict(metrics=v,execution=e)
        else:
            if ph=="counterfactual":v=read(folder/"counterfactual.json");cases=v["details"];filename=None;assert v["cases"]==16
            else:
                v=read(folder/"metrics.json");cases=read(folder/"cases.json");filename="per-case.npz" if ph=="health" else "per-case-metrics.npz"
                if ph=="quality":assert v["cases"]==64 and v["sampler_steps"]==20
            signature=[(x["episode"],x["start"]) for x in cases]
            if ph not in identities:identities[ph]=signature
            assert identities[ph]==signature
            row[ph]=dict(metrics=v,cases=cases,exit=read(r/"exits"/(j["id"]+".json")))
            if filename:
                with np.load(folder/filename,allow_pickle=False) as z:
                    for k in z.files:
                        assert np.isfinite(z[k]).all();arrays[f"{ph}/{m}/seed{s}/{k}"]=z[k].copy()
    fresh=[1042,1043,1044];summary={};effects={}
    for m,byseed in rows.items():
        summary[m]={}
        for seed,row in byseed.items():
            assert all(str(es) in row["planning"] for es in primary+fresh)
            for es,value in row["planning"].items():
                v=value["metrics"];signature=(v["episodes"],v["start_steps"])
                if es not in identities:identities[es]=signature
                assert identities[es]==signature
                arrays[f"planning/{m}/seed{seed}/eval{es}"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
            summary[m][seed]=dict(primary=statistics.mean(row["planning"][str(es)]["metrics"]["metrics"]["success_rate"] for es in primary),fresh=statistics.mean(row["planning"][str(es)]["metrics"]["metrics"]["success_rate"] for es in fresh))
            if "quality" in row:
                q=row["quality"]["metrics"];summary[m][seed].update(h7_agent_copy=q["h7_agent_ratio"],h7_moving_block_copy=q["h7_moving_block_ratio"])
            if "counterfactual" in row:summary[m][seed]["action_cf"]=100*row["counterfactual"]["metrics"]["action_identification_accuracy"]
            if "health" in row:summary[m][seed]["collapsed"]=row["health"]["metrics"]["collapsed"]
            if "training" in row:
                t=row["training"]
                summary[m][seed]["last_seconds_per_step"]=t["status"]["seconds_per_step"] if "status" in t else t["status.json"]["seconds_per_step"]
    comparisons=[(m,"original") for m in rows if m!="original"]
    if "temporal-forecast" in rows and "lower-beta" in rows:comparisons.append(("temporal-forecast","lower-beta"))
    if a.scope=="components":comparisons +=[(m,"temporal-forecast") for m in ["temporal-noise","forecast-render"]]
    for left,right in comparisons:
        key=left+"-vs-"+right;effects[key]={}
        for seed in ["1","2"]:
            effects[key][seed]={}
            for name,seeds in [("primary",primary),("fresh",fresh)]:
                delta=[];groups=[]
                for es in seeds:
                    lv=rows[left][seed]["planning"][str(es)]["metrics"];rv=rows[right][seed]["planning"][str(es)]["metrics"]
                    delta.extend(100*(np.asarray(lv["metrics"]["episode_successes"],float)-np.asarray(rv["metrics"]["episode_successes"],float)));groups.extend(lv["episodes"])
                effects[key][seed][name]=cluster_ci(delta,groups)
    result=dict(scope=a.scope,training_updates=budget,model_parameters=51097134,all40selectedjobs_exit0=True,summary=summary,paired_effects=effects,all_numeric_endpoints=rows,primary_evaluation_seeds=primary,fresh_evaluation_seeds=fresh,reporter_sha256=sha(__file__),
        limitations=["Onlytwoinitialtrainingseeds;moreseedsandfrom-scratchconfirmationpending.","Allpairedintervalsconditiononfixedweights/episodeclusters,nottraining-seeduncertainty.","Selectedcandidatesfromexploratoryscreen;noveltynotestablished.","Equalmodelparameters/updatesarenotequalcompute;forecastcallsaddtrainingwork.","Generation/healthreferenceprotocolmaydifferacrosscampaigns;donotcompareunequalmetricbanks.","Mono-componentsat2epochsversusselected3epochresultsbelongtoseparatereports."])
    assert not out.exists();out.mkdir(parents=True);(out/"followup-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"followup-per-case.npz",**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=list(rows);x=np.arange(len(methods));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,key,title in zip(axes,["primary","fresh"],["Matched primary planning cases","Fresh evaluation cases"]):
        for i,s in enumerate(["1","2"]):
            bars=ax.bar(x+(i-.5)*.35,[summary[m][s][key] for m in methods],.32,label="Training seed "+s);ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
        ax.set_xticks(x,methods,rotation=25,ha="right");ax.set_ylim(0,106);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("P0 success (%)");axes[0].legend(loc="upper left",fontsize=8)
    fig.text(.5,.01,f"Matched{budget:,}updates;51.1Mparameters;two training seeds. Academic novelty and broad stability not established.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png","svg"]:fig.savefig(out/("followup-planning."+ext),dpi=180,bbox_inches="tight")
    for f in out.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
    lines=[f"# Budget followup: {a.scope}","","| Method | Training seed | PrimaryP0% | FreshP0% |","| --- | ---: | ---: | ---: |"]
    for m in methods:
        for s in ["1","2"]:lines.append(f"| {m} | {s} | {summary[m][s]['primary']:.2f} | {summary[m][s]['fresh']:.2f} |")
    lines+=["","![Planning](followup-planning.png)","","Completeper-case/endpoints/costs/pairedintervals:followup-results.json andfollowup-per-case.npz. Sameparameter/updatebudget;twoinitialseeds,notbroadstabilityornoveltyproof.",""]
    (out/"FOLLOWUP.md").write_text("\n".join(lines));print(json.dumps({"scope":a.scope,"summary":summary}))
if __name__=="__main__":main()
