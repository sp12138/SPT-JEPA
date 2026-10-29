"""Complete matched two-epoch task evidence for evidence-selected methods."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--campaign",required=True);parser.add_argument("--parent",required=True);parser.add_argument("--output",required=True);a=parser.parse_args()
    r,parent,out=Path(a.campaign),Path(a.parent),Path(a.output);plan=read(r/"plan.json");original=read(parent/"plan.json");protocol=read(r/"protocol.json")
    for root,p in [(r,plan),(parent,original)]:
        assert all(read(root/"exits"/(j["id"]+".json"))["returncode"]==0 for j in p["jobs"]),"All selected-stage jobs must finish"
    methods=protocol["methods"];seeds=[str(s) for s in protocol["training_seeds"]]
    rows={m:{s:{"primary":{},"fresh":{}} for s in seeds} for m in methods};arrays={};identities={}
    def planning(j,kind):
        folder=Path(j["output"]);v,e=read(folder/"metrics.json"),read(folder/"execution.json")
        m,s,es=j["method"],str(j["training_seed"]),str(j["evaluation_seed"])
        assert e["returncode"]==0 and e["episodes"]==50 and not e["bf16"]
        assert v["split"]=="heldout" and v["float32_matmul_precision"]=="highest" and len(v["metrics"]["episode_successes"])==50
        signature=(v["episodes"],v["start_steps"]);key=(kind,es)
        if key not in identities:identities[key]=signature
        assert identities[key]==signature
        rows[m][s][kind][es]=dict(metrics=v,execution=e)
        arrays[f"{kind}/{m}/seed{s}/eval{es}/successes"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
    for j in original["jobs"]:
        if j["phase"] in {"reference-planning","screen-planning"}:planning(j,"primary")
        elif j["phase"]=="stage-training":
            m,s=j["method"],str(j["training_seed"]);folder=Path(j["output"]);stage=read(folder/"stage-complete.json");size=read(folder/"model_size.json")
            assert stage["initial_step"]==12491 and stage["stop_step"]==24982 and stage["scheduler_total_steps"]==124910
            assert size["total"]==51097134 and read(folder/"status.json")["step"]==24982
            rows[m][s]["training"]=dict(stage=stage,status=read(folder/"status.json"),config=read(folder/"config.json"),
                training_trace=[json.loads(x) for x in (folder/"metrics.jsonl").read_text().splitlines()],
                weights_sha256=sha(folder/"model-step0024982/weights.pt"),model_size=size)
    for j in plan["jobs"]:
        phase=j["phase"];m,s=j["method"],str(j["training_seed"]);folder=Path(j["output"])
        if phase=="fresh-planning":planning(j,"fresh");continue
        if phase=="health":
            value=read(folder/"metrics.json");cases=read(folder/"cases.json");filename="per-case.npz"
        elif phase=="quality":
            value=read(folder/"metrics.json");cases=read(folder/"cases.json");filename="per-case-metrics.npz"
            assert value["cases"]==64 and value["sampler_steps"]==20 and value["samples"]==1
        elif phase=="counterfactual":
            value=read(folder/"counterfactual.json");assert value["cases"]==16;cases=value["details"];filename=None
        else:raise AssertionError(phase)
        signature=[(x["episode"],x["start"]) for x in cases]
        if phase not in identities:identities[phase]=signature
        assert identities[phase]==signature
        rows[m][s][phase]=dict(metrics=value,cases=cases,exit=read(r/"exits"/(j["id"]+".json")))
        if filename:
            with np.load(folder/filename,allow_pickle=False) as z:
                for k in z.files:
                    assert np.isfinite(z[k]).all();arrays[f"{phase}/{m}/seed{s}/{k}"]=z[k].copy()
    summaries={};effects={}
    for m in methods:
        summaries[m]={}
        for s in seeds:
            row=rows[m][s];assert len(row["primary"])==2 and len(row["fresh"])==3
            primary=statistics.mean(x["metrics"]["metrics"]["success_rate"] for x in row["primary"].values())
            fresh=statistics.mean(x["metrics"]["metrics"]["success_rate"] for x in row["fresh"].values())
            q,h,cf=[row[k]["metrics"] for k in ["quality","health","counterfactual"]]
            summaries[m][s]=dict(primary=primary,fresh=fresh,h7_agent_copy=q["h7_agent_ratio"],h7_moving_block_copy=q["h7_moving_block_ratio"],action_cf=100*cf["action_identification_accuracy"],block_cf=100*cf["block_action_identification_accuracy"],collapsed=h["collapsed"],posterior_over_anchor=h["posterior_over_anchor"])
            if m=="original":continue
            primarycases={(ep,start) for x in row["primary"].values() for ep,start in zip(x["metrics"]["episodes"],x["metrics"]["start_steps"])}
            for kind in ["primary","fresh","fresh-excluding-primary"]:
                selection="fresh" if kind=="fresh-excluding-primary" else kind;delta=[];groups=[]
                for es,left in row[selection].items():
                    lv=left["metrics"];rv=rows["original"][s][selection][es]["metrics"]
                    for ep,start,lval,rval in zip(lv["episodes"],lv["start_steps"],lv["metrics"]["episode_successes"],rv["metrics"]["episode_successes"]):
                        if kind=="fresh-excluding-primary" and (ep,start) in primarycases:continue
                        delta.append(100*(float(lval)-float(rval)));groups.append(ep)
                effects.setdefault(m,{}).setdefault(s,{})[kind]=cluster_ci(delta,groups)
    effects["temporal-forecast-vs-lower-beta"]={}
    for seed in seeds:
        primarycases={(ep,start) for x in rows["original"][seed]["primary"].values() for ep,start in zip(x["metrics"]["episodes"],x["metrics"]["start_steps"])}
        for kind in ["primary","fresh","fresh-excluding-primary"]:
            selection="fresh" if kind=="fresh-excluding-primary" else kind;delta=[];groups=[]
            for es,left in rows["temporal-forecast"][seed][selection].items():
                lv=left["metrics"];rv=rows["lower-beta"][seed][selection][es]["metrics"]
                for ep,start,lval,rval in zip(lv["episodes"],lv["start_steps"],lv["metrics"]["episode_successes"],rv["metrics"]["episode_successes"]):
                    if kind=="fresh-excluding-primary" and (ep,start) in primarycases:continue
                    delta.append(100*(float(lval)-float(rval)));groups.append(ep)
            effects["temporal-forecast-vs-lower-beta"].setdefault(seed,{})[kind]=cluster_ci(delta,groups)
    result=dict(all_selected_stage_jobs_complete=True,total_updates=24982,model_parameters=51097134,summary=summaries,paired_effects=effects,
        complete_numeric_endpoints=rows,protocol=protocol,reporter_sha256=sha(__file__),
        limits=["Two selected diagnostic training seeds, not full final model-stability evidence.","Exact24982update budget and124910scheduler horizon; methodbeta/noise/forecast changes explicit.","Generation64clipsEuler20andCF16resetcases are a screen; not FVD or originalfullbenchmark.","Freshcase overlaps with primary excluded in secondaryeffects; bootstrapconditionalonweights.","No automatic furthertraining; decide using taskoutcomes and retainmixed/negative results."])
    assert not out.exists();out.mkdir(parents=True);(out/"stage2-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"stage2-per-case.npz",**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.arange(len(methods));fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)
    for ax,key,title in zip(axes,["primary","fresh"],["Common screening cases42/43","Fresh evaluation1042/1043/1044"]):
        for i,s in enumerate(seeds):
            bars=ax.bar(x+(i-.5)*.35,[summaries[m][s][key] for m in methods],.32,label="Training seed "+s);ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
        ax.set_xticks(x,methods,rotation=20,ha="right");ax.set_ylim(0,105);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("P0 success (%)");axes[0].legend(loc="upper left",fontsize=8)
    fig.text(.5,.01,"Matched two-epoch checkpoints;51.1Mparameters;two diagnostic training seeds. Not final stability evidence.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png","svg"]:fig.savefig(out/("stage2-planning."+ext),dpi=180,bbox_inches="tight")
    for f in out.glob("*.svg"):f.write_text("\n".join(line.rstrip() for line in f.read_text().splitlines()).rstrip()+"\n")
    lines=["# Selected two-epoch arms: matched task evidence","","Allparent16jobs andsupplement36jobs completed0. Four selected training runs stopped exactly24982updates. Same51.1Mparameters and124910stepLRhorizon; beta/noise/forecast variants remainexplicit.","","| Method | Training seed | PrimaryP0% | FreshP0% | h7agent/copy | h7movingblock/copy | ActionCF% |","| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in methods:
        for s in seeds:
            v=summaries[m][s];lines.append(f"| {m} | {s} | {v['primary']:.1f} | {v['fresh']:.2f} | {v['h7_agent_copy']:.3f} | {v['h7_moving_block_copy']:.3f} | {v['action_cf']:.2f} |")
    lines+=["","![Planning](stage2-planning.png)","","Fullnumeric/per-case/conditionaluncertainty:stage2-results.json andstage2-per-case.npz. Selected diagnostic seeds, not stable89%or from-scratch confirmation. No automatic continuation.",""]
    (out/"STAGE2-RESULTS.md").write_text("\n".join(lines));print(json.dumps({"complete":True,"summary":summaries}))
if __name__=="__main__":main()
