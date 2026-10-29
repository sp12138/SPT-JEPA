"""Audit fresh cases, strict numerical repeats, and frozen-weight probes."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--parent",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    r,old,out=Path(a.campaign),Path(a.parent),Path(a.output);plan=read(r/"plan.json");protocol=read(r/"protocol.json")
    assert all(read(r/"exits"/(j["id"]+".json"))["returncode"]==0 for j in plan["jobs"]),"All confirmation jobs must finish successfully"
    expected=[j for j in plan["jobs"] if j["phase"] in {"fresh-planning","probe","planning"}];rows={m:{str(s):{"fresh":{},"repeat":{}} for s in protocol["training_seeds"]} for m in protocol["methods"]}
    identities={};arrays={};probes={}
    for j in expected:
        m,s,phase=j["method"],str(j["training_seed"]),j["phase"];folder=Path(j["output"])
        if phase=="probe":
            v=read(folder/"probes.json");assert v["train_frames"]==10000 and v["test_frames"]==2500 and v["feature_precision"]=="FP32"
            assert v["checkpoint"]==str(old/f"{m}-seed{s}-adapt600/model-step0025582")
            probes.setdefault(m,{})[s]=v
            with np.load(folder/"probe_predictions.npz",allow_pickle=False) as z:
                for k in z.files:
                    assert np.isfinite(z[k]).all();arrays[f"probe/{m}/seed{s}/{k}"]=z[k].copy()
            continue
        v,execution=read(folder/"metrics.json"),read(folder/"execution.json");es=str(j["evaluation_seed"]);kind="fresh" if phase=="fresh-planning" else "repeat"
        assert execution["returncode"]==0 and execution["episodes"]==50 and not execution["bf16"]
        assert v["split"]=="heldout" and v["float32_matmul_precision"]=="highest" and len(v["metrics"]["episode_successes"])==50
        identity=(v["episodes"],v["start_steps"]);key=(kind,es)
        if key not in identities:identities[key]=identity
        assert identities[key]==identity
        expected_checkpoint=(old if kind=="fresh" else r)/f"{m}-seed{s}-adapt600/model-step0025582"
        assert execution["checkpoint"]==str(expected_checkpoint)
        rows[m][s][kind][es]=dict(metrics=v,execution=execution)
        arrays[f"{kind}/{m}/seed{s}/eval{es}/successes"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
    ancillary={}
    for j in plan["jobs"]:
        m,seed,phase=j["method"],str(j["training_seed"]),j["phase"]
        if phase in {"planning","fresh-planning","probe"}:continue
        folder=Path(j["output"]);record={"exit":read(r/"exits"/(j["id"]+".json"))}
        if phase=="adaptation600":
            record.update(status=read(folder/"status.json"),stage=read(folder/"stage-complete.json"),model_size=read(folder/"model_size.json"),config=read(folder/"config.json"),weights_sha256=sha(folder/"model-step0025582/weights.pt"))
            assert record["stage"]["initial_step"]==24982 and record["stage"]["stop_step"]==25582
            assert record["model_size"]["total"]==51097134
            record["training_trace"]=[json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
        elif phase=="health":
            record.update(metrics=read(folder/"metrics.json"),cases=read(folder/"cases.json"))
            with np.load(folder/"per-case.npz",allow_pickle=False) as z:
                for k in z.files:
                    assert np.isfinite(z[k]).all();arrays[f"repeat-health/{m}/seed{seed}/{k}"]=z[k].copy()
        elif phase=="quality":
            record.update(metrics=read(folder/"metrics.json"),cases=read(folder/"cases.json"))
            assert record["metrics"]["cases"]==64 and record["metrics"]["sampler_steps"]==20
            with np.load(folder/"per-case-metrics.npz",allow_pickle=False) as z:
                for k in z.files:
                    assert np.isfinite(z[k]).all();arrays[f"repeat-quality/{m}/seed{seed}/{k}"]=z[k].copy()
        elif phase=="counterfactual":
            record["metrics"]=read(folder/"counterfactual.json");assert record["metrics"]["cases"]==16
        else:raise AssertionError("Unknown phase:"+phase)
        ancillary.setdefault(m,{}).setdefault(seed,{})[phase]=record
    summaries={};effects={}
    for m,values in rows.items():
        summaries[m]={};effects[m]={}
        for s,row in values.items():
            assert len(row["fresh"])==3 and len(row["repeat"])==2
            first=[];primary_cases=set()
            for es in [42,43]:
                first_result=read(old/f"p0-{m}-seed{s}-adapt600-eval{es}/metrics.json")
                first.append(first_result["metrics"]["success_rate"])
                primary_cases.update(zip(first_result["episodes"],first_result["start_steps"]))
            fresh=[v["metrics"]["metrics"]["success_rate"] for v in row["fresh"].values()]
            repeat=[v["metrics"]["metrics"]["success_rate"] for v in row["repeat"].values()]
            origin=read(r/f"{m}-seed{s}-adapt600/resume-origin.json")
            assert origin["parent"]==str(Path("/efs/prescottsun/SPT-JEPA")/f"runs/stability_20261029/snapshots/trainseed{s}/step0024982") and origin["step"]==24982
            unseen=[];overlap=0
            for value in row["fresh"].values():
                mv=value["metrics"]
                for ep,start,success in zip(mv["episodes"],mv["start_steps"],mv["metrics"]["episode_successes"]):
                    if (ep,start) in primary_cases:overlap+=1
                    else:unseen.append(success)
            summaries[m][s]=dict(fresh_primary_exact_case_overlap=overlap,fresh_without_primary_cases=100*float(np.mean(unseen)),fresh_without_primary_count=len(unseen),first_primary=statistics.mean(first),fresh=statistics.mean(fresh),fresh_values=fresh,
                repeat_primary=statistics.mean(repeat),repeat_values=repeat,numerical_repeat_shift=statistics.mean(repeat)-statistics.mean(first),
                probe_agent_r2=probes[m][s]["agent_r2"],probe_block_r2=probes[m][s]["block_position_r2"],probe_block_angle_r2=probes[m][s]["block_angle_r2"])
            if m=="original":continue
            for kind in ["fresh","repeat","fresh-excluding-primary"]:
                differences=[];episodes=[]
                source_kind="fresh" if kind=="fresh-excluding-primary" else kind
                for es,left in row[source_kind].items():
                    lv=left["metrics"];rv=rows["original"][s][source_kind][es]["metrics"]
                    keep=np.asarray([(ep,start) not in primary_cases if kind=="fresh-excluding-primary" else True for ep,start in zip(lv["episodes"],lv["start_steps"])])
                    differences.extend(((np.asarray(lv["metrics"]["episode_successes"],float)-np.asarray(rv["metrics"]["episode_successes"],float))*100)[keep])
                    episodes.extend(np.asarray(lv["episodes"])[keep])
                effects[m].setdefault(s,{})[kind]=cluster_ci(differences,episodes)
    result=dict(all_confirmation_endpoints_complete=True,source_scope=protocol,summary=summaries,paired_effects=effects,complete_planning_endpoints=rows,complete_probes=probes,complete_training_health_quality_counterfactual=ancillary,
        reporter_sha256=sha(__file__),limitations=["Twelve600update numerical repeats reuse existing training seeds and identical initial full states/RNG.","Fresh planning evaluates the original selected short-screenweights; repeats use primary42/43cases.","Seeds1-4 are existing trained models, not new from-scratch samples.","Drop25/context-mask were selected from a larger exploratory screen; no novelty or stable89%claim.","Bootstrap conditions on fixed trained weights/episode clusters and is not training-seed uncertainty. Exact primary/fresh case overlaps are counted and also excluded in a separate paired effect."])
    assert not out.exists();out.mkdir(parents=True)
    (out/"anchor-confirmation-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"anchor-confirmation-per-case.npz",**arrays)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods=list(rows);x=np.arange(len(methods));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,kind,title in zip(axes,["fresh","repeat_primary"],["Fresh cases: eval1042/1043/1044","Same-state training repeat: eval42/43"]):
        for i,s in enumerate(["1","2","3","4"]):
            bars=ax.bar(x+(i-1.5)*.18,[summaries[m][s][kind] for m in methods],.16,label="Training seed "+s)
            ax.bar_label(bars,fmt="%.0f",padding=3,fontsize=8)
        ax.set_xticks(x,methods,rotation=20,ha="right");ax.set_title(title);ax.set_ylim(0,107);ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("P0 success (%)");axes[0].legend(fontsize=8,loc="upper left")
    fig.text(.5,.01,"Same51.1Mparameters;600updateadaptations of existing models; no from-scratch stability claim.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ["png","svg"]:fig.savefig(out/("anchor-confirmation."+ext),dpi=180,bbox_inches="tight")
    for f in out.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
    lines=["# Selected anchor leads: independent cases and numerical repeat","","All120confirmation endpoints exited0. Fresh evaluation uses the first short-screenweights; strict600update repeats restart the same2epochfullstate/RNG. These are existing training seeds, not from-scratch repetitions.","","| Method | Seed | First primary % | Fresh % | Repeat primary % | Fresh delta pp | Repeat delta pp | Agent R2 | Block position R2 |","| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in methods:
        for s in ["1","2","3","4"]:
            v=summaries[m][s];base=summaries["original"][s]
            lines.append(f"| {m} | {s} | {v['first_primary']:.1f} | {v['fresh']:.2f} | {v['repeat_primary']:.1f} | {v['fresh']-base['fresh']:+.2f} | {v['repeat_primary']-base['repeat_primary']:+.1f} | {v['probe_agent_r2']:.3f} | {v['probe_block_r2']:.3f} |")
    lines+=["","![Independent checks](anchor-confirmation.png)","","Full numerical/per-case evidence and conditional paired intervals: anchor-confirmation-results.json and anchor-confirmation-per-case.npz.","","Adaptation gains alone do not establish stable89%or a causal explanation for the anchor effect. No automatic longer training is encoded in this confirmation plan.",""]
    (out/"ANCHOR-CONFIRMATION.md").write_text("\n".join(lines));print(json.dumps({"complete":True,"summary":summaries}))
if __name__=="__main__":main()
