"""Report all paired two-epoch replication endpoints in a completed campaign."""
import argparse,json,hashlib,statistics
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args();r,out=Path(a.campaign),Path(a.output);plan=read(r/"plan.json");rows={};arrays={};identities={}
 for j in plan["jobs"]:
  exit_record=read(r/"exits"/(j["id"]+".json"));assert exit_record["returncode"]==0
  m,s,phase=j["method"],str(j["training_seed"]),j["phase"];folder=Path(j["output"]);row=rows.setdefault(m,{}).setdefault(s,{"planning":{}})
  if phase=="second-epoch":
   stage,status,size=read(folder/"stage-complete.json"),read(folder/"status.json"),read(folder/"model_size.json")
   assert stage["initial_step"]==12491 and stage["stop_step"]==24982 and stage["scheduler_total_steps"]==124910 and size["total"]==51097134 and status["step"]==24982
   row["training"]=dict(stage=stage,status=status,size=size,config=read(folder/"config.json"),trace=[json.loads(x) for x in(folder/"metrics.jsonl").read_text().splitlines()],exit=exit_record,weights_sha256=hashlib.sha256((folder/"model-step0024982/weights.pt").read_bytes()).hexdigest())
  elif phase=="planning":
   v,e=read(folder/"metrics.json"),read(folder/"execution.json");es=str(j["evaluation_seed"])
   assert e["returncode"]==0 and e["episodes"]==50 and not e["bf16"] and v["split"]=="heldout" and v["float32_matmul_precision"]=="highest" and len(v["metrics"]["episode_successes"])==50
   signature=(v["episodes"],v["start_steps"])
   if es not in identities:identities[es]=signature
   assert identities[es]==signature
   row["planning"][es]=dict(metrics=v,execution=e);arrays[f"planning/{m}/seed{s}/eval{es}"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
  else:
   if phase=="counterfactual":v=read(folder/"counterfactual.json");cases=v["details"];filename=None;assert v["cases"]==16
   else:
    v=read(folder/"metrics.json");cases=read(folder/"cases.json");filename="per-case.npz" if phase=="health" else "per-case-metrics.npz"
   signature=[(x["episode"],x["start"]) for x in cases]
   if phase not in identities:identities[phase]=signature
   assert identities[phase]==signature
   row[phase]=dict(metrics=v,cases=cases,exit=exit_record)
   if filename:
    with np.load(folder/filename,allow_pickle=False) as z:
     for k in z.files:
      assert np.isfinite(z[k]).all();arrays[f"{phase}/{m}/seed{s}/{k}"]=z[k].copy()
 methods=["original","lower-beta","temporal-forecast"];seeds=sorted(rows["original"],key=int);summary={};effects={}
 for m in methods:
  summary[m]={};assert set(rows[m])==set(seeds)
  for s,row in rows[m].items():
   assert len(row["planning"])==6
   q,h,cf=[row[x]["metrics"] for x in ["quality","health","counterfactual"]]
   summary[m][s]=dict(primary=statistics.mean(row["planning"][str(es)]["metrics"]["metrics"]["success_rate"] for es in[42,43,44]),additional=statistics.mean(row["planning"][str(es)]["metrics"]["metrics"]["success_rate"] for es in[1042,1043,1044]),
       h7_agent_copy=q["h7_agent_ratio"],h7_moving_block_copy=q["h7_moving_block_ratio"],action_cf=100*cf["action_identification_accuracy"],block_cf=100*cf["block_action_identification_accuracy"],collapsed=h["collapsed"],posterior_over_anchor=h["posterior_over_anchor"])
 for left,right in[("lower-beta","original"),("temporal-forecast","original"),("temporal-forecast","lower-beta")]:
  key=left+"-vs-"+right;effects[key]={}
  for s in seeds:
   effects[key][s]={}
   for label,evalseeds in[("primary",[42,43,44]),("additional",[1042,1043,1044])]:
    delta=[];groups=[]
    for es in evalseeds:
     l,rv=rows[left][s]["planning"][str(es)]["metrics"],rows[right][s]["planning"][str(es)]["metrics"]
     delta.extend(100*(np.asarray(l["metrics"]["episode_successes"],float)-np.asarray(rv["metrics"]["episode_successes"],float)));groups.extend(l["episodes"])
    effects[key][s][label]=cluster_ci(delta,groups)
 across={m:{k:dict(mean=statistics.mean(summary[m][s][k] for s in seeds),std=statistics.stdev(summary[m][s][k] for s in seeds) if len(seeds)>1 else None,worst=min(summary[m][s][k] for s in seeds))for k in["primary","additional"]}for m in methods}
 result=dict(all_expected_jobs_exit0=True,endpoints=len(plan["jobs"]),parameters=51097134,total_updates=24982,training_seeds=seeds,summary=summary,across_seeds=across,paired_effects=effects,complete_numeric_endpoints=rows,reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  limits=["Twoepochcheckpoint;third-epochstabilityrequiresaseparatecomparison.","Alltrainingseedsandnegativeoutcomesretained;nooutlierremoval.","Bootstrapconditionsonweightsandepisodeclusters,nottraining-seeduncertainty.","Sameparameters/updatecountnotcompute;forecastingaddswork.","Academicnoveltynotestablished;knownideasadaptedtosample-levelloss."])
 assert not out.exists();out.mkdir(parents=True);(out/"replication-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"replication-per-case.npz",**arrays)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 x=np.arange(len(seeds));fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,key,title in zip(axes,["primary","additional"],["Primary planning cases","Additional planning cases"]):
  for i,m in enumerate(methods):
   bars=ax.bar(x+(i-1)*.25,[summary[m][s][key] for s in seeds],.23,label=m);ax.bar_label(bars,fmt="%.0f",padding=3,fontsize=8)
  ax.set_xticks(x,["Training seed "+s for s in seeds]);ax.set_ylim(0,107);ax.set_title(title);ax.spines[["top","right"]].set_visible(False)
 axes[0].set_ylabel("P0 success (%)");axes[0].legend(loc="upper left",fontsize=8)
 fig.text(.5,.01,"Matched 24,982 updates; 51.1M parameters; all training seeds retained.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
 for ext in["png","svg"]:fig.savefig(out/("replication-planning."+ext),dpi=180,bbox_inches="tight")
 for f in out.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
 lines=["# Two-epoch replication","","| Method | Training seed | PrimaryP0% | AdditionalP0% | h7agent/copy | h7movingblock/copy |","| --- | ---: | ---: | ---: | ---: | ---: |"]
 for m in methods:
  for s in seeds:
   v=summary[m][s];lines.append(f"| {m} | {s} | {v['primary']:.2f} | {v['additional']:.2f} | {v['h7_agent_copy']:.3f} | {v['h7_moving_block_copy']:.3f} |")
 lines+=["","![Replication](replication-planning.png)","","Allnumericalendpointsandconditionalpairedintervals:replication-results.jsonandreplication-per-case.npz. Nooutlierremoval;notafinalthird-epochclaim.",""]
 (out/"REPLICATION.md").write_text("\n".join(lines));print(json.dumps({"summary":summary,"across_seeds":across}))
if __name__=="__main__":main()
