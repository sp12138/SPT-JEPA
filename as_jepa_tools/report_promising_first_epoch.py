"""Publish all first-epoch replication outcomes without selecting seeds."""
import argparse,json,hashlib,statistics
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--output",required=True);a=p.parse_args();r,out=Path(a.campaign),Path(a.output);plan=read(r/"plan.json");rows={};arrays={};ids={}
 assert len(plan["jobs"])==96
 for j in plan["jobs"]:
  exit_record=read(r/"exits"/(j["id"]+".json"));assert exit_record["returncode"]==0
  m,s,phase=j["method"],str(j["training_seed"]),j["phase"];folder=Path(j["output"]);row=rows.setdefault(m,{}).setdefault(s,{"planning":{}})
  if phase=="first-epoch":
   stage,status,size=read(folder/"stage-complete.json"),read(folder/"status.json"),read(folder/"model_size.json")
   assert stage["stop_step"]==12491 and stage["scheduler_total_steps"]==124910 and status["step"]==12491 and size["total"]==51097134
   row["training"]=dict(stage=stage,status=status,size=size,config=read(folder/"config.json"),trace=[json.loads(x) for x in (folder/"metrics.jsonl").read_text().splitlines()],exit=exit_record)
  elif phase=="planning":
   v,e=read(folder/"metrics.json"),read(folder/"execution.json");assert e["returncode"]==0 and e["episodes"]==50 and not e["bf16"]
   assert v["split"]=="heldout" and v["float32_matmul_precision"]=="highest" and len(v["metrics"]["episode_successes"])==50
   es=str(j["evaluation_seed"]);signature=(v["episodes"],v["start_steps"])
   if es not in ids:ids[es]=signature
   assert ids[es]==signature
   row["planning"][es]=dict(metrics=v,execution=e);arrays[f"planning/{m}/seed{s}/eval{es}"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
  else:
   if phase=="counterfactual":v=read(folder/"counterfactual.json");cases=v["details"];filename=None;assert v["cases"]==16
   else:
    v=read(folder/"metrics.json");cases=read(folder/"cases.json");filename="per-case.npz" if phase=="health" else "per-case-metrics.npz"
   signature=[(x["episode"],x["start"]) for x in cases]
   if phase not in ids:ids[phase]=signature
   assert ids[phase]==signature
   row[phase]=dict(metrics=v,cases=cases,exit=exit_record)
   if filename:
    with np.load(folder/filename,allow_pickle=False) as z:
     for k in z.files:
      assert np.isfinite(z[k]).all();arrays[f"{phase}/{m}/seed{s}/{k}"]=z[k].copy()
 summary={}
 for m,byseed in rows.items():
  summary[m]={}
  for seed,row in byseed.items():
   assert len(row["planning"])==2
   q,h,cf=[row[x]["metrics"] for x in ["quality","health","counterfactual"]]
   summary[m][seed]=dict(p0=statistics.mean(v["metrics"]["metrics"]["success_rate"] for v in row["planning"].values()),h7_agent_copy=q["h7_agent_ratio"],h7_moving_block_copy=q["h7_moving_block_ratio"],action_cf=100*cf["action_identification_accuracy"],collapsed=h["collapsed"],posterior_over_anchor=h["posterior_over_anchor"])
 result=dict(all96exits0=True,total_updates=12491,model_parameters=51097134,summary=summary,complete_numeric_endpoints=rows,reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits=["Firstepochimmaturebehaviorisnotafinalstabilityresult.","Sealednewtrainingseeds5-8retainedintact;existingseeds3/4alsoincludedbutoriginalcounterpartsnotinthisfirstepochplan.","Noautomaticfurtherbudget;readhealth/taskevidenceandpreserveallweakoutcomes."])
 assert not out.exists();out.mkdir(parents=True);(out/"first-epoch-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"first-epoch-per-case.npz",**arrays)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 methods=["original","lower-beta","temporal-forecast"];x=np.arange(4);fig,ax=plt.subplots(figsize=(9,5))
 for i,m in enumerate(methods):
  bars=ax.bar(x+(i-1)*.25,[summary[m][str(s)]["p0"] for s in [5,6,7,8]],.23,label=m);ax.bar_label(bars,fmt="%.0f",padding=3)
 ax.set_xticks(x,["Training seed "+str(s) for s in [5,6,7,8]]);ax.set_ylim(0,105);ax.set_ylabel("P0 success (%)");ax.set_title("First-epoch planning on four new training seeds");ax.legend();ax.spines[["top","right"]].set_visible(False)
 fig.text(.5,.01,"Matched 12,491 updates; 51.1M parameters. Early learning stage; mature-budget confirmation remains pending.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
 for ext in ["png","svg"]:fig.savefig(out/("first-epoch-planning."+ext),dpi=180,bbox_inches="tight")
 for f in out.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
 lines=["# Four new training seeds: first-epoch stage","","| Method | Seed | P0% | h7agent/copy | h7movingblock/copy | ActionCF% |","| --- | ---: | ---: | ---: | ---: | ---: |"]
 for m in methods:
  for seed in ["5","6","7","8"]:
   v=summary[m][seed];lines.append(f"| {m} | {seed} | {v['p0']:.1f} | {v['h7_agent_copy']:.3f} | {v['h7_moving_block_copy']:.3f} | {v['action_cf']:.2f} |")
 lines+=["","![Firstepoch](first-epoch-planning.png)","","All96endpointsrecorded;first-epochimmaturitydoesnotestablishmaturestabilityorsuperiority.Fullper-case/outcomeevidenceisretained.",""]
 (out/"FIRST-EPOCH.md").write_text("\n".join(lines));print(json.dumps({"summary":summary}))
if __name__=="__main__":main()
