"""New-case check of fixed third-epoch weights, with prior-query overlap removed."""
import argparse,json,hashlib,statistics
from pathlib import Path
import numpy as np
from as_jepa_tools.report_main_conclusions import cluster_ci
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--campaign",required=True);p.add_argument("--previous",required=True);p.add_argument("--output",required=True);a=p.parse_args();r,out=Path(a.campaign),Path(a.output)
 plan=read(r/"plan.json");previous=read(a.previous);jobs=[j for j in plan["jobs"] if j["phase"]=="new-case-planning"];assert len(jobs)==18
 rows={};ids={};arrays={}
 for j in jobs:
  assert read(r/"exits"/(j["id"]+".json"))["returncode"]==0
  f=Path(j["output"]);v,e=read(f/"metrics.json"),read(f/"execution.json");m,s,es=j["method"],str(j["training_seed"]),str(j["evaluation_seed"])
  assert e["returncode"]==0 and e["episodes"]==50 and not e["bf16"] and v["split"]=="heldout" and v["float32_matmul_precision"]=="highest"
  assert len(v["metrics"]["episode_successes"])==50 and int(es) in [2042,2043,2044]
  signature=(v["episodes"],v["start_steps"])
  if es not in ids:ids[es]=signature
  assert ids[es]==signature
  row=rows.setdefault(m,{}).setdefault(s,{})
  row[es]=dict(metrics=v,execution=e,exit=read(r/"exits"/(j["id"]+".json")))
  arrays[f"{m}/seed{s}/eval{es}/successes"]=np.asarray(v["metrics"]["episode_successes"],np.int8)
  arrays[f"{m}/seed{s}/eval{es}/episodes"]=np.asarray(v["episodes"],np.int64)
  arrays[f"{m}/seed{s}/eval{es}/starts"]=np.asarray(v["start_steps"],np.int64)
 summary={};effects={};prior={}
 for m,byseed in rows.items():
  summary[m]={}
  for s,values in byseed.items():
   assert len(values)==3
   prior[s]={(ep,start) for v in previous["all_numeric_endpoints"]["original"][s]["planning"].values() for ep,start in zip(v["metrics"]["episodes"],v["metrics"]["start_steps"])}
   scores=[];unseen=[];overlap=0
   for value in values.values():
    v=value["metrics"];scores.append(v["metrics"]["success_rate"])
    for ep,start,success in zip(v["episodes"],v["start_steps"],v["metrics"]["episode_successes"]):
     if(ep,start) in prior[s]:overlap+=1
     else:unseen.append(success)
   summary[m][s]=dict(mean=statistics.mean(scores),values=scores,exact_query_overlap_with_prior=overlap,excluding_prior=100*float(np.mean(unseen)),excluding_prior_count=len(unseen))
 for left,right in [("temporal-forecast","original"),("temporal-forecast","lower-beta"),("lower-beta","original")]:
  key=left+"-vs-"+right;effects[key]={}
  for s in ["1","2"]:
   effects[key][s]={}
   for exclude in [False,True]:
    delta=[];groups=[]
    for es,lv in rows[left][s].items():
     l,rval=lv["metrics"],rows[right][s][es]["metrics"]
     for ep,start,a,b in zip(l["episodes"],l["start_steps"],l["metrics"]["episode_successes"],rval["metrics"]["episode_successes"]):
      if exclude and(ep,start) in prior[s]:continue
      delta.append(100*(float(a)-float(b)));groups.append(ep)
    effects[key][s]["excluding_prior" if exclude else "all"]=cluster_ci(delta,groups)
 result=dict(all18endpoints_exit0=True,updates=37473,parameters=51097134,summary=summary,paired_effects=effects,complete_endpoints=rows,
  reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope="Fixedthird-epochweights,previouslyunusedplanningseeds2042/2043/2044.SameP0budget,precision,heldoutepisodesandmatchedcaseidentities.",
  limits=["Twoinitialtrainingseeds;broadertrainingstabilitypending.","Newqueryseedsdonotimplynewepisodes;exactpriorqueryoverlapsexcludedinsecondaryanalysis.","Conditionalepisodebootstrapnottraining-seeduncertainty.","Algorithmsselectedfromearlierexploration;noveltynotestablished."])
 assert not out.exists();out.mkdir(parents=True);(out/"new-case-results.json").write_text(json.dumps(result,indent=2)+"\n");np.savez_compressed(out/"new-case-per-case.npz",**arrays)
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 methods=list(rows);x=np.arange(len(methods));fig,ax=plt.subplots(figsize=(9,5))
 for i,s in enumerate(["1","2"]):
  bars=ax.bar(x+(i-.5)*.35,[summary[m][s]["mean"] for m in methods],.32,label="Training seed "+s);ax.bar_label(bars,fmt="%.1f",padding=3)
 ax.set_xticks(x,methods);ax.set_ylim(0,106);ax.set_ylabel("P0 success (%)");ax.set_title("New planning cases: evaluation seeds 2042 / 2043 / 2044");ax.legend(loc="upper left");ax.spines[["top","right"]].set_visible(False)
 fig.text(.5,.01,"Fixed 37,473-update weights; 51.1M parameters; two training seeds. Broader stability remains pending.",ha="center",fontsize=9);fig.tight_layout(rect=(0,.04,1,1))
 for ext in ["png","svg"]:fig.savefig(out/("new-case-planning."+ext),dpi=180,bbox_inches="tight")
 for f in out.glob("*.svg"):f.write_text("\n".join(x.rstrip() for x in f.read_text().splitlines()).rstrip()+"\n")
 lines=["# Third-epoch planning on additional case seeds","","| Method | Training seed | P0% | Excluding exact prior queries% | Prior query overlap |","| --- | ---: | ---: | ---: | ---: |"]
 for m in methods:
  for s in ["1","2"]:
   v=summary[m][s];lines.append(f"| {m} | {s} | {v['mean']:.2f} | {v['excluding_prior']:.2f} | {v['exact_query_overlap_with_prior']} |")
 lines+=["","![Newcases](new-case-planning.png)","","Fixedweightconfirmation,notnewtrainingseedstability.Completenumbers,conditionalpairedintervalsandqueryidentitiesinJSON/NPZ.",""]
 (out/"NEW-CASE-PLANNING.md").write_text("\n".join(lines));print(json.dumps({"summary":summary,"effects":effects}))
if __name__=="__main__":main()
