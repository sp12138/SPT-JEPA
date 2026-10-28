"""Finish both experiment families with explicit data, pinned weights and disjoint outputs."""
import argparse,json,subprocess,os,hashlib,time
from pathlib import Path
def atomic(path,value):
 tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--parent",required=True);p.add_argument("--output",required=True);p.add_argument("--data",required=True);p.add_argument("--pose",required=True);p.add_argument("--i3d",required=True);p.add_argument("--upstream",required=True);p.add_argument("--planning-data",required=True);p.add_argument("--mode",required=True);p.add_argument("--no-quality",action="store_true");a=p.parse_args()
 parent=Path(a.parent);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 exports=sorted(p for p in parent.glob("model-step*") if (p/"weights.pt").exists())
 if not exports:raise RuntimeError("No portable checkpoint; export the completed NativeDCP first")
 source=exports[-1];pin=out/"checkpoint";pin.mkdir(exist_ok=True)
 for name in ["weights.pt","config.json"]:
  if not (pin/name).exists():os.link(source/name,pin/name)
 cfg=json.loads((pin/"config.json").read_text());step=int(source.name.split("step")[1])
 manifest=dict(parent=str(parent),checkpoint=str(source),step=step,weights_sha256=hashlib.sha256((pin/"weights.pt").read_bytes()).hexdigest(),model_config=cfg,data=a.data,pose=a.pose)
 atomic(out/"checkpoint_manifest.json",manifest)
 py=os.sys.executable;jobs=[];probes=[]
 for variant in (["mean","noisy","mean-fp32","noisy-fp32"] if a.mode in ["L2","S2"] else ["mean"]):
  probes.append(dict(model=parent.name,checkpoint=str(pin),data=a.data,output=str(out/"probes"/variant),noisy=variant.startswith("noisy"),fp32=variant.endswith("fp32"),train=50000,test=5000))
 (out/"probe_jobs.json").write_text(json.dumps(probes,indent=2))
 jobs.append(dict(id="probes",command=[py,"-B","-u","-m","as_jepa_tools.probe_batch","--jobs",str(out/"probe_jobs.json"),"--output",str(out/"probes")]))
 if a.mode!="L0" and not a.no_quality:
  jobs.append(dict(id="quality",command=[py,"-B","-u","-m","as_jepa_tools.quality_complete","--checkpoint",str(pin),"--data",a.data,"--pose",a.pose,"--output",str(out/"quality"),"--i3d",a.i3d,"--cases","256","--media"]))
  jobs.append(dict(id="counterfactual",command=[py,"-B","-u","-m","as_jepa_tools.counterfactual","--checkpoint",str(pin),"--data",a.data,"--pose",a.pose,"--output",str(out/"counterfactual"),"--cases","64","--steps","50"]))
 if a.mode.startswith("L") and cfg.get("domain","standard")=="standard":
  for seed in [42,43,44]:
   jobs.append(dict(id="planning-"+str(seed),command=[py,"-B","-u","-m","as_jepa_tools.planning","--checkpoint",str(pin),"--output",str(out/"planning"/("seed"+str(seed))),"--upstream",a.upstream,"--data-root",a.planning_data,"--test-episodes",str(Path(a.data)/"test_episodes.npy"),"--seed",str(seed),"--episodes","50"]))
 atomic(out/"jobs.json",jobs);results=[]
 for job in jobs:
  atomic(out/"status.json",dict(state="running",stage=job["id"],completed=len(results),total=len(jobs),updated_at=time.time()))
  started=time.time()
  with (out/(job["id"]+".log")).open("w") as log:rc=subprocess.run(job["command"],stdout=log,stderr=subprocess.STDOUT).returncode
  results.append(dict(id=job["id"],returncode=rc,seconds=time.time()-started))
  if rc:
   atomic(out/"status.json",dict(state="failed",completed=len(results),total=len(jobs),results=results,updated_at=time.time()));raise SystemExit(rc)
 atomic(out/"status.json",dict(state="finished",completed=len(results),total=len(jobs),results=results,updated_at=time.time()))
 print(json.dumps(dict(model=parent.name,step=step,stages=len(jobs))))
if __name__=="__main__":main()
