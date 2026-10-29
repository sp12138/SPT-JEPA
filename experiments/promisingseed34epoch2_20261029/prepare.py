"""Bounded second-epoch confirmation of the reviewed existing seeds3/4."""
from pathlib import Path
import json,copy,os,datetime,subprocess
def main():
 b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";old=b/"runs/promising_epoch1_20261029";r=b/"runs/promising_seed34_epoch2_20261029"
 assert not r.exists() and not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
 parent=json.loads((old/"plan.json").read_text());templates=json.loads((b/"runs/anchor_20261029/plan.json").read_text());sha=__import__("hashlib").sha256;source=parent["jobs"][0]["source"]
 selected=[j for j in parent["jobs"] if j["phase"]=="first-epoch" and j["training_seed"] in [3,4]];assert len(selected)==4
 for j in selected:
  assert json.loads((old/"exits"/(j["id"]+".json")).read_text())["returncode"]==0
  h=json.loads((old/("health-"+j["id"])/"metrics.json").read_text());assert not h["collapsed"]
 r.mkdir();(r/"workers").mkdir();(r/"reports").mkdir();(b/"tmp/promising_seed34_epoch2_20261029").mkdir(exist_ok=True)
 jobs=[];after=[];refs=[];records=[];exp=c/"experiments/promisingseed34epoch2_20261029";(exp/"configs").mkdir(exist_ok=True);py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
 def evaluate(method,seed,checkpoint,host,gpu,dependency=None,reference=False):
  results=[]
  for phase in ["planning","health","quality","counterfactual"]:
   for es in ([42,43,44,1042,1043,1044] if phase=="planning" else [None]):
    template=next(x for x in templates["jobs"] if x["phase"]==phase and x["method"]=="original" and x["training_seed"]==1 and (phase!="planning" or x["evaluation_seed"]==42))
    child=copy.deepcopy(template);ident=f"{phase}-{method}-seed{seed}"+(f"-eval{es}" if es is not None else "");out=r/ident
    child.update(id=ident,host=host,gpu=gpu,output=str(out),source=source,method=method,training_seed=seed)
    child.pop("depends_on",None)
    if dependency:child["depends_on"]=[dependency]
    child["command"][child["command"].index("--checkpoint")+1]=str(checkpoint);child["command"][child["command"].index("--output")+1]=str(out)
    if es is not None:child["evaluation_seed"]=es;child["command"][child["command"].index("--seed")+1]=str(es)
    results.append(child)
  return results
 for j in selected:
  method,seed=j["method"],j["training_seed"];ident=f"{method}-seed{seed}-epoch2";folder=r/ident;folder.mkdir();cfg=json.loads(Path(j["config"]).read_text());cfg.update(output=str(folder),stop_after=24982,retain_steps=[24982],record_initial_weights=False)
  config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_text(config.read_text());os.link(Path(j["output"])/"last.ckpt",folder/"last.ckpt")
  (folder/"resume-origin.json").write_text(json.dumps({"parent":j["output"],"initial_step":12491,"cap":24982},indent=2)+"\n")
  job=dict(id=ident,host=j["host"],gpu=j["gpu"],output=str(folder),source=source,config=str(config),config_sha256=sha(config.read_bytes()).hexdigest(),command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","24982","--resume"],phase="second-epoch",method=method,training_seed=seed,initial_step=12491,stop_step=24982)
  jobs.append(job);records.append({k:job[k] for k in ["id","host","gpu","method","training_seed","config","config_sha256"]})
  after+=evaluate(method,seed,folder/"model-step0024982",j["host"],j["gpu"],ident)
 # Run saved original reference weights while candidates resume;reference comes first on each slot.
 for seed in [3,4]:
  host="dev2";gpu=4 if seed==3 else 5;cp=b/f"runs/stability_20261029/snapshots/trainseed{seed}/step0024982"
  assert (cp/"weights.pt").exists();refs+=evaluate("original",seed,cp,host,gpu)
 protocol=dict(root=str(r),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source["source_commit"],runs=records,model_parameters=51097134,batch_size=128,scheduler_total_steps=124910,
   scope="Onlyseeds3/4original+lower/combined24982updatecomparison;candidatefirstepochhealth/stateusepassedandcombinedP015/18vsretuned7/5. Existing2seedstrong2epochtaskleadmotivatesconfirmation. Noautomatic3epochcontinuation.",automatic_long_continuation=False)
 (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");env=copy.deepcopy(parent["environment"]);env["TMPDIR"]=str(b/"tmp/promising_seed34_epoch2_20261029")
 plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=env,scope=protocol["scope"],jobs=refs+jobs+after)
 (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(source,indent=2)+"\n")
 for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
 print(json.dumps({"root":str(r),"training_runs":4,"jobs":len(plan["jobs"])}))
if __name__=="__main__":main()
