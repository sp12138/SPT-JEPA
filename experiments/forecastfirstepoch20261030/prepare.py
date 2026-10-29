"""Advance reviewed600update trials only to their first epoch."""
from pathlib import Path
import json,copy,os,hashlib,datetime,subprocess
import torch
def main():
 b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";old=b/"runs/forecast_confirm_20261030";r=b/"runs/forecast_firstepoch_20261030"
 assert not r.exists() and not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
 audit=json.loads((old/"reports/GATE600-AUDIT.json").read_text());assert audit["all6pass"]
 parent=json.loads((old/"plan.json").read_text());templates=json.loads((b/"runs/anchor_20261029/plan.json").read_text());selected=[j for j in parent["jobs"] if j["phase"] in ["existing-forecast-gate600","new-forecast-gate600"]];assert len(selected)==6
 r.mkdir();(r/"workers").mkdir();(r/"reports").mkdir();(b/"tmp/forecast_firstepoch_20261030").mkdir(exist_ok=True);jobs=[];after=[];records=[];exp=c/"experiments/forecastfirstepoch20261030";(exp/"configs").mkdir(exist_ok=True);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
 for j in selected:
  method,seed=j["method"],j["training_seed"];ident=f"{method}-seed{seed}-epoch1";folder=r/ident;folder.mkdir();parentfolder=Path(j["output"])
  cfg=json.loads((parentfolder/"config.json").read_text());cfg.update(output=str(folder),stop_after=12491,retain_steps=[12491],record_initial_weights=False)
  config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_text(config.read_text());os.link(parentfolder/"last.ckpt",folder/"last.ckpt")
  (folder/"resume-origin.json").write_text(json.dumps({"parent":str(parentfolder),"initial_step":j["stop_step"],"cap":12491,"scope":"Reviewed600updategatepassed;newboundedoutput,preserveallparents"},indent=2)+"\n")
  new=dict(id=ident,host=j["host"],gpu=j["gpu"],output=str(folder),source=j["source"],config=str(config),config_sha256=sha(config),command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","12491","--resume"],phase="first-epoch",method=method,training_seed=seed,initial_step=j["stop_step"],stop_step=12491)
  jobs.append(new);records.append({k:new[k] for k in ["id","host","gpu","method","training_seed","initial_step","stop_step","config","config_sha256"]})
  for phase in ["planning","health","quality","counterfactual"]:
   for es in ([42,43] if phase=="planning" else [None]):
    t=next(x for x in templates["jobs"] if x["phase"]==phase and x["method"]=="original" and x["training_seed"]==1 and (phase!="planning" or x["evaluation_seed"]==42))
    child=copy.deepcopy(t);childid=f"{phase}-{ident}"+(f"-eval{es}" if es else "");out=r/childid;child.update(id=childid,host=j["host"],gpu=j["gpu"],output=str(out),source=j["source"],depends_on=[ident],method=method,training_seed=seed)
    child["command"][child["command"].index("--checkpoint")+1]=str(folder/"model-step0012491");child["command"][child["command"].index("--output")+1]=str(out)
    if es:child["evaluation_seed"]=es;child["command"][child["command"].index("--seed")+1]=str(es)
    after.append(child)
 protocol=dict(root=str(r),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=selected[0]["source"]["source_commit"],runs=records,model_parameters=51097134,batch_size=128,scheduler_total_steps=124910,basis="All6gateauditspassed,initialtensorsmatched,finite/nonzerogradientsandposteriorstateinterventions;previous2seed2epochtaskleadsupportsfirstepochconfirmation.",scope="Onlyto12491updates,thenreview.Forecastonlyexistingseeds3/4andnewseeds5-8.Noautomatic2epochcontinuation.",automatic_long_continuation=False)
 (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");env=copy.deepcopy(parent["environment"]);env["TMPDIR"]=str(b/"tmp/forecast_firstepoch_20261030")
 plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=env,scope=protocol["scope"],jobs=jobs+after)
 (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(selected[0]["source"],indent=2)+"\n")
 for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
 print(json.dumps({"root":str(r),"training_runs":6,"jobs":len(plan["jobs"]),"first_epoch_only":True}))
if __name__=="__main__":main()
