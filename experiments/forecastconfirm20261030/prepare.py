"""Confirm simpler forecast rendering and audit the selected result on new cases."""
from pathlib import Path
import json,copy,os,datetime,subprocess,hashlib
import torch

def main():
 torch.set_num_threads(2);b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";old=b/"runs/promising_20261029";r=b/"runs/forecast_confirm_20261030";assert not r.exists()
 assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
 parent=json.loads((old/"plan.json").read_text());assert len(parent["jobs"])==112
 assert all(json.loads((old/"exits"/(j["id"]+".json")).read_text())["returncode"]==0 for j in parent["jobs"])
 source=parent["jobs"][0]["source"];templates=json.loads((b/"runs/anchor_20261029/plan.json").read_text());base=json.loads((b/"runs/stability_20261029/full-L2-beta0.25-stage3-seed1/config.json").read_text())
 slots=[("dev2",i) for i in range(4)]+[("dev13",i) for i in range(4)]
 r.mkdir();(r/"workers").mkdir();(r/"reports").mkdir();(b/"tmp/forecast_confirm_20261030").mkdir(exist_ok=True)
 exp=c/"experiments/forecastconfirm20261030";(exp/"configs").mkdir(exist_ok=True);jobs=[];after=[];fresh=[];records=[];py="/efs/prescottsun/project_jepa/envs/lewm/bin/python";sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
 for seed,(host,gpu) in zip(range(1,9),slots):
  if seed in [1,2]:
   origin=old/f"forecast-render-seed{seed}-component-epoch2";phase="forecast-epoch3";cap=37473
  elif seed in [3,4]:
   origin=b/f"runs/robustness_20261029/forecast-render-seed{seed}";assert(origin/"STOP").exists()
   phase="existing-forecast-gate600";state=torch.load(origin/"last.ckpt",map_location="cpu",weights_only=False);cap=int(state["global_step"])+600;del state
  else:origin=None;phase="new-forecast-gate600";cap=600
  ident=f"forecast-render-seed{seed}-{phase}";folder=r/ident;folder.mkdir()
  cfg=json.loads((origin/"config.json").read_text()) if origin else copy.deepcopy(base)
  cfg.update(output=str(folder),seed=seed,max_steps=124910,stop_after=cap,retain_steps=[cap],record_initial_weights=origin is None,checkpoint_every=2500)
  if origin is None:cfg["robust_variant"]=True;cfg["model"].update(rho=0.,forecast_fraction=.5,forecast_seed=20261029+seed*100003,beta=.25)
  initial=0
  if origin:
   state=torch.load(origin/"last.ckpt",map_location="cpu",weights_only=False);initial=int(state["global_step"]);assert state["campaign_samples"]==initial*128 and state["optimizer_states"] and state["campaign_rng"] and state["lr_schedulers"];del state
   os.link(origin/"last.ckpt",folder/"last.ckpt");(folder/"resume-origin.json").write_text(json.dumps({"parent":str(origin),"initial_step":initial,"cap":cap,"preserve_parent_STOP":True},indent=2)+"\n")
  config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_text(config.read_text())
  j=dict(id=ident,host=host,gpu=gpu,output=str(folder),source=source,config=str(config),config_sha256=sha(config),command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step",str(cap)]+(["--resume"] if origin else []),phase=phase,method="forecast-render",training_seed=seed,initial_step=initial,stop_step=cap)
  jobs.append(j);records.append({k:j[k] for k in ["id","host","gpu","phase","method","training_seed","initial_step","stop_step","config","config_sha256"]})
  for ep in (["planning","health","quality","counterfactual"] if phase=="forecast-epoch3" else ["health"]):
   for es in ([42,43,44,1042,1043,1044,2042,2043,2044] if ep=="planning" else [None]):
    t=next(x for x in templates["jobs"] if x["phase"]==ep and x["method"]=="original" and x["training_seed"]==1 and (ep!="planning" or x["evaluation_seed"]==42))
    child=copy.deepcopy(t);cid=f"{ep}-{ident}"+(f"-eval{es}" if es is not None else "");out=r/cid
    child.update(id=cid,host=host,gpu=gpu,output=str(out),source=source,method="forecast-render",training_seed=seed,depends_on=[ident],endpoint_phase=phase)
    child["command"][child["command"].index("--checkpoint")+1]=str(folder/f"model-step{cap:07d}");child["command"][child["command"].index("--output")+1]=str(out)
    if es is not None:child["evaluation_seed"]=es;child["command"][child["command"].index("--seed")+1]=str(es)
    after.append(child)
 index=0
 for method in ["original","lower-beta","temporal-forecast"]:
  for seed in [1,2]:
   checkpoint=b/f"runs/stability_20261029/snapshots/trainseed{seed}/step0037473" if method=="original" else old/f"{method}-seed{seed}-selected-epoch3/model-step0037473"
   assert(checkpoint/"weights.pt").exists()
   for es in [2042,2043,2044]:
    host,gpu=slots[index%8];index+=1;ident=f"newcases-{method}-seed{seed}-eval{es}";out=r/ident
    t=next(x for x in templates["jobs"] if x["phase"]=="planning" and x["method"]=="original" and x["training_seed"]==1 and x["evaluation_seed"]==42);command=copy.deepcopy(t["command"])
    command[command.index("--checkpoint")+1]=str(checkpoint);command[command.index("--output")+1]=str(out);command[command.index("--seed")+1]=str(es)
    fresh.append(dict(id=ident,host=host,gpu=gpu,output=str(out),source=source,command=command,phase="new-case-planning",method=method,training_seed=seed,evaluation_seed=es))
 protocol=dict(root=str(r),created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source["source_commit"],runs=records,
  basis="Combined3epochP094/94additional93.33/91.33;forecastonly2epoch90/90matchescombinedwithinuncertainty. Testasimplerpathandnewcases.",
  model_parameters=51097134,state_dimension=64,batch_size=128,scheduler_total_steps=124910,
  scope="18fixedweightplanningchecksonnewseeds2042/2043/2044;2forecastonlythird-epochcontinuations;6forecastonly600updategates(existingseeds3/4,newseeds5-8).Allothersremainintheirexistingcappedcampaigns.",
  controls=["FixedfirstresultandretunedAS-JEPAplanningweights;sameP0budget,caseidentities,precision.","OldSTOPmarkerspreserved,fulloptimizer/scheduler/RNG/cursorresumes.","Newseedinitialtensorsmustmatchoriginal/lower/combinedfrompromisinggate600."],
  limitations=["2initialtrainingseedsnotbroadstability.","Newcasequeryoverlapmustbecountedandexcludedinasecondaryanalysis.","Sameparameters/updatesnotFLOPs;forecastingaddscompute.","Noacademicnoveltyclaim."],automatic_long_continuation=False)
 (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");env=copy.deepcopy(parent["environment"]);env["TMPDIR"]=str(b/"tmp/forecast_confirm_20261030")
 plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=env,scope=protocol["scope"],jobs=fresh+jobs+after)
 (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(source,indent=2)+"\n")
 for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
 print(json.dumps({"root":str(r),"training":8,"newfixedweightplanning":18,"jobs":len(plan["jobs"])}))
if __name__=="__main__":main()
