"""Finalize a completed, immutable campaign run with matched held-out evidence."""
import argparse,json,os,shutil,subprocess,time,fcntl
from pathlib import Path


def atomic(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--campaign",required=True)
    p.add_argument("--output",required=True);p.add_argument("--cases",type=int,default=256)
    p.add_argument("--planning-seeds",default="42,43,44");p.add_argument("--sample-cases",type=int);p.add_argument("--skip-samples",action="store_true")
    p.add_argument("--probe-only",action="store_true");a=p.parse_args()
    run=Path(a.run);campaign=Path(a.campaign);base=campaign.parent.parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/"finalizer.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=json.loads((run/"status.json").read_text())
    if status.get("state")!="finished":raise RuntimeError(f"Parent run is not finished: {status.get('state')}")
    cfg=json.loads((run/"config.json").read_text());exports=sorted(p for p in run.glob("model-step*") if (p/"config.json").exists() and (p/"weights.pt").exists())
    if not exports:raise RuntimeError("Completed parent has no portable checkpoint")
    src=exports[-1];ck=out/"checkpoint";ck.mkdir(exist_ok=True)
    if not (ck/"weights.pt").exists():shutil.copy2(src/"config.json",ck/"config.json");os.link(src/"weights.pt",ck/"weights.pt")
    model_cfg=json.loads((ck/"config.json").read_text());mode=model_cfg["mode"];domain=model_cfg.get("domain","standard");fp32=cfg.get("precision")=="32-true"
    step=int(src.name.split("step")[1]);python=os.sys.executable;data=str(base/"data/pusht");pose=str(campaign/"pose-v2/weights.pt")
    atomic(out/"checkpoint_manifest.json",dict(parent=str(run),checkpoint=str(src),checkpoint_step=step,
           parent_status=status,training_max_steps=cfg["max_steps"],reached_max_steps=status["step"]>=cfg["max_steps"],
           model_config=model_cfg,forward_precision=cfg.get("precision","bf16-mixed"),pose_weights=pose))
    jobs=[]
    probe_jobs=[]
    for noisy in ([False,True] if mode=="L2" else [False]):
        dest=out/"probes"/("noisy" if noisy else "mean")
        probe_jobs.append(dict(model=run.name,checkpoint=str(ck),data=data,output=str(dest),noisy=noisy,fp32=fp32,train=20000,test=5000))
    if mode=="L2" and not fp32:
        probe_jobs.append(dict(model=run.name,checkpoint=str(ck),data=data,output=str(out/"probes/mean-fp32"),noisy=False,fp32=True,train=20000,test=5000))
    (out/"probe_jobs.json").write_text(json.dumps(probe_jobs,indent=2))
    jobs.append(dict(id="probes",command=[python,"-u","-m","as_jepa_tools.probe_batch","--jobs",str(out/"probe_jobs.json"),"--output",str(out/"probes")]))
    if not a.probe_only:
        if mode!="L0":
            args=[python,"-u","-m","as_jepa_tools.evaluate","--checkpoint",str(ck),"--data",data,"--pose",pose,"--output",str(out/"generation/mean"),"--cases",str(a.cases),"--steps","50"]
            if fp32:args.append("--fp32")
            jobs.append(dict(id="generation-mean",command=args))
            if mode=="L2" and not a.skip_samples:
                args=[python,"-u","-m","as_jepa_tools.sample_rollouts","--checkpoint",str(ck),"--data",data,"--pose",pose,"--output",str(out/"generation/samples"),"--cases",str(a.sample_cases or a.cases),"--steps","50","--samples","4"]
                if fp32:args.append("--fp32")
                jobs.append(dict(id="generation-samples",command=args))
        if domain=="color":
            jobs.append(dict(id="colour-swaps",command=[python,"-u","-m","as_jepa_tools.color_swaps","--checkpoint",str(ck),"--data",data,"--pose",pose,"--output",str(out/"colour-swaps"),"--cases","128","--steps","50"]))
        if domain=="standard":
            for seed in a.planning_seeds.split(","):
                jobs.append(dict(id="planning-"+seed,command=[python,"-u","-m","as_jepa_tools.planning","--checkpoint",str(ck),"--output",str(out/"planning"/("seed"+seed)),"--upstream",str(base/"sources/stable-worldmodel"),"--data-root",str(base/"data/planning"),"--test-episodes",str(base/"data/pusht/test_episodes.npy"),"--seed",seed,"--episodes","50"]))
            if mode!="L0":
                args=[python,"-u","-m","as_jepa_tools.counterfactual","--checkpoint",str(ck),"--data",data,"--pose",pose,"--output",str(out/"counterfactual"),"--cases","64","--steps","50"]
                if fp32:args.append("--fp32")
                jobs.append(dict(id="counterfactual",command=args))
    def priority(job):
        if job["id"]=="probes":return 0
        if job["id"]=="generation-mean":return 1
        if job["id"]=="colour-swaps":return 2
        if job["id"].startswith("planning-"):return 3
        if job["id"]=="counterfactual":return 4
        return 5
    jobs.sort(key=priority)
    atomic(out/"jobs.json",jobs);results=[];deadline=json.loads((campaign/"plan.json").read_text())["deadline"];stopped=False
    for job in jobs:
        if (out/"STOP").exists() or time.time()>deadline-90:stopped=True;break
        log=out/(job["id"]+".log");started=time.time()
        with log.open("w") as handle:code=subprocess.run(job["command"],stdout=handle,stderr=subprocess.STDOUT).returncode
        results.append(dict(**job,returncode=code,seconds=time.time()-started,finished_at=time.time()))
        atomic(out/"status.json",dict(state="running",parent=run.name,step=step,completed=len(results),total=len(jobs),results=results,updated_at=time.time()))
    errors=any(r["returncode"] for r in results)
    atomic(out/"status.json",dict(state="stopped" if stopped else ("finished_with_errors" if errors else "finished"),parent=run.name,step=step,completed=len(results),total=len(jobs),results=results,updated_at=time.time()))
    if errors:raise SystemExit(1)


if __name__=="__main__":main()
