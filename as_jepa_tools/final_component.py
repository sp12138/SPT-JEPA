"""Run one independent final-evaluation component after a shared checkpoint is pinned."""
import argparse,json,fcntl,os,subprocess,time
from pathlib import Path


def atomic(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--parent-final",required=True);p.add_argument("--campaign",required=True)
    p.add_argument("--component",choices=["actions","samples"],required=True);p.add_argument("--output",required=True)
    p.add_argument("--sample-cases",type=int,default=64);a=p.parse_args()
    final=Path(a.parent_final);campaign=Path(a.campaign);base=campaign.parent.parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/"component.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=json.loads((campaign/"plan.json").read_text())["deadline"]
    while True:
        if (out/"STOP").exists() or time.time()>deadline-90:
            atomic(out/"status.json",dict(state="stopped",reason="deadline or STOP before checkpoint",updated_at=time.time()));return
        manifest=final/"checkpoint_manifest.json"
        if manifest.exists():
            meta=json.loads(manifest.read_text());parent=Path(meta["parent"]);status=json.loads((parent/"status.json").read_text())
            if status.get("state")=="finished" and (final/"checkpoint/weights.pt").exists():break
        atomic(out/"status.json",dict(state="waiting_parent",parent_final=str(final),component=a.component,updated_at=time.time()))
        time.sleep(30)
    ck=final/"checkpoint";meta=json.loads((final/"checkpoint_manifest.json").read_text())
    atomic(out/"checkpoint_manifest.json",meta);python=os.sys.executable;data=str(base/"data/pusht");pose=str(campaign/"pose-v2/weights.pt")
    fp32=meta["forward_precision"]=="32-true";jobs=[]
    if a.component=="actions":
        for seed in [42,43,44]:
            dest=final/"planning"/f"seed{seed}"
            command=[python,"-u","-m","as_jepa_tools.planning","--checkpoint",str(ck),"--output",str(dest),
                     "--upstream",str(base/"sources/stable-worldmodel"),"--data-root",str(base/"data/planning"),
                     "--test-episodes",str(base/"data/pusht/test_episodes.npy"),"--seed",str(seed),"--episodes","50"]
            jobs.append(dict(id=f"planning-{seed}",output=str(dest),result="metrics.json",command=command))
        dest=final/"counterfactual"
        command=[python,"-u","-m","as_jepa_tools.counterfactual","--checkpoint",str(ck),"--data",data,"--pose",pose,
                 "--output",str(dest),"--cases","64","--steps","50"]
        if fp32:command.append("--fp32")
        jobs.append(dict(id="counterfactual",output=str(dest),result="counterfactual.json",command=command))
    else:
        dest=final/"generation/samples"
        command=[python,"-u","-m","as_jepa_tools.sample_rollouts","--checkpoint",str(ck),"--data",data,"--pose",pose,
                 "--output",str(dest),"--cases",str(a.sample_cases),"--steps","50","--samples","4"]
        if fp32:command.append("--fp32")
        jobs.append(dict(id="generation-samples",output=str(dest),result="metrics.json",command=command))
    atomic(out/"jobs.json",jobs);results=[];stopped=False
    for job in jobs:
        if (out/"STOP").exists() or time.time()>deadline-90:stopped=True;break
        if (Path(job["output"])/job["result"]).exists():raise RuntimeError("Refuse to overwrite an existing measurement: "+job["output"])
        started=time.time()
        with (out/(job["id"]+".log")).open("w") as log:code=subprocess.run(job["command"],stdout=log,stderr=subprocess.STDOUT).returncode
        results.append(dict(**job,returncode=code,seconds=time.time()-started,finished_at=time.time()))
        atomic(out/"status.json",dict(state="running",component=a.component,completed=len(results),total=len(jobs),results=results,updated_at=time.time()))
    failed=any(j["returncode"] for j in results)
    atomic(out/"status.json",dict(state="stopped" if stopped else ("failed" if failed else "finished"),component=a.component,completed=len(results),total=len(jobs),results=results,updated_at=time.time()))
    if failed:raise SystemExit(1)


if __name__=="__main__":main()
