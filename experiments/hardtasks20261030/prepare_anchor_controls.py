"""Same-state, parameter-preserving visual-input/noise factorial; 600 updates only."""
import hashlib,io,json,shutil,subprocess,tarfile,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(8*1024*1024),b""):h.update(x)
 return h.hexdigest()
def main():
 b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";parent=b/"runs/hardtasks_learn2000_20261030";r=b/"runs/hardtasks_anchor_20261030";decision=b/"reports/hardtasks-gate600-review-20261030/decision-anchor2600.json"
 assert json.loads(decision.read_text())["approved_stop_step"]==2600;prior=json.loads((parent/"plan.json").read_text())
 assert all(json.loads((parent/"exits"/(j["id"]+".json")).read_text())["returncode"]==0 for j in prior["jobs"])
 assert not r.exists() and not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip();r.mkdir();source=r/"source-v1";source.mkdir();commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=c,text=True).strip()
 paths=["as_jepa_lewm","as_jepa_tools","runtime","experiments/hardtasks20261030","tests/as_jepa"]
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit,*paths],cwd=c))) as tar:tar.extractall(source,filter="data")
 manifest=dict(source=str(source),source_commit=commit,selected_paths=paths,files={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()});(r/"source-v1.json").write_text(json.dumps(manifest,indent=2))
 src=dict(source=str(source),source_commit=commit,manifest=str(r/"source-v1.json"),pythonpath=":".join(map(str,[b/"environments/completion-overlay",b/"environments/overlay",source,source/"runtime",b/"sources/stable-worldmodel",b/"sources/clear-lewm-20261030"])))
 slots={"tworoom":iter([("dev2",g) for g in range(2,8)]+[("dev13",g) for g in [0,1]]),"cube":iter([("dev13",g) for g in range(2,8)]+[("dev15",g) for g in [0,1]])};arms={"observed-base":("observed",0.),"observed-highnoise":("observed",.8),"constant-base":("zero",0.),"constant-highnoise":("zero",.8)};jobs=[];copies=[];py="/efs/prescottsun/project_jepa/envs/lewm/bin/python";exp=c/"experiments/hardtasks20261030/anchor2600";exp.mkdir(exist_ok=True);(exp/"configs").mkdir(exist_ok=True)
 for task in ["tworoom","cube"]:
  for arm,(mode,shift) in arms.items():
   for seed in [1,2]:
    old=parent/f"{task}-temporal-forecast-seed{seed}-learn2000";stage=json.loads((old/"stage-complete.json").read_text());assert stage["stop_step"]==2000
    name=f"{task}-{arm}-seed{seed}-gate2600";out=r/name;out.mkdir();cfg=json.loads((old/"config.json").read_text());cfg.update(output=str(out),checkpoint_every=600,retain_steps=[2600]);cfg["model"].update(renderer_anchor_mode=mode,flow_logit_shift=shift)
    config=out/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");shutil.copy2(config,exp/"configs"/(name+".json"));shutil.copy2(old/"last.ckpt",out/"last.ckpt");digest=sha(old/"last.ckpt");assert sha(out/"last.ckpt")==digest;copies.append(dict(id=name,parent=str(old),checkpoint_sha256=digest))
    host,gpu=next(slots[task]);base=dict(host=host,gpu=gpu,source=src,task=task,method=arm,training_seed=seed)
    jobs.append(dict(**base,id=name,output=str(out),config=str(config),config_sha256=sha(config),phase="adapt600",command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","2600","--resume"]))
    health=r/("health-"+name);jobs.append(dict(**base,id="health-"+name,output=str(health),depends_on=[name],phase="health",command=[py,"-B","-u","-m","as_jepa_tools.benchmark_health","--run",str(out),"--output",str(health)]))
    ev=r/("planning-"+name);data=Path(cfg["data"]);env={}
    if task=="cube":env=dict(LD_LIBRARY_PATH=str(b/"environments/egl-lib/usr/lib/x86_64-linux-gnu")+":/usr/lib64",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID=str(gpu),__EGL_VENDOR_LIBRARY_FILENAMES=str(b/"environments/egl-lib/10_nvidia.json"))
    jobs.append(dict(**base,id="planning-"+name,output=str(ev),depends_on=["health-"+name],phase="planning",environment=env,command=[py,"-B","-u","-m","as_jepa_tools.benchmark_evaluate","--manifest",str(data/"manifests/validation-strict.json"),"--checkpoint",str(out/"model-step0002600"),"--data",str(data),"--output",str(ev)]))
 for task in ["tworoom","cube"]:
  for seed in [1,2]:assert len(set(x["checkpoint_sha256"] for x in copies if x["id"].startswith(task+"-") and f"-seed{seed}-" in x["id"]))==1
 plan=dict(root=str(r),deadline=None,environment=prior["environment"],jobs=jobs);protocol=dict(root=str(r),source_commit=commit,starting_step=2000,stop_step=2600,additional_updates=600,training_runs=16,arms=arms,all_seeds_preserved=True,same_state_per_task_and_seed=True,unchanged_parameter_count=True,unchanged_inference_planner=True,scope="A causal adaptation test of clean visual access versus corruption, not a fresh-training or novelty claim. All four branches start at the same ordinary-noise temporal-forecast checkpoint.",no_automatic_long_training=True)
 for name,value in [("plan.json",plan),("protocol.json",protocol),("resume-audit.json",copies)]:
  (r/name).write_text(json.dumps(value,indent=2)+"\n");shutil.copy2(r/name,exp/name)
 shutil.copy2(decision,r/"decision.json");shutil.copy2(decision,exp/"decision.json");print(json.dumps(protocol),flush=True)
if __name__=="__main__":main()
