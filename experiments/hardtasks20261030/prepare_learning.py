"""Resume all matched seeds to the explicitly reviewed learning-stage cap."""
import argparse,hashlib,io,json,shutil,subprocess,tarfile,time
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--decision",required=True);a=p.parse_args();decision=json.loads(Path(a.decision).read_text());assert decision["approved_cap"]==2000
 b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";gate=b/"runs/hardtasks_gate600_20261030";r=b/"runs/hardtasks_learn2000_20261030";assert not r.exists();assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
 for task in ["tworoom","cube"]:
  review=json.loads((Path(decision["review_root"])/task/"gate-review.json").read_text());assert review["complete"] and len(review["records"])==10
 r.mkdir();source=r/"source-v1";source.mkdir();commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=c,text=True).strip();paths=["as_jepa_lewm","as_jepa_tools","runtime","experiments/hardtasks20261030"]
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit,*paths],cwd=c))) as tar:tar.extractall(source,filter="data")
 manifest=dict(source=str(source),source_commit=commit,selected_paths=paths,files={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()});(r/"source-v1.json").write_text(json.dumps(manifest,indent=2))
 src=dict(source=str(source),source_commit=commit,manifest=str(r/"source-v1.json"),pythonpath=":".join(map(str,[b/"environments/completion-overlay",b/"environments/overlay",source,source/"runtime",b/"sources/stable-worldmodel",b/"sources/clear-lewm-20261030"])))
 prior=json.loads((gate/"plan.json").read_text());jobs=[];copies=[];py="/efs/prescottsun/project_jepa/envs/lewm/bin/python";closure=b/"reports/harder-task-redirect-20261030/PUSHT-CLOSURE-REVIEWED.json";cube_slots=iter([("dev2",i) for i in [0,1]]+[("dev13",i) for i in range(4,8)]+[("dev15",i) for i in range(4)])
 for old in [x for x in prior["jobs"] if x.get("phase")=="gate600"]:
  parent=Path(old["output"]);stage=json.loads((parent/"stage-complete.json").read_text());assert stage["stop_step"]==600
  name=old["id"].replace("gate600","learn2000");out=r/name;out.mkdir();cfg=json.loads((parent/"config.json").read_text());cfg.update(output=str(out),checkpoint_every=700,retain_steps=[2000]);config=out/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n")
  shutil.copy2(parent/"last.ckpt",out/"last.ckpt");digest=sha(parent/"last.ckpt");assert sha(out/"last.ckpt")==digest;copies.append(dict(id=name,parent=str(parent),checkpoint_sha256=digest,initial_step=600))
  host,gpu=next(cube_slots) if old["task"]=="cube" else (old["host"],old["gpu"]);base=dict(host=host,gpu=gpu,source=src,task=old["task"],method=old["method"],training_seed=old["training_seed"])
  jobs.append(dict(**base,id=name,output=str(out),config=str(config),config_sha256=sha(config),command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","2000","--resume"],phase="learn2000",requires=[str(closure)] if old["task"]=="cube" else []))
  health=r/("health-"+name);jobs.append(dict(**base,id="health-"+name,output=str(health),depends_on=[name],phase="health",command=[py,"-B","-u","-m","as_jepa_tools.benchmark_health","--run",str(out),"--output",str(health)]))
  ev=r/("planning-"+name);data=Path(cfg["data"]);environment={}
  if old["task"]=="cube":environment=dict(LD_LIBRARY_PATH=str(b/"environments/egl-lib/usr/lib/x86_64-linux-gnu")+":/usr/lib64",PYOPENGL_PLATFORM="egl",MUJOCO_EGL_DEVICE_ID=str(gpu),__EGL_VENDOR_LIBRARY_FILENAMES=str(b/"environments/egl-lib/10_nvidia.json"))
  jobs.append(dict(**base,id="planning-"+name,output=str(ev),depends_on=["health-"+name],phase="planning",environment=environment,command=[py,"-B","-u","-m","as_jepa_tools.benchmark_evaluate","--manifest",str(data/"manifests/validation-strict.json"),"--checkpoint",str(out/"model-step0002000"),"--data",str(data),"--output",str(ev)]))
 plan=dict(root=str(r),deadline=None,environment=prior["environment"],jobs=jobs);(r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"resume-audit.json").write_text(json.dumps(copies,indent=2));(r/"decision.json").write_bytes(Path(a.decision).read_bytes())
 protocol=dict(source_commit=commit,root=str(r),initial_step=600,stop_step=2000,scheduler_horizon=124910,all_seeds_preserved=True,method_unchanged=True,reason=decision["reason"],validation_cases=50,no_automatic_further_training=True,jobs=60)
 (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n");exp=c/"experiments/hardtasks20261030/learning2000";exp.mkdir(exist_ok=True)
 for name in ["protocol.json","plan.json","decision.json","resume-audit.json"]:shutil.copy2(r/name,exp/name)
 print(json.dumps(protocol),flush=True)
if __name__=="__main__":main()
