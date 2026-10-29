"""Prepare a fixed five-arm, two-seed comparison on two harder control tasks."""
import hashlib,io,json,subprocess,tarfile,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";r=b/"runs/hardtasks_gate600_20261030";py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    assert not r.exists();assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
    for task in ["tworoom","cube"]:
        pre=b/"runs/hardtasks_preflight_20261030"/task
        assert json.loads((pre/"stage-complete.json").read_text())["stage_complete"]
    r.mkdir();source=r/"source-v1";source.mkdir();commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=c,text=True).strip()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit],cwd=c))) as tar:tar.extractall(source,filter="data")
    deps={name:dict(path=str(b/"sources"/name),commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=b/"sources"/name,text=True).strip()) for name in ["stable-worldmodel","clear-lewm-20261030"]}
    (r/"dependencies.json").write_text(json.dumps(deps,indent=2))
    manifest=dict(source=str(source),source_commit=commit,files={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()})
    (r/"source-v1.json").write_text(json.dumps(manifest,indent=2))
    src=dict(source=str(source),source_commit=commit,manifest=str(r/"source-v1.json"),pythonpath=":".join(str(x) for x in [b/"environments/completion-overlay",b/"environments/overlay",source,source/"runtime",b/"sources/stable-worldmodel",b/"sources/clear-lewm-20261030"]))
    slots=dict(tworoom=[("dev2",i) for i in range(2,8)]+[("dev13",i) for i in range(4)],cube=[("dev2",i) for i in range(2)]+[("dev13",i) for i in range(4,8)]+[("dev15",i) for i in range(4)])
    arms={"original":dict(beta=.25,rho=0.,forecast_fraction=0.,flow_logit_shift=0.),"lower-beta":dict(beta=.0625,rho=0.,forecast_fraction=0.,flow_logit_shift=0.),"temporal-forecast":dict(beta=.25,rho=.8,forecast_fraction=.5,flow_logit_shift=0.),"original-highnoise":dict(beta=.25,rho=0.,forecast_fraction=0.,flow_logit_shift=.8),"temporal-forecast-highnoise":dict(beta=.25,rho=.8,forecast_fraction=.5,flow_logit_shift=.8)}
    jobs=[];records=[];exp=c/"experiments/hardtasks20261030";(exp/"configs").mkdir(exist_ok=True)
    closure=str(b/"reports/harder-task-redirect-20261030/PUSHT-CLOSURE-REVIEWED.json")
    for task in ["tworoom","cube"]:
        data=b/"data/hardtasks-v1"/(task+"-metadata");card=json.loads((data/"dataset_card.json").read_text())
        for (method,seed),(host,gpu) in zip([(m,s) for m in arms for s in [1,2]],slots[task]):
            ident=f"{task}-{method}-seed{seed}-gate600";out=r/ident;out.mkdir()
            cfg=dict(output=str(out),data=str(data),dataset_kind="benchmark_hdf5",seed=seed,batch_size=128,workers=4,cpu_threads=2,max_steps=124910,max_seconds=31536000,checkpoint_every=600,record_initial_weights=True,robust_variant=True,retain_steps=[600],model=dict(mode="L2",sigma0=.1,window=8,renderer_width=384,renderer_depth=12,renderer_heads=6,forecast_seed=20261029+seed*100003,action_dim=card["model_action_dim"],**arms[method]))
            config=out/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_bytes(config.read_bytes())
            job=dict(id=ident,host=host,gpu=gpu,output=str(out),source=src,config=str(config),config_sha256=sha(config),command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","600"],phase="gate600",task=task,method=method,training_seed=seed,requires=[closure] if task=="cube" else [])
            jobs.append(job);records.append({k:job[k] for k in ["id","task","method","training_seed","host","gpu","config_sha256"]})
            health=r/("health-"+ident)
            jobs.append(dict(id="health-"+ident,host=host,gpu=gpu,output=str(health),source=src,depends_on=[ident],command=[py,"-B","-u","-m","as_jepa_tools.benchmark_health","--run",str(out),"--output",str(health)],phase="health",task=task,method=method,training_seed=seed))
    for gpu,(task,policy) in enumerate([(t,p) for t in ["tworoom","cube"] for p in ["random","released-lewm"]],4):
        ident=f"reference-{task}-{policy}-strict-validation";out=r/ident;data=b/"data/hardtasks-v1"/(task+"-metadata")
        checkpoint="random" if policy=="random" else str(b/"weights/hardtasks"/task)
        jobs.append(dict(id=ident,host="dev15",gpu=gpu,output=str(out),source=src,requires=[closure],phase="reference",task=task,method=policy,command=[py,"-B","-u","-m","as_jepa_tools.benchmark_evaluate","--manifest",str(data/"manifests/validation-strict.json"),"--checkpoint",checkpoint,"--data",str(data),"--output",str(out)]))
    protocol=dict(root=str(r),source_commit=commit,created_at=time.time(),training_runs=records,short_cap=600,batch_size=128,lr_horizon=124910,training_seeds=[1,2],global_hyperparameters=True,training_inputs=["pixels","actions"],anchor="first observed image",noise_shift=dict(base_logit_mean=.8,additional_logit_shift=.8,unchanged_std=.8,training_only=True,inference_unchanged=True),next_review="Audit full state, gradients, actual state/action use, initialization matching and baseline evaluation execution. Preserve all seeds. If healthy, one bounded learning stage may be explicitly prepared; no automatic longer training.",comparison="Original/temporal-forecast x original/high-noise factorial plus strong lower-beta control. Same parameters per task; separately report training time. No new academic novelty claim.",scope="Custom held-out episode manifests with CLEAR v0.8 task semantics; released baselines have different model size/budget and may have trained on evaluation episodes.",dependencies=deps)
    (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    tmp=b/"tmp/hardtasks20261030";tmp.mkdir(exist_ok=True)
    env=dict(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MUJOCO_GL="egl",SDL_VIDEODRIVER="dummy",PYGLET_HEADLESS="true",MPLCONFIGDIR=str(b/"cache/matplotlib"),TMPDIR=str(tmp),STABLEWM_HOME=str(b/"data/hardtasks-v1/planning"))
    plan=dict(root=str(r),deadline=None,environment=env,jobs=jobs)
    (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    for file,dest in [("protocol.json","protocol.json"),("plan.json","dispatch-plan.json")]: (exp/dest).write_bytes((r/file).read_bytes())
    print(json.dumps(dict(root=str(r),source_commit=commit,jobs=len(jobs),training=20,all_caps=600,cube_waits_for_old_closure=True)),flush=True)
if __name__=="__main__":main()
