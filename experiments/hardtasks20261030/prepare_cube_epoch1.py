"""Continue only the established Cube method and its two direct baselines."""
from pathlib import Path
import hashlib,io,json,shutil,subprocess,tarfile

B=Path("/efs/prescottsun/SPT-JEPA")
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for x in iter(lambda:f.read(8*1024*1024),b""):h.update(x)
    return h.hexdigest()
def main():
    code=B/"code";root=B/"runs/hardtasks_cube_epoch1_20261030"
    decision=code/"experiments/hardtasks20261030/decision-cube-epoch1.json"
    d=json.loads(decision.read_text())
    assert d["approved_stop_step"]==11671 and d["training_runs"]==6
    review=json.loads((B/"reports/hardtasks-sharedprior2600-review-20261030/results.json").read_text())
    assert review["all12_exit0"]
    assert not root.exists() and not subprocess.check_output(["git","status","--porcelain"],cwd=code,text=True).strip()
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    ds=BenchmarkWindows(B/"data/hardtasks-v1/cube-metadata",window=8)
    assert len(ds)==1494000 and len(ds)//128==11671
    root.mkdir();source=root/"source-v1";source.mkdir()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=code,text=True).strip()
    paths=["as_jepa_lewm","as_jepa_tools","runtime","experiments/hardtasks20261030","tests/as_jepa"]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit,*paths],cwd=code))) as tar:tar.extractall(source,filter="data")
    manifest=dict(source=str(source),source_commit=commit,selected_paths=paths,
                  files={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()})
    (root/"source-v1.json").write_text(json.dumps(manifest,indent=2))
    src=dict(source=str(source),source_commit=commit,manifest=str(root/"source-v1.json"),
             pythonpath=":".join(map(str,[B/"environments/completion-overlay",B/"environments/overlay",source,source/"runtime",B/"sources/stable-worldmodel",B/"sources/clear-lewm-20261030"])))
    learning=B/"runs/hardtasks_learn2000_20261030"
    prior=json.loads((learning/"plan.json").read_text())
    jobs=[];copies=[]
    py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    exp=code/"experiments/hardtasks20261030/cube-epoch1";exp.mkdir();(exp/"configs").mkdir()
    for method,host in [("original","dev2"),("lower-beta","dev13"),("temporal-forecast","dev15")]:
        for seed in [1,2]:
            parent=learning/f"cube-{method}-seed{seed}-learn2000"
            if method=="temporal-forecast":parent=B/"runs/hardtasks_anchor_20261030"/f"cube-observed-base-seed{seed}-gate2600"
            initial=2600 if method=="temporal-forecast" else 2000
            assert json.loads((parent/"stage-complete.json").read_text())["stop_step"]==initial
            for ident in [parent.name,"health-"+parent.name,"planning-"+parent.name]:
                assert json.loads((parent.parent/"exits"/(ident+".json")).read_text())["returncode"]==0
            h=json.loads((parent.parent/("health-"+parent.name)/"health.json").read_text())
            assert all(v for k,v in h["audit"].items() if k not in ["step","gradient_max"])
            assert all(v>0 for v in h["audit"]["gradient_max"].values())
            cfg=json.loads((parent/"config.json").read_text())
            assert cfg["model"].get("flow_logit_shift",0)==0 and cfg["model"].get("rate_prior","one-step")=="one-step"
            assert cfg["model"].get("renderer_anchor_mode","observed")=="observed"
            name=f"cube-{method}-seed{seed}-epoch1"
            out=root/name;out.mkdir()
            cfg.update(output=str(out),checkpoint_every=2000,retain_steps=[4000,8000,11671])
            config=out/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n")
            shutil.copy2(config,exp/"configs"/(name+".json"))
            shutil.copy2(parent/"last.ckpt",out/"last.ckpt")
            digest=sha(parent/"last.ckpt");assert sha(out/"last.ckpt")==digest
            copies.append(dict(id=name,parent=str(parent),checkpoint_sha256=digest,initial_step=initial,
                               model_objective_unchanged=True,optimizer_scheduler_rng_cursor_preserved=True))
            gpu=seed-1
            base=dict(host=host,gpu=gpu,source=src,task="cube",method=method,training_seed=seed)
            jobs.append(dict(**base,id=name,output=str(out),config=str(config),config_sha256=sha(config),phase="epoch1",
                             command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","11671","--resume"]))
            health=root/("health-"+name)
            jobs.append(dict(**base,id="health-"+name,output=str(health),depends_on=[name],phase="health",
                             command=[py,"-B","-u","-m","as_jepa_tools.benchmark_health","--run",str(out),"--output",str(health)]))
            ev=root/("planning-"+name);data=Path(cfg["data"])
            env=dict(LD_LIBRARY_PATH=str(B/"environments/egl-lib/usr/lib/x86_64-linux-gnu")+":/usr/lib64",
                     PYOPENGL_PLATFORM="egl",MUJOCO_GL="egl",MUJOCO_EGL_DEVICE_ID=str(gpu),
                     __EGL_VENDOR_LIBRARY_FILENAMES=str(B/"environments/egl-lib/10_nvidia.json"))
            jobs.append(dict(**base,id="planning-"+name,output=str(ev),depends_on=["health-"+name],phase="planning",environment=env,
                             command=[py,"-B","-u","-m","as_jepa_tools.benchmark_evaluate","--manifest",str(data/"manifests/validation-strict.json"),
                                      "--checkpoint",str(out/"model-step0011671"),"--data",str(data),"--output",str(ev)]))
    jobs.append(dict(id="aggregate-cube-epoch1",host="dev15",gpu=0,source=src,phase="report",output=str(root/"aggregate"),depends_on=[j["id"] for j in jobs],environment=dict(MPLCONFIGDIR=str(B/"cache/matplotlib")),command=[py,"-B","-u",str(source/"experiments/hardtasks20261030/review_cube_epoch1.py")]))
    plan=dict(root=str(root),deadline=None,environment=prior["environment"],jobs=jobs)
    protocol=dict(root=str(root),source_commit=commit,stop_step=11671,training_runs=6,scientific_endpoints=18,aggregation_jobs=1,
                  natural_training_windows=1494000,batch_size=128,window_epoch_definition="floor(1494000/128); 112 trailing windows remain",
                  scheduler_horizon=124910,methods=["original","lower-beta","temporal-forecast"],training_seeds=[1,2],
                  validation_cases=50,test_untouched=True,model_objectives_unchanged=True,
                  scope="A bounded learning comparison, not promotion of a new candidate. Resume exact full states; no noise, anchor or shared-prior long grids. Stop after first window epoch and evaluate.",
                  no_automatic_second_epoch=True)
    for name,value in [("plan.json",plan),("protocol.json",protocol),("resume-audit.json",copies),("decision.json",d)]:
        (root/name).write_text(json.dumps(value,indent=2)+"\n");shutil.copy2(root/name,exp/name)
    print(json.dumps(protocol),flush=True)
if __name__=="__main__":main()
