"""Freeze only evidence-supported continuations, mechanism controls and new-seed short gates."""
from pathlib import Path
import copy,datetime,hashlib,io,json,os,subprocess,tarfile
import torch

def main():
    torch.set_num_threads(2);b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";r=b/"runs/promising_20261029"
    assert not r.exists()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=c,text=True).strip()
    r.mkdir();(r/"workers").mkdir();(r/"reports").mkdir();(b/"tmp/promising_20261029").mkdir(exist_ok=True)
    source=r/"source-v1";source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit],cwd=c)),mode="r:") as archive:archive.extractall(source,filter="data")
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=dict(source=str(source),source_commit=commit,files={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob("*")) if p.is_file()})
    (r/"source-v1.json").write_text(json.dumps(manifest,indent=2)+"\n")
    src=dict(source=str(source),source_commit=commit,manifest=str(r/"source-v1.json"),pythonpath=":".join(map(str,[b/"environments/completion-overlay",b/"environments/overlay",source,source/"runtime",b/"sources/stable-worldmodel"])))
    # Prove framework/model equations unchanged from validated sources.
    old=b/"runs/anchor_20261029/source-v1"
    assert sha(source/"as_jepa_lewm/robust.py")==sha(old/"as_jepa_lewm/robust.py")
    assert sha(source/"as_jepa_lewm/model.py")==sha(old/"as_jepa_lewm/model.py")
    s2=b/"runs/shortscreen2_20261029";screen=b/"runs/shortscreen_20261029";held=b/"runs/robustness_20261029"
    templates=json.loads((b/"runs/anchor_20261029/plan.json").read_text())
    template=json.loads((b/"runs/stability_20261029/full-L2-beta0.25-stage3-seed1/config.json").read_text())
    slots=[("dev2",i) for i in range(8)]+[("dev13",i) for i in range(8)]+[("dev15",i) for i in range(8)]
    specs=[]
    for method in ["lower-beta","temporal-forecast"]:
        for seed in [1,2]:specs.append((method,seed,"selected-epoch3",s2/f"{method}-seed{seed}-epoch2",37473))
    for method in ["lower-beta","temporal-forecast"]:
        for seed in [3,4]:
            parent=held/f"{method}-seed{seed}";assert (parent/"STOP").exists()
            state=torch.load(parent/"last.ckpt",map_location="cpu",weights_only=False);initial=int(state["global_step"]);del state
            specs.append((method,seed,"existing-seed-gate600",parent,initial+600))
    for method in ["temporal-noise","forecast-render"]:
        for seed in [1,2]:specs.append((method,seed,"component-epoch2",screen/f"{method}-seed{seed}-epoch1",24982))
    for method in ["original","lower-beta","temporal-forecast"]:
        for seed in [5,6,7,8]:specs.append((method,seed,"new-seed-gate600",None,600))
    assert len(specs)==24
    jobs=[];after=[];records=[];exp=c/"experiments/promising20261029";(exp/"configs").mkdir(exist_ok=True);py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    for (method,seed,phase,parent,cap),(host,gpu) in zip(specs,slots):
        ident=f"{method}-seed{seed}-{phase}";folder=r/ident;folder.mkdir()
        cfg=json.loads((parent/"config.json").read_text()) if parent else copy.deepcopy(template)
        cfg.update(output=str(folder),seed=seed,max_steps=124910,stop_after=cap,retain_steps=[cap],checkpoint_every=2500,record_initial_weights=parent is None)
        if parent is None:
            cfg["model"]["beta"]=.0625 if method=="lower-beta" else .25
            if method=="temporal-forecast":
                cfg["model"].update(rho=.8,forecast_fraction=.5,forecast_seed=20261029+seed*100003);cfg["robust_variant"]=True
        initial=0
        if parent:
            state=torch.load(parent/"last.ckpt",map_location="cpu",weights_only=False);initial=int(state["global_step"]);assert state["campaign_samples"]==initial*128 and state["optimizer_states"] and state["campaign_rng"] and state["lr_schedulers"];del state
            assert initial<cap
            os.link(parent/"last.ckpt",folder/"last.ckpt")
            (folder/"resume-origin.json").write_text(json.dumps({"parent":str(parent),"initial_step":initial,"cap":cap,"preserve_parent_STOP":True,"scope":"Newoutputwithfullstatehardlink;Lightningatomicreplacementskeepallparentstateimmutable."},indent=2)+"\n")
        config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_text(config.read_text())
        command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step",str(cap)]+(["--resume"] if parent else [])
        job=dict(id=ident,host=host,gpu=gpu,output=str(folder),source=src,config=str(config),config_sha256=sha(config),command=command,phase=phase,method=method,training_seed=seed,initial_step=initial,stop_step=cap)
        jobs.append(job);records.append({k:job[k] for k in ["id","host","gpu","method","training_seed","phase","initial_step","stop_step","config","config_sha256"]})
        full_endpoint=phase in {"selected-epoch3","component-epoch2"}
        wanted=["health","quality","counterfactual","planning"] if full_endpoint else ["health"]
        for evalphase in wanted:
            evalseeds=[42,43,44,1042,1043,1044] if evalphase=="planning" else [None]
            for es in evalseeds:
                tmpl=next(j for j in templates["jobs"] if j["phase"]==evalphase and j["method"]=="original" and j["training_seed"]==1 and (evalphase!="planning" or j["evaluation_seed"]==42))
                child=copy.deepcopy(tmpl);child_id=f"{evalphase}-{ident}"+(f"-eval{es}" if es is not None else "");out=r/child_id
                child.update(id=child_id,output=str(out),host=host,gpu=gpu,source=src,depends_on=[ident],method=method,training_seed=seed,endpoint_phase=phase)
                child["command"][child["command"].index("--output")+1]=str(out)
                child["command"][child["command"].index("--checkpoint")+1]=str(folder/f"model-step{cap:07d}")
                if es is not None:
                    child["evaluation_seed"]=es;child["command"][child["command"].index("--seed")+1]=str(es)
                after.append(child)
    protocol=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),source_commit=commit,runs=records,
        basis="Twoepochseed1/2P0:original73/43,lowerbeta87/85,temporalforecast88/90.Fresh73.33/36.67,81.33/76,85.33/87.33. Prior600updateobjectivegridsdidnotconfirmstablegains.",
        model_parameters=51097134,state_dimension=64,batch_size=128,scheduler_total_steps=124910,training_seeds=[1,2,3,4,5,6,7,8],
        scope="Fourselectedrunsto3epochs;fourexistingseed3/4resumeonly600updates;fourcomponentcontrolsto2epochs;12newseed5-8runsfirst600updates. Noautomaticfurtherbudget.",
        gates=["Allnewseedandpreviouslyheldseedtrainingstopsafter600updatesandcheckpointevidence.","Newseed600updateposteriorunlearnedbehaviorisnotpersistentcollapseproof;requirecomparisonwithoriginal.","Selected/monoendpointsrequirematchedP0,quality,CF,healthreviewbeforefurtherbudget.","PreservealloldSTOP/heldplans;thisisfreshboundedoutputonly."],
        limitations=["Twoepochleadstillonlytwoinitialseeds;nextstageisconfirmationnotfinalsuperiority.","Sameparameter/updatecountdoesnotequalFLOPs;forecastingaddscompute.","Monocomponentfollowupisnecessarytoexplainstrongcombinedbenefitdespiteweakfirstepochscreen.","Noveltynotestablished;lossremainsper-sampleaveragedwithoutcrosssampleBN/SIGReg/EMA."],automatic_long_continuation=False)
    (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    env=copy.deepcopy(templates["environment"]);env["TMPDIR"]=str(b/"tmp/promising_20261029")
    plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=env,scope=protocol["scope"],protocol_sha256=sha(r/"protocol.json"),jobs=jobs+after)
    (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(src,indent=2)+"\n")
    for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
    print(json.dumps({"root":str(r),"source_commit":commit,"training_runs":24,"jobs":len(plan["jobs"]),"all_old_STOP_preserved":True}))
if __name__=="__main__":main()
