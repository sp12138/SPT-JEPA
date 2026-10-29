"""Confirm the selected anchor leads without extending any training budget."""
from pathlib import Path
import copy,datetime,hashlib,json,os,subprocess

def main():
    b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";old=b/"runs/anchor_20261029";r=b/"runs/anchorconfirm_20261029"
    assert not r.exists(),"Never duplicate an existing campaign"
    p=json.loads((old/"plan.json").read_text())
    assert len(p["jobs"])==120 and all(json.loads((old/"exits"/(j["id"]+".json")).read_text())["returncode"]==0 for j in p["jobs"])
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip(),"Commit preparation before use"
    r.mkdir();(r/"reports").mkdir();(r/"workers").mkdir();(b/"tmp/anchorconfirm_20261029").mkdir(exist_ok=True)
    methods={"original":"Matched original numerical-repeat control","drop25":"Selected25%own-anchor dropout lead","context-mask":"Secondary observed-context masking mechanism contrast; primary seed2 change is negative"}
    source=p["jobs"][0]["source"];sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    training_slots=[("dev2",i) for i in range(2,8)]+[("dev13",i) for i in range(1,7)]
    validation_slots=[("dev13",7)]+[("dev15",i) for i in range(1,8)]
    jobs=[];after=[];runs=[];exp=c/"experiments/anchorconfirm20261029";(exp/"configs").mkdir(exist_ok=True)
    py="/efs/prescottsun/project_jepa/envs/lewm/bin/python"
    for (method,seed),(host,gpu) in zip([(m,s) for m in methods for s in [1,2,3,4]],training_slots):
        ident=f"{method}-seed{seed}-adapt600";folder=r/ident;folder.mkdir()
        original=next(j for j in p["jobs"] if j["phase"]=="adaptation600" and j["method"]==method and j["training_seed"]==seed)
        cfg=json.loads(Path(original["config"]).read_text());cfg["output"]=str(folder)
        config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n");(exp/"configs"/(ident+".json")).write_text(config.read_text())
        parent=b/f"runs/stability_20261029/snapshots/trainseed{seed}/step0024982"
        os.link(parent/"full-state.ckpt",folder/"last.ckpt")
        (folder/"resume-origin.json").write_text(json.dumps({"parent":str(parent),"step":24982,"scope":"Same full-state/RNG600update numerical repeat, not a new training seed."},indent=2)+"\n")
        j=dict(id=ident,host=host,gpu=gpu,output=str(folder),source=source,config=str(config),config_sha256=sha(config),
               command=[py,"-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","25582","--resume"],
               phase="adaptation600",method=method,training_seed=seed)
        jobs.append(j);runs.append({k:j[k] for k in ["id","host","gpu","config","config_sha256","method","training_seed"]})
        for template in [j for j in p["jobs"] if j["method"]==method and j["training_seed"]==seed and j["phase"]!="adaptation600"]:
            child=copy.deepcopy(template)
            for key in ["id","output"]:child[key]=child[key].replace(str(old),str(r))
            child["command"]=[x.replace(str(old),str(r)) for x in child["command"]]
            child.update(host=host,gpu=gpu,depends_on=[ident])
            after.append(child)
    validation=[];index=0
    for method in methods:
        for seed in [1,2,3,4]:
            template=next(j for j in p["jobs"] if j["phase"]=="planning" and j["method"]==method and j["training_seed"]==seed and j["evaluation_seed"]==42)
            for es in [1042,1043,1044]:
                host,gpu=validation_slots[index%len(validation_slots)];index+=1
                ident=f"fresh-{method}-seed{seed}-eval{es}";folder=r/ident;command=copy.deepcopy(template["command"])
                command[command.index("--output")+1]=str(folder);command[command.index("--seed")+1]=str(es)
                validation.append(dict(id=ident,host=host,gpu=gpu,output=str(folder),source=source,command=command,
                    phase="fresh-planning",method=method,training_seed=seed,evaluation_seed=es,weights_scope="Original short-screen25582weights; independent of the concurrent repeat."))
            host,gpu=validation_slots[index%len(validation_slots)];index+=1
            ident=f"probe-{method}-seed{seed}";folder=r/ident;checkpoint=old/f"{method}-seed{seed}-adapt600/model-step0025582"
            validation.append(dict(id=ident,host=host,gpu=gpu,output=str(folder),source=source,
                command=[py,"-B","-u","-m","as_jepa_tools.probes","--checkpoint",str(checkpoint),"--data",str(b/"data/pusht-lance-full-v1/phase0"),"--output",str(folder),"--train","10000","--test","2500","--fp32"],
                phase="probe",method=method,training_seed=seed))
    protocol=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),model_parameters=51097134,batch_size=128,beta=.25,sigma0=.1,
        initial_step=24982,first_stop_step=25582,additional_training_updates=600,scheduler_total_steps=124910,training_seeds=[1,2,3,4],methods=methods,runs=runs,
        purpose="Check selected anchor leads on fresh evaluation seeds and repeat training from exactly the same initial full state/RNG. No longer training yet.",
        limitations=["This is mature-checkpoint adaptation, not from-scratch training.","Same-RNGrepeat measures numerical/runtime trajectory variation, not new independent training seeds.","Leads selected using eval42/43; fresh1042/1043/1044 are confirmation for these fixed weights.","Probes use privileged state only for evaluation, fit only on train episodes; no state labels enter training losses."],
        concurrency="12repeatGPUslots+8fresh/probeGPUslots+4independentselectedshortscreen2slots",automatic_long_continuation=False)
    (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    environment=copy.deepcopy(p["environment"]);environment["TMPDIR"]=str(b/"tmp/anchorconfirm_20261029")
    plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=environment,
        scope=protocol["purpose"],protocol_sha256=sha(r/"protocol.json"),jobs=jobs+validation+after)
    assert len(plan["jobs"])==120
    (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(source,indent=2)+"\n")
    for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
    print(json.dumps({"root":str(r),"jobs":len(plan["jobs"]),"repeat_runs":12,"fresh_planning_jobs":36,"probes":12,"source":source["source_commit"]}))
if __name__=="__main__":main()
