"""Supplement the already selected two-epoch arms with matched task diagnostics."""
from pathlib import Path
import copy,datetime,hashlib,json,subprocess

def main():
    b=Path("/efs/prescottsun/SPT-JEPA");c=b/"code";r=b/"runs/stage2eval_20261029";s2=b/"runs/shortscreen2_20261029";base=b/"runs/stability_20261029"
    assert not r.exists()
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=c,text=True).strip()
    parent=json.loads((s2/"plan.json").read_text());templates=json.loads((b/"runs/anchor_20261029/plan.json").read_text())
    source=templates["jobs"][0]["source"]
    slots=[("dev2",i) for i in range(2,8)]+[("dev13",i) for i in range(1,8)]+[("dev15",i) for i in range(1,8)]
    r.mkdir();(r/"workers").mkdir();(r/"reports").mkdir();(b/"tmp/stage2eval_20261029").mkdir(exist_ok=True)
    py="/efs/prescottsun/project_jepa/envs/lewm/bin/python";jobs=[];records=[];counter=0
    for method in ["original","lower-beta","temporal-forecast"]:
        for seed in [1,2]:
            if method=="original":
                cp=base/f"snapshots/trainseed{seed}/step0024982";requires=[str(cp/"weights.pt")]
            else:
                cp=s2/f"{method}-seed{seed}-epoch2/model-step0024982";requires=[str(cp.parent/"stage-complete.json"),str(cp/"weights.pt")]
            records.append(dict(method=method,training_seed=seed,checkpoint=str(cp),requires=requires))
            for es in [1042,1043,1044]:
                host,gpu=slots[counter%20];counter+=1;ident=f"fresh-{method}-seed{seed}-eval{es}";out=r/ident
                tmpl=next(j for j in templates["jobs"] if j["phase"]=="planning" and j["method"]=="original" and j["training_seed"]==1 and j["evaluation_seed"]==42)
                command=copy.deepcopy(tmpl["command"]);command[command.index("--checkpoint")+1]=str(cp);command[command.index("--output")+1]=str(out);command[command.index("--seed")+1]=str(es)
                jobs.append(dict(id=ident,host=host,gpu=gpu,output=str(out),source=source,requires=requires,command=command,phase="fresh-planning",method=method,training_seed=seed,evaluation_seed=es))
            for phase in ["health","quality","counterfactual"]:
                host,gpu=slots[counter%20];counter+=1;ident=f"{phase}-{method}-seed{seed}";out=r/ident
                tmpl=next(j for j in templates["jobs"] if j["phase"]==phase and j["method"]=="original" and j["training_seed"]==1)
                command=copy.deepcopy(tmpl["command"]);command[command.index("--checkpoint")+1]=str(cp);command[command.index("--output")+1]=str(out)
                jobs.append(dict(id=ident,host=host,gpu=gpu,output=str(out),source=source,requires=requires,command=command,phase=phase,method=method,training_seed=seed))
    env=copy.deepcopy(templates["environment"]);env["TMPDIR"]=str(b/"tmp/stage2eval_20261029")
    protocol=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),scope="Matched24982update checkpoints of the evidence-selected second-stage arms. No new training. Primaryeval42/43belong to parentcampaign; this supplement uses fresh1042/1043/1044, health32,quality64Euler20,CF16Euler20.",methods=["original","lower-beta","temporal-forecast"],training_seeds=[1,2],model_parameters=51097134,records=records,automatic_long_continuation=False,
        limitations=["Two diagnostic training seeds, exploratory2epochcheckpoints, not final3epochstability.","Fresh cases may share exactquerieswithprimary; reporter counts/excludes overlap in a separate analysis.","Lossbeta differs deliberately forlower-beta; sharedsampler/source and model size retained."])
    (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    plan=dict(root=str(r),created_at_utc=protocol["created_at_utc"],deadline=None,environment=env,scope=protocol["scope"],jobs=jobs)
    (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(source,indent=2)+"\n")
    d=c/"experiments/stage2eval20261029"
    (d/"protocol.json").write_text((r/"protocol.json").read_text());(d/"dispatch-plan.json").write_text((r/"plan.json").read_text())
    print(json.dumps({"root":str(r),"jobs":len(jobs),"training_jobs":0}))
if __name__=="__main__":main()
