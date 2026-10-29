"""Freeze a bounded four-seed gradient-routing campaign from committed source."""
from pathlib import Path
import copy,datetime,hashlib,io,json,os,shutil,subprocess,tarfile

def main():
    b=Path("/efs/prescottsun/SPT-JEPA");code=b/"code";r=b/"runs/anchor_20261029";old=b/"runs/actionrank_20261029"
    assert not r.exists(),"Do not duplicate or mutate an existing campaign"
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=code,text=True).strip(),"Commit source first"
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=code,text=True).strip()
    r.mkdir();(r/"reports").mkdir();(r/"workers").mkdir();(b/"tmp/anchor_20261029").mkdir(exist_ok=True)
    source=r/"source-v1";source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git","archive",commit],cwd=code)),mode="r:") as archive:
        archive.extractall(source,filter="data")
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest={"source":str(source),"source_commit":commit,"files":{str(p.relative_to(source)):digest(p) for p in sorted(source.rglob("*")) if p.is_file()}}
    (r/"source-v1.json").write_text(json.dumps(manifest,indent=2)+"\n")
    src={"source":str(source),"source_commit":commit,"manifest":str(r/"source-v1.json"),"pythonpath":":".join(map(str,[b/"environments/completion-overlay",b/"environments/overlay",source,source/"runtime",b/"sources/stable-worldmodel"]))}
    oldplan=json.loads((old/"plan.json").read_text())
    slots=[(j["host"],j["gpu"]) for j in oldplan["jobs"] if j["phase"]=="adaptation600"]
    assert len(set(slots))==20
    methods={"original":"Exact original continuation control.",
             "drop10":"Drop each own past anchor with probability0.1 for the whole video window.",
             "drop25":"Drop each own past anchor with probability0.25 for the whole video window.",
             "drop50":"Drop each own past anchor with probability0.5 for the whole video window.",
             "context-mask":"Mask changes between the own past anchor and first observed frame, with5x5dilation. No future target forms the mask."}
    template=json.loads((old/"original-seed1-adapt600/config.json").read_text())
    templates=[j for j in oldplan["jobs"] if j["method"]=="original" and j["training_seed"]==1 and j["phase"]!="adaptation600"]
    jobs=[];evaluations=[];runs=[];exp=code/"experiments/anchor20261029";(exp/"configs").mkdir(exist_ok=True)
    for (method,seed),(host,gpu) in zip([(m,s) for m in methods for s in [1,2,3,4]],slots):
        ident=f"{method}-seed{seed}-adapt600";folder=r/ident;folder.mkdir()
        cfg=copy.deepcopy(template);cfg.update(output=str(folder),seed=seed)
        cfg["model"]={k:v for k,v in cfg["model"].items() if k not in {"frontier_method","private_seed"}}
        cfg["model"]["frontier_method"]="original" if method=="original" else ("anchor-context-mask" if method=="context-mask" else "anchor-dropout")
        cfg["model"]["private_seed"]=20261029+seed*100003
        if method.startswith("drop"):
            cfg["model"]["anchor_dropout_p"]=int(method[4:])/100
        config=folder/"config.json";config.write_text(json.dumps(cfg,indent=2)+"\n")
        (exp/"configs"/(ident+".json")).write_text(config.read_text())
        parent=b/f"runs/stability_20261029/snapshots/trainseed{seed}/step0024982"
        assert (parent/"full-state.ckpt").is_file()
        os.link(parent/"full-state.ckpt",folder/"last.ckpt")
        (folder/"resume-origin.json").write_text(json.dumps({"parent":str(parent),"initial_step":24982,"scope":"Full optimizer/scheduler/RNG/cursor resume; atomic checkpoint replacement preserves the linked parent."},indent=2)+"\n")
        job={"id":ident,"host":host,"gpu":gpu,"output":str(folder),"source":src,"config":str(config),"config_sha256":digest(config),
             "command":["/efs/prescottsun/project_jepa/envs/lewm/bin/python","-B","-u","-m","as_jepa_tools.train_stage","--config",str(config),"--stop-step","25582","--resume"],
             "phase":"adaptation600","method":method,"training_seed":seed}
        jobs.append(job);runs.append({k:job[k] for k in ["id","host","gpu","config","config_sha256","method","training_seed"]})
        for tmpl in templates:
            j=copy.deepcopy(tmpl)
            def transform(s):return s.replace(str(old),str(r)).replace("original-seed1-adapt600",ident)
            for key in ["id","output"]:j[key]=transform(j[key])
            j["command"]=[transform(s) for s in j["command"]]
            j.update(host=host,gpu=gpu,source=src,method=method,training_seed=seed,depends_on=[ident])
            evaluations.append(j)
    protocol={"created_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"root":str(r),"model_parameters":51097134,"state_dimension":64,"batch_size":128,"beta":.25,"sigma0":.1,"initial_step":24982,"first_stop_step":25582,"additional_training_updates":600,"scheduler_total_steps":124910,"training_seeds":[1,2,3,4],"methods":methods,"runs":runs,
              "basis":"Prior short screens gave no robust planning winner. Test whether past-anchor side information permits omission of control-relevant scene information; retain all original grounding gradients.",
              "scope":"Existing mature models, short adaptation only. All candidates remain sample-level, no cross-sample statistics, EMA or additional parameters. All targets and prior inputs remain attached; preserve RGB grounding gradients. Original flow/rate, sample-local past-anchor augmentation only. Inference still uses the normal past anchor.",
              "gate":"Stop all20training at600extra updates; review finite weights, posterior/interventions, paired P0 before any continuation. Do not infer that side-information shortcuts are established before paired task/probe evidence. Rendering without an anchor is harder; raw loss units and learning balance must be reported.",
              "references":[{"title":"Classifier-free diffusion guidance: conditioning dropout is a known principle","url":"https://arxiv.org/abs/2207.12598","scope":"Related training principle only; no classifier-free guidance or inference modification is used here."}],
              "automatic_long_continuation":False}
    (r/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    env=copy.deepcopy(oldplan["environment"]);env["TMPDIR"]=str(b/"tmp/anchor_20261029")
    plan={"root":str(r),"created_at_utc":protocol["created_at_utc"],"deadline":None,"environment":env,"scope":protocol["scope"],"protocol_sha256":digest(r/"protocol.json"),"jobs":jobs+evaluations}
    (r/"plan.json").write_text(json.dumps(plan,indent=2)+"\n");(r/"dispatcher-source.json").write_text(json.dumps(src,indent=2)+"\n")
    for name in ["protocol.json","plan.json"]:(exp/("dispatch-plan.json" if name=="plan.json" else name)).write_text((r/name).read_text())
    shutil.copy2(b/"reports/anchor-contracts-20261029.log",r/"reports/cpu-contracts.log")
    print(json.dumps({"root":str(r),"source_commit":commit,"jobs":len(plan["jobs"]),"slots":slots}))
if __name__=="__main__":main()
