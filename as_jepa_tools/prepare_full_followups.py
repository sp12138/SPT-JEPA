"""Prepare dormant, immutable follow-up configs for the original decision rules.

This consumes explicit frozen training/evaluation sources; no training is run here.
Only new directories/jobs are added. Existing source/config/command entries are kept.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import yaml

from as_jepa_tools.workflow import atomic


def prepare(plan_file, source_info, registry_path):
    plan_file, registry_path = Path(plan_file), Path(registry_path)
    plan = json.loads(plan_file.read_text())
    root = Path(plan["root"])
    project = root.parents[1]
    source = json.loads(Path(source_info).read_text())
    py = plan["jobs"][0]["command"][0]
    known = {j["id"]: j for j in plan["jobs"]}
    added, trials = [], []

    def add(job):
        if job["id"] in known:
            return known[job["id"]]
        if (Path(job["output"]) / "dispatch.json").exists():
            raise RuntimeError("Unregistered existing run: " + job["id"])
        known[job["id"]] = job
        added.append(job)
        return job

    def config_job(name, config, family, host, gpu, train_source, depends=(), requires=(), resume_parent=None, expected_step=37473):
        if name in known:
            return known[name]
        out = root / name
        if out.exists():
            raise RuntimeError("Existing new-config directory: " + str(out))
        out.mkdir()
        config = copy.deepcopy(config)
        if family == "L2":
            config["output"] = str(out)
            path = out / "config.json"
            path.write_text(json.dumps(config, indent=2))
            command = [py, "-B", "-u", "-m", "as_jepa_lewm.train", str(path)]
            actual_source = train_source
            if resume_parent:
                command = [py, "-B", "-u", "-m", "as_jepa_tools.resume_lewm",
                           "--parent", str(resume_parent), "--config", str(path),
                           "--expected-step", str(expected_step),
                           "--train-source", train_source["source"],
                           "--train-pythonpath", train_source["pythonpath"]]
                actual_source = source
        else:
            config["runtime"]["output_dir"] = str(out)
            config["model"]["extras"]["campaign_output"] = str(out)
            path = out / "config.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            command = [py, "-B", "-u", str(Path(train_source["source"]) / "jobs/as_jepa/train_entry.py"), str(path)]
            actual_source = train_source
        return add(dict(id=name, host=host, gpu=gpu, output=str(out), source=actual_source,
                        config=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        depends_on=list(depends), requires=list(requires), command=command))

    def metric(name, parent, step, host, gpu, kind, depends=(), requires=()):
        out = root / name
        checkpoint = root / parent / ("model-step" + str(step).zfill(7))
        command = [py, "-B", "-u", "-m", "as_jepa_tools." + kind,
                   "--checkpoint", str(checkpoint),
                   "--data", str(project / "data/pusht-lance-full-v1/phase0"),
                   "--pose", str(root.parent / "insights12h_20261027/pose-v2/weights.pt"),
                   "--output", str(out / "results")]
        if kind == "evaluate":
            command += ["--cases", "256", "--steps", "50"]
        return add(dict(id=name, host=host, gpu=gpu, output=str(out), source=source,
                        depends_on=list(depends), requires=list(requires), command=command))

    def endpoint(parent, family, host, gpu, total, requires=()):
        name = "eval-" + parent
        out = root / name
        return add(dict(id=name, host=host, gpu=gpu, output=str(out), source=source,
                        depends_on=[parent], requires=list(requires),
                        command=[py, "-B", "-u", "-m", "as_jepa_tools.completion_evaluate",
                                 "--parent", str(root / parent), "--output", str(out / "results"),
                                 "--data", str(project / "data/pusht-lance-full-v1/phase0"),
                                 "--pose", str(root.parent / "insights12h_20261027/pose-v2/weights.pt"),
                                 "--i3d", str(project / "weights/i3d_torchscript.pt"),
                                 "--upstream", str(project / "sources/stable-worldmodel"),
                                 "--planning-data", str(project / "data/planning"),
                                 "--mode", family, "--expected-step", str(total)]))

    initial = {}
    for family in ["L2", "S2"]:
        initial[family] = [j for j in plan["jobs"] if j["id"].startswith("full-" + family + "-beta") and j["id"].endswith("-stage3-seed0")]
    if not all(initial.values()):
        raise RuntimeError("Original base price grids must already be registered")
    for family in ["L2", "S2"]:
        template_job = initial[family][0]
        template_config = (json.loads if family=="L2" else yaml.safe_load)(Path(template_job["config"]).read_text())
        prices = sorted((json.loads(Path(j["config"]).read_text())["model"]["beta"] if family=="L2" else yaml.safe_load(Path(j["config"]).read_text())["model"]["extras"]["as_jepa"]["beta"]) for j in initial[family])
        low_extensions = [min(prices)/4**k for k in [1,2,3]]
        high_extensions = [max(prices)*4**k for k in [1,2]]
        for phase in ["default", "ramp"]:
            for index, beta in enumerate(prices + low_extensions + high_extensions):
                suffix = ("ramp-" if phase=="ramp" else "") + "beta" + format(beta, ".12g")
                stem = "full-" + family + "-" + suffix
                stage, full = stem + "-stage3-seed0", stem + "-10epoch-seed0"
                activation = None
                required = []
                if phase=="ramp":
                    required.append(str(root / "selection" / (family + "-default-all-collapsed.json")))
                if beta in low_extensions + high_extensions:
                    activation = str(root / "selection" / (stem + "-ACTIVATE.json"))
                    required.append(activation)
                train_source = template_job["source"] if family=="L2" or phase=="default" else source
                cfg = copy.deepcopy(template_config)
                if family=="L2":
                    cfg["model"]["beta"] = beta
                    cfg.update(max_steps=124910, stop_after=37473, checkpoint_every=12491,
                               max_seconds=31536000, retain_steps=[37473,99928,124910])
                    if phase=="ramp":
                        cfg["ramp_steps"] = int(.05*124910)
                else:
                    cfg["model"]["extras"]["as_jepa"]["beta"] = beta
                    cfg["runtime"]["max_steps"] = 37473
                    cfg["checkpoint"]["every_n_steps"] = 12491
                    cfg["model"]["extras"]["export_every"] = 12491
                    if phase=="ramp":
                        cfg["model"]["extras"]["ramp_steps"] = int(.05*124910)
                if stage in known:
                    base = known[stage]
                    host, gpu = base["host"], base["gpu"]
                    train_source = base["source"]
                else:
                    host, gpu = ("dev13", index%8) if family=="L2" else ("dev15", index%8)
                    config_job(stage, cfg, family, host, gpu, train_source,
                               depends=["full-" + family + "-smoke"], requires=required)
                audit3 = "collapse-" + stage
                metric(audit3, stage, 37473, host, gpu, "collapse_audit", [stage])
                full_cfg = copy.deepcopy(cfg)
                gate = str(root / audit3 / "results/PASS.json")
                if family=="S2":
                    full_cfg["runtime"]["max_steps"] = 124910
                    full_cfg["resume"]["path"] = str(root / stage / "checkpoints/37473.dcp")
                config_job(full, full_cfg, family, host, gpu, train_source, [audit3], [gate],
                           resume_parent=root/stage if family=="L2" else None)
                final_audit = "collapse-" + full
                metric(final_audit, full, 124910, host, gpu, "collapse_audit", [full])
                for epoch, step in [(8,99928),(10,124910)]:
                    metric("score-" + full + "-epoch" + str(epoch), full, step, host, gpu, "evaluate", [full])
                planning = None
                if family=="L2":
                    planning = "plan-" + full + "-seed42"
                    out = root / planning
                    add(dict(id=planning, host=host, gpu=gpu, output=str(out), source=source,
                             depends_on=[final_audit], requires=[str(root/final_audit/"results/PASS.json")],
                             command=[py,"-B","-u","-m","as_jepa_tools.planning",
                                      "--checkpoint",str(root/full/"model-step0124910"),
                                      "--output",str(out/"results"),"--upstream",str(project/"sources/stable-worldmodel"),
                                      "--data-root",str(project/"data/planning"),"--test-episodes",str(project/"data/pusht-lance-full-v1/phase0/test_episodes.npy"),
                                      "--seed","42","--episodes","50"]))
                trials.append(dict(id=stem, family=family, phase=phase, beta=beta, activation=activation,
                                   stage3_audit=audit3, full_job=full, final_audit=final_audit,
                                   score8="score-"+full+"-epoch8", score10="score-"+full+"-epoch10",
                                   planning=planning))
                selected = str(root/"selection"/(stem+"-SELECTED.json"))
                endpoint(full, family, host, gpu, 124910, [selected])
                # All follow-up arms stay dormant until this exact trial is selected.
                arms = [("seed1",1,None,124910,selected),("seed2",2,None,124910,selected)]
                if family=="L2":
                    arms.append(("g0-seed0",0,"g0",124910,selected))
                arms += [("state128-seed0",0,"state128",124910,str(root/"selection"/(stem+"-CAPACITY128.json"))),
                         ("20epoch-seed0",0,None,249820,str(root/"selection"/(stem+"-EXTEND20.json")))]
                for arm, seed, change, total, arm_marker in arms:
                    arm_stem = "final-" + stem + "-" + arm
                    start_name, end_name = arm_stem+"-stage3", arm_stem+"-complete"
                    acfg = copy.deepcopy(cfg)
                    arm_host, arm_gpu = ("dev2",(gpu+seed+3)%8) if family=="L2" else ("dev15",(gpu+seed+2)%8)
                    if family=="L2":
                        acfg.update(seed=seed,max_steps=total,stop_after=37473,
                                    retain_steps=sorted(set([37473,99928,124910,total])))
                        if phase=="ramp":
                            acfg["ramp_steps"]=int(.05*total)
                        if change=="g0":
                            acfg["gap"]=0;acfg["model"]["anchor_gap"]=0
                        if change=="state128":acfg["model"]["state_width"]=32
                    else:
                        acfg["resume"]["path"]="none"
                        acfg["runtime"].update(seed=seed,max_steps=37473)
                        acfg["data"]["sampler_seed"]=seed
                        if phase=="ramp":
                            acfg["model"]["extras"]["ramp_steps"]=int(.05*total)
                        if change=="state128":acfg["model"]["extras"]["as_jepa"]["state_width"]=32
                    config_job(start_name,acfg,family,arm_host,arm_gpu,train_source,requires=[arm_marker])
                    check="collapse-"+start_name
                    metric(check,start_name,37473,arm_host,arm_gpu,"collapse_audit",[start_name])
                    acfg=copy.deepcopy(acfg)
                    if family=="S2":
                        acfg["runtime"]["max_steps"]=total
                        acfg["resume"]["path"]=str(root/start_name/"checkpoints/37473.dcp")
                    config_job(end_name,acfg,family,arm_host,arm_gpu,train_source,[check],
                               [str(root/check/"results/PASS.json")],resume_parent=root/start_name if family=="L2" else None)
                    endpoint(end_name,family,arm_host,arm_gpu,total)
    plan["jobs"] += added
    atomic(plan_file,plan)
    registry = dict(root=str(root), trials=trials, source_info=str(Path(source_info).resolve()),
                    scope="Prespecified conditional queue; dormant jobs consume no GPU. No selected config until actual metrics satisfy the original selection rule.")
    atomic(registry_path,registry)
    print(json.dumps(dict(added_jobs=len(added),total_jobs=len(plan["jobs"]),trials=len(trials))))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",required=True)
    p.add_argument("--source-info",required=True)
    p.add_argument("--registry",required=True)
    a=p.parse_args()
    prepare(a.plan,a.source_info,a.registry)


if __name__=="__main__":
    main()
