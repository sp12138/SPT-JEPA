"""Finite, gated corrected-client comparison, with a separate source screen."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import subprocess
import time

import numpy as np

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import environment,command,TRAIN_PYTHON

HOSTS=("dev13","dev15","dev6")
MODULE="experiments.robocasa_refine.compare"


def variants(parent,refine):
    result=[dict(id="zero",zero=True)]
    for arm in ("temporal-forecast","long-rollout","spatial"):
        for seed in (1,2):
            run=str(parent/"training"/f"{arm}-seed{seed}")
            result.append(dict(id=f"{arm}-s{seed}",run=run,step=5000))
            if arm in ("long-rollout","spatial"):
                result.append(dict(id=f"{arm}-online-s{seed}",run=run,step=5000,online_correction=True))
            if arm=="spatial":
                for online in (False,True):
                    result.append(dict(id=f"spatial-support{'-online' if online else ''}-s{seed}",
                        run=run,step=5000,goal_metric="support",online_correction=online))
    for arm in ("control","clean","highnoise"):
        for seed in (1,2):
            result.append(dict(id=f"refine-{arm}-s{seed}",run=str(refine/"training"/f"{arm}-seed{seed}"),step=6000))
    for size,mode,pose in (("50m",2,"native"),("50m",3,"released"),("50m",3,"native"),
                           ("2b450k",3,"released"),("8b300k",3,"released")):
        result.append(dict(id=f"robojepa-{size}-m{mode}-{pose}",reference=size,
            reference_mode=mode,reference_pose_source=pose))
    return result


def initialize(root,parent,refine):
    assert not (root/"plan.json").exists()
    target=[dict(task=t,kitchen=k,seed=s) for t in ("reach","push") for k in range(1,11) for s in (8101,8102)]
    source=[dict(task=t,kitchen=k,seed=7401) for t in ("reach","push") for k in range(1,7)]
    plan=dict(parent=str(parent),refine=str(refine),target_cases=target,source_probes=source,
        variants=variants(parent,refine),max_hours=24,
        source_selection="First two cases per task passing independent strict reset replay; no model-score criterion",
        target_raw_resolution=[480,640],target_push_objects="released random flour_bag category",
        source_cases_are_development=True,target_cases_new_but_kitchens_previously_inspected=True,
        official_precision="encoder FP16, transition BF16 as Table14; literal client transition FP16 differs",
        no_automatic_long_training=True,baseline_pretraining_budgets_not_equal=True)
    save(root/"plan.json",plan)
    print(json.dumps(dict(variants=len(plan["variants"]),target_cases=len(target),source_probes=len(source))))


def add_integration_screen(root):
    """Fix integration cadence from TRAIN action support before screen scores."""
    p=root/"plan.json";plan=json.loads(p.read_text())
    assert not (root/"source-evaluation").exists()
    assert not (root/"plan-before-integration.json").exists()
    scale=json.loads((root/"train-control-scale.json").read_text())
    n=scale["integration_steps_from_train95"]
    assert n>=1
    parent=Path(plan["parent"])
    for arm in ("long-rollout","spatial"):
        for seed in (1,2):
            plan["variants"].append(dict(id=f"{arm}-integrate-s{seed}",
                run=str(parent/"training"/f"{arm}-seed{seed}"),step=5000,action_repeat=n))
    save(root/"plan-before-integration.json",json.loads(p.read_text()))
    plan["integration_rule"]="ceil(common max macro translation / TRAIN95 four-frame translation norm); no extra environment steps"
    plan["integration_scale"]=scale
    save(p,plan);print(json.dumps(dict(variants=len(plan["variants"]),integration_steps=n)))


def case_quality(root,case,source=False):
    from experiments.robocasa_compare import evaluate as base
    base.install_public(root)
    if source:
        from experiments.robocasa_refine import validation
        validation.ROOT=root;base.make_env=validation.make_env;base.case_name=validation.name
    else:
        base.make_case(root,**case)
    path=root/"cases"/(base.case_name(**case)+".npz")
    row=json.loads(path.with_suffix(".json").read_text())
    assert row["passed"] and sha(path)==row["sha256"]
    cache=np.load(path)
    env=base.make_env(**case)
    try:
        goal=base.reset(env,case["task"],case["seed"],"goal")
        obs=base.reset(env,case["task"],case["seed"],"init")
        base.goal_proof(case["task"],goal,obs)
        state=base.physics(env)
        assert np.array_equal(state["kitchen"],cache["kitchen"])
        error={key:float(np.max(np.abs(state[key]-cache[key]))) for key in ("qpos","qvel","time")}
        for key in error:assert error[key]<=1e-8,(key,error[key])
        rgb={key:float(np.mean((np.stack([value["img_0"],value["img_1"]]).astype(float)-cache[key].astype(float))**2))
             for key,value in (("initial",obs),("goal",goal))}
        assert max(rgb.values())<5.,rgb
        return dict(passed=True,case=case,sha256=row["sha256"],maximum_physics_errors=error,rgb_mse=rgb,
                    model_scores_used=False)
    finally:env.close()


def make_cases(root,host):
    plan=json.loads((root/"plan.json").read_text());refine=Path(plan["refine"])
    items=[dict(source=True,case=c) for c in plan["source_probes"]]
    items += [dict(source=False,case=c) for c in plan["target_cases"]]
    items=[x for i,x in enumerate(items) if i%3==HOSTS.index(host)]
    todo=queue.Queue()
    for item in items:todo.put(item)
    (root/"case-logs").mkdir(exist_ok=True)
    def worker(gpu):
        rows=[]
        while not todo.empty():
            try:item=todo.get_nowait()
            except queue.Empty:break
            c=item["case"];label=("source" if item["source"] else "target")+f"-{c['task']}-{c['kitchen']}-{c['seed']}"
            proof=root/"quality"/(label+".json")
            args=[str(root/"environment/bin/python"),"-u","-m",MODULE,"--mode","case","--root",str(root),
                  "--case-root",str(refine if item["source"] else root),"--case-json",json.dumps(c),"--output",str(proof)]
            if item["source"]:args.append("--source")
            try:
                command(args,root/"case-logs"/(label+".log"),environment(root,str(gpu),True),root/"code",1800)
                rows.append(json.loads(proof.read_text()))
            except Exception as error:
                result=dict(passed=False,case=c,source=item["source"],error=repr(error),method_negative=False,model_scores_used=False)
                save(proof,result);rows.append(result)
        return rows
    # Four EGL workers leave memory for the independent parity / recovery jobs.
    with ThreadPoolExecutor(max_workers=4) as pool:rows=sum(pool.map(worker,range(4)),[])
    save(root/f"cases-{host}.json",dict(disposed=True,results=rows))


def seal_cases(root,source_only=False):
    if not source_only:assert all((root/f"cases-{h}.json").exists() for h in HOSTS)
    plan=json.loads((root/"plan.json").read_text())
    def valid(source):
        cohort=plan["source_probes"] if source else plan["target_cases"]
        result=[]
        for case in cohort:
            prefix="source" if source else "target"
            p=root/"quality"/(prefix+f"-{case['task']}-{case['kitchen']}-{case['seed']}.json")
            if json.loads(p.read_text())["passed"]:result.append(case)
        return result
    source=valid(True);selected=[]
    for task in ("reach","push"):selected += [c for c in source if c["task"]==task][:2]
    target=[] if source_only else valid(False)
    record=dict(passed=len(selected)==4,source=selected,target=target,
        target_failures=len(plan["target_cases"])-len(target),source_failures=len(plan["source_probes"])-len(source),
        technical_selection_only=True,failed_cases_are_not_model_negatives=True)
    save(root/("SOURCE_CASES_READY.json" if source_only else "CASES_READY.json"),record)
    print(json.dumps(record),flush=True)


def gates(root):
    plan=json.loads((root/"plan.json").read_text());deadline=time.time()+3600
    paths=[root/"quality"/(f"source-{c['task']}-{c['kitchen']}-{c['seed']}.json") for c in plan["source_probes"]]
    while not all(p.exists() for p in paths) and time.time()<deadline:time.sleep(5)
    assert all(p.exists() for p in paths)
    seal_cases(root,True)
    while not all((root/f"cases-{h}.json").exists() for h in HOSTS) and time.time()<deadline:time.sleep(5)
    seal_cases(root)


def evaluate_variant(root,variant,output_base,cohort,source=True):
    from experiments.robocasa_compare import evaluate as base
    base.install_public(root)
    plan=json.loads((root/"plan.json").read_text()) if source else {}
    case_root=Path(plan["refine"]) if source else root
    if source:
        from experiments.robocasa_refine import validation
        validation.ROOT=case_root;base.make_env=validation.make_env;base.case_name=validation.name
    base.initialize_singleton()
    adapter=None
    if variant.get("reference"):
        adapter=base.RoboAdapter(root,variant["reference"],variant["reference_mode"],variant["reference_pose_source"])
    elif variant.get("run"):
        run=Path(variant["run"])
        assert json.loads((run/f"audit-{variant['step']:06d}.json").read_text())["passed"]
        adapter=base.SPTAdapter(root,run,variant["step"],variant.get("online_correction",False))
        from vforge.ops import OpsConfig,initialize_ops
        initialize_ops(OpsConfig(backend="sdpa",allow_fallback=False),device=__import__("torch").device("cuda"),parallel_state=None)
    rows=[]
    for case in cohort:
        label=f"{case['task']}-{case['kitchen']}-{case['seed']}"
        out=output_base/label
        if (out/"summary.json").exists():
            rows.append(dict(case=case,passed=True,inherited=True,path=str(out)));continue
        args=argparse.Namespace(root=case_root,run=Path(variant["run"]) if variant.get("run") else None,
            step=variant.get("step",5000),reference=variant.get("reference"),reference_mode=variant.get("reference_mode",2),
            reference_pose_source=variant.get("reference_pose_source","native"),zero=variant.get("zero",False),
            history_controls=variant.get("history_controls","achieved"),online_correction=variant.get("online_correction",False),
            goal_metric=variant.get("goal_metric","uniform"),technical=False,output=out,**case)
        args.action_repeat=variant.get("action_repeat",1)
        try:
            base.evaluate(args,adapter=adapter)
            rows.append(dict(case=case,passed=True,path=str(out)))
        except Exception as error:
            record=dict(case=case,passed=False,error=repr(error),method_negative=False,path=str(out))
            save(output_base/(label+"-execution-error.json"),record);rows.append(record)
            import torch
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    save(output_base/"DONE.json",dict(disposed=True,variant=variant,rows=rows,model_unchanged=True))


def worker(root,host):
    plan=json.loads((root/"plan.json").read_text());deadline=time.time()+plan["max_hours"]*3600
    ready=lambda:(root/"SOURCE_CASES_READY.json").exists() and (root/"official-client-parity.json").exists()
    entry_deadline=min(deadline,time.time()+3600)
    while not ready() and time.time()<entry_deadline:
        if (root/"STOP").exists():return
        time.sleep(5)
    assert ready(),"Corrected evaluation entry gates incomplete"
    assert json.loads((root/"official-client-parity.json").read_text())["passed"]
    cohort=json.loads((root/"SOURCE_CASES_READY.json").read_text());assert cohort["passed"]
    todo=queue.Queue()
    for i,v in enumerate(plan["variants"]):
        if i%3==HOSTS.index(host):todo.put(v)
    def gpu_worker(gpu):
        rows=[]
        if gpu==7 and host in ("dev13","dev6"):
            size="2b450k" if host=="dev13" else "8b300k"
            proof=root/f"official-client-parity-{size}.json"
            while not proof.exists() and time.time()<deadline:
                if (root/"STOP").exists():return rows
                time.sleep(5)
        # The single recovery training owns dev15 GPUs4-7 until its full audit.
        if host=="dev15" and gpu>=4:
            recovery=Path(plan["refine"])/"training/control-seed2/RECOVERY_COMPLETE.json"
            while not recovery.exists() and time.time()<deadline:
                if (root/"STOP").exists():return rows
                time.sleep(5)
        while not todo.empty() and time.time()<deadline and not (root/"STOP").exists():
            try:v=todo.get_nowait()
            except queue.Empty:break
            if v.get("step")==6000:
                audit=Path(v["run"])/"audit-006000.json"
                while not audit.exists() and time.time()<deadline:time.sleep(5)
            if v.get("reference") in ("2b450k","8b300k"):
                proof=root/f"official-client-parity-{v['reference']}.json"
                while not proof.exists() and time.time()<deadline:time.sleep(5)
                assert json.loads(proof.read_text())["passed"]
            output=root/"source-evaluation"/v["id"]
            argv=[str(root/"environment/bin/python"),"-u","-m",MODULE,"--mode","variant","--root",str(root),
                  "--variant-json",json.dumps(v),"--output",str(output)]
            try:
                command(argv,root/"logs"/(v["id"]+".log"),environment(root,str(gpu),True),root/"code",min(12*3600,max(1,deadline-time.time())))
                record=dict(variant=v["id"],disposed=True,path=str(output),host=host,gpu=gpu)
            except Exception as error:
                record=dict(variant=v["id"],execution_error=repr(error),method_negative=False,host=host,gpu=gpu)
            save(root/"status"/(v["id"]+".json"),record);rows.append(record)
        return rows
    (root/"logs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:rows=sum(pool.map(gpu_worker,range(8)),[])
    save(root/f"source-done-{host}.json",dict(disposed=True,records=rows))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["initialize","integration","cases","case","seal","gates","worker","variant"],required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--refine",type=Path)
    p.add_argument("--host",choices=HOSTS);p.add_argument("--case-root",type=Path)
    p.add_argument("--case-json");p.add_argument("--source",action="store_true")
    p.add_argument("--variant-json");p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent,a.refine)
    elif a.mode=="integration":add_integration_screen(a.root)
    elif a.mode=="cases":make_cases(a.root,a.host)
    elif a.mode=="seal":seal_cases(a.root)
    elif a.mode=="gates":gates(a.root)
    elif a.mode=="worker":worker(a.root,a.host)
    elif a.mode=="variant":
        selected=json.loads((a.root/"SOURCE_CASES_READY.json").read_text())["source"]
        evaluate_variant(a.root,json.loads(a.variant_json),a.output,selected)
    else:
        save(a.output,case_quality(a.case_root,json.loads(a.case_json),a.source))
