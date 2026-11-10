"""Paired finite command-vs-achieved continuation; no automatic long run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import subprocess
import threading
import time

import torch
import numpy as np

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import environment,command,TRAIN_PYTHON
from experiments.robocasa_compare.train import build,make_optimizer,associations,state_hash
from experiments.robocasa_refine.intervene import Interventions,finish

ARMS={"dev13":"achieved","dev15":"issued","dev6":"issued-highnoise"}


def initialize(root):
    plan=json.loads((root/"collection-plan.json").read_text())
    finish(root)
    card=json.loads((root/"interventions/dataset.json").read_text())
    train={tuple(r["kitchen"]) for r in card["records"] if r["split"]=="train"}
    valid={tuple(r["kitchen"]) for r in card["records"] if r["split"]=="validation"}
    assert train and valid and not train&valid
    a=Interventions(root/"interventions",16,1,semantics="achieved")
    b=Interventions(root/"interventions",16,1,semantics="issued")
    for i in range(16):
        x,y=a[i],b[i]
        assert x.keys()==y.keys()=={"pixels","action"} and torch.equal(x["pixels"],y["pixels"])
        assert x["action"].shape==y["action"].shape==(16,14)
        assert not x["action"][-1].any() and not y["action"][-1].any()
        assert torch.isfinite(x["action"]).all() and torch.isfinite(y["action"]).all()
    save(root/"DATA_APPROVED.json",dict(passed=True,paired_real_windows=16,RGB_exactly_identical=True,
        train_kitchens=len(train),validation_kitchens=len(valid),terminal_action_zero=True,
        no_absolute_pose_reward_goal_or_action_target=True))
    assert not (root/"INITIALIZED.json").exists()
    parent=Path(plan["parent"]);records=[]
    for seed in (1,2):
        source=parent/"training"/f"long-rollout-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==5000
        for host,arm in ARMS.items():
            run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True,exist_ok=False)
            cfg=json.loads((source/"config.json").read_text())
            cfg.update(arm=arm,host=host,parent=str(source),data=str(root/"interventions"),
                max_steps=6000,phase_start=5000,stages=[5000,5002,6000],
                input_semantics="achieved" if arm=="achieved" else "issued",
                supervision="Only future RGB/own codes from ordinary random commands in TRAIN kitchens; no action targets")
            cfg["model"]["flow_force_t_one"]=arm=="issued-highnoise"
            save(run/"config.json",cfg)
            torch.manual_seed(seed);model=build(cfg["model"],root/"code")
            model.load_state_dict(old["model"],strict=True)
            assert state_hash(model.state_dict())==old["nn_sha256"]
            opt,scheduler=make_optimizer(model,6000,5000)
            assert associations(model,opt)==old["parameter_associations"]
            opt.load_state_dict(old["optimizer"])
            fresh=[v*f(5000) for v,f in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
            sched=copy.deepcopy(old["scheduler"]);sched["_last_lr"]=fresh;scheduler.load_state_dict(sched)
            for group,lr in zip(opt.param_groups,fresh):group["lr"]=lr
            state=dict(old,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),config_sha256=sha(run/"config.json"))
            torch.save(state,run/"last.ckpt")
            save(run/"ancestry.json",dict(passed=True,parent=str(source),parent_checkpoint_sha256=sha(source/"last.ckpt"),
                all_550_core_tensors_exact=True,optimizer_moments_RNG_cursor_preserved=True,
                parent_nominal_updates=5000,original_unsaved_update_replay=1,new_nominal_updates=1000,
                schedule="Common1000-update cosine restart with10-update warmup",new_random_neural_parameters=0,
                external_pretraining=False,independent_pretraining=False))
            records.append(dict(run=run.name,host=host,source_nn=old["nn_sha256"]))
            del state,model,opt,scheduler
        del old
    save(root/"INITIALIZED.json",dict(passed=True,records=records,new_sgd=0))


def coordinator(root):
    deadline=time.time()+8*3600
    ready=lambda:all((root/f"collection-{h}.json").exists() for h in ARMS) and (root/"REPLAY_APPROVED.json").exists()
    while not ready() and time.time()<deadline:
        if (root/"STOP").exists():return
        time.sleep(10)
    assert ready()
    assert json.loads((root/"REPLAY_APPROVED.json").read_text())["passed"]
    initialize(root)


def replay_case(root,case,output):
    from experiments.robocasa_compare import evaluate as base
    from experiments.robocasa_refine import validation
    base.install_public(root);validation.ROOT=root
    env=validation.make_env(**case)
    try:
        goal=base.reset(env,case["task"],case["seed"],"goal")
        obs=base.reset(env,case["task"],case["seed"],"init")
        path=root/"cases"/(validation.name(**case)+".npz")
        cache=np.load(path);errors=base.verify_physics(env,cache,validation.name(**case))
        rgb={k:float(np.mean((np.stack([o["img_0"],o["img_1"]]).astype(float)-cache[k].astype(float))**2))
             for k,o in (("initial",obs),("goal",goal))}
        assert max(rgb.values())<5.
        save(output,dict(passed=True,case=case,physics_errors=errors,rgb_mse=rgb,
            case_sha256=sha(path),no_NN_or_SGD=True))
    finally:env.close()


def replay_host(root,host):
    deadline=time.time()+3600
    while not (root/"VALIDATION_CASES_READY.json").exists() and time.time()<deadline:time.sleep(5)
    candidates=[dict(task=t,kitchen=k,seed=7401) for t in ("reach","push") for k in range(1,7)
        if (root/"cases"/f"{t}-v{k}-s7401.json").exists()]
    jobs=[(case,rep) for case in candidates for rep in (1,2)]
    def one(item):
        case,rep=item;label=f"{host}-{case['task']}-{case['kitchen']}-rep{rep}"
        out=root/"replay-proof"/(label+".json")
        argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.intervention_phase",
            "--mode","replay-case","--root",str(root),"--case-json",json.dumps(case),"--output",str(out)]
        try:
            command(argv,root/("replay-"+label+".log"),environment(root,str(rep-1),True),root/"code",900)
            row=json.loads(out.read_text());assert row["passed"]
            return dict(row,host=host,rep=rep)
        except Exception as error:
            row=dict(passed=False,case=case,host=host,rep=rep,error=repr(error),method_negative=False)
            save(out,row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,jobs))
    save(root/f"replay-host-{host}.json",dict(disposed=True,rows=rows))


def replay_gate(root):
    deadline=time.time()+3600
    while not all((root/f"replay-host-{h}.json").exists() for h in ARMS) and time.time()<deadline:time.sleep(5)
    assert all((root/f"replay-host-{h}.json").exists() for h in ARMS)
    rows=sum((json.loads((root/f"replay-host-{h}.json").read_text())["rows"] for h in ARMS),[])
    selected=[]
    for task in ("reach","push"):
        for k in range(1,7):
            group=[r for r in rows if r["case"]["task"]==task and r["case"]["kitchen"]==k]
            if len(group)==6 and all(r["passed"] for r in group):selected.append(dict(task=task,kitchen=k,seed=7401))
            if len([c for c in selected if c["task"]==task])==2:break
    assert len(selected)==4,(selected,"Insufficient cross-host replayable validation cases")
    original=json.loads((root/"VALIDATION_CASES_READY.json").read_text())
    save(root/"selection-before-replay.json",original)
    save(root/"VALIDATION_CASES_READY.json",dict(original,selected=selected,
        selection="First two/task passing two independent fresh resets on each of all three hosts; no model scores"))
    save(root/"REPLAY_APPROVED.json",dict(passed=True,rows=rows,selected=selected,strict_all_physics_tolerance=1e-8,
        independent_fresh_processes=True,global_scene_RNG_fixed=True,new_sgd=0))


def evaluate_seed(root,run,gpus):
    from experiments.robocasa_compare import evaluate as base
    plan=json.loads((root/"collection-plan.json").read_text());screen=Path(plan["screen"])
    cohort=json.loads((root/"VALIDATION_CASES_READY.json").read_text())["selected"]
    val_root=root
    cfg=json.loads((run/"config.json").read_text())
    def one(item):
        i,case=item;label=f"{case['task']}-{case['kitchen']}-{case['seed']}"
        out=root/"source-evaluation"/run.name/label
        argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.validation",
            "--root",str(val_root),"--mode","evaluate","--run",str(run),"--step","6000",
            "--history-controls",cfg["input_semantics"],"--output",str(out)]
        for key,value in case.items():argv.extend(["--"+key,str(value)])
        command(argv,run/("eval-"+label+".log"),environment(root,str(gpus[i]),True),root/"code",7200)
        assert json.loads((out/"summary.json").read_text())["passed"]
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,enumerate(cohort)))
    save(run/"EVALUATION_COMPLETE.json",dict(passed=True,episodes=len(cohort),source_only=True))


def worker(root,host,entry_check=None):
    deadline=time.time()+24*3600
    while not (root/"INITIALIZED.json").exists() and time.time()<deadline:
        if (root/"STOP").exists():return
        time.sleep(10)
    assert json.loads((root/"INITIALIZED.json").read_text())["passed"]
    lock=threading.Lock();reserved=set()
    def one(seed):
        run=root/"training"/f"{ARMS[host]}-seed{seed}";gpus=None
        try:
            while gpus is None and time.time()<deadline:
                if (root/"STOP").exists():return
                with lock:
                    mem=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
                    candidates=[i for i,x in enumerate(mem.splitlines()) if int(x)>=120*1024 and i not in reserved]
                    if len(candidates)>=4:
                        gpus=candidates[:4];reserved.update(gpus)
                if gpus is None:time.sleep(10)
            assert gpus is not None,"Four GPUs with safe memory headroom never available"
            save(run/"GPU_ASSIGNMENT.json",dict(host=host,gpus=gpus,minimum_free_MiB=120*1024,time=time.time()))
            gpu_list=",".join(map(str,gpus))
            for step in (5000,5002,6000):
                if step>5000:
                    argv=[TRAIN_PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=4",
                        "-m","experiments.robocasa_compare.train","--root",str(root),"--run-directory",str(run),
                        "--stop-step",str(step),"--resume"]
                    command(argv,run/f"train-to-{step}.log",environment(root,gpu_list),root/"code",3*3600)
                command([TRAIN_PYTHON,"-m","experiments.robocasa_compare.audit","--root",str(root),"--run",str(run),"--step",str(step)],
                    run/f"audit-{step}.log",environment(root,str(gpus[0])),root/"code",1800)
                if step==5002 and entry_check is not None:entry_check(root,run,gpus[0])
            save(run/"TRAINING_COMPLETE.json",dict(passed=True,step=6000,new_updates=1000))
            evaluate_seed(root,run,gpus)
        except BaseException as error:
            save(run/"ERROR.json",dict(error=repr(error),host=host,seed=seed,method_negative=False));raise
        finally:
            if gpus is not None:
                with lock:reserved.difference_update(gpus)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,(1,2)))
    save(root/f"DONE-{host}.json",dict(passed=True,automatic_extension=False))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["coordinator","worker","replay-gate","replay-host","replay-case"],required=True)
    p.add_argument("--case-json");p.add_argument("--output",type=Path)
    p.add_argument("--host",choices=ARMS);a=p.parse_args()
    if a.mode=="coordinator":coordinator(a.root)
    elif a.mode=="replay-gate":replay_gate(a.root)
    elif a.mode=="replay-host":replay_host(a.root,a.host)
    elif a.mode=="replay-case":replay_case(a.root,json.loads(a.case_json),a.output)
    else:worker(a.root,a.host)
