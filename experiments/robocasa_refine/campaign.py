"""Finite paired 5000->6000 continuation and source-validation evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import TRAIN_PYTHON,environment,command
from experiments.robocasa_compare.train import build,make_optimizer,associations,state_hash

ARMS={"dev13":"control","dev15":"clean","dev6":"highnoise"}


def initialize(root,parent):
    assert not (root/"INITIALIZED.json").exists()
    card=json.loads((parent/"packed/dataset.json").read_text())
    kitchens=sorted({tuple(r["kitchen"]) for r in card["records"] if r["split"]=="validation" and r["eligible"]})[:6]
    save(root/"validation-plan.json",dict(kitchens=kitchens,seed=7401,
        rule="First six sorted source-validation kitchens; first two geometrically valid cases per task, selected before model scores",
        target_test_kitchens_used=False))
    records=[]
    for seed in (1,2):
        source=parent/"training"/f"long-rollout-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        original=json.loads((source/"config.json").read_text())
        assert old["step"]==5000 and old["config_sha256"]==sha(source/"config.json")
        assert json.loads((source/"audit-005000.json").read_text())["passed"]
        for host,arm in ARMS.items():
            run=root/"training"/f"{arm}-seed{seed}"
            run.mkdir(parents=True,exist_ok=False)
            cfg=copy.deepcopy(original)
            cfg.update(arm=arm,host=host,max_steps=6000,phase_start=5000,data=str(root/"packed"),
                       stages=[5000,5002,6000],parent=str(source),
                       supervision="Future RGB/own codes only; no action target, state, reward, teacher or expert path")
            cfg["model"].update(clean_calibration=arm=="clean",flow_force_t_one=arm=="highnoise")
            save(run/"config.json",cfg)
            torch.manual_seed(seed)
            model=build(cfg["model"],root/"code")
            model.load_state_dict(old["model"],strict=True)
            assert state_hash(model.state_dict())==old["nn_sha256"]
            opt,scheduler=make_optimizer(model,6000,5000)
            assert associations(model,opt)==old["parameter_associations"]
            opt.load_state_dict(old["optimizer"])
            # Preserve all Adam moments/RNG/cursor. The paired new phase has a
            # common explicitly restarted 1000-update warmup/cosine schedule.
            fresh=[base*fn(5000) for base,fn in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
            sched=copy.deepcopy(old["scheduler"])
            sched["_last_lr"]=fresh
            scheduler.load_state_dict(sched)
            for group,lr in zip(opt.param_groups,fresh):group["lr"]=lr
            state=dict(old,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                       config_sha256=sha(run/"config.json"))
            torch.save(state,run/"last.ckpt")
            save(run/"ancestry.json",dict(passed=True,parent=str(source),parent_checkpoint_sha256=sha(source/"last.ckpt"),
                core_nn_exactly_equal=True,all_parameters_equal=True,optimizer_moments_preserved=True,
                full_rank_rng_and_sample_cursor_preserved=True,parent_step=5000,new_updates=0,
                known_original_unsaved_update_replay=1,schedule_change="Common 1000-update cosine restart, 10-update warmup",
                initial_learning_rates=fresh,independent_pretrainings=False))
            records.append(dict(run=run.name,host=host,parent=str(source),nn_sha256=old["nn_sha256"]))
            del state,opt,scheduler,model
        del old
    save(root/"INITIALIZED.json",dict(passed=True,records=records,new_sgd_updates=0,
          two_original_parents_crossed_with_three_objectives=True,all_candidates_sample_level=True))
    print(json.dumps(dict(passed=True,runs=len(records),source_validation_kitchens=kitchens)),flush=True)


def worker(root,host,seed):
    run=root/"training"/f"{ARMS[host]}-seed{seed}"
    gpu="0,1,2,3" if seed==1 else "4,5,6,7"
    one="0" if seed==1 else "4"
    data="/tmp/prescottsun-robocasa-compare-20261110/packed"
    try:
        for step in (5000,5002,6000):
            if step>5000:
                if step==6000:
                    deadline=time.time()+1200
                    while not (root/"VALIDATION_CASES_READY.json").exists() and time.time()<deadline:
                        time.sleep(5)
                    assert json.loads((root/"VALIDATION_CASES_READY.json").read_text())["passed"]
                argv=[TRAIN_PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=4",
                      "-m","experiments.robocasa_compare.train","--root",str(root),"--run-directory",str(run),
                      "--stop-step",str(step),"--data-root",data,"--resume"]
                command(argv,run/f"train-to-{step}.log",environment(root,gpu),root/"code",2*3600)
            command([TRAIN_PYTHON,"-m","experiments.robocasa_compare.audit","--root",str(root),"--run",str(run),"--step",str(step)],
                    run/f"audit-{step}.log",environment(root,one),root/"code",1800)
        save(run/"TRAINING_COMPLETE.json",dict(passed=True,step=6000,new_sgd_updates=1000,seed=seed))
        evaluate_seed(root,host,seed)
    except BaseException as error:
        save(run/"ERROR.json",dict(error=repr(error),host=host,seed=seed))
        raise


def evaluate_seed(root,host,seed):
    cohort=json.loads((root/"VALIDATION_CASES_READY.json").read_text())["selected"]
    base=0 if seed==1 else 4
    run=root/"training"/f"{ARMS[host]}-seed{seed}"

    def one(item):
        i,case=item
        jobid=f"{run.name}-{case['task']}-v{case['kitchen']}"
        out=root/"evaluation"/jobid
        argv=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.validation",
              "--root",str(root),"--mode","evaluate","--task",case["task"],"--kitchen",str(case["kitchen"]),
              "--seed","7401","--run",str(run),"--step","6000","--output",str(out)]
        command(argv,root/"logs"/(jobid+".log"),environment(root,str(base+i),True),root/"code",3600)
        assert json.loads((out/"summary.json").read_text())["passed"]
    (root/"logs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,enumerate(cohort)))
    save(run/"VALIDATION_COMPLETE.json",dict(passed=True,episodes=len(cohort),source_validation_only=True))


def launch(root,host):
    assert (root/"INITIALIZED.json").exists() and not (root/"STOP").exists()
    owner=root/f"launch-{host}.json";assert not owner.exists()
    # The already-authorized frozen official-reference queue keeps running.
    # H200 co-tenancy is allowed only with identified owned reference jobs and
    # enough memory for the previously measured <80GiB SPT training peak.
    mem=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
    assert all(int(x)>=90*1024 for x in mem.splitlines()),mem
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True)
    parent=Path(json.loads((root/"PARENT.json").read_text())["path"])
    for pid in apps.splitlines():
        cmd=(Path("/proc")/pid.strip()/"cmdline").read_bytes().replace(bytes([0]),b" ").decode()
        assert str(parent) in cmd and ("--reference" in cmd or "diagnose-frozen" in cmd),cmd
    records=[]
    for seed in (1,2):
        argv=["/usr/bin/timeout","--signal=TERM","--kill-after=60","21600",TRAIN_PYTHON,"-u","-m",
              "experiments.robocasa_refine.campaign","--root",str(root),"--host",host,"--seed",str(seed),"--mode","worker"]
        with (root/f"worker-{host}-seed{seed}.log").open("x") as f:
            p=subprocess.Popen(argv,cwd=root/"code",env=environment(root,""),stdin=subprocess.DEVNULL,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(pid=p.pid,seed=seed,command=argv))
    save(owner,dict(host=host,records=records,created_unix=time.time(),max_hours=6,
                   existing_reference_jobs_preserved=True,gpu_free_before_mb=[int(x) for x in mem.splitlines()]))
    print(json.dumps(dict(host=host,pids=[r["pid"] for r in records])),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--host",choices=ARMS)
    p.add_argument("--seed",type=int,choices=[1,2]);p.add_argument("--mode",choices=["initialize","launch","worker"],required=True)
    a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent)
    elif a.mode=="launch":launch(a.root,a.host)
    else:worker(a.root,a.host,a.seed)
