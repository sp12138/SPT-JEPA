"""Common RGB-goal tasks in source-validation kitchens, never test kitchens."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from experiments.robocasa_compare import evaluate as base
from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import environment

ROOT=None


def name(task,kitchen,seed):
    return f"{task}-v{kitchen}-s{seed}"


def make_env(task,kitchen,seed):
    from robots.robocasa.object_push_kitchen import ObjectPushKitchen
    from robots.robocasa.robot_env import RoboCasaEnv
    from robots.robocasa.blocking_robot_env import ReachBlockingRoboCasaEnv,ObjectPushBlockingRoboCasaEnv
    ObjectPushKitchen.DEFAULT_OBJECT_NAME="flourbag001"
    # Upstream scene creation also draws global Python/NumPy randomness.
    # Model initialization and process startup must not change a physical case.
    scene_seed=20261110+5000+kitchen+100003*seed
    random.seed(scene_seed);np.random.seed(scene_seed%(2**32))
    pair=json.loads((ROOT/"validation-plan.json").read_text())["kitchens"][kitchen-1]
    assert all(11<=x<=60 for x in pair)
    inner=RoboCasaEnv(task_name="ObjectPushKitchen" if task=="push" else "CloseStandMixerHead",
        camera_names=["robot0_agentview_left","robot0_eye_in_hand"],camera_width=320,camera_height=240,
        seed=5000+kitchen,robot="PandaOmron",chunk_size=4,wm_embodiment="franka_omron_state_delta",
        reach_task=task=="reach",reach_task_seed=seed,reach_min_distance=.25,reach_max_distance=.35,
        reach_clear_radius=1.,kitchen_split="pretrain",layout_and_style_ids=[tuple(pair)])
    # Released golden base poses belong to other seed->kitchen bindings. The
    # new explicit validation scenes capture their own base on first reset.
    inner._reach_base_pose_cache.clear()
    inner._reach_targets_cache.clear()
    if task=="reach":
        return ReachBlockingRoboCasaEnv(inner,reach_task_seed=seed,reach_min_distance=.25,
             reach_max_distance=.35,reach_clear_radius=1.,kitchen_split="pretrain",pos_tol=.001,max_steps=200)
    return ObjectPushBlockingRoboCasaEnv(inner,object_push_object_name="flourbag001",object_push_task_seed=seed,
        object_push_min_distance=.10,object_push_max_distance=.15,object_push_forward_cone_deg=60.,
        object_push_clear_radius=1.,pos_tol=.001,max_steps=200)


def make_case(args):
    env=make_env(args.task,args.kitchen,args.seed)
    try:
        goal=base.reset(env,args.task,args.seed,"goal")
        initial=base.reset(env,args.task,args.seed,"init")
        proof=base.goal_proof(args.task,goal,initial)
        state=base.physics(env)
        pair=json.loads((args.root/"validation-plan.json").read_text())["kitchens"][args.kitchen-1]
        assert np.array_equal(state["kitchen"],pair)
        path=args.root/"cases"/(name(args.task,args.kitchen,args.seed)+".npz")
        path.parent.mkdir(exist_ok=True)
        assert not path.exists()
        np.savez_compressed(path,initial=np.stack([initial["img_0"],initial["img_1"]]),
            goal=np.stack([goal["img_0"],goal["img_1"]]),**state)
        row=dict(passed=True,path=str(path),sha256=sha(path),task=args.task,kitchen=args.kitchen,
                 actual_source_validation_kitchen=pair,seed=args.seed,goal_proof=proof,
                 target_test_kitchen=False,model_scores_not_used_in_selection=True)
        save(path.with_suffix(".json"),row);print(json.dumps(row),flush=True)
    finally:env.close()


def generate_host(root,host):
    hosts=("dev13","dev15","dev6")
    cohort=[dict(task=task,kitchen=k,seed=7401) for task in ("reach","push") for k in range(1,7)]
    cohort=[c for i,c in enumerate(cohort) if i%3==hosts.index(host)]
    (root/"case-logs").mkdir(exist_ok=True)
    def one(item):
        gpu,case=item
        cmd=[str(root/"environment/bin/python"),"-u","-m","experiments.robocasa_refine.validation",
             "--root",str(root),"--mode","case"]
        for k,v in case.items():cmd.extend(["--"+k,str(v)])
        row=dict(case=case)
        try:
            with (root/"case-logs"/(name(**case)+".log")).open("x") as log:
                subprocess.run(cmd,cwd=root/"code",env=environment(root,str(gpu),True),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            row["passed"]=True
        except Exception as error:row.update(passed=False,error=repr(error),method_negative=False)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,enumerate(cohort)))
    save(root/f"cases-{host}.json",dict(disposed=True,results=rows))
    if host=="dev13":
        deadline=time.time()+1500
        while not all((root/f"cases-{h}.json").exists() for h in hosts) and time.time()<deadline:time.sleep(5)
        selected=[];failures=[]
        for task in ("reach","push"):
            valid=[]
            for k in range(1,7):
                path=root/"cases"/(name(task,k,7401)+".json")
                if path.exists() and json.loads(path.read_text()).get("passed"):
                    valid.append(dict(task=task,kitchen=k,seed=7401))
                else:failures.append(dict(task=task,kitchen=k,method_negative=False))
            selected.extend(valid[:2])
        save(root/"VALIDATION_CASES_READY.json",dict(passed=len(selected)==4,selected=selected,
             construction_failures=failures,chosen_before_model_scores=True,source_validation_only=True))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["case","cases","evaluate"],required=True)
    p.add_argument("--host",choices=["dev13","dev15","dev6"])
    p.add_argument("--task",choices=["reach","push"]);p.add_argument("--kitchen",type=int)
    p.add_argument("--seed",type=int,default=7401);p.add_argument("--run",type=Path)
    p.add_argument("--step",type=int,default=6000);p.add_argument("--reference",choices=["50m","2b","2b450k","8b300k"])
    p.add_argument("--reference-mode",type=int,default=2);p.add_argument("--zero",action="store_true")
    p.add_argument("--technical",action="store_true");p.add_argument("--output",type=Path)
    p.add_argument("--reference-pose-source",choices=["native","released"],default="native")
    p.add_argument("--history-controls",choices=["issued","achieved"],default="achieved")
    p.add_argument("--online-correction",action="store_true")
    p.add_argument("--goal-metric",choices=["uniform","support"],default="uniform")
    a=p.parse_args();ROOT=a.root;base.install_public(ROOT)
    base.make_env=make_env;base.case_name=name
    try:
        if a.mode=="case":make_case(a)
        elif a.mode=="cases":generate_host(a.root,a.host)
        else:base.evaluate(a)
    except BaseException as error:
        if a.mode=="case":
            save(a.root/"cases"/(name(a.task,a.kitchen,a.seed)+"-error.json"),
                 dict(error=repr(error),method_negative=False))
        raise
