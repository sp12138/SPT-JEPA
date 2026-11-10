"""Random command interventions in TRAIN kitchens, with two input semantics.

The two learners receive identical ordinary transition logs. No goal, reward,
success, absolute pose or policy label is returned to either learner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare.campaign import environment,command
from experiments.robocasa_compare.evaluate import install_public,reset,native_training_pose

MODULE="experiments.robocasa_refine.intervene"
HOSTS=("dev13","dev15","dev6")


def initialize(root,parent,screen):
    assert not (root/"collection-plan.json").exists()
    source=json.loads((parent/"packed/dataset.json").read_text())
    train=sorted({tuple(r["kitchen"]) for r in source["records"] if r["split"]=="train" and r["eligible"]},
        key=lambda p:hashlib.sha256(json.dumps(p).encode()).hexdigest())[:128]
    excluded={tuple(r["kitchen"]) for r in source["records"] if r["split"]=="validation"}
    assert len(train)==128 and not set(train)&excluded
    assert all(11<=a<=60 and 11<=b<=60 for a,b in train)
    records=[dict(id=f"l{p[0]}-s{p[1]}-r{seed}",kitchen=p,seed=seed) for p in train for seed in (9401,9402)]
    plan=dict(parent=str(parent),screen=str(screen),records=records,steps=16,maxnorm=.3,
        training_kitchens=128,source_validation_kitchens_excluded=True,target_kitchens_excluded=True,
        collection_policy="Independent isotropic directions; uniform radius[0,.3]; no task/goal/reward dependence",
        initial_state="Public ObjectPush init reset only; no goal observation or expert path is recorded",
        issued_and_achieved_inputs_share_exact_same_RGB_transitions=True,
        intended_training_arms={"dev13":"achieved","dev15":"issued","dev6":"issued-highnoise"},
        maximum_new_sgd_per_arm_parent=1000,original_scratch_parent_updates=5000,
        no_automatic_long_training=True)
    save(root/"collection-plan.json",plan)
    print(json.dumps(dict(episodes=len(records),maximum_transitions=len(records)*16)))


def make_env(pair,seed):
    from robots.robocasa.object_push_kitchen import ObjectPushKitchen
    from robots.robocasa.robot_env import RoboCasaEnv
    from robots.robocasa.blocking_robot_env import ObjectPushBlockingRoboCasaEnv
    ObjectPushKitchen.DEFAULT_OBJECT_NAME="flour_bag"
    scene=100000+pair[0]*64+pair[1]
    random.seed(scene+seed);np.random.seed(scene+seed)
    inner=RoboCasaEnv(task_name="ObjectPushKitchen",camera_names=["robot0_agentview_left","robot0_eye_in_hand"],
        camera_width=640,camera_height=480,seed=scene,robot="PandaOmron",chunk_size=4,
        wm_embodiment="franka_omron_state_delta",reach_task=False,kitchen_split="pretrain",
        layout_and_style_ids=[tuple(pair)])
    inner._reach_base_pose_cache.clear();inner._reach_targets_cache.clear()
    return ObjectPushBlockingRoboCasaEnv(inner,object_push_object_name="flour_bag",object_push_task_seed=seed,
        object_push_min_distance=.10,object_push_max_distance=.15,object_push_forward_cone_deg=60.,
        object_push_clear_radius=1.,pos_tol=.001,max_steps=200)


def pixels(obs):
    import cv2
    cv2.setNumThreads(1)
    result=[]
    for key in ("img_0","img_1"):
        x=obs[key];h,w=x.shape[:2];left=(w-h)//2
        assert h==480 and w==640 and x.std()>1.
        result.append(cv2.resize(x[:,left:left+h],(224,224),interpolation=cv2.INTER_AREA).transpose(2,0,1))
    return np.stack(result)


def episode(root,item):
    install_public(root)
    env=make_env(item["kitchen"],item["seed"])
    try:
        obs=reset(env,"push",item["seed"],"init")
        actual=env._env._env
        assert [actual.layout_id,actual.style_id]==item["kitchen"]
        rng=np.random.default_rng(np.random.SeedSequence([20261110,*item["kitchen"],item["seed"]]))
        images=[pixels(obs)];poses=[native_training_pose(env)];commands=[]
        for _ in range(16):
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            action=np.zeros(7,np.float32);action[:3]=direction*rng.uniform(0.,.3)
            # Reward/success/goal fields are deliberately not recorded or read.
            obs,_,_,_=env.step(action)
            images.append(pixels(obs));poses.append(native_training_pose(env));commands.append(action)
        out=root/"transitions"/(item["id"]+".npz")
        out.parent.mkdir(exist_ok=True);assert not out.exists()
        np.savez_compressed(out,pixels=np.stack(images),commands=np.stack(commands),achieved=np.diff(np.stack(poses),axis=0))
        held=int(hashlib.sha256(b"intervention-validation:"+json.dumps(item["kitchen"]).encode()).hexdigest()[:8],16)%10==0
        row=dict(passed=True,**item,path=str(out),sha256=sha(out),frames=17,transitions=16,
            split="validation" if held else "train",
            actual_train_kitchen=item["kitchen"],input_fields=["pixels","commands","achieved"],
            absolute_pose_recorded=False,no_goal_reward_success_or_action_target=True)
        save(out.with_suffix(".json"),row)
    finally:env.close()


def collect(root,host):
    plan=json.loads((root/"collection-plan.json").read_text())
    items=[x for i,x in enumerate(plan["records"]) if i%3==HOSTS.index(host)]
    (root/"collection-logs").mkdir(exist_ok=True)
    # One sequential EGL worker/GPU; neural comparison jobs may coexist.
    def one(gpu):
        rows=[]
        for item in items[gpu::8]:
            if (root/"STOP").exists():break
            out=root/"transitions"/(item["id"]+".json")
            argv=[str(root/"environment/bin/python"),"-u","-m",MODULE,"--mode","episode","--root",str(root),"--item-json",json.dumps(item)]
            try:
                if not out.exists():
                    command(argv,root/"collection-logs"/(item["id"]+".log"),environment(root,str(gpu),True),root/"code",1800)
                assert sha(out.with_suffix(".npz"))==json.loads(out.read_text())["sha256"]
                rows.append(json.loads(out.read_text()))
            except Exception as error:
                result=dict(passed=False,**item,error=repr(error),method_negative=False)
                save(root/"collection-errors"/(item["id"]+".json"),result);rows.append(result)
        return rows
    with ThreadPoolExecutor(max_workers=8) as pool:rows=sum(pool.map(one,range(8)),[])
    save(root/f"collection-{host}.json",dict(disposed=True,rows=rows))


def finish(root):
    plan=json.loads((root/"collection-plan.json").read_text())
    rows=[];errors=[]
    for item in plan["records"]:
        p=root/"transitions"/(item["id"]+".json")
        if p.exists():
            row=json.loads(p.read_text());assert row["passed"] and sha(row["path"])==row["sha256"]
            # A fresh salt is required: every source kitchen was already chosen
            # from the parent TRAIN split, so reusing that hash gives no holdout.
            row["acquisition_split"]=row["split"]
            held=int(hashlib.sha256(b"intervention-validation:"+json.dumps(row["kitchen"]).encode()).hexdigest()[:8],16)%10==0
            row["split"]="validation" if held else "train"
            rows.append(row)
        else:errors.append(item)
    # Geometrically invalid public reset tasks are acquisition failures, never
    # negative learner examples. The same remaining corpus is used by all arms.
    assert len(rows)>=128 and len({tuple(r["kitchen"]) for r in rows})>=80,(len(rows),len(errors))
    original=json.loads((Path(plan["parent"])/"packed/dataset.json").read_text())
    card=dict(passed=True,records=rows,failed_acquisitions=errors,window=16,
        action_mean=original["action_mean"],action_std=original["action_std"],
        normalizer_scope="Preserved original TRAIN-only parent convention for both input-semantic arms",
        transitions=len(rows)*16,no_goal_reward_success_or_future_action_targets=True,
        source_sha256=sha(Path(plan["parent"])/"packed/dataset.json"))
    card["new_trajectory_split"]="Independent intervention-validation: SHA256(kitchen)%10; kitchen grouped, not parent's identical hash"
    save(root/"interventions/dataset.json",card);print(json.dumps(dict(episodes=len(rows),failed=len(errors),transitions=card["transitions"])))


class Interventions:
    def __init__(self,root,window,seed,split="train",semantics="issued"):
        self.root=Path(root);self.card=json.loads((self.root/"dataset.json").read_text())
        assert self.card["passed"] and window in (8,16) and semantics in ("issued","achieved","achieved-translation")
        self.window=window;self.seed=seed;self.semantics=semantics
        self.records=[r for r in self.card["records"] if r["split"]==split]
        assert self.records
        self.cache={}
        self.mean=np.asarray(self.card["action_mean"],np.float32);self.std=np.asarray(self.card["action_std"],np.float32)
        self.n=len(self.records)*2
    def __len__(self):return self.n
    def __getitem__(self,index):
        import torch
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(index)]))
        row=int(rng.integers(self.n));record=self.records[row//2];start=row%2
        if record["path"] not in self.cache:
            if len(self.cache)>=32:self.cache.pop(next(iter(self.cache)))
            with np.load(record["path"]) as archive:self.cache[record["path"]]={k:archive[k] for k in archive.files}
        archive=self.cache[record["path"]]
        rgb=archive["pixels"][start:start+self.window].transpose(1,0,2,3,4).copy()
        if self.semantics.startswith("achieved"):
            raw=archive["achieved"][start:start+self.window-1].copy()
            if self.semantics=="achieved-translation":
                translation=raw[:,7:10].copy();raw.fill(0);raw[:,7:10]=translation
        else:
            command=archive["commands"][start:start+self.window-1]
            raw=np.zeros((self.window-1,14),np.float32);raw[:,7:10]=command[:,:3];raw[:,13]=-.04*command[:,6]
        action=np.zeros((self.window,14),np.float32);action[:-1]=(raw-self.mean)/self.std
        return dict(pixels=torch.from_numpy(rgb),action=torch.from_numpy(action))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["initialize","collect","episode","finish"],required=True)
    p.add_argument("--host",choices=HOSTS);p.add_argument("--parent",type=Path);p.add_argument("--screen",type=Path)
    p.add_argument("--item-json");a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent,a.screen)
    elif a.mode=="collect":collect(a.root,a.host)
    elif a.mode=="episode":episode(a.root,json.loads(a.item_json))
    else:finish(a.root)
