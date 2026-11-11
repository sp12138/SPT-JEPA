"""Paired tests of stationary codes and the information retained by distortion.

Targets are naturally observed future frames or their frozen V-JEPA features.
No action is a target. Features come from the already authorized visual encoder;
no public dynamics, protected diagnostic, task or proprio enters this trial.
"""
import argparse
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import time
import types
from concurrent.futures import ThreadPoolExecutor

import torch

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare import train as common
from experiments.robocasa_refine import encoder_ablation as base

MODULE="experiments.robocasa_refine.target_refinement"
HOST_ARM={"dev13":"plain","dev15":"features","dev6":"fixed"}
FRESH=("renderer.x_embedder.","renderer.anchor_embed.","renderer.final_layer.linear.",
       "renderer.feat_rope.")


def feature_distortion(self,batch,render,anchor,condition):
    from vforge.models.as_jepa.flow import flow_loss
    features=batch["features"].float()
    b,w,p,d=features.shape
    assert (p,d)==(196,1664)
    y=features[:,1:].flatten(0,1).transpose(1,2).reshape(b*(w-1),d,1,14,14)
    anchors=features[:,0].transpose(1,2).reshape(b,d,1,14,14).repeat_interleave(w-1,dim=0)
    return flow_loss(self.renderer,y,condition.flatten(0,1),anchors,self.training_shift)


def configure(model,arm,seed):
    assert arm in ("plain","fixed","features")
    if arm=="fixed":
        model.pool.requires_grad_(False)
        model.feature_adapter.requires_grad_(False)
    elif arm=="features":
        from vforge.models.as_jepa.renderer import AnchoredWorldDiT
        # Seed only a private CPU generator state; do not alter CUDA streams.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(423100+seed).get_state())
            model.renderer=AnchoredWorldDiT(replace(model.renderer.config,channels=1664,grid=14),
                state_dim=model.state_width,num_slots=model.state_slots)
        model.distortion=types.MethodType(feature_distortion,model)
    return model


def initialize(root,parent):
    torch.set_num_threads(2)
    root.mkdir(parents=True,exist_ok=False)
    for name in ("sources","environment","features","observations","cases","data.json","validation-plan.json"):
        (root/name).symlink_to(parent/name,target_is_directory=(parent/name).is_dir())
    (root/"code").symlink_to(Path(__file__).resolve().parents[2],target_is_directory=True)
    plan=json.loads((parent/"plan.json").read_text())
    plan.update(parent=str(parent),stages=[10002,15000],max_steps=15000,deadline=time.time()+6*3600,
        arms=HOST_ARM,extra_updates_per_model=5000,models=6,world_size=4,
        new_control_episodes=24,physical_cases=4,prior_head_updates=10000,
        no_automatic_long_training=True,no_new_data=True,no_future_or_optimal_action_target=True,
        mechanism="Stationary code vs unchanged control vs natural frozen-feature distortion",
        rate_normalization="Keep original12288 elements and beta=.25 in every arm; feature distortion is a mean, not an equal-unit jointELBO",
        feature_decoder="Same384x12 trunk;196 feature tokens vs256 packed pixel tokens; fresh1664-channel I/O and rebuilt14x14 RoPE",
        LR_restart="Common5000-step cosine/50 warmup; retain all compatible moments and four rank streams",
        training_data_and_model_unchanged=False)
    save(root/"plan.json",plan)
    rows=[]
    for arm in HOST_ARM.values():
      for seed in (1,2):
        source=parent/"training"/f"vjepa-spt-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==10000 and old["world_size"]==4 and old["samples"]==320000
        assert old["config_sha256"]==sha(source/"config.json")
        cfg=json.loads((source/"config.json").read_text())
        cfg.update(target_refinement=arm,parent=str(source),max_steps=15000,phase_start=10000,
            stages=[10002,15000],new_training_origin="Same10000-step parent; paired target/freeze interventions; common5000-step LR restart")
        run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True)
        save(run/"config.json",cfg)
        model=base.build(root,cfg)
        state=model.state_dict();fresh=[];retained=[]
        assert state.keys()==old["model"].keys()
        for key,value in state.items():
            if arm=="features" and key.startswith(FRESH):fresh.append(key)
            else:
                assert value.shape==old["model"][key].shape,key
                state[key]=old["model"][key];retained.append(key)
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(model.state_dict()[key],old["model"][key]) for key in retained)
        if arm!="features":assert common.state_hash(model.state_dict())==old["nn_sha256"]
        opt,scheduler=base.optimizer_for(model,cfg)
        associations=common.associations(model,opt)
        old_by_name={name:old["optimizer"]["state"][pid]
            for group,names in zip(old["optimizer"]["param_groups"],old["parameter_associations"])
            for pid,name in zip(group["params"],names) if pid in old["optimizer"]["state"]}
        moved=[];new_opt=opt.state_dict()
        for group,names in zip(new_opt["param_groups"],associations):
          for pid,name in zip(group["params"],names):
            if name in old_by_name and name not in fresh:
                new_opt["state"][pid]=copy.deepcopy(old_by_name[name]);moved.append(name)
        opt.load_state_dict(new_opt)
        now=opt.state_dict()
        for group,names in zip(now["param_groups"],associations):
          for pid,name in zip(group["params"],names):
            if name in moved:assert common.checkpoint_equal(now["state"][pid],old_by_name[name])
        rates=[rate*factor(10000) for rate,factor in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
        schedule=copy.deepcopy(old["scheduler"]);schedule["_last_lr"]=rates
        scheduler.load_state_dict(schedule)
        for group,rate in zip(opt.param_groups,rates):group["lr"]=rate
        full=dict(old,model=common.cpu(model.state_dict()),optimizer=common.cpu(opt.state_dict()),
            scheduler=scheduler.state_dict(),parameter_associations=associations,
            config_sha256=sha(run/"config.json"),nn_sha256=common.state_hash(model.state_dict()))
        assert common.checkpoint_equal(full["rank_rng"],old["rank_rng"])
        common.save_full_state(run,full)
        row=dict(passed=True,run=run.name,parent=str(source),retained_tensors=len(retained),
            fresh_tensors=fresh,retained_moments=len(moved),frozen_parameters=[n for n,p in model.named_parameters() if not p.requires_grad],
            common_tensors_moments_rankRNG_cursor_exact=True,SGD_replayed=0)
        save(run/"ancestry.json",row);rows.append(row)
        print(json.dumps(dict(initialized=run.name,fresh=len(fresh),retained_moments=len(moved))),flush=True)
        del model,opt,scheduler,old,full,state,now,new_opt,old_by_name
    save(root/"INITIALIZED.json",dict(passed=True,rows=rows,new_SGD=0))


def worker(root,host):
    """Two paired4-GPU learners, then four simultaneous physical cases each."""
    arm=HOST_ARM[host]
    try:
        assert json.loads((root/"METHOD_PROOF_APPROVED.json").read_text())["passed"]
        def one(seed):
            run=root/"training"/f"{arm}-seed{seed}"
            logs=run/"execution-v1";logs.mkdir(exist_ok=False)
            gpus=",".join(str(x) for x in range((seed-1)*4,seed*4))
            for stop in (10002,15000):
                base.call(root,[base.TRAIN_PYTHON,"-m","torch.distributed.run","--standalone","--nnodes=1",
                    "--nproc-per-node=4","-m","--",base.MODULE,"--root",str(root),"--mode","train",
                    "--run",str(run),"--stop",str(stop),"--resume"],logs/f"train-{stop}.log",gpus)
                if stop==10002:
                    base.call(root,[str(root/"environment/bin/python"),"-u","-m",base.MODULE,"--root",str(root),
                        "--mode","evaluate","--run",str(run),"--step",str(stop),"--technical","--case","2"],
                        logs/"entry.log",gpus.split(",")[0],True)
                    save(run/"ENTRY_APPROVED.json",dict(passed=True,step=stop,actual_H5_Push=True))
                else:
                    base.call(root,[base.TRAIN_PYTHON,"-u","-m",base.MODULE,"--root",str(root),"--mode","diagnostic",
                        "--run",str(run),"--step",str(stop)],logs/"diagnostic.log",gpus.split(",")[0])
                    def control(case):
                        base.call(root,[str(root/"environment/bin/python"),"-u","-m",base.MODULE,"--root",str(root),
                            "--mode","evaluate","--run",str(run),"--step",str(stop),"--case",str(case)],
                            logs/f"evaluate-{case}.log",gpus.split(",")[case],True)
                    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(control,range(4)))
            save(run/"DONE.json",dict(passed=True,step=15000,new_updates=5000))
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,(1,2)))
        save(root/f"DONE-{host}.json",dict(passed=True,host=host,arm=arm))
    except BaseException as error:
        save(root/f"ERROR-{host}.json",dict(error=repr(error),method_negative=False));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("initialize","worker"),required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--host",choices=HOST_ARM)
    a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent)
    else:worker(a.root,a.host)
