"""Give the dynamics prior the initial visual context already seen by decoding.

The pixel decoder conditions on x0 while compressed dynamics previously did not
see the same scene information. Omitting context C from an optimal log-loss prior
costs I(Z_future; C | compressed history, actions). That identity motivates this
test; it does not guarantee improvement for a finite neural model or planner.

The pooled control repeats the per-view mean at every position, keeping exactly
the same parameters, sequence length and initialization as spatial attention.
Only synchronized views of one physical example exchange information.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import types

import torch
from torch import nn

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare import train as common
from experiments.robocasa_refine import encoder_ablation as base

HOST_ARM={"dev13":"plain","dev15":"pooled","dev6":"spatial"}


class ContextBridge(nn.Module):
    def __init__(self,state_width,kind):
        super().__init__();assert kind in ("pooled","spatial")
        self.kind=kind
        self.query=nn.Linear(2*state_width,192)
        self.position=nn.Linear(3,192,bias=False)
        self.query_norm=nn.LayerNorm(192);self.context_norm=nn.LayerNorm(192)
        self.attention=nn.MultiheadAttention(192,3,dropout=0.,batch_first=True)
        self.output=nn.Linear(192,state_width)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,state,action,context):
        assert context is not None and state.shape==action.shape
        n,t,d=state.shape;assert n%2==0 and context.shape[0]%2==0
        b=n//2;p=context.shape[1];side=math.isqrt(p);assert side*side==p
        c=context.reshape(-1,2,p,192)
        assert c.shape[0] in (1,b),"Context may broadcast candidates, never independent worlds"
        if self.kind=="pooled":c=c.mean(2,keepdim=True).expand(-1,-1,p,-1)
        axis=torch.linspace(-1,1,side,device=context.device,dtype=context.dtype)
        yy,xx=torch.meshgrid(axis,axis,indexing="ij")
        xy=torch.stack((xx,yy),-1).reshape(1,p,2).expand(2,-1,-1)
        view=context.new_tensor([-1,1])[:,None,None].expand(-1,p,1)
        position=self.position(torch.cat((xy,view),-1)).reshape(1,2,p,192)
        c=self.context_norm(c+position).reshape(-1,2*p,192).expand(b,-1,-1)
        q=self.query_norm(self.query(torch.cat((state,action),-1))).reshape(b,2*t,192)
        return self.output(self.attention(q,c,c,need_weights=False)[0]).reshape(n,t,d)


def _prepare(self,batch):
    result=self._context_prepare(batch)
    # The sole observed scene argument is the first frame; future RGB is a target.
    self.context_tokens=self.feature_adapter(batch["features"][:,0].float())
    return result


def _predict(self,state,action):
    return self._context_predict(state,action)+self.context_bridge(state,action,self.context_tokens)


def _loss(self,*args,**kwargs):
    try:return self._context_loss(*args,**kwargs)
    finally:self.context_tokens=None


def attach(model,kind,seed):
    assert not hasattr(model,"context_bridge")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(112007+seed).get_state())
        model.context_bridge=ContextBridge(model.state_dim,kind)
    model.context_tokens=None
    model._context_prepare=model.prepare;model.prepare=types.MethodType(_prepare,model)
    model._context_predict=model.predict;model.predict=types.MethodType(_predict,model)
    model._context_loss=model.loss;model.loss=types.MethodType(_loss,model)
    return model


def initialize(root,parent):
    torch.set_num_threads(2);root.mkdir(parents=True,exist_ok=False)
    for name in ("sources","environment","features","observations","cases","data.json","validation-plan.json"):
        (root/name).symlink_to(parent/name,target_is_directory=(parent/name).is_dir())
    (root/"code").symlink_to(Path(__file__).resolve().parents[2],target_is_directory=True)
    plan=json.loads((parent/"plan.json").read_text())
    plan.update(parent=str(parent),arms=HOST_ARM,stages=[15002,20000],max_steps=20000,
        prior_head_updates=15000,extra_head_updates=5000,deadline=time.time()+4*3600,
        new_control_episodes=24,models=6,world_size=4,no_new_data=True,
        mechanism="The controlled prior sees initial RGB geometry already supplied to its anchored decoder",
        pooled_control="Same parameters, view/position coordinates and392 attention keys; repeat per-view mean content",
        no_automatic_long_training=True,LR_restart="Common5000-step cosine/50 warmup; compatible moments/rankRNG/cursor retained",
        no_future_or_optimal_action_target=True)
    save(root/"plan.json",plan);rows=[];initials={}
    for arm in HOST_ARM.values():
      for seed in (1,2):
        source=parent/"training"/f"plain-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==15000 and old["samples"]==480000 and old["world_size"]==4
        assert old["config_sha256"]==sha(source/"config.json")
        cfg=json.loads((source/"config.json").read_text())
        cfg.update(context_memory=None if arm=="plain" else arm,parent=str(source),phase_start=15000,
            max_steps=20000,stages=[15002,20000],new_training_origin="Same15000 plain parent plus zero-output initial-context prior; common5000-step restart")
        run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True);save(run/"config.json",cfg)
        model=base.build(root,cfg);state=model.state_dict();fresh=[]
        assert old["model"].keys()<=state.keys()
        for key in state:
            if key in old["model"]:
                assert state[key].shape==old["model"][key].shape;state[key]=old["model"][key]
            else:assert key.startswith("context_bridge.");fresh.append(key)
        model.load_state_dict(state,strict=True)
        assert all(torch.equal(v,model.state_dict()[key]) for key,v in old["model"].items())
        nn_hash=common.state_hash(model.state_dict())
        if arm=="plain":assert nn_hash==old["nn_sha256"]
        else:
            if seed in initials:assert initials[seed]==nn_hash
            else:initials[seed]=nn_hash
        opt,scheduler=base.optimizer_for(model,cfg);names=common.associations(model,opt)
        prior={name:old["optimizer"]["state"][pid] for group,groupnames in zip(old["optimizer"]["param_groups"],old["parameter_associations"])
            for pid,name in zip(group["params"],groupnames) if pid in old["optimizer"]["state"]}
        optim=opt.state_dict();moved=[]
        for group,groupnames in zip(optim["param_groups"],names):
          for pid,name in zip(group["params"],groupnames):
            if name in prior:optim["state"][pid]=copy.deepcopy(prior[name]);moved.append(name)
        opt.load_state_dict(optim)
        for group,groupnames in zip(opt.state_dict()["param_groups"],names):
          for pid,name in zip(group["params"],groupnames):
            if name in moved:assert common.checkpoint_equal(opt.state_dict()["state"][pid],prior[name])
        rates=[rate*factor(15000) for rate,factor in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
        schedule=copy.deepcopy(old["scheduler"]);schedule["_last_lr"]=rates;scheduler.load_state_dict(schedule)
        for group,rate in zip(opt.param_groups,rates):group["lr"]=rate
        full=dict(old,model=common.cpu(model.state_dict()),optimizer=common.cpu(opt.state_dict()),
            scheduler=scheduler.state_dict(),parameter_associations=names,nn_sha256=nn_hash,
            config_sha256=sha(run/"config.json"))
        assert common.checkpoint_equal(full["rank_rng"],old["rank_rng"])
        common.save_full_state(run,full)
        row=dict(passed=True,run=run.name,parent=str(source),new_tensors=fresh,retained_moments=len(moved),
            common_NN_and_moments_rankRNG_cursor_exact=True,initial_hash=nn_hash,new_SGD=0)
        save(run/"ancestry.json",row);rows.append(row);print(json.dumps(row),flush=True)
        del model,state,old,full,opt,scheduler,optim,prior
    save(root/"INITIALIZED.json",dict(passed=True,rows=rows,pooled_spatial_initials_exact=True))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("initialize","worker"),required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--host",choices=HOST_ARM)
    a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent)
    else:
        from experiments.robocasa_refine import target_refinement as phase
        phase.HOST_ARM=HOST_ARM;phase.worker(a.root,a.host)
