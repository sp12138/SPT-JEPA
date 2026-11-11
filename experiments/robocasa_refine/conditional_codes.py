"""Conditional scene changes and a stationary forecasting counterfactual.

The delta posterior pools E(x)-E(c), where c is the first actual observation in
the context window. Decoding and the spatial prior receive the same observed c.
At control time the cached visual goal is re-encoded in each actual-history
chart; it never supplies context to dynamics. Current executed actions are
inputs, naturally observed future RGB codes are targets, and no action is a label.

The forecast-only control keeps the original global readout and decoder exactly
fixed. It trains only the dynamics/action/context modules on the same Gaussian
innovation residual, without distortion gradients into dynamics. This differs
from freezing only the readout while continuing joint generator training.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from experiments.robocasa_compare.acquire import save,sha
from experiments.robocasa_compare import train as common
from experiments.robocasa_refine import encoder_ablation as base

HOST_ARM={"dev13":"global","dev15":"delta","dev6":"delta-noise","dev6-2c":"forecast"}


def _forecast_loss(self,batch,step=0,ramp_steps=0):
    try:
        pixels,_,_=self.prepare(batch)
        info=self.encode(dict(pixels=pixels,action=batch["action"]))
        z=info["emb"].float()
        means=self.mean_trajectory(z[:,0],info["act_emb"][:,:-1]).float()
        from experiments.robocasa_refine.analytic_model import innovation_error
        error=innovation_error(z,means,self.rho)
        loss=error.square().mean()
        return loss,dict(loss=loss,innovation_mse=loss,
            clean_rollout_mse=(z[:,1:]-means[:,1:]).square().mean(),
            feature_std=z.std((0,1)).mean())
    finally:self.cached=None;self.context_tokens=None


def forecast_only(model):
    for module in (model.pool,model.feature_adapter,model.renderer):module.requires_grad_(False)
    model.loss=types.MethodType(_forecast_loss,model)
    return model


class ConditionalAdapter(base.Adapter):
    @torch.no_grad()
    def _posterior(self,features):
        """features [view,time,patch,width]; fixed real chart, never goal context."""
        v,t,p,d=features.shape
        delta=features.float()-self.anchor[:,None].float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pooled=self.model.pool(self.model.feature_adapter(delta.flatten(0,1))).reshape(v,t,self.model.state_dim)
        return pooled

    @torch.no_grad()
    def initialize_goal(self,goal,initial):
        self.goal_dense=self.dense([goal]).detach()
        self.history([initial],[])
        return self.goal_for_state()

    @torch.no_grad()
    def history(self,observations,commands,realized_controls=None):
        from experiments.robocasa_compare.evaluate import RollState,control14
        features=self.dense(observations).reshape(len(observations),2,196,1664).transpose(0,1)
        self.anchor=features[:,0].detach()
        codes=self._posterior(features)
        self.origin=codes[:,:1].detach()
        codes=(codes-self.origin).transpose(0,1).reshape(1,len(observations),2,self.model.state_slots,1,self.model.state_width).float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            self.model.context_tokens=self.model.feature_adapter(self.anchor.float())
        action=torch.tensor(np.asarray(commands,np.float32).reshape(1,-1,7),device="cuda")
        return RollState(codes,(control14(action)-self.mean)/self.std)

    @torch.no_grad()
    def goal_for_state(self):
        codes=self._posterior(self.goal_dense[:,None])-self.origin
        return codes.transpose(0,1).reshape(1,1,2,self.model.state_slots,1,self.model.state_width).float()

    def encode(self,observations):
        raise RuntimeError("Conditional codes require actual history and goal_for_state, not an implicit global chart")


def initialize(root,parent,loops):
    torch.set_num_threads(2);root.mkdir(parents=True,exist_ok=False)
    for name in ("sources","environment","cases","validation-plan.json"):
        (root/name).symlink_to(parent/name,target_is_directory=(parent/name).is_dir())
    (root/"code").symlink_to(Path(__file__).resolve().parents[2],target_is_directory=True)
    oldcard=json.loads((parent/"data.json").read_text());newcard=json.loads((loops/"data.json").read_text())
    assert oldcard["mean"]==newcard["mean"] and oldcard["std"]==newcard["std"]
    records=oldcard["records"]+newcard["records"]
    assert len(records)==1216 and len({r["id"] for r in records})==1216
    assert len(oldcard["records"])==1024 and len(newcard["records"])==192
    assert all(r["split"]=="train" for r in newcard["records"])
    train={tuple(r["kitchen"]) for r in records if r["split"]=="train"}
    val={tuple(r["kitchen"]) for r in records if r["split"]=="validation"};assert not train&val
    save(root/"data.json",dict(records=records,mean=oldcard["mean"],std=oldcard["std"],
        parent_card_sha256=sha(parent/"data.json"),loop_card_sha256=sha(loops/"data.json"),
        all_valid_loops_included_without_effect_filter=True,protected_physics_excluded=True))
    (root/"features/vjepa").mkdir(parents=True);(root/"observations").mkdir()
    for card,source in ((oldcard,parent),(newcard,loops)):
        for record in card["records"]:
            for target,suffix in (("features/vjepa",".npy"),("observations",".npz")):
                path=source/target/(record["id"]+suffix);assert path.is_file()
                (root/target/path.name).symlink_to(path)
    plan=json.loads((parent/"plan.json").read_text())
    plan.update(parent=str(parent),loop_data=str(loops),arms=HOST_ARM,stages=[20002,30000],
        max_steps=30000,prior_head_updates=20000,extra_head_updates=10000,models=8,world_size=4,
        deadline=time.time()+6*3600,new_control_episodes=32,physical_cases=4,no_new_data=False,
        train_records=1120,validation_records=96,all_loops_used_as_factual_logs_not_expert_paths=True,
        mechanism="Conditional posterior and prior share actual scene; stationary forecasting control removes generator gradients",
        no_automatic_long_training=True,LR_restart="Common10000-step cosine/100warmup; compatible moments/rankRNG/cursor retained",
        no_future_or_optimal_action_target=True)
    save(root/"plan.json",plan);rows=[]
    for arm in HOST_ARM.values():
      for seed in (1,2):
        source=parent/"training"/f"spatial-seed{seed}"
        old=torch.load(source/"last.ckpt",map_location="cpu",weights_only=False)
        assert old["step"]==20000 and old["samples"]==640000 and old["world_size"]==4
        assert old["config_sha256"]==sha(source/"config.json")
        cfg=json.loads((source/"config.json").read_text())
        cfg.update(code_chart="delta" if arm.startswith("delta") else "global",forecast_only=arm=="forecast",
            flow_logit_shift=4. if arm=="delta-noise" else 0.,parent=str(source),phase_start=20000,
            max_steps=30000,stages=[20002,30000],new_training_origin="Same20000spatial parent and ordinary log mixture; common10000-step restart")
        run=root/"training"/f"{arm}-seed{seed}";run.mkdir(parents=True);save(run/"config.json",cfg)
        model=base.build(root,cfg);model.load_state_dict(old["model"],strict=True)
        assert common.state_hash(model.state_dict())==old["nn_sha256"]
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
            assert common.checkpoint_equal(opt.state_dict()["state"][pid],prior[name])
        rates=[rate*factor(20000) for rate,factor in zip(scheduler.base_lrs,scheduler.lr_lambdas)]
        schedule=copy.deepcopy(old["scheduler"]);schedule["_last_lr"]=rates;scheduler.load_state_dict(schedule)
        for group,rate in zip(opt.param_groups,rates):group["lr"]=rate
        full=dict(old,model=common.cpu(model.state_dict()),optimizer=common.cpu(opt.state_dict()),
            scheduler=scheduler.state_dict(),parameter_associations=names,config_sha256=sha(run/"config.json"))
        assert common.checkpoint_equal(full["rank_rng"],old["rank_rng"]);common.save_full_state(run,full)
        row=dict(run=run.name,passed=True,parent=str(source),retained_moments=len(moved),
            same_full_NN_as_parent=True,all_compatible_moments_rankRNG_and_cursor_exact=True,
            frozen=[n for n,p in model.named_parameters() if not p.requires_grad],new_SGD=0)
        save(run/"ancestry.json",row);rows.append(row);print(json.dumps(dict(run=run.name,passed=True,retained_moments=len(moved))),flush=True)
        del model,opt,scheduler,optim,prior,full,old
    save(root/"INITIALIZED.json",dict(passed=True,rows=rows,all_arm_initial_NN_exact=True,all_valid_logs_included=True))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=("initialize","worker"),required=True)
    p.add_argument("--parent",type=Path);p.add_argument("--loops",type=Path);p.add_argument("--host",choices=HOST_ARM)
    a=p.parse_args()
    if a.mode=="initialize":initialize(a.root,a.parent,a.loops)
    else:
        from experiments.robocasa_refine import target_refinement as phase
        phase.HOST_ARM=HOST_ARM;phase.worker(a.root,a.host)
