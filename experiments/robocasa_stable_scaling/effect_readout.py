"""Frozen-code forward-effect probes; issued actions are inputs, never targets.

Fit the next code change with either constant or state-conditioned coefficients.
At control time invert only the action response, keeping autonomous drift out of
the goal metric. This is a separate forward head, not an encoder-only probe.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.robocasa_action_probe import probe
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_stable_scaling.readout import load_pairs


def state_features(current, conditional):
    z=current.reshape(-1,128)
    one=torch.ones(len(z),1,device=z.device,dtype=z.dtype)
    return torch.cat([one,z],-1) if conditional else one


def design(current, actions, conditional):
    state=state_features(current,conditional)
    inputs=torch.cat([torch.ones_like(actions[:,:1]),actions/.3],-1)
    return (inputs[:,:,None]*state[:,None,:]).flatten(1)


def basis(weight, current, conditional):
    state=state_features(current,conditional).to(weight.dtype)
    return torch.einsum("kdf,nf->nkd",weight[:,1:,:],state)/.3


def fit(root):
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    directory=root/"heads/spt64-linear";assert not (directory/"head.pt").exists()
    plan=json.loads((root/"plan.json").read_text())
    conditional=plan["effect_state_conditioned"]
    readout=probe.Readout(root,"spt64-linear").cuda().eval().requires_grad_(False)
    before=common.state_hash(readout.state_dict())
    for path in (directory/"cache").glob("part-*.json"):
        proof=json.loads(path.read_text())
        assert proof["parent"]["full_nn_sha256"]==readout.parent["full_nn_sha256"]
        assert proof["readout_data_sha256"]==sha(root/"data.json")
    (current,following,actions),(vcurrent,vfollowing,vactions)=load_pairs(directory)
    x=design(current.double(),actions.double(),conditional)
    y=(following-current).double()
    vx=design(vcurrent.double(),vactions.double(),conditional)
    vy=(vfollowing-vcurrent).double()
    gram=x.T@x/len(x);rhs=x.T@y/len(x);energy=float(gram.diag().mean())
    assert energy>0 and np.isfinite(energy)
    best=None;scores=[]
    for alpha in plan["ridge_lambdas"]:
        weight=torch.linalg.solve(gram+alpha*energy*torch.eye(len(gram),device="cuda",dtype=torch.float64),rhs).T
        mse=float((vx@weight.T-vy).square().mean())
        assert np.isfinite(mse) and torch.isfinite(weight).all()
        scores.append(dict(alpha=alpha,validation_next_code_MSE=mse))
        if best is None or mse<best[0]:best=(mse,weight.float(),alpha)
    width=129 if conditional else 1
    weight=best[1].reshape(128,4,width)
    leaf=current[:32].clone().requires_grad_();target=following[:32].clone().requires_grad_()
    head=weight.clone().requires_grad_()
    prediction=design(leaf.detach(),actions[:32],conditional)@head.flatten(1).T
    (prediction-(target.detach()-leaf.detach())).square().mean().backward()
    assert leaf.grad is None and target.grad is None and head.grad is not None
    assert torch.isfinite(head.grad).all() and float(head.grad.norm())>0
    assert common.state_hash(readout.state_dict())==before
    assert all(not p.requires_grad and p.grad is None for p in readout.parameters())
    # Check that the extracted matrix is the exact learned action response.
    a=actions[:8];z=current[:8];zero=torch.zeros_like(a)
    change=(design(z,a,conditional)-design(z,zero,conditional))@weight.flatten(1).T
    direct=(basis(weight,z,conditional)@a.unsqueeze(-1)).squeeze(-1)
    response_error=float((change-direct).abs().max());assert response_error<1e-5
    torch.save(dict(weight=weight.cpu(),head_type="factual_forward_effect",
        state_conditioned=conditional,alpha=best[2],target="naturally observed next code change",
        actions_are_inputs_not_targets=True),directory/"head.pt")
    save(directory/"ready.json",dict(passed=True,head_sha256=sha(directory/"head.pt"),
        parent_nn_sha256=readout.parent["full_nn_sha256"],head_parameters=weight.numel(),
        state_conditioned=conditional,selected_alpha=best[2],scores=scores,
        train_transitions=len(x),validation_transitions=len(vx),validation_next_code_MSE=best[0],
        validation_copy_MSE=float(vy.square().mean()),fitted_current_action_labels=0,
        actions_are_inputs_not_targets=True,all_body_frozen_unchanged=True,
        actual_stopgradient_backward_pass=True,response_decomposition_max_error=response_error,
        target_case_feedback_used_for_selection=False,no_future_or_optimal_action_labels=True))


class EffectPolicy(probe.Policy):
    def __init__(self, root, arm):
        super().__init__(root,arm)
        saved=torch.load(root/"heads"/arm/"head.pt",map_location="cpu",weights_only=True)
        assert saved["head_type"]=="factual_forward_effect"
        self.conditional=saved["state_conditioned"]
        assert self.weight.shape==(128,4,129 if self.conditional else 1)
        self.uses_factual_action_labels=False;self.initial_weight=self.weight.clone()

    @torch.no_grad()
    def command(self, observation, goal):
        current=self.encode(observation)
        response=basis(self.weight,current,self.conditional)[0]
        value=torch.linalg.pinv(response)@(goal-current)
        assert torch.isfinite(value).all()
        value*=torch.clamp(.3/value.norm().clamp_min(1e-12),max=1.)
        action=np.zeros(7,np.float32);action[:3]=value.cpu().numpy();return action

    def unchanged(self):
        return super().unchanged() and torch.equal(self.weight,self.initial_weight)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["fit","control"],required=True)
    p.add_argument("--shard",type=int);p.add_argument("--shards",type=int,default=8);a=p.parse_args()
    if a.mode=="fit":fit(a.root)
    else:
        torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
        policy=EffectPolicy(a.root,"spt64-linear")
        for i,_ in enumerate(json.loads((a.root/"plan.json").read_text())["cases"]):
            if i%a.shards==a.shard:probe.evaluate(a.root,"spt64-linear",i,policy=policy)
