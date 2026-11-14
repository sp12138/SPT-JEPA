"""Detached action readout in the frozen forward model's response coordinates.

For the Euler control-affine model, B(z) is the exact response of the next
code to the current translation input. Project an observed code change onto
B(z), then fit a nine-parameter calibration using factual current actions.
Encoder, forward model and external visual features never receive this loss.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.robocasa_action_probe import probe
from experiments.robocasa_compare import train as common
from experiments.robocasa_compare.acquire import save, sha
from experiments.robocasa_stable_scaling.train import ControlAffineTransition


class FrozenResponse(nn.Module):
    def __init__(self, root):
        super().__init__()
        plan=json.loads((root/"plan.json").read_text())
        run=Path(plan["source"])/"training"/plan["source_models"]["spt64-linear"]
        cfg=json.loads((run/"config.json").read_text())
        assert cfg["variant"]=="spt-control-affine" and cfg["state_slots"]==4
        assert cfg.get("control_integrator","euler")=="euler" and not cfg.get("joint_views",False)
        folder=run/plan["comparison_checkpoint"]
        ready=json.loads((folder/"ready.json").read_text())
        assert ready["passed"] and sha(folder/"weights.pt")==ready["weights_sha256"]
        weights=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        self.field=ControlAffineTransition().field
        prefix="predictor.field."
        self.field.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
        assert "action_encoder.bias" not in weights
        # Windows normalizes actual XYZ inputs by the declared .3m bound.
        self.register_buffer("input_map",weights["action_encoder.weight"][:,7:10].clone()/.3)
        self.parent_nn_sha256=ready["nn_sha256"]
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def basis(self, current):
        z=current.float().reshape(-1,2,64)
        coefficients=self.field(z).reshape(-1,2,64,65)
        return (coefficients[...,1:]@self.input_map).reshape(-1,128,3)

    @torch.no_grad()
    def forward(self, current, following):
        basis=self.basis(current)
        delta=(following-current).float().reshape(-1,128,1)
        return (torch.linalg.pinv(basis)@delta).squeeze(-1)


def fit(root):
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    arm="spt64-linear";directory=root/"heads"/arm
    response=FrozenResponse(root).cuda();before=common.state_hash(response.state_dict())
    codes,actions,splits,indices=[],[],[],[]
    for shard in range(8):
        path=directory/"cache"/f"part-{shard}.npz";proof=json.loads(path.with_suffix(".json").read_text())
        assert proof["passed"] and proof["all_features_detached"] and proof["readout_frozen_unchanged"]
        assert sha(path)==proof["sha256"] and proof["parent"]["full_nn_sha256"]==response.parent_nn_sha256
        assert proof["readout_data_sha256"]==sha(root/"data.json")
        with np.load(path) as data:
            codes.extend(data["codes"]);actions.extend(data["actions"])
            splits.extend(data["splits"]);indices.extend(data["indices"])
    assert sorted(indices)==list(range(1216))
    order=np.argsort(indices)
    z=torch.from_numpy(np.stack(codes)[order]).float().cuda()
    y=torch.from_numpy(np.stack(actions)[order]).double().cuda().flatten(0,1)
    mask=np.asarray(splits)[order]=="train"
    assert sum(mask)==1120 and sum(~mask)==96
    card=json.loads((root/"data.json").read_text())["records"]
    assert not {tuple(r["kitchen"]) for r in card if r["split"]=="train"}&{
        tuple(r["kitchen"]) for r in card if r["split"]=="validation"}
    previous=z[:,:-1].reshape(-1,128);following=z[:,1:].reshape(-1,128)
    x=torch.cat([response(a,b) for a,b in zip(previous.split(512),following.split(512))]).double()
    assert torch.isfinite(x).all()
    select=torch.as_tensor(np.repeat(mask,16),device="cuda")
    xt,yt=x[select],y[select];xv,yv=x[~select],y[~select]
    gram=xt.T@xt/len(xt);rhs=xt.T@yt/len(xt);energy=float(gram.diag().mean())
    assert np.isfinite(energy) and energy>0,"Degenerate frozen action response"
    best=None;scores=[]
    for alpha in json.loads((root/"plan.json").read_text())["ridge_lambdas"]:
        weight=torch.linalg.solve(gram+alpha*energy*torch.eye(3,device="cuda",dtype=torch.float64),rhs).T
        mse=float((xv@weight.T-yv).square().mean())
        assert np.isfinite(mse) and torch.isfinite(weight).all()
        scores.append(dict(alpha=alpha,validation_RMSE=mse**.5))
        if best is None or mse<best[0]:best=(mse,weight.float().cpu(),alpha)
    leaf=xt[:32].float().detach().requires_grad_();head=best[1].cuda().clone().requires_grad_()
    loss=(leaf.detach()@head.T-yt[:32].float()).square().mean();loss.backward()
    assert leaf.grad is None and head.grad is not None and torch.isfinite(head.grad).all() and float(head.grad.norm())>0
    assert all(not p.requires_grad and p.grad is None for p in response.parameters())
    assert common.state_hash(response.state_dict())==before
    assert torch.equal(response(previous[:8],previous[:8]),torch.zeros(8,3,device="cuda"))
    torch.save(dict(weight=best[1],head_type="frozen_response_calibration",alpha=best[2]),directory/"head.pt")
    save(directory/"ready.json",dict(passed=True,head_sha256=sha(directory/"head.pt"),
        parent_nn_sha256=response.parent_nn_sha256,head_parameters=9,input_width=3,
        train_transitions=len(xt),validation_transitions=len(xv),validation_RMSE=best[0]**.5,
        validation_zero_RMSE=float(yv.square().mean().sqrt()),scores=scores,selected_alpha=best[2],
        frozen_forward_model_unchanged=True,actual_stopgradient_backward_pass=True,
        exact_zero_for_identical_codes=True,no_future_or_optimal_action_labels=True,
        head_target="same factual current issued XYZ a_t",target_case_feedback_used_for_selection=False))


class ResponsePolicy(probe.Policy):
    def __init__(self, root, arm):
        super().__init__(root,arm)
        self.uses_factual_action_labels=json.loads((root/"plan.json").read_text()).get("response_head_calibrated",True)
        self.response=FrozenResponse(root).cuda()
        assert self.response.parent_nn_sha256==self.readout.parent["full_nn_sha256"]
        self.response_hash=common.state_hash(self.response.state_dict());self.initial_weight=self.weight.clone()

    @torch.no_grad()
    def command(self, observation, goal):
        current=self.encode(observation)
        coordinates=self.response(current,goal)[0]
        value=self.weight@coordinates
        assert torch.isfinite(value).all()
        value*=torch.clamp(.3/value.norm().clamp_min(1e-12),max=1.)
        action=np.zeros(7,np.float32);action[:3]=value.cpu().numpy();return action

    def unchanged(self):
        return super().unchanged() and torch.equal(self.weight,self.initial_weight) and (
            common.state_hash(self.response.state_dict())==self.response_hash)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["fit","control"],required=True)
    p.add_argument("--shard",type=int);p.add_argument("--shards",type=int,default=8);a=p.parse_args()
    if a.mode=="fit":fit(a.root)
    else:
        torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
        policy=ResponsePolicy(a.root,"spt64-linear")
        for i,_ in enumerate(json.loads((a.root/"plan.json").read_text())["cases"]):
            if i%a.shards==a.shard:probe.evaluate(a.root,"spt64-linear",i,policy=policy)
