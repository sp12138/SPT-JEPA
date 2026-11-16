"""Detached action readout in the frozen forward model's response coordinates.

For Euler, B(z) is the exact response to the translation input. For RK4,
differentiate the full learned transition at zero input using forward AD.
Project an observed code change onto
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
        self.symmetric_endpoints=bool(plan.get("response_symmetric_endpoints",False))
        run=Path(plan["source"])/"training"/plan["source_models"]["spt64-linear"]
        cfg=json.loads((run/"config.json").read_text())
        assert cfg["variant"]=="spt-control-affine" and cfg["state_slots"]==4
        assert cfg.get('action_transition','control-affine')=='control-affine' and not cfg.get('nonlinear_action_residual',False), \
            "This readout requires an affine command map; a nonlinear map needs its actual response."
        self.integrator=cfg.get("control_integrator","euler")
        assert self.integrator in ("euler","rk4") and not cfg.get("joint_views",False)
        assert self.integrator=="euler" or not self.symmetric_endpoints
        folder=run/plan["comparison_checkpoint"]
        ready=json.loads((folder/"ready.json").read_text())
        assert ready["passed"] and sha(folder/"weights.pt")==ready["weights_sha256"]
        weights=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        self.field=ControlAffineTransition().field
        prefix="predictor.field."
        self.field.load_state_dict({k[len(prefix):]:v for k,v in weights.items() if k.startswith(prefix)},strict=True)
        if self.integrator=="rk4":
            self.rk4_transition=ControlAffineTransition("rk4")
            self.rk4_transition.field=self.field
        assert "action_encoder.bias" not in weights
        # Windows normalizes actual XYZ inputs by the declared .3m bound.
        self.register_buffer("input_map",weights["action_encoder.weight"][:,7:10].clone()/.3)
        self.parent_nn_sha256=ready["nn_sha256"]
        self.register_buffer("noise_cholesky",None)
        self.noise_model_sha256=None
        if "response_noise_covariance" in plan:
            spec=plan["response_noise_covariance"]
            path=Path(spec["path"])
            assert sha(path)==spec["sha256"]
            noise=torch.load(path,map_location="cpu",weights_only=True)
            assert noise["parent_nn_sha256"]==self.parent_nn_sha256
            assert noise["source_split"]=="TRAINonly"
            covariance=noise["covariance"].double()
            assert covariance.shape==(16,16) and torch.isfinite(covariance).all()
            assert torch.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-12)
            self.noise_cholesky=torch.linalg.cholesky(covariance)
            self.noise_model_sha256=spec["sha256"]
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def basis(self, current):
        if self.integrator=="rk4":
            current=current.float().reshape(-1,128)
            def transition(action,state):
                z=state.reshape(2,64)
                u=(action@self.input_map.T).expand_as(z)
                return self.rk4_transition(z,u).flatten()
            # Forward AD differentiates only the three hypothetical inputs.
            # Frozen NN parameters never receive the action-label loss.
            return torch.vmap(torch.func.jacfwd(transition,argnums=0))(
                current.new_zeros(len(current),3),current)
        z=current.float().reshape(-1,2,64)
        coefficients=self.field(z).reshape(-1,2,64,65)
        return (coefficients[...,1:]@self.input_map).reshape(-1,128,3)

    @torch.no_grad()
    def predict(self, current, action):
        return self.predict_input_grad(current, action)

    def predict_input_grad(self, current, action):
        """Differentiate hypothetical inputs while keeping the learned body frozen."""
        assert all(not p.requires_grad for p in self.parameters())
        z=current.detach().float().reshape(-1,2,64)
        if self.integrator=="rk4":
            u=action.float()@self.input_map.T
            return self.rk4_transition(z,u[:,None].expand_as(z)).reshape(-1,128)
        coefficients=self.field(z).reshape(-1,2,64,65)
        u=action.float()@self.input_map.T
        return (z+coefficients[...,0]+
            (coefficients[...,1:]@u[:,None,:,None]).squeeze(-1)).reshape(-1,128)

    @torch.no_grad()
    def forward(self, current, following):
        basis=self.basis(current)
        if self.symmetric_endpoints:
            # Trapezoidal approximation to the action tangent along a chord.
            # Both endpoints are naturally observed RGB encodings, never actions.
            basis=(basis+self.basis(following))*.5
        delta=(following-current).float().reshape(-1,128,1)
        if self.noise_cholesky is not None:
            # One shared channel likelihood, applied within each physical sample.
            # Its fixed mean is the learned forward model; no goal/error archive fit.
            basis=torch.linalg.solve_triangular(self.noise_cholesky,
                basis.double().reshape(-1,8,16,3),upper=False).reshape(-1,128,3)
            delta=torch.linalg.solve_triangular(self.noise_cholesky,
                delta.double().reshape(-1,8,16,1),upper=False).reshape(-1,128,1)
        return (torch.linalg.pinv(basis)@delta).squeeze(-1)


def fit_noise(root):
    """MLE of natural prediction errors with the forward mean held fixed."""
    from experiments.robocasa_stable_scaling.readout import load_pairs
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    output=root/"response-noise.pt";assert not output.exists()
    directory=root/"heads/spt64-linear"
    response=FrozenResponse(root).cuda();before=common.state_hash(response.state_dict())
    assert response.noise_cholesky is None,"Fit noise from the unweighted parent"
    for path in (directory/"cache").glob("part-*.json"):
        proof=json.loads(path.read_text())
        assert proof["parent"]["full_nn_sha256"]==response.parent_nn_sha256
        assert proof["readout_data_sha256"]==sha(root/"data.json")
    residuals=[]
    for current,following,actions in load_pairs(directory):
        predicted=torch.cat([response.predict(z,a) for z,a in
            zip(current.split(512),actions.split(512))])
        residuals.append((following-predicted).reshape(-1,16).double())
    train,validation=residuals
    covariance=train.T@train/len(train)
    eigenvalues=torch.linalg.eigvalsh(covariance)
    assert torch.isfinite(covariance).all() and bool((eigenvalues>0).all())
    logdet=torch.linalg.slogdet(covariance)[1];isotropic=covariance.diag().mean()
    def nll(residual):
        full=((torch.linalg.solve(covariance,residual.T).T*residual).sum(-1)+logdet).mean()/2
        iso=(residual.square().sum(-1)/isotropic+16*isotropic.log()).mean()/2
        return dict(full=float(full),isotropic=float(iso))
    assert common.state_hash(response.state_dict())==before
    assert all(not p.requires_grad and p.grad is None for p in response.parameters())
    torch.save(dict(covariance=covariance.cpu(),parent_nn_sha256=response.parent_nn_sha256,
        source_split="TRAINonly"),output)
    save(root/"response-noise.json",dict(passed=True,sha256=sha(output),parameters=136,
        source_split="TRAINonly",parent_nn_sha256=response.parent_nn_sha256,
        mean_fixed_to_frozen_forward_model=True,actions_are_forward_inputs_only=True,
        all_body_frozen_unchanged=True,eigenvalues=eigenvalues.cpu().tolist(),
        train_slot_residuals=len(train),validation_slot_residuals=len(validation),
        train_NLL=nll(train),validation_NLL=nll(validation),not_control_evidence=True))


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
        response_noise_covariance_sha256=response.noise_model_sha256,
        response_noise_covariance_parameters=136 if response.noise_model_sha256 else 0,
        response_symmetric_endpoints=response.symmetric_endpoints,
        response_transition_integrator=response.integrator,
        response_basis="full_transition_Jacobian_at_zero_input" if response.integrator=="rk4" else "exact_Euler_input_response",
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
        coordinates=self.response(current,goal)[0].to(self.weight.dtype)
        value=self.weight@coordinates
        assert torch.isfinite(value).all()
        value*=torch.clamp(.3/value.norm().clamp_min(1e-12),max=1.)
        action=np.zeros(7,np.float32);action[:3]=value.cpu().numpy();return action

    def unchanged(self):
        return super().unchanged() and torch.equal(self.weight,self.initial_weight) and (
            common.state_hash(self.response.state_dict())==self.response_hash)


def make_additive_head(root):
    """Invert the learned additive input map without fitting any action labels."""
    plan=json.loads((root/"plan.json").read_text());arm="spt64-linear"
    run=Path(plan["source"])/"training"/plan["source_models"][arm]
    cfg=json.loads((run/"config.json").read_text())
    assert cfg.get("pixel_additive_transport") and cfg["state_slots"]==4
    assert not cfg.get("joint_views",False)
    folder=run/plan["comparison_checkpoint"];parent=json.loads((folder/"ready.json").read_text())
    assert parent["passed"] and sha(folder/"weights.pt")==parent["weights_sha256"]
    weights=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
    assert "action_encoder.bias" not in weights
    response=weights["action_encoder.weight"][:,7:10].double()/.3
    # Both synchronized views use the same input operator during pretraining.
    response=torch.cat([response,response],0)
    assert response.shape==(128,3) and int(torch.linalg.matrix_rank(response))==3
    weight=torch.linalg.pinv(response)
    error=float((weight@response-torch.eye(3,dtype=torch.float64)).abs().max())
    assert error<1e-10 and torch.isfinite(weight).all()
    directory=root/"heads"/arm;directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/"head.pt").exists()
    calibrated=plan.get("additive_readout_calibrated",False)
    details={};label_count=0
    if calibrated:
        from experiments.robocasa_stable_scaling.readout import load_pairs
        torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
        for path in (directory/"cache").glob("part-*.json"):
            proof=json.loads(path.read_text())
            assert proof["parent"]["full_nn_sha256"]==parent["nn_sha256"]
            assert proof["readout_data_sha256"]==sha(root/"data.json")
        (z0,z1,a),(v0,v1,va)=load_pairs(directory)
        x=(z1-z0).double()@weight.T.cuda();vx=(v1-v0).double()@weight.T.cuda()
        y=a.double();vy=va.double();gram=x.T@x/len(x);rhs=x.T@y/len(x)
        energy=float(gram.diag().mean());assert energy>0 and np.isfinite(energy)
        best=None;scores=[]
        for alpha in plan["ridge_lambdas"]:
            calibration=torch.linalg.solve(gram+alpha*energy*torch.eye(3,device="cuda",dtype=torch.float64),rhs).T
            mse=float((vx@calibration.T-vy).square().mean());assert np.isfinite(mse)
            scores.append(dict(alpha=alpha,validation_RMSE=mse**.5))
            if best is None or mse<best[0]:best=(mse,calibration,alpha)
        leaf=x[:32].float().detach().requires_grad_();head=best[1].float().detach().requires_grad_()
        (leaf.detach()@head.T-y[:32].float()).square().mean().backward()
        assert leaf.grad is None and head.grad is not None and torch.isfinite(head.grad).all()
        assert float(head.grad.norm())>0
        weight=best[1].cpu()@weight;label_count=len(x)
        details=dict(train_transitions=len(x),validation_transitions=len(vx),
            validation_RMSE=best[0]**.5,selected_alpha=best[2],scores=scores,
            actual_stopgradient_backward_pass=True,head_target="same factual current XYZ input a_t")
    torch.save(dict(weight=weight.float(),head_type="intrinsic_additive_response",
        fitted_current_action_labels=label_count),directory/"head.pt")
    save(directory/"ready.json",dict(passed=True,head_sha256=sha(directory/"head.pt"),
        parent_nn_sha256=parent["nn_sha256"],input_width=128,fitted_head_parameters=9 if calibrated else 0,
        derived_head_values=weight.numel(),fitted_current_action_labels=label_count,new_body_updates=0,
        source="learned forward input operator;optional factual current-input calibration explicitly reported",
        physical_input_units="same XYZ current command divided by public .3m bound",
        uncalibrated_response_left_inverse_max_error=error,not_control_evidence=True,**details))


class AdditivePolicy(probe.Policy):
    def __init__(self,root,arm):
        super().__init__(root,arm)
        saved=torch.load(root/"heads"/arm/"head.pt",map_location="cpu",weights_only=True)
        assert saved["head_type"]=="intrinsic_additive_response"
        proof=json.loads((root/"heads"/arm/"ready.json").read_text())
        assert proof["parent_nn_sha256"]==self.readout.parent["full_nn_sha256"]
        self.uses_factual_action_labels=bool(saved["fitted_current_action_labels"])
        self.initial_weight=self.weight.clone()

    def unchanged(self):
        return super().unchanged() and torch.equal(self.weight,self.initial_weight)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--mode",choices=["fit","fit-noise","fit-additive","control","control-additive"],required=True)
    p.add_argument("--shard",type=int);p.add_argument("--shards",type=int,default=8);a=p.parse_args()
    if a.mode=="fit":fit(a.root)
    elif a.mode=="fit-noise":fit_noise(a.root)
    elif a.mode=="fit-additive":make_additive_head(a.root)
    else:
        torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
        policy=(AdditivePolicy if a.mode=="control-additive" else ResponsePolicy)(a.root,"spt64-linear")
        for i,_ in enumerate(json.loads((a.root/"plan.json").read_text())["cases"]):
            if i%a.shards==a.shard:probe.evaluate(a.root,"spt64-linear",i,policy=policy)
