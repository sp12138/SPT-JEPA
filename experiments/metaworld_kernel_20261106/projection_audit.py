"""Frozen test of tangent versus rendered-observation planning objectives."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


def run(a):
    from as_jepa_lewm.factory import build_model
    from as_jepa_lewm.observation_projection import observation_projection,warp_derivatives
    from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cfg=json.loads((a.run/'config.json').read_text());model=build_model(cfg).eval().requires_grad_(False)
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cpu',weights_only=True));before=model_sha(model)
    generator=torch.Generator().manual_seed(8301)
    image=torch.rand(2,3,128,128,generator=generator);flow=torch.randn(2,2,128,128,generator=generator)*3+.137
    flow.requires_grad_(True);weight=torch.randn(image.shape,generator=generator)
    value=model.warp(image,flow);gradient=torch.autograd.grad((value*weight).sum(),flow)[0]
    actual,gx,gy=warp_derivatives(image,flow.detach());expected=torch.stack([(gx*weight).sum(1),(gy*weight).sum(1)],1)
    output_error=float((value.detach()-actual).abs().max());gradient_error=float((gradient-expected).abs().max())
    # Pixel-coordinate and normalized-grid arithmetic have a small FP32
    # roundoff discrepancy away from exact power-of-two grid centers.
    assert output_error<2e-5 and gradient_error<1e-4,(output_error,gradient_error)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['seed']==42];rows=[]
    for case in cases:
        raw=np.load(case['path']);current=model.encoder(resize_tensor([raw['initial']]));goal=model.encoder(resize_tensor([raw['goal']]))
        past=current.new_zeros(1,4);valid=current.new_zeros(1)
        action,_,report=observation_projection(model,current,goal,valid,past,return_report=True)
        with torch.no_grad():
            role=current.new_zeros(1);drift,effect,_,var=model.fields(current,torch.zeros_like(goal),role,past,valid)
            _,_,desired,goalvar=model.fields(current,goal,role+1,past,valid);target=model.warp(current,desired);noise=var+goalvar
            original,_=model.choose_control(current,goal,valid,past)
        def objective(values):
            control=torch.as_tensor(values,dtype=torch.float32).reshape(1,4).requires_grad_(True)
            field=drift+torch.einsum('na,nachw->nchw',torch.cat([control,control.abs()],-1),effect)
            loss=((model.warp(current,field)-target).square()/noise).sum()+(control-past).square().sum()
            grad=torch.autograd.grad(loss,control)[0];return float(loss.detach()),grad.detach().numpy().astype(np.float64).flatten()
        references=[minimize(objective,start,method='L-BFGS-B',jac=True,bounds=[(-1,1)]*4,
            options=dict(maxiter=128,ftol=1e-10,gtol=1e-5))for start in[original.numpy().flatten(),action.numpy().flatten(),np.zeros(4)]]
        best=min(references,key=lambda r:r.fun)
        rows.append(dict(task=case['task'],seed=case['seed'],case_sha256=case['sha256'],tangent_action=original[0].tolist(),
            observation_action=action[0].tolist(),reference_action=best.x.tolist(),reference_cost=float(best.fun),**report))
    assert model_sha(model)==before
    result=dict(passed=True,model_updates=0,model_unchanged=True,step=a.step,warp_output_error=output_error,
        exact_displacement_gradient_error=gradient_error,rows=rows,
        scope='Nonlearned inference only.Future-observation predictions define objective;no demonstrated future action is used.No physics-success claim.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);run(p.parse_args())
