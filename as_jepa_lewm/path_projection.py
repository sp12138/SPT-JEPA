"""Nonlearned control-as-inference over complete observation paths.

An actuator can affect a later observation without changing the next image.
Myopic image inversion misses such preparatory controls.This planner scores
the consequences of complete,self-generated command words against the
goal-conditioned observation generator.No demonstrated action is referenced.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def path_cost(model,current,goal,past,valid,controls,block_steps=5):
    n=len(controls);base=current.expand(n,-1,-1,-1);goal=goal.expand(n,-1,-1,-1)
    current=base;previous=past.expand(n,-1);valid=valid.expand(n);zero=current.new_zeros(n);one=zero+1
    mapping=model.grid.permute(0,3,1,2).expand(n,-1,-1,-1).clone();cost=current.new_zeros(n)
    for index in range(controls.shape[1]*block_steps):
        command=controls[:,index//block_steps]
        drift,effect,_,_=model.fields(current,torch.zeros_like(goal),zero,previous,valid)
        _,_,desired,_=model.fields(current,goal,one,previous,valid)
        field=drift+torch.einsum('na,nachw->nchw',torch.cat([command,command.abs()],-1),effect)
        proposed=model.warp(mapping,field);reference=model.warp(mapping,desired)
        future=F.grid_sample(base,proposed.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=False)
        target=F.grid_sample(base,reference.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=False)
        # One RGB quantization unit and one normalized own-command unit.
        # A fixed observation likelihood cannot excuse contact error by
        # inflating a learned variance.There are no task-dependent weights.
        cost+=.5*((future-target)*255).square().flatten(1).sum(-1)+.5*(command-previous).square().sum(-1)
        current=future;mapping=proposed;previous=command;valid=one
    return cost


@torch.no_grad()
def path_projection(model,current,goal,valid,past,generator,warm=None,particles=128,iterations=8,blocks=3,block_steps=5):
    assert len(current)==1 and particles>=4 and iterations>=1
    mean=past[:,None].expand(1,blocks,4).clone()if warm is None else warm.reshape(1,blocks,4).clone()
    std=torch.ones_like(mean);best=None;best_cost=float('inf');history=[]
    for _ in range(iterations):
        controls=(mean+std*torch.randn(particles,blocks,4,device=current.device,generator=generator)).clamp(-1,1)
        controls[0]=mean[0].clamp(-1,1)
        if best is not None:controls[1]=best
        cost=path_cost(model,current,goal,past,valid,controls,block_steps);assert torch.isfinite(cost).all()
        values,elite=cost.topk(min(10,particles),largest=False);selected=controls[elite]
        if float(values[0])<best_cost:best_cost=float(values[0]);best=selected[0].clone()
        mean=selected.mean(0,keepdim=True);std=selected.std(0,unbiased=False,keepdim=True)
        history.append(best_cost)
    assert best is not None and best.abs().max()<=1
    commands=best[:,None].expand(-1,block_steps,-1).reshape(blocks*block_steps,4).cpu().numpy()
    return list(commands),best,dict(best_cost=best_cost,cost_history=history,particles=particles,iterations=iterations,
        native_horizon=blocks*block_steps,no_action_references=True)
