"""Nonlearned projection between future-observation distributions.

Flows are nonidentifiable auxiliary variables:image likelihood,not raw flow
distance,is the control objective.Known own past commands define the same
unit-scale prior used by the tangent approximation.No action model is fitted.
"""
import torch

from as_jepa_lewm.signed_density import signed_projection


def warp_derivatives(image,flow):
    """Border-padded bilinear warp and its exact piecewise displacement slopes."""
    n,c,h,w=image.shape
    xx=torch.arange(w,device=image.device)[None,None,:]+flow[:,0]
    yy=torch.arange(h,device=image.device)[None,:,None]+flow[:,1]
    x=xx.clamp(0,w-1);y=yy.clamp(0,h-1);x0=x.floor().long();y0=y.floor().long()
    x1=(x0+1).clamp_max(w-1);y1=(y0+1).clamp_max(h-1)
    fx=(x-x0)[:,None];fy=(y-y0)[:,None];flat=image.flatten(2)
    def read(ix,iy):return flat.gather(2,(iy*w+ix).flatten(1)[:,None].expand(-1,c,-1)).reshape(n,c,h,w)
    a,b,d,e=read(x0,y0),read(x1,y0),read(x0,y1),read(x1,y1)
    value=(a*(1-fx)+b*fx)*(1-fy)+(d*(1-fx)+e*fx)*fy
    gx=((b-a)*(1-fy)+(e-d)*fy)*((xx>0)&(xx<w-1))[:,None]
    gy=((d-a)*(1-fx)+(e-b)*fx)*((yy>0)&(yy<h-1))[:,None]
    return value,gx,gy


@torch.no_grad()
def observation_projection(model,current,goal,valid,past,max_iterations=12,return_report=False):
    n=len(current);role=current.new_zeros(n)
    drift,effect,_,variance=model.fields(current,torch.zeros_like(goal),role,past,valid)
    _,_,desired,goalvar=model.fields(current,goal,role+1,past,valid)
    target=model.warp(current,desired);noise=variance+goalvar
    initial,_=model.choose_control(current,goal,valid,past)
    def flow(action):return drift+torch.einsum('na,nachw->nchw',torch.cat([action,action.abs()],-1),effect)
    def cost(action):return((model.warp(current,flow(action))-target).square()/noise).flatten(1).sum(-1)+(action-past).square().sum(-1)
    action=initial;value=cost(action);prior_cost=cost(past)
    action=torch.where((prior_cost<value)[:,None],past,action);value=torch.minimum(value,prior_cost);start=value.clone();history=[]
    identity=torch.cat([torch.eye(4,device=current.device),torch.zeros(4,4,device=current.device)],0)[None].expand(n,-1,-1)
    for _ in range(max_iterations):
        predicted,gx,gy=warp_derivatives(current,flow(action))
        jacobian=gx[:,None]*effect[:,:,0,None]+gy[:,None]*effect[:,:,1,None]
        lifted=torch.cat([action,action.abs()],-1)
        rhs=target-predicted+torch.einsum('na,nachw->nchw',lifted,jacobian)
        basis=torch.cat([jacobian.flatten(2),identity],-1)[:,:,:,None,None]
        rhs=torch.cat([rhs.flatten(1),past],-1)[:,:,None,None]
        uncertainty=torch.cat([noise.flatten(1).expand(-1,target[0].numel()),torch.ones_like(past)],-1)[:,:,None,None]
        candidate=signed_projection(torch.zeros_like(rhs),basis,rhs,uncertainty,action)
        best=action.clone();best_cost=value.clone()
        # Common dyadic descent search;all candidates are the agent's own
        # optimization iterates,no recorded action options or references.
        for power in range(7):
            trial=action+(candidate-action)*(2.**(-power));trial_cost=cost(trial)
            improve=trial_cost<best_cost
            best=torch.where(improve[:,None],trial,best);best_cost=torch.minimum(best_cost,trial_cost)
        change=(best-action).abs().max();action,value=best,best_cost;history.append(value.cpu().tolist())
        if change<=torch.finfo(torch.float32).eps*16:break
    assert torch.all(value<=start+1e-5)and torch.isfinite(action).all()and action.abs().max()<=1
    if return_report:return action,flow(action),dict(initial_cost=cost(initial).cpu().tolist(),final_cost=value.cpu().tolist(),iterations=len(history),history=history)
    return action,flow(action)
