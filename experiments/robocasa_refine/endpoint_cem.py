"""Gaussian CEM with separate endpoint and zero-sum path covariance.

The constant temporal mode controls net robot translation. Its orthogonal
complement describes excursions returning to the same endpoint. A diagonal
action covariance cannot reduce endpoint variance while retaining those paths.
The endpoint family fits two three-vectors of variances, independently of the
choice of basis in the zero-sum subspace. No goal pose or task identity is used.
"""
import ast
import hashlib
import inspect

import numpy as np
import torch


class Proposal:
    def __init__(self,horizon,initial_std,kind):
        assert kind in ("gaussian","endpoint")
        self.kind="gaussian" if horizon==1 else kind
        self.horizon=horizon
        self.endpoint=initial_std[0].clone()
        self.path=initial_std[0].clone()

    def sample(self,mean,std,samples,maxnorm):
        noise=torch.from_numpy(np.random.normal(size=(samples,self.horizon,3))).float()
        if self.kind=="gaussian":
            delta=noise*std[None,:,:3]
        elif torch.equal(self.endpoint,self.path):
            # Exact same first-generation samples as Gaussian diagonal CEM.
            delta=noise*self.path
        else:
            delta=noise*self.path+noise.mean(1,keepdim=True)*(self.endpoint-self.path)
        translation=mean[None,:,:3]+delta
        norms=translation.norm(dim=-1,keepdim=True)
        translation=torch.where(norms<maxnorm,translation,translation/norms*maxnorm)
        actions=torch.zeros(samples,self.horizon,7)
        actions[:,:,:3]=translation
        return actions

    def refit(self,selected,momentum,reward_mask,std):
        if self.kind=="gaussian" or not bool(reward_mask.any()):return std
        # A single terminal goal makes all preceding actions eligible.
        assert bool(reward_mask.all())
        xyz=selected[:,:,:3]
        centered=xyz-xyz.mean(0,keepdim=True)
        endpoint=centered.sum(1)/(self.horizon**.5)
        path=centered-centered.mean(1,keepdim=True)
        endpoint_std=(endpoint.square().sum(0)/(len(xyz)-1)).sqrt()
        path_std=(path.square().sum((0,1))/((len(xyz)-1)*(self.horizon-1))).sqrt()
        self.endpoint=momentum*self.endpoint+(1-momentum)*endpoint_std
        self.path=momentum*self.path+(1-momentum)*path_std
        result=std.clone()
        result[:,:3]=((1-1/self.horizon)*self.path.square()+self.endpoint.square()/self.horizon).sqrt()
        return result


def make_cem(public_cem,kind):
    """Patch only sampling/covariance in the pinned public CEM's AST.

    Keep rollout, objective, global gathering, elites, mean update, budgets and
    cache lifetime unchanged. Explicit guards reject incompatible upstream code.
    """
    assert kind in ("gaussian","endpoint")
    source=inspect.getsource(public_cem)
    tree=ast.parse(source);function=tree.body[0]
    assert isinstance(function,ast.FunctionDef) and function.name=="cem"
    horizon_loops=[n for n in ast.walk(function) if isinstance(n,ast.For)
                   and isinstance(n.target,ast.Name) and n.target.id=="h"]
    assert len(horizon_loops)==1
    horizon_loop=horizon_loops[0]
    start=[i for i,n in enumerate(horizon_loop.body) if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name) and t.id=="r" for t in n.targets)]
    stop=[i for i,n in enumerate(horizon_loop.body) if isinstance(n,ast.For)
          and isinstance(n.target,ast.Name) and n.target.id=="k"]
    assert len(start)==len(stop)==1 and start[0]<stop[0]
    removed=horizon_loop.body[start[0]:stop[0]]
    assert any("action_stats.append" in ast.unparse(n) for n in removed)
    horizon_loop.body[start[0]:stop[0]]=ast.parse(
        "new_action=proposal_actions[:,h:h+1]\naction_stats.append(new_action.cpu())").body
    containers=[n for n in ast.walk(function) if isinstance(n,ast.For) and horizon_loop in n.body]
    assert len(containers)==1
    body=containers[0].body;index=body.index(horizon_loop)
    body[index:index]=ast.parse("proposal_actions=proposal.sample(mean,std,samples,maxnorm)").body
    repeats=[i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=="wm_state" for t in n.targets)
             and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
             and n.value.func.attr=="repeat"]
    assert len(repeats)==1
    function.body[repeats[0]:repeats[0]]=ast.parse(
        "assert translation_only and not use_gripper and action_repeat==1 and action_dim==7\n"
        "assert not isinstance(goal_features,dict) and actions_suffix is None\n"
        # Exact component standard deviations of the public sampler's
        # r~U[0,1], theta1/theta2~U[0,2pi] BEFORE norm projection.
        "std[:,:3]=std[:,:3]*torch.tensor([12.,12.,6.]).rsqrt()\n"
        "proposal=_proposal_factory(rollout,std[:,:3])").body
    updates=[]
    for node in ast.walk(function):
        if not isinstance(node,ast.For):continue
        for index,statement in enumerate(node.body):
            if (isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="std" for t in statement.targets)
                    and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Attribute)
                    and statement.value.func.attr=="where"):
                updates.append((node.body,index))
    assert len(updates)==1
    body,index=updates[0]
    body[index+1:index+1]=ast.parse("std=proposal.refit(selected_actions,momentum_std,reward_mask,std)").body
    ast.fix_missing_locations(tree)
    namespace=dict(public_cem.__globals__)
    namespace["_proposal_factory"]=lambda horizon,std:Proposal(horizon,std,kind)
    exec(compile(tree,f"<public-cem-{kind}>","exec"),namespace)
    result=namespace["cem"]
    result.proposal_kind=kind
    result.upstream_source_sha256=hashlib.sha256(source.encode()).hexdigest()
    return result
