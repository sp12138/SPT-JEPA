"""Rescore the CEM mean against its best final-generation sampled trajectory.

Search, sampling, elites and refitting remain the pinned public implementation.
Only the executed plan changes. Two additional model rollouts compare the final
mean and an actually evaluated sample; there is no extra simulator query, action
label or diagnostic-memory access. Better predicted cost is not physical proof.
"""
import ast
import inspect

import torch


def select_plan(world_model,initial,mean,best,goal,objective,step_func,device,
                view_weights,action_repeat,use_cache,audits,global_step):
    assert not isinstance(goal,dict) and not use_cache
    candidates=torch.stack((mean,best)).float()
    history=initial.repeat(2).to(device)
    with torch.no_grad():
        for h in range(len(mean)):
            for _ in range(action_repeat):
                future,history=step_func(candidates[:,h:h+1]/action_repeat,history,
                    no_grad=True,device=device,use_cache=False,batch=100)
        values=objective(goal.flatten(3,5),future.flatten(3,5).to(device)).cpu()
    assert values.shape[0]==2 and torch.isfinite(values).all()
    per_view=values.reshape(2,-1)
    weights=torch.tensor(view_weights or [1/per_view.shape[1]]*per_view.shape[1])
    scores=(per_view*weights).sum(1)
    selected=int(scores.argmax())
    audits.append(dict(global_step=global_step,mean_predicted_return=float(scores[0]),
        sampled_predicted_return=float(scores[1]),selected=selected,
        original_mean=mean.tolist(),best_sample=best.tolist(),
        extra_model_trajectories=2,no_extra_physics=True))
    return candidates[selected]


def make_cem(public_cem):
    """Expose the final samples without changing the preceding search bytecode."""
    tree=ast.parse(inspect.getsource(public_cem));function=tree.body[0]
    assert isinstance(function,ast.FunctionDef) and function.name=="cem"
    function.body[:0]=ast.parse("_execution_initial=wm_state").body
    returns=[n for n in function.body if isinstance(n,ast.Return)]
    assert len(returns)==1 and ast.unparse(returns[0].value)=="(rewards, mean_list, std_list)"
    index=function.body.index(returns[0])
    function.body[index:index]=ast.parse(
        "assert world_size==1 and not use_cache\n"
        "mean_list[-1]=_select_plan(world_model,_execution_initial,mean_list[-1],"
        "action_stats[top_sims.indices[0]],goal_features,objective,step_func,device,"
        "view_weights,action_repeat,use_cache,_execution_audits,global_step)").body
    ast.fix_missing_locations(tree)
    audits=[];namespace=dict(public_cem.__globals__,_select_plan=select_plan,_execution_audits=audits)
    exec(compile(tree,"<public-cem-rescored-execution>","exec"),namespace)
    result=namespace["cem"];result.execution_audits=audits
    return result
