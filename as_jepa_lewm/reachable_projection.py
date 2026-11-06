"""Observation geometry induced by our own predicted reachable futures.

No action labels/inverse learner are used.The covariance comes from the first
own CEM population and is fixed for that plan.Observation noise is one native
8-bit RGB unit,as in the existing future-observation likelihood.
"""
import torch


def prepare_metric(population):
    population=population.double()
    centered=population-population.mean(0,keepdim=True)
    _,singular,basis=torch.linalg.svd(centered,full_matrices=False)
    return basis,1+singular.square()/len(population)


def squared_metric(error,prepared):
    error=error.double();basis,variance=prepared;coordinates=error@basis.T
    orthogonal=error-coordinates@basis
    # Avoid subtracting two almost equal large quadratic forms(Woodbury).
    return orthogonal.square().sum(1)+(coordinates.square()/variance).sum(1)


@torch.no_grad()
def project(model,current,previous,goal,valid,past,generator,warm=None,particles=32,iterations=4,metric='rgb',control_prior='increment',blocks=3,block_steps=5):
    assert metric in['rgb','reachable']and control_prior in['increment','uniform']
    role=current.new_zeros(1);physical,_=model.prepare(current,previous,torch.zeros_like(goal),past,valid,role)
    desired,goal_context=model.prepare(current,previous,goal,torch.zeros_like(past),valid,role+1);target=current;targets=[]
    for _ in range(blocks*block_steps):
        desired,target=model.step(desired,target,goal_context,torch.zeros_like(past),role+1);targets.append(target)
    mean=past[:,None].expand(1,blocks,4).clone()if warm is None else warm.reshape(1,blocks,4).clone();std=torch.ones_like(mean)
    best=None;best_cost=float('inf');history=[];prepared=None
    target_path=torch.stack(targets,1).flatten(1)*255
    for _ in range(iterations):
        controls=(mean+std*torch.randn(particles,blocks,4,device=current.device,generator=generator)).clamp(-1,1);controls[0]=mean[0]
        if best is not None:controls[1]=best
        hidden=physical.expand(particles,-1,-1,-1).clone();image=current.expand(particles,-1,-1,-1);prior=past.expand(particles,-1)
        flag=role.expand(particles);context=goal_context.new_zeros(particles,96,16,16);score=current.new_zeros(particles);command_cost=score.clone();images=[]
        for index,target in enumerate(targets):
            command=controls[:,index//block_steps];hidden,image=model.step(hidden,image,context,command,flag)
            if metric=='rgb':
                score+=.5*((image-target)*255).square().flatten(1).sum(-1)+( .5*(command-prior).square().sum(-1)if control_prior=='increment'else 0)
            else:
                images.append(image)
                if control_prior=='increment':command_cost+=.5*(command-prior).square().sum(-1)
            prior=command
        if metric=='reachable':
            population=torch.stack(images,1).flatten(1)*255
            if prepared is None:prepared=prepare_metric(population)
            score=.5*squared_metric(population-target_path,prepared)+command_cost
        assert torch.isfinite(score).all()
        values,elite=score.topk(min(10,particles),largest=False);selected=controls[elite]
        if float(values[0])<best_cost:best_cost=float(values[0]);best=selected[0].clone()
        mean=selected.mean(0,keepdim=True);std=selected.std(0,unbiased=False,keepdim=True);history.append(best_cost)
    commands=best[:,None].expand(-1,block_steps,-1).reshape(blocks*block_steps,4).cpu().numpy()
    return list(commands),best,dict(best_cost=best_cost,cost_history=history,particles=particles,iterations=iterations,native_horizon=blocks*block_steps,
        metric=metric,control_prior=control_prior,covariance_population='FirstOWNCEMpopulationfixedwithinplan',observation_noise_RGB8=1,no_action_references=True)
