"""Separate observation-codec geometry from forward prediction on protected banks.

No scores, frames, states or controls from this audit may train a learner,
become a teacher, seed a planner, or count as policy success.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_endpoint_density_20261109.model import advance
from experiments.metaworld_contact_geometry_20261109.train import setup

ROOT=BASE/'runs/metaworld_contact_geometry_20261109/oracle-audit-v1'
BANK=BASE/'runs/metaworld_belief_state_20261109/contact-audit-v1'


def summary(scores,flags,distances):
    scores=np.asarray(scores);selected=int(scores.argmin());result=dict(selected=selected,
        selected_any_success=bool(flags[selected].any()),selected_final_success=bool(flags[selected,-1]),
        selected_object_distance_audit_only=float(distances[selected]),score=scores.tolist())
    for key,f in [('any',flags.any(1)),('final',flags[:,-1])]:
        auc=None
        if f.any() and (~f).any():auc=float(((scores[f,None]<scores[None,~f])+.5*(scores[f,None]==scores[None,~f])).mean())
        result['AUC_'+key]=auc
    return result


def codes(world,a,k,goal_a,goal_k):
    if world.codec_mode=='dense':
        return {'combined':(torch.nn.functional.normalize(a.float(),dim=1)-torch.nn.functional.normalize(goal_a.float(),dim=1)).square().sum(1).mean((1,2))},None
    xy,descriptor,_=world.points(a,k);gxy,gd,_=world.points(goal_a,goal_k)
    coordinate=(xy-gxy).square().mean((1,2));appearance=(descriptor-gd).square().sum(-1).mean(-1)
    return dict(combined=coordinate+appearance,coordinates=coordinate,appearance=appearance),xy


@torch.no_grad()
def run(prefix):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    path=BANK/f'prefix{prefix}.npz';assert sha(path)==read(path.with_suffix('.json'))['bank_sha256']
    data=np.load(path);flags=data['successes'];distance=np.linalg.norm(data['states60'][:,4:7]-data['target_audit_only'],axis=1)
    actual=torch.from_numpy(data['actual60']).permute(0,3,1,2).cuda().float()/255
    goal=torch.from_numpy(data['goal'].copy()).permute(2,0,1)[None].cuda().float()/255
    observed=torch.from_numpy(data['observations']).permute(0,3,1,2).cuda().float()/255
    history=torch.from_numpy(data['prefix_actions']).cuda();controls=torch.from_numpy(data['controls']).cuda()
    reports={}
    for seed in [1,2]:
        for mode in ['dense','sparse','transport']:
            cfg=read(BASE/f'runs/metaworld_contact_geometry_20261109/training/{mode}-seed{seed}/config.json')
            device,world,_,_=setup(cfg);weights=Path(cfg['output'])/'model-step0002000/weights.pt'
            assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
            world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True)
            world.eval().requires_grad_(False);before=tensor_sha(world)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a,k=world.observed(actual);ga,gk=world.observed(goal);values,xy=codes(world,a,k,ga,gk)
                gxy=world.points(ga,gk)[0] if xy is not None else None
            item=dict(weights_sha256=sha(weights),oracle_observed_configuration={key:summary(v.cpu().numpy(),flags,distance) for key,v in values.items()})
            if xy is not None:
                item['point_diagnostics']=dict(mean_abs_coordinate=float(xy.abs().mean()),
                    mean_between_image_point_std=float(xy.std(0,unbiased=False).mean()),
                    mean_within_image_point_std=float(xy.std(1,unbiased=False).mean()),
                    actual_coordinates=xy.cpu().tolist(),goal_coordinates=gxy.cpu().tolist())
            for precision in ['bf16','world32']:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                    hidden=world.parent.observe(world.parent.initial(1),observed[:1])
                    for t,action in enumerate(history,1):
                        hidden=advance(world.parent,hidden,action[None])
                        if t%15==0:hidden=world.parent.observe(hidden,observed[t//15:t//15+1])
                    state=hidden.expand(64,-1,-1,-1).clone()
                    for t in range(60):state=advance(world.parent,state,controls[:,t])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pa,pk=world.predicted(state);v,_=codes(world,pa,pk,ga,gk)
                item['predicted_'+precision]={key:summary(value.cpu().numpy(),flags,distance) for key,value in v.items()}
            assert before==tensor_sha(world);item.update(state_before=before,state_after=before,model_updates=0)
            reports[f'{mode}-seed{seed}']=item
    save(ROOT/f'prefix{prefix}.json',dict(passed=True,utc=now(),prefix=prefix,bank_sha256=sha(path),
        actual_paths=64,any_successes=int(flags.any(1).sum()),final_successes=int(flags[:,-1].sum()),reports=reports,
        protected_audit_only=True,training_teacher_proposal_eligible=False,physics_steps=0,model_updates=0,
        scope='OnepreviouslyinspectedPushcase;observed-codecORACLEisdiagnosticNOTavailabletocontroller. Allfuturecontrolpathsactuallyrealized. No performanceclaim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=int,choices=[0,30,60],required=True);a=p.parse_args();run(a.prefix)
