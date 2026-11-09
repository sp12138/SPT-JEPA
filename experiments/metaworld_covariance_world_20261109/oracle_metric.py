"""Protected scoring audit: separate forecast error from covariance goal geometry."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_covariance_world_20261109.train import setup
from experiments.metaworld_covariance_world_20261109.model import gaussian_nll

BANK=BASE/'runs/metaworld_belief_state_20261109/contact-audit-v1'
ROOT=BASE/'runs/metaworld_covariance_world_20261109/oracle-metric-v1'


def summary(value,flags):
    score=value.cpu().numpy();selected=int(score.argmin());out=dict(selected=selected,selected_any_success=bool(flags[selected].any()),selected_final_success=bool(flags[selected,-1]),scores=score.tolist())
    for name,f in [('any',flags.any(1)),('final',flags[:,-1])]:
        out['AUC_'+name]=float(((score[f,None]<score[None,~f])+.5*(score[f,None]==score[None,~f])).mean()) if f.any() and (~f).any() else None
    return out


@torch.no_grad()
def run(prefix):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    path=BANK/f'prefix{prefix}.npz';assert sha(path)==read(path.with_suffix('.json'))['bank_sha256'];data=np.load(path)
    flags=data['successes'];actual=torch.from_numpy(data['actual60']).permute(0,3,1,2).cuda().float()/255
    goal=torch.from_numpy(data['goal'].copy()).permute(2,0,1)[None].cuda().float()/255
    observed=torch.from_numpy(data['observations']).permute(0,3,1,2).cuda().float()/255
    past=torch.from_numpy(data['prefix_actions']).cuda();controls=torch.from_numpy(data['controls']).cuda()
    reports={};direction=torch.zeros(1,device='cuda',dtype=torch.long)
    for seed in [1,2]:
        for mode in ['scalar','diagonal','lowrank']:
            cfg=read(BASE/f'runs/metaworld_covariance_world_20261109/training/{mode}-seed{seed}/config.json')
            device,world,_,_=setup(cfg);weights=Path(cfg['output'])/'model-step0002000/weights.pt'
            assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
            world.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True);world.eval().requires_grad_(False);before=tensor_sha(world)
            parent=world.parent.parent;hidden=parent.observe(parent.initial(1),observed[:1])
            for t,action in enumerate(past,1):
                hidden=world.parent.advance(hidden,action[None],direction)
                if t%15==0:hidden=parent.observe(hidden,observed[t//15:t//15+1])
            state=hidden.expand(64,-1,-1,-1).clone()
            for t in range(60):state=world.parent.advance(state,controls[:,t],direction.expand(64))
            mean=world.parent.render(state,observed[-1:]);variance,basis=world.distribution(state)
            common_d=variance[:1].expand(64,-1,-1,-1);common_u=None if basis is None else basis[:1].expand(64,-1,-1)
            values=dict(actual_RGB_MSE=(goal-actual).square().mean((1,2,3)),predicted_RGB_MSE=(goal-mean).square().mean((1,2,3)),
                actual_RGB_matched_covariance=gaussian_nll(goal-actual,variance,basis),
                predicted_RGB_matched_covariance=gaussian_nll(goal-mean,variance,basis),
                actual_RGB_common_zero_control_covariance=gaussian_nll(goal-actual,common_d,common_u),
                predicted_RGB_common_zero_control_covariance=gaussian_nll(goal-mean,common_d,common_u))
            assert before==tensor_sha(world)
            reports[f'{mode}-seed{seed}']=dict(weights_sha256=sha(weights),state_before=before,state_after=before,
                metrics={k:summary(v,flags) for k,v in values.items()},model_updates=0)
    save(ROOT/f'prefix{prefix}.json',dict(passed=True,utc=now(),prefix=prefix,bank_sha256=sha(path),reports=reports,
        actual_paths=64,any_successes=int(flags.any(1).sum()),final_successes=int(flags[:,-1].sum()),
        physics_steps=0,model_updates=0,protected_audit_only=True,training_teacher_proposal_eligible=False,
        scope='OneinspectedPushcase/3branchstates,notcontrollerperformance. ACTUALfutureRGBisORACLEnotavailabletoplanner. '
        'Bothcandidate-matchedcovarianceandcommonzero-controlcovarianceareDIAGNOSTICS;notnewpolicyproposalsorhyperparameterselection.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=int,choices=[0,30,60],required=True);run(p.parse_args().prefix)
