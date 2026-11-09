"""Frozen heads on actually realized control alternatives; audit data ONLY."""
import argparse
from pathlib import Path
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from .train import setup
from .energy import cost

ROOT=BASE/'runs/metaworld_endpoint_density_20261109/causal-audit-v1'
BANK=BASE/'runs/metaworld_realized_bank_20261109'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    cfg=read(BASE/f'runs/metaworld_endpoint_density_20261109/training/{a.name}/config.json')
    _,parent,head,_,_=setup(cfg);head.eval().requires_grad_(False)
    weights=Path(cfg['output'])/'head-step0002000/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    head.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=(tensor_sha(parent),tensor_sha(head));rows=[]
    out=ROOT/a.name;out.mkdir(parents=True,exist_ok=False)
    save(out/'AUDIT_ONLY.json',dict(utc=now(),model_updates=0,candidate_performance=False,training_data_eligible=False,
        reason='All64controlalternativesphysicallyrealizedfromsamepost7zero-stepstate;8sharedphysicalcases. NEVERSGD/proposallibraryfromprotectedbanks.'))
    try:
        with torch.inference_mode():
            for path in sorted(BANK.glob('*.npz')):
                meta=read(path.with_suffix('.json'));assert meta['passed'] and sha(path)==meta['bank_sha256']
                data=np.load(path);past=torch.from_numpy(data['warm_history'].copy()).permute(0,3,1,2)[None].cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden,image=parent.prepare(dict(history_pixels=past,history_actions=torch.zeros(1,7,4,device='cuda'),
                        history_valid=torch.ones(1,8,device='cuda',dtype=torch.bool)))
                    hidden=hidden.expand(64,-1,-1,-1).clone();forecast=image.expand(64,-1,-1,-1)
                    commands=torch.from_numpy(data['controls'].copy()).cuda()
                    for t in range(60):hidden,forecast=parent.predict(hidden,forecast,commands[:,t])
                actual=parent.pixels(torch.from_numpy(data['actual60'].copy()).permute(0,3,1,2).cuda())
                anchor=(image*2-1).expand(64,-1,-1,-1);target=actual*2-1;seed=620009+meta['seed']
                matched=cost(head,hidden,anchor,target,60,cfg['mode'],seed)
                mismatched=cost(head,hidden.roll(1,0),anchor,target,60,cfg['mode'],seed)
                # Identical pure-noise conditional-mean readout for every head.
                generator=torch.Generator(device='cuda').manual_seed(seed)
                # Match cost's four integer draws before its five Gaussian fields.
                torch.randint(16,(4,),device='cuda',generator=generator)
                noises=torch.randn(5,1,3,128,128,device='cuda',generator=generator);mean=torch.zeros_like(actual)
                duration=torch.full((64,),60,device='cuda',dtype=torch.long)
                for noise in noises:
                    z=noise.expand(64,-1,-1,-1)
                    with torch.autocast('cuda',dtype=torch.bfloat16):v=head(z,torch.ones(64,device='cuda'),hidden,anchor,duration)
                    mean+=((z+v).clamp(-1,1)+1)/10
                errors=(mean-actual).square().mean((1,2,3));copy=(image-actual).square().mean((1,2,3))
                parent_error=(forecast-actual).square().mean((1,2,3))
                moving=np.linalg.norm(data['states60'][:,4:7]-data['states60'][0,4:7],axis=1)>.005
                better=(matched<mismatched).cpu().numpy()
                row=dict(task=meta['task'],bank_sha256=sha(path),mean_readout_mse=float(errors.mean()),parent_rgb_mse=float(parent_error.mean()),
                    copy_mse=float(copy.mean()),mean_over_copy=float(errors.sum()/copy.sum().clamp_min(1e-12)),
                    parent_over_copy=float(parent_error.sum()/copy.sum().clamp_min(1e-12)),
                    true_input_better_than_cyclic_wrong_input=int(better.sum()),comparisons=64,
                    moving_object_proxy_paths=int(moving.sum()),moving_object_proxy_true_better=int(better[moving].sum()),
                    mean_matched_score=float(matched.mean()),mean_wrong_score=float(mismatched.mean()),
                    score_kind='mean_squared_error' if cfg['mode']=='mean' else 'NELBO_minus_action_constant_prior')
                rows.append(row);save(out/'progress.json',dict(utc=now(),rows=rows));print(row,flush=True)
        assert before==(tensor_sha(parent),tensor_sha(head))
        save(out/'DONE.json',dict(passed=True,utc=now(),rows=rows,parent_before=before[0],parent_after=before[0],head_before=before[1],head_after=before[1],
            model_updates=0,physics_steps=0,training_data_eligible=False,candidate_performance=False,
            scope='512pairedrealizedtransitionsfrom8states,NOT512independentcases. Wronginputiscyclicactualalternative;nooptimal-actionreference. '
                'Object-coordinate5mmproxyONLYaudit/subgroup,notlearner/planner/strictcontactlabel. InputdiscriminationcanbeexplainedbyarmmotionanddoesNOTproveobjectmodelquality.'))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
