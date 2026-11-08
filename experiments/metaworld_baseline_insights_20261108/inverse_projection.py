"""CPU factual-vs-imagined inverse consistency;no goal-to-action policy.

Every inverse reference is the SAME supplied forward input at that transition,
never the next action or an unknown optimal action. No desired goal enters I.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiments.metaworld_baselines_20261107.models import PublicBaseline
from experiments.metaworld_baselines_20261107.train_native import Windows,digest

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_inverse_projection_20261108/v2'


def read(p):return json.loads(Path(p).read_text())


def main(name):
    assert not torch.cuda.is_initialized();torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    out=ROOT/name;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    j=next(j for j in read(BASE/'runs/metaworld_baselines_native_20261108/plan.json')['jobs'] if j['name']==name)
    cfg=read(j['config']);checkpoint=Path(j['output'])/f"model-step{j['epoch_steps']['100']:07d}"
    model=PublicBaseline(cfg['method'],cfg['sources']);model.load_state_dict(torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before=digest(model)
    head=model.core.inverse_model if cfg['method']=='smwm' else model.core.get_level(1).idm_loss_fn.idm
    data=Windows(cfg['data'],8,cfg['seed'],'test');indices=np.linspace(0,len(data)-1,16,dtype=np.int64)
    batch=torch.utils.data.default_collate([data[int(i)] for i in indices])
    card=read(Path(cfg['data'])/'dataset_card.json');mean=torch.tensor(card['action_mean']);std=torch.tensor(card['action_std'])
    lower=(-1-mean)/std;upper=(1-mean)/std
    # Context observations0/1 use the already-executed input0. Predictive inputs
    # and inverse references are BOTH inputs1..6 for transitions1->2..6->7.
    factual=batch['action'][:,1:7,None].transpose(1,2).contiguous()
    info=dict(pixels=model.normalize(batch['pixels'][:,:2])[:,None],action_history=batch['action'][:,:1,None].transpose(1,2))
    assert set(info)=={'pixels','action_history'} and torch.equal(factual[:,0],batch['action'][:,1:7])
    generator=torch.Generator().manual_seed(20261108)
    random=torch.randn(16,32,6,20,generator=generator).clamp(min=lower,max=upper)

    def inverse_path(prediction):
        left,right=prediction[:,:,1:-1],prediction[:,:,2:]
        output=head(left.reshape(-1,left.shape[-1]),right.reshape(-1,right.shape[-1])).reshape(*left.shape[:-1],20)
        return output

    def rollout(actions):
        payload=dict(info)
        payload['action_history']=info['action_history'].expand(-1,actions.shape[1],-1,-1)
        return model.rollout(payload,actions)['predicted_emb'].float()

    with torch.inference_mode():
        z=model.encode(dict(pixels=model.normalize(batch['pixels'])))['emb'].float()
        factual_pred=rollout(factual);assert factual_pred.shape[:3]==(16,1,8)
        assert torch.allclose(factual_pred[:,0,:2],z[:,:2],atol=1e-5,rtol=1e-5)
        observed=head(z[:,1:7].reshape(-1,z.shape[-1]),z[:,2:8].reshape(-1,z.shape[-1])).reshape(16,1,6,20)
        imagined_inverse=inverse_path(factual_pred)
        factual_projected=imagined_inverse.clamp(min=lower,max=upper);factual_projected_pred=rollout(factual_projected)
        random_pred=rollout(random);random_projected=inverse_path(random_pred).clamp(min=lower,max=upper);random_projected_pred=rollout(random_projected)
    assert digest(model)==before and not torch.cuda.is_initialized()
    tensors=[factual,observed,factual_pred,factual_projected,factual_projected_pred,random,random_pred,random_projected,random_projected_pred]
    assert all(torch.isfinite(t).all() for t in tensors)
    observed_error=(observed-factual).square().reshape(16,1,6,5,4).mean((0,1,2,3))
    imagined_error=(imagined_inverse-factual).square().reshape(16,1,6,5,4).mean((0,1,2,3))
    baseline=factual.square().reshape(16,1,6,5,4).mean((0,1,2,3))
    forward=(factual_pred[:,0,2:]-z[:,2:]).square().mean((0,2))
    projected_forward=(factual_projected_pred[:,0,2:]-z[:,2:]).square().mean((0,2))
    disagreement=(random_projected_pred[:,:,2:]-random_pred[:,:,2:]).square().mean((0,1,3))
    effect=(random_pred[:,:,2:]-z[:,None,1:2]).square().mean((0,1,3))
    def controls(a):
        raw=(a*std+mean).numpy().reshape(-1,5,4)
        return dict(within_block_temporal_variation=float(np.square(raw-raw.mean(1,keepdims=True)).sum()/max(np.square(raw).sum(),1e-20)),
            raw_axis_std=raw.std((0,1)).tolist(),raw_axis_mean=raw.mean((0,1)).tolist())
    identity=[dict(index=int(i),original_episode=int(data.base.episode_ids[data.base.locate(int(i))[0]])//5) for i in indices]
    arrays={k:v.numpy() for k,v in zip(['factual_input','inverse_observed','factual_prediction','projected_factual_input','projected_factual_prediction',
        'random_input','random_prediction','projected_random_input','projected_random_prediction'],tensors)}
    np.savez_compressed(out/'cached.npz',**arrays,indices=indices,imagined_inverse=imagined_inverse.numpy())
    d=dict(passed=True,name=name,native_epoch=100,windows=16,random_candidates_per_window=32,horizon=6,
        same_input_action_alignment=True,desired_goal_supplied_to_inverse=False,
        observed_inverse_mse_by_axis=observed_error.tolist(),imagined_inverse_mse_by_axis=imagined_error.tolist(),
        mean_input_baseline_mse_by_axis=baseline.tolist(),imagined_over_observed_inverse_mse_by_axis=(imagined_error/observed_error.clamp_min(1e-20)).tolist(),
        factual_forward_mse_by_horizon=forward.tolist(),projected_factual_forward_mse_by_horizon=projected_forward.tolist(),
        projected_over_original_factual_forward_mse=(projected_forward/forward.clamp_min(1e-20)).tolist(),
        random_projection_forward_effect_disagreement=(disagreement/effect.clamp_min(1e-20)).tolist(),
        factual_controls=controls(factual),random_controls=controls(random),projected_controls=controls(random_projected),
        model_state_before=before,model_state_after=digest(model),worldmodel_updates=0,neural_optimizers=0,gpu_context_created=False,
        new_physical_episodes=0,seconds=time.monotonic()-started,windows_identity=identity,
        scope='Factualinputsareknownexecutedforwardcontrols,notoptimal. RandomGaussian/clippedproposalsareunsimulated,notactualCEMbankreplays. '
            'P(a)=I(z,F(z,a))input-cycleprojection. Diagnosticonly,notdeployedgoal-IDMpolicy/controlsuccess. '
            'Ownlatenterrorratiosarenotsharedphysicalunits. No current/nextactionshift,no privilegedstate/reward/taskinputs/targets.')
    (out/'summary.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps({k:d[k] for k in ['passed','name','imagined_over_observed_inverse_mse_by_axis','projected_over_original_factual_forward_mse','random_projection_forward_effect_disagreement']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True,choices=['smwm-seed1','h-jepa-seed2']);a=p.parse_args()
    try:main(a.name)
    except BaseException:
        out=ROOT/a.name;out.mkdir(parents=True,exist_ok=True);(out/'ERROR.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n');raise
