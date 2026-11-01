"""Real-data CPU contracts for future-query training and native shooting cost."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from as_jepa_lewm.goal_query import GoalQueryLeWM, GoalQueryObjective
from as_jepa_lewm.fixed_forecast import FixedForecastLeWM
from as_jepa_tools.benchmark_data import make_training_dataset
from vforge.ops import OpsConfig, initialize_ops

B=Path('/efs/prescottsun/SPT-JEPA')


def kwargs(cfg):
    result=dict(cfg['model'])
    for key in list(result):
        if key.startswith('renderer_') or key in ['beta','sigma0','rho','forecast_fraction','flow_logit_shift']:
            result.pop(key)
    return result


def main(output):
    torch.set_num_threads(2)
    initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None)
    rows=[]
    for task in ['pusht','tworoom','cube']:
        parent=B/'runs/unified_three_20261101'/f'{task}-flat-seed1'
        cfg=json.loads((parent/'config.json').read_text())
        torch.manual_seed(cfg['seed']);before=torch.get_rng_state()
        model=GoalQueryLeWM(query_seed=1,**kwargs(cfg)).eval()
        after=torch.get_rng_state()
        torch.manual_seed(cfg['seed']);baseline=FixedForecastLeWM(predictor_kind='flat',**kwargs(cfg)).eval()
        assert torch.equal(after,torch.get_rng_state())
        common={k:v for k,v in model.state_dict().items() if k in baseline.state_dict()}
        assert all(torch.equal(v,baseline.state_dict()[k]) for k,v in common.items())
        ds=make_training_dataset(dict(cfg,goal_query_reference=True))
        items=[ds[0],ds[10]];batch=default_collate(items)
        for index,item in zip([0,10],items):
            ref=ds.dataset[ds.reference_index(index)]
            assert torch.equal(item['reference_pixels'],ref['pixels'][3:])
            assert torch.equal(item['pixels'],ds.dataset[index]['pixels'])
        with torch.no_grad():
            normalized=model.normalize(batch['pixels']);observed=model.encode(dict(pixels=normalized))['emb']
            original=baseline.encode(dict(pixels=normalized))['emb'];assert torch.equal(observed,original)
            actions=model.action_encoder(batch['action'])
            queries=model.query_predictions(observed[:,:3],actions[:,:7])
            future_actions=actions.clone();future_actions[:,5:]+=1
            altered=model.query_predictions(observed[:,:3],future_actions[:,:7])
            torch.testing.assert_close(queries[:,:3],altered[:,:3],rtol=0,atol=0)
            calls=[]
            hook=model.predictor.register_forward_pre_hook(lambda _,args:calls.append(tuple(v.clone() for v in args)))
            rng=torch.get_rng_state();whole,_=model.loss(batch,step=2);assert torch.equal(rng,torch.get_rng_state())
            changed=dict(batch);changed['pixels']=batch['pixels'].clone();changed['pixels'][:,3:]=0
            changed['reference_pixels']=batch['reference_pixels']*0
            model.loss(changed,step=2);hook.remove()
            assert len(calls)==2 and all(torch.equal(x,y) for x,y in zip(calls[0],calls[1]))
            parts=[]
            for i in range(2):
                sub={k:v[i:i+1] for k,v in batch.items() if torch.is_tensor(v)}
                parts.append(model.loss(sub,step=2)[0])
            torch.testing.assert_close(whole,torch.stack(parts).mean(),rtol=3e-5,atol=2e-6)
            info=dict(pixels=normalized[:,:3][:,None].expand(-1,2,-1,-1,-1,-1),
                      emb=observed[:,:3][:,None].expand(-1,2,-1,-1),
                      action_history=batch['action'][:,:2][:,None].expand(-1,2,-1,-1),
                      goal_emb=observed[:,7:])
            candidate=batch['action'][:,2:7][:,None].expand(-1,2,-1,-1)
            rolled=model.rollout(info,candidate)
            torch.testing.assert_close(rolled['predicted_query'][:,0],queries,rtol=1e-5,atol=1e-6)
            manual=-model.query_logits(queries[:,-1:],observed[:,7:]).squeeze(1)
            torch.testing.assert_close(GoalQueryObjective()(rolled)[:,0],manual,rtol=1e-5,atol=1e-6)
            cold=dict(pixels=normalized[:,2:3][:,None].expand(-1,2,-1,-1,-1,-1),goal_emb=observed[:,7:])
            assert torch.isfinite(model.rollout(cold,candidate)['goal_query_cost']).all()
        del baseline;gc.collect()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);loss,_=model.loss(batch,step=2)
            loss.backward();assert torch.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        gradients={}
        for prefix,expected in model.audit_expected_gradients().items():
            parameters=[p for n,p in model.named_parameters() if n.startswith(prefix+'.')]
            norm=sum(float(p.grad.double().square().sum()) for p in parameters if p.grad is not None)**.5
            assert norm>0 if expected=='active' else all(p.grad is None for p in parameters)
            gradients[prefix]=dict(expected=expected,norm=norm)
        rows.append(dict(task=task,common_initial_tensors_exact=len(common),initial_outer_RNG_exact=True,
                         native_rollout_score_matches_training=True,cold_start_finite=True,
                         no_future_image_or_reference_leak=True,causal_action_prefix_exact=True,
                         sample_loss_separable=True,reference_draws_independent_of_minibatch=True,
                         parameters=sum(p.numel() for p in model.parameters()),
                         trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),gradients=gradients))
        del model,optimizer,loss,batch,items,ds,common,queries,rolled,calls;gc.collect()
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,models=rows,CUDA_initialized=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Three actual full models,realtrainclips. Independentpaired reference goal loss,no minibatch statistics/optimalactions/physical-label training. Technical andcausal-input contracts,nottaskperformance.')
    with Path(output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(passed=True,models=len(rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
