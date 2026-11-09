"""Actual input identity, sample isolation and FP32 GPU-reduction checks."""
import argparse
import numpy as np
import torch
from torch.utils.data import default_collate
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,now,sha,tensor_sha
from .data import HeldPrimitives
from .model import random_terms
from .train import setup,gradients


def run():
    root=BASE/'runs/metaworld_causal_inverse_20261109';p=read(root/'plan.json')
    cfg=read(next(x['config'] for x in p['jobs'] if x['mode']=='flow_highnoise' and x['seed']==1))
    device,model,_,_=setup(cfg);data=HeldPrimitives(cfg['old_data'],cfg['new_data'],'train')
    indices=np.linspace(0,len(data)-1,64,dtype=int);batch=default_collate([data[i] for i in indices]);batch={k:v.to(device) for k,v in batch.items()}
    terms=random_terms(batch);seen=[]
    original=model.body.predict
    def traced(hidden,image,action):
        seen.append(action.detach().cpu().numpy().copy());return original(hidden,image,action)
    model.body.predict=traced;loss,_=model(batch,terms);loss.backward();gradients(model,positive=True)
    # Remove the temporary instance override. Keeping a bound method here would
    # make DataParallel replicas call the original cuda:0 module.
    del model.body.predict
    assert len(seen)==12 and all(np.array_equal(a,batch['known_input_a0'].cpu().numpy()) for a in seen[-5:])
    one=torch.cat([x.grad.flatten() for x in model.parameters() if x.requires_grad and x.grad is not None]).clone();rel={}
    for count in [2,4]:
        model.zero_grad(set_to_none=True);v,_=torch.nn.DataParallel(model,device_ids=list(range(count)))(batch,terms);v=v.mean();v.backward()
        other=torch.cat([x.grad.flatten() for x in model.parameters() if x.requires_grad and x.grad is not None])
        rel[str(count)]=float((one-other).norm()/one.norm());assert rel[str(count)]<1e-4
    # The output of one sample must not depend on another sample in this GN learner.
    small={k:v[:2].clone() for k,v in batch.items()};smallterms=tuple(v[:2] for v in terms)
    model.zero_grad(set_to_none=True);v1=model(small,smallterms)[0]
    singles=[model({k:v[i:i+1] for k,v in small.items()},tuple(v[i:i+1] for v in smallterms))[0] for i in range(2)]
    isolation=float((v1-torch.stack(singles).mean()).abs());assert isolation<1e-5
    for i,index in enumerate(indices):
        ri,t=data.rows[int(index)];r=data.records[ri];s=data.shards[r['shard']];o=r['offset'];a0=s['inputs'][o+t]
        assert np.array_equal(np.asarray(s['inputs'][o+t:o+t+5]),np.broadcast_to(a0,(5,4)))
        assert np.array_equal(batch['known_input_a0'][i].cpu().numpy(),a0)
        assert np.array_equal(batch['future_pixels'][i].permute(0,2,3,1).cpu().numpy(),s['pixels'][o+t+1:o+t+6])
    save(root/'GPU_DATAFLOW_AUDIT.json',dict(passed=True,utc=now(),relative_gradients=rel,sample_isolation_error=isolation,
        input_identity='ALL5forwardcallsreceiveEXACTSAMElabela0;observedfutureRGBbytechecks64/64.',
        inverse_label_shape=list(batch['known_input_a0'].shape),pool_counts=data.pool_counts,optimizer_updates=0,
        unknown_future_actions=False,desired_goal_or_expertpath_training=False))


@torch.no_grad()
def calibrate(config,step):
    cfg=read(config);device,model,_,_=setup(cfg);model.eval().requires_grad_(False)
    weights=Path(cfg['output'])/f'model-step{step:07d}/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    model.load_state_dict(torch.load(weights,map_location=device,weights_only=True),strict=True);before=tensor_sha(model)
    data=HeldPrimitives(cfg['old_data'],cfg['new_data'],'test');cut=data.pool_counts['ordinary_expert'];reports={}
    for pool,lo,hi in [('ordinary_expert',0,cut),('random_intervention',cut,len(data))]:
        indices=np.unique(np.linspace(lo,hi-1,min(256,hi-lo),dtype=int));values=[]
        generator=torch.Generator(device=device).manual_seed(1079001)
        for start in range(0,len(indices),32):
            batch=default_collate([data[i] for i in indices[start:start+32]]);batch={k:v.to(device) for k,v in batch.items()}
            body=model.body;hidden,current=body.prepare(batch);a0=batch['known_input_a0'];future=body.pixels(batch['future_pixels'][:,0 if cfg['mode']=='native1' else 4])
            context=model.condition(hidden,current,future);n=len(context);samples=32
            def distribution(context):
                if cfg['mode'].startswith('flow_'):
                    c=context[:,None].expand(-1,samples,-1).reshape(n*samples,-1)
                    value=torch.randn(n*samples,4,device=device,generator=generator)
                    for i in range(16):value=value-model.velocity(c,value,value.new_full((n*samples,),1-i/16))/16
                    return value.reshape(n,samples,4).clamp(-1,1)
                mean=model.base_action(context)+model.correction(context,context.new_zeros(n,4),context.new_zeros(n))
                return mean[:,None].expand(-1,samples,-1).clamp(-1,1)
            predicted=distribution(context);erased=distribution(model.condition(hidden,current,current));mean=predicted.mean(1)
            energy=(predicted-a0[:,None]).norm(dim=-1).mean(1)-.5*torch.cdist(predicted,predicted).mean((1,2))
            forecast=current;h=hidden;errors=[];copy=[]
            for t in range(5):
                h,forecast=body.predict(h,forecast,a0);target=body.pixels(batch['future_pixels'][:,t])
                errors.append((forecast-target).square().mean((1,2,3)));copy.append((current-target).square().mean((1,2,3)))
            measures=dict(current_a0_MSE=(mean-a0).square().mean(1),XYZ_MSE=(mean[:,:3]-a0[:,:3]).square().mean(1),
                grip_MSE=(mean[:,3]-a0[:,3]).square(),future_erased_a0_MSE=(erased.mean(1)-a0).square().mean(1),
                zero_a0_MSE=a0.square().mean(1),energy_score=energy,forward_RGB_MSE=torch.stack(errors).mean(0),copy_RGB_MSE=torch.stack(copy).mean(0))
            values.append({k:v.cpu().numpy() for k,v in measures.items()})
        reports[pool]=dict(rows=len(indices),metrics={k:float(np.concatenate([r[k] for r in values]).mean()) for k in values[0]})
    assert tensor_sha(model)==before
    save(Path(cfg['output'])/f'CALIBRATION-{step:07d}.json',dict(passed=True,utc=now(),step=step,reports=reports,
        NN_unchanged=True,optimizer_updates=0,scope='TESTonly/currentinputa0alreadyusedbyforward;energywithinSAMEsampleposterior,nobatchnegatives. '
            'MSEclippedposteriorpointmean,notbest-of-labelselection. Futureerasureablationnotuniquepolicyshortcutproof.'))


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--calibrate',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--step',type=int);a=p.parse_args()
    if a.calibrate:calibrate(a.config,a.step)
    else:run()
