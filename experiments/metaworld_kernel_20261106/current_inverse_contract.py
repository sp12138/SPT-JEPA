"""Current-only access,paired scratch weights,clock-free/sample-level contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.factory import build_model
from as_jepa_tools.native_inverse import NativeInversePairs
from experiments.metaworld_kernel_20261106.audit import equal


def run(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());initial={};reports=[]
    for recipe in plan['recipes']:
        cfg=json.loads((Path(recipe['run'])/'config.json').read_text())
        assert not any(cfg.get(key)for key in['initial_encoder_checkpoint','initial_model_checkpoint','initial_checkpoint','fork_data'])
        assert not any('checkpoint'in key for key in cfg['model'])
        data=NativeInversePairs(cfg);test=NativeInversePairs(cfg,'test');assert not{r['episode']for r in data.records}&{r['episode']for r in test.records}
        for index in np.linspace(0,len(test)-1,32,dtype=int):
            value=test[int(index)];assert value['action'].shape==(1,20)and value['action_mask'].tolist()==[1.]and value['horizon']==0
            assert value['current_interval_stop']<=int(value['rows'][1])
            start=value['current_interval_start'];stop=value['current_interval_stop'];record=next(r for r in test.records if r['episode']*5==value['episode']);shard=test.shards[record['shard']];original=shard['controls']
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and key.start==start and key.stop==stop,'Attemptednon-currentcommandaccess'
                    return original[key]
            shard['controls']=Guard()
            try:assert equal(value,test[int(index)])
            finally:shard['controls']=original
        torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();state={k:v.clone()for k,v in model.state_dict().items()}
        if cfg['seed']in initial:assert equal(initial[cfg['seed']],state)
        else:initial[cfg['seed']]=state
        if recipe['arm']=='phase0':
            batch=next(iter(DataLoader(Subset(test,[0,len(test)//2]),batch_size=2)));loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
            gradients={prefix:sum(float(p.grad.square().sum())for name,p in model.named_parameters()if name.startswith(prefix+'.')and p.grad is not None)**.5 for prefix in model.audit_expected_gradients()}
            assert all(v>0 for v in gradients.values())
            with torch.no_grad():
                memory=model.memory(batch);target=batch['action'];t=target.new_tensor([.2,.9]);noise=torch.randn_like(target);x=target*(1-t[:,None,None])+noise*t[:,None,None]
                field=model.action_field(x,t,t*0,memory);separate=torch.cat([model.action_field(x[j:j+1],t[j:j+1],t[j:j+1]*0,model.memory({k:v[j:j+1]for k,v in batch.items()}))for j in range(2)])
                error=float((field-separate).abs().max());assert error<1e-4,error
                assert torch.equal(field,model.action_field(x,t,t*0+10000,memory))
                def generator():return torch.Generator().manual_seed(91)
                full=model.propose(batch,generator=generator());minimal=model.propose(dict(pixels=batch['pixels']),generator=generator());assert torch.equal(full,minimal)
                assert equal(state,model.state_dict())
        reports.append(dict(arm=recipe['arm'],seed=cfg['seed'],train_episodes=len(data.records),test_episodes=len(test.records),train_windows=len(data),test_windows=len(test),current_only_guard_checks=32))
    a.output.write_text(json.dumps(dict(passed=True,models=reports,objectives=sorted({json.loads((Path(r['run'])/'config.json').read_text())['model']['objective']for r in plan['recipes']}),paired_initial_weights_within_seed=True,all_weights_random=True,model_updates=0,
        only_current_interval_label=True,clock_perturbation_exact=True,inference_pixels_only_exact=True,
        scope='No sourcecheckpoint/features/teacher.FactualalreadyexecutedCURRENT5native20Dtargetonly;no reading nextintervalcommands.'
              'FutureRGBisdesiredgoal,notfutureactionreference.Onesampleseparablecurrent-inverseloss.Allphase/sampler/noisechoicesglobal.'),indent=2)+'\n');print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
