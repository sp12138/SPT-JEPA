"""Finite observation-history test of partial observability,current labels only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_tools.native_inverse import NativeInversePairs
from as_jepa_lewm.factory import build_model
from experiments.metaworld_kernel_20261106.campaign import save,digest
from experiments.metaworld_kernel_20261106.audit import equal


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);(a.root/'results').mkdir();owners=json.loads((a.wait_parent/'launch-dev6.json').read_text())['workers'];recipes=[]
    for seed in[1,2]:
        for j,objective in enumerate(['flow','mse']):
            gpu=4+(seed-1)*2+j;cfg=json.loads((a.parent/'training'/f'episodes-seed{seed}'/'config.json').read_text());run=a.root/'training'/f'{objective}-seed{seed}';run.mkdir(parents=True)
            cfg['output']=str(run);cfg['model'].update(history_context=True,objective=objective)
            save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,one_current_20D_target=True,
                learner_inputs='PastRGB(t-5 orstart),currentRGB,goalRGB,ONLYpastalreadyexecuted5controls+validity;no future control/horizon/state/taskID/reward/success.',
                motivation='SameRGBcanhidemotion/contact/actuatorstate.Conditionalactionvariancecannotberemovedbyabiggercurrent-onlynetwork;shortobservedhistoryapproximatesbeliefstate.',
                dataset='Sameoriginaltrajectories/currentanchors/currenttargets/goalRGBsamplingasseed-matchednocontroller-historybaseline.',
                comparison='Same40kupdates/batch128,butoneadditionalCNNimageandpasttoken,not equalFLOPs.No clock/temporalaux/predicted actionhistory.'))
            owner=next(o for o in owners if o['gpu']==gpu)
            recipes.append(dict(host='dev6',gpu=gpu,seed=seed,arm=objective,run=str(run),resource_pid=owner['pid'],resource_marker='frozen-reference-worker.py',config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text());save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=old['stages'],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        scope='Fourrandomcurrent-onlyhistorycontrollers,one sample-level loss.Pastownexecutedcontrolsareinputs,neverfuturelabels.Originalcorpusonly;twoindependentinitialseeds.NoSOTA/equalFLOPclaim.'))


def contract(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());initial={};reports=[]
    for r in plan['recipes']:
        cfg=json.loads((Path(r['run'])/'config.json').read_text());data=NativeInversePairs(cfg);test=NativeInversePairs(cfg,'test')
        assert not{v['episode']for v in data.records}&{v['episode']for v in test.records}
        base_cfg=json.loads(json.dumps(cfg));base_cfg['model']['history_context']=False;base=NativeInversePairs(base_cfg,'test')
        for i in np.linspace(0,len(test)-1,64,dtype=int):
            v=test[int(i)];plain=base[int(i)];assert torch.equal(v['action'],plain['action'])and torch.equal(v['pixels'][1:],plain['pixels'])
            rec=next(x for x in test.records if x['episode']*5==v['episode']);shard=test.shards[rec['shard']];old=shard['controls'];start=v['current_interval_start'];stop=v['current_interval_stop'];n=min(v['start'],5)
            allowed={(start,stop),(start-n,start)}
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and(key.start,key.stop)in allowed,'Accessoutsideactualpast/currentinterval'
                    return old[key]
            shard['controls']=Guard()
            try:assert equal(v,test[int(i)])
            finally:shard['controls']=old
            assert v['history_valid']==np.float32(n/5)
            expected=np.zeros(20,np.float32)
            if n:expected[-n*4:]=(old[start-n:start].reshape(-1)-test.mean[-n*4:])/test.std[-n*4:]
            assert np.array_equal(expected,v['previous_action'].numpy())
        torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();state={k:v.clone()for k,v in model.state_dict().items()}
        if cfg['seed']in initial:assert equal(initial[cfg['seed']],state)
        else:initial[cfg['seed']]=state
        torch.manual_seed(cfg['seed']);plain=build_model(base_cfg)
        for k,v in plain.state_dict().items():
            if k!='kind':assert torch.equal(v,state[k]),k
        batch=next(iter(DataLoader(Subset(test,[0,len(test)//2]),batch_size=2)));loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
        gradients={p:sum(float(x.grad.square().sum())for n,x in model.named_parameters()if n.startswith(p+'.')and x.grad is not None)**.5 for p in model.audit_expected_gradients()};assert all(v>0 for v in gradients.values())
        with torch.no_grad():
            minimal={k:batch[k]for k in['pixels','previous_action','history_valid']}
            def generator():return torch.Generator().manual_seed(91)
            assert torch.equal(model.propose(batch,generator=generator()),model.propose(minimal,generator=generator()))
            t=torch.ones(2);x=torch.zeros_like(batch['action']);field=model.action_field(x,t,t*0,model.memory(minimal))
            singles=torch.cat([model.action_field(x[j:j+1],t[j:j+1],t[j:j+1]*0,model.memory({k:v[j:j+1]for k,v in minimal.items()}))for j in range(2)])
            error=float((field-singles).abs().max());assert error<1e-4,error
            assert equal(state,model.state_dict())
        reports.append(dict(objective=r['arm'],seed=r['seed'],current_and_past_interval_guards=64,shared_targets_and_CNN_decoder_initialization=True,gradients=gradients,independent_sample_max=error))
    save(a.root/'initial-contract.json',dict(passed=True,reports=reports,all_weights_random=True,model_updates=0,
        inference_inputs_only_pixels_and_own_past=True,no_future_or_current_target_actions_in_inference=True))
    print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract'],required=True)
    for k in['root','source','parent','wait-parent']:p.add_argument('--'+k,type=Path)
    a=p.parse_args();prepare(a)if a.mode=='prepare'else contract(a)
