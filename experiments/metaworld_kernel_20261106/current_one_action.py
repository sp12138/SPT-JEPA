"""Does the full-five-action requirement exclude crucial near-terminal states?"""
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
    for gpu,arm,seed,reserve in[(0,'exclude-last4',1,5),(1,'exclude-last4',2,5),(2,'all-executed',1,1),(3,'all-executed',2,1)]:
        cfg=json.loads((a.parent/'training'/f'directed-flow-seed{seed}'/'config.json').read_text());run=a.root/'training'/f'{arm}-seed{seed}';run.mkdir(parents=True)
        cfg.update(output=str(run),inverse_tail_reserve=reserve);cfg['model'].update(action_dim=4,history_context=False)
        save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,target='ONLYa_t4D,theSINGLEcurrentalreadyexecutednativeaction.Noa_t+1orlatertarget.',
            loss='One sample-levelflowmatchingloss.Noreconstructionofimage/extraauxiliaryloss/stoplabels.',
            deployment='SameRGBsensor5native/500anysuccess;oneownpredicted4Dcommandheld5native.Nofastervisualfeedback/proprio/state/reward/taskID/clock.',
            study='Botharmsidenticalcurrent4Darchitecture/initialweights/originalcorpus/goalrules/batch/40kbudget.Onlyavailablecurrent-statecoveragechanges.',
            tail='AllL-1knowncurrentnativeactions,includinglast4previouslyexcluded;lastframewithunknownnextactionEXCLUDED.'if reserve==1 else'SameL-5startsupportaspreviouscurrent5-actionmodels;deliberatelyexcludelast4knownstatesascoveragecontrol.',
            goals='Unchanged5-nativephase-mixturewherepossible;fortheNEWremaining1-4states,theACTUALlastRGBisgoal.No syntheticpadding/actionlabels.',
            limits='Versus20Dbaseline,decoderinput/outputdimensionandholdingchange;causalterminalcoveragecontrastisBETWEENthesetwo4Darms,notvs20D.'))
        owner=next(r for r in owners if r['gpu']==gpu)
        recipes.append(dict(host='dev6',gpu=gpu,arm=arm,seed=seed,run=str(run),resource_pid=owner['pid'],resource_marker='frozen-worker.py',config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text());save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=old['stages'],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        scope='SingleCURRENTnativeactiononly,twoindependentseedsxlast-statecoverage.Original2250/250trajectories;noextradata/pretraining/futureactionlabels.Evaluationkeeps5-nativeRGBcadence.Nohiddenstoprule.'))


def contract(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());initial={};reports=[]
    for r in plan['recipes']:
        cfg=json.loads((Path(r['run'])/'config.json').read_text());data=NativeInversePairs(cfg);test=NativeInversePairs(cfg,'test')
        assert not{v['episode']for v in data.records}&{v['episode']for v in test.records}
        assert data.steps==1 and data.action_dim==4
        for i in np.linspace(0,len(test)-1,64,dtype=int):
            v=test[int(i)];rec=next(x for x in test.records if x['episode']*5==v['episode']);shard=test.shards[rec['shard']];old=shard['controls'];start=v['current_interval_start']
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and key.start==start and key.stop==start+1,'Non-current native action read'
                    return old[key]
            shard['controls']=Guard()
            try:assert equal(v,test[int(i)])
            finally:shard['controls']=old
            assert v['action'].shape==(1,4)and np.array_equal(v['action'][0].numpy(),(old[start]-test.mean)/test.std)
        if test.reserve==1:
            direct_cfg=dict(cfg,inverse_balance='frames');direct=NativeInversePairs(direct_cfg,'test');offset=0;tail_checks=0
            for rec in direct.records:
                for t in range(rec['length']-5,rec['length']-1):
                    v=direct[offset+t];assert v['start']==t and v['current_interval_stop']==rec['offset']+t+1
                    assert int(v['rows'][1])==rec['offset']+rec['length']-1;tail_checks+=1
                offset+=rec['length']-1
            assert tail_checks==len(test.records)*4
        else:tail_checks=0
        torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();state={k:v.clone()for k,v in model.state_dict().items()}
        if cfg['seed']in initial:assert equal(initial[cfg['seed']],state)
        else:initial[cfg['seed']]=state
        batch=next(iter(DataLoader(Subset(test,[0,len(test)//2]),batch_size=2)));loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
        gradients={p:sum(float(v.grad.square().sum())for n,v in model.named_parameters()if n.startswith(p+'.')and v.grad is not None)**.5 for p in model.audit_expected_gradients()};assert all(v>0 for v in gradients.values())
        with torch.no_grad():
            def generator():return torch.Generator().manual_seed(91)
            assert torch.equal(model.propose(batch,generator=generator()),model.propose(dict(pixels=batch['pixels']),generator=generator()))
            t=torch.ones(2);x=torch.zeros_like(batch['action']);field=model.action_field(x,t,t*0,model.memory(batch))
            single=torch.cat([model.action_field(x[j:j+1],t[j:j+1],t[j:j+1]*0,model.memory(dict(pixels=batch['pixels'][j:j+1])))for j in range(2)])
            err=float((field-single).abs().max());assert err<1e-4,err
            assert equal(state,model.state_dict())
        reports.append(dict(arm=r['arm'],seed=cfg['seed'],train_starts=len(data),test_starts=len(test),one_current_native_guards=64,actual_near_terminal_checks=tail_checks,
            independent_sample_max=err,gradients=gradients))
    save(a.root/'initial-contract.json',dict(passed=True,reports=reports,paired_initial_weights_within_seed=True,all_weights_random=True,new_model_updates=0,
        no_future_or_fake_current_action_labels=True,scope='Sameone4Dlabel,onlylegitimateterminalstatecoveragechanges;unknownlastframeactionneverinvented.No newepisodes/extradata.'))
    print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract'],required=True)
    for k in['root','source','parent','wait-parent']:p.add_argument('--'+k,type=Path)
    a=p.parse_args();prepare(a)if a.mode=='prepare'else contract(a)
