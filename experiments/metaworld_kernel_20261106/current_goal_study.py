"""Matched current-only goal semantics and ordered visual representation study."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.factory import build_model
from as_jepa_tools.native_inverse import NativeInversePairs
from experiments.metaworld_kernel_20261106.campaign import save,digest
from experiments.metaworld_kernel_20261106.audit import equal


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False);recipes=[]
    settings=[(0,'terminal-flow','terminal','absolute','flow'),(2,'terminal-mse','terminal','absolute','mse'),
              (3,'directed-flow','phase_mixture','directed','flow'),(4,'directed-mse','phase_mixture','directed','mse'),
              (6,'terminal-directed-flow','terminal','directed','flow'),(7,'terminal-directed-mse','terminal','directed','mse')]
    for host,seed in[('dev13',1),('dev15',2)]:
        parent=a.parent/'training'/f'episodes-seed{seed}';base=json.loads((parent/'config.json').read_text())
        for gpu,arm,goals,representation,objective in settings:
            cfg=json.loads(json.dumps(base));run=a.root/'training'/f'{arm}-seed{seed}';run.mkdir(parents=True)
            cfg.update(output=str(run),inverse_goals=goals);cfg['model'].update(goal_representation=representation,objective=objective)
            save(run/'config.json',cfg);save(run/'ORIGIN.json',dict(all_weights_random=True,one_current_interval_target=True,
                source_data='Original2250train/250testtrajectories;rawnative224RGB.Onlyalreadyexecutedcurrent5commandsarelabels.',
                goal='ActualLASTnativeRGB,withoutsuccessfilter' if goals=='terminal' else 'Originalsame-phaseendpoint/randomfutureRGBmixture',
                representation=representation,limitations='Inversecontroller,notpureJEPA.Lossissample-level;nofutureactiontarget/teacher/auxiliaryloss.'))
            recipes.append(dict(arm=arm,seed=seed,host=host,gpu=gpu,run=str(run),resource_pid=0,resource_marker='unallocated',config_sha256=digest(run/'config.json')))
    old=json.loads((a.parent/'training-plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),recipes=recipes,stages=old['stages'],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        scope='Threefactorialcells(terminal,orderedrepresentation,both)xflow/MSExtwoindependentseeds;originalabsolute/mixedcellalreadytrainedascontrol.'
              'Samecurrent-onlytarget,batch128/40000updates/LRhorizon.Noexternalpretraining/extradata/privilegedinputs/taskrules.'))


def contract(a):
    torch.set_num_threads(2);plan=json.loads((a.root/'training-plan.json').read_text());states={};reports=[]
    for r in plan['recipes']:
        cfg=json.loads((Path(r['run'])/'config.json').read_text());data=NativeInversePairs(cfg);test=NativeInversePairs(cfg,'test')
        assert not{v['episode']for v in data.records}&{v['episode']for v in test.records}
        baseline_cfg=json.loads(json.dumps(cfg));baseline_cfg['inverse_goals']='phase_mixture';baseline_cfg['model']['goal_representation']='absolute'
        baseline_data=NativeInversePairs(baseline_cfg,'test')
        for i in np.linspace(0,len(test)-1,32,dtype=int):
            v=test[int(i)];base=baseline_data[int(i)];assert torch.equal(v['action'],base['action'])and torch.equal(v['pixels'][0],base['pixels'][0])
            rec=next(x for x in test.records if x['episode']*5==v['episode']);start=v['current_interval_start'];stop=v['current_interval_stop'];shard=test.shards[rec['shard']];controls=shard['controls']
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and key.start==start and key.stop==stop,'Non-current action access'
                    return controls[key]
            shard['controls']=Guard()
            try:assert equal(v,test[int(i)])
            finally:shard['controls']=controls
            if cfg['inverse_goals']=='terminal':assert int(v['rows'][1])==rec['offset']+rec['length']-1
        torch.manual_seed(cfg['seed']);model=build_model(cfg).eval();state={k:v.clone()for k,v in model.state_dict().items()}
        key=(cfg['seed'],cfg['model']['goal_representation'])
        if key in states:assert equal(state,states[key])
        else:states[key]=state
        torch.manual_seed(cfg['seed']);baseline=build_model(baseline_cfg).eval()
        for k,v in baseline.state_dict().items():
            if k!='kind':assert torch.equal(v,state[k]),k
        batch=next(iter(DataLoader(Subset(test,[0,len(test)//2]),batch_size=2)))
        loss,_=model.loss(batch);assert torch.isfinite(loss);loss.backward()
        gradients={p:sum(float(x.grad.square().sum())for n,x in model.named_parameters()if n.startswith(p+'.')and x.grad is not None)**.5 for p in model.audit_expected_gradients()}
        assert all(x>0 for x in gradients.values()),gradients
        with torch.no_grad():
            memory=model.memory(batch);t=torch.ones(2);x=torch.zeros_like(batch['action']);field=model.action_field(x,t,t*0,memory)
            singles=torch.cat([model.action_field(x[j:j+1],t[j:j+1],t[j:j+1]*0,model.memory(dict(pixels=batch['pixels'][j:j+1])))for j in range(2)])
            error=float((field-singles).abs().max());assert error<1e-4,error
            swapped=dict(pixels=batch['pixels'].flip(1));swap=float((field-model.action_field(x,t,t*0,model.memory(swapped))).square().mean().sqrt())
            bswap=float((baseline.action_field(x,t,t*0,baseline.memory(batch))-baseline.action_field(x,t,t*0,baseline.memory(swapped))).abs().max())
            assert bswap<1e-5,bswap
            if cfg['model']['goal_representation']=='directed':assert swap>1e-5,swap
            def generator():return torch.Generator().manual_seed(91)
            assert torch.equal(model.propose(batch,generator=generator()),model.propose(dict(pixels=batch['pixels']),generator=generator()))
            assert equal(state,model.state_dict())
        reports.append(dict(arm=r['arm'],seed=cfg['seed'],current_interval_guards=32,train_windows=len(data),test_windows=len(test),
            initial_absolute_swap_max=bswap,initial_model_swap_rms=swap,independent_sample_max=error,gradients=gradients))
    save(a.root/'initial-contract.json',dict(passed=True,reports=reports,all_weights_random=True,model_updates=0,shared_initial_tensor_parity=True,
        initial_absolute_role_symmetry=True,scope='DirectionclaimisstructuralatINITIALIZATIONonly;trainedabsolute-roleembeddingsneednotbesymmetric.'))
    print('PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract'],required=True)
    for k in['root','source','parent']:p.add_argument('--'+k,type=Path)
    a=p.parse_args();prepare(a)if a.mode=='prepare'else contract(a)
