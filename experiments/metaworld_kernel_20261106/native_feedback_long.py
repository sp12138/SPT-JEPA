"""Finite paired scratch a_t-only learners, with frozen native-time feedback.

NN/data/training runtime is pinned independently of this orchestration script.
The forward model never supervises an action. Its own-data 20k cost is explicit.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from experiments.metaworld_contact_20261106.launch import environment, PYTHON, wait_gpu_release
from experiments.metaworld_contact_20261106.bridge_campaign import command
from experiments.metaworld_kernel_20261106.campaign import save, digest

BASE = Path('/efs/prescottsun/SPT-JEPA')


def prepare(a):
    a.root.mkdir(); (a.root/'results').mkdir(); recipes=[]
    old=json.loads((a.wait_parent/'plan.json').read_text())
    for gpu,seed,arm in [(4,3,'absolute'),(5,3,'directed'),(6,4,'absolute'),(7,4,'directed')]:
        run=a.root/'training'/f'{arm}-seed{seed}'; run.mkdir(parents=True)
        cfg=json.loads((a.parent/'training/all-executed-seed1/config.json').read_text())
        cfg.update(seed=seed,output=str(run),max_steps=120000,retain_steps=[2,600,40000,80000,120000])
        cfg['model']['goal_representation']=arm
        assert cfg['model']['action_dim']==4 and cfg['inverse_tail_reserve']==1
        save(run/'config.json',cfg)
        world=BASE/f'runs/metaworld_native_transport_20261106/training/stencil-seed{seed}'
        assert (world/'model-step0020000/weights.pt').exists()
        assert json.loads((world/'audit-20000.json').read_text())['passed']
        previous=next(r for r in old['recipes']if r['gpu']==gpu)
        save(run/'ORIGIN.json',dict(all_actor_weights_random=True,actor_target='ONLY current native a_t4D; no a_t+1 or later action labels.',
            actor_budget=120000,actor_lr_horizon=120000,world_run=str(world),world_steps=20000,
            world_labels='Own original-data future RGB ONLY; current executed control INPUT. Never any inverse teacher/strong pretrained model.',
            deployment='Real RGB every5native; 5 sequential OWN policy draws and4 OWN forward images. Also evaluate same frozen actor held5.',
            scope='Original corpus only. 2NEW seed pairs x2representations,not4independentinitializations. No reward/state/proprio/taskID/optimal-action labels. Not equalFLOPs/publicSOTA.'))
        recipes.append(dict(host='dev6',gpu=gpu,seed=seed,arm=arm,run=str(run),world=str(world),wait_output=previous['output'],config_sha256=digest(run/'config.json')))
    confirmation=json.loads((BASE/'runs/metaworld_confirm50_20261107/plan.json').read_text())
    save(a.root/'training-plan.json',dict(source=str(a.source),evaluator=str(a.evaluator),evaluator_sha256=digest(a.evaluator),recipes=recipes,
        stages=[2,600,40000,80000,120000],manifest=old['manifest'],deadline_unix=time.time()+36*3600,
        gate_roots=[str(a.wait_parent),str(BASE/'runs/metaworld_current_control50_20261107')],
        confirmation=dict(manifest=confirmation['manifest'],tasks=confirmation['tasks'],seeds=confirmation['seeds']),
        promotion_rule='Completed 50screen:world>=85% overall,>=5pp over SAME-NNstatic5draws,bothtrainedseedspositive;completed8taskworld6mani>=85%. Never promote on partial results.',
        scope='Four NEW scratch a_t4D learners=2pairedseeds3/4 xabsolute/directed. Own eligible forward20k same-seed parents separate/count cost.120k commonLR fromscratch.0futureactionlabels. Fixedsensor5/500anysuccess/samplelevel. Complete/collect/exit, no LLM monitor.'))


def contract(a):
    import numpy as np
    import torch
    from as_jepa_tools.native_inverse import NativeInversePairs
    from as_jepa_lewm.factory import build_model
    from experiments.metaworld_kernel_20261106.audit import equal
    torch.set_num_threads(2); plan=json.loads((a.root/'training-plan.json').read_text()); states={}; reports=[]
    for r in plan['recipes']:
        cfg=json.loads((Path(r['run'])/'config.json').read_text())
        assert not any(cfg.get(k)for k in ['initial_checkpoint','initial_model_checkpoint','initial_encoder_checkpoint'])
        data=NativeInversePairs(cfg);test=NativeInversePairs(cfg,'test')
        assert data.steps==1 and test.steps==1 and data.reserve==1
        assert not {v['episode']for v in data.records}&{v['episode']for v in test.records}
        for i in np.linspace(0,len(test)-1,32,dtype=int):
            value=test[int(i)];start=value['current_interval_start'];assert value['current_interval_stop']==start+1
            record=next(v for v in test.records if v['episode']*5==value['episode']);shard=test.shards[record['shard']];original=shard['controls']
            class Guard:
                def __getitem__(self,key):
                    assert isinstance(key,slice)and key.start==start and key.stop==start+1,'Non-current native action accessed'
                    return original[key]
            shard['controls']=Guard()
            try:assert equal(value,test[int(i)])
            finally:shard['controls']=original
        torch.manual_seed(cfg['seed']);model=build_model(cfg)
        shared={k:v.clone()for k,v in model.state_dict().items()if k!='kind'and not k.startswith('goal_fusion.')}
        if cfg['seed']in states:assert equal(states[cfg['seed']],shared)
        else:states[cfg['seed']]=shared
        reports.append(dict(seed=cfg['seed'],arm=r['arm'],train_windows=len(data),test_windows=len(test),one_native_access_guards=32))
    save(a.root/'initial-contract.json',dict(passed=True,new_updates=0,all_actor_weights_random=True,shared_initial_CNN_decoder_exact=True,reports=reports,
        scope='No a_t+1 access. New representation pairs share all initialCNN/decoder tensors;kind shape/fusion differ. Actual native2 restore/backward/physical gates still required before long continuation.'))


def gate(root,plan):
    for name in plan['gate_roots']:
        p=Path(name)/'results/summary.json'
        if not p.exists():return None
        d=json.loads(p.read_text())
        assert not d['errors'],name
        expected=len(json.loads((Path(name)/'plan.json').read_text())['recipes'])
        if d['complete']!=expected:return None
    full=json.loads((Path(plan['gate_roots'][0])/'results/summary.json').read_text())
    screen=json.loads((Path(plan['gate_roots'][1])/'results/summary.json').read_text())
    f=next(s for s in full['scores']if s['arm']=='world-native4');w=next(s for s in screen['scores']if s['arm']=='world-native4');c=next(s for s in screen['scores']if s['arm']=='static-native4')
    def score(s):return sum(v['successes']for v in s['task_scores'].values())/s['episodes']
    p=json.loads((Path(plan['gate_roots'][1])/'plan.json').read_text());rows={}
    for r in p['recipes']:
        v=json.loads((Path(r['output'])/'pilot/summary.json').read_text());assert v['passed']and v['model_unchanged']
        rows[(r['arm'],r['seed'])]={(x['task'],x['seed']):x for x in v['rows']}
    deltas=[]
    for seed in [1,2]:
        x=rows[('world-native4',seed)];y=rows[('static-native4',seed)];assert x.keys()==y.keys()
        assert all(x[k]['case_sha256']==y[k]['case_sha256']for k in x)
        deltas.append(sum(int(x[k]['success'])-int(y[k]['success'])for k in x)/len(x))
    passed=score(w)>=.85 and score(w)-score(c)>=.05-1e-12 and min(deltas)>0 and f['manipulation_percent']>=85
    return dict(passed=passed,world_screen_percent=100*score(w),static_screen_percent=100*score(c),world_full6_percent=f['manipulation_percent'],seed_delta_pp=[100*x for x in deltas],new_updates=0)


def work(a):
    plan=json.loads((a.root/'training-plan.json').read_text());r=next(v for v in plan['recipes']if v['gpu']==a.gpu);run=Path(r['run']);source=Path(plan['source']);deadline=plan['deadline_unix']
    try:
        while not(Path(r['wait_output'])/'complete.json').exists():
            assert not(Path(r['wait_output'])/'execution-error.json').exists();assert time.time()<deadline;time.sleep(5)
        wait_gpu_release(a.gpu)
        assert json.loads((a.root/'initial-contract.json').read_text())['passed']
        decision=gate(a.root,plan)
        while decision is None:assert time.time()<deadline;time.sleep(10);decision=gate(a.root,plan)
        save(run/'PROMOTION.json',decision)
        if not decision['passed']:
            save(run/'complete.json',dict(passed=True,trained=False,step=0,state='NO_PROMOTION',new_updates=0));return
        assert digest(run/'config.json')==r['config_sha256']and digest(Path(plan['evaluator']))==plan['evaluator_sha256']
        env=environment(source,a.gpu)
        for step in plan['stages']:
            assert time.time()<deadline and not(a.root/'STOP').exists()and not(run/'STOP').exists();wait_gpu_release(a.gpu)
            command([PYTHON,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),'--config',str(run/'config.json'),'--stop-step',str(step)]+(['--resume']if step>2 else[]),run/f'train-{step}.log',env,source,deadline)
            command([PYTHON,'-B','-u',str(source/'experiments/metaworld_contact_20261106/audit_bridge.py'),'--run',str(run),'--step',str(step),'--output',str(run/f'audit-{step}.json')],run/f'audit-{step}.log',env,source,deadline)
            assert json.loads((run/f'audit-{step}.json').read_text())['passed']
            archive=run/f'archive-step{step:07d}';archive.mkdir();shutil.copy2(run/'last.ckpt',archive/'native.ckpt');save(archive/'manifest.json',dict(step=step,native_sha256=digest(archive/'native.ckpt')))
            if step==600:continue
            base=[PYTHON,'-B','-u',plan['evaluator'],'--run',str(run),'--step',str(step),'--mode','world','--world',r['world'],'--manifest',plan['manifest']]
            endpoint=run/f'eval-step{step:07d}'
            tail=['--budget','20','--tasks','bin-picking-v3','--seeds','42']if step==2 else['--seeds']+list(map(str,list(range(42,52))+list(range(1042,1052))))
            command(base+['--output',str(endpoint)]+tail,endpoint.with_suffix('.log'),env,source,deadline)
            if step>=40000:
                command([PYTHON,'-B','-u',str(source/'experiments/metaworld_kernel_20261106/inverse_evaluate.py'),'--run',str(run),'--step',str(step),'--manifest',plan['manifest'],'--execute','1','--output',str(run/f'hold-step{step:07d}'),'--seeds']+list(map(str,list(range(42,52))+list(range(1042,1052)))),run/f'hold-{step}.log',env,source,deadline)
            save(run/'progress.json',dict(step=step,phase='stage_complete'))
        confirmation=plan['confirmation'];endpoint=run/'confirm-final'
        command([PYTHON,'-B','-u',plan['evaluator'],'--run',str(run),'--step',str(step),'--mode','world','--world',r['world'],'--manifest',confirmation['manifest'],'--output',str(endpoint),'--tasks']+confirmation['tasks']+['--seeds']+list(map(str,confirmation['seeds'])),run/'confirm-final.log',env,source,deadline)
        proof=json.loads((endpoint/'summary.json').read_text());assert proof['passed']and proof['model_unchanged']and len(proof['rows'])==1000
        save(run/'complete.json',dict(passed=True,trained=True,step=step,current_native_action_only=True,world_pretrain_steps=20000,confirmation=True))
    except BaseException as e:save(run/'execution-error.json',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','contract','work'],required=True)
    for k in ['root','source','evaluator','parent','wait-parent']:p.add_argument('--'+k,type=Path)
    p.add_argument('--gpu',type=int);a=p.parse_args();globals()[a.mode](a)
