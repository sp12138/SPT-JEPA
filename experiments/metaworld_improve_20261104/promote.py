"""One finite,evidence-gated continuation to the original50-epoch LR horizon."""
import argparse
from collections import defaultdict
import copy
import datetime
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments.metaworld_eval_20261103.worker import read,save,digest


def observations(summary):
    rows=defaultdict(dict)
    for group in summary['groups']:
        if group['step']!=16480:continue
        for e in group['observations']:
            rows[group['arm']][(e['train_seed'],e['task'],e['seed'])]=float(e['success'])
    return rows


def screen(summary):
    data=observations(summary);reference=data['tf-control'];native=data['native-control'];keys=sorted(reference)
    assert len(keys)==320 and set(reference)==set(native)
    manip=[k for k in keys if k[1] not in ['reach-v3','reach-wall-v3']]
    def mean(d,subset):return 100*sum(d[k] for k in subset)/len(subset)
    rng=np.random.default_rng(20261104);reports=[]
    for arm,d in sorted(data.items()):
        if arm in ['tf-control','native-control']:continue
        assert set(d)==set(reference)
        cluster=defaultdict(list)
        for k in keys:cluster[(k[1],k[2])].append((d[k]-reference[k],d[k]-native[k]))
        vectors=np.asarray([np.mean(v,axis=0) for v in cluster.values()])
        samples=vectors[rng.integers(0,len(vectors),(4000,len(vectors)))].mean(1)*100
        ci=np.quantile(samples,[.025,.975],axis=0)
        gains={b:mean(d,[k for k in manip if (k[2]>=1000)==(b=='fresh')])-max(
            mean(reference,[k for k in manip if (k[2]>=1000)==(b=='fresh')]),
            mean(native,[k for k in manip if (k[2]>=1000)==(b=='fresh')])) for b in ['primary','fresh']}
        seed_gains={s:mean(d,[k for k in manip if k[0]==s])-max(
            mean(reference,[k for k in manip if k[0]==s]),mean(native,[k for k in manip if k[0]==s])) for s in [1,2]}
        supported=sum(all(any(d[k]>0 for k in manip if k[1]==task and k[0]==s) for s in [1,2])
                      for task in sorted({k[1] for k in manip}))
        score=mean(d,manip);gain=score-max(mean(reference,manip),mean(native,manip))
        checks=dict(manipulation_absolute_15pp=score>=15,manipulation_gain_10pp=gain>=10,
                    both_case_banks_positive=all(v>0 for v in gains.values()),
                    both_parent_seeds_positive=all(v>0 for v in seed_gains.values()),
                    at_least_two_manipulation_tasks_both_seeds=supported>=2,
                    overall_case_cluster_CI_positive_vs_both=bool((ci[0]>0).all()))
        reports.append(dict(arm=arm,qualified=all(checks.values()),checks=checks,
            manipulation_mean=score,manipulation_gain=gain,overall_mean=mean(d,keys),
            bank_manipulation_gains=gains,parent_seed_manipulation_gains=seed_gains,
            supported_manipulation_tasks=supported,overall_difference_CI_vs_tf=ci[:,0].tolist(),
            overall_difference_CI_vs_native=ci[:,1].tolist()))
    selected=sorted((r for r in reports if r['qualified']),key=lambda r:(r['manipulation_gain'],r['overall_mean']),reverse=True)[:2]
    return dict(passed=True,selected=[r['arm'] for r in selected],candidates=reports,
        scope='Predeclared exploratory promotion,not a final high-performance claim. CI conditional on2 trained parents;'
              'case clustering preserves shared cases across learners. No new simulator/learner labels.')


def build_long(root,previous,source,helper,selection):
    from as_jepa_lewm.factory import model_target
    assert selection['selected'] and not root.exists();root.mkdir();(root/'workers').mkdir()
    parent_plan=read(previous/'plan.json');parent_map={(r['arm'],r['seed']):r for r in parent_plan['recipes']}
    repetitions={'tf-control':2,'native-control':2}
    repetitions.update({a:(2 if len(selection['selected'])==2 else 4) for a in selection['selected']})
    recipes=[];jobs=[];tasks=sorted({j['task'] for j in read(previous/'evaluation-plan.json')['jobs']})
    for parent_seed,host in [(1,'dev13'),(2,'dev15')]:
        gpu=0
        for arm,count in repetitions.items():
            parent=Path(parent_map[(arm,parent_seed)]['run']);original=read(parent/'config.json')
            assert read(parent/'complete.json')['passed']
            archive=parent/'archive-step0016480';manifest=read(archive/'manifest.json')
            assert all(digest(archive/k)==v for k,v in manifest['files'].items())
            for repeat in range(count):
                seed=20261104+parent_seed*1009+repeat*100003
                name=f'{arm}-parent{parent_seed}-rng{repeat}';run=root/'training'/name;run.mkdir(parents=True)
                cfg=copy.deepcopy(original);cfg.update(output=str(run),seed=seed,retain_steps=[16480,16482,24720,41200])
                if 'forecast_seed' in cfg['model']:cfg['model']['forecast_seed']=20261029+seed*100003
                save(run/'config.json',cfg)
                shutil.copytree(parent/'initial-model',run/'initial-model',copy_function=os.link)
                shutil.copytree(archive,run/'inherited-archive-step0016480',copy_function=os.link)
                entry=run/'model-step0016480';entry.mkdir()
                os.link(parent/'model-step0016480/weights.pt',entry/'weights.pt')
                save(entry/'config.json',dict(_target_=model_target(cfg),**cfg['model']))
                os.link(archive/'native.ckpt',run/'last.ckpt')
                save(run/'FORK.json',dict(passed=True,parent=str(parent),parent_config=str(parent/'config.json'),
                    parent_proof=dict(native_sha256=manifest['files']['native.ckpt']),entry_step=16480,
                    continuation_rng_seed=seed,parent_seed=parent_seed,repeat=repeat,
                    scope='Randomized continuation from2 trained parents;not16 independent initializations. '
                          'Native full state preserved;only future RNG and sampler seed change under paired seed management.'))
                recipe=dict(name=name,arm=arm,seed=seed,parent_seed=parent_seed,repeat=repeat,host=host,gpu=gpu,
                            run=str(run),config_sha256=digest(run/'config.json'))
                recipes.append(recipe);gpu+=1
                for step,banks,limit in [(24720,['primary'],5),(41200,['primary','fresh'],10)]:
                    for task in tasks:
                        for bank in banks:
                            job_name=f'step{step:07d}-{name}-{task}-{bank}'
                            jobs.append(dict(name=job_name,run=str(run),model=name,arm=arm,seed=seed,parent_seed=parent_seed,
                                repeat=repeat,step=step,checkpoint=str(run/f'model-step{step:07d}'),task=task,
                                bank=bank,limit=limit,work=str(root/'evaluation'/job_name)))
        assert gpu==8
    assert len(recipes)==16 and len(jobs)==384
    save(root/'plan.json',dict(recipes=recipes,entry_step=16480,stages=[16482,24720,41200],
        selected=selection['selected'],scope='Conditional50-epoch continuation;two parent learners crossed with continuation RNG repeats. '
        'Controls receive identical total updates,original41200 LR horizon and matched future sampler seeds.'))
    save(root/'evaluation-plan.json',dict(jobs=jobs));save(root/'SELECTION.json',selection)
    dispatch=dict(read(previous/'dispatch.json'),previous_root=str(previous),plan_path=str(root/'plan.json'),
                  runtime=str(source),helper=str(helper))
    save(root/'dispatch.json',dispatch)
    save(root/'TRAINING_APPROVED.json',dict(passed=True,config_sha256={r['name']:r['config_sha256'] for r in recipes},
        scope='Scientific promotion approved;actual RNG-fork identity/full restore/backward/native2update/planning gates remain mandatory.'))
    save(root/'PREPARED.json',dict(passed=True,selected=selection['selected'],training_runs=16,
         scope='Published last;host-resident finite starters require no SSH aliases or credentials inside GPU hosts.'))
    while time.time()<dispatch['deadline_unix']:
        if all((root/f'launch-{h}.json').exists() for h in ['dev13','dev15']):break
        time.sleep(5)
    assert all((root/f'launch-{h}.json').exists() for h in ['dev13','dev15'])
    python='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    command=[python,'-B','-u',str(helper/'experiments/metaworld_improve_20261104/campaign.py'),'--mode','finish',
             '--root',str(root),'--source',str(source),'--helper',str(helper)]
    env=os.environ.copy();env.update(PYTHONPATH=str(helper)+':'+str(source),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    with (root/'completion.log').open('x') as log:
        p=subprocess.Popen(command,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(root/'completion-launch.json',dict(pid=p.pid,host='dev13',command=command))


def main(previous,root,source,helper):
    deadline=read(previous/'dispatch.json')['deadline_unix'];out=previous/'long-promotion'
    out.mkdir(exist_ok=False)
    save(out/'RULES.json',dict(deadline_unix=deadline,
         gates=['six-manipulation mean>=15%','gain>=10pp over both matched controls',
                'positive manipulation gains in both case banks and both parent seeds',
                '>=2 manipulation tasks succeed in both seeds','overall conditional case-cluster CI>0 vs both controls'],
         maximum_selected=2,long_global_cap=41200,allowed_hosts=['dev13','dev15'],
         scope='One conditional continuation,not a recurring LLM monitor or an unbounded next research round. '
               'If no qualified arm,retain all negatives and do not spend the long budget.'))
    while time.time()<deadline:
        if (previous/'results/summary.json').exists():break
        if (previous/'STOP').exists():return
        time.sleep(30)
    if not (previous/'results/summary.json').exists():
        save(out/'NO_PROMOTION.json',dict(reason='Original absolute deadline reached'));return
    summary=read(previous/'results/summary.json')
    if not summary['passed']:
        save(out/'NO_PROMOTION.json',dict(reason='Unresolved screening execution failures'));return
    result=screen(summary);save(out/'SELECTION.json',result)
    if not result['selected']:
        save(out/'NO_PROMOTION.json',dict(reason='No arm passed prespecified manipulation/replication gates'))
        return
    build_long(root,previous,source,helper,result)
    save(out/'LAUNCHED.json',dict(root=str(root),selected=result['selected'],launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def start_host(previous,root,source,helper,host):
    from experiments.metaworld_improve_20261104.campaign import launch
    deadline=read(previous/'dispatch.json')['deadline_unix']
    while time.time()<deadline:
        if (previous/'long-promotion/NO_PROMOTION.json').exists() or (previous/'STOP').exists():return
        if (root/'PREPARED.json').exists():
            assert read(root/'PREPARED.json')['passed']
            launch(root,source,helper,host,'/efs/prescottsun/project_jepa/envs/lewm/bin/python');return
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['previous','root','source','helper']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--mode',choices=['select','start-host'],default='select')
    parser.add_argument('--host',choices=['dev13','dev15']);args=parser.parse_args()
    if args.mode=='select':main(Path(args.previous),Path(args.root),Path(args.source),Path(args.helper))
    else:start_host(Path(args.previous),Path(args.root),Path(args.source),Path(args.helper),args.host)
