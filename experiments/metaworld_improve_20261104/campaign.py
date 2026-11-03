"""A finite,16-GPU matched adaptation round using the existing native trainer."""
import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

from experiments.metaworld_eval_20261103.worker import read,save,digest,command,archive,free_gpu


ARMS=[
    ('tf-control',{}),
    ('native-control',{}),
    ('high-noise',{'flow_logit_shift':math.log(4.)}),
    ('zero-anchor',{'renderer_anchor_mode':'zero'}),
    ('forecast-only',{'forecast_fraction':1.}),
    ('quadrants',{'pool_partition':'quadrants'}),
    ('shuffled-control',{'pool_partition':'shuffled'}),
    ('innovation',{}),
]


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(root,previous,source,helper):
    assert not root.exists();root.mkdir();(root/'workers').mkdir()
    old_plan=read(previous/'plan.json')['recipes']
    parents={(r['arm'],r['seed']):Path(r['run']) for r in old_plan}
    old_dispatch=read(previous/'dispatch.json');recipes=[];jobs=[];parent_proofs={}
    for arm,seed in [('temporal-forecast',1),('temporal-forecast',2),('native-lewm',1),('native-lewm',2)]:
        parent=parents[(arm,seed)];assert read(parent/'complete.json')['passed']
        path=parent/'archive-step0008240';manifest=read(path/'manifest.json')
        assert all(digest(path/name)==sha for name,sha in manifest['files'].items())
        parent_proofs[str(parent)]=dict(native_sha256=manifest['files']['native.ckpt'],
            portable_sha256=digest(parent/'model-step0008240/weights.pt'),manifest_sha256=digest(path/'manifest.json'))
    tasks=sorted({j['task'] for j in read(previous/'evaluation-plan.json')['jobs']})
    assert len(tasks)==8
    for gpu,(arm,changes) in enumerate(ARMS):
        for seed in [1,2]:
            parent=parents[('native-lewm' if arm=='native-control' else 'temporal-forecast',seed)]
            cfg=read(parent/'config.json');parent_cfg=copy.deepcopy(cfg)
            name=f'{arm}-seed{seed}';run=root/'training'/name;run.mkdir(parents=True)
            cfg['output']=str(run);cfg['retain_steps']=[8240,8242,9888,16480]
            cfg['model'].update(changes)
            if arm=='innovation':cfg['model_class']='innovation'
            save(run/'config.json',cfg)
            shutil.copytree(parent/'initial-model',run/'initial-model',copy_function=os.link)
            shutil.copytree(parent/'archive-step0008240',run/'inherited-archive-step0008240',copy_function=os.link)
            entry=run/'model-step0008240';entry.mkdir()
            os.link(parent/'model-step0008240/weights.pt',entry/'weights.pt')
            from as_jepa_lewm.factory import model_target
            save(entry/'config.json',dict(_target_=model_target(cfg),**cfg['model']))
            os.link(parent/'archive-step0008240/native.ckpt',run/'last.ckpt')
            proof=dict(passed=True,parent=str(parent),parent_config=str(parent/'config.json'),
                parent_proof=parent_proofs[str(parent)],entry_step=8240,entry_samples=8240*128,
                changed_config_keys=[k for k in cfg if cfg[k]!=parent_cfg[k]],model_intervention=changes,
                model_class=cfg['model_class'],created_utc=now(),
                scope='Adaptation from the trained same-seed parent,not a random initialization. '
                      'Full model/AdamW/scheduler/RNG/sample cursor inherited;new actual entry audit gates SGD. '
                      'Initial-model is inherited original random initialization,not the adaptation start.')
            save(run/'FORK.json',proof)
            recipe=dict(name=name,arm=arm,seed=seed,run=str(run),host='dev13' if seed==1 else 'dev15',
                        gpu=gpu,parent=str(parent),config_sha256=digest(run/'config.json'))
            recipes.append(recipe)
            for step,banks,limit in [(9888,['primary'],5),(16480,['primary','fresh'],10)]:
                for task in tasks:
                    for bank in banks:
                        job_name=f'step{step:07d}-{name}-{task}-{bank}'
                        jobs.append(dict(name=job_name,run=str(run),model=name,arm=arm,seed=seed,step=step,
                            checkpoint=str(run/f'model-step{step:07d}'),task=task,bank=bank,limit=limit,
                            work=str(root/'evaluation'/job_name)))
    save(root/'plan.json',dict(created_utc=now(),recipes=recipes,arms=ARMS,entry_step=8240,
         stages=[8242,9888,16480],lr_horizon=41200,cap=16480,
         scope='Explicit user-requested MetaWorld improvement round. Six sample-level interventions,matched TF and native controls;'
               'two same original seeds each. Native retains BN/SIGReg;not a sample-level candidate or equal-capacity comparator.'))
    save(root/'evaluation-plan.json',dict(jobs=jobs))
    dispatch=dict(old_dispatch,source=old_dispatch['source'],helper=str(helper),runtime=str(source),
                  previous_root=str(previous),plan_path=str(root/'plan.json'),allowed_hosts=['dev13','dev15'],
                  scope='16 matched continuations;8240 added updates maximum,unchanged41200 LR horizon;384 MPC endpoints. '
                        'Same cases/controller/input/budget;no new learner labels,automatic extension,next round or recurring LLM monitor.')
    save(root/'dispatch.json',dispatch)
    save(root/'TRAINING_APPROVED.json',dict(passed=True,config_sha256={r['name']:r['config_sha256'] for r in recipes},
         scope='Configuration and parent archive approvals only;each worker must pass actual restore/backward,two native updates and planning preflight before long training.'))
    save(root/'PREPARED.json',dict(passed=True,created_utc=now(),recipes=16,evaluation_endpoints=len(jobs),
         episodes=sum(j['limit'] for j in jobs),parent_proofs=parent_proofs,previous_root=str(previous)))
    assert len(jobs)==384
    print(dict(prepared=True,recipes=16,endpoints=len(jobs),root=str(root)),flush=True)


def environments(root,source,helper,gpu,evaluate=False):
    dispatch=read(root/'dispatch.json');base=Path(dispatch['base'])
    paths=[helper,source,base/'environments/completion-overlay',base/'environments/overlay',
           base/'runs/mechanism_cubeaudit_20261030/source-v1/runtime',base/'sources/stable-worldmodel',Path(dispatch['source'])]
    if evaluate:paths.insert(0,base/'environments/metaworld-sanity-20261103/lib/python3.12/site-packages')
    env=os.environ.copy();env.update(PYTHONPATH=':'.join(map(str,paths)),CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if evaluate:env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),
        LD_LIBRARY_PATH=str(base/'environments/egl-lib/usr/lib/x86_64-linux-gnu')+':'+env.get('LD_LIBRARY_PATH',''),
        __EGL_VENDOR_LIBRARY_FILENAMES=str(base/'environments/egl-lib/10_nvidia.json'))
    return env


def work(root,source,helper,host,gpu,python):
    assert host in ['dev13','dev15'];dispatch=read(root/'dispatch.json');deadline=dispatch['deadline_unix']
    previous=Path(dispatch['previous_root'])
    # Existing MPC jobs keep their GPU until their own finite campaign ends.
    while not (previous/'results/summary.json').exists():
        if time.time()>=deadline:raise TimeoutError('Waiting for original baseline completion')
        time.sleep(10)
    assert read(previous/'results/summary.json')['passed']
    import psutil
    if (previous/f'launch-{host}.json').exists():old=read(previous/f'launch-{host}.json')['jobs']
    else:old=read(previous/f'activation-{host}.json')['jobs'] if gpu<2 else read(previous/f'diagnostic-activation-v3-{host}.json')['jobs']
    owner=next(r for r in old if r['gpu']==gpu)
    while time.time()<deadline:
        try:
            p=psutil.Process(owner['pid']);alive=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE
            owned=str(previous) in p.cmdline() and any(x.endswith('/worker.py') for x in p.cmdline())
        except psutil.NoSuchProcess:alive=owned=False
        if not alive or not owned:break
        time.sleep(5)
    free_gpu(gpu)
    recipe=next(r for r in read(root/'plan.json')['recipes'] if r['host']==host and r['gpu']==gpu)
    run=Path(recipe['run']);proof=read(run/'FORK.json')
    assert digest(run/'config.json')==recipe['config_sha256']
    assert digest(run/'last.ckpt')==proof['parent_proof']['native_sha256']
    env=environments(root,source,helper,gpu)
    plan=read(root/'plan.json');entry=plan.get('entry_step',8240);stages=plan['stages']
    if proof.get('continuation_rng_seed') is not None:
        command([python,'-B','-u',str(helper/'experiments/metaworld_improve_20261104/campaign.py'),
                 '--mode','reseed','--root',str(root),'--source',str(source),'--helper',str(helper),
                 '--host',host,'--gpu',str(gpu)],run/'rng-fork.log',env,source,deadline,cap=1800)
    def audit(step,reference=False):
        cmd=[python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/audit.py'),
             '--run',str(run),'--step',str(step),'--output',str(run/f'audit-step{step}.json')]
        if reference:cmd+=['--reference-config',proof['parent_config']]
        command(cmd,run/f'audit-step{step}.log',env,source,deadline,cap=1800)
        assert read(run/f'audit-step{step}.json')['passed']
    audit(entry,True)
    for step in stages:
        if (root/'STOP').exists() or (run/'STOP').exists():raise RuntimeError('User stop marker')
        free_gpu(gpu)
        command([python,'-B','-u',str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'),
                 '--config',str(run/'config.json'),'--stop-step',str(step),'--resume'],
                 run/f'train-stage{step}.log',env,source,deadline)
        audit(step);archive(run,step)
        if step==stages[0]:
            command([python,'-B','-u',str(helper/'experiments/metaworld_eval_20261103/evaluate.py'),
                     '--checkpoint',str(run/f'model-step{step:07d}'),'--data',dispatch['data'],
                     '--manifest',dispatch['case_manifest'],'--output',str(run/'planning-preflight'),
                     '--task','reach-wall-v3','--bank','primary','--limit','1','--budget','20','--success-mode','any'],
                     run/'planning-preflight.log',environments(root,source,helper,gpu,True),source,deadline,cap=900)
            result=read(run/'planning-preflight/metrics.json')
            assert result['passed'] and result['model_state_before']==result['model_state_after']
            save(run/'PREFLIGHT_PASSED.json',dict(passed=True,step=step,created_utc=now(),
                scope='Actual full restore,2 native updates,FP32 backward and20-step MPC interface only;not task-performance evidence.'))
    save(run/'complete.json',dict(passed=True,step=stages[-1],finished_utc=now()))
    from experiments.metaworld_eval_20261103.worker import main as evaluate
    evaluate(SimpleNamespace(root=str(root),source=str(source),helper=str(helper),base=dispatch['base'],
                            mode='evaluate',python=python,gpu=gpu,host=host,run=None))


def reseed(root,host,gpu):
    # A separate process owns this CUDA generator;its context exits before
    # native training. The operational parent never keeps GPU memory resident.
    import random
    import numpy as np
    import torch
    from experiments.metaworld_eval_20261103.audit import equal
    recipe=next(r for r in read(root/'plan.json')['recipes'] if r['host']==host and r['gpu']==gpu)
    run=Path(recipe['run']);proof=read(run/'FORK.json');seed=proof['continuation_rng_seed']
    checkpoint=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    checkpoint['campaign_rng']=dict(python=random.Random(seed).getstate(),
        numpy=np.random.RandomState(seed%(2**32)).get_state(),
        torch=torch.Generator(device='cpu').manual_seed(seed).get_state(),
        cuda=[torch.Generator(device='cuda').manual_seed(seed).get_state()])
    temporary=run/'rng-fork.ckpt.tmp'
    with temporary.open('xb') as file:
        torch.save(checkpoint,file);file.flush();os.fsync(file.fileno())
    os.replace(temporary,run/'last.ckpt')
    parent=torch.load(Path(proof['parent'])/'archive-step0016480/native.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint.keys()==parent.keys()
    unchanged={k:equal(v,parent[k]) for k,v in checkpoint.items() if k!='campaign_rng'}
    assert all(unchanged.values())
    save(run/'RNG_FORK.json',dict(passed=True,seed=seed,unchanged_payload_fields=unchanged,
        changed_payload_fields=['campaign_rng'],checkpoint_sha256=digest(run/'last.ckpt'),
        scope='Independent continuation randomness only. Same2 trained parents,not4 independent pretrainings. '
              'Model/AdamW/scheduler/global_step/sample cursor exactly preserved;future sampler seed is explicit in config.'))


def launch(root,source,helper,host,python):
    assert host in ['dev13','dev15'];result=root/f'launch-{host}.json';assert not result.exists()
    deadline=read(root/'dispatch.json')['deadline_unix'];rows=[]
    env=os.environ.copy();env.update(PYTHONPATH=str(helper)+':'+str(source),PYTHONDONTWRITEBYTECODE='1')
    for gpu in range(8):
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(deadline-time.time()))),python,
             '-B','-u',str(helper/'experiments/metaworld_improve_20261104/campaign.py'),
             '--mode','work','--root',str(root),'--source',str(source),'--helper',str(helper),
             '--host',host,'--gpu',str(gpu),'--python',python]
        with (root/'workers'/f'{host}-g{gpu}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(gpu=gpu,pid=p.pid,command=cmd))
    save(result,dict(passed=True,host=host,launched_utc=now(),jobs=rows,
                    scope='One finite owner perGPU;wait original baseline collection/owned actors,then use same slot.'))
    print(dict(host=host,pids=[r['pid'] for r in rows]),flush=True)


def finish(root):
    from collections import Counter,defaultdict
    dispatch=read(root/'dispatch.json');recipes=read(root/'plan.json')['recipes'];jobs=read(root/'evaluation-plan.json')['jobs']
    while time.time()<dispatch['deadline_unix']:
        training_done=all(any((Path(r['run'])/n).exists() for n in ['complete.json','worker-failed.json']) for r in recipes)
        for j in jobs:
            run=Path(j['run']);work=Path(j['work'])
            if (run/'worker-failed.json').exists() and not (run/f'audit-step{j["step"]}.json').exists():
                if not any((work/n).exists() for n in ['blocked.json','failed.json','complete.json']):save(work/'blocked.json',dict(executed=False,reason='Training failed',run=str(run)))
        if training_done and all(any((Path(j['work'])/n).exists() for n in ['complete.json','failed.json','blocked.json']) for j in jobs):break
        if (root/'STOP').exists():break
        time.sleep(20)
    counts=Counter();outcomes=[];groups=defaultdict(list);archives=[]
    for r in recipes:
        run=Path(r['run']);counts['training_complete' if (run/'complete.json').exists() else 'training_failed' if (run/'worker-failed.json').exists() else 'training_unfinished']+=1
        for p in run.glob('archive-step*/manifest.json'):
            d=read(p);checks={k:digest(p.parent/k)==v for k,v in d['files'].items()};archives.append(dict(path=str(p),passed=all(checks.values())))
    for j in jobs:
        work=Path(j['work']);state=next((s for s in ['complete','failed','blocked'] if (work/(s+'.json')).exists()),'unfinished')
        counts['evaluation_'+state]+=1;outcomes.append(dict(**j,state=state))
        if state!='complete':continue
        metric=read(work/'output/metrics.json');assert digest(work/'output/metrics.json')==read(work/'complete.json')['metrics_sha256']
        assert metric['passed'] and metric['model_state_before']==metric['model_state_after']
        for e in metric['episodes']:groups[(j['step'],j['arm'],j['bank'],j['task'])].append(
            dict(train_seed=j['seed'],parent_seed=j.get('parent_seed',j['seed']),repeat=j.get('repeat'),**e))
    rows=[]
    for key,episodes in sorted(groups.items()):
        valid=[e for e in episodes if e['expert_goal_success']]
        rows.append(dict(step=key[0],arm=key[1],bank=key[2],task=key[3],episodes=len(episodes),
            training_seeds=sorted({e['train_seed'] for e in episodes}),success_rate=100*sum(e['success'] for e in episodes)/len(episodes),
            parent_seeds=sorted({e['parent_seed'] for e in episodes}),
            goal_valid_episodes=len(valid),goal_valid_success_rate=100*sum(e['success'] for e in valid)/len(valid) if valid else None,observations=episodes))
    passed=counts['training_complete']==len(recipes) and counts['evaluation_complete']==len(jobs) and len(archives)==3*len(recipes) and all(a['passed'] for a in archives)
    plan=read(root/'plan.json')
    save(root/'results/summary.json',dict(passed=passed,closed_utc=now(),counts=dict(counts),groups=rows,outcomes=outcomes,archives=archives,
         entry_step=plan['entry_step'],global_cap=plan['stages'][-1],added_updates=plan['stages'][-1]-plan['entry_step'],
         cohort_scope=plan['scope'],
         scope='Matched continuations. Same RGB/action data,cases,CEM,500-step any-success. '
               'Candidate objectives sample-level;native BN/SIGReg baseline explicit. Not canonicalMT50,published reproduction,equalcapacity/FLOPs or novelty evidence.'))
    save(root/'HOLD',dict(closed=True,passed=passed,reason='Finite round ended;no automatic next round'))
    print(dict(passed=passed,counts=dict(counts)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','launch','work','finish','reseed'],required=True)
    for key in ['root','source','helper']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--previous');parser.add_argument('--host',choices=['dev13','dev15']);parser.add_argument('--gpu',type=int)
    parser.add_argument('--python',default='/efs/prescottsun/project_jepa/envs/lewm/bin/python');args=parser.parse_args()
    root,source,helper=Path(args.root),Path(args.source),Path(args.helper)
    try:
        if args.mode=='prepare':prepare(root,Path(args.previous),source,helper)
        elif args.mode=='launch':launch(root,source,helper,args.host,args.python)
        elif args.mode=='finish':finish(root)
        elif args.mode=='reseed':reseed(root,args.host,args.gpu)
        else:work(root,source,helper,args.host,args.gpu,args.python)
    except Exception:
        if args.mode=='work':
            recipe=next(r for r in read(root/'plan.json')['recipes'] if r['host']==args.host and r['gpu']==args.gpu)
            save(Path(recipe['run'])/'worker-failed.json',dict(passed=False,finished_utc=now(),traceback=traceback.format_exc()))
        raise
