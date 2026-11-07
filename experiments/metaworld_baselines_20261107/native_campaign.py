"""Formal native-window coverage benchmark; valid window8 parents are inherited."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from experiments.metaworld_baselines_20261107.campaign import BASE,PYTHON,env,execute,save,sha,now
from experiments.metaworld_baselines_20261107.safe_metadata import read

ROOT=BASE/'runs/metaworld_baselines_native_20261108'
LEGACY=BASE/'runs/metaworld_baselines_20261107'
HOSTS=['dev13','dev15','dev6']
SHORT=['lewm','smwm','ac-mtm','lpwm']


def alive(pid):
    try:return 'metaworld_baselines_20261107' in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except OSError:return False


def prepare(helper):
    import numpy as np
    old=read(LEGACY/'plan.json');ROOT.mkdir(parents=True,exist_ok=False)
    data=BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'
    lengths=np.load(data/'lengths.npy');ids=np.load(data/'episode_ids.npy');training=np.load(data/'train_episodes.npy')
    lengths=lengths[np.isin(ids,training)]
    counts={n:int(np.maximum(lengths-n+1,0).sum()) for n in [2,4,8]}
    assert counts=={2:172608,4:150109,8:105486}
    jobs=[];owners=[]
    for item in old['jobs']:
        if item['method'] not in ['lewm-idm','h-jepa']:continue
        job=dict(item,kind='continue' if item['seed']==1 else 'external',runtime=str(
            BASE/'reports/metaworld_baselines_20261107/helper-panel-v4'),steps_per_epoch=412,
            epoch_steps={'20':8240,'50':20600,'100':41200})
        jobs.append(job)
        if job['kind']=='external':
            claim=read(LEGACY/'claims-v3'/(job['name']+'.json'))
            if job['method']=='h-jepa':
                owner=next(x for x in read(LEGACY/f'launch-device-v4-{claim["host"]}.json')['workers'] if x['name']==job['name'])
                owners.append(dict(name=job['name'],host=owner['host'],pair=owner['pair'],pid=owner['pid']))
            else:owners.append(dict(name=job['name'],host=claim['host'],pair=claim['pair'],pid=claim['pid']))
    for seed in [1,2,3]:
        for method in SHORT:
            frames=2 if method=='smwm' else 4
            natural=counts[frames];per_epoch=natural//256;steps=per_epoch*100
            name=f'{method}-seed{seed}';out=ROOT/'training'/name;out.mkdir(parents=True)
            cfg=dict(method=method,seed=seed,sources=str(BASE/'sources/metaworld-baselines-20261107'),output=str(out),
                data=str(data),steps=steps,global_batch=256,natural_windows=natural,steps_per_epoch=per_epoch,
                epochs=100,dataset_protocol='native_window_all_valid_starts',
                milestones=[3,100,per_epoch*20,per_epoch*50,steps],source_catalog_sha256=sha(LEGACY/'sources-v3.json'),
                scratch=True,inputs=['pixels','factual_executed_action'],reward=False,privileged_state=False,
                optimal_action_labels=False,inverse_target='same observed completed-transition forward input only')
            save(out/'config.json',cfg)
            jobs.append(dict(name=name,method=method,seed=seed,kind='fresh',config=str(out/'config.json'),output=str(out),
                runtime=str(helper),steps_per_epoch=per_epoch,epoch_steps={str(e):per_epoch*e for e in [20,50,100]}))
    for label in ['technical-0','intermediate-0','intermediate-1','final-0','final-1']:
        save(ROOT/'manifests'/(label+'.json'),read(LEGACY/'manifests'/(label+'.json')))
    save(ROOT/'plan.json',dict(jobs=jobs,legacy_owners=owners,hosts=HOSTS,helper=str(helper),deadline=old['deadline'],
        native_window_counts={str(k):v for k,v in counts.items()},source_manifest=LEGACY.as_posix()+'/sources-v3.json',
        scope='Formal native-window/full-trajectory coverage.12fresh scratch models;6 valid window8 learners continue/reuse. '
            'Each method trains100native epochs;updates/images/FLOPs differ and are disclosed. SameRGB/action data,split,casebank andCEM.'))
    save(LEGACY/'FORMAL_NATIVE_PROTOCOL.json',dict(root=str(ROOT),reason='Short native windows must include terminal/short-trajectory starts',
        prefix_pilot_methods=SHORT,valid_inherited_methods=['lewm-idm','h-jepa'],utc=now()))
    for job in old['jobs']:
        if job['method'] not in SHORT:continue
        out=Path(job['output']);save(out/'RETIRED_PROTOCOL.json',dict(reason='Biased common8prefix pilot;not an algorithm negative',formal_root=str(ROOT),utc=now()))
        claim=LEGACY/'claims-v3'/(job['name']+'.json')
        if not claim.exists():
            with claim.open('x') as stream:json.dump(dict(status='retired_protocol_pilot',host='none',pair=-1,pid=0),stream)
    print(json.dumps(dict(root=str(ROOT),fresh=12,inherited=6,counts=counts)))


def busy(host,pair):
    for owner in read(ROOT/'plan.json')['legacy_owners']:
        if owner['host']==host and owner['pair']==pair and alive(owner['pid']):return True
    for gpu in [pair*2,pair*2+1]:
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if any(v.strip().isdigit() for v in s.splitlines()):return True
    return False


def claim(host,pair):
    for job in read(ROOT/'plan.json')['jobs']:
        if job['kind']=='external':continue
        f=ROOT/'claims'/(job['name']+'.json');f.parent.mkdir(exist_ok=True)
        try:
            with f.open('x') as stream:json.dump(dict(host=host,pair=pair,pid=os.getpid(),utc=now()),stream)
        except FileExistsError:continue
        return job
    return None


def run_train(job,stop,previous,gpus,deadline,transfer=None):
    h=Path(job['runtime']);out=Path(job['output'])
    cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(h/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',job['config'],'--stop',str(stop)]
    if transfer:cmd+=['--resume',str(transfer),'--allow-output-transfer']
    elif previous:cmd+=['--resume',str(out/f'checkpoint-{previous:07d}.pt')]
    execute(cmd,out/f'train-{stop:07d}.log',env(h,gpus),h,deadline)


def audit(job,step,gpu,deadline):
    h=Path(job['runtime']);out=Path(job['output'])
    execute([PYTHON,str(h/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',job['config'],
        '--resume',str(out/f'checkpoint-{step:07d}.pt'),'--audit'],out/f'audit-{step:07d}.log',env(h,[gpu]),h,deadline)


def evaluate(job,step,gpus,deadline,technical=False,final=False):
    h=Path(job['runtime']);out=Path(job['output'])
    def one(shard,gpu):
        label='technical' if technical else ('final' if final else 'intermediate')
        dest=out/f'evaluation-{step:07d}-{shard}'
        execute([PYTHON,'-m','experiments.metaworld_eval_20261103.evaluate','--checkpoint',str(out/f'model-step{step:07d}'),
            '--data',read(job['config'])['data'],'--manifest',str(ROOT/'manifests'/f'{label}-{0 if technical else shard}.json'),
            '--output',str(dest),'--budget','20' if technical else '500'],out/(dest.name+'.log'),env(h,[gpu]),h,deadline)
        r=read(dest/'metrics.json');assert r['passed'] and r['model_state_before']==r['model_state_after']
    if technical:one(0,gpus[0]);return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(one,i,g) for i,g in enumerate(gpus)]
        for f in fs:f.result()


def work(host,pair):
    plan=read(ROOT/'plan.json');deadline=plan['deadline'];gpus=[2*pair,2*pair+1]
    while time.time()<deadline and busy(host,pair):time.sleep(15)
    while (job:=claim(host,pair)) is not None:
        out=Path(job['output']);h=Path(job['runtime'])
        try:
            if job['kind']=='fresh':
                run_train(job,3,0,gpus,deadline);audit(job,3,gpus[0],deadline)
                evaluate(job,3,gpus,deadline,technical=True)
                save(out/'PREFLIGHT_PASSED.json',dict(passed=True,actual_updates=3,native_full_coverage=True,host=host,gpus=gpus,utc=now()))
                while not (ROOT/'PANEL_APPROVED.json').exists():
                    if time.time()>deadline:raise TimeoutError('Formal native gate deadline')
                    time.sleep(15)
                run_train(job,100,3,gpus,deadline)
            else:
                fork=read(out/'FORK.json')
                run_train(job,100,0,gpus,deadline,transfer=fork['parent_checkpoint'])
            audit(job,100,gpus[0],deadline)
            if job['kind']=='continue':evaluate(job,100,gpus,deadline,technical=True)
            previous=100
            for epoch in [20,50,100]:
                stop=job['epoch_steps'][str(epoch)]
                run_train(job,stop,previous,gpus,deadline);audit(job,stop,gpus[0],deadline)
                execute([PYTHON,str(h/'experiments/metaworld_baselines_20261107/diagnose.py'),'--config',job['config'],
                    '--checkpoint',str(out/f'checkpoint-{stop:07d}.pt'),'--output',str(out/f'diagnostic-{stop:07d}.json')],
                    out/f'diagnostic-{stop:07d}.log',env(h,[gpus[1]]),h,deadline)
                evaluate(job,stop,gpus,deadline,final=epoch==100);previous=stop
            save(out/'complete.json',dict(passed=True,step=previous,native_epochs=100,host=host,gpus=gpus,utc=now()))
        except BaseException:
            save(out/'error.json',dict(passed=False,traceback=traceback.format_exc(),utc=now()));raise


def launch(host):
    p=read(ROOT/'plan.json');h=Path(p['helper']);records=[]
    for pair in range(4):
        with (ROOT/f'worker-{host}-{pair}.log').open('x') as stream:
            proc=subprocess.Popen([PYTHON,str(h/'experiments/metaworld_baselines_20261107/native_campaign.py'),
                '--mode','work','--host',host,'--pair',str(pair)],cwd=h,env=env(h,[2*pair,2*pair+1]),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(pair=pair,gpus=[2*pair,2*pair+1],pid=proc.pid))
    save(ROOT/f'launch-{host}.json',dict(host=host,workers=records,utc=now()));print(records)


def collect():
    p=read(ROOT/'plan.json')
    while time.time()<p['deadline']:
        representatives=[Path(j['output'])/'PREFLIGHT_PASSED.json' for j in p['jobs'] if j['kind']=='fresh' and j['seed']==1]
        if not (ROOT/'PANEL_APPROVED.json').exists() and all(f.exists() and read(f)['passed'] for f in representatives):
            save(ROOT/'PANEL_APPROVED.json',dict(passed=True,native_methods=SHORT,representative_gates=[str(f) for f in representatives],utc=now()))
        rows=[]
        for job in p['jobs']:
            out=Path(job['output']);row=dict(name=job['name'],method=job['method'],seed=job['seed'],kind=job['kind'],
                complete=(out/'complete.json').exists(),error=(out/'error.json').exists(),epochs={})
            if (out/'health-rank0.json').exists():row['health']=read(out/'health-rank0.json')
            for epoch,step in job['epoch_steps'].items():
                files=[out/f'evaluation-{step:07d}-{i}/metrics.json' for i in range(2)]
                if all(f.exists() for f in files):
                    episodes=[r for f in files for r in read(f)['episodes']]
                    groups={}
                    for task in sorted({r['task'] for r in episodes}):
                        selected=[r for r in episodes if r['task']==task]
                        groups[task]=dict(episodes=len(selected),successes=sum(r['success'] for r in selected),
                            percent=100*sum(r['success'] for r in selected)/len(selected))
                    row['epochs'][epoch]=dict(step=step,episodes=len(episodes),macro50=sum(g['percent'] for g in groups.values())/50,groups=groups)
            rows.append(row)
        save(ROOT/'results/summary.json',dict(utc=now(),protocol='native_window_full_coverage',records=rows,
            preflights=sum((Path(j['output'])/'PREFLIGHT_PASSED.json').exists() for j in p['jobs'] if j['kind']=='fresh'),
            errors=sum(r['error'] for r in rows),complete=sum(r['complete'] for r in rows),models=18))
        if all(r['complete'] or r['error'] for r in rows):
            save(ROOT/'CLOSED.json',dict(utc=now(),errors=sum(r['error'] for r in rows)));return
        time.sleep(30)
    save(ROOT/'DEADLINE.json',dict(utc=now(),scope='Original finite deadline; no automatic extension'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','launch','work','collect'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--host');p.add_argument('--pair',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.pair)
    else:collect()
