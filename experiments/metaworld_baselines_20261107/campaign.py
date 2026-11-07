"""Finite four-host public-baseline benchmark, native-loss gates, and collection."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_baselines_20261107'
SOURCE=BASE/'reports/metaworld_input_inverse_20261107/source-faf40148'
PYTHON='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
HOSTS=['dev13','dev15','dev6','dev6-2c']
METHODS=['lewm','lewm-idm','h-jepa','smwm','ac-mtm','lpwm']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def env(helper,gpus):
    from experiments.metaworld_input_inverse_20261107.campaign import environment
    result=environment(helper,gpus[0])
    result['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
    result['PYTHONPATH']=str(helper)+':'+str(SOURCE)+':'+result['PYTHONPATH']
    result['TOKENIZERS_PARALLELISM']='false'
    return result


def execute(cmd,log,environment,helper,deadline):
    remaining=int(deadline-time.time())
    if remaining<=0:raise TimeoutError('Finite baseline deadline reached')
    with Path(log).open('x') as stream:
        subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(remaining),*cmd],
            env=environment,cwd=helper,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            check=True,timeout=remaining+120)


def prepare(helper):
    assert (ROOT/'reference/plan.json').exists()
    sources=BASE/'sources/metaworld-baselines-20261107'
    catalog=[]
    for name in ['lewm','hjepa','smwm','acmtm','lpwm']:
        path=sources/name
        commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        files=subprocess.check_output(['git','-C',str(path),'ls-files'],text=True).splitlines()
        hashes={f:sha(path/f) for f in files if Path(f).suffix in ['.py','.yaml'] or Path(f).name in ['LICENSE','NOTICE']}
        catalog.append(dict(name=name,commit=commit,path=str(path),hashes=hashes))
    manifest=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')
    rows=manifest['rows'];tasks=sorted({c['task'] for c in rows})
    intermediate=[]
    for task in tasks:
        selected=sorted([c for c in rows if c['task']==task],key=lambda c:c['seed'])
        intermediate.extend([selected[0],selected[10]])
    for label,selected in [('intermediate',intermediate),('final',rows),('technical',rows[:1])]:
        for shard in range(2 if label!='technical' else 1):
            save(ROOT/'manifests'/f'{label}-{shard}.json',dict(manifest,rows=selected[shard::2] if label!='technical' else selected))
    jobs=[]
    for seed in [1,2,3]:
        for method in METHODS:
            name=f'{method}-seed{seed}';out=ROOT/'training'/name
            out.mkdir(parents=True,exist_ok=False)
            cfg=dict(method=method,seed=seed,sources=str(sources),output=str(out),
                data=str(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'),steps=41200,global_batch=256,
                milestones=[3,100,2500,8240,20600,41200],source_catalog_sha256=None,
                scratch=True,inputs=['pixels','factual_executed_action'],reward=False,privileged_state=False,
                optimal_action_labels=False,inverse_target='same observed completed-transition forward input only')
            save(out/'config.json',cfg)
            jobs.append(dict(name=name,method=method,seed=seed,config=str(out/'config.json'),output=str(out)))
    save(ROOT/'sources.json',dict(public=catalog,helper=str(helper),
        helper_hashes={str(p.relative_to(helper)):sha(p) for p in helper.rglob('*.py')},
        original_project_source=str(SOURCE),public_losses_verbatim=True,
        adaptation='Common LeRobot trajectory split/anchor order,224px,20D macro controls,global batch256,two GPUs. '
            'Native temporal lengths2/4/8 and native optimizers/loss coefficients retained. '
            'Warmup/cosine advances once per optimizer update with fixed100epoch horizon; '
            'this explicitly fixes ambiguous public epoch-interval versus update-count scheduler configuration. '
            'H-JEPA native DROID architecture transferred with224px/20D inputs; full native hierarchy controller is a separate track.'))
    source_sha=sha(ROOT/'sources.json')
    for job in jobs:
        cfg=read(job['config']);cfg['source_catalog_sha256']=source_sha;save(job['config'],cfg)
    save(ROOT/'plan.json',dict(created_utc=now(),helper=str(helper),jobs=jobs,hosts=HOSTS,
        steps=41200,epochs=100,global_batch=256,training_models=18,gpus_per_training=2,concurrent_pairs=16,
        total_final_model_cases=18000,intermediate_model_cases_per_checkpoint=1800,
        deadline=time.time()+5*86400,source_catalog_sha256=source_sha,
        full_hjepa_native_controller='Supplementary track pending actual source/interface gate; common-CEM scores explicitly labeled flat-controller representation transfer',
        old_goals_monitors='off',no_automatic_candidate_search=True))


def await_reference(host,pair,deadline):
    plan=read(ROOT/'reference/plan.json')
    jobs=[j for j in plan['jobs'] if j['host']==host and j['gpu'] in [2*pair,2*pair+1]]
    assert len(jobs)==2
    while True:
        if time.time()>deadline:raise TimeoutError('Reference wait expired')
        paths=[ROOT/'reference/workers'/j['name'] for j in jobs]
        if any((p/'error.json').exists() for p in paths):raise RuntimeError('Reference technical/evaluation failure, not baseline performance')
        if all((p/'complete.json').exists() for p in paths):return
        time.sleep(15)


def claim(host,pair):
    plan=read(ROOT/'plan.json')
    for job in plan['jobs']:
        path=ROOT/plan.get('claims_dir','claims')/(job['name']+'.json');path.parent.mkdir(exist_ok=True)
        try:
            with path.open('x') as stream:json.dump(dict(host=host,pair=pair,pid=os.getpid(),utc=now()),stream)
            return job
        except FileExistsError:continue
    return None


def stage(job,stop,helper,gpus,resume,deadline):
    cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(helper/'experiments/metaworld_baselines_20261107/train_native.py'),
        '--config',job['config'],'--stop',str(stop)]
    if resume:cmd+=['--resume',str(Path(job['output'])/f'checkpoint-{resume:07d}.pt')]
    execute(cmd,Path(job['output'])/f'train-{stop:07d}.log',env(helper,gpus),helper,deadline)


def audit(job,step,helper,gpu,deadline):
    execute([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/train_native.py'),
        '--config',job['config'],'--resume',str(Path(job['output'])/f'checkpoint-{step:07d}.pt'),'--audit'],
        Path(job['output'])/f'audit-{step:07d}.log',env(helper,[gpu]),helper,deadline)


def evaluate(job,step,helper,gpus,deadline,technical=False):
    def run(shard,gpu):
        label='technical' if technical else ('final' if step==41200 else 'intermediate')
        manifest=ROOT/'manifests'/f'{label}-{0 if technical else shard}.json'
        out=Path(job['output'])/f'evaluation-{step:07d}-{shard}'
        cmd=[PYTHON,'-m','experiments.metaworld_eval_20261103.evaluate','--checkpoint',
            str(Path(job['output'])/f'model-step{step:07d}'),'--data',read(job['config'])['data'],
            '--manifest',str(manifest),'--output',str(out),'--budget','20' if technical else '500']
        execute(cmd,Path(job['output'])/(out.name+'.log'),env(helper,[gpu]),helper,deadline)
        r=read(out/'metrics.json')
        assert r['passed'] and r['model_state_before']==r['model_state_after']
    if technical:run(0,gpus[0]);return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,i,gpu) for i,gpu in enumerate(gpus)]
        for f in futures:f.result()


def work(host,pair):
    plan=read(ROOT/'plan.json');helper=Path(plan['helper']);gpus=[pair*2,pair*2+1]
    await_reference(host,pair,plan['deadline'])
    while (job:=claim(host,pair)) is not None:
        out=Path(job['output'])
        try:
            stage(job,3,helper,gpus,0,plan['deadline'])
            audit(job,3,helper,gpus[0],plan['deadline'])
            evaluate(job,3,helper,gpus,plan['deadline'],technical=True)
            save(out/'PREFLIGHT_PASSED.json',dict(passed=True,actual_updates=3,global_batch=256,
                full_state_restore=True,heldout_fp32_backward=True,actual_mpc_native_steps=20,host=host,gpus=gpus,utc=now()))
            while not (ROOT/'PANEL_APPROVED.json').exists():
                if time.time()>plan['deadline']:raise TimeoutError('Baseline source/entry gate expired')
                time.sleep(15)
            stage(job,100,helper,gpus,3,plan['deadline'])
            audit(job,100,helper,gpus[0],plan['deadline'])
            for before,stop in [(100,2500),(2500,8240),(8240,20600),(20600,41200)]:
                stage(job,stop,helper,gpus,before,plan['deadline'])
                audit(job,stop,helper,gpus[0],plan['deadline'])
                execute([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/diagnose.py'),
                    '--config',job['config'],'--checkpoint',str(out/f'checkpoint-{stop:07d}.pt'),
                    '--output',str(out/f'diagnostic-{stop:07d}.json')],out/f'diagnostic-{stop:07d}.log',
                    env(helper,[gpus[1]]),helper,plan['deadline'])
                evaluate(job,stop,helper,gpus,plan['deadline'])
            save(out/'complete.json',dict(passed=True,step=41200,host=host,gpus=gpus,utc=now()))
        except BaseException:
            save(out/'error.json',dict(passed=False,host=host,gpus=gpus,traceback=traceback.format_exc(),utc=now()))
            raise


def launch(host):
    plan=read(ROOT/'plan.json');helper=Path(plan['helper']);records=[]
    for pair in range(4):
        tag=plan.get('launch_tag','')
        log=ROOT/f'pair-{tag}{host}-{pair}.log'
        with log.open('x') as stream:
            p=subprocess.Popen([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/campaign.py'),
                '--mode','work','--host',host,'--pair',str(pair)],cwd=helper,env=env(helper,[pair*2,pair*2+1]),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(pair=pair,gpus=[pair*2,pair*2+1],pid=p.pid,log=str(log)))
    save(ROOT/f'launch-{plan.get("launch_tag","")}{host}.json',dict(host=host,workers=records,utc=now()));print(json.dumps(records))


def collect():
    plan=read(ROOT/'plan.json')
    while time.time()<plan['deadline']:
        if not (ROOT/'PANEL_APPROVED.json').exists():
            gates=[Path(next(j for j in plan['jobs'] if j['method']==method and j['seed']==1)['output'])/'PREFLIGHT_PASSED.json' for method in METHODS]
            if all(g.exists() and read(g)['passed'] for g in gates):
                source_file=Path(plan.get('source_catalog',ROOT/'sources.json'))
                assert sha(source_file)==plan['source_catalog_sha256']
                save(ROOT/'PANEL_APPROVED.json',dict(passed=True,methods=METHODS,representative_native_gates=[str(g) for g in gates],
                    actual_3update_ddp_models=6,native_20step_mpc_checks=6,scratch=True,source_catalog_sha256=sha(source_file),utc=now()))
        records=[]
        for job in plan['jobs']:
            out=Path(job['output']);item=dict(name=job['name'],method=job['method'],seed=job['seed'],
                preflight=(out/'PREFLIGHT_PASSED.json').exists(),complete=(out/'complete.json').exists(),error=(out/'error.json').exists())
            if (out/'health-rank0.json').exists():item['health']=read(out/'health-rank0.json')
            scores={}
            for step in [2500,8240,20600,41200]:
                files=[out/f'evaluation-{step:07d}-{shard}/metrics.json' for shard in range(2)]
                if all(f.exists() for f in files):
                    episodes=[r for f in files for r in read(f)['episodes']]
                    groups={}
                    for task in sorted({r['task'] for r in episodes}):
                        selected=[r for r in episodes if r['task']==task]
                        groups[task]=dict(episodes=len(selected),successes=sum(r['success'] for r in selected),
                            success_percent=100*sum(r['success'] for r in selected)/len(selected))
                    scores[str(step)]=dict(groups=groups,episodes=len(episodes),
                        macro50=sum(g['success_percent'] for g in groups.values())/len(groups))
            item['scores']=scores;records.append(item)
        save(ROOT/'results/summary.json',dict(utc=now(),panel_approved=(ROOT/'PANEL_APPROVED.json').exists(),
            training_models=18,preflights=sum(r['preflight'] for r in records),complete=sum(r['complete'] for r in records),
            errors=sum(r['error'] for r in records),records=records,scope='Independent LeRobot rewardfree RGB-goal transfer; common-CEM primary. Not published JEPA-WMs or canonical MT50 reproduction.'))
        if all(r['complete'] or r['error'] for r in records):
            save(ROOT/'CLOSED.json',dict(utc=now(),complete=sum(r['complete'] for r in records),errors=sum(r['error'] for r in records)))
            return
        time.sleep(30)
    save(ROOT/'DEADLINE.json',dict(utc=now(),reason='Finite five-day cap; no automatic extension'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['prepare','launch','work','collect'])
    p.add_argument('--helper',type=Path);p.add_argument('--host');p.add_argument('--pair',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    elif a.mode=='launch':launch(a.host)
    elif a.mode=='work':work(a.host,a.pair)
    else:collect()
