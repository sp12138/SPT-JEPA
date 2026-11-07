"""Finite100+200epoch native-baseline continuation; original100epochs intact."""
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

ROOT=BASE/'runs/metaworld_baseline_scaling_20261108'
PARENT=BASE/'runs/metaworld_baselines_native_20261108'
HOSTS=['dev13','dev15','dev6']


def alive(pid):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return 'metaworld_baselines_20261107' in command or 'metaworld_baseline_scaling_20261108' in command
    except OSError:return False


def prepare(helper):
    p=read(PARENT/'plan.json');ROOT.mkdir(parents=True,exist_ok=False);jobs=[]
    for j in p['jobs']:
        out=ROOT/'training'/j['name']
        jobs.append(dict(name=j['name'],method=j['method'],seed=j['seed'],parent_output=j['output'],
            parent_config=j['config'],output=str(out),config=str(out/'config.json'),steps_per_epoch=j['steps_per_epoch'],
            epoch_steps={str(e):j['steps_per_epoch']*e for e in [100,150,200,250,300]}))
    for label in ['technical-0','intermediate-0','intermediate-1','final-0','final-1']:
        save(ROOT/'manifests'/(label+'.json'),read(PARENT/'manifests'/(label+'.json')))
    save(ROOT/'plan.json',dict(jobs=jobs,helper=str(helper),hosts=HOSTS,deadline=p['deadline'],
        native_total_epochs=300,additional_epochs=200,training_models=18,gpus_per_model=2,
        scope='Original100epoch native training/evaluation remains unchanged. Continue all18 full model/AdamW/RNG/cursor states '
              'for200more nativeepochs,with one native-warmup/cosine restart at original native peak. '
              'Not a random-init300epoch single-cosine reproduction. Same data/split/model/loss/physical cases/controller. '
              'No new candidate method or external/future-optimal-action labels. Original finite deadline retained.'))
    print(json.dumps(dict(root=str(ROOT),models=18,additional_epochs=200,deadline=p['deadline'])))


def primary_busy(host,pair):
    p=read(PARENT/'plan.json')
    # Finish original wave scheduling before taking any released pair.
    for j in p['jobs']:
        if j['kind']!='external' and not (PARENT/'claims'/(j['name']+'.json')).exists():return True
    for owner in p['legacy_owners']:
        if owner['host']==host and owner['pair']==pair and alive(owner['pid']):return True
    for j in p['jobs']:
        if j['kind']=='external':continue
        c=read(PARENT/'claims'/(j['name']+'.json'));out=Path(j['output'])
        if c['host']==host and c['pair']==pair and not (out/'complete.json').exists() and not (out/'error.json').exists():return True
    for gpu in [2*pair,2*pair+1]:
        s=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if any(v.strip().isdigit() for v in s.splitlines()):return True
    return False


def claim(host,pair):
    for j in read(ROOT/'plan.json')['jobs']:
        out=Path(j['output']);parent=Path(j['parent_output'])
        if (out/'SKIPPED_PARENT_ERROR.json').exists():continue
        if (parent/'error.json').exists() and not (parent/'complete.json').exists():
            save(out/'SKIPPED_PARENT_ERROR.json',dict(parent=str(parent),utc=now()));continue
        if not (parent/'complete.json').exists():continue
        f=ROOT/'claims'/(j['name']+'.json');f.parent.mkdir(exist_ok=True)
        try:
            with f.open('x') as stream:json.dump(dict(host=host,pair=pair,pid=os.getpid(),utc=now()),stream)
        except FileExistsError:continue
        return j
    return None


def configure(j):
    from experiments.metaworld_baselines_20261107.train_native import extended_config
    p=read(j['parent_config']);base=j['epoch_steps']['100'];assert p['steps']==base
    parent=Path(j['parent_output'])/f'checkpoint-{base:07d}.pt';assert parent.is_file()
    out=Path(j['output']);cfg=extended_config(p,out)
    save(out/'config.json',cfg)
    save(out/'FORK.json',dict(parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),inherited_step=base,
        inherited_samples=base*256,checkpointed_sgd_repeated=0,parent_config=j['parent_config'],
        original_100epoch_scores_reused_not_reexecuted=True,model_loss_data_unchanged=True,utc=now()))


def train(j,stop,previous,gpus,deadline,first=False):
    h=Path(read(ROOT/'plan.json')['helper']);out=Path(j['output'])
    cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         str(h/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',j['config'],'--stop',str(stop)]
    if first:cmd+=['--resume',read(out/'FORK.json')['parent_checkpoint'],'--allow-budget-extension']
    else:cmd+=['--resume',str(out/f'checkpoint-{previous:07d}.pt')]
    execute(cmd,out/f'train-{stop:07d}.log',env(h,gpus),h,deadline)


def audit(j,step,gpu,deadline):
    h=Path(read(ROOT/'plan.json')['helper']);out=Path(j['output'])
    execute([PYTHON,str(h/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',j['config'],
        '--resume',str(out/f'checkpoint-{step:07d}.pt'),'--audit'],out/f'audit-{step:07d}.log',env(h,[gpu]),h,deadline)


def evaluate(j,step,gpus,deadline,technical=False,full=False):
    h=Path(read(ROOT/'plan.json')['helper']);out=Path(j['output'])
    def one(shard,gpu):
        label='technical' if technical else ('final' if full else 'intermediate');dest=out/f'evaluation-{step:07d}-{shard}'
        execute([PYTHON,'-m','experiments.metaworld_eval_20261103.evaluate','--checkpoint',str(out/f'model-step{step:07d}'),
            '--data',read(j['config'])['data'],'--manifest',str(ROOT/'manifests'/f'{label}-{0 if technical else shard}.json'),
            '--output',str(dest),'--budget','20' if technical else '500'],out/(dest.name+'.log'),env(h,[gpu]),h,deadline)
        d=read(dest/'metrics.json');assert d['passed'] and d['model_state_before']==d['model_state_after']
    if technical:one(0,gpus[0]);return
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(one,i,g) for i,g in enumerate(gpus)]
        for f in futures:f.result()


def work(host,pair):
    plan=read(ROOT/'plan.json');deadline=plan['deadline'];gpus=[2*pair,2*pair+1];h=Path(plan['helper'])
    while time.time()<deadline:
        if primary_busy(host,pair):time.sleep(20);continue
        j=claim(host,pair)
        if j is None:
            if all(any((Path(x['output'])/n).exists() for n in ['complete.json','error.json','SKIPPED_PARENT_ERROR.json']) for x in plan['jobs']):return
            time.sleep(20);continue
        out=Path(j['output'])
        try:
            configure(j);base=j['epoch_steps']['100']
            train(j,base+3,base,gpus,deadline,first=True);audit(j,base+3,gpus[0],deadline)
            evaluate(j,base+3,gpus,deadline,technical=True)
            save(out/'PREFLIGHT_PASSED.json',dict(passed=True,actual_new_ddp_updates=3,inherited_step=base,
                cpu_full_restore_fp32_backward=True,actual_20native_mpc=True,host=host,gpus=gpus,utc=now()))
            train(j,base+100,base+3,gpus,deadline);audit(j,base+100,gpus[0],deadline);previous=base+100
            for epoch in [150,200,250,300]:
                step=j['epoch_steps'][str(epoch)];train(j,step,previous,gpus,deadline);audit(j,step,gpus[0],deadline)
                execute([PYTHON,str(h/'experiments/metaworld_baselines_20261107/diagnose.py'),'--config',j['config'],
                    '--checkpoint',str(out/f'checkpoint-{step:07d}.pt'),'--output',str(out/f'diagnostic-{step:07d}.json')],
                    out/f'diagnostic-{step:07d}.log',env(h,[gpus[1]]),h,deadline)
                evaluate(j,step,gpus,deadline,full=epoch in [200,300]);previous=step
            save(out/'complete.json',dict(passed=True,step=previous,cumulative_native_epochs=300,new_epochs=200,
                host=host,gpus=gpus,utc=now()))
        except BaseException:
            save(out/'error.json',dict(traceback=traceback.format_exc(),utc=now()));raise


def launch(host):
    p=read(ROOT/'plan.json');h=Path(p['helper']);records=[]
    for pair in range(4):
        with (ROOT/f'worker-{host}-{pair}.log').open('x') as stream:
            proc=subprocess.Popen([PYTHON,str(h/'experiments/metaworld_baseline_scaling_20261108/campaign.py'),
                '--mode','work','--host',host,'--pair',str(pair)],cwd=h,env=env(h,[2*pair,2*pair+1]),
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(pair=pair,pid=proc.pid,gpus=[2*pair,2*pair+1]))
    save(ROOT/f'launch-{host}.json',dict(host=host,workers=records,utc=now()));print(records)


def collect():
    p=read(ROOT/'plan.json')
    while time.time()<p['deadline']:
        rows=[]
        for j in p['jobs']:
            out=Path(j['output']);row=dict(name=j['name'],method=j['method'],seed=j['seed'],epochs={},
                complete=(out/'complete.json').exists(),error=(out/'error.json').exists(),
                skipped=(out/'SKIPPED_PARENT_ERROR.json').exists())
            if (out/'health-rank0.json').exists():row['health']=read(out/'health-rank0.json')
            for epoch,step in j['epoch_steps'].items():
                root=Path(j['parent_output']) if epoch=='100' else out
                files=[root/f'evaluation-{step:07d}-{i}/metrics.json' for i in [0,1]]
                if not all(f.exists() for f in files):continue
                reports=[read(f) for f in files];assert all(d['passed'] and d['model_state_before']==d['model_state_after'] for d in reports)
                episodes=[x for d in reports for x in d['episodes']];groups={}
                for task in sorted({x['task'] for x in episodes}):
                    selected=[x for x in episodes if x['task']==task]
                    groups[task]=dict(episodes=len(selected),successes=sum(x['success'] for x in selected),
                        percent=100*sum(x['success'] for x in selected)/len(selected))
                assert len(groups)==50 and len(episodes)==(1000 if epoch in ['100','200','300'] else 100)
                row['epochs'][epoch]=dict(step=step,episodes=len(episodes),macro50=sum(g['percent'] for g in groups.values())/50,
                    inherited=(epoch=='100'),groups=groups)
            rows.append(row)
        save(ROOT/'results/summary.json',dict(utc=now(),protocol='native100plus200_with_native_lr_restart',records=rows,
            models=18,complete=sum(r['complete'] for r in rows),errors=sum(r['error'] for r in rows),skipped=sum(r['skipped'] for r in rows)))
        if all(r['complete'] or r['error'] or r['skipped'] for r in rows):
            save(ROOT/'CLOSED.json',dict(utc=now()));return
        time.sleep(30)
    save(ROOT/'DEADLINE.json',dict(utc=now(),remaining_work_incomplete=True))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['prepare','launch','work','collect'],required=True)
    a.add_argument('--host');a.add_argument('--pair',type=int);a.add_argument('--helper',type=Path);args=a.parse_args()
    if args.mode=='prepare':prepare(args.helper)
    elif args.mode=='launch':launch(args.host)
    elif args.mode=='work':work(args.host,args.pair)
    else:collect()
