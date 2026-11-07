"""User-authorized three-host handoff of four saved step3 learners."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from experiments.metaworld_baselines_20261107.campaign import ROOT,PYTHON,read,save,sha,env,execute,stage,audit,evaluate,now

HOSTS=['dev13','dev15','dev6']


def prepare(helper):
    plan=read(ROOT/'plan.json');records=[]
    assert sha(helper/'experiments/metaworld_baselines_20261107/models.py')==sha(Path(plan['helper'])/'experiments/metaworld_baselines_20261107/models.py')
    for job in plan['jobs']:
        c=ROOT/'claims-v3'/(job['name']+'.json')
        if not c.exists() or read(c)['host']!='dev6-2c':continue
        old=Path(job['output']);parent=old/'checkpoint-0000003.pt'
        assert parent.exists() and not (old/'checkpoint-0000100.pt').exists()
        records.append(dict(name=job['name'],old=str(old),parent=str(parent),cfg=read(job['config']),job=job))
    assert len(records)==4
    for record in records:
        job=record['job'];old=Path(record['old']);new=ROOT/'training-threehost-v5'/job['name']
        new.mkdir(parents=True,exist_ok=False)
        original=record['cfg'];save(old/'config-at-threehost-retirement.json',original)
        # Old v3 workers do not implement a STOP file. Its strict native-resume
        # config equality DOES fail closed on this extra administrative field.
        # Existing in-memory work cannot be remotely killed while SSH is down;
        # the following resume is revoked without altering any hyperparameter.
        save(old/'config.json',dict(original,retired_by_user_threehost=True))
        save(old/'OWNERSHIP_RETIRED.json',dict(utc=now(),reason='Latest user requests only dev13/dev15/dev6',
            current_process_state='Unconfirmed: SSH unavailable; no further connection attempts',
            resume_guard='Strict checkpoint-config equality rejects the extra retirement field before future SGD',
            original_config=str(old/'config-at-threehost-retirement.json'),
            residual_in_memory_work='Unknown; cannot claim an immediate process kill',new_output=str(new)))
        cfg=dict(original,output=str(new));save(new/'config.json',cfg)
        save(new/'FORK.json',dict(parent_checkpoint=record['parent'],parent_checkpoint_sha256=sha(record['parent']),
            inherited_step=3,inherited_samples=768,checkpointed_sgd_repeated=0,
            unsaved_excluded_host_work='Unknown/unrecoverable; may be replayed',original_run=str(old),helper=str(helper)))
        inherited=read(old/'PREFLIGHT_PASSED.json')
        save(new/'PREFLIGHT_PASSED.json',dict(inherited,inherited=True,original_output=str(old),
            scope='Original actual3update/CPUrestore/FP32/MPC proof, not a newly executed gate'))
        job.update(output=str(new),config=str(new/'config.json'),retired_host_output=str(old),threehost_transfer=True)
        record.clear();record.update(name=job['name'],output=str(new))
    owner=read(ROOT/'collector-launch-v4.json');p=Path(f'/proc/{owner["pid"]}/cmdline')
    if p.exists():
        command=p.read_bytes().replace(b'\0',b' ').decode();assert 'campaign.py' in command and '--mode collect' in command
        os.kill(owner['pid'],signal.SIGTERM)
    save(ROOT/'collector-retired-v4.json',dict(owner,utc=now(),reason='Three-host output handoff; training children untouched'))
    save(ROOT/'plan-before-threehost-v5.json',read(ROOT/'plan.json'))
    plan.update(hosts=HOSTS,concurrent_pairs=12,excluded_hosts=['dev6-2c'],threehost_transfer_utc=now())
    save(ROOT/'plan.json',plan)
    save(ROOT/'THREEHOST_RECOVERY.json',dict(passed=True,helper=str(helper),hosts=HOSTS,transfers=records,
        original_deadline=plan['deadline'],checkpointed_sgd_repeated=0,excluded_host_unsaved_work='Unknown; do not claim zero replay',
        initial14on_allowed_hosts='Keep all current parents/SGD/evaluation and two unstarted primary recipes unchanged',
        retired_host='dev6-2c; no future SSH or new scheduling',utc=now()))


def owned_alive(pid):
    try:return 'metaworld_baselines_20261107' in Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except OSError:return False


def primary_busy_only(host,pair):
    for file in [ROOT/f'launch-v3-{host}.json',ROOT/f'launch-device-v4-{host}.json']:
        if file.exists():
            for record in read(file)['workers']:
                if record['pair']==pair and owned_alive(record['pid']):return True
    main=read(ROOT/'plan.json')
    if any(not (ROOT/'claims-v3'/(j['name']+'.json')).exists() for j in main['jobs']):return True
    for gpu in [pair*2,pair*2+1]:
        out=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if any(s.strip().isdigit() for s in out.splitlines()):return True
    return False


def work(host,pair):
    assert host in HOSTS
    recovery=read(ROOT/'THREEHOST_RECOVERY.json');helper=Path(recovery['helper']);deadline=recovery['original_deadline']
    while time.time()<deadline:
        if primary_busy_only(host,pair):time.sleep(20);continue
        job=None
        for record in recovery['transfers']:
            claim=ROOT/'threehost-recovery/claims'/(record['name']+'.json');claim.parent.mkdir(parents=True,exist_ok=True)
            try:
                with claim.open('x') as stream:
                    import json
                    json.dump(dict(host=host,pair=pair,pid=os.getpid(),utc=now()),stream)
            except FileExistsError:continue
            job=next(j for j in read(ROOT/'plan.json')['jobs'] if j['name']==record['name']);break
        if job is None:return
        output=Path(job['output']);gpus=[pair*2,pair*2+1]
        try:
            fork=read(output/'FORK.json')
            # A previously in-flight old-host100 archive may arrive after the
            # handoff. Prefer it when its ready marker is complete, so known
            # saved work is retained. No new connection to that host is made.
            old=Path(fork['original_run']);late=old/'checkpoint-0000100.pt'
            if late.exists() and (old/'ready-0000100.json').exists():
                fork.update(parent_checkpoint=str(late),parent_checkpoint_sha256=sha(late),
                    inherited_step=100,inherited_samples=25600,late_archive_adopted=True)
                save(output/'FORK.json',fork)
            entry_stop=102 if fork['inherited_step']==100 else 100
            cmd=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                str(helper/'experiments/metaworld_baselines_20261107/train_native.py'),'--config',job['config'],
                '--stop',str(entry_stop),'--resume',fork['parent_checkpoint'],'--allow-output-transfer']
            execute(cmd,output/f'train-{entry_stop:07d}.log',env(helper,gpus),helper,deadline)
            audit(job,entry_stop,helper,gpus[0],deadline);evaluate(job,entry_stop,helper,gpus,deadline,technical=True)
            for previous,stop in [(entry_stop,2500),(2500,8240),(8240,20600),(20600,41200)]:
                stage(job,stop,helper,gpus,previous,deadline);audit(job,stop,helper,gpus[0],deadline)
                execute([PYTHON,str(helper/'experiments/metaworld_baselines_20261107/diagnose.py'),
                    '--config',job['config'],'--checkpoint',str(output/f'checkpoint-{stop:07d}.pt'),
                    '--output',str(output/f'diagnostic-{stop:07d}.json')],output/f'diagnostic-{stop:07d}.log',
                    env(helper,[gpus[1]]),helper,deadline)
                evaluate(job,stop,helper,gpus,deadline)
            save(output/'complete.json',dict(passed=True,step=41200,host=host,gpus=gpus,utc=now()))
        except BaseException:
            save(output/'error.json',dict(passed=False,traceback=traceback.format_exc(),utc=now()));raise


def launch(host):
    r=read(ROOT/'THREEHOST_RECOVERY.json');helper=Path(r['helper']);records=[]
    for pair in range(4):
        log=ROOT/f'threehost-{host}-{pair}.log'
        with log.open('x') as stream:
            p=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--mode','work','--host',host,'--pair',str(pair)],
                cwd=helper,env=env(helper,[2*pair,2*pair+1]),stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(pair=pair,pid=p.pid))
    save(ROOT/f'launch-threehost-v5-{host}.json',dict(host=host,workers=records,utc=now()));print(records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','launch','work'],required=True)
    p.add_argument('--helper',type=Path);p.add_argument('--host');p.add_argument('--pair',type=int);a=p.parse_args()
    if a.mode=='prepare':prepare(a.helper)
    elif a.mode=='launch':launch(a.host)
    else:work(a.host,a.pair)
