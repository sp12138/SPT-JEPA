"""Recover only the eight failed dev15 attempts;leave healthy dev13 jobs intact."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.metaworld_eval_20261103.worker import read,save,digest


def replace(path,value):
    temporary=path.with_suffix(path.suffix+'.recovery.tmp')
    save(temporary,value);os.replace(temporary,path)


def prepare(root,source,helper,host='dev15',revision='v2',collector_record='completion-launch.json'):
    assert read(root/'EIO_DATA_CONTRACT.json')['passed']
    import psutil
    launcher=read(root/collector_record)['pid'];outer=psutil.Process(launcher)
    child=outer.children();assert len(child)==1 and '--mode' in child[0].cmdline() and 'finish' in child[0].cmdline()
    assert str(root) in child[0].cmdline() and not child[0].children()
    child[0].kill();child[0].wait(timeout=10)
    save(root/f'collector-retired-{host}-{revision}.json',dict(launcher=launcher,controller=child[0].pid,
         reason='Metadata collector now follows eight EIO recovery attempts;no native training child touched.'))
    plan=read(root/'plan.json');jobs=read(root/'evaluation-plan.json');gate=read(root/'TRAINING_APPROVED.json')
    for name in ['plan.json','evaluation-plan.json','TRAINING_APPROVED.json']:shutil.copy2(root/name,root/(f'pre-eio-{host}-{revision}-'+name))
    mapping={};rows=[]
    for recipe in plan['recipes']:
        if recipe['host']!=host:continue
        parent=Path(recipe['run']);assert (parent/'worker-failed.json').exists()
        assert 'Input/output error' in (parent/'error.txt').read_text()
        import torch
        native=torch.load(parent/'last.ckpt',map_location='cpu',weights_only=False)
        entry=int(native['global_step']);assert native['campaign_samples']==entry*128
        assert native['optimizer_states'] and native['lr_schedulers'] and native['campaign_rng']
        assert entry+2<16480
        del native
        native_sha=digest(parent/'last.ckpt')
        archive=parent/f'archive-step{entry:07d}'
        if archive.exists():
            manifest=read(archive/'manifest.json')
            assert manifest['files']['native.ckpt']==native_sha
            assert all(digest(archive/k)==v for k,v in manifest['files'].items())
        run=root/f'training-eio-{host}-{revision}'/recipe['name'];run.mkdir(parents=True)
        cfg=read(parent/'config.json');cfg['output']=str(run)
        cfg['retain_steps']=sorted({entry,entry+2,9888,16480})
        save(run/'config.json',cfg)
        shutil.copytree(parent/'initial-model',run/'initial-model',copy_function=os.link)
        if archive.exists():shutil.copytree(archive,run/f'inherited-archive-step{entry:07d}',copy_function=os.link)
        shutil.copytree(parent/f'model-step{entry:07d}',run/f'model-step{entry:07d}',copy_function=os.link)
        os.link(parent/'last.ckpt',run/'last.ckpt')
        save(run/'FORK.json',dict(passed=True,parent=str(parent),parent_config=str(parent/'config.json'),entry_step=entry,
            parent_proof=dict(native_sha256=native_sha),
            reason='HDF5 EIO. Same latest complete saved state;bounded same-sample read retries only.',
            replayed_unsaved_updates=read(parent/'status.json')['step']-entry,
            scope='Original errors and endpoints retained. No model/loss/batch/data/order/optimizer/LR changes;no OOM attribution.'))
        mapping[str(parent)]=str(run)
        stages=[entry+2]+[step for step in [9888,16480] if step>entry+2]
        recipe.update(run=str(run),original_run=str(parent),entry_step=entry,stages=stages,archive_entry=len(stages)==2,
                      runtime_source=str(source),helper=str(helper),config_sha256=digest(run/'config.json'))
        gate['config_sha256'][recipe['name']]=recipe['config_sha256']
        rows.append(dict(name=recipe['name'],parent=str(parent),run=str(run),gpu=recipe['gpu']))
    assert len(rows)==8
    corrections={}
    for j in jobs['jobs']:
        if j['run'] not in mapping:continue
        entry=read(Path(mapping[j['run']])/'FORK.json')['entry_step']
        if j['step']<=entry:continue
        original=copy.deepcopy(j);j['run']=mapping[j['run']]
        j['checkpoint']=str(Path(j['run'])/f'model-step{j["step"]:07d}')
        j['work']=str(root/f'evaluation-eio-{host}-{revision}'/j['name'])
        corrections[j['name']]=dict(original=original,effective=j)
    assert len(corrections)>0
    replace(root/'plan.json',plan);replace(root/'TRAINING_APPROVED.json',gate);replace(root/'evaluation-plan.json',jobs)
    save(root/f'EIO_RECOVERY-{host}-{revision}.json',dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source),helper=str(helper),recoveries=rows,evaluation_corrections=corrections,
        scope='Eight logical methods/seeds recovered,not eight new methods. Original blocked/unfinished endpoint paths retained;'
              'the other host stays untouched. Resume each latest complete saved state;replay only unsaved updates.'))
    print(dict(prepared=True,host=host,recovered=8,corrected_endpoints=len(corrections)),flush=True)


def launch(root,source,helper,host='dev15',revision='v2'):
    assert read(root/f'EIO_RECOVERY-{host}-{revision}.json')['passed']
    from experiments.metaworld_eval_20261103.worker import free_gpu
    dispatch=read(root/'dispatch.json');python='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    env=os.environ.copy();env.update(PYTHONPATH=str(helper)+':'+str(source),PYTHONDONTWRITEBYTECODE='1')
    rows=[]
    for recipe in read(root/'plan.json')['recipes']:
        if recipe['host']!=host:continue
        free_gpu(recipe['gpu'])
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time()))),
             python,'-B','-u',str(helper/'experiments/metaworld_improve_20261104/campaign.py'),'--mode','work',
             '--root',str(root),'--source',str(source),'--helper',str(helper),'--host',host,'--gpu',str(recipe['gpu'])]
        with (root/'workers'/f'eio-{revision}-{host}-g{recipe["gpu"]}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(gpu=recipe['gpu'],pid=p.pid,command=cmd))
    save(root/f'launch-eio-{revision}-{host}.json',dict(passed=True,jobs=rows,host=host))
    print(dict(host=host,pids=[r['pid'] for r in rows]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','launch'],required=True)
    for key in ['root','source','helper']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--host',choices=['dev13','dev15'],default='dev15');parser.add_argument('--revision',default='v2')
    parser.add_argument('--collector-record',default='completion-launch.json');args=parser.parse_args()
    if args.mode=='prepare':prepare(Path(args.root),Path(args.source),Path(args.helper),args.host,args.revision,args.collector_record)
    else:launch(Path(args.root),Path(args.source),Path(args.helper),args.host,args.revision)
