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


def prepare(root,source,helper):
    assert read(root/'EIO_DATA_CONTRACT.json')['passed']
    import psutil
    launcher=read(root/'completion-launch.json')['pid'];outer=psutil.Process(launcher)
    child=outer.children();assert len(child)==1 and '--mode' in child[0].cmdline() and 'finish' in child[0].cmdline()
    assert str(root) in child[0].cmdline() and not child[0].children()
    child[0].kill();child[0].wait(timeout=10)
    save(root/'original-collector-retired.json',dict(launcher=launcher,controller=child[0].pid,
         reason='Metadata collector now follows eight EIO recovery attempts;no native training child touched.'))
    plan=read(root/'plan.json');jobs=read(root/'evaluation-plan.json');gate=read(root/'TRAINING_APPROVED.json')
    for name in ['plan.json','evaluation-plan.json','TRAINING_APPROVED.json']:shutil.copy2(root/name,root/('pre-eio-'+name))
    mapping={};rows=[]
    for recipe in plan['recipes']:
        if recipe['host']!='dev15':continue
        parent=Path(recipe['run']);assert (parent/'worker-failed.json').exists()
        assert 'Input/output error' in (parent/'error.txt').read_text()
        archive=parent/'archive-step0008242';manifest=read(archive/'manifest.json')
        assert all(digest(archive/k)==v for k,v in manifest['files'].items())
        run=root/'training-eio-v2'/recipe['name'];run.mkdir(parents=True)
        cfg=read(parent/'config.json');cfg['output']=str(run)
        cfg['retain_steps']=[8242,8244,9888,16480]
        save(run/'config.json',cfg)
        shutil.copytree(parent/'initial-model',run/'initial-model',copy_function=os.link)
        shutil.copytree(archive,run/'inherited-archive-step0008242',copy_function=os.link)
        shutil.copytree(parent/'model-step0008242',run/'model-step0008242',copy_function=os.link)
        os.link(archive/'native.ckpt',run/'last.ckpt')
        save(run/'FORK.json',dict(passed=True,parent=str(parent),parent_config=str(parent/'config.json'),entry_step=8242,
            parent_proof=dict(native_sha256=manifest['files']['native.ckpt']),
            reason='HDF5 EIO at19:03:55UTC. Same saved state;bounded same-sample read retries only.',
            replayed_unsaved_updates=read(parent/'status.json')['step']-8242,
            scope='Original errors and endpoints retained. No model/loss/batch/data/order/optimizer/LR changes;no OOM attribution.'))
        mapping[str(parent)]=str(run)
        recipe.update(run=str(run),original_run=str(parent),entry_step=8242,stages=[8244,9888,16480],
                      runtime_source=str(source),helper=str(helper),config_sha256=digest(run/'config.json'))
        gate['config_sha256'][recipe['name']]=recipe['config_sha256']
        rows.append(dict(name=recipe['name'],parent=str(parent),run=str(run),gpu=recipe['gpu']))
    assert len(rows)==8
    corrections={}
    for j in jobs['jobs']:
        if j['run'] not in mapping:continue
        original=copy.deepcopy(j);j['run']=mapping[j['run']]
        j['checkpoint']=str(Path(j['run'])/f'model-step{j["step"]:07d}')
        j['work']=str(root/'evaluation-eio-v2'/j['name'])
        corrections[j['name']]=dict(original=original,effective=j)
    assert len(corrections)==192
    replace(root/'plan.json',plan);replace(root/'TRAINING_APPROVED.json',gate);replace(root/'evaluation-plan.json',jobs)
    save(root/'EIO_RECOVERY.json',dict(passed=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source),helper=str(helper),recoveries=rows,evaluation_corrections=corrections,
        scope='Eight logical methods/seeds recovered,not eight new methods. Original192 blocked/unfinished endpoint paths retained;'
              'healthy dev13 native jobs remain running unchanged. Failed unsaved updates are replayed from8242.'))
    print(dict(prepared=True,recovered=8,corrected_endpoints=192),flush=True)


def launch(root,source,helper):
    assert read(root/'EIO_RECOVERY.json')['passed']
    from experiments.metaworld_eval_20261103.worker import free_gpu
    dispatch=read(root/'dispatch.json');python='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
    env=os.environ.copy();env.update(PYTHONPATH=str(helper)+':'+str(source),PYTHONDONTWRITEBYTECODE='1')
    rows=[]
    for recipe in read(root/'plan.json')['recipes']:
        if recipe['host']!='dev15':continue
        free_gpu(recipe['gpu'])
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time()))),
             python,'-B','-u',str(helper/'experiments/metaworld_improve_20261104/campaign.py'),'--mode','work',
             '--root',str(root),'--source',str(source),'--helper',str(helper),'--host','dev15','--gpu',str(recipe['gpu'])]
        with (root/'workers'/f'eio-v2-dev15-g{recipe["gpu"]}.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(gpu=recipe['gpu'],pid=p.pid,command=cmd))
    save(root/'launch-eio-v2-dev15.json',dict(passed=True,jobs=rows,host='dev15'))
    print(dict(host='dev15',pids=[r['pid'] for r in rows]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','launch'],required=True)
    for key in ['root','source','helper']:parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(Path(args.root),Path(args.source),Path(args.helper))
    else:launch(Path(args.root),Path(args.source),Path(args.helper))
