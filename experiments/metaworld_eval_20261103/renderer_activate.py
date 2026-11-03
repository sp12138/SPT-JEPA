"""Replace only owned controllers; keep in-flight native training/audits intact."""
import argparse, datetime, json, os
from pathlib import Path
import subprocess, time
import psutil


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as file:json.dump(value,file,indent=2)


def main(root,host,helper):
    root=Path(root);helper=Path(helper);source=root.parent.parent/'reports/metaworld_eval_20261103/runtime-60c887e'
    assert read(root/'RENDERER_WARMUP_APPROVED.json')['passed']
    assert read(root/'DATA_IO_APPROVED.json')['source']==str(source)
    output=root/f'renderer-activation-{host}.json'
    if output.exists():print('Already activated',host);return
    dispatch=read(root/'dispatch.json');specs=[]
    for rec in read(root/f'reviews/renderer-pause-{host}.json')['rows']:
        specs.append(dict(mode='evaluate',name=f"evaluation-{host}-g{rec['gpu']}",gpu=rec['gpu'],old_parent=rec['suspended_parent']))
    if host!='dev13':
        for rec in read(root/f'reviews/renderer-training-pause-{host}.json')['rows']:
            specs.append(dict(mode='evaluate' if rec['complete'] else 'train',name=rec['name'],gpu=rec['gpu'],
                              old_parent=rec['suspended_parent'],run=rec['run'],wait_pid=rec['wait_pid'],adopt_step=rec['adopt_step']))
    assert len(specs)==8
    if host=='dev2':
        old=psutil.Process(read(root/'completion-launch.json')['pid'])
        collectors=[p for p in old.children() if any(x.endswith('/finish.py') for x in p.cmdline())]
        assert len(collectors)==1
        collectors[0].terminate()
        save(root/'original-collector-retired.json',dict(pid=collectors[0].pid,scope='Exact metadata collector only;new collector includes canonical renderer replays.'))
    jobs=[]
    for spec in specs:
        old=psutil.Process(spec['old_parent']);assert old.status()==psutil.STATUS_STOPPED
        assert any(x.endswith('/worker.py') for x in old.cmdline())
        assert str(root) in old.cmdline()
        remaining=max(1,int(dispatch['deadline_unix']-time.time()))
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(remaining),'/efs/prescottsun/project_jepa/envs/lewm/bin/python',
             '-B','-u',str(helper/'experiments/metaworld_eval_20261103/worker.py'),'--root',str(root),
             '--mode',spec['mode'],'--source',str(source),'--helper',str(helper),'--base',str(root.parent.parent),
             '--gpu',str(spec['gpu']),'--host',host]
        if spec['mode']=='train':cmd+=['--run',spec['run'],'--adopt-native-pid',str(spec['wait_pid']),'--adopt-stop-step',str(spec['adopt_step'])]
        with (root/'workers'/('renderer-v2-'+spec['name']+'.log')).open('x') as log:
            process=subprocess.Popen(cmd,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        old.kill()
        jobs.append(dict(**spec,new_launcher_pid=process.pid,command=cmd))
    if host=='dev2':
        cmd=['/usr/bin/timeout','--signal=TERM','--kill-after=60',str(max(1,int(dispatch['deadline_unix']-time.time())+300)),
             '/efs/prescottsun/project_jepa/envs/lewm/bin/python','-B','-u',str(helper/'experiments/metaworld_eval_20261103/finish.py'),'--root',str(root)]
        env=os.environ.copy();env.update(PYTHONPATH=str(helper),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
        with (root/'completion-renderer-v2.log').open('x') as log:
            collector=subprocess.Popen(cmd,env=env,cwd=helper,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'completion-renderer-v2-launch.json',dict(pid=collector.pid,host=host,command=cmd))
    save(output,dict(passed=True,host=host,helper=str(helper),source=str(source),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,
                     scope='Exact owned parent replacement;in-flight native training/audits untouched. No SGD repeat or changes to cases,weights,NN,optimizer,sampler,LR,planner orbudgets.'))
    print(json.dumps(dict(passed=True,host=host,new=[j['new_launcher_pid'] for j in jobs])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--host',required=True);parser.add_argument('--helper',required=True)
    args=parser.parse_args();main(args.root,args.host,args.helper)
