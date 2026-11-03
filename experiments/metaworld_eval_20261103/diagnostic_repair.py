"""Replace idle diagnostic controllers; preserve all failed preflight evidence."""
import argparse
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.metaworld_eval_20261103.worker import read,save
from experiments.metaworld_eval_20261103.twohost import retire_controller


def main(root,helper,host,mode,revision='v2',parent_revision=None):
    assert read(helper.parent/'twohost-smoke-b0ecb75-native/metrics.json')['passed']
    if mode=='activate':
        assert host in ['dev13','dev15']
        result=root/f'diagnostic-activation-{revision}-{host}.json';assert not result.exists()
        env=os.environ.copy();env.update(PYTHONPATH=str(helper),PYTHONDONTWRITEBYTECODE='1')
        rows=[]
        previous=root/(f'diagnostic-activation-{parent_revision}-{host}.json' if parent_revision else f'activation-{host}.json')
        for job in read(previous)['jobs']:
            if job.get('mode','evaluate')!='evaluate':continue
            retired=retire_controller(job['pid'],root,'worker.py')
            old_script=next(x for x in job['command'] if x.endswith('/worker.py'))
            old_helper=Path(old_script).parents[2]
            command=[s.replace(str(old_helper),str(helper)) for s in job['command']]
            command[3]=str(max(1,int(read(root/'dispatch.json')['deadline_unix']-time.time())))
            with (root/'workers'/f'diagnostic-{revision}-{host}-g{job["gpu"]}.log').open('x') as log:
                process=subprocess.Popen(command,env=env,cwd=helper,stdin=subprocess.DEVNULL,
                                         stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rows.append(dict(gpu=job['gpu'],pid=process.pid,retired=retired,command=command))
        assert len(rows)==6
        save(result,dict(passed=True,host=host,jobs=rows,helper=str(helper),
                         scope='Only6 idle evaluation parents replaced;native training children unchanged.'))
        print(host,[r['pid'] for r in rows],flush=True)
        return
    assert all(read(root/f'diagnostic-activation-{revision}-{h}.json')['passed'] for h in ['dev13','dev15'])
    path=root/'diagnostic-plan.json';plan=read(path)
    preserved=root/(f'diagnostic-plan-{parent_revision}.json' if parent_revision else 'diagnostic-plan-v1.json')
    assert not preserved.exists()
    shutil.copy2(path,preserved)
    corrections={}
    for job in plan['jobs']:
        original=job['work'];assert (Path(original)/'failed.json').exists()
        job['work']=str(root/f'diagnostics-native-{revision}'/job['name'])
        job['original_work']=original
        corrections[job['name']]=dict(original_work=original,work=job['work'],
                    reason='Diagnostic interface/sampling repair. Direct native rollout;at most32 unique windows/task;unavailable cross-episode controls null. No model or primary evaluator changes.')
    plan['helper']=str(helper)
    save(root/('DIAGNOSTIC_CORRECTIONS.json' if revision=='v2' else f'DIAGNOSTIC_CORRECTIONS-{revision}.json'),dict(jobs=corrections,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    temporary=root/f'diagnostic-plan-{revision}.tmp';save(temporary,plan);os.replace(temporary,path)
    print('Published12 corrected read-only diagnostics;all original failures retained',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['activate','publish'],required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--helper',required=True)
    parser.add_argument('--host',choices=['dev13','dev15'])
    parser.add_argument('--revision',default='v2');parser.add_argument('--parent-revision');args=parser.parse_args()
    main(Path(args.root),Path(args.helper),args.host,args.mode,args.revision,args.parent_revision)
