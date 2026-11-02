"""Finite8-slot diagnostic queue,one GPU per job,owned jobs only."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

from common import PY,read,write,source_check


def main(root,host):
    root=Path(root);p=read(root/'protocol.json');source_check(p['source'])
    pending=[j for j in p['jobs'] if j['host']==host];running={};results=[]
    (root/'logs').mkdir(exist_ok=True);write(root/'dispatchers'/f'{host}-start.json',dict(pid=os.getpid(),jobs=len(pending),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    while pending or running:
        for gpu,(child,log,job,start) in list(running.items()):
            code=child.poll()
            if code is None and time.monotonic()-start>1800:
                child.terminate();child.wait(timeout=30);code=124
            if code is not None:
                log.close();results.append(dict(id=job['id'],gpu=gpu,returncode=code));del running[gpu]
        for gpu in range(8):
            if not pending or gpu in running:continue
            memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
            assert memory<=1,(gpu,memory)
            job=pending.pop(0);env=dict(os.environ,PYTHONPATH=p['source']['pythonpath'],CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            command=[PY,'-B','-u',p['helper']['source']+'/experiments/lingjing_fullspec_20261102/d0_job.py','--root',str(root),'--id',job['id']]
            log=(root/'logs'/(job['id']+'.log')).open('x');child=subprocess.Popen(command,env=env,cwd=p['source']['source'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(root/'processes'/(job['id']+'.json'),dict(pid=child.pid,gpu=gpu,host=host,command=command))
            running[gpu]=(child,log,job,time.monotonic())
        if pending or running:time.sleep(5)
    write(root/'dispatchers'/f'{host}-finished.json',dict(passed=all(r['returncode']==0 for r in results),results=results,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(dict(host=host,complete=len(results),failures=sum(r['returncode']!=0 for r in results)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',required=True);a=p.parse_args();main(a.root,a.host)
