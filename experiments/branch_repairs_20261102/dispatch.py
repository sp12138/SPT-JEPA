"""Finite per-host8-slot queue;exits afterits assigned jobs,no recurringmonitor."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

from common import PY,read,write,source_check


def main(root,host,phase):
    root=Path(root);protocol=read(root/'protocol.json');source=protocol['source'];source_check(source)
    jobs=[r for r in protocol['runs'] if r['host']==host];pending=list(jobs);running={};finished=[]
    worker=Path(protocol['helper']['source'])/'experiments/branch_repairs_20261102/worker.py'
    env=dict(os.environ,PYTHONPATH=source['pythonpath'],PYTHONDONTWRITEBYTECODE='1')
    write(root/'dispatchers'/f'{host}-{phase}-start.json',dict(pid=os.getpid(),jobs=[r['id'] for r in jobs],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase))
    while pending or running:
        for gpu,(child,log,row) in list(running.items()):
            code=child.poll()
            if code is not None:
                log.close();finished.append(dict(id=row['id'],gpu=gpu,returncode=code));del running[gpu]
        for gpu in range(8):
            if pending and gpu not in running:
                row=pending.pop(0);path=root/'logs'/f'{row["id"]}-{phase}-worker.log';path.parent.mkdir(exist_ok=True)
                log=path.open('x');cmd=[PY,'-B','-u',str(worker),'--root',str(root),'--id',row['id'],'--gpu',str(gpu),'--phase',phase]
                child=subprocess.Popen(cmd,cwd=source['source'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(child,log,row)
        if pending or running:time.sleep(5)
    write(root/'dispatchers'/f'{host}-{phase}-finished.json',dict(passed=all(r['returncode']==0 for r in finished),results=finished,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(dict(host=host,phase=phase,complete=len(finished),failures=sum(r['returncode']!=0 for r in finished)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--host',choices=['dev2','dev13','dev15'],required=True);p.add_argument('--phase',choices=['preflight','gate'],required=True);a=p.parse_args();main(a.root,a.host,a.phase)
