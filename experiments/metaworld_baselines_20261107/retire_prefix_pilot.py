"""Retire only the biased short-window pilot; preserve valid window8 learners."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

ROOT=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_baselines_20261107')
SHORT={'lewm','smwm','ac-mtm','lpwm'}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def host_stop(host):
    assert host in ['dev13','dev15','dev6']
    plan=json.loads((ROOT/'plan.json').read_text())
    pilots=[j for j in plan['jobs'] if j['method'] in SHORT]
    allowed=[j['output'] for j in pilots]
    processes={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().split()
            processes[int(p.name)]=dict(ppid=int(fields[3]),cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
        except (OSError,UnicodeError):continue
    roots=set();records=[]
    for job in pilots:
        f=ROOT/'claims-v3'/(job['name']+'.json')
        if not f.exists():continue
        claim=json.loads(f.read_text())
        if claim['host']!=host:continue
        pid=claim['pid'];cmd=processes.get(pid,{}).get('cmd','')
        if not cmd:continue
        assert 'metaworld_baselines_20261107' in cmd and '--mode work' in cmd
        roots.add(pid)
        out=Path(job['output']);latest=out/'latest.json'
        records.append(dict(name=job['name'],pid=pid,latest_saved=json.loads(latest.read_text()) if latest.exists() else None))
    # Idle transfer controllers are replaced by the corrected formal dispatcher.
    transfer=ROOT/f'launch-threehost-v5-{host}.json'
    if transfer.exists():
        for owner in json.loads(transfer.read_text())['workers']:
            pid=owner['pid'];cmd=processes.get(pid,{}).get('cmd','')
            if not cmd:continue
            assert 'threehost-v5.py' in cmd
            active=[f for f in (ROOT/'threehost-recovery/claims').glob('*.json') if json.loads(f.read_text()).get('pid')==pid] if (ROOT/'threehost-recovery/claims').exists() else []
            assert not active,'An active transfer requires explicit adoption instead of idle-controller retirement'
            roots.add(pid)
    native=ROOT/'native-controller'/f'launch-v4-{host}.json'
    assert not (ROOT/'native-controller/claims').exists()
    if native.exists():
        for owner in json.loads(native.read_text())['workers']:
            pid=owner['pid'];cmd=processes.get(pid,{}).get('cmd','')
            if cmd:assert 'native-queue-v4.py' in cmd;roots.add(pid)
    selected=set(roots)
    for _ in range(12):selected.update(pid for pid,p in processes.items() if p['ppid'] in selected)
    for pid in selected-roots:
        assert any(path in processes[pid]['cmd'] for path in allowed),(pid,processes[pid]['cmd'])
    for pid in roots:
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    for pid in sorted(selected-roots,reverse=True):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    time.sleep(3)
    alive=[]
    for pid in selected:
        p=Path(f'/proc/{pid}/cmdline')
        try:
            if p.exists() and p.read_bytes():alive.append(pid)
        except OSError:pass
    save(ROOT/'coverage-recovery'/f'retired-{host}.json',dict(host=host,parents=records,signaled=sorted(selected),
        still_alive=alive,preserved_methods=['lewm-idm','h-jepa'],reason='Native shorter windows need all valid starts;prefix pilot withheld terminal observations.'))
    assert not alive,alive
    print(json.dumps(dict(host=host,retired_processes=len(selected),retired_learners=len(records))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);a=p.parse_args();host_stop(a.host)
