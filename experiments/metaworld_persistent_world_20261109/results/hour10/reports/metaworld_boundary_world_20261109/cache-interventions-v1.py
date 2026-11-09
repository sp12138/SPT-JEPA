"""Copy and verify the additional ordinary-data corpus on one host's local SSD."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime,timezone

BASE=Path('/efs/prescottsun/SPT-JEPA')
SOURCE=BASE/'data/metaworld-random-interventions-native128-20261109-v1'
TARGET=Path('/tmp/prescottsun-random-interventions-native128-20261109-v1')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(host):
    assert json.loads((SOURCE/'manifest.json').read_text())['passed'];TARGET.mkdir(exist_ok=True);rows=[]
    for shard in range(12):
        src=SOURCE/str(shard);dst=TARGET/str(shard);dst.mkdir(exist_ok=True)
        d=json.loads((src/'manifest.json').read_text())
        for name,expected in d['files'].items():
            assert sha(src/name)==expected
            target=dst/name
            if not target.exists() or sha(target)!=expected:
                temp=dst/(name+f'.{os.getpid()}.tmp');shutil.copyfile(src/name,temp);assert sha(temp)==expected;os.replace(temp,target)
            rows.append(dict(shard=shard,file=name,sha256=expected,bytes=target.stat().st_size))
        shutil.copyfile(src/'manifest.json',dst/'manifest.json')
    shutil.copyfile(SOURCE/'manifest.json',TARGET/'manifest.json')
    result=dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),host=host,source=str(SOURCE),target=str(TARGET),
        manifest_sha256=sha(SOURCE/'manifest.json'),files=rows,all24array_SHAs_verified=True,optimizer_updates=0)
    report=BASE/'reports/metaworld_boundary_world_20261109'/f'INTERVENTION-CACHE-{host}.json';report.write_text(json.dumps(result,indent=2)+'\n')
    print(host,'verified',sum(r['bytes'] for r in rows),'bytes',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--host',required=True);main(a.parse_args().host)
