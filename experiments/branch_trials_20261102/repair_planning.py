"""Repeat only the failed PushT CLI integration; preserve original failures."""
import argparse
import os
from pathlib import Path
import subprocess

from common import B,PY,read,write,sha,source_check


def main(root,ident,gpu):
    root=Path(root);protocol=read(root/'protocol.json');source=protocol['source'];source_check(source)
    row=next(r for r in protocol['runs'] if r['id']==ident)
    assert row['task']=='pusht' and row['arm']=='function'
    assert read(root/'audits'/f'{ident}-step2/audit.json')['passed']
    original=read(root/'worker-records'/f'{ident}-preflight.json')
    assert not original['passed'] and [r['returncode'] for r in original['results']]==[0,0,2]
    memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
    assert memory<=1
    cfg=read(Path(row['run'])/'config.json');out=root/'evaluations'/f'{ident}-step2-primary-cli-repair'
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,**protocol['environment'],PYTHONPATH=source['pythonpath'],CUDA_VISIBLE_DEVICES=str(gpu),
             MUJOCO_EGL_DEVICE_ID=str(gpu),ASJEPA_QUERY_AUDIT=str(out/'query-objective-proof.json'),PYTHONDONTWRITEBYTECODE='1')
    cmd=[PY,'-B','-u','-m','as_jepa_tools.planning','--checkpoint',str(Path(row['run'])/'model-step0000002'),
         '--output',str(out),'--upstream',str(B/'sources/stable-worldmodel'),
         '--data-root',str(B/'data/hardtasks-v1/planning'),'--seed','42','--episodes','2','--split','heldout',
         '--test-episodes',str(Path(cfg['data'])/'test_episodes.npy')]
    with (out/'repair.log').open('x') as f:
        child=subprocess.run(cmd,cwd=source['source'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
    assert child.returncode==0,(ident,child.returncode,str(out))
    proof=read(out/'query-objective-proof.json')
    assert proof['objective']=='GoalQueryObjective' and proof['finite'] and proof['cost_shape'][-1]==300
    assert (out/'metrics.json').exists()
    write(root/'repairs'/f'{ident}-preflight.json',dict(passed=True,source=source,command=cmd,
          original_failure_sha256=sha(root/'worker-records'/f'{ident}-preflight.json'),
          objective=proof,checkpoint_sha256=sha(Path(row['run'])/'model-step0000002/weights.pt'),
          source_only_change='PushT CLI override ordering;model/training/data unchanged',returncode=child.returncode))
    print(dict(passed=True,id=ident))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);p.add_argument('--gpu',required=True,type=int);a=p.parse_args();main(a.root,a.id,a.gpu)
