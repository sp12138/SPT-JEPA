"""Finite CPU-only frozen diagnostics,never a training or policy launcher."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def run(a):
    from experiments.metaworld_contact_20261106.launch import environment,PYTHON
    plan=json.loads((a.root/'training-plan.json').read_text());seed=1 if a.host=='dev13'else 2
    source=Path(plan['source']);env=environment(source,0);env['CUDA_VISIBLE_DEVICES']=''
    output=a.root/'diagnostics'/a.host;output.mkdir(parents=True,exist_ok=False)
    intervention=a.root.parent/'metaworld_controlspan_20261106_v2/physical-response/fullseq-free-seed1'
    try:
        for step in [1000,5000,20000]:
            for recipe in [r for r in plan['recipes']if r['seed']==seed]:
                run=Path(recipe['run']);gate=run/f'audit-{step}.json'
                while not gate.exists():
                    assert not(run/'execution-error.json').exists()and time.time()<plan['deadline_unix']
                    time.sleep(10)
                assert json.loads(gate.read_text())['passed']
                name=f'{recipe["arm"]}-seed{seed}-step{step:07d}'
                cmd=[PYTHON,'-B','-u',str(a.script),'--run',str(run),'--step',str(step),
                    '--interventions',str(intervention),'--output',str(output/name),'--device','cpu','--threads','2']
                with(output/(name+'.log')).open('x')as log:
                    subprocess.run(cmd,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                        timeout=max(1,int(plan['deadline_unix']-time.time())),check=True)
        (output/'complete.json').write_text(json.dumps(dict(passed=True,model_updates=0,device='cpu'))+'\n')
    except BaseException as error:
        (output/'execution-error.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--script',type=Path,required=True)
    p.add_argument('--host',choices=['dev13','dev15'],required=True);run(p.parse_args())
