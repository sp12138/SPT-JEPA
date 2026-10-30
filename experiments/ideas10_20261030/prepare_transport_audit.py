"""Freeze eight read-only posterior-transport diagnostics; no training."""
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    b=Path('/efs/prescottsun/SPT-JEPA');code=b/'code';root=b/'runs/ideas10_transport_audit_20261031'
    assert not root.exists()
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=code,check=True,stdout=subprocess.DEVNULL)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
    root.mkdir();source=root/'source-v1';source.mkdir()
    archive=subprocess.check_output(['git','archive',revision,'as_jepa_lewm','as_jepa_tools','experiments/ideas10_20261030'],cwd=code)
    with tarfile.open(fileobj=io.BytesIO(archive))as tar:tar.extractall(source,filter='data')
    write(root/'source-v1.json',dict(source=str(source),source_commit=revision,files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()for p in source.rglob('*')if p.is_file()}))
    src=json.loads((b/'runs/ideas10_response600_20261030/source-record.json').read_text())
    old=src['source'];src=dict(src,source=str(source),source_commit=revision,manifest=str(root/'source-v1.json'),pythonpath=src['pythonpath'].replace(old,str(source)))
    env=json.loads((b/'runs/ideas10_response600_20261030/plan.json').read_text())['environment']
    jobs=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            for stage in ['fresh600','mature']:
                if stage=='fresh600':
                    run=b/f'runs/mechanism_rate600_20261030/{task}-fresh-rate-on-seed{seed}';step=600
                elif task=='cube':
                    run=b/f'runs/hardtasks_cube_epoch1_20261030/cube-temporal-forecast-seed{seed}-epoch1';step=11671
                else:
                    run=b/f'runs/hardtasks_learn2000_20261030/tworoom-temporal-forecast-seed{seed}-learn2000';step=2000
                assert (run/f'model-step{step:07d}/weights.pt').exists()
                name=f'{task}-{stage}-seed{seed}'
                jobs.append(dict(id=name,host='dev2'if task=='tworoom'else'dev13',gpu=4+(seed-1)*2+(stage=='mature'),source=src,output=str(root/name),
                    command=['/efs/prescottsun/project_jepa/envs/lewm/bin/python','-B','-u',str(source/'experiments/ideas10_20261030/root/audit_noise_transport.py'),'--run',str(run),'--step',str(step),'--output',str(root/name)]))
    write(root/'plan.json',dict(root=str(root),deadline=None,environment=env,jobs=jobs))
    write(root/'protocol.json',dict(source_commit=revision,scope='Eight frozen-checkpoint eval-only diagnostics. No optimizer, no additional training, no parameter mutation.32fixed validation windows x16MC draws.',
        hypothesis='Fixed scalar posterior correlation may regularize clean dynamics toward the wrong Jacobian. Quantify, rather than infer from synthetic rotations.',
        limitations='Transport is a different posterior; realized noise growth is reported. Existing rate-off results do not establish this is the main planning bottleneck. Structured variational predecessors must be credited.'))
    print(json.dumps(dict(root=str(root),source_commit=revision,jobs=len(jobs))))


if __name__=='__main__':main()
