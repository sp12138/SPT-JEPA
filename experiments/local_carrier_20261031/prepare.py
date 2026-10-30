"""Prepare only: fixed local/global1024 calibration, memory gate, reviewed600 gate."""
import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
CAL=B/'runs/local_carrier_calibration_20261031'
PRE=B/'runs/local_carrier_preflight_20261031'
ROOT=B/'runs/local_carrier600_20261031'
OLD=B/'runs/ideas_validate_calibration_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF='experiments/local_carrier_20261031/prepare.py'


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def assignments():
    for gpu,(task,seed,arm) in enumerate(itertools.product(('tworoom','cube'),(1,2),('local','dense'))):
        yield dict(id=f'{task}-spatial-{arm}-seed{seed}',host='dev15',gpu=gpu,task=task,
                   training_seed=seed,arm=arm,family='spatial_carrier')


def source(record):
    src=read(record);root=Path(src['source']);manifest=read(src['manifest'])
    assert manifest['source_commit']==src['source_commit']
    required=['as_jepa_lewm/spatial_carrier.py','as_jepa_lewm/factory.py',SELF,
              'experiments/local_carrier_20261031/calibrate.py',
              'experiments/ideas_validate_20261031/carrier/calibrate.py',
              'experiments/ideas_validate_20261031/preflight.py','experiments/ideas_validate_20261031/audit.py']
    for name in required:assert digest(root/name)==manifest['files'][name],name
    assert digest(__file__)==digest(root/SELF),'Execute the frozen preparation helper'
    return src


def calibration(args):
    assert not CAL.exists() and not PRE.exists() and not ROOT.exists()
    src=source(args.source_record);CAL.mkdir();write(CAL/'source-record.json',src)
    env=read(B/'runs/ideas_validate600_20261031/plan.json')['environment'];jobs=[]
    for gpu,(task,seed) in enumerate(itertools.product(('tworoom','cube'),(1,2))):
        name=f'calibrate-{task}-seed{seed}';out=CAL/name;out.mkdir()
        reference=OLD/name/'calibration.json';assert read(reference)['measured_frames']==128
        jobs.append(dict(id=name,host='dev15',gpu=gpu,task=task,training_seed=seed,phase='calibration',
            source=src,output=str(out),command=[PY,'-B','-u',str(Path(src['source'])/'experiments/local_carrier_20261031/calibrate.py'),
            '--reference-calibration',str(reference),'--output',str(out/'calibration.json'),
            '--source-commit',src['source_commit'],'--device','cuda','--batch-size','8']))
    write(CAL/'plan.json',dict(root=str(CAL),deadline=None,environment=env,jobs=jobs))
    write(CAL/'protocol.json',dict(source_commit=src['source_commit'],created_at=time.time(),
        frames_per_task_seed=128,one_shared_encoder_pass=True,split='train',
        reference='Exact initial frames/episodes/rows from original Carrier calibration; no physical labels or validation.'))
    print(json.dumps(dict(calibration=str(CAL),jobs=4)))


def calibration_ready(src):
    plan=read(CAL/'plan.json');assert read(CAL/'source-record.json')==src
    assert len(plan['jobs'])==4
    for job in plan['jobs']:
        assert read(CAL/'exits'/(job['id']+'.json'))['returncode']==0
        result=read(Path(job['output'])/'calibration.json')
        assert result['source_commit']==src['source_commit'] and result['measured_frames']==128
        assert result['model_seed']==job['training_seed']
        assert set(result['arms'])=={'local','dense'}
    return plan['environment']


def config(allocation,out,step):
    task,seed,arm=allocation['task'],allocation['training_seed'],allocation['arm']
    path=CAL/f'calibrate-{task}-seed{seed}'/'calibration.json';stats=read(path)
    cfg=read(stats['config']);assert digest(stats['config'])==stats['config_sha256']
    assert cfg['seed']==seed and cfg['batch_size']==128 and cfg['max_steps']==124910
    assert cfg['model']['beta']==.25 and cfg['model']['sigma0']==.1
    assert cfg['model']['rho']==.8 and cfg['model']['forecast_fraction']==.5
    cfg=copy.deepcopy(cfg);cfg.update(output=str(out),model_class='spatial_carrier',record_initial_weights=True,
        checkpoint_every=2 if step==2 else 200,retain_steps=[2] if step==2 else [200,400,600],
        spatial_calibration=dict(path=str(path),sha256=digest(path)),
        audit_expectations=dict(gradients=dict(action_encoder='active'),
            initial_unchanged_prefixes=['spatial_projection','spatial_mean','spatial_scale']))
    cfg['model'].update(stats['arms'][arm]['model_kwargs'])
    assert cfg['model']['state_slots']==64 and cfg['model']['state_width']==16
    assert not cfg.get('initial_weights') and not cfg.get('initial_encoder_checkpoint')
    return cfg


def preflight(args):
    src=source(args.source_record);env=calibration_ready(src);assert not PRE.exists();PRE.mkdir()
    write(PRE/'source-record.json',src);jobs=[]
    for allocation in assignments():
        name=allocation['id'];out=PRE/name;out.mkdir();cfg=config(allocation,out,2);cp=out/'config.json';write(cp,cfg)
        common={k:v for k,v in allocation.items() if k!='id'};common['source']=src
        jobs.append(dict(**common,id=name,output=str(out),config=str(cp),config_sha256=digest(cp),phase='train2',
            command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(cp),'--stop-step','2']))
        dest=PRE/('gradient-'+name);local=f"{allocation['task']}-spatial-local-seed{allocation['training_seed']}"
        jobs.append(dict(**common,id=dest.name,output=str(dest),phase='gradient',depends_on=sorted(set([name,local])),
            command=[PY,'-B','-u',str(Path(src['source'])/'experiments/ideas_validate_20261031/preflight.py'),
                '--run',str(out),'--output',str(dest),'--expected-step','2',
                '--reference-initial',str(PRE/local/'initial-model/weights.pt')]))
    write(PRE/'plan.json',dict(root=str(PRE),deadline=None,environment=env,jobs=jobs))
    write(PRE/'protocol.json',dict(created_at=time.time(),source_commit=src['source_commit'],batch_size=128,
        require='All16 exits0, fullstate/frozen/gradient/pairedinitial pass, actual batch128 peak memory reviewed. '
                'Write review.json with passed=true before creating the separate600 plan. No automatic batch reduction.'))
    print(json.dumps(dict(preflight=str(PRE),jobs=16)))


def training(args):
    src=source(args.source_record);env=calibration_ready(src);assert not ROOT.exists()
    assert read(PRE/'source-record.json')==src and read(PRE/'review.json')['passed'] is True
    preplan=read(PRE/'plan.json');assert len(preplan['jobs'])==16
    for job in preplan['jobs']:
        assert read(PRE/'exits'/(job['id']+'.json'))['returncode']==0
        if job['phase']=='gradient':assert read(Path(job['output'])/'preflight.json')['passed']
    ROOT.mkdir();write(ROOT/'source-record.json',src);jobs=[];records=[]
    for allocation in assignments():
        name=allocation['id'];out=ROOT/name;out.mkdir();cfg=config(allocation,out,600);cp=out/'config.json';write(cp,cfg)
        common={k:v for k,v in allocation.items() if k!='id'};common['source']=src
        jobs.append(dict(**common,id=name,output=str(out),config=str(cp),config_sha256=digest(cp),phase='train600',
            command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(cp),'--stop-step','600']))
        previous=name
        for phase,module in [('health','as_jepa_tools.benchmark_health'),('info','as_jepa_tools.benchmark_information_audit')]:
            dest=ROOT/(phase+'-'+name)
            jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase=phase,
                command=[PY,'-B','-u','-m',module,'--run',str(out),'--output',str(dest)]));previous=dest.name
        dest=ROOT/('transition-'+name)
        jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase='transition',
            command=[PY,'-B','-u',str(Path(src['source'])/'experiments/ideas_validate_20261031/audit.py'),
                '--run',str(out),'--output',str(dest),'--cases','64','--manifest',
                str(Path(src['source'])/f"experiments/ideas_validate_20261031/audit-windows-{allocation['task']}.json")]))
        previous=dest.name;dest=ROOT/('planning-'+name)
        jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase='planning',
            environment=dict(MUJOCO_EGL_DEVICE_ID=str(allocation['gpu'])),
            command=[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',
                str(Path(cfg['data'])/'manifests/validation-strict.json'),'--checkpoint',str(out/'model-step0000600'),
                '--data',cfg['data'],'--output',str(dest)]))
        records.append(dict(**allocation,config_sha256=digest(cp),calibration=cfg['spatial_calibration']))
    write(ROOT/'plan.json',dict(root=str(ROOT),deadline=None,environment=env,jobs=jobs))
    write(ROOT/'protocol.json',dict(created_at=time.time(),source_commit=src['source_commit'],stop_step=600,
        runs=records,preflight_review_sha256=digest(PRE/'review.json'),
        purpose='Fixed local addressed channel compression versus fixed global compression; matched1024-D learned model.',
        unchanged='Native sample-level TF loss, beta.25/sigma.1, Trainer/checkpoint/data/planner; no learned chart/head or privileged labels.',
        controls='local/dense learned tensors and initialization RNG identical; frozen encoder same seed. Fixed buffers and projection compute differ.',
        limits='Old64-D controls differ in learned capacity, state/rate dimension and compute. Original random-feature readout improved position, '
               'not established pose recovery.600 is inside1249 warmup. No automatic continuation or historical hold changes.'))
    print(json.dumps(dict(root=str(ROOT),jobs=len(jobs))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibration','preflight','training'])
    p.add_argument('--source-record',required=True);args=p.parse_args()
    dict(calibration=calibration,preflight=preflight,training=training)[args.phase](args)
