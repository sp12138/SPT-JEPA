"""Prepare separate immutable calibration, preflight and24-slot short-gate plans."""
import argparse
import copy
import hashlib
import io
import json
import subprocess
import tarfile
import time
from pathlib import Path


B = Path('/efs/prescottsun/SPT-JEPA')
CODE = B/'code'
ROOT = B/'runs/ideas_validate600_20261031'
CAL = B/'runs/ideas_validate_calibration_20261031'
PRE = B/'runs/ideas_validate_preflight_20261031'
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
REFERENCE = B/'runs/mechanism_rate600_20261030'
ARMS = [('carrier', 'identity'), ('carrier', 'unconstrained'), ('carrier', 'bounded'),
        ('pathquery', 'symbol'), ('pathquery', 'completion'), ('pathquery', 'transform')]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(str(path))
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration():
    assert not ROOT.exists() and not CAL.exists() and not PRE.exists()
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=CODE, check=True,
                   stdout=subprocess.DEVNULL)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    ROOT.mkdir(); CAL.mkdir()
    source = ROOT/'source-v1'; source.mkdir()
    archive = subprocess.check_output(['git', 'archive', commit, 'as_jepa_lewm', 'as_jepa_tools',
        'tests/as_jepa', 'experiments/ideas_validate_20261031'], cwd=CODE)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter='data')
    write(ROOT/'source-v1.json', dict(source=str(source), source_commit=commit,
        files={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}))
    old = read(B/'runs/ideas10_relative600_20261031/source-record.json')
    src = dict(old, source=str(source), source_commit=commit, manifest=str(ROOT/'source-v1.json'),
               pythonpath=old['pythonpath'].replace(old['source'], str(source)))
    environment = read(B/'runs/ideas10_relative600_20261031/plan.json')['environment']
    write(ROOT/'source-record.json', src)
    jobs=[]
    for task,seed,host,gpu in [('tworoom',1,'dev2',0),('tworoom',2,'dev13',0),
                               ('cube',1,'dev15',0),('cube',2,'dev15',1)]:
        name=f'calibrate-{task}-seed{seed}';out=CAL/name;out.mkdir()
        cfg=REFERENCE/f'{task}-fresh-rate-on-seed{seed}'/'config.json'
        jobs.append(dict(id=name,host=host,gpu=gpu,source=src,output=str(out),
            phase='training-only-calibration',task=task,training_seed=seed,
            command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/carrier/calibrate.py'),
                     '--config',str(cfg),'--output',str(out/'calibration.json'),'--device','cuda',
                     '--source-commit',commit,'--episodes','128','--batch-size','8','--selection-seed','1042']))
    write(CAL/'plan.json',dict(root=str(CAL),deadline=None,environment=environment,jobs=jobs))
    write(CAL/'protocol.json',dict(created_at=time.time(),source_commit=commit,
        frames_per_task_seed=128,split='train',reused_across_three_carrier_arms=True,
        scope='Only frozen initial encoder measurements; no fitting, validation images, physical or optimal-action labels.'))
    print(json.dumps(dict(calibration=str(CAL),root=str(ROOT),source_commit=commit,jobs=len(jobs))))


def training():
    assert ROOT.exists() and CAL.exists() and not PRE.exists()
    assert not (ROOT/'plan.json').exists()
    calplan=read(CAL/'plan.json')
    exits=[read(CAL/'exits'/(j['id']+'.json')) for j in calplan['jobs']]
    assert len(exits)==4 and all(x['returncode']==0 for x in exits)
    src=read(ROOT/'source-record.json');source=Path(src['source'])
    # This preparation code must be exactly the archived version used by jobs.
    assert Path(__file__).read_bytes()==(source/'experiments/ideas_validate_20261031/prepare.py').read_bytes()
    environment=calplan['environment'];PRE.mkdir()
    jobs=[];prejobs=[];records=[]
    for ti,task in enumerate(['tworoom','cube']):
        for seed in [1,2]:
            reference=REFERENCE/f'{task}-fresh-rate-on-seed{seed}'
            calibration_path=CAL/f'calibrate-{task}-seed{seed}'/'calibration.json'
            stats=read(calibration_path)
            assert stats['source_commit']==src['source_commit'] and stats['measured_frames']==128
            assert stats['model_seed']==seed and stats['dataset']==read(reference/'config.json')['data']
            for ai,(family,arm) in enumerate(ARMS):
                name=f'{task}-{family}-{arm}-seed{seed}';out=ROOT/name;out.mkdir()
                cfg=read(reference/'config.json')
                cfg.update(output=str(out),model_class=family,checkpoint_every=200,
                           retain_steps=[200,400,600],record_initial_weights=True,
                           audit_expectations={'gradients':{'action_encoder':'active'}})
                if family=='carrier':
                    cfg['model'].update(carrier_chart=arm,carrier_bound=.5,carrier_hidden=64,**stats['model_kwargs'])
                    cfg['carrier_calibration']=dict(path=str(calibration_path),sha256=digest(calibration_path))
                else:
                    cfg['model'].update(path_query_mode=arm,path_query_weight=1.,path_query_seed=20261031)
                config=out/'config.json';write(config,cfg)
                slot=ti*12+(seed-1)*6+ai;host=['dev2','dev13','dev15'][slot//8];gpu=slot%8
                common=dict(host=host,gpu=gpu,source=src,task=task,family=family,arm=arm,training_seed=seed)
                jobs.append(dict(**common,id=name,output=str(out),config=str(config),config_sha256=digest(config),
                    phase='train600',command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(config),'--stop-step','600']))
                previous=name
                for phase,module in [('health','as_jepa_tools.benchmark_health'),('info','as_jepa_tools.benchmark_information_audit')]:
                    dest=ROOT/(phase+'-'+name)
                    jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase=phase,
                        command=[PY,'-B','-u','-m',module,'--run',str(out),'--output',str(dest)]))
                    previous=dest.name
                dest=ROOT/('transition-'+name)
                jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase='transition',
                    command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/audit.py'),
                             '--run',str(out),'--output',str(dest),'--cases','64',
                             '--manifest',str(source/f'experiments/ideas_validate_20261031/audit-windows-{task}.json')]))
                previous=dest.name;dest=ROOT/('planning-'+name)
                jobs.append(dict(**common,id=dest.name,output=str(dest),depends_on=[previous],phase='planning',
                    environment={'MUJOCO_EGL_DEVICE_ID':str(gpu)},command=[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate',
                        '--manifest',str(Path(cfg['data'])/'manifests/validation-strict.json'),
                        '--checkpoint',str(out/'model-step0000600'),'--data',cfg['data'],'--output',str(dest)]))
                records.append(dict(id=name,task=task,seed=seed,family=family,arm=arm,
                    original_control=str(reference),config_sha256=digest(config),host=host,gpu=gpu))
                if seed==1:
                    po=PRE/name;po.mkdir();pcfg=copy.deepcopy(cfg)
                    pcfg.update(output=str(po),batch_size=2,workers=0,checkpoint_every=2,retain_steps=[2])
                    pc=po/'config.json';write(pc,pcfg)
                    # Four checks per host, independently occupied before training.
                    pre_slot=ti*6+ai;ph=['dev2','dev13','dev15'][pre_slot//4];pg=pre_slot%4
                    c=dict(host=ph,gpu=pg,source=src,task=task,family=family,arm=arm)
                    prejobs.append(dict(**c,id=name,output=str(po),config=str(pc),config_sha256=digest(pc),
                        command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(pc),'--stop-step','2']))
                    go=PRE/('gradient-'+name)
                    prejobs.append(dict(**c,id=go.name,output=str(go),depends_on=[name],
                        command=[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/preflight.py'),
                            '--run',str(po),'--output',str(go),'--expected-step','2',
                            '--reference-initial',str(reference/'initial-model/weights.pt')]))
    write(ROOT/'plan.json',dict(root=str(ROOT),deadline=None,environment=environment,jobs=jobs))
    write(PRE/'plan.json',dict(root=str(PRE),deadline=None,environment=environment,jobs=prejobs))
    write(ROOT/'protocol.json',dict(created_at=time.time(),source_commit=src['source_commit'],
        root=str(ROOT),preflight=str(PRE),calibration=str(CAL),runs=records,stop_step=600,
        purpose='Validate two conditional ideas with six direct-control arms, not claim six or ten innovations.',
        carrier='Frozen initialized spatialRP64, fixed train-only coordinate statistics; identical identity/free/bounded residual charts at initialization. Report total/frozen/trainable parameters and whether the bound actually activates.',
        pathquery='Same auxiliary head and causal rollout; symbols/scalar-completion/full-path transformations, within-clip RGB queries only. No future state or query enters the predictor. CE normalized by log(classcount), globalweight1.',
        fairness='Same task split, seeds, original550initialtensors, underlyingTF objective, native50caseCEM/evaluator. Extra parameters and compute matched within each family; different acrossfamilies. Calibration access explicitly recorded.',
        required_review=['fullstate_and_initial_tensor_identity','frozen_and_active_gradients','state_information_and_generator_use',
            'teacherforced_vs_shiftedcarry_vs_causalforecast','samecase_native_planning_all_seeds'],
        limitations=['600updates remain inside1249warmup; neither superiority nor permanentfailure follows from one early gate.',
            'Carrier cannot recover information missing from its frozen measurement; spatialcarrier statistics use128trainingframes per task/seed.',
            'Within-clip path queries have known static/query-only shortcuts; complete transform requireshalves to expose allclasses.',
            'Class counts and active output rows differ; equalhead parameters do not imply identical effectivecapacity.',
            'Noautomatic longertraining or historicalgridresume. Preserveallcheckpoints/holds.']))
    print(json.dumps(dict(root=str(ROOT),preflight=str(PRE),jobs=len(jobs),preflight_jobs=len(prejobs))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['calibration','training'])
    args=parser.parse_args()
    calibration() if args.phase=='calibration' else training()
