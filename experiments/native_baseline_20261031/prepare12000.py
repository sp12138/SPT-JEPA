"""Prepare ONLY two held native Cube6000 -> 12000 maturity references; no launch."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

B = Path('/efs/prescottsun/SPT-JEPA')
PARENT = B/'runs/native_lewm6000_20261031'
INITIAL = B/'runs/native_lewm600_20261031'
ROOT = B/'runs/native_cube12000_20261031'
REPORT = B/'reports/native-lewm6000-20261031/results.json'
FRESH_ROOT = B/'runs/native_rawfresh6000_20261031'
STEPS = (8000, 10000, 12000)
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'


def read(path): return json.loads(Path(path).read_text())


def load_helpers():
    """Reuse the already committed native audit and cross-epoch sampler helper."""
    record=read(PARENT/'helper-source-record.json')
    assert record['source_commit']=='a8b77922f26a7ee9397d4bbad78e65649c361eef'
    path=Path(record['source'])/'experiments/native_baseline_20261031/prepare6000.py'
    spec=importlib.util.spec_from_file_location('_frozen_native6000_preparation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=read(record['manifest'])
    for relative in ['experiments/native_baseline_20261031/prepare6000.py',
                     'experiments/native_baseline_20261031/bn_calibrate.py',
                     'experiments/carrier_progress_20261031/prepare_next.py',
                     'experiments/carrier_progress_20261031/prepare_learning.py']:
        assert module.digest(Path(record['source'])/relative)==manifest['files'][relative]
    source=read(PARENT/'source-record.json')
    assert source['source_commit']=='7a1ab4feb1a44cb3a6a37f5f7f2ad6a0e110487f'
    for manifest_key in ['manifest','runtime_manifest']:
        frozen=read(source[manifest_key])
        for relative,h in frozen['files'].items():
            assert module.digest(Path(frozen['source'])/relative)==h
    sys.path[:0]=source['pythonpath'].split(':')
    return module,source,record


def audit(base,run,step,parent):
    import torch
    from as_jepa_tools.benchmark_data import make_training_dataset
    torch.set_num_threads(2)
    report=base.audit(run,step,parent)
    cfg=read(run/'config.json');assert cfg['seed'] in (1,2)
    assert cfg['data']==str(B/'data/hardtasks-v1/cube-metadata')
    assert cfg['batch_size']==128 and cfg['model']['window']==8 and cfg['model']['action_dim']==25
    assert cfg['precision']=='bf16-mixed' and cfg['float32_matmul_precision']=='high'
    ds=make_training_dataset(cfg)
    try:
        cursors={str(s):base.resume_sequence(ds,s*cfg['batch_size'],cfg['seed'],cfg['batch_size']) for s in (6000,*STEPS)}
        assert all(r['passed'] for r in cursors.values())
    finally:
        if getattr(ds,'file',None) is not None:ds.file.close()
    report.update(sampler_at_stage_boundaries=cursors,
        operation='Original full native state; additional stop/archive/resume boundaries at8000 and10000. No uninterrupted-bitwise trajectory claim.')
    return report


def archive(args):
    base,source,helpers=load_helpers();run,parent,out=Path(args.run),Path(args.parent_run),Path(args.output)
    assert args.step in STEPS and not (out/'fullstate.json').exists()
    expected_initial={8000:6000,10000:8000,12000:10000}[args.step]
    stage=read(run/'stage-complete.json')
    assert stage['stage_complete'] and stage['initial_step']==expected_initial and stage['stop_step']==args.step
    assert stage['scheduler_total_steps']==124910 and stage['config_sha256']==base.digest(run/'config.json')
    report=audit(base,run,args.step,parent)
    destination=run/f'checkpoint-step{args.step:07d}.ckpt'
    assert not destination.exists();shutil.copy2(run/'last.ckpt',destination)
    assert base.digest(destination)==report['checkpoint_sha256']
    report.update(archived_checkpoint=str(destination),stage_complete=stage,
        native_training_status=read(run/'status.json'),native_dispatch=read(run/'dispatch.json'),
        model_source=source,reused_helpers=helpers,archive_helper_sha256=base.digest(__file__))
    base.write(out/'fullstate.json',report)
    print(json.dumps(dict(passed=True,step=args.step,archive=str(destination))))


def prepare(args):
    base,source,helpers=load_helpers();root=Path(args.root);helper=Path(args.helper_root).resolve()
    assert not root.exists() and (PARENT/'HOLD_DISPATCH').exists()
    assert (FRESH_ROOT/'HOLD_DISPATCH').exists()
    oldplan=read(PARENT/'plan.json');oldjobs={j['id']:j for j in oldplan['jobs']}
    review=read(REPORT);assert review['completed_endpoints']=={'native6000':32,'rawfresh':4}
    rows=[];fresh=B/'runs/carrier_cube_freshcases_20261031/fresh-manifest.json'
    train_manifest=B/'reports/native-lewm600-20261031/source-bncal-v1/train128-cube.json'
    for seed in (1,2):
        name=f'cube-native-lewm-seed{seed}';parent=PARENT/name
        for job in oldplan['jobs']:
            if job.get('task')=='cube' and job.get('training_seed')==seed:
                assert read(PARENT/'exits'/(job['id']+'.json'))['returncode']==0 and job['source']==source
        fresh_job='fresh-planning-raw-'+name
        assert read(FRESH_ROOT/'exits'/(fresh_job+'.json'))['returncode']==0
        cfg=read(parent/'config.json');assert base.digest(parent/'config.json')==oldjobs['learn6000-'+name]['config_sha256']
        health=read(PARENT/f'fullstate-{name}/preflight.json');arch=read(PARENT/f'archive-{name}/fullstate.json')
        state=read(PARENT/f'transition-{name}/state-audit.json');status=read(parent/'status.json')
        assert health['passed'] and arch['passed'] and status['state']=='finished' and status['step']==6000
        assert base.digest(parent/'last.ckpt')==base.digest(parent/'checkpoint-step0006000.ckpt')==arch['checkpoint_sha256']==health['checkpoint_sha256']
        assert base.digest(parent/'model-step0006000/weights.pt')==health['weights_sha256']==state['weights_sha256']
        report=audit(base,parent,6000,parent)
        assert report['checkpoint_sha256']==arch['checkpoint_sha256']
        original=next(r for r in review['rows'] if r['id']==name)
        assert original['fullstate_passed'] and original['model_size']['trainable']==18035396
        primary=Path(cfg['data'])/'manifests/validation-strict.json';fm=read(fresh)
        assert fm['task']=='cube' and fm['campaign_split']=='validation' and len(fm['pairs'])==50
        assert fm['fresh_selection']['primary_manifest_sha256']==base.digest(primary)
        assert fm['metadata_card_sha256']==base.digest(Path(cfg['data'])/'dataset_card.json')
        cal=read(PARENT/f'calibrated-{name}/calibration.json')
        assert base.digest(train_manifest)==cal['manifest_sha256'] and cal['training_only']
        assert read(train_manifest)['split']=='train' and len(read(train_manifest)['windows'])==128
        config=copy.deepcopy(cfg);config.update(output=str(root/name),checkpoint_every=2000,retain_steps=list(STEPS))
        assert base.semantic(config)==base.semantic(cfg)
        rows.append(dict(id=name,seed=seed,parent=parent,config=config,audit=report,health=health,
            initial=str(INITIAL/name/'initial-model/weights.pt'),historical_planning=original['planning'],
            original_archive=parent/'checkpoint-step0006000.ckpt',primary=primary))
    root.mkdir(parents=True);jobs=[];records=[]
    for slot,row in enumerate(rows):
        name=row['id'];run=root/name;run.mkdir();base.write(run/'config.json',row['config'])
        shutil.copy2(row['original_archive'],run/'last.ckpt');assert base.digest(run/'last.ckpt')==row['audit']['checkpoint_sha256']
        base.write(run/'pre-resume-fullstate.json',dict(row['audit'],copy_byte_identical=True,
            preserved_parent_archive=str(row['original_archive']),parent=str(row['parent']),parent_source=source))
        common=dict(host='dev2',gpu=slot,source=source,task='cube',training_seed=row['seed'],family='native-lewm',arm='history7-matched',hold=True)
        previous=[]
        def add(phase,dest,command,**extra):
            nonlocal previous
            ident=phase+'-'+name;jobs.append(dict(**common,id=ident,phase=phase,output=str(dest),depends_on=previous,command=command,**extra));previous=[ident]
        for step in STEPS:
            add(f'learn{step}',run,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(run/'config.json'),'--stop-step',str(step),'--resume'],config=str(run/'config.json'),config_sha256=base.digest(run/'config.json'))
            dest=root/f'archive{step}-{name}'
            add(f'archive{step}',dest,[PY,'-B','-u',str(helper/'prepare12000.py'),'archive','--step',str(step),'--run',str(run),'--parent-run',str(row['parent']),'--output',str(dest)])
        frozen=Path(source['source']);dest=root/f'fullstate-{name}'
        add('fullstate',dest,[PY,'-B','-u',str(frozen/'experiments/ideas_validate_20261031/preflight.py'),'--run',str(run),'--output',str(dest),'--expected-step','12000','--initial-weights',row['initial']])
        dest=root/f'transition-{name}'
        add('transition',dest,[PY,'-B','-u',str(frozen/'experiments/ideas_validate_20261031/audit.py'),'--run',str(run),'--output',str(dest),'--cases','64','--manifest',str(frozen/'experiments/ideas_validate_20261031/audit-windows-cube.json')])
        def evaluate(phase,checkpoint,manifest):
            dest=root/(phase+'-'+name)
            add(phase,dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),'--checkpoint',str(checkpoint),'--data',row['config']['data'],'--output',str(dest)],
                environment={'MUJOCO_EGL_DEVICE_ID':str(slot)},manifest_sha256=base.digest(manifest))
        evaluate('planning',run/'model-step0012000',row['primary'])
        evaluate('fresh-planning',run/'model-step0012000',fresh)
        cal=root/('calibrated-'+name);bn_helper=Path(helpers['source'])/'experiments/native_baseline_20261031/bn_calibrate.py'
        add('calibration',root/('calibration-job-'+name),[PY,'-B','-u',str(bn_helper),'calibrate','--run',str(run),'--step','12000','--manifest',str(train_manifest),'--output',str(cal)],artifact_output=str(cal))
        evaluate('planning-calibrated',cal/'model',row['primary'])
        evaluate('fresh-planning-calibrated',cal/'model',fresh)
        records.append(dict(id=name,parent=str(row['parent']),output=str(run),host='dev2',gpu=slot,seed=row['seed'],initial_step=6000,stop_step=12000,
            resume_original_archive=str(row['original_archive']),resume_sha256=row['audit']['checkpoint_sha256'],initial_weights=row['initial'],
            parent_health_passed=True,parent_loss=row['health']['loss'],parent_planning=row['historical_planning'],model_parameters=18035396,
            sampler_at_stage_boundaries=row['audit']['sampler_at_stage_boundaries'],optimizer_groups=1,optimizer_parameter_tensors=297,
            config_sha256=base.digest(run/'config.json')))
    assert len(jobs)==26 and sum(j['phase'] in ['planning','fresh-planning','planning-calibrated','fresh-planning-calibrated'] for j in jobs)==8
    base.write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=oldplan['environment'],jobs=jobs))
    base.write(root/'source-record.json',source)
    base.write(root/'protocol.json',dict(prepared_only=True,source=source,reused_helpers=helpers,runs=records,jobs=26,final_planning_evaluations=8,
        initial_step=6000,stop_step=12000,archive_steps=list(STEPS),scheduler_total=124910,scheduler_warmup=1249,
        initial_training_windows_per_seed=6000*128,additional_training_windows_per_seed=6000*128,total_training_windows_per_seed=12000*128,frames_per_window=8,
        fixed_calibration=dict(path=str(train_manifest),sha256=base.digest(train_manifest),training_clips=128,observation_vectors=1024,prediction_vectors=896),
        primary_manifest=dict(path=str(rows[0]['primary']),sha256=base.digest(rows[0]['primary'])),fresh_manifest=dict(path=str(fresh),sha256=base.digest(fresh)),
        helper_files={str(helper/'prepare12000.py'):base.digest(helper/'prepare12000.py')},parent_review=dict(path=str(REPORT),sha256=base.digest(REPORT)),
        requires_before_launch='Root reviews CPU evidence, commits/freezes prepare12000 helper, repoints archive commands, verifies source/input hashes and actual dev2 GPUs0/1 availability; no active plan or dispatcher is created.',
        rationale='Two healthy existing native Cube seeds receive an equally exposed12000-update maturity reference before new-method claims. Native6000 raw primary24/22 and fresh12/14 are retained. TF12000 primary18/20 and fresh14/16 are contextual root-reported references, requiring exact artifact matching in the eventual collector.',
        limitations='Native18.04M versus TF/F approximately51.1M: equal update/window exposure is not equal capacity or compute. Raw native and fixed train128 preciseBN remain separate uniform variants. Historical Cube evaluations have observed repeat variation; no bitwise outcome claim or posthoc best variant.',
        rule='Only two Cube runs; no TwoRoom, candidate grid or new objective. Resume original raw6000 full-state archive, never BN-calibrated exports. All model/loss/Trainer/optimizer/scheduler/sampler code stays7a. Use existing staged stops8000/10000/12000; each full archive and native state audit gates the following stage. Additional operational resume boundaries are disclosed; no uninterrupted-bitwise trajectory equivalence is claimed. Keep all6k holds and archives. At12000 run fullstate/gradient and64-window transition audits, raw primary/fresh50, fixed train-only calibration, calibrated primary/fresh50. No automatic continuation beyond12000.'))
    (root/'HOLD_DISPATCH').write_text('Prepared only: two native Cube maturity references; root review and immutable helper freeze required.\n')
    summary=dict(root=str(root),prepared_only=True,launch=False,runs=2,jobs=26,final_evaluations=8,source_commit=source['source_commit'],
        audits_passed=2,slots=[dict(host='dev2',gpu=g) for g in (0,1)],archive_steps=list(STEPS),runs_summary=records,
        candidate_plan_sha256=base.digest(root/'candidate-plan.json'),protocol_sha256=base.digest(root/'protocol.json'),helper_sha256=base.digest(__file__))
    base.write(root/'preparation-summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','archive'])
    parser.add_argument('--root',default=str(ROOT));parser.add_argument('--helper-root',default=str(Path(__file__).resolve().parent))
    parser.add_argument('--run');parser.add_argument('--parent-run');parser.add_argument('--output');parser.add_argument('--step',type=int)
    args=parser.parse_args();prepare(args) if args.phase=='prepare' else archive(args)
