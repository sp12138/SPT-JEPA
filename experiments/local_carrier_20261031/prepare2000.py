"""Prepare only eight full-state600→1200→2000 continuations; never launch."""
import argparse,copy,importlib,itertools,json,shutil,sys,time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.carrier_progress_20261031.prepare_learning import (
    B,PYTHON,activate_source,checkpoint_semantic,digest,read,rng_digest,semantic,write)
from experiments.carrier_progress_20261031.prepare_next import resume_sequence

PARENT=B/'runs/local_isotropic600_20261031'
ROOT=B/'runs/local_isotropic2000_20261031'
SELF='experiments/local_carrier_20261031/prepare2000.py'
TRAINING_COMMIT='32fad0cc04ca46cb9e6f5516a9aa0525b65b4cea'
_OPS_READY=False


def allocations():
    for gpu,(task,seed,arm) in enumerate(itertools.product(('tworoom','cube'),(1,2),('local','dense'))):
        yield dict(id=f'{task}-spatial-{arm}-seed{seed}',host='dev15',gpu=gpu,
            task=task,training_seed=seed,arm=arm,family='spatial_carrier')


def continuation_config(cfg,out):
    result=copy.deepcopy(cfg);result.update(output=str(out),checkpoint_every=1200,retain_steps=[1200,2000])
    assert semantic(result)==semantic(cfg)
    assert cfg['model_class']=='spatial_carrier' and cfg['batch_size']==128 and cfg['max_steps']==124910
    assert cfg['model']['state_slots']==64 and cfg['model']['state_width']==16
    assert len(set(cfg['model']['spatial_scale']))==1
    return result


def source_checks(helper,training):
    assert training['source_commit']==TRAINING_COMMIT
    records={}
    for kind,src in [('training',training),('helper',helper)]:
        manifest=read(src['manifest']);assert manifest['source_commit']==src['source_commit']
        names=manifest['files'] if kind=='training' else [SELF,
            'experiments/carrier_progress_20261031/prepare_learning.py','experiments/carrier_progress_20261031/prepare_next.py']
        for name in names:assert digest(Path(src['source'])/name)==manifest['files'][name],name
        records[kind]=dict(source=src,manifest_sha256=digest(src['manifest']))
    assert digest(__file__)==digest(Path(helper['source'])/SELF)
    assert helper['runtime_source']==training['runtime_source'] and helper['runtime_manifest']==training['runtime_manifest']
    return records


def nested_equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(nested_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a==b


def audit_state(run,step,parent,initial,training,portable_override=None):
    """Native CPU model/optimizer/scheduler loads plus frozen-buffer/state audit."""
    activate_source(training)
    import torch
    from experiments.ideas_validate_20261031.preflight import checkpoint_report
    from as_jepa_lewm.train import TrainModule
    from as_jepa_lewm.factory import gradient_expectations
    from as_jepa_tools.benchmark_data import make_training_dataset
    from vforge.ops import OpsConfig,initialize_ops
    global _OPS_READY
    torch.set_num_threads(2)
    if not _OPS_READY:
        initialize_ops(OpsConfig(backend='sdpa_math',allow_fallback=False),device=torch.device('cpu'),parallel_state=None);_OPS_READY=True
    for name in ('as_jepa_lewm.train','as_jepa_lewm.factory','as_jepa_lewm.spatial_carrier','as_jepa_lewm.robust'):
        assert Path(importlib.import_module(name).__file__).is_relative_to(Path(training['source'])),name
    run,parent=Path(run),Path(parent);cfg=read(run/'config.json');assert semantic(cfg)==semantic(read(parent/'config.json'))
    assert cfg['model_class']=='spatial_carrier'
    state=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    portable_path=Path(portable_override) if portable_override else run/f'model-step{step:07d}/weights.pt'
    portable=torch.load(portable_path,map_location='cpu',weights_only=True)
    report=checkpoint_report(state,portable,cfg,step);checks=report['checks']
    checks['complete_model_keys']=set(state['state_dict'])=={'model.'+k for k in portable}
    checks['saved_training_semantics']=checkpoint_semantic(state['hyper_parameters'])==semantic(cfg)
    checks['scheduler_step']=all(s['last_epoch']==step for s in state['lr_schedulers'])
    checks['optimizer_scheduler_lr_equal']=([g['lr'] for g in state['optimizer_states'][0]['param_groups']]==state['lr_schedulers'][0]['_last_lr'])
    module=TrainModule(cfg);module.load_state_dict(state['state_dict'],strict=True);checks['native_model_strict_load']=True
    native=module.configure_optimizers();optimizer=native['optimizer'];scheduler=native['lr_scheduler']['scheduler']
    named={id(p):n for n,p in module.model.named_parameters()};current=[[named[id(p)] for p in g['params']] for g in optimizer.param_groups]
    trainable={n for n,p in module.model.named_parameters() if p.requires_grad}
    ordered=[n.removeprefix('model.') for n in state['state_dict'] if n.removeprefix('model.') in trainable]
    checks['optimizer_parameter_order']=current==[[n for n in ordered if not n.startswith('renderer.')],[n for n in ordered if n.startswith('renderer.')]]
    optimizer.load_state_dict(state['optimizer_states'][0]);scheduler.load_state_dict(state['lr_schedulers'][0])
    checks['native_optimizer_roundtrip']=nested_equal(state['optimizer_states'][0],optimizer.state_dict())
    checks['native_scheduler_roundtrip']=nested_equal(state['lr_schedulers'][0],scheduler.state_dict())
    original=torch.load(initial,map_location='cpu',weights_only=True)
    # Enumerate every actual frozen parameter, not a Carrier64-specific branch.
    frozen=[n for n,p in module.model.named_parameters() if not p.requires_grad]
    buffers=['spatial_projection','spatial_mean','spatial_scale'];registered=dict(module.model.named_buffers())
    assert all(n in registered and n in portable and n in original for n in buffers)
    frozen+=buffers;assert len(frozen)==len(set(frozen)) and frozen
    different=[n for n in frozen if n not in original or n not in portable or not torch.equal(original[n],portable[n])]
    checks['all_frozen_parameters_and_measurement_buffers_unchanged']=not different
    sequence=resume_sequence(make_training_dataset(cfg),state['campaign_samples'],cfg['seed'],cfg['batch_size'])
    checks['sampler_cursor_sequence']=sequence['passed']
    report.update(run=str(run),parent=str(parent),initial_weights=str(initial),
        checkpoint_sha256=digest(run/'last.ckpt'),portable_path=str(portable_path),portable_sha256=digest(portable_path),
        config_sha256=digest(run/'config.json'),source_commit=training['source_commit'],
        optimizer_parameter_tensors=[len(g) for g in current],gradient_expectations=gradient_expectations(module.model,cfg),
        frozen_tensors=dict(equal=not different,compared_tensors=len(frozen),names=frozen,different=different),
        sampler=sequence,rng_component_sha256={k:rng_digest(v) for k,v in state['campaign_rng'].items()},
        passed=all(checks.values()),scope='CPU-only strict native model/optimizer/scheduler load and frozen tensor/full-state audit. '
            'No training update or GPU. Stored RNG/cursor preserved; not a claim of bitwise uninterrupted GPU trajectories.')
    assert report['passed'],report
    return report


def archive(args):
    helper,training=read(args.source_record),read(args.training_source_record);source_checks(helper,training)
    assert args.expected_step in (1200,2000)
    out=Path(args.output);assert not (out/'fullstate.json').exists()
    report=audit_state(args.audit_run,args.expected_step,args.parent_run,args.initial_weights,training)
    stage=read(Path(args.audit_run)/'stage-complete.json')
    assert stage['stage_complete'] and stage['stop_step']==args.expected_step and stage['initial_step']==(600 if args.expected_step==1200 else 1200)
    target=Path(args.audit_run)/f'checkpoint-step{args.expected_step:07d}.ckpt';assert not target.exists()
    shutil.copy2(Path(args.audit_run)/'last.ckpt',target);assert digest(target)==report['checkpoint_sha256']
    report.update(archived_checkpoint=str(target),native_stage_complete=stage,helper_source_commit=helper['source_commit'])
    out.mkdir(parents=True,exist_ok=True);write(out/'fullstate.json',report)
    print(json.dumps(dict(passed=True,step=args.expected_step,archive=str(target))),flush=True)


def jobs_for(rec,root,helper,training):
    """Eight bounded endpoints; second native segment depends on first archive."""
    root=Path(root);name=rec['id'];out=root/name;cfg=out/'config.json'
    common={k:rec[k] for k in ('host','gpu','task','training_seed','family','arm')};jobs=[];dependency=[]
    for step in (1200,2000):
        ident=f'learn{step}-{name}'
        jobs.append(dict(**common,source=training,id=ident,output=str(out),config=str(cfg),config_sha256=digest(cfg),phase=f'learn{step}',depends_on=dependency,
            command=[PYTHON,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(cfg),'--stop-step',str(step),'--resume']))
        dest=root/f'fullstate{step}-{name}'
        jobs.append(dict(**common,source=helper,id=dest.name,output=str(dest),phase=f'fullstate{step}',depends_on=[ident],
            command=[PYTHON,'-B','-u',str(Path(helper['source'])/SELF),'--source-record',str(root/'helper-source-record.json'),
                '--training-source-record',str(root/'source-record.json'),'--audit-run',str(out),'--parent-run',rec['parent'],
                '--initial-weights',rec['initial_weights'],'--expected-step',str(step),'--output',str(dest)]));dependency=[dest.name]
    for phase,module in [('health','as_jepa_tools.benchmark_health'),('info','as_jepa_tools.benchmark_information_audit')]:
        dest=root/(phase+'-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase=phase,depends_on=dependency,
            command=[PYTHON,'-B','-u','-m',module,'--run',str(out),'--output',str(dest)]));dependency=[dest.name]
    dest=root/('transition-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase='transition',depends_on=dependency,
        command=[PYTHON,'-B','-u',str(Path(training['source'])/'experiments/ideas_validate_20261031/audit.py'),'--run',str(out),'--output',str(dest),
            '--cases','64','--manifest',str(Path(training['source'])/f'experiments/ideas_validate_20261031/audit-windows-{rec["task"]}.json')]));dependency=[dest.name]
    dest=root/('planning-'+name);jobs.append(dict(**common,source=training,id=dest.name,output=str(dest),phase='planning',depends_on=dependency,
        environment=dict(MUJOCO_EGL_DEVICE_ID=str(rec['gpu'])),command=[PYTHON,'-B','-u','-m','as_jepa_tools.benchmark_evaluate',
            '--manifest',str(Path(rec['config']['data'])/'manifests/validation-strict.json'),'--checkpoint',str(out/'model-step0002000'),
            '--data',rec['config']['data'],'--output',str(dest)]))
    return jobs


def prepare(args):
    root,parent=Path(args.root),Path(args.parent);assert not root.exists()
    helper,training=read(args.source_record),read(parent/'source-record.json');compatibility=source_checks(helper,training)
    plan=read(parent/'plan.json');assert len(plan['jobs'])==40
    for j in plan['jobs']:assert read(parent/'exits'/(j['id']+'.json'))['returncode']==0
    records=[]
    for rec in allocations():
        previous=parent/rec['id'];cfg=read(previous/'config.json');stage=read(previous/'stage-complete.json')
        job=next(j for j in plan['jobs'] if j['id']==rec['id']);assert digest(previous/'config.json')==job['config_sha256']
        assert stage['stage_complete'] and stage['stop_step']==600
        cal=cfg['spatial_calibration'];assert digest(cal['path'])==cal['sha256']
        assert read(cal['path'])['normalization']=='global_rms'
        initial=previous/'initial-model/weights.pt';pre=audit_state(previous,600,previous,initial,training)
        records.append(dict(**rec,parent=str(previous),initial_weights=str(initial),config=continuation_config(cfg,root/rec['id']),pre_audit=pre))
        print(json.dumps(dict(pre_resume_passed=rec['id'])),flush=True)
    if args.dry_run:
        print(json.dumps(dict(dry_run=True,audits_passed=8,allocations=list(allocations()))));return
    root.mkdir();write(root/'source-record.json',training);write(root/'helper-source-record.json',helper)
    write(root/'source-compatibility.json',compatibility);jobs=[];copies=[]
    for rec in records:
        out=root/rec['id'];out.mkdir();write(out/'config.json',rec['config']);previous=Path(rec['parent'])
        shutil.copy2(previous/'last.ckpt',out/'last.ckpt');assert digest(out/'last.ckpt')==rec['pre_audit']['checkpoint_sha256']
        post=audit_state(out,600,previous,rec['initial_weights'],training,previous/'model-step0000600/weights.pt')
        assert post['rng_component_sha256']==rec['pre_audit']['rng_component_sha256']
        assert post['checkpoint_sha256']==rec['pre_audit']['checkpoint_sha256'] and post['sampler']==rec['pre_audit']['sampler']
        write(out/'pre-resume-fullstate.json',dict(rec['pre_audit'],copy_byte_identical=True,copied_to=str(out/'last.ckpt')))
        write(out/'post-copy-fullstate.json',post);copies.append(dict(id=rec['id'],checkpoint_sha256=post['checkpoint_sha256'],passed=True))
        jobs.extend(jobs_for(rec,root,helper,training));print(json.dumps(dict(post_copy_passed=rec['id'])),flush=True)
    assert len(jobs)==64 and len({j['id'] for j in jobs})==64
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=plan['environment'],jobs=jobs))
    write(root/'resume-audit.json',copies)
    write(root/'protocol.json',dict(created_at=time.time(),parent=str(parent),root=str(root),training_source_commit=training['source_commit'],
        helper_source_commit=helper['source_commit'],initial_step=600,intermediate_step=1200,stop_step=2000,runs=8,jobs=64,
        allocations=list(allocations()),native_fullstate_archives=[1200,2000],scheduler_horizon=124910,warmup_steps=1249,
        semantic_config_unchanged=True,original_calibration_unchanged=True,validation_cases=50,transition_windows=64,
        purpose='One bounded maturity check:1024-state factual learning at600 is real but slower than64-state Carrier; no cross-task planning superiority.',
        candidate_only=True,activation='Root reviews preparation and closes the old completed dispatcher before explicitly activating candidate-plan; helper writes no plan.json and launches nothing.',
        departures=['Fresh independent directories; original600 held checkpoints/config/source/STOPs untouched.',
            'Common native1200 resume boundary permits full-state archival without Trainer edits.',
            'Operational checkpoint cadence1200, retained portable1200/2000. No optimizer/scheduler/data/loss/calibration reset.',
            'Training and all GPU evaluation use original32fad0c. Separately frozen CPU helper imports that exact training source.'],
        stop_policy='Failed1200 audit/archive blocks2000. Stop at2000, no automatic further training.',
        state_scope='All frozen parameters and spatial projection/mean/scale compared to original initial tensors; native optimizer/scheduler load and RNG/cursor byte preservation. '
                    'Deterministic BenchmarkWindows has no random augmentation; no bitwise uninterrupted-GPU trajectory claim.'))
    assert not (root/'plan.json').exists();print(json.dumps(dict(candidate=str(root/'candidate-plan.json'),runs=8,endpoints=64,launched=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True);p.add_argument('--training-source-record')
    p.add_argument('--root',default=str(ROOT));p.add_argument('--parent',default=str(PARENT));p.add_argument('--dry-run',action='store_true')
    p.add_argument('--audit-run');p.add_argument('--parent-run');p.add_argument('--initial-weights');p.add_argument('--expected-step',type=int,choices=[1200,2000]);p.add_argument('--output')
    a=p.parse_args()
    if a.audit_run:
        assert a.training_source_record and a.parent_run and a.initial_weights and a.expected_step and a.output and not a.dry_run;archive(a)
    else:prepare(a)
