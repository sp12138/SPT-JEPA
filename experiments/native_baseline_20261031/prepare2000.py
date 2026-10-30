"""Prepare four native full-state continuations; no launch, commit or source freeze."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys

B=Path('/efs/prescottsun/SPT-JEPA')
PARENT=B/'runs/native_lewm600_20261031'
ROOT=B/'runs/native_lewm2000_20261031'
REPORT=B/'reports/native-lewm600-20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
OPERATIONAL={'output','checkpoint_every','retain_steps','is_resuming','slurm.job_id','slurm.task_id'}

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def semantic(x):return {k:v for k,v in x.items()if k not in OPERATIONAL}
def activate():
    src=read(PARENT/'protocol.json')['source'];sys.path[:0]=src['pythonpath'].split(':');return src
def exact(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b)and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(exact(v,b[k])for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(exact(v,w)for v,w in zip(a,b))
    return a==b

def audit(run,step,parent):
    import torch
    from as_jepa_lewm.train import TrainModule,WindowSampler
    from as_jepa_tools.benchmark_data import make_training_dataset
    from experiments.ideas_validate_20261031.preflight import checkpoint_report
    cfg=read(run/'config.json');assert semantic(cfg)==semantic(read(parent/'config.json'))
    assert cfg['model_class']=='native_lewm'and cfg['max_steps']==124910
    state=torch.load(run/'last.ckpt',map_location='cpu',weights_only=False)
    portable=torch.load(run/f'model-step{step:07d}/weights.pt',map_location='cpu',weights_only=True)
    report=checkpoint_report(state,portable,cfg,step)
    assert semantic(state['hyper_parameters'])==semantic(cfg)
    module=TrainModule(cfg);module.load_state_dict(state['state_dict'],strict=True)
    oc=module.configure_optimizers();opt=oc['optimizer'];scheduler=oc['lr_scheduler']['scheduler']
    saved=state['optimizer_states'][0];names={id(p):n for n,p in module.named_parameters()}
    ordered=[[names[id(p)]for p in g['params']]for g in opt.param_groups]
    assert len(saved['param_groups'])==len(ordered)==1,'Native AdamW has one group, not AS-JEPA renderer groups'
    assert len(saved['state'])==sum(map(len,ordered))==297
    for group,new in zip(saved['param_groups'],opt.param_groups):
        assert len(group['params'])==len(new['params'])
        for pid,param in zip(group['params'],new['params']):
            assert saved['state'][pid]['exp_avg'].shape==param.shape
            assert saved['state'][pid]['exp_avg_sq'].shape==param.shape
            assert int(saved['state'][pid]['step'])==step
    opt.load_state_dict(saved);scheduler.load_state_dict(state['lr_schedulers'][0])
    assert exact(saved,opt.state_dict())and exact(state['lr_schedulers'][0],scheduler.state_dict())
    assert scheduler.last_epoch==step
    assert [g['lr']for g in opt.param_groups]==scheduler.get_last_lr()
    dataset=make_training_dataset(cfg);cursor=state['campaign_samples'];kw=dict(seed=cfg['seed'],batch_size=cfg['batch_size'])
    resumed=list(itertools.islice(iter(WindowSampler(dataset,start=cursor,**kw)),32))
    uninterrupted=list(itertools.islice(iter(WindowSampler(dataset,start=0,**kw)),cursor,cursor+32))
    assert resumed==uninterrupted
    if getattr(dataset,'file',None)is not None:dataset.file.close()
    report.update(optimizer_group_names=ordered,optimizer_group_count=1,optimizer_state_tensors=297,
        optimizer_scheduler_roundtrip_exact=True,sampler_next32=resumed,sampler_cursor_exact=True,
        learning_rates=scheduler.get_last_lr(),scheduler_last_epoch=scheduler.last_epoch,
        checkpoint_sha256=digest(run/'last.ckpt'),portable_sha256=digest(run/f'model-step{step:07d}/weights.pt'),
        initial_weights_sha256=digest(parent/'initial-model/weights.pt'),saved_training_semantics_equal=True,
        passed=all(report['checks'].values()),
        scope='CPU load/save roundtrip and native optimizer parameter order/shape checks; no model forward or update, no claim of bitwise GPU resumed trajectory equivalence.')
    assert report['passed'];return report

def archive(a):
    activate();run=Path(a.run);parent=Path(a.parent_run);out=Path(a.output)
    r=audit(run,2000,parent);stage=read(run/'stage-complete.json');assert stage['stage_complete']and stage['stop_step']==2000
    dest=run/'checkpoint-step0002000.ckpt';assert not dest.exists();shutil.copy2(run/'last.ckpt',dest)
    assert digest(dest)==r['checkpoint_sha256'];r['archived_checkpoint']=str(dest);write(out/'fullstate.json',r)
    print(json.dumps(dict(passed=True,archive=str(dest))))

def prepare(a):
    src=activate();root=Path(a.root);assert not root.exists();helper=Path(a.helper_root).resolve()
    oldplan=read(PARENT/'plan.json');review=read(REPORT/'results.json');assert len(review['rows'])==4
    for job in oldplan['jobs']:assert read(PARENT/'exits'/(job['id']+'.json'))['returncode']==0
    records=[]
    for task in ('tworoom','cube'):
        for seed in (1,2):
            n=f'{task}-native-lewm-seed{seed}';parent=PARENT/n
            r=audit(parent,600,parent);cfg=read(parent/'config.json');cfg.update(output=str(root/n),checkpoint_every=200,retain_steps=[600,1200,2000])
            assert semantic(cfg)==semantic(read(parent/'config.json'));records.append((task,seed,n,parent,r,cfg))
    root.mkdir(parents=True);jobs=[];runs=[]
    for slot,(task,seed,n,parent,r,cfg)in enumerate(records):
        out=root/n;out.mkdir();write(out/'config.json',cfg);shutil.copy2(parent/'last.ckpt',out/'last.ckpt')
        assert digest(out/'last.ckpt')==r['checkpoint_sha256'];write(out/'pre-resume-fullstate.json',r)
        data=Path(cfg['data']);source=Path(src['source']);windows=source/f'experiments/ideas_validate_20261031/audit-windows-{task}.json'
        trainmanifest=REPORT/f'source-bncal-v1/train128-{task}.json'
        original_cal=read(REPORT/f'calibrated-{n}/calibration.json');assert digest(trainmanifest)==original_cal['manifest_sha256']
        common=dict(host='dev2',gpu=slot,source=src,task=task,training_seed=seed,family='native-lewm',arm='history7-matched')
        previous=[]
        def add(phase,dest,cmd,**extra):
            nonlocal previous
            ident=phase+'-'+n;jobs.append(dict(**common,id=ident,phase=phase,output=str(dest),depends_on=previous,command=cmd,**extra));previous=[ident]
        add('learn2000',out,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step','2000','--resume'],config=str(out/'config.json'),config_sha256=digest(out/'config.json'))
        dest=root/('archive-'+n);add('archive',dest,[PY,'-B','-u',str(helper/'prepare2000.py'),'archive','--run',str(out),'--parent-run',str(parent),'--output',str(dest)])
        dest=root/('fullstate-'+n);add('fullstate',dest,[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/preflight.py'),'--run',str(out),'--output',str(dest),'--expected-step','2000','--initial-weights',str(parent/'initial-model/weights.pt')])
        dest=root/('transition-'+n);add('transition',dest,[PY,'-B','-u',str(source/'experiments/ideas_validate_20261031/audit.py'),'--run',str(out),'--output',str(dest),'--cases','64','--manifest',str(windows)])
        def plan_command(checkpoint,dest):return [PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(data/'manifests/validation-strict.json'),'--checkpoint',str(checkpoint),'--data',str(data),'--output',str(dest)]
        dest=root/('planning-'+n);add('planning',dest,plan_command(out/'model-step0002000',dest),environment={'MUJOCO_EGL_DEVICE_ID':str(slot)})
        cal=root/('calibrated-'+n);add('calibration',root/('calibration-job-'+n),[PY,'-B','-u',str(helper/'bn_calibrate.py'),'calibrate','--run',str(out),'--step','2000','--manifest',str(trainmanifest),'--output',str(cal)],artifact_output=str(cal))
        dest=root/('planning-calibrated-'+n);add('planning-calibrated',dest,plan_command(cal/'model',dest),environment={'MUJOCO_EGL_DEVICE_ID':str(slot)})
        runs.append(dict(id=n,parent=str(parent),output=str(out),initial_step=600,stop_step=2000,checkpoint_sha256=r['checkpoint_sha256'],training_manifest=str(trainmanifest),training_manifest_sha256=digest(trainmanifest),optimizer_group_count=r['optimizer_group_count'],learning_rates_at600=r['learning_rates']))
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=oldplan['environment'],jobs=jobs))
    write(root/'protocol.json',dict(prepared_only=True,source=src,runs=runs,stop_step=2000,scheduler_total=124910,scheduler_warmup=1249,
        helper_files={str(helper/f):digest(helper/f)for f in ['prepare2000.py','bn_calibrate.py']},
        requires_before_launch='Root reviews, commits/freezes helper sources, replaces their candidate command paths with immutable equivalents, verifies hashes and confirms actual dev2g0-3 availability. No plan.json/dispatcher is created.',
        rule='Resume original full-state600, never calibrated exports. Preserve original modules/native loss/BN/AdamW one group/scheduler/RNG/sampler. No restart or semantic config change. Fixed train128 preciseBN only on copied2000 export, never fed into training. Raw and calibrated primary50 both mandatory. Stop2000; no automatic longer continuation.'))
    print(json.dumps(dict(prepared_only=True,root=str(root),jobs=len(jobs),full_state_pass=4,optimizer_groups=[r['optimizer_group_count']for r in runs])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','archive']);p.add_argument('--root',default=str(ROOT));p.add_argument('--helper-root',default=str(Path(__file__).resolve().parent));p.add_argument('--run');p.add_argument('--parent-run');p.add_argument('--output');a=p.parse_args()
    if a.phase=='prepare':prepare(a)
    else:archive(a)
