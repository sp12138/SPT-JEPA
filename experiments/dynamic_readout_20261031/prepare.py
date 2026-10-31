"""Three-update technical preflight;600 remains a separate unactivated candidate."""
import argparse,copy,hashlib,itertools,json
from pathlib import Path
B=Path('/efs/prescottsun/SPT-JEPA')
OLD=B/'runs/local_isotropic600_20261031'
PRE=B/'runs/dynamic_readout_preflight_20261031'
ROOT=B/'runs/dynamic_readout600_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF='experiments/dynamic_readout_20261031/prepare.py'
PREFLIGHT='experiments/dynamic_readout_20261031/preflight.py'
PREFIX='dynamic_readout_module.'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def allocations():
    for gpu,(task,seed,mode)in enumerate(itertools.product(['tworoom','cube'],[1,2],['static','state'])):
        yield dict(id=f'{task}-readout-{mode}-seed{seed}',task=task,training_seed=seed,arm=mode,host='dev15',gpu=gpu,family='spatial_carrier')
def source(path):
    s=read(path);m=read(s['manifest']);assert m['source_commit']==s['source_commit']
    for n,d in m['files'].items():assert sha(Path(s['source'])/n)==d,n
    assert sha(__file__)==m['files'][SELF]
    proof=read(Path(s['source'])/'experiments/dynamic_readout_20261031/results/summary.json')
    assert proof['passed']and proof['raw_chart_default_regression']
    for n,d in proof['source_hashes'].items():assert m['files'][n]==d,n
    cpu=read(Path(s['source'])/'experiments/dynamic_readout_20261031/results/contracts.json')
    assert cpu['passed']and cpu['tests']==5 and not cpu['cuda_initialized']
    return s
def update_config(old,allocation,output,step):
    c=copy.deepcopy(old);assert step in [3,600]
    assert c['model_class']=='spatial_carrier'and c['model']['spatial_mode']=='local'
    assert c['batch_size']==128 and c['max_steps']==124910 and c['record_initial_weights']
    assert c['model']['beta']==.25 and c['model']['sigma0']==.1 and c['model']['rho']==.8 and c['model']['forecast_fraction']==.5
    assert not c.get('initial_weights')and not c.get('initial_encoder_checkpoint')
    c.update(output=str(output),checkpoint_every=3 if step==3 else 200,retain_steps=[3]if step==3 else[200,400,600])
    c['model']['spatial_readout_mode']=allocation['arm']
    assert c['model'].get('spatial_chart','none')=='none'and c['model'].get('spatial_basis','identity')=='identity'
    for name in ['input.weight','input.bias','output.weight','output.bias']:
        c['audit_expectations']['gradients'][PREFIX+name]='active'
    return c
def config(a,out,step):
    old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}';cfg=read(old/'config.json')
    cal=cfg['spatial_calibration'];assert sha(cal['path'])==cal['sha256']and read(cal['path'])['normalization']=='global_rms'
    assert len(set(cfg['model']['spatial_scale']))==1
    return update_config(cfg,a,out,step),old
def jobs_for(a,cfg,reference,root,s,step):
    name=a['id'];out=root/name;common={k:v for k,v in a.items()if k!='id'};common['source']=s;jobs=[];prev=[]
    def add(phase,dest,command,**extra):
        nonlocal prev
        ident=phase+'-'+name;jobs.append(dict(**common,id=ident,phase=phase,output=str(dest),depends_on=prev,command=command,**extra));prev=[ident]
    add('train'+str(step),out,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step',str(step)],
        config=str(out/'config.json'),config_sha256=sha(out/'config.json'))
    dest=root/('fullstate-'+name)
    add('fullstate',dest,[PY,'-B','-u',str(Path(s['source'])/PREFLIGHT),'--run',str(out),'--output',str(dest),
        '--expected-step',str(step),'--reference-initial',str(reference/'initial-model/weights.pt')])
    if step==600:
        dest=root/('health-'+name);add('health',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)])
        dest=root/('transition-'+name);add('transition',dest,[PY,'-B','-u',str(Path(s['source'])/'experiments/ideas_validate_20261031/audit.py'),
            '--run',str(out),'--output',str(dest),'--cases','64','--manifest',str(Path(s['source'])/f'experiments/ideas_validate_20261031/audit-windows-{a["task"]}.json')])
        dest=root/('planning-'+name);manifest=Path(cfg['data'])/'manifests/validation-strict.json'
        add('planning',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
            '--checkpoint',str(out/'model-step0000600'),'--data',cfg['data'],'--output',str(dest)],
            manifest_sha256=sha(manifest),environment=dict(MUJOCO_EGL_DEVICE_ID=str(a['gpu'])))
    return jobs
def prepare(args):
    s=source(args.source_record);pre=args.phase=='preflight';root,step=(PRE,3)if pre else(ROOT,600)
    assert not root.exists()
    if not pre:assert read(PRE/'review.json')['passed']and read(PRE/'source-record.json')==s
    root.mkdir();write(root/'source-record.json',s);jobs=[];runs=[]
    for a in allocations():
        out=root/a['id'];cfg,reference=config(a,out,step);write(out/'config.json',cfg)
        jobs.extend(jobs_for(a,cfg,reference,root,s,step));runs.append(dict(**a,reference=str(reference),calibration=cfg['spatial_calibration']))
    assert len(jobs)==(16 if pre else 40)
    env=read(OLD/'plan.json')['environment']
    # Existing primary protocol, including native Cube EGL libraries, unchanged.
    env.update(PYOPENGL_PLATFORM='egl',LD_LIBRARY_PATH=str(B/'environments/egl-lib/usr/lib/x86_64-linux-gnu')+':/usr/lib64',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(B/'environments/egl-lib/10_nvidia.json'))
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=env,jobs=jobs))
    write(root/'protocol.json',dict(source=s,step=step,runs=runs,candidate_only=True,
        intervention='Old raw local1024 carrier and native TF loss;optional p+M(u,pos)(p-u),with shared18->64->64 MLP;STATIC zeroes state inputs.',
        parameter_departure=dict(extra_trainable=5376,original_tensors=553,new_tensors=4,static_inactive_state_input_elements=1024),
        audit='Three updates allow delayed head gates;no first-step all-nonzero or weight-change requirement. Native fullstate/optimizer/frozen210/initial553+4,head tensor gradients and memory must pass.',
        controls='Original identity-local600 primary50 and same64 saved validation windows;no newcalibration/cohort/freshcases.',
        raw_inverse_scope='Source retains frozen0c inverse code, but every readout arm requires chartnone and never invokes that inverse. This plan does not authorize raw-chart replay or change its failed certificates.',
        unchanged='Native Trainer/checkpoints,original data/sampler/actions/noise/loss/beta/sigma/rho,globalRMS,model seed,batch128,max_steps124910 and warmup.',
        authorization='Only8 real3-update preflights+8audits approved. A600 candidate requires actual preflight review and explicit separate launch approval. No automatic >600;all historical HOLDs unchanged.'))
    print(json.dumps(dict(root=str(root),endpoints=len(jobs),activated=False)),flush=True)
def review(args):
    import torch
    torch.set_num_threads(2);s=source(args.source_record);assert read(PRE/'source-record.json')==s
    plan=read(PRE/'candidate-plan.json');assert len(plan['jobs'])==16
    for j in plan['jobs']:assert read(PRE/'exits'/(j['id']+'.json'))['returncode']==0
    rows=[];head_reference=None
    for a in allocations():
        run=PRE/a['id'];cfg,old=config(a,run,3);assert cfg==read(run/'config.json')
        folder=PRE/('fullstate-'+a['id']);native=read(folder/'preflight.json');report=read(folder/'readout-audit.json')
        assert native['passed']and report['passed']and all(native['checkpoint']['checks'].values())
        assert report['original_initial_tensors_exact']==553 and report['new_tensors']==4 and report['new_parameters']==5376
        assert report['frozen_tensors_unchanged']==210 and report['head_gradient']['passed']
        init=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        head={k:v.clone()for k,v in init.items()if k.startswith(PREFIX)}
        if head_reference is None:head_reference=head
        else:assert all(torch.equal(v,head_reference[k])for k,v in head.items())
        size,oldsize=read(run/'model_size.json'),read(old/'model_size.json')
        assert size['total']-oldsize['total']==size['trainable']-oldsize['trainable']==5376
        rows.append(dict(id=a['id'],**report,all8_initial_heads_equal=True,native_report_sha256=sha(folder/'preflight.json')))
    write(PRE/'review.json',dict(passed=True,source=s,rows=rows,scope='All16 native technical endpoints reviewed. No600 launch authorization or task-performance claim.'))
    print(json.dumps(dict(passed=True,runs=8,peak_gib=[r['peak_gib']for r in rows])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','review','candidate600']);p.add_argument('--source-record',required=True);a=p.parse_args()
    review(a)if a.phase=='review'else prepare(a)
