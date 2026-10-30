"""Prepare a four-run coordinate-factorization control; never launch jobs.

Reuse the fixed local1024 measurements, means and scalar scales. The only model
intervention is the predeclared orthogonal basis after normalization. This is a
mechanism control, not a new algorithm or an information-improvement claim.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
OLD=B/'runs/local_isotropic600_20261031'
PRE=B/'runs/local_basis_preflight_20261031'
ROOT=B/'runs/local_basis600_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
SELF='experiments/local_carrier_20261031/prepare_basis.py'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')
def source(path):
    s=read(path);m=read(s['manifest']);assert s['source_commit']==m['source_commit']
    for n,h in m['files'].items():assert sha(Path(s['source'])/n)==h,n
    assert sha(__file__)==m['files'][SELF]
    return s
def allocations():
    for gpu,(task,seed) in enumerate(itertools.product(('tworoom','cube'),(1,2)),2):
        yield dict(id=f'{task}-basis-orthogonal-seed{seed}',task=task,training_seed=seed,host='dev13',gpu=gpu,
                   family='spatial_carrier',arm='local-orthogonal')
def config(a,out,step):
    old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'
    c=copy.deepcopy(read(old/'config.json'));cal=c['spatial_calibration']
    assert sha(cal['path'])==cal['sha256'] and read(cal['path'])['normalization']=='global_rms'
    assert c['model']['spatial_mode']=='local' and len(set(c['model']['spatial_scale']))==1
    assert c['batch_size']==128 and c['max_steps']==124910 and c['record_initial_weights']
    assert not c.get('initial_weights') and not c.get('initial_encoder_checkpoint')
    c.update(output=str(out),checkpoint_every=2 if step==2 else 200,retain_steps=[2] if step==2 else [200,400,600])
    c['model']['spatial_basis']='orthogonal'
    c['audit_expectations']['initial_unchanged_prefixes'].append('spatial_orthogonal_basis')
    return c,old
def prepare(a):
    s=source(a.source_record);stage=a.phase;root=PRE if stage=='preflight' else ROOT;step=2 if stage=='preflight' else 600
    assert not root.exists()
    if stage=='training':
        assert read(PRE/'review.json')['passed'] and read(PRE/'source-record.json')==s
    oldplan=read(OLD/'plan.json');assert read(OLD/'source-record.json')['source_commit']=='32fad0cc04ca46cb9e6f5516a9aa0525b65b4cea'
    root.mkdir();write(root/'source-record.json',s);jobs=[];records=[]
    for alloc in allocations():
        n=alloc['id'];out=root/n;c,old=config(alloc,out,step);write(out/'config.json',c)
        common={k:v for k,v in alloc.items() if k!='id'};common['source']=s;prev=[]
        def add(phase,dest,cmd,**extra):
            nonlocal prev
            ident=phase+'-'+n;jobs.append(dict(**common,id=ident,phase=phase,output=str(dest),depends_on=prev,command=cmd,**extra));prev=[ident]
        add('train'+str(step),out,[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step',str(step)],
            config=str(out/'config.json'),config_sha256=sha(out/'config.json'))
        dest=root/('fullstate-'+n)
        add('fullstate',dest,[PY,'-B','-u',str(Path(s['source'])/'experiments/ideas_validate_20261031/preflight.py'),
            '--run',str(out),'--output',str(dest),'--expected-step',str(step),'--reference-initial',str(old/'initial-model/weights.pt')])
        if stage=='training':
            dest=root/('health-'+n);add('health',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_health','--run',str(out),'--output',str(dest)])
            dest=root/('transition-'+n);add('transition',dest,[PY,'-B','-u',str(Path(s['source'])/'experiments/ideas_validate_20261031/audit.py'),
                '--run',str(out),'--output',str(dest),'--cases','64','--manifest',str(Path(s['source'])/f'experiments/ideas_validate_20261031/audit-windows-{alloc["task"]}.json')])
            # Existing identity-local600 controls have primary50 only. Keep this
            # short control exactly matched; do not compare against fresh2000.
            manifest=Path(c['data'])/'manifests/validation-strict.json';dest=root/('planning-'+n)
            add('planning',dest,[PY,'-B','-u','-m','as_jepa_tools.benchmark_evaluate','--manifest',str(manifest),
                '--checkpoint',str(out/'model-step0000600'),'--data',c['data'],'--output',str(dest)],
                environment={'MUJOCO_EGL_DEVICE_ID':str(alloc['gpu'])},manifest_sha256=sha(manifest))
        records.append(dict(**alloc,reference=str(old),calibration=c['spatial_calibration']))
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=oldplan['environment'],jobs=jobs))
    write(root/'protocol.json',dict(source=s,step=step,runs=records,candidate_only=True,
        intervention='One fixed signed/permuted Hadamard1024 after unchanged local normalization; FP64 accumulation, FP32 output, no new learned parameters.',
        evidence='CPU orthogonality/default-loss-gradient-RNG contracts and cached-state distance preservation; inspect committed basis contracts.',
        interpretation='Information and observed Euclidean goal distances preserved to rounding. Predictor/renderer coordinate factorization both change. Same RNG noise is equal in distribution, not pathwise conjugacy. A short outcome alone is not a mature planning verdict.',
        unchanged='Native sample-level objective, trainer, optimizer, schedule, samples, model initialization, calibration, evaluator and all historical holds.',
        policy='Root must activate after source/capacity review; all eight real preflight endpoints and complete initial-tensor checks required before four fresh600 runs. Stop600; no automatic extension.'))
    print(json.dumps(dict(root=str(root),jobs=len(jobs),launched=False)))
def review(a):
    import torch
    s=source(a.source_record);assert read(PRE/'source-record.json')==s
    p=read(PRE/'candidate-plan.json');assert len(p['jobs'])==8
    for j in p['jobs']:assert read(PRE/'exits'/(j['id']+'.json'))['returncode']==0
    rows=[]
    for alloc in allocations():
        n=alloc['id'];run=PRE/n;c,old=config(alloc,run,2);assert c==read(run/'config.json')
        report=read(PRE/('fullstate-'+n)/'preflight.json');assert report['passed']
        original=torch.load(old/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        initial=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True)
        final=torch.load(run/'model-step0000002/weights.pt',map_location='cpu',weights_only=True)
        key='spatial_orthogonal_basis';assert set(initial)==set(original)|{key}
        assert all(torch.equal(v,initial[k]) for k,v in original.items())
        assert torch.equal(initial[key],final[key]);q=initial[key].double()
        assert torch.equal(q.T@q,torch.eye(1024,dtype=torch.double))
        rows.append(dict(id=n,all_original_initial_tensors_equal=len(original),only_extra_key=key,
            frozen_basis_unchanged=True,orthogonality_exact_fp64=True,fullstate_gradient_report_sha256=sha(PRE/('fullstate-'+n)/'preflight.json'),
            reference_initial_sha256=sha(old/'initial-model/weights.pt'),initial_sha256=sha(run/'initial-model/weights.pt'),passed=True))
    write(PRE/'review.json',dict(passed=True,source=s,rows=rows,scope='All real two-update checks and every original initial tensor; not a functional600-step learning verdict.'))
    print(json.dumps(dict(passed=True,runs=len(rows))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['preflight','review','training']);p.add_argument('--source-record',required=True);a=p.parse_args()
    review(a) if a.phase=='review' else prepare(a)
