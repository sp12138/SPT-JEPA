"""CPU algebra on immutable predictor tensors and existing 64-window caches.

Never imports a world model, calls a network/encoder, or changes a cohort.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/predictor_span_20261031'
OUT=B/'reports/predictor-span-20261031'
PREFIX='predictor.transformer.'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def artifact(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def basis(matrix):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    tol=np.finfo(np.float64).eps*max(matrix.shape)*s[0] if len(s)and s[0]>0 else 0.
    rank=int((s>tol).sum());return u[:,:rank],s,tol
def orthogonal(x,q):return x-(x@q)@q.T
def ratio(a,b):return float(a/b)if b>0 else None

def cpu_contracts():
    # Rank1 direction with ambient dimension4: no harm when true changes lie
    # along it, irreducible error when an orthogonal action effect is required.
    W=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    gamma=np.array([2.,3.]);beta=np.array([.2,-.1]);bias=np.array([0.,0.,.4,0.])
    V=W*gamma;V-=V.mean(1,keepdims=True);c=W@beta+bias;q,s,tol=basis(V)
    assert q.shape==(4,1)
    z=np.array([[1.,-1.],[-2.,2.],[.5,-.5]])
    increments=z@(W*gamma).T+c
    np.testing.assert_allclose(orthogonal(increments-c,q),0,atol=2e-15)
    endpoint=increments.sum(0);np.testing.assert_allclose(orthogonal(endpoint-3*c,q),0,atol=2e-15)
    target=endpoint+np.array([0.,0.,0.,2.]);res=orthogonal(target-3*c,q)
    assert abs(np.square(res).sum()-4)<1e-13
    prediction=endpoint+q[:,0]*3
    error=np.square(prediction-target).sum()
    assert abs(error-(np.square(res).sum()+9))<1e-13
    return dict(layernorm_affine_rank=True,inductive_carry_endpoint=True,
        exact_projection_oracle=True,pythagorean_error_partition=True,
        counterexample_low_rank_zero_oracle=True,model_calls=0)

def specs():
    rows=[]
    for family,stage,campaign in [('local',2000,'local_isotropic2000_20261031'),
        ('local',6000,'local_isotropic6000_20261031'),('basis',2000,'local_basis2000_20261031'),
        ('F64',6000,'carrier_learn6000_20261031'),('TF64',6000,'carrier_learn6000_20261031')]:
        for task in ['tworoom','cube']:
            for seed in [1,2]:
                suffix={'local':'spatial-local','basis':'basis-orthogonal','F64':'carrier-identity','TF64':'temporal-forecast'}[family]
                root=B/'runs'/campaign;name=f'{task}-{suffix}-seed{seed}';run=root/name
                row=dict(id=f'{family}-{stage}-{task}-seed{seed}',family=family,step=stage,task=task,seed=seed,
                    run=str(run),source_record=str(root/'source-record.json'),config=str(run/'config.json'),
                    weights=str(run/f'model-step{stage:07d}'/'weights.pt'))
                if family!='TF64':
                    folder=root/f'transition-{name}';row.update(cache_kind='audit',cache=str(folder/'state-audit.npz'),
                        audit=str(folder/'state-audit.json'),windows=str(folder/'windows.json'))
                elif task=='cube':
                    folder=B/'reports/tf6000-pose-20261031'/f'seed{seed}-features'
                    row.update(cache_kind='pose',cache=str(folder/'features.npz'),audit=str(folder/'extraction.json'),
                        windows=str(B/'runs/ideas_validate600_20261031'/f'transition-cube-carrier-identity-seed{seed}'/'windows.json'))
                else:row['cache_kind']='none'
                rows.append(row)
    return rows

def prepare(args):
    inputs={};sources={};rows=specs();windows={}
    for row in rows:
        cfg=read(row['config']);assert cfg['model'].get('mode','L2')=='L2'and cfg['model'].get('carry',True)
        assert not cfg['model'].get('batchnorm_state',False)and not cfg['model'].get('batchnorm_projector',False)
        if row['family']in ['local','basis']:assert cfg['model']['spatial_mode']=='local'
        for key in ['config','weights','source_record','cache','audit','windows']:
            if key in row:inputs[row[key]]=artifact(row[key])
        src=read(row['source_record']);root=Path(src['source'])
        if str(root)not in sources:
            manifest=read(src['manifest']);verified={}
            for rel in ['as_jepa_lewm/model.py','as_jepa_lewm/robust.py','as_jepa_lewm/carrier.py',
                        'as_jepa_lewm/spatial_carrier.py','experiments/ideas_validate_20261031/audit.py']:
                p=root/rel
                if p.exists():
                    assert sha(p)==manifest['files'][rel];verified[rel]=artifact(p)
                    if Path(rel).name in ['robust.py','carrier.py','spatial_carrier.py']:
                        assert 'def predict('not in p.read_text(),'Unexpected prediction override'
            sources[str(root)]=dict(source=src,manifest=artifact(src['manifest']),verified=verified)
        if row['cache_kind']=='audit':
            a=read(row['audit']);assert a['weights_sha256']==inputs[row['weights']]['sha256']
            assert a['config_sha256']==inputs[row['config']]['sha256'] and a['step']==row['step']
            assert a['manifest_sha256']==inputs[row['windows']]['sha256']and a['cases']==64
        elif row['cache_kind']=='pose':
            a=read(row['audit']);assert a['weights']['sha256']==inputs[row['weights']]['sha256']
            assert a['config']['sha256']==inputs[row['config']]['sha256']and a['step']==row['step']
            assert a['reference_provenance']['validation_manifest']['sha256']==inputs[row['windows']]['sha256']
        if 'windows'in row:
            w=read(row['windows']);assert w['count']==64 and w['window']==8
            if row['task']in windows:assert windows[row['task']]==w
            else:windows[row['task']]=w
    for rel in ['module.py','lewm.py']:
        p=B/'sources/stable-worldmodel/stable_worldmodel/wm/lewm'/rel;inputs[str(p)]=artifact(p)
    write(args.plan,dict(helper_sha256=sha(__file__),protocol=artifact(Path(__file__).with_name('PROTOCOL.md')),
        cpu_contracts=cpu_contracts(),rows=rows,sources=sources,inputs=inputs,
        output=str(args.output),scope='20 frozen checkpoints;18 existing caches,64 identical validation windows per task;TwoRoomTF has no cache and rank-only is reported. No model call, encoder, GPU, labels, new cohorts, new probe fitting or optimization.'))
    print(json.dumps(dict(plan=str(args.plan),rows=len(rows),caches=sum(x['cache_kind']!='none'for x in rows),passed=True)),flush=True)

def error_partition(pred,target,base,h,c,q,loose):
    delta=target-base;adjusted=delta-h*c
    outside=orthogonal(adjusted,q);oracle=np.square(outside).mean(-1)
    pdelta=pred-base;pin=orthogonal(pdelta-h*c,q)
    error=np.square(pred-target).mean(-1);carry=np.square(delta).mean(-1)
    reachable_error=np.square((pred-target)@q).sum(-1)/pred.shape[-1]
    relaxed=np.square(orthogonal(delta,loose)).mean(-1)
    per=[]
    for i in range(pred.shape[1]):
        per.append(dict(horizon=i+1,prediction_mse=float(error[:,i].mean()),carry_mse=float(carry[:,i].mean()),
            oracle_mse=float(oracle[:,i].mean()),oracle_over_error=ratio(oracle[:,i].sum(),error[:,i].sum()),
            oracle_over_true_change=ratio(oracle[:,i].sum(),carry[:,i].sum())))
    return dict(prediction_mse=float(error.mean()),carry_mse=float(carry.mean()),oracle_mse=float(oracle.mean()),
        oracle_over_error=ratio(oracle.sum(),error.sum()),oracle_over_true_change=ratio(oracle.sum(),carry.sum()),
        reachable_error_mse=float(reachable_error.mean()),relaxed_span_oracle_mse=float(relaxed.mean()),
        forecast_outside_affine_span_mse=float(np.square(pin).mean()),forecast_outside_affine_span_max_abs=float(np.abs(pin).max()),
        forecast_outside_over_displacement=ratio(np.square(pin).sum(),np.square(pdelta).sum()),
        pythagorean_relative_gap=ratio(abs(error.sum()-oracle.sum()-reachable_error.sum()),error.sum()),
        per_horizon=per,per_window_oracle_mse=oracle.mean(1).tolist(),per_window_error_mse=error.mean(1).tolist())

def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only visibility must be explicit'
    import torch
    torch.set_num_threads(2);plan=read(args.plan);assert sha(__file__)==plan['helper_sha256']
    assert sha(Path(__file__).with_name('PROTOCOL.md'))==plan['protocol']['sha256']
    for p,a in plan['inputs'].items():assert sha(p)==a['sha256'],p
    for src in plan['sources'].values():
        assert sha(src['manifest']['path'])==src['manifest']['sha256']
        for a in src['verified'].values():assert sha(a['path'])==a['sha256']
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False);start=time.time();results=[];observed_saved={}
    for spec in plan['rows']:
        state=torch.load(spec['weights'],map_location='cpu',weights_only=True,mmap=True)
        arr=lambda k:state[PREFIX+k].detach().double().numpy().copy()
        W,b,gamma,beta=[arr(k)for k in ['output_proj.weight','output_proj.bias','norm.weight','norm.bias']]
        assert W.shape[1]==192 and b.shape==(W.shape[0],)and gamma.shape==beta.shape==(192,)
        assert not any(k.startswith('pred_proj.')for k in state)
        V=W*gamma;V-=V.mean(1,keepdims=True);c=W@beta+b
        q,s,tol=basis(V);qW,sW,_=basis(W);loose,sloose,_=basis(np.column_stack([W,b]))
        result=dict(**{k:spec[k]for k in ['id','family','task','seed','step','cache_kind']},
            weight_shape=list(W.shape),rank_W=qW.shape[1],rank_W_b=loose.shape[1],rank_LN_effective=q.shape[1],
            singular_value_largest=float(s[0]),singular_value_smallest_retained=float(s[q.shape[1]-1]),
            singular_value_last=float(s[-1]),rank_tolerance=tol,
            bias_norm=float(np.linalg.norm(b)),layernorm_offset_norm=float(np.linalg.norm(c)),
            offset_outside_tangent_norm=float(np.linalg.norm(orthogonal(c,q))))
        if spec['cache_kind']!='none':
            with np.load(spec['cache'])as z:
                observed=z['observed'if spec['cache_kind']=='audit'else'validation_observed'].astype(np.float64)
                clean=z['clean_rollout'].astype(np.float64);teacher=z['teacher_forced_prefix'].astype(np.float64)
            assert observed.shape==(64,8,W.shape[0])and clean.shape==teacher.shape==(64,7,W.shape[0])
            assert all(np.isfinite(x).all()for x in [observed,clean,teacher,W,b,gamma,beta])
            observed_saved[spec['id']]=observed
            target=observed[:,1:];h=np.arange(1,8,dtype=float)[None,:,None]
            result['clean']=error_partition(clean,target,observed[:,:1],h,c,q,loose)
            result['one_step']=error_partition(teacher,target,observed[:,:-1],np.ones_like(h),c,q,loose)
            # All cached predictions are FP32. A numerical residual is allowed,
            # not silently described as mathematically exact zero.
            for key in ['clean','one_step']:
                assert result[key]['forecast_outside_affine_span_max_abs']<2e-4,(spec['id'],key,result[key])
                assert result[key]['pythagorean_relative_gap']<1e-5
            if spec['cache_kind']=='audit':
                a=read(spec['audit']);assert abs(result['clean']['prediction_mse']-a['conditions']['clean_rollout']['mse'])<1e-10
            result['cache_rows']=448
        else:result['limitation']='No cached TwoRoom TF6000 state audit; exact full-rank output result only, no new model call.'
        results.append(result);print(json.dumps(dict(id=spec['id'],rank=result['rank_LN_effective'],clean=result.get('clean',{}).get('oracle_over_error'),one_step=result.get('one_step',{}).get('oracle_over_error'))),flush=True)
        del state
    observation_checks=[]
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            a=observed_saved[f'local-2000-{task}-seed{seed}'];b=observed_saved[f'local-6000-{task}-seed{seed}'];assert np.array_equal(a,b)
            spec=next(x for x in plan['rows']if x['id']==f'basis-2000-{task}-seed{seed}')
            state=torch.load(spec['weights'],map_location='cpu',weights_only=True,mmap=True)
            Q=state['spatial_orthogonal_basis'].double().numpy();rotated=observed_saved[spec['id']]
            diff=float(np.abs(a@Q-rotated).max());assert diff<2e-6
            observation_checks.append(dict(task=task,seed=seed,local2000_equals6000=True,basis_rotation_max_abs=diff))
    result=dict(passed=True,helper_sha256=sha(__file__),protocol_sha256=plan['protocol']['sha256'],plan_sha256=sha(args.plan),
        device='CPU',model_calls=0,cuda_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.time()-start,
        source_provenance=plan['sources'],input_provenance=plan['inputs'],cpu_contracts=cpu_contracts(),
        observation_checks=observation_checks,rows=results,
        scope=plan['scope'],limitation='Oracle lower bounds allow arbitrary hidden sequences; they are necessary reachability bounds, not sufficient executable-action models or planning success bounds.64D learned/fixed representations differ; raw MSE is not cross-family comparable. No saved local1024 fixed physical-readout coefficients were available, so no physical decomposition is claimed.')
    assert not result['cuda_initialized'];write(out/'results.json',result)
    print(json.dumps(dict(passed=True,output=str(out),elapsed_seconds=result['elapsed_seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','contracts'])
    p.add_argument('--plan',type=Path,default=ROOT/'plan.json');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.phase=='prepare':prepare(a)
    elif a.phase=='run':run(a)
    else:print(json.dumps(cpu_contracts()))
if __name__=='__main__':main()
