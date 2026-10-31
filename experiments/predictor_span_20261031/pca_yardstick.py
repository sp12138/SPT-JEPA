"""Exactly two train-only affine191 PCA fits on existing Cube local caches."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/predictor_span_pca_20261031'
OUT=B/'reports/predictor-span-pca-20261031'
V1=B/'runs/predictor_span_20261031/plan.json'
CACHE=B/'reports/direct-rgb-information-20261031/features-and-calibration.npz'
DIRECT=B/'reports/direct-rgb-information-20261031/results.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def artifact(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def orthogonal(x,q):return x-(x@q)@q.T
def basis(x):
    u,s,_=np.linalg.svd(x,full_matrices=False)
    rank=int((s>s[0]*max(x.shape)*np.finfo(float).eps).sum());return u[:,:rank]
def score(delta,q,c):
    residual=orthogonal(delta-c,q);error=np.square(residual).mean(-1);energy=np.square(delta).mean(-1)
    return dict(mse=float(error.mean()),fraction_of_true_change=float(error.sum()/energy.sum()),
        per_horizon_mse=error.mean(0).tolist(),per_window_mse=error.mean(1).tolist())
def contracts():
    # Independent known-axis affine projection, no exploratory PCA fit.
    q=np.array([[1.],[0.],[0.]]);c=np.array([0.,2.,0.]);x=np.array([[3.,2.,4.],[-2.,2.,0.]])
    r=orthogonal(x-c,q);np.testing.assert_array_equal(r,[[0.,0.,4.],[0.,0.,0.]])
    assert float(np.square(r).sum())==16
    return dict(exact_affine_oracle=True,diagnostic_pca_fit_count=0)
def prepare(args):
    v1=read(V1);direct=read(DIRECT);assert sha(CACHE)==direct['files']['features-and-calibration.npz']['sha256']
    rows=[r for r in v1['rows']if r['family']=='local'and r['task']=='cube'and r['step']==6000]
    assert [r['seed']for r in rows]==[1,2]
    inputs={str(p):artifact(p)for p in [V1,DIRECT,CACHE,B/'reports/predictor-span-20261031/results.json']}
    manifests=direct['protocol']['manifests'];assert len(manifests['train']['windows'])==128 and len(manifests['validation']['windows'])==64
    assert not {x['episode']for x in manifests['train']['windows']} & {x['episode']for x in manifests['validation']['windows']}
    for row in rows:
        for key in ['weights','cache','windows','config','source_record']:
            path=row[key];assert sha(path)==v1['inputs'][path]['sha256'];inputs[path]=artifact(path)
        assert read(row['windows'])['windows']==manifests['validation']['windows']
        ref=next(x for x in direct['references']if x['seed']==row['seed'])
        for key in ['calibration','extraction']:
            a=ref[key];assert sha(a['path'])==a['sha256'];inputs[a['path']]=artifact(a['path'])
    write(args.plan,dict(helper_sha256=sha(__file__),protocol=artifact(Path(__file__).with_name('PCA_PROTOCOL.md')),
        inputs=inputs,rows=rows,output=str(args.output),rank=191,train_clips=128,validation_clips=64,
        train_deltas=896,validation_deltas=448,pca_fit_count=2,cpu_contracts=contracts(),
        scope='Cube seeds1/2 only; cached oldViT local1024, original scalar normalization and Euclidean geometry. Exactly two PCA fits on TRAIN adjacent deltas, affine offset=train mean. No heldout fitting/newcohort/labels/encoder/network/GPU, rank grid or seed search. Preserve v1.'))
    print(json.dumps(dict(passed=True,plan=str(args.plan),pca_fit_count=0)),flush=True)
def run(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    torch.set_num_threads(2);p=read(args.plan);assert sha(__file__)==p['helper_sha256']
    assert sha(Path(__file__).with_name('PCA_PROTOCOL.md'))==p['protocol']['sha256']
    for name,a in p['inputs'].items():assert sha(name)==a['sha256'],name
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=False);start=time.time();rows=[];fits={};count=0
    with np.load(CACHE)as cached:
        for row in p['rows']:
            seed=row['seed'];train=cached[f'oldViT_seed{seed}__train'];val=cached[f'oldViT_seed{seed}__validation']
            assert train.shape==(128,8,1024)and val.shape==(64,8,1024)and train.dtype==val.dtype==np.float64
            assert np.isfinite(train).all()and np.isfinite(val).all()
            td=np.diff(train,axis=1);vd=np.diff(val,axis=1);flat=td.reshape(896,1024);c=flat.mean(0)
            # Only these two SVDs fit a data-dependent subspace. No val used.
            _,s,vh=np.linalg.svd(flat-c,full_matrices=False);q=vh[:191].T;count+=1
            np.testing.assert_allclose(q.T@q,np.eye(191),atol=5e-14,rtol=0)
            train_pca=score(td,q,c);val_pca=score(vd,q,c)
            tail=float(np.square(s[191:]).sum()/(896*1024));assert abs(train_pca['mse']-tail)<1e-12
            fits[f'seed{seed}__center']=c;fits[f'seed{seed}__basis']=q;fits[f'seed{seed}__singular_values']=s
            state=torch.load(row['weights'],map_location='cpu',weights_only=True,mmap=True)
            a=lambda key:state['predictor.transformer.'+key].double().numpy()
            W,b,gamma,beta=[a(k)for k in ['output_proj.weight','output_proj.bias','norm.weight','norm.bias']]
            V=W*gamma;V-=V.mean(1,keepdims=True);current_q=basis(V);current_c=W@beta+b
            loose_q=basis(np.column_stack([W,b]));assert current_q.shape==(1024,191)and loose_q.shape==(1024,193)
            current={split:score(d,current_q,current_c)for split,d in [('train',td),('validation',vd)]}
            loose={split:score(d,loose_q,np.zeros(1024))for split,d in [('train',td),('validation',vd)]}
            with np.load(row['cache'])as old:observed=old['observed'].astype(float)
            maxdiff=float(np.abs(val-observed).max());mse=float(np.square(val-observed).mean());assert maxdiff<2e-5 and mse<1e-10
            item=dict(seed=seed,training_pca_rank=191,current_rank=191,loose_rank=193,
                unchanged_cache_vs_original_fp32=dict(max_abs=maxdiff,mse=mse,
                    reason='DirectRGB reused cached fullpatch projection+FP64 normalization; native cached observations used FP32 arithmetic. No renormalization performed.'),
                training_centroid_norm=float(np.linalg.norm(c)),pca=dict(train=train_pca,validation=val_pca),
                current=current,loose=loose,train_tail_energy_identity=tail,
                validation_pca_over_current=val_pca['mse']/current['validation']['mse'],
                validation_pca_over_loose=val_pca['mse']/loose['validation']['mse'],
                singular_values_explained_fraction=float(np.square(s[:191]).sum()/np.square(s).sum()),
                current_offset_distance_from_train_mean_outside_current_subspace=float(np.linalg.norm(orthogonal(c-current_c,current_q))))
            rows.append(item);print(json.dumps(dict(seed=seed,train=[train_pca['mse'],current['train']['mse']],
                validation=[val_pca['mse'],current['validation']['mse'],loose['validation']['mse']],
                pca_over_current=item['validation_pca_over_current'])),flush=True)
            del state
    assert count==2 and not torch.cuda.is_initialized()
    np.savez_compressed(out/'pca-fit.npz',**fits)
    result=dict(passed=True,helper_sha256=sha(__file__),plan_sha256=sha(args.plan),protocol_sha256=p['protocol']['sha256'],
        pca_fit_count=count,rank=191,training_deltas_per_seed=896,validation_deltas_per_seed=448,model_calls=0,
        cuda_initialized=False,seconds=time.time()-start,inputs=p['inputs'],rows=rows,pca_artifact=artifact(out/'pca-fit.npz'),
        scope=p['scope'],limitations='PCA is optimal on the training sample for affine reconstruction, not guaranteed optimal on the population or validation. It uses no actions and is not a world model, objective proposal or attainable planning guarantee. Current versus PCA may differ in subspace and affine offset; fixed low-rank alone does not determine task performance.128 independent training clips with correlated within-clip deltas; no new repeats.')
    write(out/'results.json',result);print(json.dumps(dict(passed=True,output=str(out),pca_fit_count=count)),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run']);p.add_argument('--plan',type=Path,default=ROOT/'plan.json');p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    (prepare if a.phase=='prepare'else run)(a)
if __name__=='__main__':main()
