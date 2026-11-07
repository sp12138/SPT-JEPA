"""Read-only action support and physical-trace audit; no NN or new control.

Covariance is diagnostic,not a proposed learned prior or new candidate loss.
Executed controls are not assumed optimal. Correlation is not a failure cause.
"""
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
PARENT=BASE/'runs/metaworld_baselines_native_20261108'
ROOT=BASE/'runs/metaworld_baseline_insights_20261108/action-support-v1'
TASKS=['bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3']


def read(path):return json.loads(Path(path).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ROOT.mkdir(exist_ok=False)
    data=BASE/'data/metaworld-lerobot-b46a1c18-packed-v1';card=read(data/'dataset_card.json')
    with h5py.File(card['h5_path'],'r') as file:actions=np.asarray(file['action'],np.float64)
    ids,offsets,lengths=[np.load(data/(n+'.npy')) for n in ['episode_ids','offsets','lengths']]
    selected=set(np.load(data/'train_episodes.npy').tolist());mask=np.zeros(len(actions),bool)
    for ep,offset,length in zip(ids,offsets,lengths):
        if ep in selected:mask[offset:offset+max(0,length-1)]=True
    train=actions[mask];assert train.shape==(172608,20)
    mean,std=[np.asarray(card['action_'+s],np.float64) for s in ['mean','std']]
    assert np.allclose(train.mean(0),mean,atol=2e-6,rtol=1e-5)
    z=(train-mean)/std;center=z.mean(0);zc=z-center
    covariance=zc.T@zc/len(zc);eigen,vec=np.linalg.eigh(covariance)
    order=np.argsort(eigen)[::-1];eigen=eigen[order];vec=vec[:,order]
    fraction=np.maximum(eigen,0)/np.maximum(eigen,0).sum();dims={str(q):int(np.searchsorted(np.cumsum(fraction),q)+1) for q in [.95,.99]}
    basis={q:vec[:,:k] for q,k in dims.items()}
    train_residual={q:np.maximum(0,(zc*zc).sum(1)-np.square(zc@v).sum(1)) for q,v in basis.items()}
    cutoffs={q:float(np.quantile(v,.99)) for q,v in train_residual.items()}
    common_axis_std=np.sqrt(np.mean(std.reshape(5,4)**2,axis=0))

    def describe(blocks):
        blocks=np.asarray(blocks,np.float64).reshape(-1,20);norm=(blocks-mean)/std-center
        a=blocks.reshape(-1,5,4)/common_axis_std
        difference=(np.diff(a,axis=1)**2).mean(axis=(0,1))
        temporal=(a-a.mean(1,keepdims=True))**2
        result=dict(blocks=len(blocks),native_adjacent_difference_energy_by_axis=difference.tolist(),
            within_block_temporal_variation_fraction=float(temporal.sum()/max(float((a*a).sum()),1e-20)),
            residual={})
        for q,v in basis.items():
            residual=np.maximum(0,(norm*norm).sum(1)-np.square(norm@v).sum(1))
            result['residual'][q]=dict(subspace_dimensions=dims[q],median_squared_distance=float(np.median(residual)),
                p95_squared_distance=float(np.quantile(residual,.95)),train_p99_squared_distance=cutoffs[q],
                fraction_above_train_p99=float((residual>cutoffs[q]).mean()))
        return result

    sources={};verified=0;models={};bank=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')
    expected={(r['task'],r['seed']):r for r in bank['rows']}
    common=[c for i in [0,1] for c in read(PARENT/f'manifests/intermediate-{i}.json')['rows']]
    common_keys={(c['task'],c['seed']) for c in common};assert len(common_keys)==100
    plan=read(PARENT/'plan.json')
    for seed in [1,2]:
        job=next(j for j in plan['jobs'] if j['method']=='smwm' and j['seed']==seed)
        out=Path(job['output']);step=job['epoch_steps']['100'];rows=[];weights=set();seen=set();blocks=[];hard=[]
        for i in [0,1]:
            report=read(out/f'evaluation-{step:07d}-{i}/metrics.json')
            assert report['passed'] and report['model_updates']==0 and report['model_state_before']==report['model_state_after']
            weights.add(report['weights_sha256'])
            for r in report['episodes']:
                key=r['task'],r['seed'];assert key not in seen;seen.add(key)
                assert r['case_sha256']==expected[key]['sha256'] and r['initial_rgb_replay']['passed']
                assert sha(r['trace'])==r['trace_sha256']
                with np.load(r['trace']) as trace:a=trace['actions'];s=trace['successes']
                assert a.shape==(r['steps'],4) and len(s)==len(a)<=500
                assert np.isfinite(a).all() and np.max(np.abs(a))<=1 and bool(s.any())==r['success']
                complete=a[:len(a)//5*5].reshape(-1,20);blocks.append(complete)
                if r['task'] in TASKS:hard.append(complete)
                rows.append(r);verified+=1
        assert len(rows)==1000 and seen==set(expected) and len(weights)==1
        sources[f'smwm100-seed{seed}-all50']=describe(np.concatenate(blocks))
        sources[f'smwm100-seed{seed}-hard6']=describe(np.concatenate(hard))
        paired={}
        for epoch in [20,50,100]:
            e=job['epoch_steps'][str(epoch)]
            stage=[r for i in [0,1] for r in read(out/f'evaluation-{e:07d}-{i}/metrics.json')['episodes']]
            selected_rows=[r for r in stage if (r['task'],r['seed']) in common_keys]
            assert len(selected_rows)==100
            paired[str(epoch)]=dict(cases=100,successes=sum(r['success'] for r in selected_rows),
                macro50_percent=100*np.mean([r['success'] for r in selected_rows]))
        models[str(seed)]=dict(native100epoch_macro50=100*np.mean([r['success'] for r in rows]),
            native100epoch_hard6_successes=sum(r['success'] for r in rows if r['task'] in TASKS),
            native100epoch_hard6_episodes=sum(r['task'] in TASKS for r in rows),matched100case_learning_curve=paired,
            weights_sha256=next(iter(weights)))
    floors={name:[] for name in ['zero','uniform','train-mean']}
    for i in range(16):
        for r in read(BASE/f'runs/metaworld_baselines_20261107/floors/shard{i:02}/summary.json')['rows']:
            if r['task'] not in TASKS:continue
            assert sha(r['trace'])==r['trace_sha256']
            with np.load(r['trace']) as trace:a=trace['actions'].copy()
            floors[r['policy']].append(a[:len(a)//5*5].reshape(-1,20))
    for name,values in floors.items():sources['floor-'+name+'-hard6']=describe(np.concatenate(values))
    result=dict(passed=True,model_updates=0,neural_models_loaded=0,render_calls=0,new_physical_episodes=0,
        actual_swmw100_traces_verified=verified,train_factual_macro_blocks=len(train),
        train_covariance_eigenvalues=eigen.tolist(),train_cumulative_variance=np.cumsum(fraction).tolist(),
        train_effective_rank=float(np.exp(-sum(p*np.log(p) for p in fraction if p>0))),
        dimensions=dims,train=describe(train),sources=sources,models=models,
        scope='Fixed-data descriptive support audit,not a causal test or proposed prior. PCA dimensions/cutoffs come only fromtrain controls; '
              'allphase last paddedcontrols excluded. Executedactions are not optimal labels. '
              'Matched100case curves are separate from1000case full scores. Same120hard physicalcases crossed2trainedmodels,not240independentcases.')
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','actual_swmw100_traces_verified','train_effective_rank','dimensions','models']}))


if __name__=='__main__':main()
