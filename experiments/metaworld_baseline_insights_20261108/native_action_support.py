"""Same48 native/CEM factual-control support contrast; CPU,0NN/SGD/physics."""
import hashlib
import json
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')


def read(p):return json.loads(p.read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=BASE/'runs/metaworld_baseline_insights_20261108/native-action-support-v1';root.mkdir(exist_ok=False)
    data=BASE/'data/metaworld-lerobot-b46a1c18-packed-v1';card=read(data/'dataset_card.json')
    with h5py.File(card['h5_path'],'r') as f:a=np.asarray(f['action'],np.float64)
    train_ids=set(np.load(data/'train_episodes.npy'));keep=np.zeros(len(a),bool)
    for ep,off,length in zip(np.load(data/'episode_ids.npy'),np.load(data/'offsets.npy'),np.load(data/'lengths.npy')):
        if ep in train_ids:keep[off:off+max(0,length-1)]=True
    train=a[keep];assert train.shape==(172608,20)
    mean,std=[np.asarray(card['action_'+s],np.float64) for s in ['mean','std']]
    z=(train-mean)/std;center=z.mean(0);z-=center
    eigen,vec=np.linalg.eigh(z.T@z/len(z));order=np.argsort(eigen)[::-1];eigen=eigen[order];vec=vec[:,order]
    k=int(np.searchsorted(np.cumsum(eigen/eigen.sum()),.99)+1);basis=vec[:,:k]
    residual=lambda v:np.maximum(0,np.square(v).sum(1)-np.square(v@basis).sum(1))
    cutoff=float(np.quantile(residual(z),.99));old=read(BASE/'runs/metaworld_baseline_insights_20261108/action-support-v1/summary.json')
    assert k==12 and np.isclose(cutoff,old['train']['residual']['0.99']['train_p99_squared_distance'],rtol=1e-8)
    axis_std=np.sqrt(np.mean(std.reshape(5,4)**2,axis=0))

    def describe(blocks):
        v=(blocks-mean)/std-center;r=residual(v);native=blocks.reshape(-1,5,4)/axis_std
        temporal=float(np.square(native-native.mean(1,keepdims=True)).sum()/np.square(native).sum())
        return dict(blocks=len(blocks),temporal_variation_fraction=temporal,
            fraction_above_train_p99_subspace_residual=float((r>cutoff).mean()),median_subspace_squared_residual=float(np.median(r)))

    plan=read(BASE/'runs/metaworld_native100_pilot_20261108/plan.json');groups={'train':describe(train)};verified=0
    for j in plan['jobs']:
        native=[];cem=[]
        for i in [0,1]:
            d=read(BASE/f'runs/metaworld_native100_pilot_20261108/{j["name"]}/formal-{i}/metrics.json')
            assert d['passed'] and d['model_state_before']==d['model_state_after'] and d['weights_sha256']==j['weights_sha256']
            native.extend(d['episodes'])
        assert len(native)==len(j['common_cem_rows'])==12
        assert {(r['task'],r['seed'],r['case_sha256']) for r in native}=={(r['task'],r['seed'],r['case_sha256']) for r in j['common_cem_rows']}
        for label,rows in [('native',native),('cem',j['common_cem_rows'])]:
            blocks=[]
            for r in rows:
                assert sha(r['trace'])==r['trace_sha256']
                with np.load(r['trace']) as f:a=f['actions'];s=f['successes'];assert a.shape==(500,4) and not s.any()
                assert np.isfinite(a).all() and np.max(np.abs(a))<=1
                blocks.append(a.reshape(-1,20));verified+=1
            groups[j['name']+'-'+label]=describe(np.concatenate(blocks))
    d=dict(passed=True,groups=groups,actual_trace_files_verified=verified,train99variance_dimensions=k,train_p99_squared_residual=cutoff,
        model_updates=0,neural_models_loaded=0,new_physical_episodes=0,render_calls=0,
        scope='Same12physicalcases x4frozenparents x2controllers=96existingtraces. Blockswithinatrajectoryarenotindependentcases. '
            'GlobalTRAINlinearsubspace-tail diagnostic,not calibratedconditional OODprobability or causalfailureproof. '
            'Nativehorizon/solver/prior/feedback/computechangejointly. Sameknownexecutedforwardinputs,notoptimal-actionlabels.')
    (root/'summary.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))


if __name__=='__main__':main()
