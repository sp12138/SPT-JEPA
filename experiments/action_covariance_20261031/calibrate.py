"""Train128-only fixed action correlation; CPU, no RGB/model/simulator imports."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from sklearn.covariance import LedoitWolf
from analyze import spectrum, shape_stats, quantiles

B=Path('/efs/prescottsun/SPT-JEPA')
COHORTS={'cube':B/'runs/carrier_metric_20261031/train-clips128.json',
         'tworoom':B/'reports/native-lewm600-20261031/source-bncal-v1/train128-tworoom.json'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for task,manifest_path in COHORTS.items():
        out=root/task;out.mkdir(exist_ok=True);assert not(out/'calibration.json').exists()
        m=read(manifest_path);data=B/f'data/hardtasks-v1/{task}-metadata';card=read(data/'dataset_card.json')
        assert m['split']=='train'and m['count']==128 and m['window']==8 and m['frameskip']==5
        assert m['dataset_card_sha256']==sha(data/'dataset_card.json')
        for name,h in m['metadata_sha256'].items():assert sha(data/name)==h
        ids,offsets,lengths=[np.load(data/f'{k}.npy')for k in ['episode_ids','offsets','lengths']]
        allowed=set(np.load(data/'train_episodes.npy').tolist());held=set(np.load(data/'validation_episodes.npy').tolist());assert not allowed&held
        provenance=[];chunks=[];dim=len(card['action_mean']);payload_source=None
        if task=='cube':
            payload_source=B/'reports/action-covariance-20261031/training-action-statistics.npz'
            with np.load(payload_source)as f:cached=f['raw_train_actions'].copy()
            old=read(payload_source.with_name('results.json'));assert old['manifest_sha256']==sha(manifest_path)
            assert digest(cached)==old['train']['raw_action_bytes_sha256']
        # Cube reuses the just-verified cache; TwoRoom reads only action/ep_idx slices.
        h5=h5py.File(card['h5_path'],'r',swmr=True)if task=='tworoom'else None
        try:
            for i,w in enumerate(m['windows']):
                ep=w['episode_array_index'];assert int(ids[ep])==w['episode']and w['episode']in allowed
                start=int(offsets[ep])+w['start'];assert w['rows']==(start+5*np.arange(8)).tolist()
                assert start+25<=int(offsets[ep]+lengths[ep])
                x=np.asarray(h5['action'][start:start+25],dtype=np.float32)if h5 is not None else cached[i]
                if h5 is not None:assert np.all(np.asarray(h5['ep_idx'][start:start+25])==w['episode'])
                assert x.shape==(25,dim)and np.isfinite(x).all();chunks.append(x)
                provenance.append(dict(episode=w['episode'],start_row=start,end_row_exclusive=start+25,action_sha256=digest(x)))
        finally:
            if h5 is not None:h5.close()
        assert len({w['episode']for w in provenance})==128
        raw=np.stack(chunks);a=(raw.astype(np.float64)-np.array(card['action_mean']))/np.array(card['action_std']);x=a.reshape(128,-1);n=x.shape[1]
        var=x.var(0);assert np.all(var>0),'Zero empirical variance: stop, no exploration prior is silently added'
        fit=LedoitWolf().fit(x);sd=np.sqrt(fit.covariance_.diagonal());R=fit.covariance_/sd[:,None]/sd[None,:]
        R=(R+R.T)/2;L=np.linalg.cholesky(R)
        np.testing.assert_allclose(L@L.T,R,atol=2e-15,rtol=1e-14);np.testing.assert_allclose(R.diagonal(),1,atol=2e-15)
        assert np.linalg.eigvalsh(R).min()>0 and np.isfinite(L).all()
        loo=[]
        for i in range(128):
            f=LedoitWolf().fit(np.delete(x,i,axis=0));d=x[i]-f.location_;loo.append(float(d@f.precision_@d/n))
        channels=[]
        for c in range(dim):
            v=a[:,:,c];channels.append(dict(index=c,raw_min=float(raw[:,:,c].min()),raw_max=float(raw[:,:,c].max()),
                normalized_mean=float(v.mean()),normalized_std=float(v.std()),within_clip_std=quantiles(v.std(1)),
                exact_constant_clips=int((np.ptp(v,axis=1)==0).sum()),
                lag_correlation={str(l):float(np.corrcoef(v[:,:-l].ravel(),v[:,l:].ravel())[0,1])for l in [1,2,5,10,20]}))
        within=[t for t in range(24)if t%5!=4];between=[t for t in range(24)if t%5==4]
        def corr(ix):return [float(np.corrcoef(a[:,ix,c].ravel(),a[:,np.array(ix)+1,c].ravel())[0,1])for c in range(dim)]
        np.savez_compressed(out/'matrix.npz',raw_train_actions=raw,location=fit.location_,covariance=fit.covariance_,correlation=R,cholesky=L,loo=np.asarray(loo))
        result=dict(task=task,manifest=str(manifest_path),manifest_sha256=sha(manifest_path),dataset_card_sha256=sha(data/'dataset_card.json'),
            metadata_sha256=m['metadata_sha256'],action_mean=card['action_mean'],action_std=card['action_std'],
            raw_dimension=dim,horizon=5,action_block=5,trajectory_dimension=n,coordinate_order='raw step major, action channel minor',
            cohort_count=128,distinct_training_episodes=128,no_validation_overlap=True,provenance=provenance,
            raw_actions_sha256=digest(raw),payload_bytes=int(raw.nbytes),new_h5_action_bytes=int(raw.nbytes)if h5 is not None else 0,
            reused_payload_source=str(payload_source)if payload_source else None,empirical_variance_min=float(var.min()),zero_variance_coordinates=int((var==0).sum()),
            shrinkage=float(fit.shrinkage_),empirical_spectrum=spectrum(np.cov(x,rowvar=False,bias=True)),correlation_spectrum=spectrum(R),
            channels=channels,block_internal_correlation=corr(within),block_boundary_correlation=corr(between),
            shape={k:quantiles(v)for k,v in shape_stats(a).items()},leave_one_episode_out=quantiles(loo),
            matrix_sha256=sha(out/'matrix.npz'),helper_sha256=sha(__file__),dependency_sha256=sha(Path(__file__).with_name('analyze.py')),
            definition='R = diag(C_LW)^(-1/2) C_LW diag(C_LW)^(-1/2); positive-diagonal Cholesky; no mean injection, eigenfloor, truncation, arbitrary mixture, clipping or outcome tuning',
            scope='CPU training-action statistics only. No encoder, renderer, learned-model fit, new simulator, goal/physical labels or planning.')
        (out/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k]for k in ['task','trajectory_dimension','empirical_variance_min','shrinkage','channels','block_internal_correlation','block_boundary_correlation','leave_one_episode_out']}),flush=True)
if __name__=='__main__':main()
