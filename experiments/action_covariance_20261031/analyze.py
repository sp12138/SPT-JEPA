"""Small train128 action-only CPU statistics; no model, rollout, or candidate fit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py
import hdf5plugin  # noqa: F401
from scipy.fft import dct
from sklearn.covariance import LedoitWolf

B=Path('/efs/prescottsun/SPT-JEPA');M=B/'runs/carrier_metric_20261031/train-clips128.json';D=B/'data/hardtasks-v1/cube-metadata'
POOL=B/'runs/mechanism_cubeaudit_recovery_20261030/pool';OUT=B/'reports/action-covariance-20261031'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def quantiles(a):
    a=np.asarray(a,float);return dict(mean=float(a.mean()),q=np.quantile(a,[0,.05,.25,.5,.75,.95,1]).tolist())
def shape_stats(a):
    a=np.asarray(a,dtype=np.float64);coeff=dct(a,type=2,axis=1,norm='ortho');power=np.square(coeff).sum(2);total=power.sum(1)
    return dict(rms=np.sqrt(np.square(a).mean((1,2))),velocity_energy=np.square(np.diff(a,axis=1)).mean((1,2)),
        acceleration_energy=np.square(np.diff(a,n=2,axis=1)).mean((1,2)),
        dc_fraction=np.divide(power[:,0],total,out=np.zeros_like(total),where=total>0),
        high_frequency_fraction=np.divide(power[:,7:].sum(1),total,out=np.zeros_like(total),where=total>0),
        spectral_centroid=np.divide((power*np.arange(25)[None]).sum(1),24*total,out=np.zeros_like(total),where=total>0))
def score(x,fit):
    d=x.reshape(len(x),-1)-fit.location_;return np.einsum('ni,ij,nj->n',d,fit.precision_,d)/125
def spectrum(c):
    eig=np.linalg.eigvalsh(c);positive=np.maximum(eig,0);p=positive/positive.sum()
    return dict(eigenvalues=eig[::-1].tolist(),participation=float(positive.sum()**2/np.square(positive).sum()),
        entropy_effective_rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),rank_numpy=int(np.linalg.matrix_rank(c)),
        dimensions95=int(np.searchsorted(np.cumsum(positive[::-1])/positive.sum(),.95)+1),trace=float(np.trace(c)))


def main():
    OUT.mkdir(exist_ok=True);assert not(OUT/'results.json').exists(),'Preserve previous review'
    manifest=read(M);card=read(D/'dataset_card.json');assert manifest['split']=='train'and manifest['count']==128 and manifest['window']==8 and manifest['frameskip']==5
    assert manifest['dataset_card_sha256']==sha(D/'dataset_card.json')
    for name,h in manifest['metadata_sha256'].items():assert sha(D/name)==h
    ids,offsets,lengths=[np.load(D/f)for f in ['episode_ids.npy','offsets.npy','lengths.npy']]
    allowed=set(np.load(D/'train_episodes.npy').tolist());held=set(np.load(D/'validation_episodes.npy').tolist());assert not allowed&held
    clips=[];provenance=[]
    with h5py.File(card['h5_path'],'r',swmr=True)as f:
        for w in manifest['windows']:
            ep=w['episode_array_index'];assert int(ids[ep])==w['episode']and w['episode']in allowed
            start=int(offsets[ep])+w['start'];assert w['rows']==(start+5*np.arange(8)).tolist()and start+25<=int(offsets[ep]+lengths[ep])
            x=np.asarray(f['action'][start:start+25],dtype=np.float32);assert x.shape==(25,5)and np.isfinite(x).all()
            assert np.all(np.asarray(f['ep_idx'][start:start+25])==w['episode'])
            clips.append(x);provenance.append(dict(episode=w['episode'],start_row=start,end_row_exclusive=start+25,action_sha256=digest(x)))
    assert len({w['episode']for w in provenance})==128
    raw=np.stack(clips);mean=np.asarray(card['action_mean']);std=np.asarray(card['action_std']);a=(raw.astype(np.float64)-mean)/std;flat=a.reshape(128,125)
    # The ONLY fitted statistics use these128 training trajectories; no candidate/physical outcome enters here.
    fit=LedoitWolf().fit(flat);loo=[];loo_alpha=[]
    for i in range(128):
        f=LedoitWolf().fit(np.delete(flat,i,axis=0));loo.append(float(score(a[i:i+1],f)[0]));loo_alpha.append(float(f.shrinkage_))
    cutoff=float(np.quantile(loo,.95));cov=np.cov(flat,rowvar=False,bias=True);marg=np.sqrt(np.diag(fit.covariance_));cor=fit.covariance_/marg[:,None]/marg[None,:]
    eig,q=np.linalg.eigh(cor);assert eig.min()>0;factor=(q*np.sqrt(eig))@q.T
    np.testing.assert_allclose(factor@factor.T,cor,atol=1e-12);np.testing.assert_allclose(np.diag(cor),1,atol=1e-12)
    temporal=np.corrcoef(a.transpose(0,2,1).reshape(-1,25),rowvar=False)
    channel=[]
    for c in range(5):
        v=a[:,:,c];channel.append(dict(index=c,raw_min=float(raw[:,:,c].min()),raw_max=float(raw[:,:,c].max()),
            mean_normalized=float(v.mean()),overall_std=float(v.std()),within_clip_std=quantiles(v.std(1)),exact_constant_clips=int((np.ptp(v,axis=1)==0).sum()),
            lag_correlation={str(l):float(np.corrcoef(v[:,:-l].ravel(),v[:,l:].ravel())[0,1])for l in [1,2,5,10,20]}))
    def transition_corr(indices):
        return [float(np.corrcoef(a[:,indices,c].ravel(),a[:,np.asarray(indices)+1,c].ravel())[0,1])for c in range(5)]
    within=[t for t in range(24)if t%5!=4];between=[t for t in range(24)if t%5==4]
    train=dict(provenance=provenance,raw_action_bytes_sha256=digest(raw),action_payload_bytes=int(raw.nbytes),normalized_shape=list(a.shape),
        channels=channel,block_internal_adjacent_correlation=transition_corr(within),block_boundary_adjacent_correlation=transition_corr(between),
        temporal_correlation=temporal.tolist(),empirical_covariance=spectrum(cov),shrunken_covariance=spectrum(fit.covariance_),shrinkage=float(fit.shrinkage_),
        correlation_spectrum=spectrum(cor),support_insample=quantiles(score(a,fit)),support_leave_one_episode_out=quantiles(loo),
        loo_shrinkage=quantiles(loo_alpha),reference_loo95=cutoff,shape={k:quantiles(v)for k,v in shape_stats(a).items()},raw_Box_outside_fraction=float((np.abs(raw)>1).mean()))
    # Outcome labels are first accessed AFTER the estimator and fixed references are finalized.
    cases=[];families_ref=None
    for path in sorted(POOL.glob('case*.npz')):
        with np.load(path)as p:
            x=p['actions'].astype(np.float64);families=p['family'].tolist();assert x.shape==(39,25,5)
            if families_ref is None:families_ref=families
            else:assert families==families_ref
            z=(x-mean)/std;d=score(z,fit);shape=shape_stats(z);success=p['success'].any(1);pose=p['physical_cost'][:,-1]
            cases.append(dict(case=int(p['case_index']),support=d.tolist(),above_train_loo95=(d>cutoff).tolist(),
                shape={k:v.tolist()for k,v in shape.items()},outside_fraction=(np.abs(x)>1).mean((1,2)).tolist(),success=success.tolist(),pose=pose.tolist(),sha256=sha(path)))
    assert len(cases)==50
    groups={name:[i for i,f in enumerate(families_ref)if f==name]for name in sorted(set(families_ref))}
    groups.update(nonoracle=[i for i,f in enumerate(families_ref)if f in ['zero','gaussian']or f.startswith('cem_')],
        oracle_neighborhood=[i for i,f in enumerate(families_ref)if f.startswith('recorded')],all=list(range(39)))
    summary={}
    for name,ix in groups.items():
        ds=np.asarray([[r['support'][i]for i in ix]for r in cases]);ss=np.asarray([[r['success'][i]for i in ix]for r in cases],bool)
        flags=ds>cutoff;casefrac=flags.mean(1);draw=casefrac[np.random.default_rng(1042).integers(50,size=(5000,50))].mean(1)
        summary[name]=dict(candidates_per_case=len(ix),support=quantiles(ds),above_loo95_fraction=float(flags.mean()),above_loo95_case_bootstrap=np.quantile(draw,[.025,.975]).tolist(),
            outside_Box_fraction=float(np.asarray([[r['outside_fraction'][i]for i in ix]for r in cases]).mean()),
            shape={k:quantiles(np.asarray([[r['shape'][k][i]for i in ix]for r in cases]))for k in shape_stats(a)},
            hold_success_fraction=float(ss.mean()),successful_support=quantiles(ds[ss])if ss.any()else None,failed_support=quantiles(ds[~ss])if(~ss).any()else None,
            case_mean_support=ds.mean(1).tolist())
    # Algebraic prospective control only: symmetric correlation factor preserves initial coordinate means/variances;
    # no candidate generation, cost evaluation, simulator call or deployment.
    np.savez_compressed(OUT/'training-action-statistics.npz',raw_train_actions=raw,normalized_train_actions=a,location=fit.location_,covariance=fit.covariance_,
        precision=fit.precision_,correlation=cor,correlation_factor=factor,loo_scores=np.asarray(loo))
    source=B/'sources/stable-worldmodel/stable_worldmodel/planning/solver/cem.py'
    result=dict(train=train,groups=summary,cases=cases,families=families_ref,manifest=str(M),manifest_sha256=sha(M),data_card_sha256=sha(D/'dataset_card.json'),
        cem_source=str(source),cem_source_sha256=sha(source),helper_sha256=sha(__file__),stats_sha256=sha(OUT/'training-action-statistics.npz'),
        scope='CPU only; original128 training episodes,first25 raw actions each; no RGB/full action-column/model/GPU/simulator/planning reads or calls. LW fit never sees validation candidates or success. LOO95 is a descriptive reference, not calibrated OOD probability; candidate outcomes only stratify the frozen statistic.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    compact={**result,'cases':None,'train':{k:v for k,v in train.items()if k not in ['provenance','temporal_correlation']}}
    (OUT/'summary.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(train={k:train[k]for k in ['channels','shrinkage','support_insample','support_leave_one_episode_out','reference_loo95','shape']},
        rank={k:train[k]['participation']for k in ['empirical_covariance','shrunken_covariance','correlation_spectrum']},
        groups={k:{x:v[x]for x in ['candidates_per_case','above_loo95_fraction','outside_Box_fraction','hold_success_fraction','support']}for k,v in summary.items()}),indent=2))


if __name__=='__main__':main()
