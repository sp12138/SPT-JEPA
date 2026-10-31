"""Read-only CPU screen of one predeclared block norm on a complete old pool."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B / 'runs/cube_pool_reuse2000_20261031'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def interval(x):
    x=np.asarray(x,dtype=float)
    draws=x[np.random.default_rng(1042).integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n=len(x))

def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'results.json').exists()
    rows=[];costs={};max_error=0.;provenance=[]
    for seed in [1,2]:
        rpath=ROOT/f'local6000-seed{seed}/result.json';r=read(rpath)
        assert r['passed'] and r['state_before']==r['state_after'] and r['rng_before']==r['rng_after']
        assert len(r['cases'])==50 and r['spec']['dimension']==1024
        provenance.append(dict(seed=seed,result=str(rpath),result_sha256=sha(rpath),spec=r['spec'],cases=r['cases']))
        for rec in r['cases']:
            assert sha(rec['file'])==rec['sha256']
            with np.load(rec['file']) as f:
                goal=f['goal'];family=f['family'];ph=f['physical_cost'][:,-1];pos=f['position_error_m'][:,-1];angle=f['orientation_error_deg'][:,-1];hold=f['success'].any(1)
                assert goal.shape==(1024,) and len(ph)==39
                ids=np.flatnonzero(np.isin(family,['zero','gaussian'])|np.char.startswith(family,'cem_'))
                assert len(ids)==22
                for endpoint,key,oldkey in [('encoded','true_endpoint','encoded_cost'),('predicted','predicted_endpoint','predicted_cost')]:
                    z=f[key];assert z.shape==(39,1024) and z.dtype==goal.dtype==np.float32
                    sq=(z-goal).reshape(39,64,16)**2
                    block=sq.sum(-1);mean=block.mean(-1);maximum=block.max(-1)
                    old=f[oldkey];assert np.isfinite(block).all()
                    error=float(np.abs(mean*64-old).max());max_error=max(max_error,error)
                    assert np.allclose(mean*64,old,rtol=2e-6,atol=1e-8)
                    costs[f'seed{seed}_case{rec["case"]:04d}_{endpoint}']=np.stack([mean,maximum])
                    for group,indices in [('nonoracle',ids),('all',np.arange(39))]:
                        base=int(indices[np.argmin(mean[indices])]);candidate=int(indices[np.argmin(maximum[indices])]);original=int(indices[np.argmin(old[indices])])
                        assert base==original
                        entry=dict(seed=seed,case=rec['case'],endpoint=endpoint,group=group,mean_index=base,max_index=candidate,mean_family=str(family[base]),max_family=str(family[candidate]),changed=base!=candidate,coverage_hold=float(hold[indices].any()),physical_oracle_cost=float(ph[indices].min()))
                        for name,i in [('mean',base),('max',candidate)]:
                            for k,v in dict(pose_cost=ph[i],position_cm=pos[i]*100,angle_deg=angle[i],joint=(pos[i]<=.03 and angle[i]<=15),hold=hold[i]).items():entry[f'{name}_{k}']=float(v)
                            entry[f'{name}_physical_rank']=float(spearmanr(ph[indices],(mean if name=='mean' else maximum)[indices]).statistic)
                        rows.append(entry)
    assert len(rows)==400
    metrics=['pose_cost','position_cm','angle_deg','joint','hold','physical_rank']
    summaries=[]
    for seed in [1,2]:
        for endpoint in ['encoded','predicted']:
            for group in ['nonoracle','all']:
                rr=[x for x in rows if (x['seed'],x['endpoint'],x['group'])==(seed,endpoint,group)]
                assert len(rr)==50 and len({x['case']for x in rr})==50
                summaries.append(dict(seed=seed,endpoint=endpoint,group=group,changed=sum(x['changed']for x in rr),mean={k:interval([x['mean_'+k]for x in rr])for k in metrics},max={k:interval([x['max_'+k]for x in rr])for k in metrics},paired_max_minus_mean={k:interval([x['max_'+k]-x['mean_'+k]for x in rr])for k in metrics}))
    paired=[]
    for endpoint in ['encoded','predicted']:
        for group in ['nonoracle','all']:
            rr=sorted([x for x in rows if x['endpoint']==endpoint and x['group']==group],key=lambda x:(x['seed'],x['case']))
            assert [x['case']for x in rr[:50]]==[x['case']for x in rr[50:]]
            paired.append(dict(endpoint=endpoint,group=group,conditional_two_seed_case_average={k:interval((np.array([x['max_'+k]-x['mean_'+k]for x in rr]).reshape(2,50)).mean(0))for k in metrics}))
    np.savez_compressed(output/'costs.npz',**costs)
    result=dict(passed=True,script_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),read_only_model_calls=0,recomputed_original_max_absolute_error=max_error,all400_original_selection_replays_exact=True,summaries=summaries,paired=paired,cases=rows,provenance=provenance,cost_arrays_sha256=sha(output/'costs.npz'))
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([x for x in summaries if x['group']=='nonoracle'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
