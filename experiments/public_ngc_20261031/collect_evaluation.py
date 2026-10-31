"""CPU aggregate; refuse to read any outcomes until all48 endpoints exit0."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from common import read,write,sha


def paired(a,b):
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    assert delta.ndim==1 and len(delta)==50
    rng=np.random.default_rng(20261031)
    boot=delta[rng.integers(0,50,size=(10000,50))].mean(1)*100
    return dict(delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,out=Path(a.root),Path(a.output);plan=read(root/'plan.json');assert len(plan['jobs'])==48
    exits={j['id']:read(root/'exits'/(j['id']+'.json'))for j in plan['jobs']}
    assert all(x['returncode']==0 for x in exits.values()),'Do not inspect partial outcomes'
    digest={}
    def check(path,expected):
        path=str(path)
        if path not in digest:digest[path]=sha(path)
        assert digest[path]==expected,path
    sources=set();rows=[];lookup={};first={};casehash={}
    for j in plan['jobs']:
        check(j['spec'],j['spec_sha256']);s=read(j['spec']);m=read(Path(j['output'])/'metrics.json');d=m['ngc_diagnostic'];ref=read(s['reference_metrics'])
        assert s['numeric_variant']==d['numeric_variant']=='uniform-fp32-direct-attachment'
        for path,h in s['immutable_inputs'].items():check(path,h)
        for source in [s['model_source'],s['helper_source']]:
            for key in ['manifest','runtime_manifest']:
                if source[key]in sources:continue
                manifest=read(source[key])
                for f,h in manifest['files'].items():check(Path(manifest['source'])/f,h)
                sources.add(source[key])
        for f,h in s['helper_files'].items():check(Path(s['helper_source']['source'])/'experiments/public_ngc_20261031'/f,h)
        check(s['graph_gate'],d['graph_gate_sha256']);gate=read(s['graph_gate']);assert gate['passed']and len(gate['graphs'])==12
        check(d['graph_report'],d['graph_report_sha256']);g=read(d['graph_report']);check(Path(d['graph_report']).with_name('graph.npz'),g['graph_sha256'])
        assert g['checkpoint']==s['checkpoint']and g['weights_sha256']==s['immutable_inputs'][str(Path(s['checkpoint'])/'weights.pt')]
        assert g['cohort_sha256']==sha(s['cohort'])and g['numeric_variant']==s['numeric_variant']
        for key in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol','checkpoint_audit']:assert m[key]==ref[key],(j['id'],key)
        assert m['pairs']==read(s['manifest'])['pairs']and len(m['pairs'])==50
        key=(s['task'],s['case_set'])
        if key in casehash:assert casehash[key]==m['manifest_sha256']
        casehash[key]=m['manifest_sha256']
        state=d['model_audit'];assert state['state_before']==state['state_after']and state['actual_scaler_checked']
        check(state['model_source'],state['model_source_sha256']);assert Path(state['model_source']).is_relative_to(Path(s['model_source']['source']))
        check(j['spec'],d['spec_sha256']);check(Path(s['helper_source']['source'])/'experiments/public_ngc_20261031/stable_cost.py',d['stable_cost_sha256'])
        bank=d['shared_bank_audit'];assert bank['extra_model_calls']==0 and bank['metric_order']==['l2','ngc']
        directory=Path(j['output']);check(directory/'banks.json',bank['banks_sha256']);check(directory/'costs.npz',bank['costs_sha256'])
        banks=read(directory/'banks.json');assert len(banks)==bank['calls']
        assert m['solver']['batch_size']==1 and m['solver']['n_steps']==30 and m['solver']['num_samples']==300
        assert len(banks)>=1500 and len(banks)%30==0
        # Native CEM completes all30 iterations for each single-case batch.
        # First solve includes all50 cases, so calls0,30,...1470 are common
        # iteration-zero banks; later replans can omit successful environments.
        initial_calls=np.arange(50)*30
        assert all(banks[i]['shape']==[2,1,300]for i in initial_calls)
        with np.load(directory/'costs.npz')as f:
            costs=f['costs'];offset=f['call_offsets'];assert np.isfinite(costs).all()and len(offset)==len(banks)+1
            assert np.all(np.diff(offset)==1)
            initial=costs[:,offset[initial_calls]].copy()
        assert initial.shape[0]==2 and initial.shape[1]==50
        ranks=rankdata(initial,axis=-1);center=ranks-ranks.mean(-1,keepdims=True)
        denom=np.sqrt((center[0]**2).sum(-1)*(center[1]**2).sum(-1));valid=denom>0
        corr=np.divide((center[0]*center[1]).sum(-1),denom,out=np.zeros_like(denom),where=valid)
        y=np.asarray(m['metrics']['episode_successes'],dtype=int);old=np.asarray(ref['metrics']['episode_successes'],dtype=int)
        assert y.shape==old.shape==(50,)and np.isin(y,[0,1]).all()and np.isin(old,[0,1]).all()
        replay=None
        if s['mode']=='l2':
            exact=bool(np.array_equal(y,old));assert exact==d['original_replay_successes_exact']
            replay=dict(exact=exact,historical_successes=old.tolist(),historical_percent=float(old.mean()*100),current_percent=float(y.mean()*100),
                changed_indices=np.flatnonzero(y!=old).tolist(),gains=np.flatnonzero((y==1)&(old==0)).tolist(),losses=np.flatnonzero((y==0)&(old==1)).tolist(),
                reference=s['reference_metrics'],paired=paired(y,old))
        row=dict(id=j['id'],task=s['task'],family=s['family'],seed=s['seed'],step=s['step'],dimension=s['dimension'],mode=s['mode'],case_set=s['case_set'],
            checkpoint=s['checkpoint'],model_source_commit=s['model_source']['source_commit'],parameters=m['checkpoint_audit']['parameters'],
            successes=y.tolist(),percent=float(y.mean()*100),success_count=int(y.sum()),cases=50,manifest_sha256=m['manifest_sha256'],
            immutable_state=True,actual_scaler_checked=True,model_state_hash=state['state_before'],replay=replay,
            seconds=exits[j['id']]['finished_at']-exits[j['id']]['started_at'],
            first_bank=dict(metadata=[banks[i]for i in initial_calls],call_indices=initial_calls.tolist(),mean_spearman=float(corr[valid].mean())if valid.any()else None,finite_spearman_cases=int(valid.sum()),
                spearman_per_case=[float(corr[i])if valid[i]else None for i in range(50)],argmin_agreement=float((initial[0].argmin(-1)==initial[1].argmin(-1)).mean()),
                score_sha256=bank['costs_sha256']),metrics_path=str(directory/'metrics.json'),metrics_sha256=sha(directory/'metrics.json'))
        rows.append(row);lookup[(s['task'],s['family'],s['seed'],s['case_set'],s['mode'])]=row;first[j['id']]=initial
    comparisons=[];combined=[]
    for task in ['tworoom','cube']:
        for family in ['local1024','native-raw','TF64']:
            for case in ['primary','fresh']:
                deltas=[]
                for seed in [1,2]:
                    l2=lookup[(task,family,seed,case,'l2')];ngc=lookup[(task,family,seed,case,'ngc')]
                    assert l2['checkpoint']==ngc['checkpoint']and l2['model_state_hash']==ngc['model_state_hash']
                    y,z=np.array(ngc['successes']),np.array(l2['successes']);deltas.append(y-z)
                    keys=['actions_sha256','goal_sha256','predicted_sha256']
                    same_rows={k:[aa[k]==bb[k]for aa,bb in zip(l2['first_bank']['metadata'],ngc['first_bank']['metadata'])]for k in keys}
                    same={k:all(v)for k,v in same_rows.items()}
                    score_rows=np.all(first[l2['id']]==first[ngc['id']],axis=(0,2))
                    identical_inputs=np.all(np.asarray(list(same_rows.values())),axis=0)
                    assert np.all(score_rows[identical_inputs]),'Identical first bank inputs produced different saved cost arrays'
                    score_equal=bool(score_rows.all())
                    comparisons.append(dict(task=task,family=family,seed=seed,case_set=case,l2=l2['percent'],ngc=ngc['percent'],
                        **paired(y,z),gains=np.flatnonzero((y==1)&(z==0)).tolist(),losses=np.flatnonzero((y==0)&(z==1)).tolist(),
                        first_bank_equal=same,first_bank_equal_counts={k:sum(v)for k,v in same_rows.items()},first_scores_exact=score_equal,
                        first_scores_exact_cases=int(score_rows.sum()),first_input_equal_cases=int(identical_inputs.sum())))
                mean_delta=np.mean(deltas,axis=0)
                combined.append(dict(task=task,family=family,case_set=case,**paired(mean_delta,np.zeros(50)),
                    l2=float(np.mean([lookup[(task,family,s,case,'l2')]['percent']for s in [1,2]])),
                    ngc=float(np.mean([lookup[(task,family,s,case,'ngc')]['percent']for s in [1,2]]))))
    write(out/'results.json',dict(passed=True,root=str(root),source_commit=plan['jobs'][0]['source']['source_commit'],rows=rows,comparisons=comparisons,combined=combined,
        all48exit0=True,immutable_inputs_checked=len(digest),manifest_hashes={str(k):v for k,v in casehash.items()},
        limitations='Case-paired bootstrap10000/fixedseed20261031; two-seed summary resamples the same50case indices after averaging seed differences, conditional on these two trained seeds. Primary/fresh separate. CI excludes environment-repeat variance and training-seed uncertainty. Current L2 retained regardless of historical replay differences. Subsequent CEM populations may diverge; common-bank claim limited to checked first bank. Both arms pay both-score recording cost; runtime is not bare L2 speed. Model capacity/exposure differ across families; no learning-algorithm novelty claim.'))
    print(__import__('json').dumps(dict(passed=True,rows=48,comparisons=24,combined=combined)))


if __name__=='__main__':main()
