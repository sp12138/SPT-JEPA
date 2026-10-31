"""CPU aggregate; refuse to read any outcomes until all48 endpoints exit0."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'public_ngc_20261031'))
from common import read,write,sha


def paired(a,b):
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    assert delta.ndim==1 and len(delta)==50
    rng=np.random.default_rng(20261031)
    boot=delta[rng.integers(0,50,size=(10000,50))].mean(1)*100
    return dict(delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--white-summary',required=True);a=p.parse_args()
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
        assert s['mode']==d['mode'] and s['controller_mode']=='correlated'
        for path,h in s['combined_immutable_inputs'].items():check(path,h)
        for source in [s['model_source'],s['helper_source'],s['combined_helper_source'],s['correlation_source']]:
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
        cd=m['controller_diagnostic'];stats=cd['stats'];assert cd['mode']=='correlated'
        assert cd['source']==s['correlation_source'] and cd['combined_source']==s['combined_helper_source']
        assert cd['mapped_objective_metadata_verified'] and d['original_replay_successes_exact'] is None
        assert d['white_reference_success_vector_equal']==bool(np.array_equal(y,old))
        check(j['spec'],cd['combined_spec_sha256']);check(s['reference_metrics'],cd['reference_white_sha256'])
        check(s['calibration'],cd['calibration_sha256']);check(Path(s['calibration']).with_name('matrix.npz'),cd['matrix_sha256'])
        cal=read(s['calibration']);assert cal['zero_variance_coordinates']==0
        assert stats['cost_calls']==bank['calls'] and stats['solves']>=1 and stats['execution_plans']>=50
        assert not stats['identity_exact_bypass']
        startup=read(directory/'first-plan.json')
        assert startup['first_bank']==stats['first_bank'] and startup['first_plan_finite']
        assert startup['first_return_map_max_absolute_error']<=2e-6
        for file,h in stats['model_imports'].items():
            check(file,h);assert Path(file).is_relative_to(Path(s['model_source']['source']))
        for file,h in s['combined_helper_files'].items():
            check(Path(s['combined_helper_source']['source'])/'experiments/public_ngc_covariance_20261031'/file,h)
        row=dict(id=j['id'],task=s['task'],family=s['family'],seed=s['seed'],step=s['step'],dimension=s['dimension'],mode=s['mode'],case_set=s['case_set'],
            checkpoint=s['checkpoint'],model_source_commit=s['model_source']['source_commit'],parameters=m['checkpoint_audit']['parameters'],
            successes=y.tolist(),percent=float(y.mean()*100),success_count=int(y.sum()),cases=50,manifest_sha256=m['manifest_sha256'],
            immutable_state=True,actual_scaler_checked=True,model_state_hash=state['state_before'],replay=None,
            controller=dict(calibration_sha256=cd['calibration_sha256'],matrix_sha256=cd['matrix_sha256'],stats=stats,
                mapped_objective_metadata_verified=True,white_reference=s['reference_metrics'],white_reference_sha256=cd['reference_white_sha256']),
            seconds=exits[j['id']]['finished_at']-exits[j['id']]['started_at'],
            first_bank=dict(metadata=[banks[i]for i in initial_calls],call_indices=initial_calls.tolist(),mean_spearman=float(corr[valid].mean())if valid.any()else None,finite_spearman_cases=int(valid.sum()),
                spearman_per_case=[float(corr[i])if valid[i]else None for i in range(50)],argmin_agreement=float((initial[0].argmin(-1)==initial[1].argmin(-1)).mean()),
                score_sha256=bank['costs_sha256']),metrics_path=str(directory/'metrics.json'),metrics_sha256=sha(directory/'metrics.json'))
        rows.append(row);lookup[(s['task'],s['family'],s['seed'],s['case_set'],s['mode'])]=row;first[j['id']]=initial
    assert len(lookup)==48
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
                    assert all(same.values()) and score_equal,'Correlated L2/NGC initial banks differ'
                    assert l2['controller']['stats']['first_bank']==ngc['controller']['stats']['first_bank']
                    comparisons.append(dict(task=task,family=family,seed=seed,case_set=case,l2=l2['percent'],ngc=ngc['percent'],
                        **paired(y,z),gains=np.flatnonzero((y==1)&(z==0)).tolist(),losses=np.flatnonzero((y==0)&(z==1)).tolist(),
                        first_bank_equal=same,first_bank_equal_counts={k:sum(v)for k,v in same_rows.items()},first_scores_exact=score_equal,
                        first_scores_exact_cases=int(score_rows.sum()),first_input_equal_cases=int(identical_inputs.sum())))
                mean_delta=np.mean(deltas,axis=0)
                combined.append(dict(task=task,family=family,case_set=case,**paired(mean_delta,np.zeros(50)),
                    l2=float(np.mean([lookup[(task,family,s,case,'l2')]['percent']for s in [1,2]])),
                    ngc=float(np.mean([lookup[(task,family,s,case,'ngc')]['percent']for s in [1,2]]))))
    white=read(a.white_summary);assert white['passed'] and white['all48exit0'] and len(white['rows'])==48
    white_lookup={(r['task'],r['family'],r['seed'],r['case_set'],r['mode']):r for r in white['rows']}
    assert len(white_lookup)==48
    factorial=[];factorial_combined=[]
    for task in ['tworoom','cube']:
        for family in ['local1024','native-raw','TF64']:
            for case in ['primary','fresh']:
                seed_deltas=[];cells=[]
                for seed in [1,2]:
                    key=(task,family,seed,case)
                    cl,cn=lookup[key+('l2',)],lookup[key+('ngc',)]
                    wl,wn=white_lookup[key+('l2',)],white_lookup[key+('ngc',)]
                    for c,w in [(cl,wl),(cn,wn)]:
                        check(w['metrics_path'],w['metrics_sha256'])
                        actual_white=read(w['metrics_path'])
                        assert w['successes']==actual_white['metrics']['episode_successes']
                        assert w['percent']==float(np.mean(w['successes'])*100)
                        assert c['controller']['white_reference']==w['metrics_path']
                        assert c['controller']['white_reference_sha256']==w['metrics_sha256']
                        for field in ['checkpoint','model_state_hash','parameters','step','dimension','manifest_sha256','model_source_commit']:
                            assert c[field]==w[field],(c['id'],field)
                    arrays=[np.array(r['successes'])for r in [wl,wn,cl,cn]]
                    w0,w1,c0,c1=arrays
                    contrasts=dict(white_ngc_minus_l2=w1-w0,correlated_ngc_minus_l2=c1-c0,
                        correlated_minus_white_l2=c0-w0,correlated_minus_white_ngc=c1-w1,
                        interaction=(c1-c0)-(w1-w0))
                    seed_deltas.append(contrasts);cells.append([r['percent']for r in [wl,wn,cl,cn]])
                    factorial.append(dict(task=task,family=family,case_set=case,seed=seed,
                        cells=dict(zip(['white_l2','white_ngc','correlated_l2','correlated_ngc'],cells[-1])),
                        contrasts={k:paired(v,np.zeros(50))for k,v in contrasts.items()},
                        interaction_per_case=contrasts['interaction'].tolist()))
                factorial_combined.append(dict(task=task,family=family,case_set=case,
                    cells=dict(zip(['white_l2','white_ngc','correlated_l2','correlated_ngc'],np.mean(cells,axis=0).tolist())),
                    contrasts={k:paired(np.mean([v[k]for v in seed_deltas],axis=0),np.zeros(50))for k in seed_deltas[0]}))
    result=dict(passed=True,root=str(root),source_commit=plan['jobs'][0]['source']['source_commit'],rows=rows,comparisons=comparisons,combined=combined,
        all48exit0=True,immutable_inputs_checked=len(digest),manifest_hashes={str(k):v for k,v in casehash.items()},
        factorial=factorial,factorial_combined=factorial_combined,white_summary=str(a.white_summary),white_summary_sha256=sha(a.white_summary),
        historical_replay=dict(exact=sum(r['replay']['exact']for r in white['rows']if r['replay']),
            differences=[dict(id=r['id'],**r['replay'])for r in white['rows']if r['replay']and not r['replay']['exact']]),
        limitations='Case-paired bootstrap10000/fixedseed20261031; average the two seed differences per case before resampling. Conditional on these two trained seeds; CI excludes environment repeat variability and seed-population uncertainty. Primary/fresh separate. Contemporary white L2, white NGC, correlated L2 and correlated NGC are retained without historical or best-of selection. The 2x2 interaction is a difference-in-differences of success indicators, not a causal proof excluding simulation repeat noise. First-iteration common-bank proof applies within a proposal regime; later CEM populations diverge, and white/correlated banks intentionally differ. Both arms pay both-score recording cost, so runtime is not bare L2 latency. Model dimensions/capacities/exposures differ; controller comparison is not a new learned algorithm.')
    write(out/'results.json',result)
    # Keep all per-case success/contrasts and diagnostics locally, large bank lists on EFS.
    compact=__import__('copy').deepcopy(result)
    for row in compact['rows']:row['first_bank'].pop('metadata')
    compact['full_results_source']=str(out/'results.json');compact['full_results_sha256']=sha(out/'results.json')
    write(out/'results-compact.json',compact)
    print(__import__('json').dumps(dict(passed=True,rows=48,comparisons=24,factorial_combined=factorial_combined)))


if __name__=='__main__':main()
