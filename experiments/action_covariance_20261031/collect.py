"""Collect only a complete48-endpoint bounded controller comparison; CPU analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
B=Path('/efs/prescottsun/SPT-JEPA')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ci(d):
    d=np.asarray(d,float);draw=np.random.default_rng(1042).integers(len(d),size=(5000,len(d)))
    return dict(delta_pp=float(100*d.mean()),case_bootstrap95_pp=(100*np.quantile(d[draw].mean(1),[.025,.975])).tolist())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.output)
    plan=read(root/'plan.json');assert len(plan['jobs'])==48
    exits={j['id']:read(root/'exits'/(j['id']+'.json'))for j in plan['jobs']};assert all(x['returncode']==0 for x in exits.values()),'No partial-results review'
    assert (root/'HOLD_DISPATCH').exists(),'Close owned idle dispatchers before collecting'
    out.mkdir(parents=True,exist_ok=True);assert not(out/'results.json').exists(),'Preserve previous collection'
    rows=[];grouped={};samebank={};certificates=[]
    for job in plan['jobs']:
        assert sha(job['spec'])==job['spec_sha256'];s=read(job['spec']);m=read(Path(job['output'])/'metrics.json');old=read(s['reference_metrics']);first=read(Path(job['output'])/'first-plan.json')
        assert first['first_plan_finite']
        for k in ['pairs','manifest_sha256','checkpoint','solver','plan','normalization','split','protocol','checkpoint_audit']:assert m[k]==old[k],(job['id'],k)
        assert sha(s['manifest'])==m['manifest_sha256']and len(m['pairs'])==50
        assert m['normalization']=='train episodes'and m['checkpoint_audit']['strict_equal']
        diag=m['controller_diagnostic'];assert diag['checkpoint_files_unchanged']and diag['matched_cases_checkpoint_solver_plan_normalization']
        assert diag['source']==s['source']and diag['helper_files']==s['helper_files']and diag['spec_sha256']==sha(job['spec'])
        assert diag['matrix_sha256']==sha(Path(s['calibration']).with_name('matrix.npz'))
        y=np.asarray(m['metrics']['episode_successes'],int);historical=np.asarray(old['metrics']['episode_successes'],int);assert y.shape==historical.shape==(50,)and np.isin(y,[0,1]).all()
        key=(s['task'],s['family'],s['training_seed'],s['case_set']);grouped.setdefault(key,{})[s['mode']]=y
        row={k:s[k]for k in ['task','family','step','training_seed','case_set','mode','checkpoint']};row.update(success_percent=float(100*y.mean()),successes=y.tolist(),historical_percent=float(100*historical.mean()),
            historical_successes=historical.tolist(),original_replay_changed_cases=np.flatnonzero(y!=historical).tolist()if s['mode']=='original'else None,
            source_commit=s['source']['source_commit'],parameters=m['checkpoint_audit']['parameters'],manifest_sha256=m['manifest_sha256'],metrics_path=str(Path(job['output'])/'metrics.json'),metrics_sha256=sha(Path(job['output'])/'metrics.json'))
        rows.append(row)
        if s['mode']=='correlated':
            assert first['first_return_map_max_absolute_error']<=2e-6 and diag['stats']['cost_calls']>0 and diag['stats']['execution_plans']>0
            bk=(s['task'],s['case_set']);samebank.setdefault(bk,[]).append(first['first_bank'])
        certificates.append(dict(id=job['id'],first_plan_sha256=sha(Path(job['output'])/'first-plan.json'),source_imports=first['model_imports']))
    comparisons=[]
    for key,values in grouped.items():
        assert set(values)=={'original','correlated'};task,family,seed,case=key;u,v=values['original'],values['correlated'];comparison=dict(task=task,family=family,seed=seed,case_set=case,original_percent=float(100*u.mean()),correlated_percent=float(100*v.mean()),
            gained_cases=np.flatnonzero((v==1)&(u==0)).tolist(),lost_cases=np.flatnonzero((v==0)&(u==1)).tolist(),**ci(v-u));comparisons.append(comparison)
    means=[]
    for task,family,case in sorted({(k[0],k[1],k[3])for k in grouped}):
        u=np.stack([grouped[(task,family,s,case)]['original']for s in [1,2]]);v=np.stack([grouped[(task,family,s,case)]['correlated']for s in [1,2]])
        means.append(dict(task=task,family=family,case_set=case,original_percent=float(100*u.mean()),correlated_percent=float(100*v.mean()),**ci((v-u).mean(0))))
    samebank_report=[]
    for key,banks in samebank.items():
        assert len(banks)==6
        white={x['white_sha256']for x in banks};mapped={x['normalized_action_sha256']for x in banks}
        assert len(white)==len(mapped)==1,(key,white,mapped)
        samebank_report.append(dict(task=key[0],case_set=key[1],models=6,white_sha256=next(iter(white)),mapped_sha256=next(iter(mapped))))
    result=dict(root=str(root),complete48=True,all_exits0=True,helper_commit=plan['jobs'][0]['source']['source_commit'],plan_sha256=sha(root/'plan.json'),rows=rows,
        paired_comparisons=comparisons,two_seed_means=means,correlated_first_bank_agreement=samebank_report,first_plan_certificates=certificates,
        inference='Paired comparisons use the newly recomputed original arm, never best-of historical/replay. Bootstrap unit is case; averaged-seed intervals resample the same case across both seeds and condition on the two trained models. No environment-repeat variance or training-seed population uncertainty included. Primary/fresh stay separate. Later CEM populations differ; only the first random bank is common within task/case set.',
        scope='Common controller intervention. No changed world-model learning objective, metric, weights, labels, action bounds or sample budget. Candidate differs by task: TwoRoom F64free vs Cube local1024. Exposure/capacity mismatches remain.')
    write(out/'results.json',result);write(out/'summary.json',{k:v for k,v in result.items()if k not in ['rows','first_plan_certificates']})
    text='# Fixed action-correlation CEM: complete48-endpoint comparison\n\nAll48 endpoints exited0; original/Cholesky arms retain the same checkpoint, cases, native source, scaler, goal metric and300×30 CEM budget. All first-plan costs/actions are finite and fixed-map checks pass. No training, clipping or optimal-action supervision was added.\n\n'
    text+='| Task | Family | Seed | Primary original → correlated | Fresh original → correlated |\n|---|---|---:|---:|---:|\n'
    lookup={(v['task'],v['family'],v['seed'],v['case_set']):v for v in comparisons}
    for task,family,seed in sorted({(k[0],k[1],k[2])for k in grouped}):
        p=lookup[(task,family,seed,'primary')];f=lookup[(task,family,seed,'fresh')]
        text+=f"| {task} | {family} | {seed} | {p['original_percent']:.0f} → {p['correlated_percent']:.0f}% | {f['original_percent']:.0f} → {f['correlated_percent']:.0f}% |\n"
    text+='\n| Task | Family | Cases | Two-seed mean original → correlated | Paired change,95% case CI (pp) |\n|---|---|---|---:|---:|\n'
    for v in means:
        lo,hi=v['case_bootstrap95_pp'];text+=f"| {v['task']} | {v['family']} | {v['case_set']} | {v['original_percent']:.0f} → {v['correlated_percent']:.0f}% | {v['delta_pp']:+.1f} [{lo:+.1f},{hi:+.1f}] |\n"
    replays=[r for r in rows if r['mode']=='original'and r['original_replay_changed_cases']]
    text+=f"\n{len(replays)}/24 new original-arm success vectors differ from historical references. These are retained separately; the new original arm is the paired comparator. Exact changed-case indices and all per-seed confidence intervals are in results.json.\n\n"
    text+=result['inference']+'\n\n'+result['scope']+'\n\n'
    text+='This control tests search parameterization, not a new learning algorithm. TwoRoom and Cube use different candidate families; native/candidate capacity differs and Cube local6000 has less exposure than native/TF12000. Existing colored-noise CEM precedent is [iCEM](https://proceedings.mlr.press/v155/pinneri21a.html). Matrices were frozen solely from pre-existing training128 cohorts. No task results select covariance, factorization or model. No further runs are triggered.\n'
    (out/'RESULTS.md').write_text(text);print(json.dumps(dict(complete48=True,means=means,original_replays_different=len(replays))))
if __name__=='__main__':main()
