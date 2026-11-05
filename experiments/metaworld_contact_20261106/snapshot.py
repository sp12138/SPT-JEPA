"""Export numerical evidence,checking every reported main/fresh physical trace."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

from experiments.metaworld_contact_20261106.collect import summary, ci, MANIP


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(root, output):
    output.mkdir(parents=True,exist_ok=False)
    index=[]
    def write(relative, value, source=None):
        path=output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2)+'\n')
        index.append(dict(path=str(relative),sha256=digest(path),bytes=path.stat().st_size,source=str(source) if source else None))
    def copy(path, relative=None):
        if not path.exists():return
        content=path.read_bytes();target=output/(relative or path.relative_to(root));target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        index.append(dict(path=str(target.relative_to(output)),sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),source=str(path)))
    value=json.loads((root/'review/summary.json').read_text())
    for name in ['summary.json','compact.json']:copy(root/'review'/name)
    groups={}
    old_path=root.parent/'metaworld_improve_20261104/results/summary.json'
    old=json.loads(old_path.read_text())
    for arm in ['tf-control','native-control']:
        rows=[dict(r,training_seed=r['train_seed']) for g in old['groups'] if g['arm']==arm and g['step']==16480 for r in g['observations']]
        assert len(rows)==320
        groups['main',arm]=dict(rows=rows,**summary(rows))
    write('baseline-rows.json',dict(source=str(old_path),source_sha256=digest(old_path),groups=[dict(arm=arm,**groups['main',arm]) for arm in ['tf-control','native-control']]))
    for phase,cohort in [('long-v1','main'),('fresh-v1','fresh')]:
        for group in value['groups']:
            if group['phase']!=phase:continue
            if group['step'] not in [16480,40000]:continue
            assert len(group['rows'])==(320 if cohort=='main' else 160)
            groups[cohort,group['arm']]=group
    for arm in ['tf-control','native-control']:
        groups['main4',arm]=groups['main',arm]
    for arm in ['local-mse','local-flow']:
        extra=next(g for g in value['groups'] if g['phase']=='repeat-v1' and g['arm']==arm and g['step']==40000)
        assert len(extra['rows'])==320
        rows=groups['main',arm]['rows']+extra['rows']
        assert {r['training_seed'] for r in rows}=={1,2,3,4}
        groups['main4',arm]=dict(rows=rows,**summary(rows))
    arms=['tf-control','native-control','local-mse','local-flow']
    trace_checks={};tables={}
    for cohort in ['main','main4','fresh']:
        expected=None;tables[cohort]={}
        for arm in arms:
            group=groups[cohort,arm]
            identity={(r['task'],r['seed'],r['training_seed']):r['case_sha256'] for r in group['rows']}
            assert len(identity)==len(group['rows'])
            if expected is None:expected=identity
            if cohort=='main4':
                # Controls have two trained models;candidate repeats3/4 share
                # exactly the same physical cases,not new physical samples.
                physical={(t,s):h for (t,s,_),h in identity.items()}
                assert physical=={(t,s):h for (t,s,_),h in expected.items()},(cohort,arm)
            else:assert identity==expected,(cohort,arm,'physicalcase mismatch')
            for row in group['rows']:
                path=row['trace']
                if path not in trace_checks:
                    assert digest(path)==row['trace_sha256'],path
                    trace_checks[path]=row['trace_sha256']
            tables[cohort][arm]=dict(**summary(group['rows']),case_cluster95=ci(group['rows']),
                per_training_seed={seed:summary([r for r in group['rows'] if r['training_seed']==seed]) for seed in sorted({r['training_seed'] for r in group['rows']})})
    write('trace-checks.json',dict(passed=True,traces=len(trace_checks),sha256_by_path=trace_checks,
        common_case_hashes_across_all4_methods=True,main_cases_per_task=20,fresh_cases_per_task=10,
        candidate_trained_models=4,baseline_trained_models=2))
    diagnostic=[];factorials={}
    for path in sorted((root/'diagnostics').glob('*/summary.json')):
        v=json.loads(path.read_text());assert v['passed'] and v['model_unchanged'] and v['model_updates']==0
        copy(path)
        for row in v['rows']:
            if row['task'] not in MANIP:continue
            expert=row['paths']['expert_diagnostic'];cem=row['paths']['recorded_cem']
            valid=expert['any_success']
            if valid:
                diagnostic.append(dict(model=v['name'],task=row['task'],seed=row['seed'],
                    failed_cem_preferred_at30=cem['macro_goal_mse'][6]<expert['macro_goal_mse'][6],
                    cem_failed=not cem['any_success']))
            if v['name'].startswith('tf-control'):
                f=expert['factorial']['goal_mse'];factorials.setdefault(row['task'],[]).append(f[1]/f[0])
    assert len(diagnostic)==46 and all(r['failed_cem_preferred_at30'] and r['cem_failed'] for r in diagnostic)
    for phase in ['', 'long-v1','repeat-v1','vit-v1','vit-mse-v1','marginal-local-v1','history-v1','fresh-v1','broad-v1']:
        folder=root/phase
        for pattern in ['*plan*.json','launch-*.json','GOALS_APPROVED.json']:
            for path in folder.glob(pattern):copy(path)
        for pattern in ['bridge/*/config.json','bridge/*/status.json','bridge/*/complete.json','bridge/*/FORK.json',
                        'bridge/*/ORIGIN.json','bridge/*/audit-*.json','bridge/*/eval-step0040000/summary.json']:
            for path in folder.glob(pattern):copy(path)
    for pattern in ['confirmation10k/*/summary.json','confirmation10k/*/parent-removal-proof.json',
                    'fresh-v1/*/*/summary.json','fresh-v1/*/complete.json','finite-collector-launch-v3.json']:
        for path in root.glob(pattern):copy(path)
    write('analysis.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tables=tables,
        frozen_diagnosis=dict(valid_model_cases=len(diagnostic),failed_cem_preferred_at30=sum(r['failed_cem_preferred_at30'] for r in diagnostic),
            rows=diagnostic,tf_goal_arm_wrong_object_cost_fraction={k:dict(min=min(v),max=max(v)) for k,v in factorials.items()}),
        scope=value['scope'],traces_checked=len(trace_checks),
        cautions=['Main20 physicalcases/task were used in the exploratory investigation;fresh10 were fixed after freezing these CNNs.',
            'Case intervals condition on the trainedmodels and do not measure initialization uncertainty;main4 candidates have4 models,controls2.',
            'Same corpus/trajectorysplit/evaluation but different phase sampling,models,objectives,training and inferencecompute.',
            'Action reconstruction uses factual loggedactions,no oracle optimal-action labels;this is a goal-conditioned inverse controller,not a pureforecastloss win.',
            'PickPlace is trainingseed-sensitive and Reach/ReachWall regress;no unified alltask solution or novelty/SOTA claim.']))
    write('INDEX.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=index.copy()))
    print(json.dumps(dict(passed=True,files=len(index),traces=len(trace_checks),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.root,a.output)
