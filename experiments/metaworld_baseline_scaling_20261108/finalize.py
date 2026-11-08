"""One-shot baseline closeout after all existing primary recipes finish.

No launcher, timer, neural inference or simulation. Missing final trace audits
are completed once; existing successful audits are reused.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path('/efs/prescottsun/SPT-JEPA')
ROOT = BASE/'runs/metaworld_baseline_scaling_20261108'
HARD = {'bin-picking-v3', 'door-open-v3', 'drawer-open-v3',
        'peg-insert-side-v3', 'pick-place-v3', 'push-v3'}


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source = ROOT/'results/summary.json'
    summary = read(source)
    assert summary['complete'] == 18 and summary['errors'] == 0 and summary['skipped'] == 0
    assert (ROOT/'CLOSED.json').exists()
    plan = read(ROOT/'plan.json')
    assert len(plan['jobs']) == 18 and all(r['complete'] and not r['error'] for r in summary['records'])
    audits = []
    new_audits = 0
    for job in plan['jobs']:
        p = ROOT/'audits'/f"{job['name']}-epoch300.json"
        if not p.exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('phase-audit.py')),
                            '--name', job['name'], '--epoch', '300'], check=True)
            new_audits += 1
        audit = read(p)
        assert audit['passed'] and audit['actual_trace_files_verified'] == 1000
        assert audit['name'] == job['name'] and audit['native_epoch'] == 300
        assert audit['weights_sha256'] == digest(Path(job['output'])/f"model-step{job['epoch_steps']['300']:07d}/weights.pt")
        audits.append(dict(path=str(p), sha256=digest(p)))
    records = {r['name']:r for r in summary['records']}
    models = []
    for job in plan['jobs']:
        r = records[job['name']]
        epochs = {}
        for epoch in ['100', '200', '300']:
            score = r['epochs'][epoch]
            assert score['episodes'] == 1000 and len(score['groups']) == 50
            assert all(g['episodes'] == 20 for g in score['groups'].values())
            count = sum(score['groups'][t]['successes'] for t in HARD)
            epochs[epoch] = dict(macro50=score['macro50'], hard6_successes=count,
                                 hard6_episodes=120, hard6_percent=100*count/120)
        paired = {}
        for epoch in ['100', '300']:
            parent = Path(job['parent_output'] if epoch == '100' else job['output'])
            step = job['epoch_steps'][epoch]
            rows = [row for i in [0, 1] for row in read(parent/f'evaluation-{step:07d}-{i}/metrics.json')['episodes']]
            paired[epoch] = {(v['task'],v['seed']):(v['case_sha256'], bool(v['success'])) for v in rows}
            assert len(paired[epoch]) == 1000
        assert set(paired['100']) == set(paired['300'])
        assert all(paired['100'][k][0] == paired['300'][k][0] for k in paired['100'])
        gained = sum(not paired['100'][k][1] and paired['300'][k][1] for k in paired['100'])
        lost = sum(paired['100'][k][1] and not paired['300'][k][1] for k in paired['100'])
        models.append(dict(name=job['name'], method=job['method'], seed=job['seed'], epochs=epochs,
                           paired_scaling_gained_cases=gained, paired_scaling_lost_cases=lost))
    methods = {}
    for method in sorted({j['method'] for j in plan['jobs']}):
        rows = sorted([m for m in models if m['method'] == method], key=lambda m:m['seed'])
        assert sorted(m['seed'] for m in rows) == [1,2,3]
        epochs = {}
        for epoch in ['100', '200', '300']:
            values = [m['epochs'][epoch]['macro50'] for m in rows]
            count = sum(m['epochs'][epoch]['hard6_successes'] for m in rows)
            epochs[epoch] = dict(mean_macro50=sum(values)/3, per_seed_macro50=values,
                hard6_successes=count, hard6_model_case_episodes=360, hard6_unique_physical_cases=120,
                hard6_percent=100*count/360)
        tasks = {}
        for task in sorted(records[rows[0]['name']]['epochs']['300']['groups']):
            tasks[task] = {e:sum(records[m['name']]['epochs'][e]['groups'][task]['percent'] for m in rows)/3
                           for e in ['100','200','300']}
        methods[method] = dict(epochs=epochs, tasks=tasks,
            gain100_to300_pp=epochs['300']['mean_macro50']-epochs['100']['mean_macro50'])
    result = dict(status='COMPLETE_HOLD_NO_FURTHER_BASELINE_ROUND',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), passed=True,
        primary_recipes=18, trained_seeds_per_method=3, actual_300epoch_trace_files_verified=18000,
        phase_audits_created_here=new_audits, existing_phase_audits_reused=18-new_audits,
        unique_physical_cases_per_model=1000, models=models, methods=methods,
        ranking300=sorted(methods, key=lambda k:methods[k]['epochs']['300']['mean_macro50'], reverse=True),
        sources=dict(summary_path=str(source), summary_sha256=digest(source), final_trace_audits=audits,
                     source_sha256=digest(__file__)),
        protocol='Scratch RGB/actual forward-input/rewardfree on independent LeRobot MetaWorld corpus and fixed trajectory split. '
                 'CommonCEM300x15/top10/H6/execute3/context2/f5/500nativeanysuccess,50tasksx20samephysicalcases/model. '
                 '100+200continuation with one nativeLRrestart,not scratch300singlecosine;capacity/frames/FLOPs differ between methods. '
                 'Inverse targets only actual inputs of completed transitions. Native baseline batchregs retained. '
                 'Not official gatedJEPA-WMs data,public100step-final protocol,canonicalMT50 or universalSOTA claim.',
        uncertainty='Three learned seeds share the same1000physicalcases;model-case outcomes are not independent physical cases. '
                    'Per-seed values and paired changes reported;no causal attribution from aggregate scaling.',
        optional_native_controller='Closed partial by user;690complete audited episodes,11interrupted shards/170unstarted deferred. '
                                   'No complete1000case native model. See separate native-controller/closeout-20261108/summary.json.',
        additional_training_updates=0, neural_models_loaded=0, new_physical_episodes=0,
        no_further_baseline_extension=True, no_automatic_candidate_promotion=True)
    out = ROOT/'results/final'
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'HOLD').write_text('Baseline campaign closed after the existing300epoch recipes; no further baseline branches or extensions.\n')
    print(json.dumps(dict(status=result['status'], ranking300=result['ranking300'],
                          means300={k:v['epochs']['300']['mean_macro50'] for k,v in methods.items()})))


if __name__ == '__main__':
    main()
