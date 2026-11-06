"""Finite read-only result collection; never launches training or model calls."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def collect(root):
    plan = json.loads((root / 'training-plan.json').read_text())
    runs, groups, identities = [], {}, {}
    for recipe in plan['recipes']:
        run = Path(recipe['run'])
        row = dict(arm=recipe['arm'], seed=recipe['seed'], host=recipe['host'], gpu=recipe['gpu'], run=str(run))
        for name in ['status', 'progress', 'execution-error', 'complete']:
            path = run / (name + '.json')
            if path.exists():
                row[name] = json.loads(path.read_text())
        row['stages'] = []
        for step in recipe.get('stages',plan['stages']):
            objectives=plan.get('objectives',['default']);completed=0
            for objective in objectives:
                path=run/(f'eval-step{step:07d}'if objective=='default'else f'eval-{objective}-step{step:07d}')/'summary.json'
                if not path.exists():continue
                result = json.loads(path.read_text())
                audit = json.loads((run / f'audit-{step}.json').read_text())
                assert result['passed'] and result['model_unchanged'] and audit['passed']
                for case in result['rows']:
                    key = (case['task'], case['seed'])
                    assert identities.setdefault(key, case['case_sha256']) == case['case_sha256']
                    trace = Path(case['trace'])
                    assert hashlib.sha256(trace.read_bytes()).hexdigest() == case['trace_sha256']
                completed+=1
                if step==2:continue
                group=groups.setdefault((recipe['arm'],objective,step),[])
                group.extend(dict(training_seed=recipe['seed'],**case)for case in result['rows'])
            if completed==len(objectives):row['stages'].append(step)
        runs.append(row)
    scores = []
    for (arm, objective,step), rows in sorted(groups.items()):
        tasks = sorted(set(r['task'] for r in rows))
        table = {}
        for task in tasks:
            selected = [r for r in rows if r['task'] == task]
            table[task] = dict(successes=sum(r['success'] for r in selected), episodes=len(selected),
                               percent=100 * sum(r['success'] for r in selected) / len(selected))
        manipulation = [r for r in rows if r['task'] not in ['reach-v3', 'reach-wall-v3']]
        cohorts = {}
        for name, seeds in [('original_seed_ids', [1, 2]), ('independent_seed_ids', [3, 4])]:
            selected = [r for r in manipulation if r['training_seed'] in seeds]
            cohorts[name] = dict(episodes=len(selected), successes=sum(r['success'] for r in selected),
                                  percent=100 * sum(r['success'] for r in selected) / len(selected) if selected else None)
        scores.append(dict(arm=arm,objective=objective, step=step, trained_seeds=sorted(set(r['training_seed'] for r in rows)),
                           episodes=len(rows), task_scores=table, cohorts=cohorts,
                           manipulation_percent=100 * sum(r['success'] for r in manipulation) / len(manipulation),
                           complete_training_seed_cohort=len(set(r['training_seed']for r in rows))==4))
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), runs=runs, scores=scores,
                complete_runs=sum('complete' in r for r in runs), effective_errors=sum('execution-error' in r for r in runs),
                strict_constraint_candidate=True, performance_objective_achieved=False,
                scope='No automatic success/promotion from a screening score. Cases are shared across training seeds;'
                      'not independent physical replications. Old 78.75% reference is action-supervised, different model/budget.')


def main(a):
    deadline = json.loads((a.root / 'training-plan.json').read_text())['deadline_unix']
    while True:
        result = collect(a.root)
        save(a.root / 'results/summary.json', result)
        if not a.watch or result['complete_runs'] + result['effective_errors'] == len(result['runs']) or time.time() >= deadline:
            print(json.dumps(dict(complete_runs=result['complete_runs'], errors=result['effective_errors'], scores=result['scores'])), flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    main(p.parse_args())
