"""One finite completion collector for the prescribed jobs; never launches work."""
import argparse
from collections import defaultdict
import datetime
import json
from pathlib import Path
import time

from experiments.metaworld_eval_20261103.worker import digest, read, save


def main(root):
    root=Path(root);dispatch=read(root/'dispatch.json');jobs=read(root/'evaluation-plan.json')['jobs']
    recipes=read(root/'plan.json')['recipes'];deadline=dispatch['deadline_unix']
    while time.time()<deadline:
        training_disposed=all((Path(r['run'])/'complete.json').exists() or (Path(r['run'])/'worker-failed.json').exists() for r in recipes)
        for job in jobs:
            work=Path(job['work']);run=Path(job['run'])
            if (work/'complete.json').exists() or (work/'failed.json').exists() or (work/'blocked.json').exists():continue
            audit=run/f"audit-step{job['step']}.json"
            if (run/'worker-failed.json').exists() and (not audit.exists() or not read(audit)['passed']):
                save(work/'blocked.json',dict(executed=False,reason='Required native training stage failed',
                                             training_failure=str(run/'worker-failed.json')))
        disposed=all(any((Path(job['work'])/name).exists() for name in ['complete.json','failed.json','blocked.json']) for job in jobs)
        if training_disposed and disposed:break
        if (root/'STOP').exists():break
        time.sleep(20)
    counts=defaultdict(int);outcomes=[];observations=[];archive_checks=[]
    for recipe in recipes:
        run=Path(recipe['run']);state='complete' if (run/'complete.json').exists() else 'failed' if (run/'worker-failed.json').exists() else 'unfinished'
        counts['training_'+state]+=1
        for path in sorted(run.glob('archive-step*/manifest.json')):
            manifest=read(path);checks={name:(path.parent/name).exists() and digest(path.parent/name)==sha for name,sha in manifest['files'].items()}
            archive_checks.append(dict(path=str(path),passed=bool(checks) and all(checks.values()),checks=checks))
    for job in jobs:
        work=Path(job['work']);state=next((name for name in ['complete','failed','blocked'] if (work/(name+'.json')).exists()),'unfinished')
        counts['evaluation_'+state]+=1;outcomes.append(dict(**job,state=state))
        if state!='complete':continue
        metric=read(work/'output/metrics.json')
        assert digest(work/'output/metrics.json')==read(work/'complete.json')['metrics_sha256']
        assert metric['passed'] and metric['model_state_before']==metric['model_state_after']
        for episode in metric['episodes']:observations.append(dict(arm=job['arm'],train_seed=job['seed'],step=job['step'],bank=job['bank'],**episode))
    grouped=defaultdict(list)
    for row in observations:grouped[(row['step'],row['arm'],row['bank'],row['task'])].append(row)
    groups=[]
    for key,rows in sorted(grouped.items()):
        seeds=sorted({r['train_seed'] for r in rows});valid=[r for r in rows if r['expert_goal_success']]
        groups.append(dict(step=key[0],arm=key[1],bank=key[2],task=key[3],training_seeds=seeds,
                           matched_both_training_seeds=seeds==[1,2],episodes=len(rows),
                           success_rate=100*sum(r['success'] for r in rows)/len(rows),
                           goal_valid_episodes=len(valid),goal_valid_success_rate=100*sum(r['success'] for r in valid)/len(valid) if valid else None,
                           goal_construction_failures=len(rows)-len(valid)))
    complete=(counts['training_complete']==8 and counts['evaluation_complete']==192
              and len(archive_checks)==32 and all(c['passed'] for c in archive_checks))
    summary=dict(passed=complete,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=dict(counts),
                 groups=groups,observations=observations,outcomes=outcomes,archives=archive_checks,
                 deadline_utc=dispatch['deadline_utc'],
                 scope='Prespecified independent LeRobot RGB image-goal extension on8tasks,2training seeds,500-native-step any-success. '
                       'All endpoints,cases,failed goal constructions and negative results retained. '
                       'Not canonicalMT50,officialJEPA-WMs reproduction,convergence or equalFLOPs. No automatic continuation.')
    save(root/'results/summary.json',summary)
    save(root/'HOLD',dict(closed=True,passed=complete,reason='Finite queue ended; no next round',summary=str(root/'results/summary.json')))
    print(json.dumps(dict(passed=complete,counts=dict(counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    args=parser.parse_args();main(Path(args.root))
