"""Read-only dependency audit for intentionally inconsistent RGB goals.

This is not an evaluation of reaching the substituted goal. Physical task goals
remain unchanged. Only the first action has identical current context/RNG.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA/runs')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(a):
    plan=json.loads((a.root/'plan.json').read_text())
    while True:
        path=a.root/'results/summary.json'
        summary=json.loads(path.read_text())if path.exists()else dict(complete=0,errors=0)
        if summary['complete']+summary['errors']==len(plan['recipes']):break
        assert a.wait and time.time()<plan['deadline_unix'],'Diagnostic not complete'
        time.sleep(10)
    assert not summary['errors'],'Retain execution failures; do not infer an algorithm outcome'
    manifest=json.loads(Path(plan['manifest']).read_text());cases={(r['task'],r['seed']):r for r in manifest['rows']}
    for c in cases.values():
        assert digest(c['path'])==c['sha256']and digest(c['physical_parent_path'])==c['physical_parent_sha256']
        with zipfile.ZipFile(c['path'])as x,zipfile.ZipFile(c['physical_parent_path'])as y:
            assert x.namelist()==y.namelist()
            assert all(x.read(n)==y.read(n)for n in x.namelist()if n!='goal.npy')
    paths={
        'flow-world':'metaworld_current_control_full_20261107/world-native4-seed{seed}/pilot/summary.json',
        'flow-held':'metaworld_one_native_20261107/training/all-executed-seed{seed}/eval-step0040000/summary.json',
        'macro20-best':'metaworld_current_goal_20261107/training/directed-flow-seed{seed}/eval-step0040000/summary.json',
        'mse-world':'metaworld_native_clock_controls_20261107/mse-world-native4-seed{seed}/pilot/summary.json'}
    groups={};evidence=[]
    for recipe in plan['recipes']:
        alt_path=Path(recipe['output'])/'pilot/summary.json';ref_path=BASE/paths[recipe['arm']].format(seed=recipe['seed'])
        alt=json.loads(alt_path.read_text());ref=json.loads(ref_path.read_text())
        assert alt['passed']and ref['passed']and alt['model_unchanged']and ref['model_unchanged']
        assert alt['model_sha256']==ref['model_sha256']
        if recipe['kind']=='world':assert alt['world_sha256']==ref['world_sha256']
        original={(r['task'],r['seed']):r for r in ref['rows']}
        for row in alt['rows']:
            key=(row['task'],row['seed']);c=cases[key];r=original[key]
            assert row['case_sha256']==c['sha256']and r['case_sha256']==c['physical_parent_sha256']
            assert digest(row['trace'])==row['trace_sha256']and digest(r['trace'])==r['trace_sha256']
            x=np.load(row['trace'])['actions'];y=np.load(r['trace'])['actions'];delta=x[0]-y[0]
            groups.setdefault(recipe['arm'],[]).append(dict(task=key[0],case_seed=key[1],actor_seed=recipe['seed'],
                original_task_success_correct_goal=bool(r['success']),original_task_success_substituted_goal=bool(row['success']),
                first_native_xyz_l2=float(np.linalg.norm(delta[:3])),first_native_grip_abs=float(abs(delta[3])),first_native_max_abs=float(abs(delta).max()),
                case_sha256=c['sha256'],physical_parent_sha256=c['physical_parent_sha256'],proper_trace_sha256=r['trace_sha256'],substituted_trace_sha256=row['trace_sha256']))
        evidence.append(dict(arm=recipe['arm'],seed=recipe['seed'],correct_summary=str(ref_path),correct_sha256=digest(ref_path),substituted_summary=str(alt_path),substituted_sha256=digest(alt_path)))
    scores={}
    for arm,rows in groups.items():
        assert len(rows)==32
        scores[arm]=dict(episodes=len(rows),physical_cases=16,trained_actors=2,
            original_task_successes_correct_goal=sum(r['original_task_success_correct_goal']for r in rows),
            original_task_successes_substituted_goal=sum(r['original_task_success_substituted_goal']for r in rows),
            mean_first_native_xyz_l2=float(np.mean([r['first_native_xyz_l2']for r in rows])),
            mean_first_native_grip_abs=float(np.mean([r['first_native_grip_abs']for r in rows])),
            first_native_equal_within_1e5=sum(r['first_native_max_abs']<=1e-5 for r in rows),rows=rows)
    result=dict(passed=True,scores=scores,evidence=evidence,new_updates=0,new_simulator_runs=0,
        scope='GOALINPUTDEPENDENCYONLY. Allnon-goalNPZmembersbyteidentical,NNs/RNG/physicaltaskunchanged,onlygoalRGBsubstitutedfromnextSAME-taskcase.'
              'Original-task success under inconsistent instructions is NOT benchmark performance or success at the new goal. '
              'First-native command differences isolate goal-input influence at identical current context. '
              'Similar first actions do NOT prove goal ignorance (early prefixes can legitimately agree); later trajectories also change state. '
              '16sharedphysicalcases crossed2actors,not32independentphysicalcases.No optimal-action/future-command labels or newNN/simulatorruns in this audit.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in s.items()if n!='rows'}for k,s in scores.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait',action='store_true');run(p.parse_args())
