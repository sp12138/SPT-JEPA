"""Promotion must require both learners, gains, and non-regressing ATE."""
from campaign import ARMS, SEEDS, select, read_results


rows=[]
for arm in ARMS:
    for seed in SEEDS:
        for planner in [1,2,3]:
            for clip in range(16):
                fidelity = .2 if arm=='tf-idm' or (arm=='f8' and seed==1) else 0.
                if arm=='variance-joint':fidelity=.3
                ate = 2. if arm=='variance-joint' else 1.
                rows.append(dict(arm=arm,model_seed=seed,planner_seed=planner,clip=clip,
                                 metrics={'frechet/skill':fidelity,'ate/end_distance_xyz':ate}))
r=select(rows)
assert r['selected']==['tf-idm'] and r['continue_arms']==['tf','tf-idm'],r
assert not r['summary']['f8']['qualifies']
assert not r['summary']['variance-joint']['qualifies']
for row in rows:row['metrics']['frechet/skill']=0.
assert select(rows)['continue_arms']==[]
print('PASS: paired gain, two learners, ATE regression, and no-promotion controls')

# Replay the real public-writer -> wrapper-writer lifecycle that interrupted
# collection: an empty file, then a valid public result without stage, then the
# finalized wrapper result. None of the intermediate states is a method score.
import tempfile
import json
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);path=root/'evaluations/one/result.json';path.parent.mkdir(parents=True)
    path.write_text('')
    assert read_results(root)==([],[str(path)])
    value=dict(passed=True,arm='temporal-forecast',model_seed=1,planner_seed=1,clip=0,metrics={})
    path.write_text(json.dumps(value))
    assert read_results(root)==([],[str(path)])
    value.update(arm='tf',stage='screen',step=5840)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value));temporary.replace(path)
    assert read_results(root)==([value],[])
print('PASS: incomplete/public/finalized evaluation-file lifecycle')
