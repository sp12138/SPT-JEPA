"""Promotion must require both learners, gains, and non-regressing ATE."""
from campaign import ARMS, SEEDS, select


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
