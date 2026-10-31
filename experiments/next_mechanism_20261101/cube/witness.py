"""Small exact witnesses; no dependencies, labels for learning, or model calls."""
import json
from pathlib import Path

goal=(2.,-1.,.5,3.)
goal_history=[goal]*3
centered=[[x-y for x,y in zip(g,goal)]for g in goal_history]
assert all(v==0 for row in centered for v in row)
# A goal-tube projector inferred from these observed differences has rank zero.
# Its orthogonal complement is the identity: no nuisance direction is learned.

# Already squared errors in two blocks: hand, task object.
# Mean selects the object-correct candidate; max selects the incorrect one.
errors=[[4.,0.],[2.5,2.5]]
means=[sum(x)/len(x)for x in errors];maxima=[max(x)for x in errors]
assert means[0]<means[1] and maxima[0]>maxima[1]

# Same fully observed world law and same exact goal can support different
# externally specified success criteria. No reward-free observational statistic
# distinguishes which component the evaluator chooses to ignore.
goal2=(0,0);candidate_a=(0,1);candidate_b=(1,0)
hand_success=[c[0]==goal2[0]for c in [candidate_a,candidate_b]]
object_success=[c[1]==goal2[1]for c in [candidate_a,candidate_b]]
assert hand_success==[True,False] and object_success==[False,True]

result=dict(passed=True,repeated_goal_history_centered=centered,goal_tube_rank=0,
    mean_vs_max_counterexample=dict(squared_block_errors=errors,mean=means,max=maxima,mean_selects=0,max_selects=1,object_errors=[0.,2.5]),
    nonidentifiability=dict(goal=goal2,candidates=[candidate_a,candidate_b],hand_success=hand_success,object_success=object_success),
    scope='No universal guarantee for discovering evaluator-specific nuisance equivalence from unchanged reward-free observations. This does not preclude exact-goal reaching or useful task-relevant inductive biases.')
Path(__file__).with_name('witness-results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
