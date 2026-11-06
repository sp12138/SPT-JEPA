"""CPU-only order-of-interventions audit on OUR forward-model trajectories.

Privileged simulation state is used only for this diagnostic,never a learner,
planner,training target,or data proposal.No inverse-policy trajectory is read.
No rendering or neural inference is performed,so no GPU context is needed.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.metaworld_eval_20261103.environment import make_environment,physics_state,close_environment


def run(a):
    import mujoco
    a.output.mkdir(parents=True,exist_ok=False);manifest=json.loads(a.manifest.read_text());rows=[]
    tasks=['bin-picking-v3','pick-place-v3','push-v3','drawer-open-v3','door-open-v3','peg-insert-side-v3']
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    def snapshot(env,observation):
        value=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,value,spec)
        return dict(integration=value,observation=observation.copy(),previous=env._prev_obs.copy(),
            count=env.curr_path_length,rng=copy.deepcopy(env.np_random.bit_generator.state))
    def restore(env,saved):
        mujoco.mj_setState(env.model,env.data,saved['integration'],spec);mujoco.mj_forward(env.model,env.data)
        env._prev_obs=saved['previous'].copy();env.curr_path_length=saved['count'];env.np_random.bit_generator.state=copy.deepcopy(saved['rng'])
    def outcome(env,saved,commands):
        restore(env,saved)
        for command in commands:observation,*_=env.step(command)
        return observation.copy(),physics_state(env)
    for arm in ['free','linear','signed','stencil']:
        run=a.parent/'training'/f'{arm}-seed2';assert json.loads((run/'config.json').read_text())['model_class']=='causal_transport'
        assert json.loads((run/'ORIGIN.json').read_text())['all_weights_random']
        evaluation=run/'eval-projection-step0020000'
        for task,seed in itertools.product(tasks,[42,1042]):
            progress=json.loads((evaluation/'progress.json').read_text());trace=next(r for r in progress['rows']if r['task']==task and r['seed']==seed)
            path=Path(trace['trace']);assert hashlib.sha256(path.read_bytes()).hexdigest()==trace['trace_sha256']
            actions=np.load(path)['actions'];case=next(r for r in manifest['rows']if r['task']==task and r['seed']==seed)
            assert trace['case_sha256']==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(task,seed,cached['rand_vec'])
            try:
                assert np.array_equal(state,cached['initial_state'])
                assert all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
                initial=snapshot(env,state);nearest=initial;distance=float(np.linalg.norm(state[:3]-state[4:7]));closest_step=0
                for index,command in enumerate(actions):
                    state,*_=env.step(command);d=float(np.linalg.norm(state[:3]-state[4:7]))
                    if d<distance:distance=d;nearest=snapshot(env,state);closest_step=index+1
                stages={}
                for label,saved in [('initial',initial),('own_nearest',nearest)]:
                    # Restore determinism includes hidden motor/integration state.
                    zero=np.zeros((2,4),np.float32);first,first_physics=outcome(env,saved,zero);second,second_physics=outcome(env,saved,zero)
                    assert np.array_equal(first,second)and all(np.array_equal(v,second_physics[k])for k,v in first_physics.items())
                    pairs=[];base=saved['observation']
                    for i,j in itertools.combinations(range(4),2):
                        for si,sj in itertools.product([-1,1],repeat=2):
                            left=np.zeros(4,np.float32);right=left.copy();left[i]=si;right[j]=sj
                            ab,_=outcome(env,saved,[left]*5+[right]*5);ba,_=outcome(env,saved,[right]*5+[left]*5)
                            pairs.append(dict(axes=[i,j],signs=[si,sj],object_commutator=float(np.linalg.norm(ab[4:7]-ba[4:7])),
                                hand_commutator=float(np.linalg.norm(ab[:3]-ba[:3])),
                                object_delta_ab=(ab[4:7]-base[4:7]).tolist(),object_delta_ba=(ba[4:7]-base[4:7]).tolist()))
                    stages[label]=dict(object_position=base[4:7].tolist(),hand_position=base[:3].tolist(),pairs=pairs)
                rows.append(dict(arm=arm,seed=seed,task=task,source_trace_sha256=trace['trace_sha256'],closest_own_step=closest_step,
                    closest_hand_object_distance=distance,stages=stages))
                (a.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
                print(json.dumps(dict(arm=arm,task=task,seed=seed,closest_distance=distance,step=closest_step)),flush=True)
            finally:close_environment(env)
    report=dict(passed=True,rows=rows,model_updates=0,neural_models_loaded=0,render_calls=0,
        scope='State-only diagnostic of signed-axis intervention ordering on initial and nearest-object states reached by our own forward20k controllers.'
              'State selected for diagnosis only,not model/planner input or new trainingdata.No demonstration/future-action reference or inverse model.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['parent','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
