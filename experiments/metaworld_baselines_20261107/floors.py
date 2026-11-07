"""Goal-blind physical controls to establish the task-wise success floor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_baselines_20261107')


def main(shard):
    from experiments.metaworld_eval_20261103.environment import make_environment,physics_state,close_environment
    from experiments.metaworld_baselines_20261107.reference import save
    manifest=json.loads(Path('/efs/prescottsun/SPT-JEPA/runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json').read_text())
    card=json.loads(Path('/efs/prescottsun/SPT-JEPA/data/metaworld-lerobot-b46a1c18-packed-v1/dataset_card.json').read_text())
    mean=np.clip(np.asarray(card['action_mean'],np.float32).reshape(5,4),-1,1)
    out=ROOT/'floors'/f'shard{shard:02}';out.mkdir(parents=True,exist_ok=False)
    rows=[];started=time.monotonic()
    for case in manifest['rows'][shard::16]:
        assert hashlib.sha256(Path(case['path']).read_bytes()).hexdigest()==case['sha256']
        cached=np.load(case['path'])
        for index,kind in enumerate(['zero','uniform','train-mean']):
            env,initial,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(initial,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
            rng=np.random.default_rng(20261107+case['seed']*1009+index)
            actions=[];successes=[]
            try:
                for step in range(500):
                    if kind=='zero':action=np.zeros(4,np.float32)
                    elif kind=='uniform':action=rng.uniform(-1,1,4).astype(np.float32)
                    else:action=mean[step%5].copy()
                    _,_,terminated,truncated,info=env.step(action)
                    actions.append(action);successes.append(bool(info['success']))
                    if successes[-1] or terminated or truncated:break
            finally:close_environment(env)
            trace=out/f"{case['task']}-seed{case['seed']}-{kind}.npz"
            np.savez_compressed(trace,actions=np.asarray(actions),successes=np.asarray(successes))
            row=dict(task=case['task'],seed=case['seed'],policy=kind,success=any(successes),
                first100_any_success=any(successes[:100]),steps=len(successes),case_sha256=case['sha256'],
                expert_goal_success=case['expert_any_success'],trace=str(trace),trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
    save(out/'summary.json',dict(passed=True,rows=rows,episodes=len(rows),seconds=time.monotonic()-started,
        model_updates=0,policy_inputs=[],scope='Goal-blind controls share exact physical cases and500native any-success. '
            'The train-mean policy uses only train-set global action statistics. '
            'No taskID/state/goal/reward/success is passed to a policy; success is evaluator termination only.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();main(a.shard)
