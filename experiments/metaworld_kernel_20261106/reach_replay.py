"""CPU-only replay of SAVED controls;privileged geometry is diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def run(a):
    sys.path.insert(0,str(a.source))
    from experiments.metaworld_eval_20261103.environment import make_environment,physics_state,close_environment,expert
    base=Path('/efs/prescottsun/SPT-JEPA');runs=base/'runs';cases=json.loads((runs/'metaworld_eval_20261103/cases-v2/manifest.json').read_text())['rows']
    cases={(r['task'],r['seed']):r for r in cases if r['task']=='reach-v3'and r['seed']in[42,43,1042,1043]}
    methods=[('current40',runs/'metaworld_current_inverse_20261107/training','episodes','eval-step0040000'),
             ('directed20',runs/'metaworld_current_goal_20261107/training','directed-flow','eval-step0020000'),
             ('local-inverse40',runs/'metaworld_current_local_20261107/training','flow','eval-step0040000'),
             ('local-RGB5',runs/'metaworld_intervention_finetune_20261107/training','causal-and-goal','eval-projection-step0005000')]
    a.output.mkdir(parents=True,exist_ok=False);rows=[];experts={}
    for key,case in cases.items():
        cached=np.load(case['path']);env,state,_=make_environment(*key,cached['rand_vec']);policy=expert(base/'sources/lerobot-mw-f5c9bc02',key[0])
        try:
            for step in range(500):
                state,_,terminated,truncated,info=env.step(np.clip(policy.get_action(state),-1,1).astype(np.float32))
                if info['success']or terminated or truncated:break
            assert step+1==case['expert_steps']and bool(info['success'])==case['expert_any_success']
            experts[str(key)]=dict(steps=step+1,success=bool(info['success']),goal_hand_distance=float(np.linalg.norm(state[:3]-env._target_pos)),
                goal_TCP_distance=float(np.linalg.norm(env.tcp_center-env._target_pos)),goal_hand=state[:3].tolist(),target=env._target_pos.tolist())
        finally:close_environment(env)
    for method,path,arm,endpoint in methods:
        for model_seed in[1,2]:
            source=path/f'{arm}-seed{model_seed}'/endpoint/'summary.json';summary=json.loads(source.read_text());assert summary['passed']and summary['model_unchanged']
            for row in summary['rows']:
                key=(row['task'],row['seed'])
                if key not in cases:continue
                case=cases[key];cached=np.load(case['path']);assert hashlib.sha256(Path(row['trace']).read_bytes()).hexdigest()==row['trace_sha256']
                trace=np.load(row['trace']);env,state,_=make_environment(*key,cached['rand_vec']);initial=physics_state(env)
                assert np.array_equal(state,cached['initial_state'])and all(np.array_equal(v,cached['initial_physics_'+k])for k,v in initial.items())
                target=env._target_pos.copy();hand=[];tcp=[];successes=[];radial=[]
                try:
                    for action in trace['actions']:
                        error=target-state[:3];radial.append(float(np.dot(error,action[:3])/max(1e-12,np.linalg.norm(error)*np.linalg.norm(action[:3]))))
                        state,_,_,_,info=env.step(action.astype(np.float32));hand.append(state[:3].copy());tcp.append(env.tcp_center.copy());successes.append(bool(info['success']))
                    assert np.array_equal(successes,trace['successes'])
                finally:close_environment(env)
                hand=np.asarray(hand);tcp=np.asarray(tcp);distance=np.linalg.norm(hand-target,axis=1);tail=trace['actions'][-100:,:3]
                physical=a.output/f'{method}-model{model_seed}-case{key[1]}.npz';np.savez_compressed(physical,hand=hand,tcp=tcp,target=target,distance=distance,action=trace['actions'],radial_cosine=radial)
                value=dict(method=method,model_seed=model_seed,case_seed=key[1],success=bool(any(successes)),steps=len(distance),
                    min_hand_target_distance=float(distance.min()),final_hand_target_distance=float(distance[-1]),final_error_xyz=(target-hand[-1]).tolist(),
                    tail_command_mean=tail.mean(0).tolist(),tail_command_std=tail.std(0).tolist(),last100_radial_cosine=float(np.mean(radial[-100:])),
                    saved_trace_successes_exact=True,geometry_file=str(physical),trace_sha256=row['trace_sha256'])
                rows.append(value);print(json.dumps(value),flush=True)
    result=dict(passed=True,rows=rows,expert_goal_geometry=experts,new_model_updates=0,new_controller_trials=0,renderer_calls=0,
        scope='ONLYCPUphysicsreplayofpreviouslyexecutedcontrols.Model/policyoutputsunchanged.Allprivilegedpositions/targets/statesareDIAGNOSTICONLY;neverlearner/plannerinputornewtrainingdata.'
              'Fourphysicalcasesrepeatedacrosstrainedmodels/methods,notindependentreplicates.ExpertsONLYreconstructexistinggoalphysicalstate,neverteachthemodel.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);run(p.parse_args())
