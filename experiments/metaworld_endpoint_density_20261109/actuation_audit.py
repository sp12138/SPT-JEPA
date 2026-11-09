"""Protocol audit ONLY: raw TRAIN proprioception never enters a learner."""
from pathlib import Path
import json
import traceback
import numpy as np
import pyarrow.parquet as pq
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,now
from experiments.metaworld_eval_20261103.environment import make_environment,physics_state,close_environment

ROOT=BASE/'runs/metaworld_endpoint_density_20261109/actuation-audit-v1'


if __name__=='__main__':
    ROOT.mkdir(exist_ok=False)
    save(ROOT/'AUDIT_ONLY.json',dict(utc=now(),model_updates=0,nn_loads=0,training_data_eligible=False,
        scope='RawTRAIN4DstateisreadONLYtoauditactuatorunits/timing. No state/reward/goal/action-policy labelsenterany learner. Notfittedadapterornewalgorithmperformance.'))
    try:
        source=BASE/'data/metaworld-lerobot-b46a1c18';columns=['observation.state','action','frame_index','episode_index','index']
        arrays={name:[] for name in columns}
        for path in sorted((source/'data').rglob('*.parquet')):
            data=pq.ParquetFile(path).read(columns=columns).to_pydict()
            for name in columns:arrays[name].extend(data[name])
        order=np.argsort(arrays['index']);state=np.asarray(arrays['observation.state'],np.float64)[order]
        inputs=np.clip(np.asarray(arrays['action'],np.float64)[order],-1,1)
        episodes=np.asarray(arrays['episode_index'])[order];frames=np.asarray(arrays['frame_index'])[order]
        train=set((np.load(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1/train_episodes.npy')//5).tolist())
        assert len(train)==2250
        index=np.arange(2,len(state)-3);base=np.isin(episodes[index],list(train))&(episodes[index]==episodes[index+1])&(frames[index+1]==frames[index]+1)
        base&=state[index,2]>.1
        delta=state[index+1,:3]-state[index,:3];fits=[]
        for lag in [-2,-1,0,1,2]:
            mask=base&(episodes[index]==episodes[index+lag]);a=inputs[index+lag,:3];y=delta
            matrix=np.linalg.lstsq(a[mask],y[mask],rcond=None)[0]
            residual=y[mask]-a[mask]@matrix
            steady=mask&(np.abs(inputs[index,:3]-inputs[index-1,:3]).max(1)<.05)&(np.abs(inputs[index-1,:3]-inputs[index-2,:3]).max(1)<.05)
            steady_matrix=np.linalg.lstsq(a[steady],y[steady],rcond=None)[0]
            fits.append(dict(lag=lag,pairs=int(mask.sum()),least_squares_gain_matrix=matrix.tolist(),steady_pairs=int(steady.sum()),
                steady_gain_matrix=steady_matrix.tolist(),r2=(1-(residual**2).sum(0)/((y[mask]-y[mask].mean(0))**2).sum(0)).tolist()))
        same=base;grip_delta=state[index+1,3]-state[index,3];middle=(state[index,3]>.1)&(state[index,3]<.9)
        gripper={}
        for sign in [-1,1]:
            select=same&middle&(inputs[index,3]*sign>.5)
            gripper[str(sign)]=dict(pairs=int(select.sum()),mean_width_delta=float(grip_delta[select].mean()) if select.any() else None)
        save(ROOT/'DATA.json',dict(passed=True,utc=now(),raw_rows=len(state),train_episodes=len(train),
            state_min=state.min(0).tolist(),state_max=state.max(0).tolist(),lag_fits=fits,gripper=gripper,source_fps=read(source/'meta/info.json')['fps'],
            caveat='Expertfeedbackcorrelation/inertia/contacts can affectlagfits;maximumR2aloneisNOTproof ofoff-by-one misalignment. StateauditONLY,notNNinput/targetoradapterfitting.'))
        case=next(c for c in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'] if c['task']=='reach-v3')
        cached=np.load(case['path']);assert sha(case['path'])==case['sha256'];trials=[];properties=None
        for axis in range(4):
            for sign in [-1,1]:
                env,initial,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
                assert np.array_equal(initial,cached['initial_state'])
                assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
                command=np.zeros(4,np.float32);command[axis]=sign*(.2 if axis<3 else 1.)
                states=[initial[:4].copy()]
                try:
                    properties=dict(frame_skip=env.frame_skip,dt=env.dt,timestep=env.model.opt.timestep,
                        action_scale=float(env.action_scale),action_space_low=env.action_space.low.tolist(),action_space_high=env.action_space.high.tolist())
                    for _ in range(15):
                        obs,_,_,_,_=env.step(command);states.append(obs[:4].copy())
                finally:close_environment(env)
                values=np.asarray(states);changes=np.diff(values,axis=0)
                trials.append(dict(axis=axis,sign=sign,command=command.tolist(),states=values.tolist(),
                    late_mean_delta=changes[5:].mean(0).tolist(),gain=changes[5:].mean(0).tolist() if axis==3 else (changes[5:].mean(0)/command[axis]).tolist()))
        save(ROOT/'SIMULATOR.json',dict(passed=True,utc=now(),properties=properties,trials=trials,physics_steps=8*15,
            model_updates=0,nn_loads=0,training_data_eligible=False,candidate_performance=False,
            scope='OnecachedReachphysicalstate/8constantactuatorprobes. PrivilegedstateONLYprotocolaudit,nottrainedcontrollerorperformance.'))
        print('Actuation data/simulator audit complete',flush=True)
    except BaseException:
        save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
