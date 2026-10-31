"""One approved9-restoration Cube reset audit, using native World.evaluate.

No learned model, optimizer, CEM, protocol patch or task-success experiment.
The fixed one-step command is only a diagnostic of reset-state persistence.
"""
from __future__ import annotations
import argparse,hashlib,inspect,json,random,sys,time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA')
CASES=[3,28,35]
ACTION=[.3,-.2,.1,.2,-.1]
FIELDS=['qpos','qvel','act','qacc','qacc_warmstart','ctrl','act_dot','qfrc_applied','xfrc_applied',
        'mocap_pos','mocap_quat','userdata','plugin_state','eq_active','sensordata',
        'qfrc_actuator','qfrc_constraint','cfrc_ext','xpos','xquat','site_xpos','site_xmat']
MODEL_FIELDS=['body_mass','body_inertia','geom_friction','geom_size','dof_damping','actuator_gainprm',
              'actuator_biasprm','actuator_ctrlrange','qpos0','cam_pos','cam_quat','light_pos','geom_rgba']

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def difference(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    delta=a.astype(np.float64)-b.astype(np.float64)
    return dict(exact_equal=bool(np.array_equal(a,b)),different_elements=int(np.count_nonzero(a!=b)),
                max_abs=float(np.abs(delta).max())if delta.size else 0.,l2=float(np.linalg.norm(delta)))

def compare(records):
    # All three repeat pairs are reported; there is no result-based selection.
    result=[]
    for case in CASES:
        for a,b in [(0,1),(0,2),(1,2)]:
            left=records[a,case];right=records[b,case];stages={}
            for stage in ['before_dataset_set_state','after_dataset_set_state','after_dataset_set_target','before_native_run','after_one_step']:
                x,y=left[stage],right[stage];assert set(x)==set(y)
                stages[stage]={k:difference(x[k],y[k])for k in x}
            result.append(dict(case_index=case,repeats=[a,b],stages=stages))
    return result

@contextmanager
def native_counts(CubeEnv):
    """Count the real native constructor/reset settling steps; delegate unchanged."""
    reset,step=CubeEnv.reset,CubeEnv.step;counts={};depth={}
    def row(env):return counts.setdefault(id(env),dict(reset_calls=0,steps_inside_reset=0,steps_outside_reset=0))
    def wrapped_reset(env,*args,**kwargs):
        row(env)['reset_calls']+=1;depth[id(env)]=depth.get(id(env),0)+1
        try:return reset(env,*args,**kwargs)
        finally:depth[id(env)]-=1
    def wrapped_step(env,action,*args,**kwargs):
        row(env)['steps_inside_reset'if depth.get(id(env),0)else'steps_outside_reset']+=1
        return step(env,action,*args,**kwargs)
    try:
        CubeEnv.reset=wrapped_reset;CubeEnv.step=wrapped_step;yield counts
    finally:CubeEnv.reset=reset;CubeEnv.step=step

def cpu_contracts():
    z=np.zeros(4,dtype=np.float64);one=z.copy();one[2]=.25
    assert difference(z,z)==dict(exact_equal=True,different_elements=0,max_abs=0.,l2=0.)
    assert difference(z,one)==dict(exact_equal=False,different_elements=1,max_abs=.25,l2=.25)
    class Fake:
        def reset(self,*,seed=None):self.step(seed);return seed
        def step(self,action):return action
    reset,step=Fake.reset,Fake.step;obj=Fake()
    with native_counts(Fake)as counts:
        assert obj.reset(seed=17)==17 and obj.step(3)==3
        assert counts[id(obj)]==dict(reset_calls=1,steps_inside_reset=1,steps_outside_reset=1)
    assert Fake.reset is reset and Fake.step is step
    return dict(exact_difference_oracle=True,delegated_native_return_values_preserved=True,counting_hooks_restored=True,
                learned_models=0,real_environments=0,source_sha256=sha(__file__))

def prepare(args):
    original=read(args.reference_spec);manifest=read(original['manifest'])
    assert original['task']=='cube'and original['case_set']=='primary'and len(manifest['pairs'])==50
    assert manifest['campaign_split']=='validation'
    source=original['source'];files={}
    for mk,sk in [('manifest','source'),('runtime_manifest','runtime_source')]:
        record=read(source[mk])
        for rel,digest in record['files'].items():
            path=Path(source[sk])/rel;assert sha(path)==digest,path
        files[source[mk]]=sha(source[mk])
    old=read(original['reference_metrics']);assert old['manifest_sha256']==sha(original['manifest'])
    for path,digest in original['immutable_inputs'].items():
        if Path(path).name=='weights.pt':continue # No models/weights will be loaded in this audit.
        assert sha(path)==digest;files[path]=digest
    native=B/'sources/stable-worldmodel/stable_worldmodel';site=Path('/efs/prescottsun/project_jepa/envs/lewm/lib/python3.12/site-packages')
    critical=[native/'world/world.py',native/'world/env_pool.py',native/'envs/ogbench/cube_env.py',native/'wrapper/default.py',
        site/'ogbench/manipspace/envs/manipspace_env.py',site/'ogbench/manipspace/envs/env.py',site/'ogbench/manipspace/controllers/diff_ik.py']
    files.update({str(p):sha(p)for p in critical})
    plan=dict(source=source,reference_spec=str(args.reference_spec),reference_spec_sha256=sha(args.reference_spec),
        native_evaluator=original['native_evaluator'],manifest=original['manifest'],manifest_sha256=sha(original['manifest']),
        data=original['data'],cases=[dict(index=i,pair=manifest['pairs'][i])for i in CASES],repeats=3,
        action=ACTION,eval_budget_diagnostic=1,original_protocol=manifest['protocol'],original_policy_seed=manifest['policy_seed'],
        output=str(args.output),source_files=files,helper_sha256=sha(__file__),cpu_contracts=cpu_contracts(),
        scope='Fixed primary cases3,28,35,3fresh Worlds with3envs each, exactly9 dataset restorations and9 explicit bounded one-step probes. Original World.evaluate dataset reset/callable path; native constructor initialization and random settling remain and are counted. No model, training, CEM, newP0 or reset correction. Physical labels only diagnostics. Root must freeze and check actualdev2g5free before this one audit.')
    write(args.plan,plan);print(json.dumps(dict(plan=str(args.plan),cases=CASES,repeats=3,passed=True)))

def snapshot(env,mujoco):
    model,data=env._model,env._data;out={}
    state_type=mujoco.mjtState.mjSTATE_INTEGRATION
    full=np.empty(mujoco.mj_stateSize(model,state_type),np.float64);mujoco.mj_getState(model,data,full,state_type)
    out['integration']=full
    for key in FIELDS:
        if hasattr(data,key):out[key]=np.asarray(getattr(data,key)).copy()
    out['time']=np.asarray(data.time,np.float64);out['ncon']=np.asarray(data.ncon,np.int64)
    for key in ['_prev_qpos','_prev_qvel']:
        if hasattr(env,key):out[key]=np.asarray(getattr(env,key)).copy()
    for key in ['_success','_clear_lewm_hold_count','cur_task_id']:
        value=getattr(env,key,None)
        if value is not None:out[key]=np.asarray(value)
    if hasattr(env,'_target_effector_pose'):
        out['target_effector_matrix']=np.asarray(env._target_effector_pose.as_matrix()).copy()
    ik=getattr(env,'_ik',None)
    if ik is not None:
        for key in ['qpos','qvel','qacc_warmstart','ctrl','time']:
            out['ik_data_'+key]=np.asarray(getattr(ik._data,key)).copy()
        # Scratch buffers are named as such; differences alone imply no functional use.
        for key in ['_err','_site_quat','_site_quat_inv','_err_quat','_jac']:
            out['ik_scratch'+key]=np.asarray(getattr(ik,key)).copy()
    for key in MODEL_FIELDS:
        if hasattr(model,key):out['model_'+key]=np.asarray(getattr(model,key)).copy()
    out['model_timestep']=np.asarray(model.opt.timestep,np.float64)
    out['model_gravity']=np.asarray(model.opt.gravity).copy()
    assert all(np.isfinite(x).all()for x in out.values()),'Nonfinite recorded state'
    return out

def run(args):
    plan=read(args.plan);assert plan['helper_sha256']==sha(__file__)
    assert [x['index']for x in plan['cases']]==CASES and plan['repeats']==3 and plan['action']==ACTION
    for path,digest in plan['source_files'].items():assert sha(path)==digest,path
    for mk,sk in [('manifest','source'),('runtime_manifest','runtime_source')]:
        for rel,digest in read(plan['source'][mk])['files'].items():assert sha(Path(plan['source'][sk])/rel)==digest
    sys.path[:0]=plan['source']['pythonpath'].split(':')
    import torch,mujoco,stable_worldmodel as swm
    from hydra import initialize_config_dir,compose
    from omegaconf import OmegaConf,open_dict
    from clear_lewm.protocols import protocol_from_dict
    from clear_lewm.runner import _install_task_success
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    assert inspect.getfile(CubeEnv)==str(B/'sources/stable-worldmodel/stable_worldmodel/envs/ogbench/cube_env.py')
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    write(out/'input-provenance.json',dict(plan=plan,plan_sha256=sha(args.plan),script=str(Path(__file__).resolve()),
        script_sha256=sha(__file__),mujoco_version=mujoco.__version__,numpy_version=np.__version__,started_at=time.time()))
    manifest=read(plan['manifest']);card=read(Path(plan['data'])/'dataset_card.json');spec=protocol_from_dict(manifest['protocol'])
    dataset=swm.data.load_dataset(card['h5_path'],keys_to_cache=['action'])
    pairs=[x['pair']for x in plan['cases']];records={};run_metadata=[];started=time.time()
    class FixedAction(swm.policy.BasePolicy):
        def get_action(self,info,**kwargs):
            command=np.repeat(np.asarray(ACTION,np.float32)[None],self.env.num_envs,axis=0)
            assert np.all(command>=self.env.single_action_space.low)and np.all(command<=self.env.single_action_space.high)
            return command
    with native_counts(CubeEnv)as counters:
        for repeat in range(3):
            counters.clear()  # Per-fresh-World counts; copied into metadata before the next trial.
            # Same global seed as original evaluator; Gym's independent native reset behavior remains untouched.
            seed=int(manifest['policy_seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            with initialize_config_dir(config_dir=str(B/'sources/stable-worldmodel/scripts/plan/config'),version_base=None):cfg=compose(config_name='cube')
            with open_dict(cfg):
                cfg.world.num_envs=3;cfg.world.max_episode_steps=2*spec.eval_budget
                cfg.plan_config.history_len=3;cfg.solver.batch_size=1
            world=swm.World(**cfg.world,image_shape=(224,224));_install_task_success(world,'cube',spec);world.set_policy(FixedAction())
            envs=[x.unwrapped for x in world.envs.envs]
            for case,env in zip(CASES,envs):
                records[repeat,case]={}
                original_state=env.set_state;original_target=env.set_target_pos
                def state_hook(self,*a,_original=original_state,_case=case,**kw):
                    store=records[repeat,_case];store['before_dataset_set_state']=snapshot(self,mujoco)
                    result=_original(*a,**kw);store['after_dataset_set_state']=snapshot(self,mujoco);return result
                def target_hook(self,*a,_original=original_target,_case=case,**kw):
                    result=_original(*a,**kw);records[repeat,_case]['after_dataset_set_target']=snapshot(self,mujoco);return result
                env.set_state=MethodType(state_hook,env);env.set_target_pos=MethodType(target_hook,env)
            original_run=world._run;pixel_inputs={}
            def run_hook(*a,**kw):
                assert kw.get('max_steps')==1 and kw.get('mode')=='wait'
                for rank,(case,env)in enumerate(zip(CASES,envs)):
                    records[repeat,case]['before_native_run']=snapshot(env,mujoco)
                    pixel_inputs[case]=dict(initial=array_sha(world.infos['pixels'][rank]),goal=array_sha(world.infos['goal'][rank]))
                return original_run(*a,**kw)
            world._run=run_hook
            try:
                metrics=world.evaluate(dataset=dataset,start_steps=[p['start_step']for p in pairs],goal_offset=spec.goal_offset,
                    eval_budget=1,episodes_idx=[p['episode_id']for p in pairs],callables=OmegaConf.to_container(cfg.eval.callables,resolve=True),video=None)
                for rank,(case,env)in enumerate(zip(CASES,envs)):
                    records[repeat,case]['after_one_step']=snapshot(env,mujoco)
                    # Dataset restore must recover the exact stored qpos/qvel.
                    chunks=dataset.load_chunk(np.asarray([pairs[rank]['episode_id']]),np.asarray([pairs[rank]['start_step']]),np.asarray([pairs[rank]['start_step']+1]))
                    for key in ['qpos','qvel']:
                        expected=np.asarray(chunks[0][key][0]);np.testing.assert_array_equal(records[repeat,case]['after_dataset_set_state'][key],expected)
                run_metadata.append(dict(repeat=repeat,inputs=pixel_inputs,next_pixels={case:array_sha(world.infos['pixels'][rank])for rank,case in enumerate(CASES)},
                    actual_native_counts={case:dict(counters[id(env)])for case,env in zip(CASES,envs)},
                    env_np_random_seeds={case:int(getattr(env,'_np_random_seed',-1))for case,env in zip(CASES,envs)},
                    native_metric_diagnostic_only={k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in metrics.items()}))
            finally:world.close()
            print(json.dumps(dict(repeat=repeat,recorded_cases=CASES)),flush=True)
    assert len(records)==9
    assert sum(row['actual_native_counts'][case]['steps_outside_reset']for row in run_metadata for case in CASES)==9
    arrays={f'r{repeat}_case{case}_{phase}_{key}':value for(repeat,case),stages in records.items()for phase,fields in stages.items()for key,value in fields.items()}
    np.savez_compressed(out/'states.npz',**arrays)
    comparisons=compare(records)
    summary=[]
    for row in comparisons:
        stages=row['stages'];after=stages['before_native_run'];step=stages['after_one_step']
        summary.append(dict(case_index=row['case_index'],repeats=row['repeats'],restored_qpos_equal=after['qpos']['exact_equal'],restored_qvel_equal=after['qvel']['exact_equal'],
            post_restore_different_fields=[k for k,v in after.items()if not v['exact_equal']],
            post_restore_integration=after['integration'],post_step_qpos=step['qpos'],post_step_qvel=step['qvel'],post_step_integration=step['integration']))
    result=dict(passed=True,plan_sha256=sha(args.plan),helper_sha256=sha(__file__),cases=plan['cases'],repeats=3,action=ACTION,
        source=plan['source'],metadata=run_metadata,comparison_summary=summary,comparisons=comparisons,
        npz_sha256=sha(out/'states.npz'),seconds=time.time()-started,learned_models_loaded=0,training_steps=0,
        explicit_dataset_restorations=9,explicit_bounded_probe_steps=9,actual_total_native_counts={k:sum(x['actual_native_counts'][case][k]for x in run_metadata for case in CASES)for k in ['reset_calls','steps_inside_reset','steps_outside_reset']},
        limitations='At most3cases×3freshWorld repetitions. No task-performance or thresholded chance claim. Equal input image/qpos/qvel does not assert equal full simulator state. Hidden-state variation plus different fixed one-step outcomes establishes a reset reproducibility issue for these samples, not which residual field causes it or its contribution to fullP0. IK scratch differences alone need not affect computation. No reset intervention or protocol correction was tested.')
    write(out/'results.json',result)
    print(json.dumps(dict(output=str(out),passed=True,counts=result['actual_total_native_counts'],comparisons=summary)),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','contracts'])
    p.add_argument('--reference-spec',default=str(B/'runs/action_projection_evaluation_20261031/specs/original-primary-cube-native-cal-seed1.json'))
    p.add_argument('--plan',default=str(B/'runs/cube_reset_state_audit_20261031/plan.json'))
    p.add_argument('--output',default=str(B/'reports/cube-reset-state-20261031/diagnostic-v1'))
    a=p.parse_args()
    if a.phase=='contracts':print(json.dumps(cpu_contracts()))
    elif a.phase=='prepare':prepare(a)
    else:run(a)

if __name__=='__main__':main()
