"""Own-command branching experiments;no expert command is read or replayed.

This corpus is a SEPARATE causal-identification study.It is not automatically
added to original-data learners.Any performance comparison using it must give
the identical corpus and protocol to its baselines.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(native,packed,output):
    output.mkdir(parents=True,exist_ok=False);split=json.loads((packed/'split-manifest.json').read_text());mapping={}
    for bundle in json.loads((packed/'pack-index.json').read_text())['bundles']:
        for phase in bundle['phases']:
            key=int(phase['original_episode']);task=int(phase['task_id']);assert mapping.setdefault(key,task)==task
    training=set(split['train_original_episodes']);rows=[];images=[]
    for shard in range(12):
        directory=native/str(shard);manifest=json.loads((directory/'manifest.json').read_text());assert manifest['passed']
        values=np.load(directory/'pixels.npy',mmap_mode='r')
        for r in manifest['records']:
            if r['split']!='train':continue
            assert r['episode']in training
            for quantile in [.25,.5,.75,1.]:
                frame=int(round((r['length']-1)*quantile));image=np.asarray(values[r['offset']+frame]).copy()
                rows.append(dict(task_id=mapping[r['episode']],episode=r['episode'],frame=frame,sha256=hashlib.sha256(image.tobytes()).hexdigest()));images.append(image)
    assert len(rows)==9000 and len(set(r['episode']for r in rows))==2250
    np.save(output/'pixels.npy',np.stack(images));save(output/'manifest.json',dict(passed=True,rows=rows,task_names=split['task_names'],
        pixels_sha256=digest(output/'pixels.npy'),training_only=True,no_actions_read=True,scope='UniformtemporalquartilesoforiginalTRAINRGBonly,notexpertcommandsoroptimal-actionlabels.'))


def active_contact(env):
    import mujoco
    roots=[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_BODY,n)for n in ['leftclaw','rightclaw']]
    def top(body):
        while env.model.body_parentid[body]:body=int(env.model.body_parentid[body])
        return body
    robot=top(roots[0]);touched=False
    for i,contact in enumerate(env.data.contact):
        for left,right in [(int(contact.geom1),int(contact.geom2)),(int(contact.geom2),int(contact.geom1))]:
            body=int(env.model.geom_bodyid[left]);finger=False
            while body:
                if body in roots:finger=True;break
                body=int(env.model.body_parentid[body])
            if not finger:continue
            body=int(env.model.geom_bodyid[right]);dynamic=False
            if top(body)==robot:continue
            while body:
                if env.model.body_jntnum[body]:dynamic=True;break
                body=int(env.model.body_parentid[body])
            if dynamic:
                force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,i,force);touched|=bool(force[0]>0)
    return touched


def words(rng):
    result=[('zero',np.zeros((5,4),np.float32))]
    for axis,sign in itertools.product(range(4),[-1,1]):
        command=np.zeros(4,np.float32);command[axis]=sign;result.append((f'axis{axis}-sign{sign}',np.tile(command,(5,1))))
    for i,j in itertools.combinations(range(4),2):
        left=np.zeros(4,np.float32);right=left.copy();left[i]=rng.choice([-1,1]);right[j]=rng.choice([-1,1])
        result.extend([(f'pair{i}{j}-forward',np.stack([left]*5+[right]*5)),(f'pair{i}{j}-reverse',np.stack([right]*5+[left]*5)),
            (f'pair{i}{j}-simultaneous',np.tile(left+right,(10,1)))])
    for index in range(2):result.append((f'random{index}',np.repeat(rng.uniform(-1,1,(2,4)).astype(np.float32),5,axis=0)))
    assert len(result)==29;return result


def collect(a):
    import mujoco
    from as_jepa_lewm.factory import build_model
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');a.output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((a.bank/'manifest.json').read_text());assert metadata['passed']and metadata['training_only']and metadata['no_actions_read']
    images=np.load(a.bank/'pixels.npy',mmap_mode='r');cfg=json.loads((a.policy/'config.json').read_text());assert cfg['model_class']=='goal_transport'
    model=build_model(cfg).cuda().eval().requires_grad_(False);checkpoint=a.policy/'model-step0010000/weights.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True));before=model_sha(model)
    excluded=set()
    for case in json.loads(a.eval_manifest.read_text())['rows']:
        vector=np.load(case['path'])['rand_vec'];excluded.add((case['task'],hashlib.sha256(vector.tobytes()).hexdigest()))
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;records=[];qa=[];started=time.time()
    with torch.inference_mode():
        for task_id in a.tasks:
            task=metadata['task_names'][str(task_id)];options=[i for i,r in enumerate(metadata['rows'])if r['task_id']==task_id];assert len(options)==180
            for parent_index in range(a.parents):
                seed=20000+task_id*100+parent_index;rng=np.random.default_rng(seed);choice=int(rng.choice(options));goal=model.encoder(torch.from_numpy(np.asarray(images[choice]).copy()).permute(2,0,1)[None].cuda())
                env,_,vector=make_environment(task,seed);vector_sha=hashlib.sha256(vector.tobytes()).hexdigest();assert(task,vector_sha)not in excluded
                parent=a.output/f'task{task_id:02d}-parent{parent_index}.npz';frames=[resize_tensor([pixels(env)])[0].permute(1,2,0).numpy()];controls=[];current_command=None
                try:
                    for step in range(1,a.native_steps+1):
                        if(step-1)%5==0:
                            current=model.encoder(torch.from_numpy(frames[-1]).permute(2,0,1)[None].cuda())
                            previous=torch.as_tensor(controls[-1],device='cuda')[None]if controls else current.new_zeros(1,4)
                            current_command=model.choose_control(current,goal,current.new_tensor([bool(controls)]),previous)[0][0].cpu().numpy();assert np.isfinite(current_command).all()and abs(current_command).max()<=1
                        env.step(current_command.astype(np.float32));controls.append(current_command.copy());frames.append(resize_tensor([pixels(env)])[0].permute(1,2,0).numpy())
                        if step not in a.context_steps:continue
                        integration=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,integration,spec)
                        prior_obs=env._prev_obs.copy();count=env.curr_path_length;generator_state=copy.deepcopy(env.np_random.bit_generator.state);physical=physics_state(env)
                        context_id=f'task{task_id:02d}-parent{parent_index}-step{step:03d}';contact=active_contact(env);last_images=[]
                        for name,command_word in words(rng):
                            mujoco.mj_setState(env.model,env.data,integration,spec);mujoco.mj_forward(env.model,env.data);mujoco.mj_setState(env.model,env.data,integration,spec)
                            env._prev_obs=prior_obs.copy();env.curr_path_length=count;env.np_random.bit_generator.state=copy.deepcopy(generator_state)
                            assert all(np.array_equal(v,physics_state(env)[k])for k,v in physical.items())
                            actual=resize_tensor([pixels(env)])[0].permute(1,2,0).numpy();gate=rgb_replay_check(frames[-1],actual);assert gate['passed'],gate
                            branch=[]
                            for command in command_word:
                                env.step(command);branch.append(resize_tensor([pixels(env)])[0].permute(1,2,0).numpy())
                            path=a.output/(context_id+'-'+name+'.npz');np.savez_compressed(path,pixels=np.stack(branch),controls=command_word)
                            records.append(dict(parent=str(parent),prefix_start=step-35,prefix_stop=step+1,branch=str(path),branch_sha256=digest(path),
                                group=f'task{task_id:02d}-parent{parent_index}',context=context_id,task_id=task_id,parent_index=parent_index,
                                split='train'if parent_index==0 else'validation',word=name,native_frames=36+len(branch),learner_fields=['pixels','controls']))
                            if len(branch)==10:last_images.append(branch[-1].astype(np.float32))
                        mujoco.mj_setState(env.model,env.data,integration,spec);mujoco.mj_forward(env.model,env.data);mujoco.mj_setState(env.model,env.data,integration,spec)
                        env._prev_obs=prior_obs.copy();env.curr_path_length=count;env.np_random.bit_generator.state=copy.deepcopy(generator_state)
                        assert all(np.array_equal(v,physics_state(env)[k])for k,v in physical.items())
                        qa.append(dict(context=context_id,active_finger_external_contact=contact,
                            ten_step_branch_RGB_variance=float(np.var(np.stack(last_images),axis=0).mean()),state_used_only_for_simulator_restore_and_diagnostic=True))
                        print(json.dumps(dict(context=context_id,branches=29,contact=contact,total_branches=len(records))),flush=True)
                    np.savez_compressed(parent,pixels=np.stack(frames),controls=np.stack(controls))
                    save(a.output/f'task{task_id:02d}-parent{parent_index}.json',dict(task=task,seed=seed,rand_vec=vector.tolist(),rand_vec_sha256=vector_sha,
                        sampled_goal=metadata['rows'][choice],parent_sha256=digest(parent),state_reward_success_labels_stored=False))
                    save(a.output/'progress.json',dict(records=records,diagnostic_only=qa,elapsed_seconds=time.time()-started))
                finally:close_environment(env)
    assert model_sha(model)==before
    save(a.output/'manifest.json',dict(passed=True,records=records,diagnostic_only=qa,policy_unchanged=True,policy_sha256=before,
        policy_checkpoint_sha256=digest(checkpoint),policy_ancestry='OwnscratchRGBforward20k+ownRGBgoal10k;costsharedbyallfuturecomparators.No inverse ancestry.',
        new_training_updates=0,corpus_not_used_in_primary_models=True,
        scope='Ownnonoptimalcommandexperimentsonly.GoalbankoriginalTRAINRGB,neverexpertcommands.Noactionlabels;recordedcommandsarephysicalinputs.'
              'Allbranchesofaparentshareonesplit.Privilegedstateonlysimulatorrestore/diagnostics,notlearnerfields.Evaluationcasesexcluded.'
              'Anyfuturetrainingcomparisonmustuseidenticalaugmentedcorpusforcandidateandbaselines;notpublicSOTA.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['bank','collect'],required=True);p.add_argument('--output',type=Path,required=True)
    for key in ['native','packed','bank','policy','eval-manifest']:p.add_argument('--'+key,type=Path)
    p.add_argument('--tasks',type=int,nargs='+');p.add_argument('--parents',type=int,default=2);p.add_argument('--native-steps',type=int,default=100)
    p.add_argument('--context-steps',type=int,nargs='+',default=[40,60,80,100]);a=p.parse_args()
    if a.mode=='bank':bank(a.native,a.packed,a.output)
    else:collect(a)
