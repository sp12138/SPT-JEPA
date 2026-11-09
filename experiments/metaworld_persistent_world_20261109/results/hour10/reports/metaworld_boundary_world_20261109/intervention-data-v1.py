"""Fresh ordinary transition logs for causal-coverage diagnostics.

Scripted collection reaches a fixed, presampled time, without success filtering.
Subsequent controls are our independent Gaussian draws, not demonstrations.
No action is a learning target. Desired-goal/evaluation/audit banks are excluded.
This ADDS a data budget; resulting scores cannot claim an equal-data baseline win.
"""
import argparse
import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.campaign import command
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
from experiments.metaworld_horizon_match_20261109.bank import capture,restore,mjstate,byte
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,expert,configuration,rgb_replay_check,close_environment

ROOT=BASE/'runs/metaworld_interventions_20261109'
LEROBOT=BASE/'sources/lerobot-mw-f5c9bc02'


def vector_hash(value):return hashlib.sha256(np.asarray(value,np.float64).tobytes()).hexdigest()


def prepare(source):
    ROOT.mkdir(exist_ok=False);protected=collections.defaultdict(set);manifests=[]
    for path in [BASE/'runs/metaworld_eval_20261103/cases-v2/manifest.json',
                 BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json']:
        data=read(path);manifests.append(dict(path=str(path),sha256=sha(path),rows=len(data['rows'])))
        for row in data['rows']:
            with np.load(row['path']) as z:protected[row['task']].add(vector_hash(z['rand_vec']))
    jobs=[]
    for task_index,task in enumerate(sorted(configuration(LEROBOT)['TASK_POLICY_MAPPING'])):
        for split_index,split in enumerate(['train','test']):
            index=len(jobs);seed=500901+task_index*997+split_index
            prefix=int(np.random.default_rng(seed+700001).integers(7,201));name=f'{task}-{split}-seed{seed}'
            jobs.append(dict(index=index,name=name,task=task,seed=seed,prefix=prefix,split=split,output=str(ROOT/'cases'/name)))
    assert len(jobs)==100
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        helper_sha256=sha(__file__),jobs=jobs,protected_vectors={k:sorted(v) for k,v in protected.items()},
        exclusion_manifests=manifests,deadline=time.time()+7200,physical_cases=100,branches_per_case=8,
        random_native_per_branch=60,stored_history_native=7,
        scope='NEWADDITIONALDATA,notexistingfairbaselinebudget. 50tasksx2freshcases,casegroupedtrain/test. '
        'ScriptedexpertcollectorONLYselectsphysicalstateatPRESAMPLEDnativeprefix7..200;NOsuccess/rewardfiltering or optimal-actiontargets. '
        'Then8branchesofOURcontrols(0zero+7independentwhiteGaussianheld5). Allobservations/knownexecutedinputsordinarylogs. '
        'Modelneverreadsstate/reward/success/taskID/policy/desiredgoal. Expert-collectedcontext,NOTexpert-free. '
        'Alloldprotectedcasevectorsstrictlyexcluded;no protectedtrajectory/goalRGBreadforlearning.',
        training_launched=False,automatic_training=False))


@torch.no_grad()
def gather(job):
    p=read(ROOT/'plan.json');assert sha(__file__)==p['helper_sha256'];torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    task=job['task'];env,state,vector=make_environment(task,job['seed'])
    assert vector_hash(vector) not in p['protected_vectors'].get(task,[]),'Protectedphysicalcasecollision'
    policy=expert(LEROBOT,task);frames=[byte(rgb_tensor(pixels(env),FactualRecurrent))];known=[];native=0
    try:
        for _ in range(job['prefix']):
            a=np.clip(policy.get_action(state),-1,1).astype(np.float32);state,_,term,trunc,_=env.step(a);native+=1
            assert not term and not trunc;known.append(a);frames.append(byte(rgb_tensor(pixels(env),FactualRecurrent)))
        history=np.stack(frames[-8:]);past=np.asarray(known[-7:],np.float32);snapshot=capture(env)
        pre=pixels(env);restore(env,snapshot);canonical=mjstate(env).copy();branch=pixels(env)
        history[-1]=byte(rgb_tensor(branch,FactualRecurrent));canonicalization=rgb_replay_check(pre,branch)
        rng=np.random.default_rng(job['seed']+900001);controls=rng.standard_normal((8,12,4)).clip(-1,1).astype(np.float32)
        controls=np.repeat(controls,5,axis=1);controls[0]=0;all_images=[];all_inputs=[];end_states=[];zero=None;gates=[]
        for i in range(9):
            restore(env,snapshot);assert np.array_equal(mjstate(env),canonical)
            gate=rgb_replay_check(branch,pixels(env));assert gate['passed'],gate;gates.append(gate)
            images=list(history);inputs=list(past);start_state=state.copy()
            for a in controls[i if i<8 else 0]:
                state,_,term,trunc,_=env.step(a);native+=1;assert not term and not trunc
                inputs.append(a.copy());images.append(byte(rgb_tensor(pixels(env),FactualRecurrent)))
            final=mjstate(env).copy()
            if i==0:zero=dict(mj=final,state=state.copy(),raw=pixels(env))
            if i==8:
                assert np.array_equal(final,zero['mj']) and np.array_equal(state,zero['state'])
                repeat=rgb_replay_check(zero['raw'],pixels(env));assert repeat['passed']
            else:
                inputs.append(np.zeros(4,np.float32));all_images.append(np.stack(images));all_inputs.append(np.stack(inputs));end_states.append(state.copy())
        images=np.stack(all_images);inputs=np.stack(all_inputs)
        assert images.shape==(8,68,128,128,3) and inputs.shape==(8,68,4) and images.dtype==np.uint8
        assert np.isfinite(inputs).all() and np.max(np.abs(inputs))<=1
        assert np.array_equal(inputs[:,7:67],controls) and np.array_equal(inputs[:,:7],np.broadcast_to(past,(8,7,4)))
        out=Path(job['output']);out.mkdir(exist_ok=False);path=out/'transitions.npz'
        np.savez_compressed(path,pixels=images,known_input=inputs)
        save(out/'DONE.json',dict(passed=True,utc=now(),job=job,stored=str(path),sha256=sha(path),rand_vec_hash=vector_hash(vector),
            native_steps=native,branches=8,stored_frames=8*68,train_model_fields=['pixels','known_input'],
            nine_initial_integration_restores_exact=True,all_initial_RGB_gates=gates,zero_replay_exact=True,zero_rgb=repeat,
            pre_to_canonical_RGB=canonicalization,expert_policy_source_sha256=sha(inspect.getfile(type(policy))),
            between_branch_endpoint_object4to7_std_audit_only=np.asarray(end_states)[:,4:7].std(0).tolist(),
            source_hashes=p['source_hashes'],training_updates=0,protected_vectors_excluded=True,
            split_group='All8siblingsandhistoryshareonephysicalcaseandonesplit;NOT8independentcases.',
            no_expert_path_or_control_targets=True,no_goal_reward_success_state_taskID_model_fields=True))
    finally:close_environment(env)


def work(host,gpu):
    p=read(ROOT/'plan.json');s=Path(p['source'])
    assert {str(f.relative_to(s)):sha(f) for f in s.rglob('*.py')}==p['source_hashes']
    for job in p['jobs']:
        claim=ROOT/'claims'/(job['name']+'.json');claim.parent.mkdir(exist_ok=True)
        try:
            with claim.open('x') as f:json.dump(dict(host=host,gpu=gpu,pid=os.getpid(),utc=now()),f)
        except FileExistsError:continue
        if (Path(job['output'])/'DONE.json').exists():continue
        try:
            command([PYTHON,str(Path(__file__).resolve()),'gather','--name',job['name']],Path(job['output']).with_suffix('.log'),environment(s,gpu),s,p['deadline'],cap=1800)
        except BaseException:
            save(ROOT/'errors'/(job['name']+'.json'),dict(utc=now(),traceback=traceback.format_exc()));raise


def preflight():
    p=read(ROOT/'plan.json');j=next(x for x in p['jobs'] if x['task']=='push-v3' and x['split']=='train')
    gather(j);d=read(Path(j['output'])/'DONE.json');assert d['passed'] and d['zero_replay_exact']
    save(ROOT/'PREFLIGHT.json',dict(passed=True,utc=now(),inherited_planned_case=j['name'],native_steps=d['native_steps'],
        real8branches_sourceRGB_knowninput_alignment_and9physicsrestores_passed=True,training_updates=0))


def launch(host):
    p=read(ROOT/'plan.json');assert read(ROOT/'PREFLIGHT.json')['passed'];s=Path(p['source']);workers=[]
    for gpu in range(8):
        with (ROOT/f'worker-{host}-{gpu}.log').open('x') as f:
            child=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'work','--host',host,'--gpu',str(gpu)],cwd=s,
                env=environment(s,gpu)|{'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(dict(gpu=gpu,pid=child.pid))
    save(ROOT/f'launch-{host}.json',dict(utc=now(),workers=workers,host=host));print(host,workers,flush=True)


def pack():
    p=read(ROOT/'plan.json');data=BASE/'data/metaworld-random-interventions-native128-20261109-v1';data.mkdir(exist_ok=False)
    proofs=[]
    for job in p['jobs']:
        d=read(Path(job['output'])/'DONE.json');assert d['passed'] and sha(d['stored'])==d['sha256'];proofs.append(d)
    for shard in range(12):
        chosen=[d for d in proofs if d['job']['index']%12==shard];n=len(chosen)*8*68;s=data/str(shard);s.mkdir()
        rgb=np.lib.format.open_memmap(s/'pixels.npy',mode='w+',dtype=np.uint8,shape=(n,128,128,3))
        action=np.lib.format.open_memmap(s/'known_input.npy',mode='w+',dtype=np.float32,shape=(n,4));records=[];offset=0
        for d in chosen:
            j=d['job']
            with np.load(d['stored']) as z:
                for b in range(8):
                    rgb[offset:offset+68]=z['pixels'][b];action[offset:offset+68]=z['known_input'][b]
                    records.append(dict(episode=1000000+j['index']*8+b,offset=offset,length=68,split=j['split']));offset+=68
        rgb.flush();action.flush();del rgb,action
        save(s/'manifest.json',dict(passed=True,physical_stride=1,all_phases_grouped_split=True,terminal_macro_padding_recovered=True,
            no_macro_padding_ever_created=True,records=records,files={f:sha(s/f) for f in ['pixels.npy','known_input.npy']},
            branch_sibling_split_grouped=True,stored_frames=n,native_pairs=len(records)*67))
    save(data/'manifest.json',dict(passed=True,utc=now(),source_plan_sha256=sha(ROOT/'plan.json'),physical_cases=100,episodes=800,
        train_cases=50,test_cases=50,branches_per_case=8,pixels=54400,random_native_inputs=48000,
        additional_data_budget=True,ordinary_expert_collected_context_and_OUR_random_interventions=True,
        no_goal_reward_success_state_taskID_or_optimal_action_model_fields=True,
        all_protected_case_vectors_excluded=True,hashes={str(f.relative_to(data)):sha(f) for f in data.glob('*/manifest.json')}))
    return str(data)


def collect():
    p=read(ROOT/'plan.json')
    while time.time()<p['deadline']:
        done=[read(Path(j['output'])/'DONE.json') for j in p['jobs'] if (Path(j['output'])/'DONE.json').exists()]
        errors=list(map(str,ROOT.glob('errors/*.json')))
        save(ROOT/'results/summary.json',dict(utc=now(),completed_cases=len(done),planned_cases=100,errors=errors,
            native_steps=sum(x['native_steps'] for x in done),source='Extraordinarytransitiondata,NOTevaluationperformance.',training_updates=0))
        if len(done)==100:
            assert not errors;directory=pack()
            save(ROOT/'CLOSED.json',dict(passed=True,utc=now(),data=directory,metadata_and_all100_source_bytes_verified=True,
                ordinary_transition_training_eligible=True,automatic_training=False,optimizer_updates=0));return
        time.sleep(15)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','preflight','gather','work','launch','collect']);a.add_argument('--source',type=Path);a.add_argument('--host');a.add_argument('--gpu',type=int);a.add_argument('--name');args=a.parse_args()
    if args.mode=='prepare':prepare(args.source)
    elif args.mode=='preflight':preflight()
    elif args.mode=='gather':gather(next(x for x in read(ROOT/'plan.json')['jobs'] if x['name']==args.name))
    elif args.mode=='work':work(args.host,args.gpu)
    elif args.mode=='launch':launch(args.host)
    else:collect()
