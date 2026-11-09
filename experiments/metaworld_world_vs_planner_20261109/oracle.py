"""PRIVILEGED DIAGNOSTIC ONLY. Never count as our algorithm or train from it.

Replace only the forecast function by actual simulator transitions, preserving
CEM64x4/H60/held5/execute15/RGBterminalcost/native500. No reward or native
object-goal distance enters the objective. Full simulator state is privileged.
"""
import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import numpy as np
import torch
import mujoco
from experiments.metaworld_transition_only_20261109.common import BASE,PYTHON,read,save,sha,now,environment
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_horizon_match_20261109.bank import capture,restore,mjstate,SPEC

ORIGINAL=BASE/'runs/metaworld_world_vs_planner_20261109'
ROOT=ORIGINAL/'recovery-v3'
OLD=BASE/'runs/metaworld_transition_only_20261109'
CORRELATED=False
HORIZON=60
BLOCK=5
VARIANT='white'


def restore_exact(env,state):
    restore(env,state)
    # mj_forwardrefreshesderivedpositions. Reapplyintegration/warmstartstate
    # afterwards, so real execution starts from the exact preplanning state.
    mujoco.mj_setState(env.model,env.data,state['mj'],SPEC)
    assert np.array_equal(mjstate(env),state['mj'])


@torch.no_grad()
def optimize(env,snapshot,goal,generator,warm):
    mean=torch.zeros(12,4,device='cuda') if warm is None else torch.as_tensor(warm,device='cuda').reshape(12,BLOCK,4).mean(1)
    std=torch.ones_like(mean);best=None;best_score=float('inf');iterations=[]
    for iteration in range(4):
        raw=torch.randn(64,12,4,device='cuda',generator=generator)
        if CORRELATED:
            from experiments.metaworld_correlated_search_20261109.evaluate import mixing
            raw=torch.einsum('ij,nja->nia',mixing(raw.device),raw)
        candidates=(mean[None]+std[None]*raw).clamp(-1,1);candidates[0]=mean
        if best is not None:candidates[1]=best
        prefix=hashlib.sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
        commands=candidates[:, :,None,:].expand(-1,-1,BLOCK,-1).reshape(64,HORIZON,4).cpu().numpy()
        scores=[]
        for path in commands:
            restore_exact(env,snapshot)
            actual_limit=env.max_path_length
            # The learned predictor also forecasts a full60steps near episode
            # end. Lift ONLY imagined administrative truncation,notphysics.
            env.max_path_length=max(actual_limit,int(env.curr_path_length)+HORIZON+1)
            try:
                for control in path:env.step(control.astype(np.float32))
                image=rgb_tensor(pixels(env),FactualRecurrent)
                scores.append(float((image-goal).square().mean()))
            finally:env.max_path_length=actual_limit
        costs=torch.tensor(scores,device='cuda');assert torch.isfinite(costs).all()
        values,indices=costs.topk(10,largest=False);elite=candidates[indices]
        if float(values[0])<best_score:best_score=float(values[0]);best=elite[0].clone()
        mean=elite.mean(0);std=elite.std(0,unbiased=False).clamp_min(1e-6)
        iterations.append(dict(iteration=iteration,first_bank_prefix_sha256=prefix,min_cost=float(values[0]),mean_cost=float(costs.mean())))
    return best[:,None].expand(-1,BLOCK,-1).reshape(HORIZON,4).cpu().numpy(),dict(best_terminal_RGB_mse=best_score,
        iterations=iterations,imagined_native_calls=64*4*HORIZON,horizon_native=HORIZON,control_hold_native=BLOCK)


def prefix(env,commands):
    actions=[];flags=[];observation=None
    for control in commands:
        observation,_,terminated,truncated,info=env.step(control.astype(np.float32))
        actions.append(control.copy());flags.append(bool(info['success']))
        if flags[-1] or terminated or truncated:break
    return np.asarray(actions),np.asarray(flags),observation.copy(),mjstate(env).copy()


def run(task):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.20)
    plan=read(ROOT/'plan.json');case=next(c for c in plan['cases'] if c['task']==task)
    out=ROOT/task;out.mkdir(exist_ok=True);env=None
    def terminate(signum,frame):raise TimeoutError('Finite oracle diagnostic terminated')
    signal.signal(signal.SIGTERM,terminate)
    try:
        assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(task,case['seed'],cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        initial=rgb_replay_check(cached['initial'],pixels(env));assert initial['passed']
        goal=rgb_tensor(cached['goal'],FactualRecurrent);generator=torch.Generator(device='cuda').manual_seed(610000+case['seed'])
        reference=read(OLD/'evaluation/h15-inverse1-seed1-s10000-H60-e15/metrics.json')
        ref=next(r for r in reference['episodes'] if r['task']==task and r['seed']==case['seed'])
        expected_prefix=ref['planning'][0]['first_bank_prefix_sha256']
        if CORRELATED:
            from experiments.metaworld_correlated_search_20261109.evaluate import mixing
            contract_generator=torch.Generator(device='cuda').manual_seed(610000+case['seed'])
            raw=torch.randn(64,12,4,device='cuda',generator=contract_generator)
            candidates=torch.einsum('ij,nja->nia',mixing(raw.device),raw).clamp(-1,1)
            candidates[0]=0
            expected_prefix=hashlib.sha256(candidates[:,:6].cpu().numpy().tobytes()).hexdigest()
            if VARIANT=='long':
                short=read(ORIGINAL/'correlated-v4'/task/'progress.json')
                assert expected_prefix==short['decisions'][0]['iterations'][0]['first_bank_prefix_sha256']
        actions=[];successes=[];decisions=[];warm=None;started=time.monotonic();imagined=0
        checkpoint=out/'checkpoint-latest.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['source_sha256']==sha(__file__)
            restore_exact(env,saved['physics']);generator.set_state(saved['generator'])
            actions=list(saved['actions']);successes=list(saved['successes']);decisions=saved['decisions'];warm=saved['warm'];imagined=saved['imagined']
            assert len(actions)==env.curr_path_length and env.max_path_length==500
            save(out/f'RESUME-{os.getpid()}.json',dict(utc=now(),native_steps=len(actions),full_physics_rng_cursor_restored=True,candidate_performance=False))
        while len(actions)<500:
            assert time.time()<plan['deadline']
            snapshot=capture(env);commands,detail=optimize(env,snapshot,goal,generator,warm);imagined+=detail['imagined_native_calls']
            if not decisions:assert detail['iterations'][0]['first_bank_prefix_sha256']==expected_prefix
            count=min(15,500-len(actions))
            restore_exact(env,snapshot);shadow=prefix(env,commands[:count]);imagined+=len(shadow[0])
            restore_exact(env,snapshot);actual=prefix(env,commands[:count])
            assert all(np.array_equal(a,b) for a,b in zip(shadow,actual)),'Simulation restore changed chosen execution'
            executed,flags,observation,final=actual;actions.extend(executed);successes.extend(flags.tolist())
            image=rgb_tensor(pixels(env),FactualRecurrent);detail.update(native_step=len(actions),actual_RGB_goal_mse=float((image-goal).square().mean()),
                actual_steps=len(executed),chosen_prefix_replay_exact=True)
            decisions.append(detail)
            next_warm=np.concatenate([commands[count:],np.repeat(commands[-1:],count,axis=0)],axis=0)
            partial=out/'trace-partial.npz';np.savez_compressed(partial,actions=np.asarray(actions,np.float32),successes=np.asarray(successes,bool))
            payload=dict(physics=capture(env),generator=generator.get_state(),actions=np.asarray(actions),successes=np.asarray(successes),
                decisions=decisions,warm=next_warm,imagined=imagined,source_sha256=sha(__file__),candidate_performance=False)
            tmp=checkpoint.with_name(checkpoint.name+f'.{os.getpid()}.tmp');torch.save(payload,tmp);os.replace(tmp,checkpoint)
            save(out/'progress.json',dict(utc=now(),native_steps=len(actions),success=bool(any(successes)),decisions=decisions,
                imagined_native_calls=imagined,candidate_performance=False,information='PRIVILEGED_SIMULATOR_AUDIT_ONLY'))
            print(task,len(actions),bool(any(successes)),detail['actual_RGB_goal_mse'],flush=True)
            if successes[-1] or len(executed)<count:break
            warm=next_warm
        trace=out/'trace.npz';np.savez_compressed(trace,actions=np.asarray(actions,np.float32),successes=np.asarray(successes,bool))
        save(out/'DONE.json',dict(passed=True,utc=now(),task=task,case_seed=case['seed'],case_sha256=case['sha256'],success=bool(any(successes)),
            native_steps=len(actions),imagined_native_calls=imagined,seconds=time.monotonic()-started,initial_rgb=initial,
            all_selected_execution_replays_exact=True,initial_candidate_bank_contract_verified=True,
            initial_bank_matches_white_reference=not CORRELATED,proposal='fixed_DCT_covariance' if CORRELATED else 'original_independent',
            horizon_native=HORIZON,control_hold_native=BLOCK,action_knots=12,execute_native=15,
            trace=str(trace),trace_sha256=sha(trace),
            candidate_performance=False,training_data_eligible=False,model_updates=0,
            information='PRIVILEGEDtrue-simulatorprediction;NOTourlearnedscratchalgorithm/NOTfairSOTAscore/NOtrainingorproposalreuse.'))
    except BaseException:
        save(out/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc(),candidate_performance=False));raise
    finally:
        if env is not None:close_environment(env)


def prepare(source):
    old_plan=read(ORIGINAL/'plan.json');ROOT.mkdir(exist_ok=False);cases=old_plan['cases']
    assert len(cases)==3
    inherited={} if CORRELATED else {r['task']:str(ORIGINAL/r['task']/'DONE.json') for r in cases if (ORIGINAL/r['task']/'DONE.json').exists()}
    for path in inherited.values():assert read(path)['passed'] and not read(path)['candidate_performance']
    save(ROOT/'AUDIT_ONLY_PRIVILEGED_NO_TRAINING.json',dict(utc=now(),candidate_performance=False,
        excluded='Neveraggregateasouralgorithm/SOTAbaseline;nevertrainfromstates/RGB/actions oruseasactionproposals/goalpathteachers.'))
    save(ROOT/'plan.json',dict(utc=now(),source=str(source),deadline=old_plan['deadline'],cases=cases,inherited=inherited,
        variant=VARIANT,horizon_native=HORIZON,control_hold_native=BLOCK,
        jobs=[dict(task=task,host=host,gpu=7) for task,host in [('door-open-v3','dev13'),('drawer-open-v3','dev15'),('push-v3','dev6')] if task not in inherited],
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},model_updates=0,
        design='Samecachedinitialcases/RGBgoals/initialGaussianbanks/CEM64x4/top10/H60/held5/execute15/native500anysuccess. '
            'Replaceforecastwithtruephysics+renderedRGB. CorrelatedvariantchangesONLYfixedproposal-timecovariance,likefrozenlearnerprobe. '
            'Noreward/native-object-target/objectstateobjective;fullstatecloningISprivileged.',
        interpretation='Diagnosticceilingon3selecteddevelopmentalphysicalcases,notouralgorithmperformance. '
            'Everychosenprefixreplayedtwicewithfullstate/RNGrestoreandcheckedbitexact. '
            'v2Door/Pushfirst450nativepartialrecords/errorsretained;no recoverablestatewaswritten,v3replaysONLYthese2unfinisheddiagnostics. '
            '0SGDrepeated;completedDrawerv2inherited/notrerun. ImaginedtimecapliftedtoallowfullH60likelearnedpredictor;real500unchanged. '
            'v3fullphysics/privateRNG/actions/warmplancheckpointsafteveryactualprefix. Originaldeadlineunchanged. '
            'Coloredv4runs3newpairedcases;doesNOTinheritwhiteperformancesascolored. White3validresultsretained/notrerun.'))
    if VARIANT=='long':
        p=read(ROOT/'plan.json')
        p['interpretation']=('PrivilegedH120/held10vscompletedH60/held5coloredCEM. Same12actionknots/64x4/RGBobjective/native500/execute15andinitialGaussian-knotbank. '
            'Temporal dilation changes horizon ANDhold;not equalphysicscompute orsinglefactorholdindependentcomparison. '
            'No expertcontrols,policylabels,reward/objectstatecost. Oldshortresultsretainednotrerun. NEVERcandidateperformance/training/proposaldata.')
        p['design']='Truephysics/RGBgoal/CEM64x4/top10/H120/held10/execute15/native500,12actionknots;fixedDCTproposal-timecovariance.'
        save(ROOT/'plan.json',p)


def launch(host):
    p=read(ROOT/'plan.json');s=Path(p['source']);job=next(j for j in p['jobs'] if j['host']==host)
    with (ROOT/f'worker-{host}.log').open('x') as f:
        child=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=30','14400',PYTHON,'-m','experiments.metaworld_world_vs_planner_20261109.oracle','work','--task',job['task'],
            '--variant',p.get('variant','white')],
            cwd=s,env=environment(s,job['gpu']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    save(ROOT/f'launch-{host}.json',dict(job,pid=child.pid,utc=now(),candidate_performance=False));print(job,child.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','launch','work']);p.add_argument('--source',type=Path);p.add_argument('--host');p.add_argument('--task')
    p.add_argument('--variant',choices=['white','correlated','long'],default='white');a=p.parse_args()
    VARIANT=a.variant;CORRELATED=a.variant in ['correlated','long']
    if a.variant=='correlated':ROOT=ORIGINAL/'correlated-v4'
    if a.variant=='long':ROOT=ORIGINAL/'long-horizon-v1';HORIZON=120;BLOCK=10
    if a.mode=='prepare':prepare(a.source)
    elif a.mode=='launch':launch(a.host)
    else:run(a.task)
