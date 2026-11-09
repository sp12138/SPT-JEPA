"""Protected multi-task physical test of actuator loops and input-inverse closure.

Never training/teacher/proposal data or policy-performance evidence. The
closure head reconstructs the SAME executed input from adjacent observed RGB,
and here servos toward an already observed branch image, not an expert future.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent
from experiments.metaworld_horizon_match_20261109.model import build
from experiments.metaworld_horizon_match_20261109.bank import capture,restore,mjstate,byte
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

ROOT=BASE/'runs/metaworld_loop_geometry_20261109/compare-v1'
PARENT=BASE/'runs/metaworld_belief_state_20261109'


def auc(score,flags):
    if not flags.any() or flags.all():return None
    return float(((score[flags,None]<score[None,~flags])+.5*(score[flags,None]==score[None,~flags])).mean())


@torch.no_grad()
def run(task,prefix):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=read(BASE/'runs/metaworld_horizon_match_20261109/training/h60-seed2/config.json')
    weights=Path(cfg['output'])/'model-step0012000/weights.pt'
    assert sha(weights)==read(weights.parent/'SEALED.json')['weights_sha256']
    inverse=build(cfg).cuda().eval().requires_grad_(False)
    inverse.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True);before=tensor_sha(inverse)
    case=next(c for c in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'] if c['task']==task)
    reference=read(PARENT/'evaluation'/f'joint_anchor-seed2-s500-{task}'/'metrics.json')['episodes'][0]
    assert sha(reference['trace'])==reference['trace_sha256'] and case['sha256']==reference['case_sha256']
    assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);logged=np.load(reference['trace'])
    assert len(logged['actions'])>=prefix
    env,state,_=make_environment(task,case['seed'],cached['rand_vec'])
    assert np.array_equal(state,cached['initial_state'])
    assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
    gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'];native=0
    try:
        for t,a in enumerate(logged['actions'][:prefix]):
            state,_,term,trunc,info=env.step(a);native+=1
            assert bool(info['success'])==bool(logged['successes'][t]) and not term and not trunc
        snapshot=capture(env);branch_raw=pixels(env);restore(env,snapshot);canonical=mjstate(env).copy()
        initial_state=np.asarray(state).copy();initial_image=rgb_tensor(branch_raw,FactualRecurrent)
        target_feature=inverse.vision(initial_image)
        goal=byte(rgb_tensor(cached['goal'],FactualRecurrent)).astype(float)/255
        raw=np.random.default_rng(470009).standard_normal((64,12,4)).clip(-1,1).astype(np.float32)
        white=np.repeat(raw,5,axis=1);u,v=raw[:,0],raw[:,1]
        loop=np.repeat(np.stack([u,v,-u,-v],1),15,axis=1);white[0]=0;loop[0]=0
        assert np.max(np.abs(loop.astype(np.float64).sum(1)))<1e-10
        reports={};out=ROOT/f'{task}-prefix{prefix}';out.mkdir(exist_ok=False)
        for kind,controls,closure in [('white',white,False),('loop',loop,False),('white-inverse',white,True),('loop-inverse',loop,True)]:
            frames=[];states=[];flags=[];realized=[];zero=None
            for i in range(65):
                restore(env,snapshot);assert np.array_equal(mjstate(env),canonical)
                start_gate=rgb_replay_check(branch_raw,pixels(env));assert start_gate['passed']
                row=[];commands=[]
                actions=controls[i] if i<64 else zero['actions']
                for a in actions:
                    state,_,term,trunc,info=env.step(a);native+=1;assert not term and not trunc
                    row.append(bool(info['success']));commands.append(a.copy())
                if closure and i<64:
                    for _ in range(15):
                        image=rgb_tensor(pixels(env),FactualRecurrent);feature=inverse.vision(image)
                        action=inverse.inverse(torch.cat([feature,target_feature,target_feature-feature],1)).float().clamp(-1,1)[0].cpu().numpy()
                        state,_,term,trunc,info=env.step(action);native+=1;assert not term and not trunc
                        row.append(bool(info['success']));commands.append(action.copy())
                frame=byte(rgb_tensor(pixels(env),FactualRecurrent));integration=mjstate(env)
                if i==0:zero=dict(actions=np.asarray(commands),integration=integration.copy(),state=state.copy(),flags=row,raw=pixels(env))
                if i==64:
                    assert np.array_equal(integration,zero['integration']) and np.array_equal(state,zero['state']) and row==zero['flags']
                    zero_rgb=rgb_replay_check(zero['raw'],pixels(env));assert zero_rgb['passed']
                else:frames.append(frame);states.append(state.copy());flags.append(row);realized.append(commands)
            frames=np.asarray(frames);states=np.asarray(states);flags=np.asarray(flags,bool);realized=np.asarray(realized,np.float32)
            score=((frames.astype(float)/255-goal)**2).mean((1,2,3));selected=int(score.argmin())
            path=out/(kind+'.npz');np.savez_compressed(path,actual=frames,states_audit_only=states,successes=flags,
                controls=realized,initial_state_audit_only=initial_state,goal=(goal*255).round().astype(np.uint8))
            reports[kind]=dict(bank=str(path),bank_sha256=sha(path),paths=64,native_horizon=realized.shape[1],
                any_successes=int(flags.any(1).sum()),final_successes=int(flags[:,-1].sum()),
                actual_RGB_selected=selected,actual_RGB_selected_any_success=bool(flags[selected].any()),
                actual_RGB_selected_final_success=bool(flags[selected,-1]),AUC_any=auc(score,flags.any(1)),AUC_final=auc(score,flags[:,-1]),
                mean_TCP_return_error_m=float(np.linalg.norm(states[:,:3]-initial_state[:3],axis=1).mean()),
                TCP_endpoint_std_m=states[:,:3].std(0).tolist(),gripper_endpoint_std_audit_only=float(states[:,3].std()),
                mean_object_displacement_m=float(np.linalg.norm(states[:,4:7]-initial_state[4:7],axis=1).mean()),
                object_endpoint_std_m=states[:,4:7].std(0).tolist(),all65_branch_restores_exact=True,
                zero_REALIZED_controls_replay_exact=True,zero_rgb_gate=zero_rgb,
                scope='Object4:7isdiagnosticposeproxy,notuniversalreward. Closed-loopzero replayusesitsRECORDEDcontrols,notassumedRGB-policy-bitdeterminism.')
        assert before==tensor_sha(inverse)
        save(out/'DONE.json',dict(passed=True,utc=now(),task=task,prefix=prefix,case_sha256=case['sha256'],
            source_trace_sha256=reference['trace_sha256'],source_initial_rgb_gate=gate,source_initial_physics_exact=True,
            inverse_weights_sha256=sha(weights),inverse_before=before,inverse_after=before,reports=reports,
            physics_steps=native,model_updates=0,protected_audit_only=True,training_teacher_proposal_eligible=False,
            design='Matched64whitevsu,v,-u,-vloops;60native/marginalclippedGaussian. Optional15stepFP32INPUT-inverseclosuretowardalreadyOBSERVEDbranchRGB. No expertfuture/path/actionlabels.',
            scope='8previouslyinspectedcasesx3branchstates. Thesearephysicalinterventions,NOT500stepagentperformance or independenttrajectoryconfirmation.'))
    finally:close_environment(env)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--prefix',type=int,choices=[0,30,60],required=True);a=p.parse_args();run(a.task,a.prefix)
