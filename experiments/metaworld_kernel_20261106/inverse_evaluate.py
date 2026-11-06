"""RGB goal controller evaluation;only self-generated commands enter execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
from experiments.metaworld_contact_20261106.bridge_campaign import TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest


def run(a):
    from as_jepa_lewm.factory import build_model
    a.feedback_native=getattr(a,'feedback_native',5)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((a.run/'config.json').read_text());model=build_model(cfg).cuda().eval().requires_grad_(False)
    assert model.visual_source=='local'and not model.temporal_aux and model.condition_horizon=='none'
    assert not model.history_context or(model.chunk==1 and a.execute==1)
    assert model.action_dim in[4,20]and(model.action_dim==20 or not model.history_context)
    assert a.feedback_native in[1,5]and(a.feedback_native==5 or(not model.history_context and a.execute==1))
    assert a.execute<=model.chunk
    model.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cuda',weights_only=True));before=model_sha(model)
    card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text());mean=np.asarray(card['action_mean'][:model.action_dim],np.float32);std=np.asarray(card['action_std'][:model.action_dim],np.float32)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['task']in a.tasks and r['seed']in a.seeds]
    cases.sort(key=lambda r:(a.tasks.index(r['task']),a.seeds.index(r['seed'])));assert len(cases)==len(a.tasks)*len(a.seeds);rows=[]
    with torch.inference_mode():
        for case in cases:
            assert digest(Path(case['path']))==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate
            image=cached['initial'];past=image;previous_action=torch.zeros(1,20,device='cuda');history_valid=torch.zeros(1,device='cuda')
            goal=resize_tensor([cached['goal']]).cuda();actions=[];successes=[];calls=0
            generator=torch.Generator(device='cuda').manual_seed(778000+case['seed'])
            try:
                while len(actions)<a.budget:
                    if a.noise_policy=='fixed':generator.manual_seed(778000+case['seed'])
                    current=resize_tensor([image]).cuda();batch=dict(pixels=torch.stack([current,goal],1))
                    if model.history_context:
                        batch.update(pixels=torch.stack([resize_tensor([past]).cuda(),current,goal],1),previous_action=previous_action,history_valid=history_valid)
                    old_image=image
                    proposed=model.propose(batch,generator=generator,steps=12)[0,:a.execute].float().cpu().numpy()
                    native=np.clip(proposed*std+mean,-1,1).reshape(-1,4);assert np.isfinite(native).all();calls+=1
                    if model.action_dim==4:native=np.repeat(native,a.feedback_native,axis=0)
                    elif a.feedback_native==1:native=native[:1]
                    for j,command in enumerate(native):
                        _,_,terminated,truncated,info=env.step(command.astype(np.float32));actions.append(command.copy());successes.append(bool(info['success']))
                        if successes[-1]or terminated or truncated or len(actions)>=a.budget:break
                        if(j+1)%a.feedback_native==0:image=pixels(env)
                    if model.history_context and len(actions)>=5:
                        actual=np.asarray(actions[-5:]).reshape(20);previous_action=torch.from_numpy(((actual-mean)/std)[None]).cuda();history_valid.fill_(1);past=old_image
                    if successes[-1]or terminated or truncated:break
            finally:close_environment(env)
            trace=a.output/f'{case["task"]}-seed{case["seed"]}.npz';np.savez_compressed(trace,actions=np.stack(actions),successes=np.asarray(successes))
            row=dict(task=case['task'],seed=case['seed'],success=bool(any(successes)),steps=len(actions),calls=calls,case_sha256=case['sha256'],
                expert_goal_success=case['expert_any_success'],initial_rgb=gate,trace=str(trace),trace_sha256=digest(trace))
            rows.append(row);save(a.output/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
    assert before==model_sha(model)
    save(a.output/'summary.json',dict(passed=True,rows=rows,model_unchanged=True,model_sha256=before,step=a.step,execute=a.execute,
        training_action_blocks=model.chunk,training_current_native_actions=model.action_dim//4,native_hold=a.feedback_native if model.action_dim==4 else 1,feedback_native=a.feedback_native,
        noise_policy=a.noise_policy,own_past_history=model.history_context,new_updates=0,no_future_demonstrated_command_reference_in_controller=True,
        scope='OriginalLeRobotcorpus,commonRGBgoals/resetstates,500nativeanysuccess,DECLAREDfeedback_native1or5,12ownflowintegrationsteps.'
              'OnlyRGBcurrent/goalplusdeclaredpastRGB/OWNexecutedcontrolhistory;no clock/taskID/proprio/reward/successinputs.EachcommandcomesfromownGaussianproposal.'
              'Chunk6modelsarelabelledfuture-targetBASELINESONLY;newcurrent-onlycandidatechunk1.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in['run','manifest','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--execute',type=int,choices=[1,3],default=1);p.add_argument('--budget',type=int,default=500)
    p.add_argument('--noise-policy',choices=['fresh','fixed'],default='fresh')
    p.add_argument('--feedback-native',type=int,choices=[1,5],default=5)
    p.add_argument('--tasks',nargs='+',default=TASKS);p.add_argument('--seeds',nargs='+',type=int,default=[42,1042]);run(p.parse_args())
