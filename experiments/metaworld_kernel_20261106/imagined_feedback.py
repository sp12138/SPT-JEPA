"""Frozen current-only controller with causal imagined feedback inside sensor gaps."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from as_jepa_lewm.factory import build_model
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_eval_20261103.evaluate import resize_tensor,model_sha
from experiments.metaworld_contact_20261106.bridge_campaign import TASKS
from experiments.metaworld_kernel_20261106.campaign import save,digest
from as_jepa_tools.supervision_contract import validate_training_supervision


def composed_image(reference,coordinates,original_grid):
    """Render the composed backward map once, preserving the source resolution."""
    h,w=reference.shape[-2:]
    if (h,w)==(128,128):grid=coordinates.permute(0,2,3,1)
    else:
        y=(torch.arange(h,device=reference.device,dtype=torch.float32)+.5)*(2/h)-1
        x=(torch.arange(w,device=reference.device,dtype=torch.float32)+.5)*(2/w)-1
        yy,xx=torch.meshgrid(y,x,indexing='ij')
        identity=torch.stack([xx,yy],-1)[None]
        offset=coordinates-original_grid.permute(0,3,1,2)
        grid=identity+F.interpolate(offset,(h,w),mode='bilinear',align_corners=False).permute(0,2,3,1)
    return F.grid_sample(reference.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)


def sample_policy(model,batch,generator,steps,solver,draw='stochastic'):
    if draw=='component-median':
        assert model.objective=='mixture'
        logits,location,_=model.distribution(model.memory(batch))
        index=logits.argmax(-1)[...,None,None].expand(-1,-1,1,model.action_dim)
        # Median of the highest-weight latent component, then the SAME actuator
        # clip map. This is not claimed to be the MAP of a mixed discrete/
        # continuous output measure, or an optimal physical action.
        return location.gather(2,index).squeeze(2).maximum(model.action_lower).minimum(model.action_upper)
    if solver=='euler'or model.objective!='flow':return model.propose(batch,generator=generator,steps=steps)
    memory=model.memory(batch);horizon=model.duration(memory)
    value=torch.randn(len(memory),model.chunk,model.action_dim,device=memory.device,generator=generator)
    for i in range(steps):
        time=value.new_full((len(value),),1-i/steps)
        velocity=model.action_field(value,time,horizon,memory)
        velocity=model.action_field(value-velocity/(2*steps),time-1/(2*steps),horizon,memory)
        value=value-velocity/steps
    return value


def run(a):
    cfg=json.loads((a.run/'config.json').read_text());validate_training_supervision(cfg)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');a.output.mkdir(parents=True,exist_ok=False)
    assert a.world_action=='own'or a.mode=='world'
    assert a.render_state=='sequential'or a.mode=='world'
    assert cfg['model']['action_dim']in[4,20]and not cfg['model']['history_context']and cfg['model']['chunk']==1
    assert cfg['model']['action_dim']==20 or cfg.get('inverse_tail_reserve')==1
    controller=build_model(cfg).cuda().eval().requires_grad_(False)
    controller.load_state_dict(torch.load(a.run/f'model-step{a.step:07d}/weights.pt',map_location='cuda',weights_only=True))
    assert json.loads((a.run/f'audit-{a.step}.json').read_text())['passed']
    world=None
    if a.mode!='static':
        wc=json.loads((a.world/'config.json').read_text());assert wc['model_class']=='causal_transport'and wc['model']['mode']=='stencil'
        assert not wc.get('initial_checkpoint')and not wc.get('initial_encoder_checkpoint')
        world=build_model(wc).cuda().eval().requires_grad_(False)
        world.load_state_dict(torch.load(a.world/'model-step0020000/weights.pt',map_location='cuda',weights_only=True))
    before=dict(controller=model_sha(controller),world=model_sha(world)if world is not None else None);card=json.loads((Path(cfg['data'])/'dataset_card.json').read_text())
    mean=np.asarray(card['action_mean'][:4],np.float32);std=np.asarray(card['action_std'][:4],np.float32)
    cases=[r for r in json.loads(a.manifest.read_text())['rows']if r['task']in a.tasks and r['seed']in a.seeds]
    cases.sort(key=lambda r:(a.tasks.index(r['task']),a.seeds.index(r['seed'])));assert len(cases)==len(a.tasks)*len(a.seeds);rows=[]
    with torch.inference_mode():
        for case in cases:
            assert digest(Path(case['path']))==case['sha256'];cached=np.load(case['path']);env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])and all(np.array_equal(v,cached['initial_physics_'+k])for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed'],gate
            image=cached['initial'];goal=resize_tensor([cached['goal']]).cuda();actions=[];successes=[];plans=0;forecasts=0;calls=0
            past=goal.new_zeros((1,4),dtype=torch.float32);valid=goal.new_zeros((1,),dtype=torch.float32);generator=torch.Generator(device='cuda').manual_seed(778000+case['seed'])
            try:
                while len(actions)<a.budget:
                    actual=resize_tensor([image]).cuda();belief=world.encoder(actual)if world is not None else None;commands=[]
                    if a.render_state!='sequential':
                        coordinates=world.grid.permute(0,3,1,2).clone()
                        reference=belief if a.render_state=='composed128'else actual.float()/255.
                        actor_belief=None
                    predicted_past=past
                    slot_noise=generator.get_state()if a.noise_coupling=='observation'else None
                    for j in range(5):
                        if slot_noise is not None:generator.set_state(slot_noise)
                        if j==0 or a.mode=='static':current=actual
                        elif a.render_state=='composed224':current=(actor_belief*255).round().clamp(0,255).byte()
                        else:current=F.interpolate(belief*255,(224,224),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255).byte()
                        proposed=sample_policy(controller,dict(pixels=torch.stack([current,goal],1)),generator,a.policy_steps,a.policy_solver,a.policy_draw)[0,0,:4].float().cpu().numpy();calls+=1
                        command=np.clip(proposed*std+mean,-1,1).astype(np.float32);assert np.isfinite(command).all();commands.append(command)
                        control=torch.from_numpy(command[None]).cuda()
                        if a.mode=='world'and j<4:
                            forecast_control=control if a.world_action=='own'else torch.zeros_like(control)if a.world_action=='zero'else-control
                            drift,effect,_,_=world.fields(belief,torch.zeros_like(belief),valid*0,predicted_past,valid)
                            displacement=drift+torch.einsum('na,nachw->nchw',torch.cat([forecast_control,forecast_control.abs()],-1),effect)
                            if a.render_state=='sequential':belief=world.warp(belief,displacement)
                            else:
                                # Compose coordinate maps, not repeatedly resampled RGB.
                                # The SAME learned field now reads this once-rendered
                                # state at the next step; no new NN/label is introduced.
                                coordinates=world.warp(coordinates,displacement)
                                actor_belief=composed_image(reference,coordinates,world.grid)
                                belief=actor_belief if a.render_state=='composed128'else F.interpolate(actor_belief,(128,128),mode='bilinear',align_corners=False,antialias=True)
                            assert torch.isfinite(belief).all();forecasts+=1
                            predicted_past=forecast_control
                        past=control;valid.fill_(1)
                    plans+=1
                    for command in commands:
                        _,_,terminated,truncated,info=env.step(command);actions.append(command.copy());successes.append(bool(info['success']))
                        if successes[-1]or terminated or truncated or len(actions)>=a.budget:break
                    if successes[-1]or terminated or truncated:break
                    image=pixels(env)
            finally:close_environment(env)
            trace=a.output/f'{case["task"]}-seed{case["seed"]}.npz';np.savez_compressed(trace,actions=np.stack(actions),successes=np.asarray(successes))
            row=dict(task=case['task'],seed=case['seed'],success=bool(any(successes)),steps=len(actions),calls=calls,plans=plans,world_forecasts=forecasts,
                case_sha256=case['sha256'],expert_goal_success=case['expert_any_success'],initial_rgb=gate,trace=str(trace),trace_sha256=digest(trace))
            rows.append(row);save(a.output/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
    after=dict(controller=model_sha(controller),world=model_sha(world)if world is not None else None);assert before==after
    save(a.output/'summary.json',dict(passed=True,rows=rows,model_unchanged=True,model_sha256=before['controller'],world_sha256=before['world'],step=a.step,mode=a.mode,world_action=a.world_action,noise_coupling=a.noise_coupling,
        render_state=a.render_state,policy_objective=controller.objective,policy_solver=a.policy_solver,policy_steps=a.policy_steps,policy_draw=a.policy_draw,policy_field_passes_per_call=a.policy_steps*(2 if a.policy_solver=='midpoint'else 1)if controller.objective=='flow'else 1,
        independent_action_latents_per_sensor_interval=(1 if a.noise_coupling=='observation'else 5)if controller.objective=='flow'else 0,
        categorical_logistic_draws_per_sensor_interval=5 if controller.objective=='mixture'and a.policy_draw=='stochastic'else 0,
        feedback_native=5,action_native=1,training_action_native=controller.action_dim//4,new_updates=0,current_action_only=controller.action_dim==4,
        scope='Frozencontrolleratdeclaredstep,declaredtraining_action_native1or5.Allscratchoriginaldata;noexternalteacher.'
              'The4DcontrollerusesONLYcurrentnativea_tlabels;20Dmodelsareexplicit5-control-targetcomparators.World20kloadedONLYforcopy/worldmodes.'
              'RealRGBONLYevery5native;fourintermediateobservationsareOWNforecast,neveractualfutureframes/states/actions.'
              'CopycontrolgetsSAME128->224intermediateimageprocessinganddeclaredown-noisecouplingasworld;onlyforwardevolutiondisabled.'
              'ObservationcouplingreusesONEownGaussianlatentthrough5imagineddecisions,resamplesaftereachREALcameraimage;nativemoderesamplesperdecision.Noadditionaltuningparameter.'
              'StaticmodeusesONLYoriginalactual224RGBthroughoutthe5nativeslot;noworldloaded/resize/proxyimage.'
              'World-actionzero/negateareexplicitNEGATIVEdiagnostics:onlyimagineddynamicsusesfalseowncontrol;actualphysicalcontrolsarenotchanged.'
              'Macro20modelsarelabelledcurrent5-controlTARGETcomparators,onlytheirFIRSTnativecomponentusedforthisstochasticclocktest.'
              'Extra5xpolicycallsandworldpretrainingcost(ifworldmode)explicit;notan equalFLOPs claim.No newSGD,smallpilotnotconfirmation.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in['run','manifest','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--world',type=Path);p.add_argument('--mode',choices=['copy','world','static'],required=True);p.add_argument('--budget',type=int,default=500)
    p.add_argument('--world-action',choices=['own','zero','negate'],default='own')
    p.add_argument('--step',type=int,default=40000)
    p.add_argument('--noise-coupling',choices=['native','observation'],default='native')
    p.add_argument('--policy-draw',choices=['stochastic','component-median'],default='stochastic')
    p.add_argument('--render-state',choices=['sequential','composed128','composed224'],default='sequential')
    p.add_argument('--policy-solver',choices=['euler','midpoint'],default='euler');p.add_argument('--policy-steps',type=int,default=12)
    p.add_argument('--tasks',nargs='+',default=TASKS);p.add_argument('--seeds',type=int,nargs='+',default=[42,1042]);run(p.parse_args())
