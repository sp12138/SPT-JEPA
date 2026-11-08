"""Protected-audit visualization, not training data or an action library."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .probe import forecast,realize
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,tensor_sha,now
from experiments.metaworld_transition_only_20261109.evaluate import rgb_tensor
from experiments.metaworld_horizon_match_20261109.model import build
from experiments.metaworld_horizon_match_20261109.bank import capture,byte
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment
from experiments.metaworld_world_vs_planner_20261109.oracle import restore_exact

ROOT=BASE/'runs/metaworld_gradient_grounding_20261109/ieee-v2'


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.30)
    case=next(c for c in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'] if c['task']=='push-v3')
    cached=np.load(case['path']);assert sha(case['path'])==case['sha256'];rows=[]
    for seed in [1,2]:
        cfg=read(BASE/f'runs/metaworld_horizon_match_20261109/training/h60-seed{seed}/config.json')
        model=build(cfg).cuda().eval().requires_grad_(False)
        weights=Path(cfg['output'])/'model-step0012000/weights.pt';model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));before=tensor_sha(model)
        record=next(r for r in read(ROOT/f'seed{seed}/DONE.json')['rows'] if r['task']=='push-v3')
        assert sha(record['trace'])==record['trace_sha256'];controls=np.load(record['trace'])['refined_controls']
        env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
        try:
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
            assert rgb_replay_check(cached['initial'],pixels(env))['passed']
            with torch.inference_mode():
                image=rgb_tensor(cached['initial'],model);goal=rgb_tensor(cached['goal'],model)
                hidden=model.observe(model.initial(1),image)
                for _ in range(7):
                    hidden,_=model.predict(hidden,image,torch.zeros(1,4,device='cuda'))
                    env.step(np.zeros(4,np.float32));image=rgb_tensor(pixels(env),model);hidden=model.observe(hidden,image)
                snapshot=capture(env);predicted=forecast(model,hidden,image,torch.from_numpy(controls).cuda())
                scores=(predicted-goal).square().mean((1,2,3)).cpu().numpy()
                assert np.allclose(scores,record['details']['refined_predicted'],atol=1e-7,rtol=0)
                canvas=Image.new('RGB',(4*256,4*290),'white');draw=ImageDraw.Draw(canvas)
                for index,path in enumerate(controls):
                    actual=realize(env,snapshot,path,goal,model);ref=next(r for r in record['results'] if r['kind']=='refined' and r['index']==index)
                    assert abs(actual['rgb_goal_mse']-ref['rgb_goal_mse'])<1e-6
                    terminal=rgb_tensor(pixels(env),model)
                    for column,(label,frame) in enumerate([('Observed start',image),('Predicted 60 steps',predicted[index:index+1]),('Actual 60 steps',terminal),('Goal (planning only)',goal)]):
                        canvas.paste(Image.fromarray(byte(frame)).resize((256,256)),(column*256,index*290))
                        draw.text((column*256+5,index*290+257),label,fill='black')
                    draw.text((5,index*290+274),f'Seed{seed} path{index}: predicted RGB {scores[index]:.5f}, actual RGB {actual["rgb_goal_mse"]:.5f}, object-goal {ref["object_goal_distance_audit_only"]:.3f}m',fill='black')
                target=ROOT/f'push-predicted-vs-actual-seed{seed}.png';canvas.save(target)
                rows.append(dict(seed=seed,path=str(target),sha256=sha(target),case_sha256=case['sha256'],all4_predicted_costs_reproduced=True,all4_actual_costs_reproduced=True))
        finally:close_environment(env)
        assert before==tensor_sha(model)
    save(ROOT/'VISUAL_AUDIT.json',dict(passed=True,utc=now(),rows=rows,model_updates=0,physics_steps=2*(7+4*60),
        candidate_performance=False,training_data_eligible=False,scope='Replays only8 retained audit paths for visualization. Nevertrain or seedproposals fromthese protectedframes/controls.'))
