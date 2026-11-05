"""Render saved MetaWorld actions; no model, planner, SGD or new score cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.metaworld_eval_20261103.environment import (
    make_environment, pixels, physics_state, rgb_replay_check, close_environment,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_checked(env):
    before=physics_state(env);clock=float(env.data.time)
    rng=json.dumps(env.np_random.bit_generator.state,sort_keys=True)
    rgb=pixels(env)
    assert float(env.data.time)==clock
    assert json.dumps(env.np_random.bit_generator.state,sort_keys=True)==rng
    assert all(np.array_equal(v,physics_state(env)[k]) for k,v in before.items())
    return rgb


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def label_frame(actual,goal,arm,step,total,success):
    canvas=Image.new('RGB',(960,532),'#101827');draw=ImageDraw.Draw(canvas)
    canvas.paste(Image.fromarray(actual),(0,52));canvas.paste(Image.fromarray(goal),(480,52))
    name='Temporal-forecast' if arm=='tf-control' else 'Native LeWM'
    draw.text((12,7),f'{name} | step {step}/{total}',font=font(19),fill='white')
    draw.text((12,31),f'Replayed execution | success={int(success)}',font=font(13),fill='#8de6ad' if success else '#ffb599')
    draw.text((493,10),'Goal RGB (cached expert reference)',font=font(20),fill='white')
    return canvas


def run(a):
    import imageio.v2 as imageio
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    selection=json.loads(Path(a.selection).read_text())
    manifest=json.loads(Path(a.manifest).read_text())['rows']
    proofs=[]
    for case in selection['cases']:
        task,seed=case['task'],case['seed']
        row=next(c for c in manifest if c['task']==task and c['seed']==seed)
        assert sha(case['trace'])==case['trace_sha256'] and sha(row['path'])==row['sha256']
        trace=np.load(case['trace']);cached=np.load(row['path'])
        assert row['expert_any_success']
        env,state,_=make_environment(task,seed,cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        directory=root/f'{case["arm"]}-{task}';directory.mkdir()
        total=len(trace['actions']);key_steps=sorted(set([0,min(100,total),min(250,total),total]))
        if total<100:key_steps=sorted(set([0,(total//3)//5*5,(2*total//3)//5*5,total]))
        frames=[];frame_steps=[];actual_success=[];diagnostics=[]
        try:
            initial=render_checked(env);rgb_check=rgb_replay_check(cached['initial'],initial)
            assert rgb_check['passed'],rgb_check
            frames.append(initial);frame_steps.append(0)
            Image.fromarray(initial).save(directory/'step000.png')
            initial_object=np.asarray(env._get_pos_objects()).reshape(-1)[:3].copy()
            for step,action in enumerate(trace['actions'],1):
                _,_,terminated,truncated,info=env.step(action.astype(np.float32))
                actual_success.append(bool(info['success']))
                assert actual_success[-1]==bool(trace['successes'][step-1]),(task,step,'saved outcome differs')
                obj=np.asarray(env._get_pos_objects()).reshape(-1)[:3].copy()
                target=np.asarray(env._target_pos).reshape(-1)[:3].copy()
                diagnostics.append(dict(step=step,object_xyz=obj.tolist(),target_xyz=target.tolist(),
                                        object_goal_distance=float(np.linalg.norm(obj-target)),
                                        tcp_xyz=np.asarray(env.tcp_center).tolist(),success=actual_success[-1]))
                if step%5==0 or step==total:
                    rgb=render_checked(env);frames.append(rgb);frame_steps.append(step)
                    if step in key_steps:Image.fromarray(rgb).save(directory/f'step{step:03d}.png')
            assert np.array_equal(np.asarray(actual_success),trace['successes'])
            assert any(actual_success)==case['success']
        finally:
            close_environment(env)
        Image.fromarray(cached['goal']).save(directory/'goal.png')
        labelled=[label_frame(rgb,cached['goal'],case['arm'],step,total,
                              False if step==0 else actual_success[step-1]) for rgb,step in zip(frames,frame_steps)]
        with imageio.get_writer(directory/'replay.mp4',fps=10,codec='libx264',quality=8,
                                macro_block_size=2,ffmpeg_params=['-threads','2']) as video:
            for frame in labelled:video.append_data(np.asarray(frame))
        # Five native actions/frame, 10fps presentation. Hold the last frame for
        # readability; these duplicates do not represent extra simulator steps.
        small=[im.resize((768,426),Image.Resampling.LANCZOS) for im in labelled]
        duration=[100]*len(small);duration[-1]=1200
        small[0].save(directory/'replay.gif',save_all=True,append_images=small[1:],duration=duration,loop=0,optimize=True)
        proof=dict(task=task,arm=case['arm'],case_seed=seed,parent_seed=case['parent_seed'],source_trace=case['trace'],
             source_trace_sha256=case['trace_sha256'],source_case_sha256=row['sha256'],
             steps=total,success=case['success'],all_step_successes_identical=True,initial_physics_exact=True,
             initial_rgb_check=rgb_check,render_preserves_physics_time_rng=True,goal_is_cached_expert_rgb=True,
             model_calls=0,SGD_updates=0,new_benchmark_episodes=0,frame_steps=frame_steps,key_steps=key_steps,
             object_initial=initial_object.tolist(),object_final=diagnostics[-1]['object_xyz'],
             object_max_lift=float(max(d['object_xyz'][2] for d in diagnostics)-initial_object[2]),
             object_final_goal_distance=diagnostics[-1]['object_goal_distance'],
             scope='Exact action replay for visualization. Diagnostic state never enters a learner or planner. Selected examples do not establish population performance.')
        (directory/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        (directory/'diagnostics.json').write_text(json.dumps(diagnostics)+'\n')
        proofs.append(proof)
        print(json.dumps(dict(task=task,arm=case['arm'],success=case['success'],steps=total,
                   max_lift=proof['object_max_lift'],final_object_distance=proof['object_final_goal_distance'])),flush=True)
    (root/'summary.json').write_text(json.dumps(dict(passed=True,selection=selection['selection'],
         cases=proofs,created=time.time(),presentation_fps=10,actions_per_frame=5),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['selection','manifest','output']:p.add_argument('--'+key,required=True)
    run(p.parse_args())
