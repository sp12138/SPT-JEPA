"""Audit one success and matched failures; NEVER use these paths as supervision."""
from pathlib import Path
import traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,now
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

ROOT=BASE/'runs/metaworld_success_failure_audit_20261109'
OLD=BASE/'runs/metaworld_transition_only_20261109'
CASES=[('h15-inverse1-seed1',60,15),('h15-inverse1-seed1',30,15),
       ('h15-inverse1-seed2',60,15),('h15-inverse0-seed1',60,15)]


def run():
    ROOT.mkdir(exist_ok=False)
    save(ROOT/'AUDIT_ONLY_NO_TRAINING.json',dict(utc=now(),reason='SuccessfulandfailedEVALUATIONtraces areONLYforanalysis/visualization. '
        'Neveruseasactiontargets,trainingframes,goalpathteachers,retrieval/proposals orfuture-actionreference.0NNupdates/0CUDAcontexts;EGLrenderingonly.'))
    bank=read(OLD/'manifests/final.json')['rows'];case=next(c for c in bank if c['task']=='push-v3' and c['seed']==6042)
    assert sha(case['path'])==case['sha256'];cached=np.load(case['path']);reports=[];images=[]
    for name,horizon,execute in CASES:
        endpoint=OLD/'evaluation'/f'{name}-s10000-H{horizon}-e{execute}'
        meta=read(endpoint/'metrics.json');row=next(e for e in meta['episodes'] if e['task']=='push-v3' and e['seed']==6042)
        assert meta['model_state_before']==meta['model_state_after'] and row['case_sha256']==case['sha256']
        assert sha(row['trace'])==row['trace_sha256'];trace=np.load(row['trace']);actions=trace['actions'];expected=trace['successes']
        env,state,_=make_environment('push-v3',6042,cached['rand_vec'])
        assert np.array_equal(state,cached['initial_state'])
        assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
        gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
        captured={0:cached['initial'].copy()};trajectory=[];flags=[];obj0=np.asarray(env._get_pos_objects()).reshape(-1,3)[0].copy()
        try:
            for t,action in enumerate(actions,1):
                state,_,_,_,info=env.step(action.astype(np.float32));flags.append(bool(info['success']))
                tcp=np.asarray(env.tcp_center).copy();obj=np.asarray(env._get_pos_objects()).reshape(-1,3)[0].copy();target=np.asarray(env._target_pos).copy()
                trajectory.append(np.r_[tcp,obj,np.linalg.norm(tcp-obj),np.linalg.norm(obj-target),np.linalg.norm(obj-obj0),state[3],action[3]])
                if t in [50,100,155,500]:captured[t]=pixels(env)
            assert np.array_equal(np.asarray(flags),expected),'Actualsuccessreplaymismatch'
        finally:close_environment(env)
        values=np.stack(trajectory);dest=ROOT/f'{name}-H{horizon}-e{execute}.npz'
        np.savez_compressed(dest,measurements=values,successes=np.asarray(flags),actions=actions)
        report=dict(name=name,horizon=horizon,execute=execute,case_seed=6042,success=bool(any(flags)),steps=len(flags),
            source_trace=row['trace'],source_trace_sha256=row['trace_sha256'],actual_success_replay_exact=True,initial_physics_exact=True,initial_rgb=gate,
            min_tcp_object_distance_m=float(values[:,6].min()),min_object_native_target_distance_m=float(values[:,7].min()),
            final_object_native_target_distance_m=float(values[-1,7]),max_object_displacement_m=float(values[:,8].max()),
            final_gripper_observation=float(values[-1,9]),mean_gripper_input=float(values[:,10].mean()),
            measurement_columns=['tcp_xyz','object_xyz','tcp_object_distance','native_object_goal_distance','object_displacement','gripper_observation','executed_gripper_input'],
            measurements=str(dest),measurements_sha256=sha(dest))
        reports.append(report);images.append(captured);save(ROOT/'progress.json',dict(rows=reports,utc=now()));print(report,flush=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    tile=224;left=240;rowheight=270;top=42;canvas=Image.new('RGB',(left+6*tile,top+len(images)*rowheight),'white');draw=ImageDraw.Draw(canvas)
    for j,label in enumerate(['initial','native50','native100','native155','native500','goal']):draw.text((left+j*tile+8,10),label,fill='black',font=font)
    for i,(frames,report) in enumerate(zip(images,reports)):
        y=top+i*rowheight;draw.multiline_text((8,y+20),report['name']+'\nH'+str(report['horizon'])+' execute'+str(report['execute'])+'\nsuccess='+str(report['success'])+'\nsteps='+str(report['steps'])+'\nmin object-goal='+f'{100*report["min_object_native_target_distance_m"]:.2f}cm',font=font,fill='black',spacing=8)
        for j,t in enumerate([0,50,100,155,500]):
            x=left+j*tile
            if t in frames:canvas.paste(Image.fromarray(frames[t]).resize((tile,tile)),(x,y))
            else:draw.rectangle((x,y,x+tile-1,y+tile-1),fill='#eeeeee');draw.text((x+15,y+80),'episode ended\nat '+str(report['steps']),fill='black',font=font)
        canvas.paste(Image.fromarray(cached['goal']).resize((tile,tile)),(left+5*tile,y))
    png=ROOT/'success-vs-failures.png';canvas.save(png)
    save(ROOT/'SUMMARY.json',dict(passed=True,utc=now(),rows=reports,figure=str(png),figure_sha256=sha(png),
        cases=1,episodes_replayed=4,native_steps=sum(r['steps'] for r in reports),model_updates=0,gpu_contexts=0,
        scope='OnephysicallymatcheddevelopmentalPushcase,notgeneralizableperformance. Native3DstateONLYdiagnosis/plotannotations;neverlearner/plannerinput. '
            'Rowsafterepisodeendareblank,nofabricatedfutureframes. This auditsrecordedowncontrols,notexpert/optimalactionlabels.'))


if __name__=='__main__':
    try:run()
    except BaseException:
        save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
