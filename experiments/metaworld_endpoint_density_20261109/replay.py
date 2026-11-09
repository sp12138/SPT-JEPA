"""Selected same-parent Drawer success/failure replay; NEVER training data."""
from pathlib import Path
import traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from experiments.metaworld_transition_only_20261109.common import BASE,read,save,sha,now
from experiments.metaworld_eval_20261103.environment import make_environment,pixels,physics_state,rgb_replay_check,close_environment

ORIGINAL=BASE/'runs/metaworld_endpoint_density_20261109'
ROOT=ORIGINAL/'selected-drawer-replay'


if __name__=='__main__':
    ROOT.mkdir(exist_ok=False);records=[];pictures=[]
    save(ROOT/'AUDIT_ONLY.json',dict(candidate_performance=False,training_data_eligible=False,
        model_updates=0,nn_loads=0,reason='SelecteddevelopmentalDrawer6042/parentseed2;forensicreplayonly. Neverlearn/proposefromthesetraces/images.'))
    try:
        case=next(c for c in read(BASE/'runs/metaworld_transition_only_20261109/manifests/screen.json')['rows'] if c['task']=='drawer-open-v3')
        cached=np.load(case['path']);assert sha(case['path'])==case['sha256']
        for mode in ['mean','uniform','highnoise']:
            meta=read(ORIGINAL/f'evaluation/{mode}-seed2-s2000-drawer-open-v3/metrics.json');row=meta['episodes'][0]
            assert row['case_sha256']==case['sha256'] and sha(row['trace'])==row['trace_sha256']
            source=np.load(row['trace']);actions=source['actions'];env,state,_=make_environment(case['task'],case['seed'],cached['rand_vec'])
            assert np.array_equal(state,cached['initial_state'])
            assert all(np.array_equal(v,cached['initial_physics_'+k]) for k,v in physics_state(env).items())
            gate=rgb_replay_check(cached['initial'],pixels(env));assert gate['passed']
            frames={0:cached['initial'].copy()};flags=[];observations=[]
            try:
                for t,action in enumerate(actions,1):
                    obs,_,_,_,info=env.step(action.astype(np.float32));flags.append(bool(info['success']));observations.append(obs.copy())
                    if t in [50,100,138,152,500]:frames[t]=pixels(env)
                assert np.array_equal(np.asarray(flags),source['successes'])
            finally:close_environment(env)
            target=ROOT/f'{mode}-measurements.npz';np.savez_compressed(target,actions=actions,observations=np.asarray(observations),successes=np.asarray(flags))
            records.append(dict(mode=mode,source_trace=row['trace'],source_trace_sha256=row['trace_sha256'],
                native_steps=len(actions),success=bool(any(flags)),initial_physics_exact=True,initial_rgb_gate=gate,
                all_success_flags_replayed_exact=True,measurements=str(target),measurements_sha256=sha(target)))
            pictures.append(frames)
        tile=224;left=160;top=40;height=265;canvas=Image.new('RGB',(left+6*tile,top+3*height),'white');draw=ImageDraw.Draw(canvas)
        try:font=ImageFont.truetype('DejaVuSans.ttf',16)
        except OSError:font=ImageFont.load_default()
        for j,label in enumerate(['initial','native50','native100','native138','native152','goal']):draw.text((left+j*tile+5,8),label,fill='black',font=font)
        for i,(row,frames) in enumerate(zip(records,pictures)):
            y=top+i*height;draw.multiline_text((5,y+30),row['mode']+'\nparentseed2\nsuccess='+str(row['success'])+'\nsteps='+str(row['native_steps']),fill='black',font=font,spacing=8)
            for j,t in enumerate([0,50,100,138,152]):
                x=left+j*tile
                if t in frames:canvas.paste(Image.fromarray(frames[t]).resize((tile,tile)),(x,y))
                else:draw.text((x+15,y+80),'episode ended\nat '+str(row['native_steps']),fill='black',font=font)
            canvas.paste(Image.fromarray(cached['goal']).resize((tile,tile)),(left+5*tile,y))
        figure=ROOT/'drawer-successes-vs-mean.png';canvas.save(figure)
        save(ROOT/'SUMMARY.json',dict(passed=True,utc=now(),records=records,figure=str(figure),figure_sha256=sha(figure),
            native_steps=sum(r['native_steps'] for r in records),model_updates=0,nn_loads=0,gpu_cuda_contexts=0,
            scope='Selectedonephysicalcase/sameoriginalparentseed2. Bothflowpoliciesphysicallysucceed;meanfails. '
                'Successflagsreplayedexact. Notindependenttraining-seedconfirmation orhighperformanceclaim. ProtectedartifactsNEVERtraining/proposals.'))
    except BaseException:
        save(ROOT/'ERROR.json',dict(utc=now(),traceback=traceback.format_exc()));raise
