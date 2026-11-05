"""CPU-only frozen geometry diagnosis; no change to the public planner."""
import argparse
import json
from pathlib import Path
import sys

from common import OLD, bootstrap, digest_state


def run(a):
    dump=bootstrap(OLD)
    import torch
    import numpy as np
    from torch.nn import functional as F
    from bridge import cost_adapter
    from droid_data import DROIDClipReader
    from droid_eval import img_transform, clip_metrics
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    directory=OLD/'training'/f'{a.arm}-seed{a.seed}'
    core=torch.load(directory/'droid_lewm_epoch_100_object.ckpt',map_location='cpu',weights_only=False).eval().requires_grad_(False)
    before=digest_state(core)
    assert before==json.loads((directory/'epoch100-ready.json').read_text())['model_state_sha256']
    model=core.get_level(1) if a.arm=='lewm' else cost_adapter(core)
    stats=core.normalizer_artifact['stats']['action']
    mean=torch.tensor(np.asarray(stats['mean']).reshape(-1)).float()
    std=torch.tensor(np.asarray(stats['std']).reshape(-1)).float()
    manifest=OLD/'public/h_jepa/droid_assets/droid_clips_waypoint_curated16v2_5fps_gw36.json'
    reader=DROIDClipReader(None,['left_mp4_path'],37,5,frozen_clips=str(manifest),deterministic_getitem=True)
    tf=img_transform(256)
    rows=[];cache=[]
    old=json.loads((Path(a.root)/'diagnostics-v2'/f'{a.arm}-seed{a.seed}.json').read_text())
    with torch.no_grad():
        for k in range(16):
            obs,actions,_,_,info=reader[k]
            pixels=tf(obs['visual'])
            z=torch.cat([model.encode(dict(pixels=part[None]))['embed_0'] for part in pixels.split(8)],1)
            factual=((actions[:36]-mean)/std).float()
            variants=dict(factual=factual,clipped=factual.clamp(-2,2),zero=(-mean/std).expand_as(factual),
                          reverse=factual.flip(0).clamp(-2,2))
            values={}
            for label,act in variants.items():
                for h in [1,7,36]:
                    data=dict(pixels=torch.zeros(1,1,1,3,1,1),embed_0=z[:,None,:1],
                              goal_embed_0=z[:,None,h:h+1],action=torch.zeros(1,1,1,7))
                    cost=float(model.get_cost(data,act[None,None,:h]))
                    end=data['predicted_embed_0'][0,0,-1]
                    goal=z[0,h]
                    values[f'{label}_l2_{h}']=cost
                    values[f'{label}_cosine_{h}']=float(1-F.cosine_similarity(end[None],goal[None]))
                    if label in ['zero','factual'] and h==36:
                        values[label+'_end_over_start_norm']=float(end.norm()/z[0,0].norm().clamp_min(1e-30))
            # Clipping factual actions is a posthoc reference, not a supplied
            # policy or an optimal achievable bound for this benchmark.
            clipped_raw=variants['clipped']*std+mean
            clipped_reference=clip_metrics(clipped_raw,actions[:36],info['grasp_pos'])
            comparison={str(h):abs(values[f'factual_l2_{h}']-old['rows'][k]['values'][f'factual_{h}'])/
                        max(abs(old['rows'][k]['values'][f'factual_{h}']),1e-20) for h in [1,7,36]}
            rows.append(dict(clip=k,values=values,cpu_gpu_relative_difference=comparison,
                             clipped_factual_reference=clipped_reference))
            cache.append(dict(clip=k,embedding=z.cpu(),normalized_factual=factual))
            dump(Path(a.output).with_suffix('.progress.json'),dict(arm=a.arm,seed=a.seed,completed=k+1))
    assert digest_state(core)==before
    value=dict(passed=True,arm=a.arm,seed=a.seed,model_state_unchanged=True,model_sha256=before,rows=rows,
               caveat='Retrospective CPU geometry only. Cosine is not a new reported planning score. Clipped factual reference is not an optimal policy or a performance upper bound.')
    target=Path(a.output);dump(target,value)
    torch.save(cache,target.with_suffix('.pt'))
    print(json.dumps(dict(passed=True,arm=a.arm,seed=a.seed,clips=16)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','arm','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--seed',type=int,required=True);run(p.parse_args())
