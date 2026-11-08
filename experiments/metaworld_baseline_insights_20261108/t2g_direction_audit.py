"""Paired start-anchor vs future-goal inequalities on identical RGB triples."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_t2g_adapted_20261108'


def main(root=ROOT,verify_public_forward=True):
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');assert not torch.cuda.is_initialized()
    source=BASE/'reports/metaworld_baseline_insights_20261108/t2g-source-572784c/temporal_metric.py'
    spec=importlib.util.spec_from_file_location('t2g_direction',source);public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    results={};identity=None
    for name in ['smwm-seed1','h-jepa-seed2']:
        out=root/name;proof=json.loads((out/'summary.json').read_text());assert proof['passed']
        values=torch.from_numpy(np.load(out/'rgb_features.npz')['embedding']);clips=[];originals=set()
        for e in json.loads((out/'episodes.json').read_text())['episodes']:
            if e['split']!=2 or e['length']<4:continue
            originals.add(e['original_episode'])
            for start in range(e['length']-3):clips.append((e['offset']+start,min(16,e['length']-start)))
        state_path=out/'metric-state.pt';assert hashlib.sha256(state_path.read_bytes()).hexdigest()==proof['metric_state_sha256']
        state=torch.load(state_path,weights_only=True,map_location='cpu');metric=public.TemporalMetric(values.shape[-1])
        metric.load_state_dict(state,strict=True);metric.eval().requires_grad_(False);generator=torch.Generator().manual_seed(1002)
        inequalities={k:[] for k in ['raw_start','raw_goal','t2g_start','t2g_goal','equal_temporal_gap']};triples=[]
        with torch.inference_mode():
            for start in range(0,len(clips),128):
                groups={};parts={k:[] for k in ['anchor','near','far','k_near','k_far']}
                for offset,length in clips[start:start+128]:groups.setdefault(length,[]).append(offset)
                for length,offsets in groups.items():
                    index=torch.tensor(offsets)[:,None]+torch.arange(length)[None,:]
                    p=public.make_temporal_pairs(values[index],max_steps=15,min_steps=1,gap=2,generator=generator,sampling='legacy')
                    for k in parts:parts[k].append(p[k])
                    anchors=torch.tensor(offsets)+p['t'];triples.extend(torch.stack([anchors,anchors+p['k_near'],anchors+p['k_far']],dim=1).tolist())
                # Match the original fitter's concatenated128-example head
                # batches,including numerical ties;do not infer per group.
                p={k:torch.cat(v) for k,v in parts.items()};a,n,g=p['anchor'],p['near'],p['far']
                ra=(a-n).norm(dim=-1);rg=(a-g).norm(dim=-1);ng=(n-g).norm(dim=-1)
                ma=metric(a,n);mg=metric(a,g);mn=metric(n,g)
                for key,good in [('raw_start',ra<rg),('raw_goal',ng<rg),('t2g_start',ma<mg),('t2g_goal',mn<mg),('equal_temporal_gap',p['k_near']==p['k_far']-p['k_near'])]:
                    inequalities[key].extend(good.tolist())
        assert len(triples)==proof['heldout']['triples']
        reproduced=float(np.mean(inequalities['t2g_start']))
        if verify_public_forward:
            print(name,'forward_reproduction',reproduced,'original',proof['heldout']['rank_accuracy'],flush=True)
            assert abs(reproduced-proof['heldout']['rank_accuracy'])<1e-12
        current=hashlib.sha256(np.asarray(triples,np.int64).tobytes()).hexdigest()
        if identity is None:identity=current
        assert identity==current and all(torch.equal(v,metric.state_dict()[k]) for k,v in state.items())
        same=np.asarray(inequalities['equal_temporal_gap']);assert same.sum()>0
        results[name]=dict(triples=len(triples),unique_original_test_trajectories=len(originals),triples_sha256=current,
            all_triples={k:float(np.mean(v)) for k,v in inequalities.items() if k!='equal_temporal_gap'},
            equal_gap_triples=int(same.sum()),equal_gap={k:float(np.mean(np.asarray(v)[same])) for k,v in inequalities.items() if k!='equal_temporal_gap'})
    assert not torch.cuda.is_initialized()
    d=dict(passed=True,evaluated_head_root=str(root),verify_public_forward=verify_public_forward,results=results,worldmodels_loaded=0,worldmodel_updates=0,head_updates=0,gpu_context_created=False,new_physical_episodes=0,
        scope='SAME3541heldoutRGBtriplesandpublicsamplingseedasoriginalheadtest. Startinequalityd(a,n)<d(a,g);goalinequalityd(n,g)<d(a,g),a<n<g. '
            'Equal-gapsubsetcontrolsleft/righttemporalspacing. Thisisolatesanchordirectiononthesametriples,notphysicalcausality/controlsuccess. '
            'No action/state/reward/tasklabelsused;frozenheadonlyNNinference;within-trajectorytriplescorrelated.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root/'direction-audit.json').open('x') as f:json.dump(d,f,indent=2)
    print(json.dumps(results))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--goal-head',action='store_true')
    a=p.parse_args();main(a.root,not a.goal_head)
