"""Goal-terminal ordering of cached factual features;frozen head inference only."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

BASE=Path('/efs/prescottsun/SPT-JEPA')
SCENE=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'
ROOT=BASE/'runs/metaworld_t2g_adapted_20261108'


def main(root=ROOT):
    torch.set_num_threads(8);assert not torch.cuda.is_initialized()
    source=BASE/'reports/metaworld_baseline_insights_20261108/t2g-source-572784c/temporal_metric.py'
    spec=importlib.util.spec_from_file_location('t2g_goal_audit',source);public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    meta=np.load(SCENE/'samples.npz');episodes=meta['original_episodes'];split=meta['splits'];tasks=meta['tasks'];result={}
    for name in ['smwm-seed1','h-jepa-seed2']:
        proof=json.loads((root/name/'summary.json').read_text());assert proof['passed']
        path=root/name/'metric-state.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==proof['metric_state_sha256']
        state=torch.load(path,map_location='cpu',weights_only=True)
        metric=public.TemporalMetric(state['encoder.0.weight'].shape[1]);metric.load_state_dict(state,strict=True);metric.eval().requires_grad_(False)
        z=np.load(SCENE/name/'features.npz')['embedding'][:len(episodes)];rows=[];sources=[];goals=[];groups=[]
        for ep in np.unique(episodes[split==2]):
            index=np.flatnonzero((episodes==ep)&(split==2))
            if len(index)<4:continue
            groups.append((int(ep),index,len(sources)));sources.extend(index[:-1]);goals.extend([index[-1]]*(len(index)-1))
        with torch.inference_mode():distances=metric(torch.from_numpy(z[sources]),torch.from_numpy(z[goals])).numpy()
        for ep,index,start in groups:
            raw=np.square(z[index[:-1]].astype(np.float64)-z[index[-1]].astype(np.float64)).sum(1)
            learned=distances[start:start+len(index)-1]
            def accuracy(d):return float(np.mean([float(d[i]>d[j])+.5*float(d[i]==d[j]) for j in range(1,len(d)) for i in range(j)]))
            rows.append(dict(original_episode=ep,task=int(tasks[index[0]]),raw=accuracy(raw),t2g=accuracy(learned)))
        by_task={str(t):{k:float(np.mean([r[k] for r in rows if r['task']==t])) for k in ['raw','t2g']} for t in range(50)}
        assert all(torch.equal(v,metric.state_dict()[k]) for k,v in state.items())
        result[name]=dict(original_test_trajectories=len(rows),task_values=by_task,rows=rows,
            macro50={k:float(np.mean([r[k] for r in by_task.values()])) for k in ['raw','t2g']})
    assert not torch.cuda.is_initialized()
    d=dict(passed=True,evaluated_head_root=str(root),results=result,head_optimizer_updates=0,worldmodel_optimizer_updates=0,worldmodels_loaded=0,gpu_context_created=False,new_physical_episodes=0,
        scope='Same4cachedphase0factualframes/250heldoutoriginaltrajectoriesastemporal-order.json. '
            'Terminalfuturegoalexcludedfromcomparisons;only3preterminalstatescompared,NOTtrivialgoalzero-distancepairs. '
            'ThisisGOAL-terminalordering,differentfromtrainingforward-anchortripletstatistic. No control/counterfactual/shortestpathproof. '
            'No state/action/reward/tasklabelsintoNN;taskmetadataONLYposthocmetricaggregation;frozenownheadinferenceonly.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (root/'goal-order-audit.json').open('x') as f:json.dump(d,f,indent=2)
    print(json.dumps({k:v['macro50'] for k,v in result.items()}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();main(a.root)
