"""No-NN/no-SGD temporal ordering audit of cached frozen factual features."""
import json
from pathlib import Path

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'


def main():
    sample=np.load(ROOT/'samples.npz');episodes=sample['original_episodes'];split=sample['splits'];tasks=sample['tasks']
    groups={}
    for name in ['smwm-seed1','h-jepa-seed2','smwm-seed1-epoch200','h-jepa-seed2-epoch200']:
        features=np.load(ROOT/name/'features.npz');result={}
        for field in ['embedding','backbone_cls','patch_grid2','raw_rgb32']:
            z=features[field][:len(episodes)].astype(np.float64);rows=[]
            for ep in np.unique(episodes[split==2]):
                index=np.flatnonzero((episodes==ep)&(split==2))
                if len(index)<4:continue
                # Goal itself is excluded from comparisons to avoid a trivial
                # guaranteed zero-distance ordering at the terminal frame.
                distances=np.square(z[index[:-1]]-z[index[-1]]).sum(1)
                comparisons=[float(distances[i]>distances[j])+.5*float(distances[i]==distances[j])
                    for j in range(1,len(distances)) for i in range(j)]
                rows.append(dict(original_episode=int(ep),task=int(tasks[index[0]]),comparisons=len(comparisons),
                    temporal_order_accuracy=float(np.mean(comparisons))))
            assert len({r['task'] for r in rows})==50
            by_task={str(t):float(np.mean([r['temporal_order_accuracy'] for r in rows if r['task']==t])) for t in range(50)}
            result[field]=dict(macro50_temporal_order_accuracy=float(np.mean(list(by_task.values()))),
                original_test_trajectories=len(rows),task_values=by_task,rows=rows)
        groups[name]=result
    out=ROOT/'temporal-order.json';out.write_text(json.dumps(dict(passed=True,groups=groups,neural_inference=0,worldmodel_updates=0,
        scope='Factual demonstration temporal ordering with heldout originaltrajectories;not shortest-path distance,action ranking,or controlsuccess. '
            'Atmost4sampledphase0frames/trajectory,goalterminalexcludedfromcomparisons. '
            'No state/action/reward labels used. Orderedpairs within anepisodearenot independentcases.'),indent=2)+'\n')
    print(json.dumps({n:{f:round(v['macro50_temporal_order_accuracy'],4) for f,v in r.items()} for n,r in groups.items()}))


if __name__=='__main__':main()
