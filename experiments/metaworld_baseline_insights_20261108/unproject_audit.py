"""Fit an internal encoder-geometry map;no privileged labels or new NN SGD.

Targets are the SAME frozen encoder's own pre-projector CLS,not physical state,
teacher features,pixels,actions,rewards,or task labels. This is a baseline cost
diagnostic,not a new sample-level world-model or a novelty claim.
"""
import json
from pathlib import Path

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
SOURCE=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'
ROOT=BASE/'runs/metaworld_goal_geometry_20261108'


def ordering(z,episodes,mask):
    scores=[]
    for ep in np.unique(episodes[mask]):
        index=np.flatnonzero((episodes==ep)&mask)
        if len(index)<4:continue
        d=np.square(z[index[:-1]]-z[index[-1]]).sum(1)
        scores.append(np.mean([float(d[i]>d[j])+.5*float(d[i]==d[j]) for j in range(1,len(d)) for i in range(j)]))
    return float(np.mean(scores)),len(scores)


def main():
    ROOT.mkdir(exist_ok=False)
    # Deliberately access ONLY split and original-episode metadata here.
    meta=np.load(SOURCE/'samples.npz');splits=meta['splits'];episodes=meta['original_episodes'];n=len(episodes)
    fit=splits==0;validation=splits==1;test=splits==2;reports={}
    for name in ['smwm-seed1','h-jepa-seed2']:
        frozen=json.loads((SOURCE/name/'FROZEN.json').read_text());assert frozen['passed']
        features=np.load(SOURCE/name/'features.npz');x=features['embedding'][:n].astype(np.float64);y=features['backbone_cls'][:n].astype(np.float64)
        xm=x[fit].mean(0);xs=np.maximum(x[fit].std(0),1e-6);ym=y[fit].mean(0);a=(x-xm)/xs
        gram=a[fit].T@a[fit]/sum(fit);cross=a[fit].T@(y[fit]-ym)/sum(fit)
        eigen,vec=np.linalg.eigh(gram);projected=vec.T@cross;best=None
        for alpha in [1e-6,1e-4,1e-2,1.]:
            w=vec@(projected/(np.maximum(eigen,0)[:,None]+alpha));prediction=a[validation]@w+ym
            error=float(np.square(prediction-y[validation]).mean())
            if best is None or error<best[0]:best=error,alpha,w
        error,alpha,w=best;prediction=a@w+ym
        out=ROOT/name;out.mkdir()
        # This RGB-derived cost map is isolated from all privileged state probes.
        np.savez_compressed(out/'encoder_geometry_metric.npz',linear=(w/xs[:,None]).astype(np.float32))
        baseline=float(np.square(y[test]-ym).mean());test_error=float(np.square(prediction[test]-y[test]).mean())
        reports[name]=dict(passed=True,alpha=alpha,internal_feature_test_relative_mse=test_error/max(baseline,1e-20),
            temporal_order_test={k:ordering(z,episodes,test) for k,z in [('projected',x),('backbone',y),('recovered_backbone',prediction)]},
            frozen_weights_sha256=frozen['weights_sha256'],worldmodel_sgd=0,
            target='Sameimage/ownencoderbackboneCLS. NO physicalstate/taskID/reward/action/pixel/teacher targets.',
            scope='Validationselectsinternal-featureMSE,notcontrolsuccess/temporaltest. Testdevelopmentalpreviouslyinspected. '
                'Costmapnotdeployed/controlnotyettested. OriginalNNparentsnativebatchregs,soNOTanentirelysample-levelcandidate. '
                'Affineinverseprojector/metricmatchingisnotclaimednovel.')
        (out/'summary.json').write_text(json.dumps(reports[name],indent=2)+'\n')
    (ROOT/'summary.json').write_text(json.dumps(dict(passed=True,reports=reports,model_updates=0,neural_inference=0,
        accessed_training_fields=['frozen_embedding','same_encoder_backbone_cls','split_codes','original_episode_ids']),indent=2)+'\n')
    print(json.dumps(reports))


if __name__=='__main__':main()
