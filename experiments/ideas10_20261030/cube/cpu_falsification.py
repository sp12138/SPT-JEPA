"""CPU-only idea checks. Frozen validation-bank fits are exploratory cross-fitting.

No checkpoints, original pools, or scores are modified. No physical labels enter fits.
Cross-fit labels are actual encoded outcome costs from existing validation cases;
therefore these numbers MUST NOT be called a new held-out benchmark result.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def synthetic():
    # Two independent Gaussian source coordinates, each variance one.
    # R_i=.5 log(1/D_i), total R=1 nat. Large field occupies 99% of pixels.
    allocations = np.linspace(0, 1, 10001)
    distortion = np.stack([np.exp(-2 * allocations), np.exp(-2 * (1-allocations))], 1)
    avg = .99 * distortion[:, 0] + .01 * distortion[:, 1]
    worst = distortion.max(1)
    a, w = int(np.argmin(avg)), int(np.argmin(worst))
    # Backward support of f(s)=s^2 at goal 1 is {-1,+1}, not mean zero.
    predecessor = [-1., 1.]
    mean = float(np.mean(predecessor))
    # Two equal-norm prediction errors give opposite goal-cost regret signs.
    pred, goal = 1., 0.
    deltas = [-.5, .5]
    regret = [(pred+d-goal)**2-(pred-goal)**2 for d in deltas]
    # Closure under reversible translations on a 9-position chain.
    connected = {0}
    for _ in range(9):
        connected |= {s+a for s in list(connected) for a in [-1,1] if 0<=s+a<9}
    return dict(
        rate_allocation=dict(total_nats=1, area_weights=[.99,.01],
            average_optimal_rate=[float(allocations[a]),float(1-allocations[a])],
            average_optimal_distortion=distortion[a].tolist(),
            minimax_optimal_rate=[float(allocations[w]),float(1-allocations[w])],
            minimax_optimal_distortion=distortion[w].tolist()),
        inverse_support=dict(predecessors=predecessor,backward_mean=mean,
            mean_forward_outcome=mean**2,goal=1),
        signed_error=dict(error_norm_squared=[d*d for d in deltas],
            exact_regret=regret,first_order=[2*(pred-goal)*d for d in deltas]),
        orbit=dict(n_positions=9,n_positions_merged=len(connected),
            uniform_goal_identification_upper_bound=1/len(connected)))


def fit_predict(x,y,x_test):
    mean=x.mean(0); sd=np.maximum(x.std(0),1e-6)
    a=np.column_stack([np.ones(len(x)),(x-mean)/sd])
    b=np.column_stack([np.ones(len(x_test)),(x_test-mean)/sd])
    penalty=np.eye(a.shape[1]);penalty[0,0]=0
    return b@np.linalg.solve(a.T@a+penalty,a.T@y)


def summarize(y):
    return dict(mean=float(np.mean(y)),median=float(np.median(y)),
        positive_fraction=float(np.mean(np.asarray(y)>0)))


def bank(run, output):
    results={}
    labels=['temporal-forecast-s1','temporal-forecast-s2','lewm']
    for label in labels:
        cases=[]
        for case in range(50):
            s=np.load(run/f'score-{label}'/f'scores-case{case:04d}.npz')
            p=np.load(run/'pool'/f'case{case:04d}.npz')
            c=s['predicted_cost'][:,-1].astype(float)
            t=s['encoded_cost'][:,-1].astype(float)
            m=s['rollout_mse'][:,-1].astype(float)
            family=s['family']
            nonoracle=np.isin(family,['gaussian','zero'])|np.char.startswith(family,'cem_')
            assert len(c)==39 and nonoracle.sum()==22
            scale=max(float(s['encoded_cost'][0,0]),1e-10)
            z=s['predicted_embeddings'].astype(float)
            act=p['actions'].astype(float)
            motion=((z-z[:,:1])**2).mean(-1)/scale
            rough=((z[:,2:]-2*z[:,1:-1]+z[:,:-2])**2).mean((1,2))/scale
            energy=(act**2).mean((1,2))
            jerk=(np.diff(act,axis=1)**2).mean((1,2))
            # Only quantities available before executing this candidate.
            features=np.stack([np.log1p(c/scale),np.log1p(motion[:,-1]),
                np.log1p(motion[:,1]),np.log1p(motion.mean(1)),
                np.log1p(rough),np.log1p(energy),np.log1p(jerk)],1)
            cases.append(dict(case=case,c=c/scale,t=t/scale,m=m/scale,
                x=features,nonoracle=nonoracle,success=p['success'].any(1),family=family,
                position=p['position_error_m'][:,-1],orientation=p['orientation_error_deg'][:,-1]))
        kinds=['random','cem','winner_all','winner_nonoracle','recorded']
        optimism={k:[] for k in kinds};direction={k:[] for k in kinds};norm={k:[] for k in kinds}
        cross=[]
        for fold in range(5):
            train=[c for c in cases if c['case']%5!=fold]
            test=[c for c in cases if c['case']%5==fold]
            tx=np.concatenate([c['x'][c['nonoracle']] for c in train])
            ty=np.concatenate([(c['t']-c['c'])[c['nonoracle']] for c in train])
            tm=np.concatenate([c['m'][c['nonoracle']] for c in train])
            for c in test:
                rc=fit_predict(tx,ty,c['x'])
                uc=fit_predict(tx,tm,c['x'])
                for group,mask in [('all',np.ones(39,bool)),('nonoracle',c['nonoracle'])]:
                    ids=np.flatnonzero(mask)
                    methods={'original':c['c'],'signed_regret_ridge':c['c']+rc,
                        'norm_squared_ridge':c['c']+np.maximum(uc,0),'true_cost_oracle':c['t']}
                    record={'case':c['case'],'fold':fold,'group':group}
                    for method,cost in methods.items():
                        i=int(ids[np.argmin(cost[ids])])
                        record[method]=dict(index=i,success=bool(c['success'][i]),
                            encoded_regret=float(c['t'][i]-c['t'][ids].min()),
                            true_cost=float(c['t'][i]),position_m=float(c['position'][i]),
                            orientation_deg=float(c['orientation'][i]))
                    cross.append(record)
        for c in cases:
            d=c['t']-c['c']
            groups={'random':np.flatnonzero(c['family']=='gaussian'),
                'cem':np.flatnonzero(np.char.startswith(c['family'],'cem_')),
                'winner_all':[int(np.argmin(c['c']))],
                'winner_nonoracle':[int(np.flatnonzero(c['nonoracle'])[np.argmin(c['c'][c['nonoracle']])])],
                'recorded':[0]}
            for kind,idx in groups.items():
                optimism[kind].append(float(d[idx].mean()))
                direction[kind].append(float((d-c['m'])[idx].mean()))
                norm[kind].append(float(c['m'][idx].mean()))
        aggregate={}
        for group in ['all','nonoracle']:
            rows=[r for r in cross if r['group']==group]
            aggregate[group]={m:dict(success_pct=100*np.mean([r[m]['success'] for r in rows]),
                encoded_regret=np.mean([r[m]['encoded_regret'] for r in rows]),
                position_cm=100*np.mean([r[m]['position_m'] for r in rows]),
                orientation_deg=np.mean([r[m]['orientation_deg'] for r in rows]),
                changed_vs_original=sum(r[m]['index']!=r['original']['index'] for r in rows))
                for m in ['original','signed_regret_ridge','norm_squared_ridge','true_cost_oracle']}
        results[label]=dict(optimism={k:summarize(v) for k,v in optimism.items()},
            directional_term={k:summarize(v) for k,v in direction.items()},
            norm_term={k:summarize(v) for k,v in norm.items()},crossfit=aggregate,cases=cross)
    output.write_text(json.dumps(dict(scope='CPU exploratory diagnostic on already-used validation bank; not a new benchmark. Fits use encoded outcomes only; physical labels are reporting only.',
        protocol='5 folds case_index modulo 5; fit 40 cases x 22 nonoracle candidates; ridge 1, fixed 7 pre-execution features, no tuning. No uncertainty/calibration guarantee.',
        models=results),indent=2))
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'synthetic.json').write_text(json.dumps(synthetic(),indent=2))
    if a.run:
        r=bank(a.run,a.output/'frozen-regret.json')
        print(json.dumps({k:{'optimism':v['optimism'],'crossfit':v['crossfit']} for k,v in r.items()},indent=2))


if __name__=='__main__':main()
