"""CPU falsification of action-order and hidden-history mechanisms.

Physical states below train diagnostic regressions ONLY. No model/policy is trained.
The common bank is previously used validation data, not a new benchmark.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def fit(x,y,xt):
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-8)
    u=np.column_stack([np.ones(len(x)),(x-mean)/sd])
    v=np.column_stack([np.ones(len(xt)),(xt-mean)/sd])
    penalty=np.eye(u.shape[1]);penalty[0,0]=0
    return v@np.linalg.solve(u.T@u+penalty,u.T@y)


def synthetic():
    def step(x,a):return min(1.,max(0.,x+a))
    result={}
    for x in [.5,.95]:
        ab=step(step(x,.1),-.1);ba=step(step(x,-.1),.1)
        result[str(x)]=dict(ab=ab,ba=ba,defect=ab-ba)
    eps=.1
    # f=(1,0), g=(0,x): exact time-eps flows.
    point=np.array([0.,0.]);f=lambda p:p+np.array([eps,0.]);g=lambda p:p+np.array([0.,eps*p[0]])
    defect=g(f(point))-f(g(point))
    rx=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    ry=np.array([[0,0,1],[0,1,0],[-1,0,0]],float)
    point=np.array([0.,0.,1.]);rotdefect=ry@rx@point-rx@ry@point
    return dict(clipped_boundary=result,smooth_shear=dict(epsilon=eps,defect=defect.tolist(),
        epsilon_squared=eps**2),actuator_without_object=dict(defect=rotdefect.tolist(),
        norm=float(np.linalg.norm(rotdefect))),hidden_velocity=dict(current_x=0,
        velocities=[-1,1],next_positions=[-1,1],best_frame_only_prediction=0,
        irreducible_mse=1.,history_aware_mse=0.,required_binary_memory_nats=float(np.log(2))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cases=[]
    for case in range(50):
        b=np.load(a.bank/f'case{case:04d}.npz')
        q=b['qpos'];pos=b['positions'];actions=b['actions'];family=b['family']
        non=np.isin(family,['gaussian','zero'])|np.char.startswith(family,'cem_')
        x=[];h=[];y=[];mask=[]
        assert q.shape==(39,26,21)
        for t in [1,6,11,16,21]:
            x.append(np.concatenate([q[:,t],actions[:,t:t+4].reshape(39,-1)],1))
            h.append(np.concatenate([x[-1],q[:,t]-q[:,t-1]],1))
            y.append(pos[:,t+4]-pos[:,t]);mask.extend(non)
        cases.append(dict(case=case,x=np.concatenate(x),h=np.concatenate(h),y=np.concatenate(y),mask=np.array(mask)))
    groups={}
    for group in ['all','nonoracle']:
        preds={k:[] for k in ['current_qpos','qpos_plus_one_step_history','no_motion']};truth=[];bycase=[]
        for fold in range(5):
            train=[c for c in cases if c['case']%5!=fold];test=[c for c in cases if c['case']%5==fold]
            m=lambda c:np.ones(len(c['y']),bool) if group=='all' else c['mask']
            y=np.concatenate([c['y'][m(c)] for c in train])
            for c in test:
                mc=m(c);yt=c['y'][mc];row={'case':c['case'],'fold':fold,'n':len(yt)}
                for name,k in [('current_qpos','x'),('qpos_plus_one_step_history','h')]:
                    xp=np.concatenate([t[k][m(t)] for t in train]);yp=fit(xp,y,c[k][mc]);preds[name].append(yp)
                    row[name+'_mse']=float(np.square(yp-yt).mean())
                preds['no_motion'].append(np.zeros_like(yt));truth.append(yt);bycase.append(row)
        yt=np.concatenate(truth);sst=np.square(yt-yt.mean(0)).sum(0)
        groups[group]={'n':len(yt),'cases':bycase,'methods':{}}
        for name,values in preds.items():
            yp=np.concatenate(values);err=np.square(yp-yt)
            groups[group]['methods'][name]=dict(mse=float(err.mean()),
                coordinate_r2=(1-err.sum(0)/sst).tolist(),
                mean_displacement_error_cm=float(np.linalg.norm(yp-yt,axis=1).mean()*100))
    r=dict(scope='Physical-state diagnostic only. No encoder/policy training; no new benchmark claim.',
        protocol='50 held validation cases, 39 candidates, times1/6/11/16/21, horizon4 raw steps; 5 folds case%5, fixed ridge1; baseline full qpos plus exact upcoming four actions, history adds qpos[t]-qpos[t-1].',
        synthetic=synthetic(),history=groups)
    a.output.write_text(json.dumps(r,indent=2))
    print(json.dumps({'synthetic':r['synthetic'],'history':{k:{'n':v['n'],'methods':v['methods']} for k,v in groups.items()}},indent=2))


if __name__=='__main__':main()
