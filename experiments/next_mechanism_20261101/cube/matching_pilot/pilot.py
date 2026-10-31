"""Fixed CPU cross-fitted compatibility diagnostic; not a benchmark model."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/cube_pool_reuse2000_20261031'
MARKER='DIAGNOSTIC_ONLY_NOT_FOR_BENCHMARK'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def head():return nn.Sequential(nn.Linear(253,192),nn.GELU(),nn.Linear(192,192),nn.GELU(),nn.Linear(192,1))

def load_fit_data(seed,episode_by_case):
    path=ROOT/f'tf12000-seed{seed}/result.json';r=read(path)
    assert r['passed']and r['state_before']==r['state_after'] and len(r['cases'])==50
    cfgpath=Path(r['spec']['config']);cfg=read(cfgpath);cardpath=Path(cfg['data'])/'dataset_card.json';card=read(cardpath)
    mean=np.asarray(card['action_mean']);std=np.asarray(card['action_std']);assert mean.shape==std.shape==(5,) and (std>0).all()
    data=[]
    for rec in r['cases']:
        assert sha(rec['file'])==rec['sha256']
        with np.load(rec['file'])as z:
            # Deliberately never access goal or any physical/success fields here.
            initial=z['initial'].copy();endpoints=z['true_endpoint'].copy();actions=z['raw_actions'].copy()
        assert initial.shape==(64,) and endpoints.shape==(39,64)and actions.shape==(39,25,5)
        actions-=mean;actions/=std
        data.append(dict(case=rec['case'],episode=episode_by_case[rec['case']],initial=initial,endpoints=endpoints,actions=actions.reshape(39,125),file=rec['file'],sha256=rec['sha256']))
    data.sort(key=lambda x:x['episode'])
    assert set(data[0])=={'case','episode','initial','endpoints','actions','file','sha256'}
    return data,dict(result=str(path),result_sha256=sha(path),config=str(cfgpath),config_sha256=sha(cfgpath),card=str(cardpath),card_sha256=sha(cardpath),action_mean=mean.tolist(),action_std=std.tolist(),original_spec=r['spec'])

def inputs(initial,goal,actions,rms):
    n,c=actions.shape[:2]
    assert initial.shape==goal.shape==(n,64)and actions.shape==(n,c,125)
    return torch.cat([(initial/rms)[:,None].expand(n,c,64),((goal-initial)/rms)[:,None].expand(n,c,64),actions],-1)

def fork_eval(model,data,rms):
    ce=[];acc=[];pred_acc=[];observed_acc=[];cases=[]
    with torch.no_grad():
        for d in data:
            initial=torch.from_numpy(d['initial']);endpoints=torch.from_numpy(d['endpoints']);actions=torch.from_numpy(d['actions'])
            logits=[]
            for lo in range(0,39,8):
                g=endpoints[lo:lo+8];x=inputs(initial[None].expand(len(g),64),g,actions[None].expand(len(g),39,125),rms)
                logits.append(model(x.reshape(-1,253)).reshape(len(g),39))
            logits=torch.cat(logits);target=torch.arange(39)
            loss=F.cross_entropy(logits,target,reduction='none');correct=logits.argmax(-1)==target
            # Prediction/encoded fork accuracy are read-only reference scores.
            with np.load(d['file'])as z:pred=torch.from_numpy(z['predicted_endpoint'].copy())
            pc=((endpoints[:,None]-pred[None])**2).sum(-1);ec=((endpoints[:,None]-endpoints[None])**2).sum(-1)
            ce.append(float(loss.mean()));acc.append(float(correct.float().mean()));pred_acc.append(float((pc.argmin(-1)==target).float().mean()));observed_acc.append(float((ec.argmin(-1)==target).float().mean()))
            cases.append(dict(case=d['case'],episode=d['episode'],ce=ce[-1],accuracy=acc[-1],predicted_mse_accuracy=pred_acc[-1],true_encoded_accuracy=observed_acc[-1]))
    return dict(ce=float(np.mean(ce)),accuracy=float(np.mean(acc)),predicted_mse_accuracy=float(np.mean(pred_acc)),true_encoded_accuracy=float(np.mean(observed_acc)),cases=cases)

def contracts(model,d,rms):
    i=torch.from_numpy(d['initial'])[None];g=torch.from_numpy(d['endpoints'][7])[None];a=torch.from_numpy(d['actions'])[None]
    perm=torch.arange(38,-1,-1)
    with torch.no_grad():
        x=inputs(i,g,a,rms).reshape(-1,253);xp=inputs(i,g,a[:,perm],rms).reshape(-1,253);xz=inputs(i,g,torch.zeros_like(a),rms).reshape(-1,253)
        original=model(x).reshape(39);changed=model(xp).reshape(39);zero=model(xz).reshape(39)
        # CPU GEMM's remainder rows can use different arithmetic. Same-shape
        # individual candidates test exact functional independence separately.
        single=torch.cat([model(row[None])for row in x]).reshape(39)
        singlep=torch.cat([model(row[None])for row in xp]).reshape(39)
        singlez=torch.cat([model(row[None])for row in xz]).reshape(39)
    error=float((original[perm]-changed).abs().max());zeroerror=float((zero-zero[0]).abs().max())
    exactperm=float((single[perm]-singlep).abs().max());exactzero=float((singlez-singlez[0]).abs().max())
    record=dict(batched_permutation_max_error=error,batched_zero_action_candidate_max_error=zeroerror,single_candidate_permutation_max_error=exactperm,single_candidate_zero_action_max_error=exactzero,batched_rtol=1e-5,batched_atol=1e-6)
    print(json.dumps(dict(contract=record)),flush=True)
    assert exactperm==0 and exactzero==0
    assert torch.allclose(original[perm],changed,rtol=1e-5,atol=1e-6)and torch.allclose(zero,zero[0].expand_as(zero),rtol=1e-5,atol=1e-6)
    return record

def interval(x):
    x=np.asarray(x,float);ids=np.random.default_rng(1042).integers(len(x),size=(5000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[ids].mean(1),[.025,.975]).tolist(),n=len(x))

def main(out):
    out=Path(out);assert not out.exists();out.mkdir(parents=True)
    started=time.monotonic();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    invpath=B/'reports/cube-pool-reuse-review-20261031/inventory.json';inv=read(invpath)
    episode_by_case={x['case']:x['pair']['episode_id']for x in inv['cases_provenance']};ordered=sorted(episode_by_case.values());assert len(set(ordered))==50
    fold_by_episode={ep:i%2 for i,ep in enumerate(ordered)}
    meta=B/'data/hardtasks-v1/cube-metadata';val=np.load(meta/'validation_episodes.npy');train=np.load(meta/'train_episodes.npy')
    assert np.isin(ordered,val).all()and not np.isin(ordered,train).any()
    write(out/'protocol.json',dict(marker=MARKER,script_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),inventory_sha256=sha(invpath),fold_by_episode=fold_by_episode,updates=600,batch=16,lr=.001,weight_decay=.01,threads=2,max_seconds=900,all_sources_validation=True))
    models=[];sources=[];init_hash={};fit_reports=[]
    for seed in [1,2]:
        data,source=load_fit_data(seed,episode_by_case);sources.append(dict(seed=seed,**source))
        for fold in [0,1]:
            training=[d for d in data if fold_by_episode[d['episode']]!=fold];heldout=[d for d in data if fold_by_episode[d['episode']]==fold]
            assert len(training)==len(heldout)==25 and not ({d['episode']for d in training}&{d['episode']for d in heldout})
            values=np.concatenate([np.concatenate([d['initial'][None],d['endpoints']],0)for d in training]);rms=float(np.sqrt(np.mean(values.astype(np.float64)**2)))
            assert math.isfinite(rms)and rms>0
            runseed=20261101+seed;torch.manual_seed(runseed);rng=np.random.default_rng(runseed);model=head();initialhash=digest(model)
            if seed in init_hash:assert init_hash[seed]==initialhash
            else:init_hash[seed]=initialhash
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            ini=torch.from_numpy(np.stack([d['initial']for d in training]));ends=torch.from_numpy(np.stack([d['endpoints']for d in training]));acts=torch.from_numpy(np.stack([d['actions']for d in training]))
            trace=[];fitstart=time.monotonic();model.train()
            for step in range(1,601):
                if time.monotonic()-started>900:
                    torch.save(dict(marker=MARKER,step=step-1,model=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,fold=fold),out/f'interrupted-seed{seed}-fold{fold}.pt')
                    write(out/'INTERRUPTED.json',dict(marker=MARKER,seed=seed,fold=fold,completed_updates=step-1,seconds=time.monotonic()-started));raise RuntimeError('Fixed900-second ceiling reached; no continuation')
                ri=torch.from_numpy(rng.integers(25,size=16));gi=torch.from_numpy(rng.integers(39,size=16))
                x=inputs(ini[ri],ends[ri,gi],acts[ri],rms);logits=model(x.reshape(-1,253)).reshape(16,39);loss=F.cross_entropy(logits,gi)
                assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                if step==1 or step%100==0:trace.append(dict(step=step,loss=float(loss.detach())));print(json.dumps(dict(seed=seed,fold=fold,**trace[-1])),flush=True)
            model.eval();check=contracts(model,heldout[0],rms)
            for v in model.state_dict().values():assert torch.isfinite(v).all()
            path=out/f'DIAGNOSTIC_ONLY-seed{seed}-fold{fold}.pt';torch.save(dict(marker=MARKER,updates=600,model=model.state_dict(),optimizer=optimizer.state_dict(),rms=rms,seed=seed,fold=fold,runseed=runseed,train_episodes=[d['episode']for d in training],heldout_episodes=[d['episode']for d in heldout]),path)
            fit_reports.append(dict(seed=seed,heldout_fold=fold,updates=600,rms=rms,parameters=sum(p.numel()for p in model.parameters()),initial_hash=initialhash,final_hash=digest(model),checkpoint=str(path),checkpoint_sha256=sha(path),seconds=time.monotonic()-fitstart,trace=trace,contracts=check,train_episodes=[d['episode']for d in training],heldout_episodes=[d['episode']for d in heldout]))
            models.append((seed,fold,model,rms,training,heldout))
    # All four fixed fits have completed before opening any physical outcomes.
    selection_rows=[];evaluations=[]
    for seed,fold,model,rms,training,heldout in models:
        evaluations.append(dict(seed=seed,heldout_fold=fold,train=fork_eval(model,training,rms),heldout=fork_eval(model,heldout,rms)))
        for d in heldout:
            with np.load(d['file'])as z:
                goal=torch.from_numpy(z['goal'].copy())[None];ini=torch.from_numpy(d['initial'])[None];act=torch.from_numpy(d['actions'])[None]
                with torch.no_grad():score=model(inputs(ini,goal,act,rms).reshape(-1,253)).reshape(39).numpy()
                family=z['family'];ids=np.flatnonzero(np.isin(family,['zero','gaussian'])|np.char.startswith(family,'cem_'));assert len(ids)==22
                ph=z['physical_cost'][:,-1];pos=z['position_error_m'][:,-1];angle=z['orientation_error_deg'][:,-1];hold=z['success'].any(1)
                for group,indices in [('nonoracle',ids),('all',np.arange(39))]:
                    r=dict(seed=seed,heldout_fold=fold,case=d['case'],episode=d['episode'],group=group,physical_oracle_cost=float(ph[indices].min()),coverage_hold=float(hold[indices].any()))
                    for name,cost in [('head',-score),('predicted_mse',z['predicted_cost']),('true_encoded_mse',z['encoded_cost'])]:
                        i=int(indices[np.argmin(cost[indices])]);r[name]=dict(index=i,family=str(family[i]),pose_cost=float(ph[i]),position_cm=float(pos[i]*100),angle_deg=float(angle[i]),joint=float(pos[i]<=.03 and angle[i]<=15),hold=float(hold[i]))
                    selection_rows.append(r)
    metrics=['pose_cost','position_cm','angle_deg','joint','hold'];summaries=[]
    for seed in [1,2]:
        for group in ['nonoracle','all']:
            rr=sorted([r for r in selection_rows if r['seed']==seed and r['group']==group],key=lambda r:r['episode']);assert len(rr)==50
            summaries.append(dict(seed=seed,group=group,selectors={name:{k:interval([r[name][k]for r in rr])for k in metrics}for name in ['head','predicted_mse','true_encoded_mse']},head_minus={name:{k:interval([r['head'][k]-r[name][k]for r in rr])for k in metrics}for name in ['predicted_mse','true_encoded_mse']}))
    paired=[]
    for group in ['nonoracle','all']:
        rr=sorted([r for r in selection_rows if r['group']==group],key=lambda r:(r['seed'],r['episode']));assert [r['episode']for r in rr[:50]]==[r['episode']for r in rr[50:]]
        paired.append(dict(group=group,head_minus={name:{k:interval(np.array([r['head'][k]-r[name][k]for r in rr]).reshape(2,50).mean(0))for k in metrics}for name in ['predicted_mse','true_encoded_mse']}))
    result=dict(marker=MARKER,passed=True,seconds=time.monotonic()-started,random_branch_chance=1/39,uniform_ce=math.log(39),fit_reports=fit_reports,evaluations=evaluations,summaries=summaries,paired=paired,cases=selection_rows,source_provenance=sources,protocol_sha256=sha(out/'protocol.json'),model_encoder_renderer_simulator_gpu_calls=0,all_four_fits_completed_before_physical_read=True,limitation='Validation-source cross-fitting with additional executed fork data. No original benchmark claim. Bootstrap is conditional/descriptive and does not account for refitting or environment repeats.')
    write(out/'results.json',result);print(json.dumps(dict(passed=True,seconds=result['seconds'],output=str(out/'results.json'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
