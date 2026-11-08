"""CPU-only frozen scene readouts. Privileged labels are diagnostic only.

No probe,coefficient,state,task ID,or reward is supplied to a world model or
planner. Closed-form readouts are not candidate objectives or control policies.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import h5py
import hdf5plugin
import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
DATA=BASE/'data/metaworld-lerobot-b46a1c18-packed-v1'
RAW=BASE/'data/metaworld-lerobot-b46a1c18'
ROOT=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)


def prepare():
    import pyarrow.parquet as pq
    ROOT.mkdir(exist_ok=False)
    idx=read(DATA/'pack-index.json');card=read(DATA/'dataset_card.json')
    train=set(np.load(DATA/'train_episodes.npy')//5);test=set(np.load(DATA/'test_episodes.npy')//5)
    assert len(train)==2250 and len(test)==250 and not train&test
    phases=[p for p in idx['phases'] if p['phase']==0]
    assert len(phases)==2500 and {p['original_episode'] for p in phases}==train|test
    fit=set();validation=set()
    for task in range(50):
        episodes=sorted(p['original_episode'] for p in phases if p['task_id']==task and p['original_episode'] in train)
        assert len(episodes)==45
        np.random.default_rng(20261108+task).shuffle(episodes)
        fit.update(episodes[:40]);validation.update(episodes[40:])
    examples=[]
    for p in phases:
        split=0 if p['original_episode'] in fit else (1 if p['original_episode'] in validation else 2)
        for t in np.unique(np.linspace(0,p['length']-1,min(4,p['length']),dtype=np.int64)):
            examples.append(dict(packed_index=p['offset']+int(t),original_episode=p['original_episode'],task=p['task_id'],split=split))
    examples.sort(key=lambda r:r['packed_index'])
    indices=np.asarray([r['packed_index'] for r in examples]);assert np.all(np.diff(indices)>0)
    with h5py.File(card['h5_path'],'r') as f:raw_rows=np.asarray(f['raw_row'])[indices]
    labels=np.full((len(examples),7),np.nan,np.float32)
    for file in idx['raw_files']:
        chosen=np.flatnonzero((raw_rows>=file['start'])&(raw_rows<file['end']))
        if not len(chosen):continue
        table=pq.read_table(RAW/file['path'],columns=['index','episode_index','task_id','observation.state','observation.environment_state'])
        selected=table.take((raw_rows[chosen]-file['start']).astype(np.int64))
        assert np.array_equal(np.asarray(selected['index']),raw_rows[chosen])
        state=np.asarray(selected['observation.environment_state'].to_pylist(),np.float32)
        robot=np.asarray(selected['observation.state'].to_pylist(),np.float32)
        assert state.shape==(len(chosen),39) and np.allclose(robot,state[:,:4],atol=1e-6)
        assert np.array_equal(np.asarray(selected['episode_index']),[examples[i]['original_episode'] for i in chosen])
        assert np.array_equal(np.asarray(selected['task_id']),[examples[i]['task'] for i in chosen])
        labels[chosen]=state[:,:7]
    assert np.isfinite(labels).all()
    task_names=read(DATA/'split-manifest.json')['task_names']
    bank=read(BASE/'runs/metaworld_input_inverse_20261107_v1/followup/cases/manifest.json')['rows']
    online_labels=[];online_tasks=[]
    for c in bank:
        assert sha(c['path'])==c['sha256']
        with np.load(c['path']) as cached:online_labels.append(cached['initial_state'][:7])
        online_tasks.append(next(int(k) for k,v in task_names.items() if v==c['task']))
    np.savez_compressed(ROOT/'samples.npz',indices=indices,raw_rows=raw_rows,labels=labels,
        tasks=np.asarray([r['task'] for r in examples]),splits=np.asarray([r['split'] for r in examples]),
        original_episodes=np.asarray([r['original_episode'] for r in examples]),online_labels=np.asarray(online_labels),online_tasks=np.asarray(online_tasks))
    save(ROOT/'selection.json',dict(examples=examples,online_cases=bank,samples_sha256=sha(ROOT/'samples.npz'),
        physical_episodes=2500,fit_episodes=len(fit),validation_episodes=len(validation),test_episodes=len(test),
        samples=len(examples),models=['smwm-seed1','h-jepa-seed2'],
        scope='Atmost4phase0frames/originalepisode. Probes fit40traintrajectories/task,selectridgeon5traintrajectories/task,test5originallyheldouttrajectories/task. '
            'TaskIDusedonlytoformconditional diagnosticlabels/metrics;neverNN/plannerinput. '
            'Native39Dlabels[:7]=handXYZ/gripper/firstreportedobjectXYZ;firstobjectnotuniversallytasktarget. '
            'Onlineinitial1000cases evaluated separately,not newphysicalepisodes. '
            'No readout deployment,no worldmodelSGD,no strong pretrained teachers.'))
    print(json.dumps(dict(prepared=True,samples=len(examples),physical_episodes=2500)),flush=True)


def features(name,out,smoke=False,epoch=100):
    import torch
    import torch.nn.functional as F
    from transformers import ViTModel
    from experiments.metaworld_baselines_20261107.models import PublicBaseline
    from experiments.metaworld_baselines_20261107.train_native import digest
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    selection=read(ROOT/'selection.json');assert sha(ROOT/'samples.npz')==selection['samples_sha256']
    assert name in selection['models']
    sample=np.load(ROOT/'samples.npz');indices=sample['indices'][:2] if smoke else sample['indices']
    online_cases=selection['online_cases'][:2] if smoke else selection['online_cases']
    parent='metaworld_baselines_native_20261108' if epoch==100 else 'metaworld_baseline_scaling_20261108'
    job=next(j for j in read(BASE/f'runs/{parent}/plan.json')['jobs'] if j['name']==name)
    cfg=read(job['config']);checkpoint=Path(job['output'])/f"model-step{job['epoch_steps'][str(epoch)]:07d}"
    model=PublicBaseline(cfg['method'],cfg['sources']);model.load_state_dict(torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before=digest(model)
    encoders=[m for m in model.modules() if isinstance(m,ViTModel)];assert len(encoders)==1
    captured=[]
    hook=encoders[0].register_forward_hook(lambda m,a,r:captured.append(r.last_hidden_state.detach()))
    arrays={k:[] for k in ['embedding','backbone_cls','patch_grid2','raw_rgb32']}
    card=read(DATA/'dataset_card.json');started=time.monotonic()

    def one(images):
        x=torch.from_numpy(np.stack(images))
        if x.shape[-1]==3:x=x.permute(0,3,1,2)
        x=x.float()
        if x.shape[-2:]!=(224,224):x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True).round().clamp(0,255)
        x=x.byte();captured.clear()
        with torch.inference_mode():
            z=model.encode(dict(pixels=model.normalize(x[:,None])))['emb'][:,0].float()
            assert len(captured)==1
            tokens=captured[0].float();n=int(math.isqrt(tokens.shape[1]-1));assert n*n==tokens.shape[1]-1
            patches=tokens[:,1:].reshape(len(x),n,n,-1).permute(0,3,1,2)
            values=dict(embedding=z,backbone_cls=tokens[:,0],patch_grid2=F.adaptive_avg_pool2d(patches,(2,2)).flatten(1),
                raw_rgb32=F.adaptive_avg_pool2d(x.float()/255,(32,32)).flatten(1))
        for k,v in values.items():arrays[k].append(v.numpy().copy())

    with h5py.File(card['h5_path'],'r') as f:
        for start in range(0,len(indices),16):
            one(np.asarray(f['pixels'][indices[start:start+16]]))
            if start%512==0:save(out/'progress.json',dict(offline_frames=min(start+16,len(indices)),seconds=time.monotonic()-started))
    for start in range(0,len(online_cases),16):
        images=[]
        for c in online_cases[start:start+16]:
            with np.load(c['path']) as cached:images.append(cached['initial'].copy())
        one(images)
    hook.remove();assert digest(model)==before and not torch.cuda.is_initialized()
    all_features={k:np.concatenate(v) for k,v in arrays.items()}
    assert all(len(a)==len(indices)+len(online_cases) and np.isfinite(a).all() for a in all_features.values())
    np.savez_compressed(out/'features.npz',**all_features)
    save(out/'FROZEN.json',dict(passed=True,name=name,model_updates=0,model_state_before=before,model_state_after=digest(model),
        weights_sha256=sha(checkpoint/'weights.pt'),samples_sha256=selection['samples_sha256'],
        gpu_context_created=False,offline_frames=len(indices),online_initial_frames=len(online_cases),seconds=time.monotonic()-started,
        technical_smoke_only=smoke,training_epoch=epoch,checkpoint=str(checkpoint),source_sha256=sha(__file__),feature_shapes={k:list(v.shape) for k,v in all_features.items()},
        raw_rgb_features_sha256=hashlib.sha256(all_features['raw_rgb32'].tobytes()).hexdigest()))


def fit(out):
    sample=np.load(ROOT/'samples.npz');feature=np.load(out/'features.npz');n=len(sample['indices'])
    tasks=sample['tasks'];labels=sample['labels'].astype(np.float64);splits=sample['splits'];training=splits==0
    mean=np.zeros((50,7));std=np.zeros((50,7));floor=np.asarray([.01,.01,.01,.05,.01,.01,.01])
    for t in range(50):
        y=labels[training&(tasks==t)];assert len(y)>0
        mean[t]=y.mean(0);std[t]=np.maximum(y.std(0),floor)
    y=(labels-mean[tasks])/std[tasks];online_tasks=sample['online_tasks'];online_labels=sample['online_labels']
    online_y=(online_labels-mean[online_tasks])/std[online_tasks]
    groups={'hand_xyz':[0,1,2],'gripper':[3],'first_object_xyz':[4,5,6]}

    def metrics(pred,target,physical,ids):
        result={}
        for group,axes in groups.items():
            error=(pred[:,axes]-target[:,axes])**2;base=target[:,axes]**2
            physical_error=error*np.square(std[ids][:,axes]);task_errors=[];task_bases=[];rms=[]
            for t in range(50):
                mask=ids==t
                task_errors.append(float(error[mask].mean()));task_bases.append(float(base[mask].mean()))
                rms.append(float(np.sqrt(physical_error[mask].mean())))
            result[group]=dict(macro_normalized_mse=float(np.mean(task_errors)),
                relative_mse_to_fit_task_mean=float(np.mean(task_errors)/max(np.mean(task_bases),1e-20)),
                macro_rmse_native_units=float(np.mean(rms)),tasks=50,
                task_rmse_native_units={str(t):rms[t] for t in range(50)})
        predicted_physical=pred*std[ids]+mean[ids]
        relative_error=(predicted_physical[:,:3]-predicted_physical[:,4:7])-(physical[:,:3]-physical[:,4:7])
        rms=[float(np.sqrt(np.square(relative_error[ids==t]).mean())) for t in range(50)]
        result['hand_minus_firstobject_xyz']=dict(macro_rmse_native_units=float(np.mean(rms)),tasks=50,
            task_rmse_native_units={str(t):rms[t] for t in range(50)})
        return result

    results={};rng=np.random.default_rng(20261108)
    feature_sets={k:feature[k] for k in feature.files}
    # Nonlinear diagnostic of the ACTUAL model embedding; no trained NN.
    emb=feature_sets['embedding'].astype(np.float64);em=emb[:n][training].mean(0);es=np.maximum(emb[:n][training].std(0),1e-6)
    normalized=(emb-em)/es
    frequencies=rng.normal(size=(emb.shape[1],512))/math.sqrt(emb.shape[1]);phase=rng.uniform(0,2*np.pi,512)
    feature_sets['embedding_rff512']=np.sqrt(2/512)*np.cos(normalized@frequencies+phase)
    # Privileged robot-only baseline is diagnostic;never deployed.
    # The target coordinates are already task-centered/scaled. Give this
    # privileged diagnostic the SAME coordinate system,so it can exactly read
    # its own hand input. Raw robot input would lack the task means that RGB
    # can infer from backgrounds,an unfair correlation baseline.
    robot=np.concatenate([y[:,:4],online_y[:,:4]])
    feature_sets['robot_state_diagnostic_only']=np.concatenate([robot,robot**2],axis=1)
    for name,values in feature_sets.items():
        x=np.asarray(values,np.float64);xm=x[:n][training].mean(0);xs=np.maximum(x[:n][training].std(0),1e-6);x=(x-xm)/xs
        a=x[:n][training];target=y[training];gram=a.T@a/len(a);cross=a.T@target/len(a)
        eigen,vec=np.linalg.eigh(gram);projected=vec.T@cross;best=None
        for alpha in [.001,.01,.1,1.]:
            w=vec@(projected/(np.maximum(eigen,0)[:,None]+alpha))
            score=float(np.square(x[:n][splits==1]@w-y[splits==1]).mean())
            if best is None or score<best[0]:best=score,alpha,w
        score,alpha,w=best
        results[name]=dict(ridge_alpha=alpha,validation_normalized_mse=score,
            offline_test=metrics(x[:n][splits==2]@w,y[splits==2],labels[splits==2],tasks[splits==2]),
            online_initial=metrics(x[n:]@w,online_y,online_labels,online_tasks))
    assert read(out/'FROZEN.json')['passed']
    assert results['robot_state_diagnostic_only']['offline_test']['hand_xyz']['macro_rmse_native_units']<.002
    save(out/'summary-v2.json',dict(passed=True,results=results,frozen=read(out/'FROZEN.json'),analysis_source_sha256=sha(__file__),
        correction='Readoutv2normalizestheprivilegedrobotinputlikeitsconditionaltarget;original100epochv1summariesretained. '
            'Closedformfittingreusesfrozenfeatures,with0NN/SGDduringfitting. Addsper-taskandrelativehand/objecterrors.',
        selection_sha256=sha(ROOT/'samples.npz'),worldmodel_updates=0,readout='Closedformridge,alpha selected ONLY onoriginaltraintrajectoryvalidation. '
            'Conditionaltaskmean/scale fromfitlabels,taskIDnot a NNfeature. ModelinputsRGBonly. '
            'Firstobjectnotuniversallytarget. Failedlinear/RFFreadouts donotproveinformationabsence. '
            'No coefficients saved/deployed,no policytraining,stateorprobeinputs toworldmodel/planner.'))
    print(json.dumps(dict(passed=True,model=out.name,results={k:v['offline_test']['first_object_xyz'] for k,v in results.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','run','smoke','fit'],required=True);parser.add_argument('--name')
    parser.add_argument('--epoch',type=int,choices=[100,200,300],default=100)
    args=parser.parse_args();out=ROOT/(str(args.name)+(f'-epoch{args.epoch}' if args.epoch!=100 else '')+('-smoke' if args.mode=='smoke' else ''))
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='fit':fit(out)
        else:
            out.mkdir(exist_ok=False);features(args.name,out,smoke=args.mode=='smoke',epoch=args.epoch)
            if args.mode=='run':fit(out)
    except BaseException:
        save((ROOT if args.mode=='prepare' else out)/'ERROR.json',dict(traceback=traceback.format_exc()));raise
