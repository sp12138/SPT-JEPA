"""Offline diagnostic only: decode factual forecasts,never deploy state readouts.

NN inference is completed using RGB and executed forward inputs before any
privileged labels are loaded. A separate stage fits ephemeral readouts from
previously cached frozen RGB features. No readout coefficients are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_factual_effects_20261108'
SCENE=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'
HARD={'bin-picking-v3','door-open-v3','drawer-open-v3','peg-insert-side-v3','pick-place-v3','push-v3'}


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def neural(name):
    import torch
    from experiments.metaworld_baselines_20261107.models import PublicBaseline
    from experiments.metaworld_baselines_20261107.train_native import Windows,digest
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    assert not torch.cuda.is_initialized()
    out=ROOT/name;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    j=next(j for j in read(BASE/'runs/metaworld_baselines_native_20261108/plan.json')['jobs'] if j['name']==name)
    cfg=read(j['config']);model=PublicBaseline(cfg['method'],cfg['sources'])
    weights=Path(j['output'])/f"model-step{j['epoch_steps']['100']:07d}/weights.pt"
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False);before=digest(model)
    data=Windows(cfg['data'],8,cfg['seed'],'test');idx=read(Path(cfg['data'])/'pack-index.json')
    phase={p['original_episode']*5+p['phase']:p for p in idx['phases']}
    choices={t:[] for t in range(50)};fallback={t:[] for t in range(50)}
    for i,ep in enumerate(data.base.episodes):
        eid=int(data.base.episode_ids[ep]);p=phase[eid]
        assert int(data.base.offsets[ep])==p['offset']
        lo=int(data.base.ends[i-1]) if i else 0
        for start in {0,int(data.base.counts[i])-1}:
            fallback[p['task_id']].append((lo+start,eid//5))
        if p['phase']!=0:continue
        choices[p['task_id']].append((lo+int(data.base.counts[i])//2,eid//5))
    selected=[];tasks=[];originals=[]
    for task,options in choices.items():
        assert options,(task,options)
        pair=[options[0],options[-1]] if len(options)>=2 else [options[0],next(c for c in reversed(fallback[task]) if c[0]!=options[0][0])]
        for index,ep in pair:
            selected.append(index);tasks.append(task);originals.append(ep)
    assert len(selected)==100 and len(set(selected))==100
    observed=[];predicted=[];packed_rows=[];actions=[]
    with torch.inference_mode():
        for start in range(0,100,4):
            items=[data.base[i] for i in selected[start:start+4]]
            batch=torch.utils.data.default_collate(items)
            pixels=model.normalize(batch['pixels'])
            z=model.encode(dict(pixels=pixels))['emb'].float()
            inputs=batch['action'][:,1:7,None].transpose(1,2).contiguous()
            info=dict(pixels=pixels[:,:2,None].transpose(1,2),action_history=batch['action'][:,:1,None].transpose(1,2))
            pred=model.rollout(info,inputs)['predicted_emb'][:,0].float()
            assert pred.shape==z.shape and torch.allclose(pred[:,:2],z[:,:2],atol=1e-5,rtol=1e-5)
            assert torch.equal(inputs[:,0],batch['action'][:,1:7])
            observed.append(z.numpy());predicted.append(pred.numpy());packed_rows.append(batch['rows'].numpy());actions.append(inputs[:,0].numpy())
    assert digest(model)==before and not torch.cuda.is_initialized()
    np.savez_compressed(out/'rgb_forecast_cache.npz',observed=np.concatenate(observed),predicted=np.concatenate(predicted),
        packed_rows=np.concatenate(packed_rows),inputs=np.concatenate(actions),indices=np.asarray(selected),tasks=np.asarray(tasks),originals=np.asarray(originals))
    frozen=dict(passed=True,name=name,windows=100,unique_original_trajectories=len(set(originals)),tasks=50,windows_per_task=2,
        selection='Two distinct phase0 original trajectories/task when available;otherwise two distinct windows from same original,explicitly not independent trajectories.',
        weights_sha256=sha(weights),model_state_before=before,model_state_after=digest(model),worldmodel_updates=0,
        gpu_context_created=False,new_physical_episodes=0,privileged_labels_loaded_during_nn=False,
        model_inputs=['pixels','executed_forward_input'],input_action_indices='1..6fortransitions1->2..6->7;input0history',
        same_input_alignment=True,cache_sha256=sha(out/'rgb_forecast_cache.npz'),seconds=time.monotonic()-started)
    (out/'FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n');print(json.dumps(frozen),flush=True)


def analyze(name):
    # This stage imports NO neural modules and never calls a planner.
    import h5py
    import hdf5plugin
    import pyarrow.parquet as pq
    out=ROOT/name;frozen=read(out/'FROZEN.json');assert frozen['passed'] and sha(out/'rgb_forecast_cache.npz')==frozen['cache_sha256']
    cache=np.load(out/'rgb_forecast_cache.npz');sample=np.load(SCENE/'samples.npz');features=np.load(SCENE/name/'features.npz')
    assert read(SCENE/name/'FROZEN.json')['passed']
    train=sample['splits']==0;valid=sample['splits']==1;ids=sample['tasks'];n=len(ids)
    labels=sample['labels'].astype(np.float64);means=np.zeros((50,7));scales=np.zeros((50,7))
    for t in range(50):
        ys=labels[train&(ids==t)];means[t]=ys.mean(0);scales[t]=np.maximum(ys.std(0),[.01,.01,.01,.05,.01,.01,.01])
    target=(labels-means[ids])/scales[ids]
    x=features['embedding'][:n].astype(np.float64);xm=x[train].mean(0);xs=np.maximum(x[train].std(0),1e-6);x=(x-xm)/xs
    a=x[train];eigen,vec=np.linalg.eigh(a.T@a/len(a));cross=vec.T@(a.T@target[train]/len(a));best=None
    for alpha in [.001,.01,.1,1.]:
        w=vec@(cross/(np.maximum(eigen,0)[:,None]+alpha));loss=float(np.square(x[valid]@w-target[valid]).mean())
        if best is None or loss<best[0]:best=loss,alpha,w
    loss,alpha,w=best
    assert alpha==read(SCENE/name/'summary-v2.json')['results']['embedding']['ridge_alpha']
    card=read(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1/dataset_card.json')
    unique,inverse=np.unique(cache['packed_rows'].reshape(-1),return_inverse=True)
    with h5py.File(card['h5_path'],'r') as f:raw_rows=np.asarray(f['raw_row'])[unique][inverse].reshape(100,8)
    actual=np.full((100,8,7),np.nan,np.float64);idx=read(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1/pack-index.json')
    for source in idx['raw_files']:
        wi,ti=np.where((raw_rows>=source['start'])&(raw_rows<source['end']))
        if not len(wi):continue
        p=BASE/'data/metaworld-lerobot-b46a1c18'/source['path']
        table=pq.read_table(p,columns=['index','episode_index','task_id','observation.environment_state'])
        rows=table.take((raw_rows[wi,ti]-source['start']).astype(np.int64))
        assert np.array_equal(np.asarray(rows['index']),raw_rows[wi,ti])
        assert np.array_equal(np.asarray(rows['episode_index']),cache['originals'][wi])
        assert np.array_equal(np.asarray(rows['task_id']),cache['tasks'][wi])
        actual[wi,ti]=np.asarray(rows['observation.environment_state'].to_pylist())[:,:7]
    assert np.isfinite(actual).all()
    task=cache['tasks'];observed=((cache['observed']-xm)/xs@w)*scales[task,None]+means[task,None]
    predicted=((cache['predicted']-xm)/xs@w)*scales[task,None]+means[task,None]
    copy=np.broadcast_to(observed[:,1:2],observed.shape)
    names=read(BASE/'data/metaworld-lerobot-b46a1c18-packed-v1/split-manifest.json')['task_names']
    masks={'all50':np.ones(100,dtype=bool),'hard6':np.asarray([names[str(t)] in HARD for t in task])}
    results={}
    for scope,mask in masks.items():
        results[scope]={}
        for group,axes in {'hand_xyz':[0,1,2],'gripper':[3],'first_object_xyz':[4,5,6]}.items():
            gt=actual[mask][:,:,axes];ob=observed[mask][:,:,axes];pr=predicted[mask][:,:,axes];cp=copy[mask][:,:,axes]
            def rmse(a):return np.sqrt(np.square(a).mean((0,2))).tolist()
            results[scope][group]=dict(windows=int(mask.sum()),physical_displacement_rms_by_horizon=rmse(gt[:,2:]-gt[:,1:2]),
                observed_readout_rmse_by_horizon=rmse(ob[:,2:]-gt[:,2:]),forward_readout_rmse_by_horizon=rmse(pr[:,2:]-gt[:,2:]),
                persistence_readout_rmse_by_horizon=rmse(cp[:,2:]-gt[:,2:]),
                observed_delta_readout_rmse_by_horizon=rmse((ob[:,2:]-ob[:,1:2])-(gt[:,2:]-gt[:,1:2])),
                forward_delta_readout_rmse_by_horizon=rmse((pr[:,2:]-ob[:,1:2])-(gt[:,2:]-gt[:,1:2])),
                latent_forecast_to_readout_difference_rmse_by_horizon=rmse(pr[:,2:]-ob[:,2:]))
    # Save diagnostic predictions/labels only;never save deployable coefficients.
    np.savez_compressed(out/'diagnostic_only.npz',actual=actual,observed_readout=observed,forward_readout=predicted,tasks=task,originals=cache['originals'])
    result=dict(passed=True,name=name,results=results,frozen=frozen,readout_ridge_alpha=alpha,
        readout_validation_normalized_mse=loss,coefficients_saved=False,worldmodel_updates=0,neural_modules_loaded_in_analysis=False,
        cache_sha256=frozen['cache_sha256'],analysis_source_sha256=sha(__file__),
        scope='DiagnosticONLY. 100heldoutwindows,2pertask,commonmodels;distinctoriginaltrajectorycountdisclosedinfrozenrecord. '
            'Ephemeralreadoutfit40originaltraintrajectories/task,alphavalidation5originaltrain/task;notfitontest. '
            'State/tasklabelsandreadoutcoefficientsNEVERsuppliedtoNN/planner/learnedproposal/targets. '
            'PhysicalRMSEreadoutsarenotphysicalsimulatorsorinformationbounds;offmanifoldlatentdecodingcanbebiased. '
            'Firstreportedobjectisnotuniversaltasktarget. Factualcontrolsknownexecuted,notoptimal/counterfactual. '
            'No newphysicalepisode/GPU/worldmodelSGD/controllerdeployment/performanceclaim.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,name=name,hard6=results['hard6'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['neural','analyze'],required=True)
    p.add_argument('--name',choices=['smwm-seed1','h-jepa-seed2'],required=True);args=p.parse_args()
    try:(neural if args.mode=='neural' else analyze)(args.name)
    except BaseException:
        out=ROOT/args.name;out.mkdir(parents=True,exist_ok=True)
        (out/f'ERROR-{args.mode}.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
