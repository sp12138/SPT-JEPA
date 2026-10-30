"""Approved frozen Cube TF6000 information/dynamics probe. Never train or launch."""
from __future__ import annotations
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import time

REPO=next(p for p in Path(__file__).resolve().parents if (p/'experiments/carrier_progress_20261031/orientation_probe.py').is_file())
sys.path.insert(0,str(REPO))
import numpy as np
from experiments.carrier_progress_20261031 import orientation_probe as pose
from experiments.carrier_progress_20261031 import carrier_spatial_probe as spatial

B=Path('/efs/prescottsun/SPT-JEPA')
OUTPUT=B/'reports/tf6000-pose-20261031'
MODES=('observed_future','clean_rollout','teacher_forced_prefix','initial_carry','shifted_observed_carry')
SOURCE=B/'runs/carrier_learn6000_20261031/source-record.json'
COMMIT='8e0f821bba84e7ed8c0f16500d7c880fde56d4cf'


def record(path):
    path=Path(path);return dict(path=str(path),sha256=pose.digest(path),bytes=path.stat().st_size)


def references(seed):
    name=f'cube-carrier-identity-seed{seed}'
    cal=B/'runs/carrier_metric_calibration_20261031'/('calibration-'+name)
    meta=pose.read(cal/'calibration.json')
    old=B/'runs/ideas_validate600_20261031'/('transition-'+name)
    now=B/'runs/carrier_learn6000_20261031'/('transition-'+name)
    a0,a1=pose.read(old/'state-audit.json'),pose.read(now/'state-audit.json')
    train_path=Path(meta['train_manifest']);val_path=old/'windows.json'
    assert pose.digest(train_path)==meta['train_manifest_sha256']
    assert pose.digest(val_path)==a0['manifest_sha256']==a1['manifest_sha256']==pose.digest(now/'windows.json')
    assert pose.digest(cal/'training-embeddings.npz')==meta['training_embeddings_sha256']
    assert a0['step']==600 and a1['step']==6000
    manifests=dict(train=pose.read(train_path),validation=pose.read(val_path))
    data=Path(meta['data'])
    for split,manifest in manifests.items():spatial.assert_metadata(data,manifest,split)
    assert [len(manifests[k]['windows']) for k in ['train','validation']]==[128,64]
    assert not {r['episode'] for r in manifests['train']['windows']} & {r['episode'] for r in manifests['validation']['windows']}
    with np.load(cal/'training-embeddings.npz') as z:train=z['clips'].copy()
    with np.load(old/'state-audit.npz') as z:observed0=z['observed'].copy()
    with np.load(now/'state-audit.npz') as z:
        observed=z['observed'].copy();actions=z['actions'].copy()
        states={k:z[k].copy() for k in MODES}
    np.testing.assert_array_equal(observed,observed0)
    assert train.shape==(128,8,64) and observed.shape==(64,8,64)
    assert all(v.shape==(64,7,64) for v in states.values())
    assert np.isfinite(train).all() and all(np.isfinite(v).all() for v in states.values())
    return dict(data=data,manifests=manifests,train=train,observed=observed,actions=actions,states=states,
        provenance={k:record(p) for k,p in dict(calibration=cal/'calibration.json',
            training_manifest=train_path,validation_manifest=val_path,
            training_embeddings=cal/'training-embeddings.npz',old_observed=old/'state-audit.npz',
            current_states=now/'state-audit.npz',current_audit=now/'state-audit.json').items()},
        observed600_equals6000=True)


def prediction_states(model,observed,actions):
    """Only clean_rollout gets an initial state; no observed future argument."""
    import torch
    from experiments.ideas_validate_20261031.audit import clean_rollout,prefix_prediction
    with torch.no_grad():
        embedded=model.action_encoder(actions)
        target=observed[:,1:]
        return dict(observed_future=target,
            clean_rollout=clean_rollout(model,observed[:,0],embedded,observed.shape[1]-1),
            teacher_forced_prefix=prefix_prediction(model,observed,embedded),
            initial_carry=observed[:,:1].expand_as(target),shifted_observed_carry=observed[:,:-1])


def source_contract():
    source=pose.read(SOURCE);assert source['source_commit']==COMMIT
    manifest=pose.read(source['manifest']);assert manifest['source_commit']==COMMIT
    checks={}
    # Pure analysis helpers may reside in the new snapshot, but their arithmetic is pinned.
    for name,module in [('experiments/carrier_progress_20261031/orientation_probe.py',pose),
                        ('experiments/carrier_progress_20261031/carrier_spatial_probe.py',spatial)]:
        expected=manifest['files'][name];assert pose.digest(module.__file__)==expected
        checks[name]=expected
    sys.path[:0]=source['pythonpath'].split(':')
    for module_name,name in [('as_jepa_lewm.model','as_jepa_lewm/model.py'),
        ('as_jepa_lewm.robust','as_jepa_lewm/robust.py'),('as_jepa_lewm.factory','as_jepa_lewm/factory.py'),
        ('as_jepa_tools.benchmark_data','as_jepa_tools/benchmark_data.py'),
        ('experiments.ideas_validate_20261031.audit','experiments/ideas_validate_20261031/audit.py')]:
        module=importlib.import_module(module_name)
        expected=manifest['files'][name]
        assert pose.digest(module.__file__)==pose.digest(Path(source['source'])/name)==expected
        checks[name]=dict(sha256=expected,loaded_path=str(module.__file__))
    return dict(source_record=record(SOURCE),source_commit=COMMIT,verified_files=checks)


def fresh_output(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError('Preserve completed extraction/analysis')
    path.mkdir(parents=True,exist_ok=True);return path


def extract(args):
    import torch
    from torch.utils.data import DataLoader,Subset
    contract=source_contract()
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from as_jepa_lewm.factory import build_model
    from vforge.ops import OpsConfig,initialize_ops
    refs=references(args.seed)
    run=B/'runs/carrier_learn6000_20261031'/f'cube-temporal-forecast-seed{args.seed}'
    cfg=pose.read(run/'config.json');checkpoint=run/'model-step0006000';weights=checkpoint/'weights.pt'
    full=pose.read(run.parent/('fullstate6000-'+run.name)/'fullstate.json')
    assert full['passed'] and full['step']==6000 and full['source_commit']==COMMIT
    assert pose.digest(weights)==full['portable_sha256']
    assert pose.digest(run/'config.json')==full['config_sha256']
    assert cfg['data']==str(refs['data']) and cfg['seed']==args.seed and cfg['model']['window']==8
    assert cfg.get('robust_variant') and not cfg.get('model_class')
    out=fresh_output(args.output);torch.set_num_threads(2);torch.manual_seed(1042)
    torch.set_float32_matmul_precision('highest');device=torch.device(args.device)
    initialize_ops(OpsConfig(backend='sdpa' if device.type=='cuda' else 'sdpa_math',allow_fallback=False),device=device,parallel_state=None)
    model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    assert model.state_dim==64 and not model.training and not any(p.requires_grad for p in model.parameters())
    started=time.time();saved={};encoder_batches=0
    with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
        for split,manifest in refs['manifests'].items():
            dataset=BenchmarkWindows(refs['data'],window=8,split=split)
            loader=DataLoader(Subset(dataset,[r['index'] for r in manifest['windows']]),
                              batch_size=args.batch_size,num_workers=0,shuffle=False)
            observed,actions=[],[];offset=0
            for batch in loader:
                assert not any(k.startswith('privileged') for k in batch)
                for i,episode in enumerate(batch['episode'].tolist()):
                    expected=manifest['windows'][offset+i]
                    assert episode==expected['episode'] and batch['rows'][i].tolist()==expected['rows']
                offset+=len(batch['episode'])
                moved={k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
                pixels,_,_=model.prepare(moved)
                observed.append(model.encode(dict(pixels=pixels))['emb'].float().cpu())
                actions.append(moved['action'].float().cpu());encoder_batches+=1
            observed=torch.cat(observed);actions=torch.cat(actions)
            saved[split+'_observed']=observed.numpy()
            if split=='validation':
                np.testing.assert_array_equal(actions.numpy(),refs['actions'])
                outputs={k:[] for k in MODES}
                for start in range(0,len(observed),args.batch_size):
                    states=prediction_states(model,observed[start:start+args.batch_size].to(device),
                        actions[start:start+args.batch_size].to(device))
                    for name,value in states.items():outputs[name].append(value.cpu().numpy())
                saved.update({k:np.concatenate(v) for k,v in outputs.items()})
                saved['validation_actions']=actions.numpy()
    assert saved['train_observed'].shape==(128,8,64) and saved['validation_observed'].shape==(64,8,64)
    assert all(saved[k].shape==(64,7,64) for k in MODES)
    assert all(np.isfinite(v).all() for v in saved.values())
    # A frozen forward must not update any serialized buffer or weight.
    original=torch.load(weights,map_location='cpu',weights_only=True)
    assert original.keys()==model.state_dict().keys()
    assert all(torch.equal(value.cpu(),original[name]) for name,value in model.state_dict().items())
    np.savez_compressed(out/'features.npz',**saved)
    meta=dict(seed=args.seed,step=6000,run=str(run),weights=record(weights),config=record(run/'config.json'),
        source_contract=contract,reference_provenance=refs['provenance'],data=str(refs['data']),
        manifests=refs['manifests'],features=record(out/'features.npz'),source=record(__file__),
        frames_encoded=1536,encoder_batches=encoder_batches,old_F_encoder_calls=0,state_dim=64,
        precision='float32/highest; eval; no autocast or posterior noise',
        model_state_unchanged=True,seconds=time.time()-started,
        scope='Only pixels/action observations enter extraction. No privileged labels, optimizer, BN calibration, planner or training update.')
    (out/'extraction.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seed=args.seed,frames_encoded=1536,seconds=meta['seconds'],output=str(out),model_state_unchanged=True)))


def fit_and_predict(train,test,target):
    """One fit per readout; test may concatenate all conditions without refitting."""
    gram,cross,tn,vn,normalization=spatial.standardized_grams(train,test)
    return spatial.readouts_from_grams(gram,cross,tn,vn,target,True),normalization


def score_scalar(x):
    return dict(xyz_cm=x['xyz_euclidean']['mean_cm'],xyz_within3cm=x['xyz_euclidean']['fraction_within3cm'],
        pose_deg=x['orientation_angles']['mean_deg'],pose_within15deg=x['orientation_angles']['fraction_within15deg'],
        joint=x['joint_readout_within3cm_and15deg_fraction'],
        xyz_r2=x['xyz_features']['variance_weighted_r2'],pose_r2=x['orientation_features']['variance_weighted_r2'])


def analyze(args):
    import h5py,hdf5plugin  # noqa: F401
    features=Path(args.features);meta=pose.read(features/'extraction.json');refs=references(meta['seed'])
    assert pose.digest(features/'features.npz')==meta['features']['sha256']
    assert refs['manifests']==meta['manifests'] and refs['provenance']==meta['reference_provenance']
    old_report_path=B/f'reports/carrier-spatial600-20261031/seed{meta["seed"]}-readouts/results.json'
    old_report=pose.read(old_report_path)
    assert pose.digest(spatial.__file__)==old_report['source_sha256']
    assert pose.digest(pose.__file__)==old_report['orientation_helper_sha256']
    old_orientation=pose.read(B/'reports/carrier-orientation600-20261031/results.json')
    task_source=Path(pose.cube_symmetry_angle_deg.__code__.co_filename)
    assert pose.digest(task_source)==old_orientation['pinned_task_source_sha256']
    card=pose.read(refs['data']/'dataset_card.json');manifests=refs['manifests']
    with h5py.File(card['h5_path'],'r',swmr=True) as h5:
        tq=pose.load_quaternions(h5,manifests['train']).reshape(-1,4)
        vq=pose.load_quaternions(h5,manifests['validation'])[:,1:].reshape(-1,4)
        txyz=spatial.load_xyz(h5,manifests['train']).reshape(-1,3)
        vxyz=spatial.load_xyz(h5,manifests['validation'])[:,1:].reshape(-1,3)
    ty=np.concatenate([pose.invariant(tq),txyz],axis=-1);vy=np.concatenate([pose.invariant(vq),vxyz],axis=-1)
    with np.load(features/'features.npz') as stored:
        tf_train=stored['train_observed'].reshape(1024,64).astype(np.float64)
        tf_states={k:stored[k].reshape(448,64).astype(np.float64) for k in MODES}
    models={'TF6000':(tf_train,tf_states),'Fidentity6000':(refs['train'].reshape(1024,64).astype(np.float64),
        {k:v.reshape(448,64).astype(np.float64) for k,v in refs['states'].items()})}
    out=fresh_output(args.output);rows=[];predictions={};reproduction={}
    for model,(train,states) in models.items():
        outputs,normalization=fit_and_predict(train,np.concatenate([states[k] for k in MODES]),ty)
        for kind,(tp,vp,configuration) in outputs.items():
            conditions={}
            for i,mode in enumerate(MODES):
                values=vp[i*448:(i+1)*448]
                conditions[mode]=spatial.score(values,vy,vq,ty,tq,64)
                predictions[model+'__'+kind+'__'+mode]=values
            if model=='Fidentity6000':
                old=next(r for r in old_report['rows'] if r['space']=='pooled4x4' and r['feature']=='rp64' and r['readout']==kind)['validation']
                a,b=score_scalar(conditions['observed_future']),score_scalar(old)
                delta={k:a[k]-b[k] for k in a};assert max(abs(v) for v in delta.values())<2e-5
                reproduction[kind]=delta
            rows.append(dict(seed=meta['seed'],model=model,readout=kind,configuration=configuration,
                standardization=normalization,train=spatial.score(tp,ty,tq,ty,tq,128),conditions=conditions))
    constant=np.repeat(ty.mean(0,keepdims=True),len(vy),axis=0)
    np.savez_compressed(out/'predictions.npz',**predictions)
    result=dict(seed=meta['seed'],rows=rows,constant_reference=spatial.score(constant,vy,vq,ty,tq,64),
        codebook_reference=spatial.score(vy,vy,vq,ty,tq,64),old_F_observation_reproduction=reproduction,
        extraction=record(features/'extraction.json'),old_F_spatial_report=record(old_report_path),
        source=record(__file__),helpers=dict(pose=record(pose.__file__),spatial=record(spatial.__file__)),
        pinned_task_source=record(task_source),
        predictions=record(out/'predictions.npz'),training_frames=1024,validation_frames=448,
        no_validation_tuning=True,old_F_model_calls=0,
        scope='Diagnostic physical labels fit readouts only. Same original train/val episodes and codebook. '
              'F observed/train codes reused from600, invariant at6000; F6000 causal states reused without model calls. '
              'Readout accessibility and forecast degradation are not total information or planning/three-step hold guarantees.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary=dict(seed=meta['seed'],full_results=record(out/'results.json'),
        rows=[dict(model=r['model'],readout=r['readout'],train=score_scalar(r['train']),
                   conditions={k:score_scalar(v) for k,v in r['conditions'].items()}) for r in rows],
        constant=score_scalar(result['constant_reference']),codebook=score_scalar(result['codebook_reference']),
        old_F_observation_reproduction=reproduction,scope=result['scope'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


def contracts():
    import torch
    class Toy:
        action_encoder=staticmethod(lambda a:a)
        @staticmethod
        def predict(z,a):return z+.3*a[:,:,:z.shape[-1]]
    gen=torch.Generator().manual_seed(1042);z=torch.randn(3,8,4,generator=gen);a=torch.randn(3,8,4,generator=gen)
    baseline=prediction_states(Toy(),z,a);changed=z.clone();changed[:,1:]+=27
    future=prediction_states(Toy(),changed,a)
    assert torch.equal(baseline['clean_rollout'],future['clean_rollout'])
    assert not torch.equal(baseline['teacher_forced_prefix'],future['teacher_forced_prefix'])
    padding=a.clone();padding[:,-1]+=100
    assert torch.equal(baseline['clean_rollout'],prediction_states(Toy(),z,padding)['clean_rollout'])
    altered=a.clone();altered[:,0]+=1
    assert not torch.equal(baseline['clean_rollout'],prediction_states(Toy(),z,altered)['clean_rollout'])
    rng=np.random.default_rng(1042);x=rng.normal(size=(40,8));test=rng.normal(size=(28,8));y=rng.normal(size=(40,5))
    out,_=fit_and_predict(x,test,y);w,b,_=pose.ridge_fit(x,y)
    assert np.max(np.abs(out['linear'][1]-(test@w+b)))<1e-10
    from experiments.carrier_progress_20261031.orientation_rbf_probe import kernel_fit_predict
    _,expected,_=kernel_fit_predict(x,y,test);assert np.max(np.abs(expected-out['rbf'][1]))<1e-10
    altered=test.copy();altered[14:]+=90;again,_=fit_and_predict(x,altered,y)
    for kind in ['linear','rbf']:
        np.testing.assert_allclose(out[kind][1][:14],again[kind][1][:14],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(out[kind][0],again[kind][0],atol=1e-12,rtol=1e-12)
    return dict(passed=True,future_observations_cannot_change_clean_forecast=True,
        padded_final_action_ignored=True,valid_action_changes_forecast=True,
        unchanged_primal_linear_and_fixed_RBF=True,test_rows_do_not_change_fitting_or_other_predictions=True,
        real_model_calls=0,GPU_calls=0,source_sha256=pose.digest(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['extract','analyze','run','contracts'])
    p.add_argument('--seed',type=int,choices=[1,2]);p.add_argument('--features');p.add_argument('--output')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--batch-size',type=int,default=4);a=p.parse_args()
    if a.phase=='contracts':print(json.dumps(contracts(),indent=2))
    elif a.phase=='run':
        assert a.seed and a.output
        extraction=copy.copy(a);extraction.output=str(Path(a.output)/f'seed{a.seed}-features')
        extract(extraction)
        analysis=copy.copy(a);analysis.features=extraction.output
        analysis.output=str(Path(a.output)/f'seed{a.seed}-readouts');analyze(analysis)
    elif a.phase=='extract':
        assert a.seed and a.output;extract(a)
    else:
        assert a.features and a.output;analyze(a)
