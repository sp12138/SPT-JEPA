"""Six bounded frozen Cube extractions and the predeclared CPU physical probes."""
import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

B = Path('/efs/prescottsun/SPT-JEPA')
PY = '/efs/prescottsun/project_jepa/envs/lewm/bin/python'
MODES = ('observed_future', 'clean_rollout', 'teacher_forced_prefix', 'initial_carry', 'shifted_observed_carry')
TRAIN = B/'runs/carrier_metric_20261031/train-clips128.json'
VAL = B/'runs/native_cube12000_20261031/transition-cube-native-lewm-seed1/windows.json'
HELPERS = {
 'orientation_probe.py':'3dbe1fd09eabfc85073a264bff384624dafbc6996d3302c26baf8335699598b6',
 'orientation_rbf_probe.py':'a48d3402a9fe3f40878b0d61ed639da0bb45fc12f095771ef2940d607951794d',
 'carrier_spatial_probe.py':'b01d6ded47aa91e67048f3b6a8acdb37a454c4300f125d787b53a42e97ad64c6'}


def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source_check(source):
    for name in ['manifest','runtime_manifest']:
        m=read(source[name])
        for relative,digest in m['files'].items():assert sha(Path(m['source'])/relative)==digest,relative
def complete_state_hash(model):
    import torch
    h=hashlib.sha256()
    for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
        for name,t in items:
            a=t.detach().cpu().contiguous();h.update(str((kind,name,tuple(a.shape),a.dtype)).encode())
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def array_check(states, dimension):
    import numpy as np
    assert states['observed'].shape==(64,8,dimension)
    for key in MODES:
        assert states[key].shape==(64,7,dimension),key
        assert np.isfinite(states[key]).all(),key
    np.testing.assert_array_equal(states['observed_future'],states['observed'][:,1:])
    np.testing.assert_array_equal(states['initial_carry'],np.repeat(states['observed'][:,:1],7,axis=1))
    np.testing.assert_array_equal(states['shifted_observed_carry'],states['observed'][:,:-1])
def decomposition(observed, forecast, target):
    import numpy as np
    a=observed-target;b=forecast-observed
    terms=dict(observation_mse=float(np.square(a).mean()),decoded_forecast_discrepancy_mse=float(np.square(b).mean()),
               cross_term=float(2*(a*b).mean()),forecast_mse=float(np.square(forecast-target).mean()))
    terms['identity_abs_error']=abs(terms['forecast_mse']-sum(terms[k]for k in ['observation_mse','decoded_forecast_discrepancy_mse','cross_term']))
    assert terms['identity_abs_error']<1e-9
    return terms


def cohorts():
    import numpy as np
    train,val=read(TRAIN),read(VAL);data=B/'data/hardtasks-v1/cube-metadata'
    assert sha(TRAIN)=='16070202b03fe85f44e997be574a65edaec308006891e9b498a11528eb1d6cac'
    assert sha(VAL)=='6ea9cf49e5b0f319038882818cb0c58f3d52b162cf4ca3c69ce5f1981f2aa731'
    assert train==read(B/'reports/native-lewm600-20261031/source-bncal-v1/train128-cube.json')
    for split,count,m in [('train',128,train),('validation',64,val)]:
        assert m['split']==split and m['count']==count and len(m['windows'])==count and m['window']==8 and m['frameskip']==5
        assert sha(data/'dataset_card.json')==m['dataset_card_sha256']
        for name,digest in m['metadata_sha256'].items():assert sha(data/name)==digest
        eps={x['episode']for x in m['windows']};assert len(eps)==count and eps<=set(np.load(data/(split+'_episodes.npy')).tolist())
        assert all(len(x['rows'])==8 and all(b-a==5 for a,b in zip(x['rows'],x['rows'][1:])) for x in m['windows'])
    assert not {x['episode']for x in train['windows']}&{x['episode']for x in val['windows']}
    return train,val


def prepare(args):
    import numpy as np
    root=Path(args.root);assert not root.exists(),'Preserve earlier candidates'
    train,val=cohorts();helper=read(args.source_record) if args.source_record else None
    script=Path(helper['source'])/'experiments/native_baseline_20261031/readability.py' if helper else Path('<FREEZE_REQUIRED>/readability.py')
    if helper:source_check(helper);assert sha(script)==sha(__file__)
    jobs=[];specs=[];seen=set()
    for family,step,parent in [('native',6000,'native_lewm6000_20261031'),('native',12000,'native_cube12000_20261031'),('tf',12000,'cube_tf12000_20261031')]:
        old=B/'runs'/parent;source=read(old/'source-record.json')
        assert source['source_commit'].startswith('7a1ab4f' if family=='native' else '8e0f821')
        if source['source_commit'] not in seen:source_check(source);seen.add(source['source_commit'])
        for seed in [1,2]:
            name=f'cube-native-lewm-seed{seed}' if family=='native' else f'cube-temporal-forecast-seed{seed}'
            run=old/name;checkpoint=run/f'model-step{step:07d}';cfg=read(run/'config.json')
            assert cfg['seed']==seed and cfg['model']['window']==8 and cfg['model']['action_dim']==25
            weightsha=sha(checkpoint/'weights.pt');cache=None;dim=192 if family=='native' else 64
            if family=='native':
                d=old/f'transition-{name}';meta=read(d/'state-audit.json')
                assert meta['weights_sha256']==weightsha and meta['step']==step and meta['state_dim']==dim and meta['manifest_sha256']==sha(VAL)
                assert read(d/'windows.json')==val
                with np.load(d/'state-audit.npz') as f:array_check(f,dim)
                cache=dict(path=str(d/'state-audit.npz'),sha256=sha(d/'state-audit.npz'),report=str(d/'state-audit.json'),report_sha256=sha(d/'state-audit.json'))
            else:
                audit=read(old/f'fullstate12000-{name}/fullstate.json');assert audit['portable_sha256']==weightsha and audit['passed']
            ident=f'{family}{step}-seed{seed}';out=root/ident
            spec=dict(id=ident,family=family,step=step,seed=seed,run=str(run),checkpoint=str(checkpoint),weights_sha256=weightsha,
                config_sha256=sha(run/'config.json'),data=cfg['data'],dimension=dim,model_source=source,helper_source=helper,
                helper_sha256=sha(__file__),train_manifest=str(TRAIN),train_manifest_sha256=sha(TRAIN),
                val_manifest=str(VAL),val_manifest_sha256=sha(VAL),validation_cache=cache,output=str(out),
                budget=dict(rgb_frames=1024 if cache else 1536,predict_calls=0 if cache else 112,action_encoder_calls=0 if cache else 8,renderer_calls=0),
                helpers=HELPERS,precision='FP32; PyTorch highest; original audit SDPA; eval/no-grad; encode batch4/predict batch8')
            sp=root/'specs'/(ident+'.json');write(sp,spec);specs.append(spec)
            jobs.append(dict(id=ident,host='dev2',gpu=len(jobs),hold=True,phase='frozen-raw-readability-extraction',output=str(out),source=helper,
                             command=[PY,'-B','-u',str(script),'extract','--spec',str(sp)]))
    assert len(jobs)==6 and sum(s['budget']['rgb_frames'] for s in specs)==7168 and sum(s['budget']['predict_calls']for s in specs)==224
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},jobs=jobs))
    write(root/'protocol.json',dict(specs=specs,helper=helper,total_rgb_frames=7168,total_predict_calls=224,
         scope='Six raw fixed-model diagnostics; no BN calibration, training, renderer, counterfactual rollout, planner or new cohort. Fixed physical readouts only after all caches pass.'))
    (root/'HOLD_DISPATCH').write_text('Candidate preserved until frozen-source review and actual GPU availability checks.\n')
    write(root/'contracts.json',dict(passed=True,models=6,native_caches_reused=4,training_rows_identical=True,validation_disjoint=True,source_and_checkpoint_checks=True,shape_checks=True,cuda_calls=0,probe_fits=0))
    print(json.dumps(dict(root=str(root),jobs=6,frames=7168,predict_calls=224,frozen=helper is not None)))


def extract(args):
    spec=read(args.spec);source=spec['model_source']
    # Dispatcher owns a common helper source; each model runs in its ORIGINAL import universe.
    env={**os.environ,'PYTHONPATH':source['pythonpath'],'PYTHONDONTWRITEBYTECODE':'1'}
    os.chdir(source['source']);os.execve(PY,[PY,'-B','-u',str(Path(__file__).resolve()),'extract-inner','--spec',args.spec],env)


def extract_inner(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Subset
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from experiments.ideas_validate_20261031.audit import clean_rollout,prefix_prediction
    from vforge.ops import OpsConfig,initialize_ops
    s=read(args.spec);out=Path(s['output']);assert not (out/'extraction.json').exists()
    assert sha(__file__)==s['helper_sha256'];source_check(s['model_source']);cohorts()
    assert os.environ['PYTHONPATH']==s['model_source']['pythonpath']
    weights=Path(s['checkpoint'])/'weights.pt';cfg=read(Path(s['run'])/'config.json')
    assert sha(weights)==s['weights_sha256'] and sha(Path(s['run'])/'config.json')==s['config_sha256']
    torch.set_num_threads(2);torch.manual_seed(1042);torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda');initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=device,parallel_state=None)
    model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    clsfile=Path(inspect.getsourcefile(type(model)));assert clsfile.is_relative_to(Path(s['model_source']['source']))
    imported={}
    names=['as_jepa_lewm.factory','as_jepa_tools.benchmark_data','experiments.ideas_validate_20261031.audit','vforge.ops']
    names += [c.__module__ for c in type(model).__mro__ if c.__module__.startswith(('as_jepa','stable_worldmodel'))]
    if s['family']=='native':names+=['stable_worldmodel.wm.lewm.module','stable_pretraining.backbone.utils']
    for name in dict.fromkeys(names):
        module=importlib.import_module(name);path=Path(inspect.getsourcefile(module)).resolve()
        if name.startswith(('as_jepa','experiments.')):assert path.is_relative_to(Path(s['model_source']['source']))
        if name.startswith('vforge.'):assert path.is_relative_to(Path(s['model_source']['runtime_source']))
        imported[name]=dict(path=str(path),sha256=sha(path))
    before=complete_state_hash(model);counts=dict(rgb_frames=0,predict_calls=0,action_encoder_calls=0,renderer_calls=0)
    def encoder_hook(module,inputs):
        counts['rgb_frames']+=len(inputs[0]);assert counts['rgb_frames']<=s['budget']['rgb_frames']
    def action_hook(module,inputs):
        counts['action_encoder_calls']+=1;assert counts['action_encoder_calls']<=s['budget']['action_encoder_calls']
    def renderer_hook(module,inputs):raise AssertionError('No renderer permitted')
    handles=[model.encoder.register_forward_pre_hook(encoder_hook),model.action_encoder.register_forward_pre_hook(action_hook)]
    if getattr(model,'renderer',None) is not None:handles.append(model.renderer.register_forward_pre_hook(renderer_hook))
    original_predict=model.predict
    def predict(*a,**kw):
        counts['predict_calls']+=1;assert counts['predict_calls']<=s['budget']['predict_calls']
        return original_predict(*a,**kw)
    model.predict=predict
    start=time.time();torch.cuda.reset_peak_memory_stats();arrays={}
    with torch.no_grad():
        for split,mp in [('train',s['train_manifest'])]+([] if s['validation_cache'] else [('validation',s['val_manifest'])]):
            manifest=read(mp);ds=BenchmarkWindows(s['data'],window=8,split=split);values=[];actions=[];offset=0
            for raw in DataLoader(Subset(ds,[x['index']for x in manifest['windows']]),batch_size=4,shuffle=False,num_workers=0):
                for i,rows in enumerate(raw['rows'].tolist()):
                    want=manifest['windows'][offset+i];assert rows==want['rows'] and int(raw['episode'][i])==want['episode']
                offset+=len(raw['episode']);pixels=model.normalize(raw['pixels'].to(device))
                values.append(model.encode({'pixels':pixels})['emb'].float())
                if split=='validation':actions.append(raw['action'].to(device).float())
            if ds.file is not None:ds.file.close()
            z=torch.cat(values);assert z.shape==(len(manifest['windows']),8,s['dimension']) and torch.isfinite(z).all()
            if split=='train':
                np.savez_compressed(out/'training-embeddings.npz',clips=z.cpu().numpy());del z,values
            else:
                act=torch.cat(actions);clean=[];teacher=[]
                for i in range(0,64,8):
                    a=model.action_encoder(act[i:i+8]);clean.append(clean_rollout(model,z[i:i+8,0],a,7));teacher.append(prefix_prediction(model,z[i:i+8],a))
                states=dict(observed=z,actions=act,observed_future=z[:,1:],clean_rollout=torch.cat(clean),teacher_forced_prefix=torch.cat(teacher),
                            initial_carry=z[:,:1].expand(-1,7,-1),shifted_observed_carry=z[:,:-1])
                arrays={k:v.cpu().numpy()for k,v in states.items()};array_check(arrays,s['dimension'])
                np.savez_compressed(out/'validation-states.npz',**arrays)
    assert counts==s['budget'],(counts,s['budget']);assert not model.training
    assert all(not m.training for m in model.modules())
    after=complete_state_hash(model);assert after==before and sha(weights)==s['weights_sha256']
    assert all(sha(v['path'])==v['sha256'] for v in imported.values())
    for h in handles:h.remove()
    files={p.name:sha(p)for p in out.glob('*.npz')}
    validation=s['validation_cache'] or dict(path=str(out/'validation-states.npz'),sha256=files['validation-states.npz'])
    assert sha(validation['path'])==validation['sha256']
    report=dict(passed=True,spec=s,spec_sha256=sha(args.spec),files=files,validation=validation,state_hash_before=before,state_hash_after=after,
        model_class_file=str(clsfile),model_class_file_sha256=sha(clsfile),complete_parameters_and_buffers_unchanged=True,actual_calls=counts,
        seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,torch_version=torch.__version__,device=torch.cuda.get_device_name(),
        helper_source_sha256=sha(__file__),model_source=s['model_source'],actual_imported_modules=imported,input_precision='FP32/highest; no autocast',raw_eval_bn=True)
    write(out/'extraction.json',report);print(json.dumps({k:report[k]for k in ['passed','actual_calls','seconds','peak_gib','state_hash_after']}),flush=True)


def analyze(args):
    import numpy as np
    import h5py,hdf5plugin
    from experiments.carrier_progress_20261031 import orientation_probe as pose
    from experiments.carrier_progress_20261031 import orientation_rbf_probe as rbf
    from experiments.carrier_progress_20261031 import carrier_spatial_probe as spatial
    root=Path(args.root);out=Path(args.output);assert not (out/'results.json').exists()
    for module in [pose,rbf,spatial]:assert sha(module.__file__)==HELPERS[Path(module.__file__).name]
    plan=read(root/'plan.json');assert len(plan['jobs'])==6 and (root/'HOLD_DISPATCH').exists()
    train,val=cohorts();card=read(B/'data/hardtasks-v1/cube-metadata/dataset_card.json')
    with h5py.File(card['h5_path'],'r',swmr=True)as h5:
        tq=pose.load_quaternions(h5,train).reshape(-1,4);vq=pose.load_quaternions(h5,val)[:,1:].reshape(-1,4)
        tx=spatial.load_xyz(h5,train).reshape(-1,3);vx=spatial.load_xyz(h5,val)[:,1:].reshape(-1,3)
    ty=np.concatenate([pose.invariant(tq),tx],1);vy=np.concatenate([pose.invariant(vq),vx],1)
    refs=dict(constant_train_mean=spatial.score(np.repeat(ty.mean(0,keepdims=True),448,0),vy,vq,ty,tq,64),
              true_target_with_training_codebook=spatial.score(vy,vy,vq,ty,tq,64))
    rows=[];predictions={};frames=0;calls=0
    for job in plan['jobs']:
        assert read(root/'exits'/(job['id']+'.json'))['returncode']==0
        d=Path(job['output']);r=read(d/'extraction.json');s=r['spec'];assert r['passed'] and r['state_hash_before']==r['state_hash_after']
        assert sha(d/'training-embeddings.npz')==r['files']['training-embeddings.npz'] and sha(r['validation']['path'])==r['validation']['sha256']
        frames+=r['actual_calls']['rgb_frames'];calls+=r['actual_calls']['predict_calls']
        with np.load(d/'training-embeddings.npz')as f:x=f['clips'].reshape(1024,s['dimension']).astype(float)
        with np.load(r['validation']['path'])as f:
            array_check(f,s['dimension']);states={k:f[k].reshape(448,s['dimension']).astype(float)for k in MODES}
        joined=np.concatenate([states[k]for k in MODES]);matrix,intercept,_=pose.ridge_fit(x,ty)
        train_rbf,test_rbf,config=rbf.kernel_fit_predict(x,ty,joined)
        fits=dict(linear=(x@matrix+intercept,joined@matrix+intercept,dict(ridge_mean_loss_units=.001)),rbf=(train_rbf,test_rbf,config))
        result=dict(id=s['id'],family=s['family'],step=s['step'],seed=s['seed'],dimension=s['dimension'],extraction=r,readouts={})
        for kind,(tp,vp,configuration)in fits.items():
            values={k:v for k,v in zip(MODES,np.split(vp,len(MODES)))}
            scores={k:spatial.score(v,vy,vq,ty,tq,64)for k,v in values.items()}
            parts={k:{'xyz':decomposition(values['observed_future'][:,81:],values[k][:,81:],vy[:,81:]),
                      'tensor81':decomposition(values['observed_future'][:,:81],values[k][:,:81],vy[:,:81])}for k in MODES if k!='observed_future'}
            result['readouts'][kind]=dict(configuration=configuration,training=spatial.score(tp,ty,tq,ty,tq,128),validation=scores,decomposition=parts)
            for k,v in values.items():predictions[s['id']+'__'+kind+'__'+k]=v
        rows.append(result)
    assert frames==7168 and calls==224
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'readout-predictions.npz',**predictions)
    report=dict(passed=True,rows=rows,references=refs,physical_labels_diagnostic_only=True,total_rgb_frames=frames,total_predict_calls=calls,
        train_manifest_sha256=sha(TRAIN),val_manifest_sha256=sha(VAL),helper_sha256=sha(__file__),predictions_sha256=sha(out/'readout-predictions.npz'),
        scope='Same128 training clips and448 held-out future frames; fixed linear/RBF probes, no validation tuning. Native192 versus TF64 differs in representation/probe/model capacity. Physical-unit accessibility, not latent-MSE comparison or planning causality. Joint3cm/15deg readout is not task success or hold. Teacher/shifted carry receive additional real observations.')
    write(out/'results.json',report)
    summary=[]
    for r in rows:
        summary.append(dict(id=r['id'],dimension=r['dimension'],readouts={kind:{mode:dict(xyz_cm=v['xyz_euclidean']['mean_cm'],xyz3cm=v['xyz_euclidean']['fraction_within3cm'],pose_deg=v['orientation_angles']['mean_deg'],pose15=v['orientation_angles']['fraction_within15deg'],joint=v['joint_readout_within3cm_and15deg_fraction'])for mode,v in fit['validation'].items()}for kind,fit in r['readouts'].items()}))
    write(out/'summary.json',dict(passed=True,rows=summary,full_results=dict(path=str(out/'results.json'),sha256=sha(out/'results.json')),scope=report['scope']))
    print(json.dumps(summary,indent=2))


def contracts(args):
    import numpy as np
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    cohorts();rng=np.random.default_rng(1042)
    for dim in [64,192]:
        z=rng.normal(size=(64,8,dim));a=dict(observed=z,observed_future=z[:,1:],clean_rollout=z[:,1:]+.1,teacher_forced_prefix=z[:,1:]+.2,initial_carry=np.repeat(z[:,:1],7,axis=1),shifted_observed_carry=z[:,:-1]);array_check(a,dim)
    bn=torch.nn.BatchNorm1d(3).eval();before=complete_state_hash(bn);bn(torch.ones(4,3));assert complete_state_hash(bn)==before
    p=rng.normal(size=(10,3));y=rng.normal(size=(10,3));o=rng.normal(size=(10,3));decomposition(o,p,y)
    assert not torch.cuda.is_initialized();write(args.output,dict(passed=True,shape_dimensions=[64,192],cohort_exact=True,scalar_buffer_hash=True,frozen_bn_unchanged=True,cross_term_identity=True,model_calls=0,probe_fits=0,cuda_initialized=False))
    print('CPU contracts passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['contracts','prepare','extract','extract-inner','analyze']);p.add_argument('--root');p.add_argument('--source-record');p.add_argument('--spec');p.add_argument('--output');a=p.parse_args()
    globals()[a.phase.replace('-','_')](a)
