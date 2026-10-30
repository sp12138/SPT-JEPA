"""Two frozen-encoder extractions, then fixed CPU readouts; no model training.

All large feature/prediction arrays belong on EFS. Extract never reads labels.
Analyze reuses exact train/validation rows, fixed ridge and train-median RBF.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from experiments.carrier_progress_20261031 import orientation_probe as pose

B=Path('/efs/prescottsun/SPT-JEPA')
PROJECTION_SEED=20261030
RIDGE=.001
SPACES=('pooled4x4','fullpatch')
FEATURES=('rp64','rp192','full')


def projections(dim):
    """Preserve the original64 columns; append two independent fixed blocks."""
    blocks=[]
    for block in range(3):
        seed=[PROJECTION_SEED,dim]if block==0 else[PROJECTION_SEED,dim,block]
        rng=np.random.default_rng(np.random.SeedSequence(seed))
        blocks.append((rng.standard_normal((dim,64))/math.sqrt(64)).astype(np.float32))
    return blocks[0],np.concatenate(blocks,axis=1)/np.float32(math.sqrt(3))


def array_hash(array):
    import hashlib
    array=np.ascontiguousarray(array)
    h=hashlib.sha256(str((array.shape,array.dtype)).encode());h.update(array.tobytes());return h.hexdigest()


def assert_metadata(data,manifest,split):
    assert manifest['split']==split
    for name,sha in manifest['metadata_sha256'].items():assert pose.digest(data/name)==sha
    assert pose.digest(data/'dataset_card.json')==manifest['dataset_card_sha256']
    allowed=set(np.load(data/(split+'_episodes.npy')).tolist())
    episodes=[r['episode']for r in manifest['windows']]
    assert len(set(episodes))==len(episodes)and set(episodes)<=allowed


def extract(args):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader,Subset
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from as_jepa_lewm.factory import build_model
    import as_jepa_lewm.carrier as carrier_source
    from vforge.ops import OpsConfig,initialize_ops
    out=Path(args.output)
    if out.exists()and any(out.iterdir()):raise FileExistsError('Use a fresh extraction directory')
    out.mkdir(parents=True,exist_ok=True)
    name=f'cube-carrier-identity-seed{args.seed}'
    gate=B/'runs/ideas_validate600_20261031';run=gate/name
    cal=B/'runs/carrier_metric_calibration_20261031'/('calibration-'+name)
    calibration=pose.read(cal/'calibration.json');audit_dir=gate/('transition-'+name)
    audit=pose.read(audit_dir/'state-audit.json')
    cfg=pose.read(run/'config.json');weights=run/'model-step0000600/weights.pt'
    assert cfg['model_class']=='carrier'and cfg['model']['carrier_chart']=='identity'
    assert pose.digest(weights)==calibration['weights_sha256']==audit['weights_sha256']
    assert pose.digest(calibration['train_manifest'])==calibration['train_manifest_sha256']
    assert pose.digest(audit_dir/'windows.json')==audit['manifest_sha256']
    assert pose.digest(cal/'training-embeddings.npz')==calibration['training_embeddings_sha256']
    manifests=dict(train=pose.read(calibration['train_manifest']),validation=pose.read(audit_dir/'windows.json'))
    data=Path(cfg['data'])
    for split,manifest in manifests.items():assert_metadata(data,manifest,split)
    assert not {r['episode']for r in manifests['train']['windows']}.intersection(r['episode']for r in manifests['validation']['windows'])
    assert [len(manifests[k]['windows'])for k in ['train','validation']]==[128,64]
    with np.load(cal/'training-embeddings.npz')as saved:train_reference=saved['clips']
    with np.load(audit_dir/'state-audit.npz')as saved:val_reference=saved['observed']
    references=dict(train=train_reference,validation=val_reference)
    assert train_reference.shape==(128,8,64)and val_reference.shape==(64,8,64)
    torch.set_num_threads(2);torch.manual_seed(1042);torch.set_float32_matmul_precision('highest')
    device=torch.device(args.device)
    initialize_ops(OpsConfig(backend='sdpa'if device.type=='cuda'else'sdpa_math',allow_fallback=False),device=device,parallel_state=None)
    model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    assert not model.encoder.training and carrier_source.PROJECTION_SEED==PROJECTION_SEED
    mean,scale=[x.detach().cpu().numpy()for x in [model.carrier_mean,model.carrier_scale]]
    started=time.time();files={};projection_records={};validation={};dimensions={};tokens_side=None
    matrix_cache={};encoder_calls=0
    with torch.no_grad():
        for split,manifest in manifests.items():
            ds=BenchmarkWindows(data,window=manifest['window'],split=split)
            indices=[r['index']for r in manifest['windows']]
            loader=DataLoader(Subset(ds,indices),batch_size=args.batch_size,shuffle=False,num_workers=0)
            arrays={};offset=0;raw_error=0.;normalized_error=0.
            for batch in loader:
                for j,episode in enumerate(batch['episode'].tolist()):
                    expected=manifest['windows'][offset+j]
                    assert episode==expected['episode']and batch['rows'][j].tolist()==expected['rows']
                pixels=model.normalize(batch['pixels'].to(device)).float();bs,window=pixels.shape[:2]
                with torch.autocast(device_type=device.type,enabled=False):
                    encoded=model.encoder(pixels.flatten(0,1),interpolate_pos_encoding=True).last_hidden_state
                    patches=encoded[:,1:].float();encoder_calls+=1
                    side=math.isqrt(patches.shape[1]);assert side*side==patches.shape[1]
                    if tokens_side is None:tokens_side=side
                    assert side==tokens_side
                    grid=patches.transpose(1,2).reshape(bs*window,patches.shape[-1],side,side)
                    representations=dict(pooled4x4=F.adaptive_avg_pool2d(grid,(4,4)).permute(0,2,3,1).flatten(1),
                                         fullpatch=patches.flatten(1))
                    for space,raw in representations.items():
                        dim=raw.shape[-1];dimensions[space]=dim
                        if space not in matrix_cache:
                            p64,p192=projections(dim)
                            if space=='pooled4x4':np.testing.assert_array_equal(p64,model.carrier_projection.cpu().numpy())
                            matrix_cache[space]=(torch.from_numpy(p64).to(device),torch.from_numpy(p192).to(device))
                            projection_records[space]=dict(input_dim=dim,rp64_sha256=array_hash(p64),rp192_sha256=array_hash(p192),
                                nested_max_abs_error=float(np.max(np.abs(p192[:,:64]-p64/np.float32(math.sqrt(3))))))
                        p64,p192=matrix_cache[space]
                        projected=dict(full=raw,rp64=raw@p64,rp192=raw@p192)
                        for kind,value in projected.items():
                            key=space+'__'+kind
                            if key not in arrays:
                                filename=split+'__'+key+'.npy'
                                arrays[key]=np.lib.format.open_memmap(out/filename,mode='w+',dtype=np.float32,
                                    shape=(len(indices),window,value.shape[-1]))
                                files[filename]=dict(split=split,space=space,feature=kind,shape=list(arrays[key].shape))
                            arrays[key][offset:offset+bs]=value.reshape(bs,window,-1).cpu().numpy()
                        if space=='pooled4x4':
                            raw_code=projected['rp64'].reshape(bs,window,64).cpu().numpy()
                            normalized=(raw_code-mean)/scale
                            expected=references[split][offset:offset+bs]
                            np.testing.assert_allclose(normalized,expected,rtol=2e-5,atol=2e-5)
                            recovered_raw=expected*scale+mean
                            np.testing.assert_allclose(raw_code,recovered_raw,rtol=2e-5,atol=2e-5)
                            normalized_error=max(normalized_error,float(np.max(np.abs(normalized-expected))))
                            raw_error=max(raw_error,float(np.max(np.abs(raw_code-recovered_raw))))
                offset+=bs
            assert offset==len(indices)
            for value in arrays.values():value.flush()
            del arrays
            if ds.file is not None:ds.file.close()
            validation[split]=dict(rp64_normalized_reference_max_abs_error=normalized_error,
                rp64_raw_recovered_reference_max_abs_error=raw_error,rtol=2e-5,atol=2e-5,passed=True)
    for name,record in files.items():record.update(sha256=pose.digest(out/name),bytes=(out/name).stat().st_size)
    report=dict(seed=args.seed,run=str(run),checkpoint_step=600,weights_sha256=calibration['weights_sha256'],
        config_sha256=pose.digest(run/'config.json'),calibration=str(cal),audit_directory=str(audit_dir),
        calibration_sha256=pose.digest(cal/'calibration.json'),audit_sha256=pose.digest(audit_dir/'state-audit.json'),
        training_reference_sha256=pose.digest(cal/'training-embeddings.npz'),validation_reference_sha256=pose.digest(audit_dir/'state-audit.npz'),
        training_manifest_sha256=calibration['train_manifest_sha256'],validation_manifest_sha256=audit['manifest_sha256'],
        data=str(data),manifests=manifests,files=files,dimensions=dimensions,patch_grid_side=tokens_side,
        projection_seed=PROJECTION_SEED,projection_rule='64-column blocks: SeedSequence([seed,input_dim]) first; [seed,input_dim,block] for1,2. Concatenated192 divided by sqrt3.',
        projection_records=projection_records,carrier_code_reproduction=validation,
        encoder_calls=encoder_calls,encoded_frames=1536,device=str(device),precision='float32/highest, no autocast',
        model_source_sha256=pose.digest(carrier_source.__file__),source_sha256=pose.digest(__file__),
        numpy_version=np.__version__,torch_version=torch.__version__,seconds=time.time()-started,
        scope='Frozen encoder; exact existing train/validation RGB windows. No physical/goal/action labels are used in encoding. Full features and projections retained on EFS; no optimizer or model/policy update.')
    (out/'extraction.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seed=args.seed,seconds=report['seconds'],dimensions=dimensions,code_reproduction=validation)),flush=True)


def standardized_grams(train,test):
    center=train.mean(0);scale=np.maximum(train.std(0),1e-12)
    x=(train-center)/scale;tx=(test-center)/scale
    gram=x@x.T;cross=tx@x.T
    return gram,cross,np.square(x).sum(1),np.square(tx).sum(1),dict(std_min=float(scale.min()),
        std_max=float(scale.max()),floored_coordinates=int((train.std(0)<1e-12).sum()),
        standardized_gram_mean_diagonal=float(np.trace(gram)/len(gram)))


def readouts_from_grams(gram,cross,train_norm,test_norm,target,with_rbf):
    n=len(gram);mean=target.mean(0);centered=target-mean
    alpha=np.linalg.solve(gram+n*RIDGE*np.eye(n),centered)
    outputs=dict(linear=(gram@alpha+mean,cross@alpha+mean,dict(ridge_mean_loss_units=RIDGE,dual=True)))
    if with_rbf:
        distance=np.maximum(train_norm[:,None]+train_norm[None]-2*gram,0.)
        bandwidth=float(np.median(np.sqrt(distance[np.triu_indices(n,k=1)])))
        if not math.isfinite(bandwidth)or bandwidth<=0:raise ValueError('Nonpositive training-median bandwidth')
        kernel=np.exp(-distance/(2*bandwidth**2))
        test_distance=np.maximum(test_norm[:,None]+train_norm[None]-2*cross,0.)
        test_kernel=np.exp(-test_distance/(2*bandwidth**2))
        columns,grand=kernel.mean(0),kernel.mean()
        centered_kernel=kernel-kernel.mean(1,keepdims=True)-columns[None]+grand
        centered_test=test_kernel-test_kernel.mean(1,keepdims=True)-columns[None]+grand
        alpha=np.linalg.solve(centered_kernel+n*RIDGE*np.eye(n),centered)
        outputs['rbf']=(centered_kernel@alpha+mean,centered_test@alpha+mean,
            dict(ridge_mean_loss_units=RIDGE,bandwidth=bandwidth,kernel_centered=True,
                 bandwidth_selection='All off-diagonal training distances; fixed median, no validation tuning'))
    return outputs


def load_xyz(h5,manifest):
    rows=np.asarray([r['rows']for r in manifest['windows']]);unique,inverse=np.unique(rows.reshape(-1),return_inverse=True)
    expected=np.repeat([r['episode']for r in manifest['windows']],rows.shape[1])
    np.testing.assert_array_equal(h5['ep_idx'][unique][inverse],expected)
    return h5['privileged_block_0_pos'][unique][inverse].reshape(*rows.shape,3)


def score(prediction,target,quaternions,train_target,train_q,windows):
    orientation=prediction[:,:81];xyz=prediction[:,81:]
    decoded,_=pose.decode_codebook(orientation,train_target[:,:81],train_q)
    angles=pose.cube_symmetry_angle_deg(decoded,quaternions)
    error_m=np.linalg.norm(xyz-target[:,81:],axis=-1)
    return dict(orientation_features=pose.feature_metrics(orientation,target[:,:81],train_target[:,:81].mean(0)),
        orientation_angles=pose.angle_metrics(angles,windows),
        xyz_features=pose.feature_metrics(xyz,target[:,81:],train_target[:,81:].mean(0)),
        xyz_euclidean=dict(mean_cm=float(error_m.mean()*100),median_cm=float(np.median(error_m)*100),
            p90_cm=float(np.quantile(error_m,.9)*100),rmse_cm=float(np.sqrt(np.square(error_m).mean())*100),
            fraction_within3cm=float((error_m<=.03).mean()),per_frame_cm=(error_m*100).tolist()),
        joint_readout_within3cm_and15deg_fraction=float(((error_m<=.03)&(angles<=15)).mean()))


def analyze(args):
    import h5py,hdf5plugin  # noqa: F401
    features=Path(args.features);meta=pose.read(features/'extraction.json')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'results.json').exists():raise FileExistsError('Preserve the predeclared analysis')
    manifests=meta['manifests'];data=Path(meta['data']);card=pose.read(data/'dataset_card.json')
    for split,manifest in manifests.items():assert_metadata(data,manifest,split)
    with h5py.File(card['h5_path'],'r',swmr=True)as h5:
        train_q=pose.load_quaternions(h5,manifests['train']).reshape(-1,4)
        val_q=pose.load_quaternions(h5,manifests['validation'])[:,1:].reshape(-1,4)
        train_xyz=load_xyz(h5,manifests['train']).reshape(-1,3)
        val_xyz=load_xyz(h5,manifests['validation'])[:,1:].reshape(-1,3)
    train_y=np.concatenate([pose.invariant(train_q),train_xyz],axis=-1)
    val_y=np.concatenate([pose.invariant(val_q),val_xyz],axis=-1)
    rows=[];predictions={};started=time.time()
    for space in SPACES:
        for feature in FEATURES:
            names=[split+'__'+space+'__'+feature+'.npy'for split in ['train','validation']]
            for name in names:assert pose.digest(features/name)==meta['files'][name]['sha256']
            train=np.load(features/names[0],mmap_mode='r').reshape(1024,-1).astype(np.float64)
            val=np.load(features/names[1],mmap_mode='r')[:,1:].reshape(448,-1).astype(np.float64)
            gram,cross,tn,vn,standardization=standardized_grams(train,val)
            outputs=readouts_from_grams(gram,cross,tn,vn,train_y,space=='pooled4x4'and feature in ['rp64','full'])
            for kind,(train_prediction,val_prediction,configuration)in outputs.items():
                row=dict(seed=meta['seed'],space=space,feature=feature,dimensions=train.shape[1],readout=kind,
                    configuration=configuration,standardization=standardization,
                    train=score(train_prediction,train_y,train_q,train_y,train_q,128),
                    validation=score(val_prediction,val_y,val_q,train_y,train_q,64))
                rows.append(row);predictions[space+'__'+feature+'__'+kind]=val_prediction
            del train,val,gram,cross,outputs
    constant=np.repeat(train_y.mean(0,keepdims=True),len(val_y),axis=0)
    constant_result=score(constant,val_y,val_q,train_y,train_q,64)
    oracle_result=score(val_y,val_y,val_q,train_y,train_q,64)
    # The original RP64 linear/RBF diagnostics used equivalent affine-normalized
    # features. Retain numerical differences; do not refit or tune anything.
    prior_linear=pose.read(B/'reports/carrier-orientation600-20261031/results.json')
    prior_rbf=pose.read(B/'reports/carrier-orientation-rbf600-20261031/results.json')
    old_linear=next(r for r in prior_linear['rows']if r['seed']==meta['seed']and r['arm']=='identity')
    old_rbf=next(r for r in prior_rbf['rows']if r['seed']==meta['seed'])
    baseline_differences={}
    for kind in ['linear','rbf']:
        new=next(r for r in rows if r['space']=='pooled4x4'and r['feature']=='rp64'and r['readout']==kind)['validation']
        old=old_linear['scores']['cube24_tensor81']['observed_future']if kind=='linear'else old_rbf['validation']['rbf']
        baseline_differences[kind]=dict(invariant_r2_difference=new['orientation_features']['variance_weighted_r2']-old['features']['variance_weighted_r2'],
            mean_angle_difference_deg=new['orientation_angles']['mean_deg']-old['angles']['mean_deg'])
    np.savez_compressed(out/'readout-predictions.npz',**predictions)
    report=dict(seed=meta['seed'],rows=rows,constant_reference=constant_result,true_label_decoder_reference=oracle_result,
        rp64_existing_probe_differences=baseline_differences,extraction=str(features),extraction_sha256=pose.digest(features/'extraction.json'),
        source_sha256=pose.digest(__file__),orientation_helper_sha256=pose.digest(pose.__file__),
        training_clips=128,training_frames=1024,validation_clips=64,validation_future_frames=448,ridge=RIDGE,
        seconds=time.time()-started,no_validation_tuning=True,
        scope='Fixed readout diagnostic, not a training algorithm. Full features use primal-equivalent dual ridge; RBF only two predeclared4x4 controls.1024 correlated frames are128 episodes; high-dimensional failure does not prove missing information. Joint3cm/15deg readout accuracy is neither planning success nor three-step hold.')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Frozen Cube spatial readout diagnostic','',
        '| Space | Features | Readout | Train pose R² | Val pose R² | Mean angle | ≤15° | Mean xyz cm | ≤3 cm | Joint |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        train=row['train'];val=row['validation']
        lines.append(f"| {row['space']} | {row['feature']} ({row['dimensions']}) | {row['readout']} | "
            f"{train['orientation_features']['variance_weighted_r2']:.4f} | {val['orientation_features']['variance_weighted_r2']:.4f} | "
            f"{val['orientation_angles']['mean_deg']:.2f}° | {100*val['orientation_angles']['fraction_within15deg']:.2f}% | "
            f"{val['xyz_euclidean']['mean_cm']:.2f} | {100*val['xyz_euclidean']['fraction_within3cm']:.2f}% | "
            f"{100*val['joint_readout_within3cm_and15deg_fraction']:.2f}% |")
    lines+=['',report['scope'],'',
        'Ridge is fixed in coordinate-standardized mean-loss units, not matched effective degrees of freedom across dimensions. Readout differences combine accessible information and finite-sample inductive bias. Constant and training-codebook quantization references, all frame errors and RP64 reproduction checks are retained in results.json.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(seed=meta['seed'],seconds=report['seconds'],rows=[dict(space=r['space'],feature=r['feature'],readout=r['readout'],
        orientation_r2=r['validation']['orientation_features']['variance_weighted_r2'],
        angle_deg=r['validation']['orientation_angles']['mean_deg'],xyz_cm=r['validation']['xyz_euclidean']['mean_cm'],
        xyz_within3cm=r['validation']['xyz_euclidean']['fraction_within3cm'],
        joint=r['validation']['joint_readout_within3cm_and15deg_fraction'])for r in rows]),indent=2),flush=True)


def contracts():
    from experiments.carrier_progress_20261031.orientation_rbf_probe import kernel_fit_predict
    rng=np.random.default_rng(32);errors=[]
    for dim in [7,83]:
        x=rng.normal(size=(23,dim));tx=rng.normal(size=(11,dim));y=rng.normal(size=(23,5))
        gram,cross,tn,vn,_=standardized_grams(x,tx)
        outputs=readouts_from_grams(gram,cross,tn,vn,y,True)
        w,b,_=pose.ridge_fit(x,y)
        error=max(float(np.max(np.abs(outputs['linear'][0]-(x@w+b)))),float(np.max(np.abs(outputs['linear'][1]-(tx@w+b)))))
        assert error<1e-10;errors.append(error)
        train,test,config=kernel_fit_predict(x,y,tx)
        np.testing.assert_allclose(outputs['rbf'][0],train,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(outputs['rbf'][1],test,rtol=1e-11,atol=1e-11)
        assert outputs['rbf'][2]['bandwidth']==config['bandwidth']
    p64,p192=projections(3072)
    expected=(np.random.default_rng(np.random.SeedSequence([20261030,3072])).standard_normal((3072,64))/8).astype(np.float32)
    np.testing.assert_array_equal(p64,expected)
    np.testing.assert_array_equal(p192[:,:64],p64/np.float32(math.sqrt(3)))
    print(json.dumps(dict(passed=True,dual_primal_max_abs_errors=errors,rbf_matches_existing_formula=True,
        nested_rp64_original_formula_exact=True,no_model_or_encoder_calls=True)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['extract','analyze','contracts'])
    parser.add_argument('--seed',type=int,choices=[1,2]);parser.add_argument('--output')
    parser.add_argument('--features');parser.add_argument('--device',default='cpu');parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.phase=='contracts':contracts()
    elif args.phase=='extract':
        if args.seed is None or args.output is None:parser.error('extract requires --seed and --output')
        extract(args)
    else:
        if args.features is None or args.output is None:parser.error('analyze requires --features and --output')
        analyze(args)


if __name__=='__main__':main()
