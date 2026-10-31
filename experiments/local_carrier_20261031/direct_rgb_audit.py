"""One fixed Cube RGB-block measurement audit; CPU, no encoder or model calls."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import h5py,hdf5plugin  # noqa: F401
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.carrier_progress_20261031 import carrier_spatial_probe as old
from experiments.carrier_progress_20261031 import local_carrier_probe as local
from experiments.carrier_progress_20261031 import orientation_probe as pose
from experiments.local_carrier_20261031.summarize_calibration import stats

B=Path('/efs/prescottsun/SPT-JEPA')
FEATURES=B/'reports/carrier-spatial600-20261031'
PRIOR=B/'reports/carrier-local1024-20261031/results.json'
CAL=B/'runs/local_isotropic_calibration_20261031'
OUT=B/'reports/direct-rgb-information-20261031'
NOISE_SEED=1042
MEAN=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)


def artifact(path):return dict(path=str(path),sha256=pose.digest(path))


def pixels_to_measurement(pixels,matrix):
    assert pixels.dtype==torch.uint8 and pixels.shape[-3:]==(3,224,224)
    leading=pixels.shape[:-3];flat=pixels.reshape(-1,3,224,224)
    normalized=(flat.float()/255.-MEAN)/STD
    recovered=((normalized*STD+MEAN)*255.).round().clamp(0,255).byte()
    assert torch.equal(recovered,flat)
    render=F.interpolate(recovered.float(),size=(64,64),mode='area').round().byte()
    # Addresses=(block row,block column); inside each block=(row,column,RGB).
    hwc=(render.float()/127.5-1.).permute(0,2,3,1)
    blocks=hwc.reshape(-1,8,8,8,8,3).permute(0,1,3,2,4,5).reshape(-1,64,192)
    values=(blocks@matrix).reshape(*leading,1024)
    return values.numpy(),render.reshape(*leading,3,64,64)


def run(args):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    out=Path(args.output);assert not out.exists()
    src=pose.read(args.source_record);manifest=pose.read(src['manifest'])
    assert manifest['source_commit']==src['source_commit']
    for name,sha in manifest['files'].items():assert pose.digest(Path(src['source'])/name)==sha
    assert pose.digest(__file__)==manifest['files']['experiments/local_carrier_20261031/direct_rgb_audit.py']
    out.mkdir();started=time.time();matrix=local.channel_projection()/np.float32(4.)
    assert matrix.shape==(192,16)
    # Check the exact persisted original local projection, without constructing a model.
    wp=B/'runs/local_isotropic600_20261031/cube-spatial-local-seed1/initial-model/weights.pt'
    weights=torch.load(wp,map_location='cpu',weights_only=True)
    np.testing.assert_array_equal(matrix,weights['spatial_projection'].numpy());del weights
    tm=torch.from_numpy(matrix)
    # Address/order contract uses a synthetic64 image before any data/labels.
    impulse=np.zeros((1,8,8,8,8,3),np.float32);impulse[0,2,3,5,6,1]=1
    blocks=impulse.transpose(0,1,3,2,4,5).reshape(1,64,192)
    measured=(blocks@matrix).reshape(1,8,8,16)
    np.testing.assert_array_equal(measured[0,2,5],matrix[(3*8+6)*3+1]);assert np.count_nonzero(np.linalg.norm(measured,axis=-1))==1
    metas=[pose.read(FEATURES/f'seed{s}-features/extraction.json') for s in (1,2)]
    assert metas[0]['manifests']==metas[1]['manifests']
    manifests=metas[0]['manifests'];data=Path(metas[0]['data']);card=pose.read(data/'dataset_card.json')
    assert [len(manifests[k]['windows']) for k in ('train','validation')]==[128,64]
    for split,m in manifests.items():old.assert_metadata(data,m,split)
    assert not {v['episode'] for v in manifests['train']['windows']} & {v['episode'] for v in manifests['validation']['windows']}
    cals=[pose.read(CAL/f'calibrate-cube-seed{s}/calibration.json') for s in (1,2)]
    for key in ('selected_window_indices','training_episode_ids','starts','measured_h5_rows'):
        assert cals[0][key]==cals[1][key]
    cal=cals[0];assert cal['measured_frames']==128 and cal['dataset_card_sha256']==pose.digest(data/'dataset_card.json')
    assert len(set(cal['training_episode_ids']))==128 and set(cal['training_episode_ids'])<=set(np.load(data/'train_episodes.npy').tolist())
    protocol=dict(source=src,helper=artifact(__file__),task='cube',raw_seed_independent=True,
        training_clips=128,training_frames=1024,validation_clips=64,validation_future_frames=448,
        manifests=manifests,normalization='Original128 training initial frames: coordinate mean and one population globalRMS. No floor/tuning.',
        calibration=artifact(CAL/'calibrate-cube-seed1/calibration.json'),projection_rule='Exact existing shared192x16 buffer: SeedSequence([20261030,192,16]),N(0,1)/4.',
        projection_sha256=old.array_hash(matrix),projection_weights=artifact(wp),
        pixel_rule='Original224 uint8 -> ImageNet normalize -> inverse normalize/round224 -> area64/round. Must byte-match BenchmarkWindows.render.',
        block_order='64 row-major blocks; within8x8 block flatten HWC (RGB fastest); same projection at all addresses.',
        readout='Unchanged train-only coordinate-standardized ridge mean-loss.001 and training-median centered RBF; original cube24-invariant81 features and train quaternion codebook.',
        noise=dict(sigma=.1,seed=NOISE_SEED,kind='Independent single-frame Gaussian; identical draw per representation, train/validation separate; not an AR trajectory experiment.'),
        physical_labels='Diagnostic readout only; projection/calibration/measurement fixed before labels.',
        blocked_tworoom='No existing matching128train/64val local1024 feature cache; information audit used128/128 without saved arrays. No cohort invented or encoder called.',
        gpu_or_world_model_calls=0,training_authorized=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    raw={};native_equal_frames=0
    for split,m in manifests.items():
        ds=BenchmarkWindows(data,window=8,split=split);values=[]
        for record in m['windows']:
            batch=ds[record['index']]
            assert int(batch['episode'])==record['episode'] and batch['rows'].tolist()==record['rows']
            value,render=pixels_to_measurement(batch['pixels'],tm)
            assert torch.equal(render,batch['render']);native_equal_frames+=len(render);values.append(value)
        raw[split]=np.stack(values);ds.file.close()
    with h5py.File(card['h5_path'],'r',swmr=True) as h5:
        cal_values=[]
        for row,ep in zip(cal['measured_h5_rows'],cal['training_episode_ids']):
            assert int(h5['ep_idx'][row])==ep
            pixels=torch.from_numpy(np.asarray(h5['pixels'][row]).copy()).permute(2,0,1)[None]
            value,_=pixels_to_measurement(pixels,tm);cal_values.append(value[0])
        cal_values=np.stack(cal_values).astype(np.float64)
        mean=cal_values.mean(0);scale=float(np.sqrt(cal_values.var(0).mean()));assert np.isfinite(scale) and scale>0
        normalized={k:(v.astype(np.float64)-mean)/scale for k,v in raw.items()}
        # Feature arrays fixed before any privileged physical fields are read.
        tq=pose.load_quaternions(h5,manifests['train']).reshape(-1,4)
        vq=pose.load_quaternions(h5,manifests['validation'])[:,1:].reshape(-1,4)
        ty=np.concatenate([pose.invariant(tq),old.load_xyz(h5,manifests['train']).reshape(-1,3)],-1)
        vy=np.concatenate([pose.invariant(vq),old.load_xyz(h5,manifests['validation'])[:,1:].reshape(-1,3)],-1)
    reps={'rawRGB':normalized};references=[];prior=pose.read(PRIOR)
    assert pose.digest(old.__file__)==prior['protocol']['spatial_helper_sha256']
    assert pose.digest(pose.__file__)==prior['protocol']['orientation_helper_sha256']
    for seed,meta,calibration in zip((1,2),metas,cals):
        arrays={}
        for split in ('train','validation'):
            name=split+'__fullpatch__full.npy';path=FEATURES/f'seed{seed}-features'/name
            assert pose.digest(path)==meta['files'][name]['sha256'];patches=np.load(path,mmap_mode='r')
            projected=local.local_project(patches.reshape(-1,49152),matrix,2).reshape(len(patches),8,1024)
            kw=calibration['arms']['local']['model_kwargs'];ms=np.asarray(kw['spatial_scale']);assert len(set(ms))==1
            arrays[split]=(projected.astype(np.float64)-np.asarray(kw['spatial_mean']))/ms[0]
        reps[f'oldViT_seed{seed}']=arrays
        references.append(dict(seed=seed,extraction=artifact(FEATURES/f'seed{seed}-features/extraction.json'),
            calibration=artifact(CAL/f'calibrate-cube-seed{seed}/calibration.json'),prior_rows=[r for r in prior['rows'] if r['seed']==seed and r['representation']=='cell8_shared16']))
    rng=np.random.default_rng(NOISE_SEED);noise={k:rng.standard_normal(v.shape)*.1 for k,v in normalized.items()}
    rows=[];predictions={};energies={};reproduced=[]
    for name,arrays in reps.items():
        energies[name]={k:stats(v) for k,v in arrays.items()}
        for corruption in ('clean','sigma0.1'):
            z={k:v+(noise[k] if corruption!='clean' else 0) for k,v in arrays.items()}
            tx=z['train'].reshape(1024,1024);vx=z['validation'][:,1:].reshape(448,1024)
            gram,cross,tn,vn,standardization=old.standardized_grams(tx,vx)
            outputs=old.readouts_from_grams(gram,cross,tn,vn,ty,True)
            for readout,(tp,vp,configuration) in outputs.items():
                tr=old.score(tp,ty,tq,ty,tq,128);va=old.score(vp,vy,vq,ty,tq,64)
                row=dict(representation=name,corruption=corruption,readout=readout,train=tr,validation=va,configuration=configuration,standardization=standardization);rows.append(row)
                predictions[name+'__'+corruption+'__'+readout+'__train']=tp
                predictions[name+'__'+corruption+'__'+readout+'__validation']=vp
                if name.startswith('oldViT') and corruption=='clean':
                    seed=int(name[-1]);ref=next(r for r in prior['rows'] if r['seed']==seed and r['representation']=='cell8_shared16' and r['readout']==readout)
                    differences={k:abs(va[k][metric]-ref['validation'][k][metric]) for k,metric in [('xyz_euclidean','mean_cm'),('orientation_features','variance_weighted_r2'),('orientation_angles','mean_deg')]}
                    assert max(differences.values())<1e-5,differences;reproduced.append(dict(seed=seed,readout=readout,differences=differences))
                print(json.dumps(dict(representation=name,corruption=corruption,readout=readout,val_cm=va['xyz_euclidean']['mean_cm'],val_deg=va['orientation_angles']['mean_deg'])),flush=True)
    np.savez_compressed(out/'features-and-calibration.npz',raw_calibration=cal_values,mean=mean,scale=scale,projection=matrix,**{name+'__'+split:value for name,arr in reps.items() for split,value in arr.items()})
    np.savez_compressed(out/'predictions.npz',**predictions)
    result=dict(protocol=protocol,rows=rows,energies=energies,references=references,old_readout_reproduction=reproduced,
        numerical_contracts=dict(native_render_byte_equal_frames=native_equal_frames,normalized224_byte_roundtrip_exact=True,projection_buffer_equal=True,block_order_impulse_pass=True),
        raw_calibration=dict(global_rms=scale,min_coordinate_std=float(cal_values.std(0).min()),max_coordinate_std=float(cal_values.std(0).max()),floor_applied=False),
        files={n:artifact(out/n) for n in ('features-and-calibration.npz','predictions.npz')},
        prior=artifact(PRIOR),seconds=time.time()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        limitations='Cube only. Fixed diagnostic readout accessibility is not total information or planning. No bootstrap/model/projection tuning, new cohort, encoder/GPU/training, or TwoRoom extrapolation.')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary={k:v for k,v in result.items() if k not in ('protocol','rows','references')}
    summary['rows']=[dict(representation=r['representation'],corruption=r['corruption'],readout=r['readout'],train_xyz_cm=r['train']['xyz_euclidean']['mean_cm'],validation_xyz_cm=r['validation']['xyz_euclidean']['mean_cm'],train_pose_r2=r['train']['orientation_features']['variance_weighted_r2'],validation_pose_r2=r['validation']['orientation_features']['variance_weighted_r2'],validation_angle_deg=r['validation']['orientation_angles']['mean_deg'],within3cm=r['validation']['xyz_euclidean']['fraction_within3cm'],within15deg=r['validation']['orientation_angles']['fraction_within15deg'],joint=r['validation']['joint_readout_within3cm_and15deg_fraction']) for r in rows]
    summary['full_results']=artifact(out/'results.json');(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),rows=len(rows),seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True);p.add_argument('--output',default=str(OUT));run(p.parse_args())
