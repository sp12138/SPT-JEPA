"""Three predeclared1024-D fixed measurements of saved Cube patch features.

CPU-only: no encoder, world-model, simulator, GPU, or validation tuning.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from experiments.carrier_progress_20261031 import carrier_spatial_probe as old
from experiments.carrier_progress_20261031 import orientation_probe as pose

B=Path('/efs/prescottsun/SPT-JEPA')
SEED=20261030
OUT=1024
BLOCK=128
MODES=('patch16_shared4','cell8_shared16','dense1024')


def channel_projection():
    generator=np.random.default_rng(np.random.SeedSequence([SEED,192,16]))
    return generator.standard_normal((192,16)).astype(np.float32)


def local_project(flat,matrix,pool_factor):
    grid=np.asarray(flat).reshape(-1,16,16,192)
    if pool_factor==2:
        grid=grid.reshape(-1,8,2,8,2,192).mean(axis=(2,4))
    else:
        assert pool_factor==1
    return (grid@matrix).reshape(len(flat),OUT)


def dense_project(inputs):
    dim=next(iter(inputs.values())).shape[1]
    outputs={name:np.empty((len(x),OUT),dtype=np.float32) for name,x in inputs.items()}
    blocks=[]
    for block,start in enumerate(range(0,OUT,BLOCK)):
        seed=[SEED,dim,OUT,block]
        matrix=np.random.default_rng(np.random.SeedSequence(seed)).standard_normal((dim,BLOCK),dtype=np.float32)
        matrix*=np.float32(1/math.sqrt(OUT))
        for name,x in inputs.items():outputs[name][:,start:start+BLOCK]=x@matrix
        blocks.append(dict(start=start,columns=BLOCK,seed_sequence=seed,sha256=old.array_hash(matrix)))
    return outputs,blocks


def contracts():
    matrix=channel_projection();w4=matrix[:,:4]/np.float32(2.);w16=matrix/np.float32(4.)
    impulse=np.zeros((1,16,16,192),dtype=np.float32);impulse[0,3,5,7]=1.
    fine=local_project(impulse.reshape(1,-1),w4,1).reshape(1,16,16,4)
    coarse=local_project(impulse.reshape(1,-1),w16,2).reshape(1,8,8,16)
    np.testing.assert_array_equal(fine[0,3,5],w4[7])
    np.testing.assert_array_equal(coarse[0,1,2],w16[7]/4)
    assert np.count_nonzero(np.linalg.norm(fine,axis=-1))==1
    assert np.count_nonzero(np.linalg.norm(coarse,axis=-1))==1
    tiny=np.arange(12,dtype=np.float32).reshape(2,6)
    blocks_a,record_a=dense_project(dict(train=tiny,validation=tiny[:1]))
    blocks_b,record_b=dense_project(dict(train=tiny,validation=tiny[:1]))
    assert record_a==record_b
    np.testing.assert_array_equal(blocks_a['train'],blocks_b['train'])
    np.testing.assert_allclose(blocks_a['train'][:1],blocks_a['validation'],rtol=2e-6,atol=2e-6)
    return dict(passed=True,spatial_address_preserved=True,pooling_is_mean2x2=True,
                shared_channel_prefix=True,dense_deterministic=True,output_dimensions=OUT)


def run(args):
    root=Path(args.features_root);out=Path(args.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Preserve the predeclared probe result')
    out.mkdir(parents=True,exist_ok=True)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    check=contracts()
    protocol=dict(projection_seed=SEED,representations=list(MODES),output_dimensions=OUT,
        channel_matrix_rule='N(0,1), SeedSequence([20261030,192,16]); first4/sqrt4 or all16/sqrt16, shared at every address.',
        dense_rule='Independent128-column blocks: SeedSequence([20261030,input_dim,1024,block]); N(0,1/1024).',
        dense_projection_block_columns=BLOCK,training_clips=128,training_frames=1024,
        validation_clips=64,validation_future_frames=448,readouts=['linear','rbf'],
        coordinate_standardization='Training mean/std only, floor1e-12.',
        ridge_mean_loss_units=old.RIDGE,rbf_bandwidth='Median off-diagonal standardized training distance.',
        label_use='Existing diagnostic readouts only; neither projection nor feature extraction sees labels.',
        original_feature_root=str(root),source_sha256=pose.digest(__file__),
        spatial_helper_sha256=pose.digest(old.__file__),orientation_helper_sha256=pose.digest(pose.__file__))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];provenance=[];references=[];started=time.time();predictions={}
    matrix=channel_projection();w4=matrix[:,:4]/np.float32(2.);w16=matrix/np.float32(4.)
    for seed in (1,2):
        features=root/f'seed{seed}-features';meta=pose.read(features/'extraction.json')
        prior_path=root/f'seed{seed}-readouts/results.json';prior=pose.read(prior_path)
        assert pose.digest(old.__file__)==prior['source_sha256']
        assert pose.digest(pose.__file__)==prior['orientation_helper_sha256']
        assert meta['seed']==seed and meta['patch_grid_side']==16 and meta['dimensions']['fullpatch']==49152
        assert meta['checkpoint_step']==600
        manifests=meta['manifests'];data=Path(meta['data']);card=pose.read(data/'dataset_card.json')
        for split,manifest in manifests.items():old.assert_metadata(data,manifest,split)
        assert not {x['episode'] for x in manifests['train']['windows']} & {x['episode'] for x in manifests['validation']['windows']}
        assert [len(manifests[k]['windows']) for k in ('train','validation')]==[128,64]
        inputs={};source_arrays={}
        for split in ('train','validation'):
            name=split+'__fullpatch__full.npy';path=features/name
            assert pose.digest(path)==meta['files'][name]['sha256']
            value=np.load(path,mmap_mode='r')
            assert value.shape==((128,8,49152) if split=='train' else (64,8,49152))
            inputs[split]=value.reshape(1024,-1) if split=='train' else value[:,1:].reshape(448,-1)
            source_arrays[split]=dict(path=str(path),sha256=meta['files'][name]['sha256'],shape=list(value.shape))
        # All three fixed representations are produced before physical labels
        # are loaded. They cannot adapt to this seed's probe performance.
        projected={
            'patch16_shared4':{k:local_project(v,w4,1) for k,v in inputs.items()},
            'cell8_shared16':{k:local_project(v,w16,2) for k,v in inputs.items()},
        }
        projected['dense1024'],block_records=dense_project(inputs)
        with h5py.File(card['h5_path'],'r',swmr=True) as h5:
            train_q=pose.load_quaternions(h5,manifests['train']).reshape(-1,4)
            val_q=pose.load_quaternions(h5,manifests['validation'])[:,1:].reshape(-1,4)
            train_xyz=old.load_xyz(h5,manifests['train']).reshape(-1,3)
            val_xyz=old.load_xyz(h5,manifests['validation'])[:,1:].reshape(-1,3)
        train_y=np.concatenate([pose.invariant(train_q),train_xyz],axis=-1)
        val_y=np.concatenate([pose.invariant(val_q),val_xyz],axis=-1)
        feature_records={}
        for mode in MODES:
            x=projected[mode]['train'];vx=projected[mode]['validation']
            assert x.shape==(1024,OUT) and vx.shape==(448,OUT)
            assert np.isfinite(x).all() and np.isfinite(vx).all()
            feature_records[mode]={k:old.array_hash(v) for k,v in projected[mode].items()}
            gram,cross,tn,vn,std=old.standardized_grams(x.astype(np.float64),vx.astype(np.float64))
            readouts=old.readouts_from_grams(gram,cross,tn,vn,train_y,True)
            for kind,(tp,vp,configuration) in readouts.items():
                row=dict(seed=seed,representation=mode,dimensions=OUT,readout=kind,configuration=configuration,
                    standardization=std,train=old.score(tp,train_y,train_q,train_y,train_q,128),
                    validation=old.score(vp,val_y,val_q,train_y,train_q,64))
                rows.append(row)
                predictions[f'seed{seed}__{mode}__{kind}__train']=tp
                predictions[f'seed{seed}__{mode}__{kind}__validation']=vp
                print(json.dumps(dict(seed=seed,representation=mode,readout=kind,
                    train_xyz_cm=row['train']['xyz_euclidean']['mean_cm'],
                    validation_xyz_cm=row['validation']['xyz_euclidean']['mean_cm'],
                    train_pose_r2=row['train']['orientation_features']['variance_weighted_r2'],
                    validation_pose_r2=row['validation']['orientation_features']['variance_weighted_r2'],
                    angle_deg=row['validation']['orientation_angles']['mean_deg'],
                    within3cm=row['validation']['xyz_euclidean']['fraction_within3cm'],
                    joint=row['validation']['joint_readout_within3cm_and15deg_fraction'])),flush=True)
        references.append(dict(seed=seed,artifact=dict(path=str(prior_path),sha256=pose.digest(prior_path)),
            rows=[r for r in prior['rows'] if (r['space'],r['feature']) in [('pooled4x4','rp64'),('pooled4x4','full'),('fullpatch','full')]],
            constant_reference=prior['constant_reference'],true_label_decoder_reference=prior['true_label_decoder_reference']))
        provenance.append(dict(seed=seed,extraction=dict(path=str(features/'extraction.json'),sha256=pose.digest(features/'extraction.json')),
            source_arrays=source_arrays,training_manifest_sha256=meta['training_manifest_sha256'],
            validation_manifest_sha256=meta['validation_manifest_sha256'],
            projection_sha256=dict(shared4=old.array_hash(w4),shared16=old.array_hash(w16)),
            dense_projection_blocks=block_records,projected_feature_sha256=feature_records))
        del projected,inputs,gram,cross,readouts
    assert len(rows)==12
    np.savez_compressed(out/'readout-predictions.npz',**predictions)
    report=dict(protocol=protocol,contracts=check,rows=rows,provenance=provenance,previous_references=references,
        seconds=time.time()-started,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        predictions_sha256=pose.digest(out/'readout-predictions.npz'),
        no_gpu_or_encoder_calls=True,no_validation_tuning=True,
        interpretation='Exactly three fixed1024-D measurements, two predeclared readouts and both seeds. '
            'Physical labels train only diagnostic readouts.1024 correlated training frames are128 episodes. '
            'High-dimensional ridge/RBF failure does not prove absent information; readout scores are not planning success. '
            'Local-vs-dense differences mix retained information, covariance and finite-sample readout bias. '
            'A new world-model algorithm and broad novelty are not established.')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Fixed local-vs-global1024-D measurement diagnostic','',
        '| Seed | Measurement | Readout | Train pose R² | Val pose R² | Train xyz cm | Val xyz cm | Val angle | ≤3cm | ≤15° | Joint |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t,v=r['train'],r['validation']
        lines.append(f"| {r['seed']} | {r['representation']} | {r['readout']} | "
            f"{t['orientation_features']['variance_weighted_r2']:.4f} | {v['orientation_features']['variance_weighted_r2']:.4f} | "
            f"{t['xyz_euclidean']['mean_cm']:.2f} | {v['xyz_euclidean']['mean_cm']:.2f} | "
            f"{v['orientation_angles']['mean_deg']:.2f}° | {100*v['xyz_euclidean']['fraction_within3cm']:.2f}% | "
            f"{100*v['orientation_angles']['fraction_within15deg']:.2f}% | {100*v['joint_readout_within3cm_and15deg_fraction']:.2f}% |")
    lines+=['',report['interpretation'],'',f"CPU wall time {report['seconds']:.1f}s; process peakRSS {report['peak_rss_gib']:.2f}GiB. "
            'Same original readout helpers, train-only standardization, mean-loss ridge.001, training-median RBF bandwidth, '
            'training-only quaternion codebook and cube24-invariant pose error. No extra parameter search.',
            '', 'All training and validation frame errors, prior RP64/full-feature controls, matrix/array hashes and prediction arrays are retained.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(out),rows=12,seconds=report['seconds'],peak_rss_gib=report['peak_rss_gib'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-root',default=str(B/'reports/carrier-spatial600-20261031'))
    parser.add_argument('--output',default=str(B/'reports/carrier-local1024-20261031'))
    args=parser.parse_args();run(args)
