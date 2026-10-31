"""One predeclared CPU-only comparison on existing frozen Cube patch caches."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from experiments.carrier_progress_20261031 import carrier_spatial_probe as old
from experiments.carrier_progress_20261031 import local_carrier_probe as local
from experiments.carrier_progress_20261031 import orientation_probe as pose

B=Path('/efs/prescottsun/SPT-JEPA');SEED=20261030
read,sha=pose.read,pose.digest

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def cells(x):return x.reshape(-1,8,2,8,2,192).transpose(0,1,3,2,4,5).reshape(-1,64,4,192)
def stable_contrast(x):
    d=x-x[...,:1,:]
    return d-d.mean(axis=-2,keepdims=True)

def matrices():
    p=local.channel_projection()/np.float32(4.)
    g=np.random.default_rng(np.random.SeedSequence([SEED,768,16])).standard_normal((768,16)).astype(np.float32)/np.float32(8.)
    g=g.reshape(4,192,16);a=p.astype(np.float64)[None]/4+g.astype(np.float64)-g.astype(np.float64).mean(0,keepdims=True)
    return p,g,a

def project(flat,p,g):
    z0=local.local_project(flat,p,2)
    q=cells(flat);contrast=stable_contrast(q)
    z1=z0+(contrast.reshape(-1,64,768)@g.reshape(768,16)).reshape(-1,1024)
    return z0,z1

def tensor_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def contracts(p,g,a):
    v=np.arange(192,dtype=np.float32)/193
    q=np.tile(v,(2,64,4,1));flat=q.reshape(2,8,8,2,2,192).transpose(0,1,3,2,4,5).reshape(2,49152)
    z0,z1=project(flat,p,g);assert np.array_equal(z0,z1)
    delta=np.zeros((4,192));delta[0]=v;delta[1]=-v
    old_delta=delta.mean(0)@p;new_delta=delta.reshape(768)@a.reshape(768,16)
    assert np.count_nonzero(old_delta)==0 and np.linalg.norm(new_delta)>0
    np.testing.assert_allclose(a.sum(0),p,rtol=0,atol=1e-15)
    rank0=int(np.linalg.matrix_rank(np.tile(p.astype(np.float64)/4,(4,1))))
    rank1=int(np.linalg.matrix_rank(a.reshape(768,16)));assert rank0==rank1==16
    cov=np.full((4,4),1/256)+np.eye(4)/64-np.ones((4,4))/256
    np.testing.assert_array_equal(cov,np.eye(4)/64)
    return dict(passed=True,dc_fp32_exact=True,old_contrast_norm=float(np.linalg.norm(old_delta)),new_contrast_norm=float(np.linalg.norm(new_delta)),
        old_rank=rank0,new_rank=rank1,nullity_each=752,old_structural_zero_sum_dimension=576,coupled_gaussian_covariance=cov.tolist(),
        equivalent_A_block_sum_error=float(np.max(np.abs(a.sum(0)-p))))

def summarize_energy(flat,p,g,a,shape):
    # Accumulate original-patch DC/contrast energies, not cross-coordinate MSE.
    allrows=[];fp32_error=fp64_error=0.;old_null=0.
    for start in range(0,len(flat),64):
        x=np.asarray(flat[start:start+64]);q=cells(x).astype(np.float64);dc=q.mean(-2,keepdims=True);contrast=q-dc
        raw=(q*q).sum((1,2,3));ce=(contrast*contrast).sum((1,2,3));de=4*(dc*dc).sum((1,2,3))
        np.testing.assert_allclose(raw,de+ce,rtol=2e-13,atol=1e-12)
        delta=(contrast.reshape(-1,64,768)@g.reshape(768,16).astype(np.float64)).reshape(-1,1024)
        old_null=max(old_null,float(np.max(np.abs(contrast.mean(-2)@p))))
        z0,z1=project(x,p,g)
        direct=(cells(x).reshape(-1,64,768)@a.astype(np.float32).reshape(768,16)).reshape(-1,1024)
        ref=(q.reshape(-1,64,768)@a.reshape(768,16)).reshape(-1,1024)
        fp32_error=max(fp32_error,float(np.max(np.abs(z1-direct))))
        fp64_error=max(fp64_error,float(np.max(np.abs(z1-ref))))
        allrows.extend(np.stack([raw,de,ce,(delta*delta).sum(1),(z0.astype(np.float64)**2).sum(1),(z1.astype(np.float64)**2).sum(1)],1).tolist())
    e=np.asarray(allrows);s=e.sum(0)
    return dict(frames=len(e),full_patch_energy=float(s[0]),dc_energy=float(s[1]),contrast_energy=float(s[2]),
        contrast_fraction=float(s[2]/s[0])if s[0]>0 else 0.,
        added_contrast_measurement_energy=float(s[3]),contrast_gain_relative_to_expectation=float(s[3]/(.25*s[2]))if s[2]>0 else 0.,
        old_output_energy=float(s[4]),new_output_energy=float(s[5]),
        direct_A_fp32_max_abs_error=fp32_error,FP64_formula_max_abs_error=fp64_error,old_zero_sum_numerical_residual_max=old_null,
        per_clip_energies=e.reshape(*shape,6).mean(1).tolist())

def paired(a,b):
    d=np.asarray(a).reshape(64,7).mean(1)-np.asarray(b).reshape(64,7).mean(1)
    rng=np.random.default_rng(20261031);boots=d[rng.integers(64,size=(10000,64))].mean(1)
    return dict(delta=float(d.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),per_clip_delta=d.tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.time();p,g,amat=matrices();contract=contracts(p,g,amat)
    np.savez_compressed(out/'matrices.npz',old=p,independent_G=g,equivalent_A=amat)
    protocol=dict(source_sha256=sha(__file__),matrix_sha256=sha(out/'matrices.npz'),seed=SEED,
        matrix_rule='DC coupled: P old; G from SeedSequence([20261030,768,16])/8; A_i=P/4+G_i-Gbar.',
        designs_tested=['old_mean_then_projection','dc_coupled_concat'],no_matrix_selection=True,
        calls=dict(encoder=0,model_forward=0,renderer=0,simulator=0,gpu=0),physical_labels='Fixed diagnostic readouts only; never projection/calibration.',
        spatial_helper_sha256=sha(old.__file__),pose_helper_sha256=sha(pose.__file__),local_helper_sha256=sha(local.__file__))
    write(out/'protocol.json',protocol)
    prior=read(B/'reports/carrier-local1024-20261031/results.json')
    assert sha(old.__file__)==prior['protocol']['spatial_helper_sha256'] and sha(pose.__file__)==prior['protocol']['orientation_helper_sha256']
    import torch,h5py,hdf5plugin
    torch.set_num_threads(2);rows=[];provenance=[];energy=[];calibration=[];comparisons=[];predictions={}
    for seed in [1,2]:
        features=B/f'reports/carrier-spatial600-20261031/seed{seed}-features';meta=read(features/'extraction.json')
        manifests=meta['manifests'];data=Path(meta['data']);card=read(data/'dataset_card.json')
        for split,m in manifests.items():old.assert_metadata(data,m,split)
        assert not {w['episode']for w in manifests['train']['windows']} & {w['episode']for w in manifests['validation']['windows']}
        original=Path(meta['run'])/'model-step0000600/weights.pt';assert sha(original)==meta['weights_sha256']
        spec=read(B/f'runs/public_ngc_covariance_evaluation_20261031/specs/corr-l2-primary-cube-local1024-seed{seed}.json')
        current=Path(spec['checkpoint'])/'weights.pt';assert sha(current)==spec['immutable_inputs'][str(current)]
        w0=torch.load(original,map_location='cpu',weights_only=True,mmap=True);w1=torch.load(current,map_location='cpu',weights_only=True,mmap=True)
        e0={k[8:]:v for k,v in w0.items()if k.startswith('encoder.')};e1={k[8:]:v for k,v in w1.items()if k.startswith('encoder.')}
        assert set(e0)==set(e1) and all(torch.equal(e0[k],e1[k])for k in e0)
        np.testing.assert_array_equal(p,w1['spatial_projection'].numpy())
        calpath=B/f'runs/local_isotropic_calibration_20261031/calibrate-cube-seed{seed}/calibration.json';cal=read(calpath)
        assert tensor_hash(e0)==cal['encoder_initial_sha256']
        encoder_digest=tensor_hash(e0);del w0,w1,e0,e1
        arrays={};projected={};sources={}
        for split in ['train','validation']:
            path=features/(split+'__fullpatch__full.npy');assert sha(path)==meta['files'][path.name]['sha256']
            x=np.load(path,mmap_mode='r');assert x.shape==((128,8,49152)if split=='train'else(64,8,49152))
            arrays[split]=x;flat=x.reshape(-1,49152);projected[split]=project(flat,p,g)
            sources[split]=dict(path=str(path),sha256=sha(path),shape=list(x.shape))
            energy.append(dict(seed=seed,split=split,kind='raw_observation',**summarize_energy(flat,p,g,amat,x.shape[:2])))
            increments=np.asarray(x[:,1:]-x[:,:-1]);energy.append(dict(seed=seed,split=split,kind='temporal_increment',**summarize_energy(increments.reshape(-1,49152),p,g,amat,(len(x),7))))
        # Exact historical128 calibration rows all available; no new RGB.
        trainrows=[r for w in manifests['train']['windows']for r in w['rows']];assert len(set(trainrows))==len(trainrows)
        lookup={r:i for i,r in enumerate(trainrows)};indices=[lookup[r]for r in cal['measured_h5_rows']];assert len(indices)==128
        cals=[]
        for mode,raw in zip(['mean','concat'],projected['train']):
            z=raw[indices].astype(np.float64);mean=z.mean(0);sigma=float(np.sqrt(z.var(0).mean()));assert sigma>0
            cals.append(dict(mode=mode,mean=mean.tolist(),isotropic_scale=sigma,raw_calibration_sha256=old.array_hash(raw[indices])))
        oldcfg=cal['arms']['local']['model_kwargs'];meanerror=float(np.max(np.abs(np.asarray(oldcfg['spatial_mean'])-cals[0]['mean'])))
        scaleerror=abs(float(oldcfg['spatial_scale'][0])-cals[0]['isotropic_scale'])
        assert meanerror<2e-5 and scaleerror<2e-5
        calibration.append(dict(seed=seed,source=str(calpath),source_sha256=sha(calpath),rows=cal['measured_h5_rows'],old_mean_max_abs_error=meanerror,old_scale_abs_error=scaleerror,measurements=cals))
        # Only now are diagnostic labels loaded; projections are fully fixed.
        with h5py.File(card['h5_path'],'r',swmr=True)as h:
            tq=pose.load_quaternions(h,manifests['train']).reshape(-1,4);vq=pose.load_quaternions(h,manifests['validation'])[:,1:].reshape(-1,4)
            ty=np.concatenate([pose.invariant(tq),old.load_xyz(h,manifests['train']).reshape(-1,3)],1)
            vy=np.concatenate([pose.invariant(vq),old.load_xyz(h,manifests['validation'])[:,1:].reshape(-1,3)],1)
        for i,mode in enumerate(['mean','concat']):
            x=projected['train'][i].astype(np.float64);vx=projected['validation'][i].reshape(64,8,1024)[:,1:].reshape(448,1024).astype(np.float64)
            gram,cross,tn,vn,std=old.standardized_grams(x,vx);fitted=old.readouts_from_grams(gram,cross,tn,vn,ty,True)
            for kind,(tp,vp,configuration)in fitted.items():
                row=dict(seed=seed,measurement=mode,readout=kind,configuration=configuration,standardization=std,
                    train=old.score(tp,ty,tq,ty,tq,128),validation=old.score(vp,vy,vq,ty,tq,64))
                if mode=='mean':
                    reference=next(r for r in prior['rows']if r['seed']==seed and r['representation']=='cell8_shared16'and r['readout']==kind)
                    for field,metric in [('xyz_euclidean','mean_cm'),('orientation_angles','mean_deg'),('orientation_features','variance_weighted_r2')]:
                        assert abs(row['validation'][field][metric]-reference['validation'][field][metric])<2e-5,(seed,kind,field)
                    row['old_readout_reproduced']=True
                rows.append(row);predictions[f'seed{seed}-{mode}-{kind}']=vp
        for kind in ['linear','rbf']:
            oldr,newr=[next(r['validation']for r in rows if r['seed']==seed and r['measurement']==m and r['readout']==kind)for m in ['mean','concat']]
            comparisons.append(dict(seed=seed,readout=kind,xyz_cm=paired(newr['xyz_euclidean']['per_frame_cm'],oldr['xyz_euclidean']['per_frame_cm']),
                angle_deg=paired(newr['orientation_angles']['per_frame_deg'],oldr['orientation_angles']['per_frame_deg'])))
        provenance.append(dict(seed=seed,extraction=str(features/'extraction.json'),extraction_sha256=sha(features/'extraction.json'),sources=sources,
            cached_encoder_weights=str(original),cached_encoder_weights_sha256=meta['weights_sha256'],current_local_checkpoint=str(current.parent),
            current_weights_sha256=sha(current),encoder_tensors_exact=True,encoder_tensor_hash=encoder_digest,
            train_manifest_sha256=meta['training_manifest_sha256'],validation_manifest_sha256=meta['validation_manifest_sha256'],
            train_episodes=128,val_episodes=64,train_frames=1024,val_future_frames=448))
        print(json.dumps(dict(seed=seed,completed=True)),flush=True)
    np.savez_compressed(out/'probe-predictions.npz',**predictions)
    result=dict(passed=True,protocol=protocol,contracts=contract,provenance=provenance,energy=energy,calibration=calibration,rows=rows,comparisons=comparisons,
        constant_and_codebook_references=[dict(seed=r['seed'],constant=r['constant_reference'],true_label_codebook=r['true_label_decoder_reference'])for r in prior['previous_references']],
        prior_readout_path=str(B/'reports/carrier-local1024-20261031/results.json'),prior_readout_sha256=sha(B/'reports/carrier-local1024-20261031/results.json'),
        seconds=time.time()-started,scope='Observation-only fixed-probe diagnostic. No prediction/planning conclusion, no raw latent MSE cross-coordinate comparison, no novel algorithm claim. Both matrices rank16/cell; conditioning and retained directions differ.64 paired validation clips,128 training clips with correlated frames. Physical labels never select measurement.')
    write(out/'results.json',result)
    compact=json.loads(json.dumps(result));
    for c in compact['calibration']:
        for m in c['measurements']:m.pop('mean')
    for e in compact['energy']:e.pop('per_clip_energies')
    for row in compact['rows']:
        for split in ['train','validation']:
            row[split]['xyz_euclidean'].pop('per_frame_cm');row[split]['orientation_angles'].pop('per_frame_deg');row[split]['orientation_angles'].pop('per_window_mean_deg')
    compact.update(full_results_path=str(out/'results.json'),full_results_sha256=sha(out/'results.json'))
    write(out/'results-compact.json',compact)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],rows=8,output=str(out))),flush=True)

if __name__=='__main__':main()
