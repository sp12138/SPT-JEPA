"""Two failed TR TF graphs only: save frozen nodes, then CPU distance diagnosis."""
import argparse
from pathlib import Path
import time
from common import *

ORIGINAL=B/'runs/public_ngc_nodes_20261031'


def prepare(a):
    root=Path(a.root);assert not root.exists();source=read(a.source_record);source_check(source)
    script=Path(source['source'])/'experiments/public_ngc_20261031/radius_probe.py'
    assert sha(script)==sha(__file__);jobs=[];records=[]
    for seed,host,gpu in [(1,'dev2',7),(2,'dev13',3)]:
        ident=f'tworoom-TF64-seed{seed}';old=ORIGINAL/'specs'/(ident+'.json');s=read(old)
        assert read(ORIGINAL/'exits'/(ident+'.json'))['returncode']==1
        assert 'latent coincides with graph nodes' in (Path(s['output'])/'dispatch.log').read_text()
        verify(s);verify_cohort(s['cohort'],s['data'])
        rec=dict(id=ident,original_spec=str(old),original_spec_sha256=sha(old),model_source=s['model_source'],
            output=str(root/ident),helper_source=source,helper_sha256=sha(script),
            budget=dict(rgb=5000,encoder=157,predict=0,action_encoder=0,renderer=0))
        sp=root/'specs'/(ident+'.json');write(sp,rec);records.append(rec)
        jobs.append(dict(id=ident,host=host,gpu=gpu,hold=True,source=source,output=rec['output'],spec=str(sp),spec_sha256=sha(sp),
            command=[PY,'-B','-u',str(script),'capture','--spec',str(sp)]))
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},jobs=jobs))
    write(root/'protocol.json',dict(records=records,original_root=str(ORIGINAL),max_rgb=10000,max_encoder_calls=314,
        retained='All old failures, source, rows and evaluation HOLD remain unchanged.',
        purpose='Save the two exact frozen node sets before any NGC construction. Then CPU-only FP32-mm/direct32/direct64 radius and duplicate comparison.',
        prohibited='No graph rebuilding, planning, tuning, floor, changed k, centering treatment, predictor, renderer, training or new cases.'))
    (root/'HOLD_DISPATCH').write_text('Candidate only; activate after frozen helper and actual-free check.\n')
    print(json.dumps(dict(root=str(root),jobs=2,held=True)))


def capture(a):
    rec=read(a.spec)
    if not a.inner:reexec(rec,__file__)
    import inspect,h5py,numpy as np,torch
    import stable_worldmodel as swm
    from clear_lewm.runner import _image_transform
    from vforge.ops import OpsConfig,initialize_ops
    assert sha(__file__)==rec['helper_sha256']and sha(rec['original_spec'])==rec['original_spec_sha256']
    s=read(rec['original_spec']);verify(s);m=verify_cohort(s['cohort'],s['data'])
    out=Path(rec['output']);out.mkdir(exist_ok=True);assert not(out/'nodes.npz').exists()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=swm.wm.utils.load_pretrained(s['checkpoint']).cuda().eval().requires_grad_(False)
    weights=torch.load(Path(s['checkpoint'])/'weights.pt',map_location='cpu',weights_only=True)
    assert set(weights)==set(model.state_dict())and all(torch.equal(v,model.state_dict()[k].cpu())for k,v in weights.items());del weights
    actual=Path(inspect.getsourcefile(type(model))).resolve();assert actual.is_relative_to(Path(s['model_source']['source']))
    counts=dict(rgb=0,encoder=0,predict=0,action_encoder=0,renderer=0)
    def forbidden(*args,**kwargs):raise AssertionError('Only exact frozen RGB encoding authorized')
    model.predict=forbidden;model.rollout=forbidden
    hooks=[getattr(model,n).register_forward_pre_hook(forbidden)for n in ['action_encoder','renderer']if getattr(model,n,None)is not None]
    def counted(module,args):
        counts['rgb']+=len(args[0]);counts['encoder']+=1
        assert counts['rgb']<=5000 and counts['encoder']<=157
    hooks.append(model.encoder.register_forward_pre_hook(counted))
    before,rng_before=state_hash(model),rng_hash();rows=[x['row']for x in m['nodes']];transform=_image_transform(224)
    card=read(Path(s['data'])/'dataset_card.json');latents=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with h5py.File(card['h5_path'],'r',swmr=True)as h5,torch.inference_mode():
        np.testing.assert_array_equal(h5['ep_idx'][rows],[x['episode']for x in m['nodes']])
        for i in range(0,5000,32):
            pixels=torch.from_numpy(h5['pixels'][rows[i:i+32]]).permute(0,3,1,2)
            z=model.encode({'pixels':transform(pixels).cuda()[:,None]})['emb'][:,0]
            assert z.shape==(len(rows[i:i+32]),64)and torch.isfinite(z).all();latents.append(z.float().cpu().numpy())
            if i==0:write(out/'startup.json',dict(actual_model=str(actual),actual_model_sha256=sha(actual),counts=counts.copy(),finite=True))
    # This durable artifact is written before any radius/cost computation.
    nodes=np.concatenate(latents);np.savez_compressed(out/'nodes.npz',nodes=nodes,rows=np.asarray(rows))
    torch.cuda.synchronize();after,rng_after=state_hash(model),rng_hash()
    assert before==after and rng_before==rng_after and counts==rec['budget'];verify(s)
    assert all(not m.training for m in model.modules())and all(not p.requires_grad and p.grad is None for p in model.parameters())
    report=dict(passed=True,id=rec['id'],spec_sha256=sha(a.spec),original_spec_sha256=rec['original_spec_sha256'],
        nodes_sha256=sha(out/'nodes.npz'),cohort_sha256=sha(s['cohort']),counts=counts,state_before=before,state_after=after,
        rng_before=rng_before,rng_after=rng_after,actual_model=str(actual),actual_model_sha256=sha(actual),
        seconds=time.perf_counter()-start,peak_device_bytes=torch.cuda.max_memory_allocated(),
        scope='Exact5000 original train rows; saved float32 codes. No NGC constructor or distance kernel called on GPU.')
    write(out/'capture.json',report);print(json.dumps(dict(passed=True,id=rec['id'],counts=counts,seconds=report['seconds'])),flush=True)


def compare_nodes(nodes):
    import numpy as np,torch
    assert nodes.ndim==2 and nodes.dtype==np.float32 and np.isfinite(nodes).all()
    z=torch.from_numpy(nodes).cpu();double=z.double();radii={name:[]for name in ['fp32_mm','fp32_direct','fp64_direct']}
    counts=dict(mm_zero_but_direct_positive=0,direct32_zero_but_direct64_positive=0)
    maxerror=dict(fp32_mm=0.,fp32_direct=0.);sumerror=dict(fp32_mm=0.,fp32_direct=0.);pairs=0
    with torch.inference_mode():
        for i in range(0,len(z),32):
            d={'fp32_mm':torch.cdist(z[i:i+32],z,compute_mode='use_mm_for_euclid_dist'),
               'fp32_direct':torch.cdist(z[i:i+32],z,compute_mode='donot_use_mm_for_euclid_dist'),
               'fp64_direct':torch.cdist(double[i:i+32],double,compute_mode='donot_use_mm_for_euclid_dist')}
            for name,values in d.items():
                assert torch.isfinite(values).all();radii[name].extend(torch.sort(values,dim=1,stable=True).values[:,14].tolist())
            ref=d['fp64_direct'];pairs+=ref.numel()
            counts['mm_zero_but_direct_positive']+=int(((d['fp32_mm']==0)&(ref>0)).sum())
            counts['direct32_zero_but_direct64_positive']+=int(((d['fp32_direct']==0)&(ref>0)).sum())
            for name in maxerror:
                error=(d[name].double()-ref).abs();maxerror[name]=max(maxerror[name],float(error.max()));sumerror[name]+=float(error.sum())
    _,inverse,multiplicity=np.unique(nodes,axis=0,return_inverse=True,return_counts=True)
    exact_duplicate_radius=(multiplicity[inverse]>=15)
    arrays={k:np.asarray(v)for k,v in radii.items()};np.testing.assert_array_equal(arrays['fp64_direct']==0,exact_duplicate_radius)
    energy=float(np.square(nodes.astype(np.float64)).sum(1).mean());centered=nodes.astype(np.float64)-nodes.astype(np.float64).mean(0)
    report=dict(nodes=len(nodes),dimension=nodes.shape[1],unique_float32_rows=len(multiplicity),max_exact_multiplicity=int(multiplicity.max()),
        rows_with_at_least15_exact_copies=int(exact_duplicate_radius.sum()),mean_squared_norm=energy,
        total_centered_variance=float(np.square(centered).sum(1).mean()),feature_variance_quantiles=np.quantile(centered.var(0),[0,.5,1]).tolist(),
        radii={k:dict(zero_count=int((v==0).sum()),quantiles=np.quantile(v,[0,.01,.5,.99,1]).tolist())for k,v in arrays.items()},
        fp32_mm_radius_zero_but_fp64_positive=int(((arrays['fp32_mm']==0)&(arrays['fp64_direct']>0)).sum()),
        pair_counts=counts,all_pair_absolute_error={k:dict(max=maxerror[k],mean=sumerror[k]/pairs)for k in maxerror},
        precision='All distance comparisons run on CPU on the same saved float32 rows. CPU FP32-mm is the official formula but not a bitwise replay of its CUDA kernel.',
        scope='No fit, labels, graph change or production repair. Centered variance is descriptive only; no centered metric is evaluated.')
    return report,arrays


def compare(a):
    import numpy as np,torch
    root=Path(a.root);plan=read(root/'plan.json');assert len(plan['jobs'])==2;torch.set_num_threads(2);rows=[]
    for j in plan['jobs']:
        assert read(root/'exits'/(j['id']+'.json'))['returncode']==0 and sha(j['spec'])==j['spec_sha256']
        out=Path(j['output']);r=read(out/'capture.json');assert r['passed']and r['state_before']==r['state_after']and r['rng_before']==r['rng_after']
        assert sha(out/'nodes.npz')==r['nodes_sha256']and r['counts']==dict(rgb=5000,encoder=157,predict=0,action_encoder=0,renderer=0)
        with np.load(out/'nodes.npz')as f:nodes=f['nodes'].copy()
        result,arrays=compare_nodes(nodes);np.savez_compressed(out/'cpu-radii.npz',**arrays)
        result.update(id=j['id'],capture=r,cpu_radii_sha256=sha(out/'cpu-radii.npz'));write(out/'diagnosis.json',result);rows.append(result)
    write(root/'diagnosis.json',dict(passed=True,rows=rows,new_rgb=10000,new_encoder_calls=314,production_changed=False,planning_held=True))
    print(json.dumps(dict(passed=True,rows=[dict(id=r['id'],unique=r['unique_float32_rows'],zero_radii={k:v['zero_count']for k,v in r['radii'].items()})for r in rows])))


def contract(a):
    import numpy as np,torch
    torch.set_num_threads(1)
    nodes=np.ones((40,3),np.float32)*1000;nodes[:,0]+=np.arange(40,dtype=np.float32)*.001
    r,arrays=compare_nodes(nodes);assert r['unique_float32_rows']==40 and r['radii']['fp64_direct']['zero_count']==0
    np.testing.assert_allclose(arrays['fp32_direct'],arrays['fp64_direct'],rtol=1e-6,atol=1e-8)
    repeated=nodes.copy();repeated[:16]=nodes[0];d,_=compare_nodes(repeated);assert d['radii']['fp64_direct']['zero_count']==16
    write(a.output,dict(passed=True,tests=2,helper_sha256=sha(__file__),synthetic_cancellation=r,synthetic_duplicates=d,
        trained_model_calls=0,gpu_calls=0))
    print(json.dumps(dict(passed=True,tests=2)))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--root',required=True);q.add_argument('--source-record',required=True)
    q=sub.add_parser('capture');q.add_argument('--spec',required=True);q.add_argument('--inner',action='store_true')
    q=sub.add_parser('compare');q.add_argument('--root',required=True)
    q=sub.add_parser('contract');q.add_argument('--output',required=True)
    args=p.parse_args();globals()[args.mode](args)
