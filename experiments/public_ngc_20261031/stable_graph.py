"""Node-only derived graph and CUDA numerical gate; never loads a world model."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from common import *
from stable_cost import StableNGCCost
from vendor.cost import NGCCost
from vendor.graph import build_graph,neighbor_links


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();s=read(a.spec);verify(s)
    assert s['numeric_variant']=='uniform-fp32-direct-attachment'
    m=verify_cohort(s['cohort'],s['data']);origin=read(s['origin_report']);out=Path(s['output']);out.mkdir(exist_ok=True)
    assert not(out/'graph.json').exists()and not(out/'graph.npz').exists();start=time.perf_counter()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats();graph_seconds=0.
    if s['origin_kind']=='reuse-exact-graph':
        shutil.copyfile(s['origin_array'],out/'graph.npz');assert sha(out/'graph.npz')==origin['graph_sha256']
        with np.load(out/'graph.npz')as f:nodes=f['nodes'].copy();dist=f['graph_distance'].copy();rows=f['rows'].copy();bridges=f['bridges'].tolist()
        components=origin['original_components'];bridge_lengths=origin['bridge_lengths']
    else:
        assert s['origin_kind']=='rebuild-fp64-from-saved-nodes'
        from scipy.spatial.distance import cdist
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        with np.load(s['origin_array'])as f:nodes=f['nodes'].copy();rows=f['rows'].copy()
        tick=time.perf_counter();z,dist64,bridges=build_graph(nodes,NEIGHBORS,'mutual');graph_seconds=time.perf_counter()-tick
        np.testing.assert_array_equal(z,nodes.astype(np.float64));assert np.isfinite(dist64).all()
        distance=cdist(z,z);np.fill_diagonal(distance,np.inf);linked=neighbor_links(distance,NEIGHBORS,'mutual')
        components,_=connected_components(csr_matrix(linked),directed=False);bridge_lengths=[float(distance[i,j])for i,j in bridges]
        dist=dist64.astype(np.float32)
        np.savez_compressed(out/'graph.npz',nodes=nodes,graph_distance=dist,rows=rows,bridges=np.asarray(bridges,dtype=np.int64).reshape(-1,2))
        del z,dist64,distance,linked
    assert nodes.dtype==np.float32 and dist.dtype==np.float32 and nodes.shape==(5000,s['dimension'])and dist.shape==(5000,5000)
    assert np.isfinite(nodes).all()and np.isfinite(dist).all()and np.all(dist>=0)
    np.testing.assert_array_equal(rows,[x['row']for x in m['nodes']]);np.testing.assert_array_equal(np.diag(dist),np.zeros(NODES))
    np.testing.assert_allclose(dist,dist.T,rtol=1e-6,atol=0);assert len(bridges)==components-1
    cost=StableNGCCost(nodes,dist,NEIGHBORS,ANCHORS).cuda();z64=cost.nodes.double()
    old=NGCCost(nodes,dist,NEIGHBORS,ANCHORS).cuda()if s['origin_kind']=='rebuild-fp64-from-saved-nodes'else None
    before=rng_hash();radii=[];refs=[];oldradii=[];maxabs=maxrel=sumabs=0.;pairs=0;old_fail_chunks=0;mm_false_zero=0
    # Dimension-dependent floating arithmetic bound, not a metric/radius floor.
    relative_bound=8*(s['dimension']+4)*np.finfo(np.float32).eps
    write(out/'startup.json',dict(id=s['id'],origin_kind=s['origin_kind'],nodes_sha256=sha(s['origin_array']),
        model_calls=0,new_rgb=0,device=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),finite=True))
    with torch.inference_mode():
        for i in range(0,NODES,BATCH):
            q=cost.nodes[i:i+BATCH]
            direct=torch.cdist(q,cost.nodes,compute_mode='donot_use_mm_for_euclid_dist')
            ref=torch.cdist(z64[i:i+BATCH],z64,compute_mode='donot_use_mm_for_euclid_dist')
            assert torch.isfinite(direct).all()and torch.isfinite(ref).all()
            assert torch.equal(direct==0,ref==0)
            error=(direct.double()-ref).abs();positive=ref>0
            rel=float((error[positive]/ref[positive]).max());assert rel<=relative_bound
            maxabs=max(maxabs,float(error.max()));maxrel=max(maxrel,rel);sumabs+=float(error.sum());pairs+=ref.numel()
            radius=cost.attach(q)[2];reference=torch.sort(ref,dim=1,stable=True).values[:,14]
            assert torch.all(radius>0)and torch.all(reference>0)
            radii.extend(radius.cpu().tolist());refs.extend(reference.cpu().tolist())
            value=cost({'predicted_emb':q[:,None,None],'goal_emb':q[:,None]});assert torch.equal(value,torch.zeros_like(value))
            if old is not None:
                mm=torch.cdist(q,old.nodes,compute_mode='use_mm_for_euclid_dist')
                oldradius=torch.sort(mm,dim=1,stable=True).values[:,14]
                oldradii.extend(oldradius.cpu().tolist());mm_false_zero+=int(((mm==0)&positive).sum())
                try:old.attach(q)
                except AssertionError as e:
                    assert str(e)=='latent coincides with graph nodes';old_fail_chunks+=1
                    assert torch.any(oldradius==0)
    torch.cuda.synchronize();assert before==rng_hash();verify(s)
    np.savez_compressed(out/'radius-audit.npz',direct32=np.asarray(radii),direct64=np.asarray(refs),original_cuda_mm=np.asarray(oldradii))
    report=dict(passed=True,id=s['id'],spec_sha256=sha(a.spec),checkpoint=s['checkpoint'],
        weights_sha256=s['immutable_inputs'][str(Path(s['checkpoint'])/'weights.pt')],cohort=s['cohort'],cohort_sha256=sha(s['cohort']),
        graph_sha256=sha(out/'graph.npz'),dimension=s['dimension'],nodes=5000,k=15,anchors=4,rule='mutual',
        numeric_variant=s['numeric_variant'],origin_kind=s['origin_kind'],origin_report=s['origin_report'],origin_report_sha256=sha(s['origin_report']),
        origin_array=s['origin_array'],origin_array_sha256=sha(s['origin_array']),original_components=int(components),bridges=bridges,bridge_lengths=bridge_lengths,
        radius_quantiles=np.quantile(radii,[0,.01,.5,.99,1]).tolist(),radius64_quantiles=np.quantile(refs,[0,.01,.5,.99,1]).tolist(),
        distance_reference=dict(ordered_pairs=pairs,max_absolute_error=maxabs,mean_absolute_error=sumabs/pairs,max_relative_error=maxrel,
            dimension_relative_bound=relative_bound,reference='CUDA FP64 direct on promoted identical saved float32 nodes; no new encoding.'),
        original_cuda_mm=None if old is None else dict(zero_radius_count=int(np.count_nonzero(np.asarray(oldradii)==0)),
            radius_quantiles=np.quantile(oldradii,[0,.01,.5,.99,1]).tolist(),positive_distance_pairs_rounded_zero=mm_false_zero,attach_assertion_chunks=old_fail_chunks),
        state_before=origin['state_before'],state_after=origin['state_after'],state_provenance='Exact original capture state hashes; no model loaded or called in this derived job.',
        rng_before=before,rng_after=rng_hash(),counts=dict(rgb=0,encoder=0,predict=0,action_encoder=0,renderer=0),
        graph_seconds=graph_seconds,total_seconds=time.perf_counter()-start,peak_device_bytes=torch.cuda.max_memory_allocated(),
        radius_audit_sha256=sha(out/'radius-audit.npz'),torch_version=torch.__version__,
        scope='Only saved-node graph derivation and numerical gate. All original weights/buffers/cohorts/source unchanged; uniform stable attachment. No planning, floor, tuning or model calls.')
    write(out/'graph.json',report);print(json.dumps(dict(id=s['id'],passed=True,seconds=report['total_seconds'],minimum_radius=min(radii),original_cuda_mm=report['original_cuda_mm'])),flush=True)


if __name__=='__main__':main()
