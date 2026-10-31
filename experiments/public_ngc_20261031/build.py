"""Encode exactly5000 train-only images, then build the unmodified public graph."""
import argparse
import inspect
from pathlib import Path
import time
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--inner',action='store_true');a=p.parse_args()
    s=read(a.spec)
    if not a.inner: reexec(s,__file__)
    import h5py,numpy as np,torch
    import stable_worldmodel as swm
    from clear_lewm.runner import _image_transform
    from vforge.ops import OpsConfig,initialize_ops
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from vendor.graph import build_graph,neighbor_links
    from vendor.cost import NGCCost
    verify(s); m=verify_cohort(s['cohort'],s['data']); out=Path(s['output']);out.mkdir(exist_ok=True)
    assert not (out/'graph.json').exists() and not (out/'graph.npz').exists()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=swm.wm.utils.load_pretrained(s['checkpoint']).cuda().eval().requires_grad_(False)
    weights=torch.load(Path(s['checkpoint'])/'weights.pt',map_location='cpu',weights_only=True)
    assert set(weights)==set(model.state_dict()) and all(torch.equal(v,model.state_dict()[k].cpu())for k,v in weights.items());del weights
    actual=Path(inspect.getsourcefile(type(model))).resolve();assert actual.is_relative_to(Path(s['model_source']['source']))
    assert all(not x.training for x in model.modules())
    before,rng_before=state_hash(model),rng_hash();counts=dict(rgb=0,encoder=0,predict=0,action_encoder=0,renderer=0)
    def forbid(*args,**kwargs):raise AssertionError('Only frozen RGB encoding is authorized')
    model.predict=forbid;model.rollout=forbid
    hooks=[]
    for name in ['action_encoder','renderer']:
        if getattr(model,name,None) is not None:hooks.append(getattr(model,name).register_forward_pre_hook(forbid))
    def counted(module,args):
        counts['encoder']+=1;counts['rgb']+=len(args[0]);assert counts['encoder']<=157 and counts['rgb']<=5000
    hooks.append(model.encoder.register_forward_pre_hook(counted))
    transform=_image_transform(224);card=read(Path(s['data'])/'dataset_card.json');rows=[x['row']for x in m['nodes']]
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();emb=[]
    with h5py.File(card['h5_path'],'r',swmr=True)as h5,torch.inference_mode():
        assert np.array_equal(h5['ep_idx'][rows],np.array([x['episode']for x in m['nodes']]))
        for i in range(0,NODES,BATCH):
            pixels=torch.from_numpy(h5['pixels'][rows[i:i+BATCH]]).permute(0,3,1,2)
            z=model.encode({'pixels':transform(pixels).cuda()[:,None]})['emb'][:,0]
            assert z.ndim==2 and z.shape[0]==len(rows[i:i+BATCH]) and torch.isfinite(z).all()
            emb.append(z.float().cpu().numpy())
            if i==0:write(out/'startup.json',dict(actual_model=str(actual),source_sha256=sha(actual),
                rows=len(z),dimension=z.shape[1],finite=True,state_before=before,counts=counts.copy()))
    torch.cuda.synchronize();encoded_seconds=time.perf_counter()-started;latents=np.concatenate(emb)
    assert counts==dict(rgb=5000,encoder=157,predict=0,action_encoder=0,renderer=0)
    assert latents.shape==(NODES,s['dimension'])
    after,rng_after=state_hash(model),rng_hash();assert before==after and rng_before==rng_after
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    for h in hooks:h.remove()
    del model;torch.cuda.empty_cache()
    tick=time.perf_counter();nodes,distances,bridges=build_graph(latents,NEIGHBORS,'mutual')
    assert np.isfinite(nodes).all() and np.isfinite(distances).all() and np.all(distances>=0)
    np.testing.assert_array_equal(np.diag(distances),np.zeros(NODES))
    np.testing.assert_allclose(distances,distances.T,rtol=1e-12,atol=1e-12)
    graph_seconds=time.perf_counter()-tick;euclidean=cdist(nodes,nodes);np.fill_diagonal(euclidean,np.inf)
    linked=neighbor_links(euclidean,NEIGHBORS,'mutual');components,_=connected_components(csr_matrix(linked),directed=False)
    assert len(bridges)==components-1
    bridge_lengths=[float(euclidean[i,j])for i,j in bridges]
    degree=linked.sum(1);edge_lengths=euclidean[linked];del euclidean,linked
    objective=NGCCost(nodes,distances,NEIGHBORS,ANCHORS).cuda();radii=[]
    with torch.inference_mode():
        for i in range(0,NODES,BATCH):
            q=torch.from_numpy(latents[i:i+BATCH]).cuda()
            radii.extend(objective.attach(q)[2].cpu().tolist())
            # Finite, exact self-goal zeros; no validation/query-state tuning.
            value=objective({'predicted_emb':q[:,None,None], 'goal_emb':q[:,None]})
            assert torch.equal(value,torch.zeros_like(value))
    assert len(radii)==NODES and min(radii)>0
    np.savez_compressed(out/'graph.npz',nodes=nodes.astype(np.float32),graph_distance=distances.astype(np.float32),
                        rows=np.asarray(rows),bridges=np.asarray(bridges,dtype=np.int64).reshape(-1,2))
    # Recheck saved precision and no change in any immutable inputs.
    with np.load(out/'graph.npz')as f:assert np.isfinite(f['graph_distance']).all()and np.isfinite(f['nodes']).all()
    verify(s)
    report=dict(passed=True,spec_sha256=sha(a.spec),id=s['id'],checkpoint=s['checkpoint'],
        weights_sha256=s['immutable_inputs'][str(Path(s['checkpoint'])/'weights.pt')],cohort=s['cohort'],cohort_sha256=sha(s['cohort']),
        graph_sha256=sha(out/'graph.npz'),dimension=s['dimension'],nodes=NODES,k=15,anchors=4,rule='mutual',
        original_components=int(components),bridges=bridges,bridge_lengths=bridge_lengths,
        degree_quantiles=np.quantile(degree,[0,.25,.5,.75,1]).tolist(),
        original_edge_length_quantiles=np.quantile(edge_lengths,[0,.25,.5,.75,1]).tolist(),
        radius_quantiles=np.quantile(radii,[0,.01,.5,.99,1]).tolist(),unique_nodes=int(len(np.unique(latents,axis=0))),
        state_before=before,state_after=after,rng_before=rng_before,rng_after=rng_after,counts=counts,
        actual_model_source=str(actual),actual_model_source_sha256=sha(actual),model_source=s['model_source'],
        encoded_seconds=encoded_seconds,graph_seconds=graph_seconds,total_seconds=time.perf_counter()-started,
        peak_device_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,
        scope='Training RGB only; original frozen eval encoder/BN/normalization. No action/physical labels, renderer, predictor, optimizer or simulator. No floors, resampling or tuning.')
    write(out/'graph.json',report);print(json.dumps(dict(id=s['id'],passed=True,counts=counts,seconds=report['total_seconds'])),flush=True)


if __name__=='__main__':main()
