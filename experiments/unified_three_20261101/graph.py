"""One public NGC graph per frozen task/seed carrier; no new learner.

Reuse four existing graphs only after exact encoder/buffer comparisons and an
actual32-frame forward check. PushT uses the same public5000/15/4 rule.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from common import B, dataset, read, sha, state_hash, write
from as_jepa_lewm.factory import build_model
from clear_lewm.runner import _image_transform
from vforge.ops import OpsConfig, initialize_ops


def encoding_state(weights):
    return {k:v for k,v in weights.items() if k.startswith(('encoder.','spatial_projection','spatial_mean','spatial_scale'))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,out=Path(a.root),Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=read(root/f'{a.task}-flat-seed{a.seed}/config.json')
    weights_path=root/f'{a.task}-flat-seed{a.seed}/initial-model/weights.pt'
    weights=torch.load(weights_path,map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(1042)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=build_model(cfg).cuda().eval().requires_grad_(False);model.load_state_dict(weights,strict=True)
    before=state_hash(model);transform=_image_transform(224)
    public=Path(__file__).resolve().parents[1]/'public_ngc_20261031'
    sys.path.insert(0,str(public))
    from stable_cost import StableNGCCost
    from vendor.graph import build_graph
    from experiments.public_ngc_20261031.common import select_rows
    reference=None;encoder_calls=0;rgb_reads=0;forward_error=0.
    if a.task!='pusht':
        spec=read(B/f'runs/public_ngc_stable_evaluation_v2_20261031/specs/ngc-primary-{a.task}-local1024-seed{a.seed}.json')
        gate=read(spec['graph_gate']);assert gate['passed']
        entry=next(x for x in gate['graphs'] if x['id']==spec['model_id']);assert sha(entry['report'])==entry['report_sha256']
        reference=read(entry['report']);assert reference['passed']
        path=Path(entry['report']).with_name('graph.npz');assert sha(path)==reference['graph_sha256']
        old=torch.load(Path(spec['checkpoint'])/'weights.pt',map_location='cpu',weights_only=True,mmap=True)
        left,right=encoding_state(weights),encoding_state(old)
        assert left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        with np.load(path) as bank:nodes=bank['nodes'].copy();distances=bank['graph_distance'].copy();rows=bank['rows'][:32].copy()
        import h5py,hdf5plugin
        card=read(Path(cfg['data'])/'dataset_card.json')
        with h5py.File(card['h5_path'],'r') as h5:images=np.asarray(h5['pixels'][rows])
        with torch.inference_mode():
            pixels=transform(torch.from_numpy(images.copy()).permute(0,3,1,2)).cuda()[:,None]
            encoded=model.encode(dict(pixels=pixels))['emb'][:,0].cpu().numpy()
        forward_error=float(np.max(np.abs(encoded-nodes[:32])))
        np.testing.assert_allclose(encoded,nodes[:32],rtol=1e-5,atol=1e-5)
        encoder_calls=1;rgb_reads=32;node_rows=rows.tolist()
        reference=dict(report=entry['report'],report_sha256=entry['report_sha256'],encoding_tensors_exact=len(left))
    else:
        ds=dataset(cfg,'train');heldout=np.load(ds.root/'test_episodes.npy')
        cohort=select_rows(np.arange(len(ds.lengths)),ds.offsets,ds.lengths,ds.episodes,heldout)
        assert len(cohort)==5000 and not set(x['episode'] for x in cohort)&set(heldout.tolist())
        ds.open();latents=[]
        with torch.inference_mode():
            for start in range(0,5000,32):
                rows=[x['row'] for x in cohort[start:start+32]]
                images=np.array(ds.p224[rows]);rgb_reads+=len(images)
                pixels=transform(torch.from_numpy(images).permute(0,3,1,2)).cuda()[:,None]
                latents.append(model.encode(dict(pixels=pixels))['emb'][:,0].float().cpu().numpy());encoder_calls+=1
        nodes,distances,bridges=build_graph(np.concatenate(latents),15,'mutual')
        path=out/'graph.npz'
        np.savez_compressed(path,nodes=nodes.astype(np.float32),graph_distance=distances.astype(np.float32),
                            rows=np.asarray([x['row'] for x in cohort]),bridges=np.asarray(bridges,dtype=np.int64).reshape(-1,2))
        nodes,distances=nodes.astype(np.float32),distances.astype(np.float32)
        node_rows=cohort;assert encoder_calls==157 and rgb_reads==5000
    assert nodes.shape==(5000,1024) and distances.shape==(5000,5000)
    assert np.isfinite(nodes).all() and np.isfinite(distances).all() and (distances>=0).all()
    assert np.array_equal(np.diag(distances),np.zeros(5000))
    cost=StableNGCCost(nodes,distances,15,4).cuda();radii=[]
    with torch.inference_mode():
        for start in range(0,5000,32):
            z=torch.from_numpy(nodes[start:start+32]).cuda();radii.extend(cost.attach(z)[2].cpu().tolist())
            value=cost(dict(predicted_emb=z[:,None,None],goal_emb=z[:,None]))
            assert torch.equal(value,torch.zeros_like(value))
    assert min(radii)>0 and state_hash(model)==before
    write(out/'graph.json',dict(passed=True,task=a.task,seed=a.seed,graph=str(path),graph_sha256=sha(path),
          reference=reference,weights=str(weights_path),weights_sha256=sha(weights_path),nodes=5000,k=15,anchors=4,
          numeric_variant='uniform-fp32-direct-attachment',encoder_calls=encoder_calls,rgb_reads=rgb_reads,
          forward_max_error=forward_error,state_before=before,state_after=before,node_rows=node_rows,
          radius_quantiles=np.quantile(radii,[0,.01,.5,.99,1]).tolist(),
          scope='Train-only public NGC control;no learning,optimal actions,physical labels or task-specific graph tuning.'))
    print('Graph passed',a.task,a.seed,flush=True)


if __name__=='__main__':main()
