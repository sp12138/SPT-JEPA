"""One fixed train-only PushT carrier calibration per seed; no labels/fit."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import B, dataset, read, reference, sha, windows, write, state_hash
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM
from vforge.ops import OpsConfig, initialize_ops


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=read(reference('pusht',a.seed));assert cfg['batch_size']==128 and cfg['all_phase_data']
    torch.set_num_threads(2);torch.manual_seed(a.seed);torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    kwargs=dict(cfg['model'],state_slots=64,state_width=16,spatial_mode='local')
    model=SpatialCarrierLeWM(**kwargs).cuda().eval();before=state_hash(model)
    ds=dataset(cfg,'train');chosen=windows(ds,128);test=set(np.load(ds.root/'test_episodes.npy').tolist())
    assert not set(r['episode'] for r in chosen)&test
    values=[]
    with torch.no_grad():
        for batch in DataLoader(Subset(ds,[r['index'] for r in chosen]),batch_size=8,num_workers=0):
            values.append(model.raw_carrier(model.normalize(batch['pixels'][:,:1].cuda()))[:,0].cpu().double())
    z=torch.cat(values);assert z.shape==(128,1024) and torch.isfinite(z).all()
    mean=z.mean(0);rms=(z-mean).square().mean().sqrt();assert rms>0
    assert state_hash(model)==before
    write(a.output,dict(passed=True,seed=a.seed,source_config=str(reference('pusht',a.seed)),
         source_config_sha256=sha(reference('pusht',a.seed)),dataset_card_sha256=sha(ds.root/'dataset_card.json'),
         heldout_ids_sha256=sha(ds.root/'test_episodes.npy'),windows=chosen,measured_frames=128,
         normalization='global_rms',model_kwargs=dict(spatial_mode='local',state_slots=64,state_width=16,
         spatial_mean=mean.float().tolist(),spatial_scale=[float(rms)]*1024),
         state_before=before,state_after=before,scope='Train-only fixed random encoder;no physical/action labels or fitting.'))
    print('Calibration passed',a.output,flush=True)


if __name__=='__main__':main()
