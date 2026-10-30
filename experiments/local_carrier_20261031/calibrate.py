"""Joint local/dense statistics on the original128 initial training frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM,projection_matrix,PROJECTION_SEED
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.ideas_validate_20261031.carrier.calibrate import file_hash,tensor_hash
from vforge.ops import OpsConfig,initialize_ops


def statistics(values):
    assert values.shape==(128,1024) and torch.isfinite(values).all()
    values=values.double();mean=values.mean(0);raw=values.std(0,unbiased=False)
    return mean.float(),raw.clamp_min(1e-6).float(),raw


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-calibration',required=True)
    p.add_argument('--output',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args();out=Path(args.output)
    assert not out.exists();reference=json.loads(Path(args.reference_calibration).read_text())
    assert reference['measured_frames']==128 and reference['selection_seed']==1042
    cfg=json.loads(Path(reference['config']).read_text());assert file_hash(reference['config'])==reference['config_sha256']
    assert cfg['dataset_kind']=='benchmark_hdf5' and not cfg.get('initial_encoder_checkpoint')
    assert not any(k.startswith('carrier_') for k in cfg['model'])
    kwargs=dict(cfg['model'],state_slots=64,state_width=16,spatial_mode='local')
    device=torch.device(args.device);torch.set_num_threads(2);torch.manual_seed(cfg['seed'])
    torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa' if device.type=='cuda' else 'sdpa_math',allow_fallback=False),device=device,parallel_state=None)
    model=SpatialCarrierLeWM(**kwargs).to(device).eval()
    encoder_hash=tensor_hash(model.encoder.state_dict());assert encoder_hash==reference['encoder_initial_sha256']
    dense=torch.from_numpy(projection_matrix('dense')).to(device)
    projections=dict(local=tensor_hash({'spatial_projection':model.spatial_projection}),
                     dense=tensor_hash({'spatial_projection':dense}))
    data=Path(cfg['data']);assert file_hash(data/'dataset_card.json')==reference['dataset_card_sha256']
    assert file_hash(data/'train_episodes.npy')==reference['train_episode_ids_sha256']
    ds=BenchmarkWindows(data,window=cfg['model']['window'],split='train',seed=cfg['seed'])
    indices=reference['selected_window_indices'];assert len(indices)==128
    loader=DataLoader(Subset(ds,indices),batch_size=args.batch_size,num_workers=0)
    values=dict(local=[],dense=[]);episodes=[];starts=[];rows=[]
    with torch.no_grad():
        for batch in loader:
            pixels=model.normalize(batch['pixels'][:,:1].to(device));patches=model.patch_features(pixels)
            values['local'].append(model.project_patches(patches)[:,0].cpu())
            with torch.autocast(device_type=device.type,enabled=False):
                values['dense'].append((patches.float().flatten(2)@dense)[:,0].cpu())
            episodes.extend(int(x) for x in batch['episode']);starts.extend(int(x) for x in batch['start'])
            rows.extend(int(x) for x in batch['rows'][:,0])
    assert episodes==reference['training_episode_ids'] and starts==reference['starts'] and rows==reference['measured_h5_rows']
    assert len(set(episodes))==128 and set(episodes)<=set(np.load(data/'train_episodes.npy').tolist())
    assert tensor_hash(model.encoder.state_dict())==encoder_hash
    arms={}
    for mode in ('local','dense'):
        mean,scale,raw=statistics(torch.cat(values[mode]))
        counts=model.spatial_parameter_counts()
        if mode=='dense':counts.update(fixed_projection_elements=dense.numel(),fixed_projection_bytes=dense.numel()*dense.element_size())
        arms[mode]=dict(model_kwargs=dict(spatial_mode=mode,state_slots=64,state_width=16,
            spatial_mean=mean.tolist(),spatial_scale=scale.tolist()),projection_sha256=projections[mode],
            raw_std_min=float(raw.min()),raw_std_mean=float(raw.mean()),raw_std_max=float(raw.max()),
            scale_floor_coordinates=int((raw<1e-6).sum()),parameter_counts=counts)
    result=dict(source_commit=args.source_commit,source_sha256=file_hash(__file__),
        model_source_sha256=file_hash(__import__('as_jepa_lewm.spatial_carrier',fromlist=['__file__']).__file__),
        reference_calibration=str(Path(args.reference_calibration)),reference_calibration_sha256=file_hash(args.reference_calibration),
        config=reference['config'],config_sha256=reference['config_sha256'],dataset=str(data),model_seed=cfg['seed'],
        dataset_card_sha256=reference['dataset_card_sha256'],train_episode_ids_sha256=reference['train_episode_ids_sha256'],
        encoder_initial_sha256=encoder_hash,selected_window_indices=indices,training_episode_ids=episodes,
        starts=starts,measured_h5_rows=rows,measured_frames=128,projection_seed=PROJECTION_SEED,
        device=str(device),precision='FP32/highest/no autocast',scale_floor=1e-6,arms=arms,
        scope='Exactly the old initial128 training frames; one shared frozen encoder pass for both fixed projections. '
              'No validation, physical labels, fitting, optimizer or new frame selection.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),frames=128,encoder_sha256=encoder_hash,
                         parameter_counts={k:v['parameter_counts'] for k,v in arms.items()})),flush=True)


if __name__=='__main__':main()
