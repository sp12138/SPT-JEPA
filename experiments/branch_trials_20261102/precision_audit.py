"""Matched-input precision decomposition; frozen networks and no simulation."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from common import dataset,read,write,sha
from as_jepa_tools.evaluate import load_model
from experiments.trajectory_search_20261101.common import state_hash
from experiments.ideas_validate_20261031.audit import move_batch
from vforge.ops import OpsConfig,initialize_ops


def main(root,ident,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    p=read(root/'protocol.json');row=next(r for r in p['runs'] if r['id']==ident)
    assert row['family'] in ['decoder','round2']
    run=Path(row['run']);cfg=read(run/'config.json')
    assert read(root/'worker-records'/f'{ident}-gate.json')['passed']
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=load_model(run/'model-step0002000',device='cuda');model.eval();model.requires_grad_(False)
    before=state_hash(model);cohort=read(root/'cohorts'/f'{row["task"]}.json')[:8]
    ds=dataset(cfg,'validation');indices=[x['index'] for x in cohort]
    arrays={k:[] for k in ['fp32','bf16','bf16_encoder_fp32_pool','fp32_encoder_bf16_pool','fp32_train']}
    with torch.no_grad():
        for raw in DataLoader(Subset(ds,indices),batch_size=4,num_workers=0):
            batch=move_batch(raw,'cuda');pixels=model.normalize(torch.cat([batch['pixels'],batch['anchor224'][:,None]],1))
            b,t=pixels.shape[:2]
            def manual(encoder_bf16,pool_bf16):
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=encoder_bf16):
                    tokens=model.encoder(pixels.flatten(0,1),interpolate_pos_encoding=True).last_hidden_state
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=pool_bf16):
                    pooled=model.pool(tokens if pool_bf16 else tokens.float())
                    return model.state_bn(pooled.reshape(b*t,model.state_dim)).reshape(b,t,model.state_dim).float()
            values={}
            for name,enc,pool in [('fp32',False,False),('bf16',True,True),('bf16_encoder_fp32_pool',True,False),('fp32_encoder_bf16_pool',False,True)]:
                values[name]=manual(enc,pool)
            for name,enabled in [('fp32',False),('bf16',True)]:
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
                    native=model.encode(dict(pixels=pixels))['emb'].float()
                assert torch.equal(native,values[name]),name
            model.train();values['fp32_train']=manual(False,False);model.eval()
            assert torch.equal(values['fp32_train'],values['fp32'])
            for name,value in values.items():arrays[name].append(value.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()};reference=arrays['fp32'][:,:8];stats={}
    for name,all_values in arrays.items():
        value=all_values[:,:8];flat=value.reshape(-1,model.state_dim);ref=reference.reshape(flat.shape)
        centered_error=(flat-flat.mean(0))-(ref-ref.mean(0))
        d=dict(observer_std=float(flat.std(0,ddof=1).mean()),centered_rms=float(np.sqrt(np.mean((flat-flat.mean(0))**2))),
               temporal_rms=float(np.sqrt(np.mean(np.diff(value,axis=1)**2))),
               fp32_centered_error_rms=float(np.sqrt(np.mean(centered_error**2))),
               fp32_total_error_rms=float(np.sqrt(np.mean((value-reference)**2))))
        if hasattr(model,'information'):
            mu=torch.from_numpy(all_values[:,:8]).cuda().reshape(8,8,model.state_slots,model.state_width)
            anchor=torch.from_numpy(all_values[:,8]).cuda().reshape(8,model.state_slots,model.state_width)
            d['information_nats']=float(model.information(mu[:,1:],anchor))
        stats[name]=d
    after=state_hash(model);assert before==after
    np.savez_compressed(out/'embeddings.npz',**arrays)
    report=dict(passed=True,id=ident,task=row['task'],seed=row['seed'],cohort=cohort,precision=stats,
                state_before=before,state_after=after,weights_sha256=sha(run/'model-step0002000/weights.pt'),
                main_and_anchor_frames=8*9,encoder_passes_per_frame=5,model_updates=0,simulation_calls=0,
                native_encode_manual_exact=True,fp32_train_eval_exact=True,
                runtime_source=dict(model=inspect.getfile(type(model)),encoder_method=inspect.getfile(model.encode),helper=str(Path(__file__).resolve())),
                scope='Same8 validation windows;precision intervention on frozen encodings,not a training fix or planning gain.')
    write(out/'result.json',report);print(json.dumps(dict(id=ident,precision=stats)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--id',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.id,a.output)
