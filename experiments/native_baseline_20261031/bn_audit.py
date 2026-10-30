"""Frozen native600 precision/BN diagnosis on exact last-train and heldout clips.

No optimizer, recalibration artifact or planner change. Every BN buffer mutation
is restored. Batch-statistics branches are diagnostic extra-information controls.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset


def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args()
    root=Path(args.run);out=Path(args.output)
    if(out/'bn-audit.json').exists():raise FileExistsError('Preserve completed diagnosis')
    out.mkdir(parents=True,exist_ok=True);cfg=read(root/'config.json')
    assert cfg['model_class']=='native_lewm'and cfg['precision']=='bf16-mixed'
    assert cfg['float32_matmul_precision']=='high'and cfg['batch_size']==128
    from as_jepa_lewm.factory import build_model
    from as_jepa_lewm.train import WindowSampler
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    from experiments.ideas_validate_20261031.audit import verify_manifest
    torch.set_num_threads(2);device=torch.device(args.device);torch.manual_seed(1042)
    model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    weights=root/'model-step0000600/weights.pt'
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    checkpoint=torch.load(root/'last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['global_step']==600 and checkpoint['campaign_samples']==600*128
    del checkpoint
    original_buffers={k:v.clone()for k,v in model.named_buffers()}
    batchnorms={k:m for k,m in model.named_modules()if isinstance(m,nn.BatchNorm1d)}
    assert set(batchnorms)=={'projector.net.1','pred_proj.net.1'}
    audit_dir=root.parent/('transition-'+root.name)
    validation_manifest=read(audit_dir/'windows.json')
    reports=[];manifests={};cached={};started=time.time()
    precision_policies=[('eval_fp32','highest',False),('train_matmul_fp32','high',False),('train_bf16','high',True)]
    with torch.no_grad():
        for split in ['train','validation']:
            ds=BenchmarkWindows(cfg['data'],window=8,split=split)
            if split=='train':
                cursor=(600-1)*128
                indices=list(itertools.islice(iter(WindowSampler(ds,start=cursor,seed=cfg['seed'],batch_size=128)),128))
                manifest=dict(split=split,selection='exact last training batch at update599, before final update',sample_cursor=cursor,windows=[])
                for index in indices:
                    ep,start=ds.locate(index)
                    manifest['windows'].append(dict(index=index,episode=int(ds.episode_ids[ep]),start=start,
                        rows=(int(ds.offsets[ep])+start+np.arange(8)*ds.stride).tolist()))
            else:
                verify_manifest(ds,validation_manifest);manifest=validation_manifest
                indices=[r['index']for r in manifest['windows']]
            manifests[split]=manifest
            all_actions=[];values={policy[0]:[]for policy in precision_policies};offset=0
            for batch in DataLoader(Subset(ds,indices),batch_size=4,num_workers=0,shuffle=False):
                for j,row in enumerate(batch['rows'].tolist()):
                    assert row==manifest['windows'][offset+j]['rows']
                    assert int(batch['episode'][j])==manifest['windows'][offset+j]['episode']
                pixels=model.normalize(batch['pixels'].to(device));bs=len(pixels)
                all_actions.append(batch['action'].to(device))
                for name,precision,amp in precision_policies:
                    torch.set_float32_matmul_precision(precision)
                    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                        value=model.encoder(pixels.flatten(0,1),interpolate_pos_encoding=True).last_hidden_state[:,0]
                    values[name].append(value.reshape(bs,8,192))
                offset+=bs
            if ds.file is not None:ds.file.close()
            actions=torch.cat(all_actions);n=len(actions)
            for policy,precision,amp in precision_policies:
                cls=torch.cat(values[policy]);cached[split+'__cls__'+policy]=cls.float().cpu().numpy()
                torch.set_float32_matmul_precision(precision)
                modes=['running','projector_batch','prediction_batch','both_batch']
                if policy=='train_bf16':modes.append('native_train_dropout')
                for mode in modes:
                    model.eval()
                    for k,buffer in model.named_buffers():buffer.copy_(original_buffers[k])
                    if mode in ['projector_batch','both_batch','native_train_dropout']:model.projector.train()
                    if mode in ['prediction_batch','both_batch','native_train_dropout']:model.pred_proj.train()
                    if mode=='native_train_dropout':model.predictor.train()
                    inputs={};hooks=[]
                    for name,bn in batchnorms.items():
                        def hook(module,args,key=name):inputs[key]=args[0].detach().float()
                        hooks.append(bn.register_forward_pre_hook(hook))
                    torch.manual_seed(1042)
                    try:
                        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):
                            emb=model.projector(cls.flatten(0,1)).reshape(n,8,192)
                            act=model.action_encoder(torch.nan_to_num(actions,0.))
                            pred=model.predict(emb[:,:7],act[:,:7])
                            native_mse=(pred-emb[:,1:]).pow(2).mean()
                            sigreg=model.sigreg(emb.transpose(0,1))
                        error=(pred.float()-emb[:,1:].float()).double()
                        mean_error=error.mean((0,1));mse=float(error.square().mean())
                        copy=float((emb[:,1:].float()-emb[:,:-1].float()).double().square().mean())
                        bn_report={}
                        for name,bn in batchnorms.items():
                            observed=inputs[name].double();mean=observed.mean(0);variance=observed.var(0,unbiased=False)
                            running_mean=original_buffers[name+'.running_mean'].double()
                            running_var=original_buffers[name+'.running_var'].double()
                            bn_report[name]=dict(batch_mode=bn.training,
                                standardized_mean_shift_rms=float(((mean-running_mean).square()/(running_var+bn.eps)).mean().sqrt()),
                                running_variance_median=float(running_var.median()),observed_variance_median=float(variance.median()),
                                mean_variance_ratio=float((variance/(running_var+bn.eps)).mean()),
                                observed_mean_rms=float(mean.square().mean().sqrt()),running_mean_rms=float(running_mean.square().mean().sqrt()))
                        row=dict(split=split,clips=n,precision_policy=policy,bn_mode=mode,
                            native_pred_loss=float(native_mse),native_sigreg=float(sigreg),native_loss=float(native_mse+.09*sigreg),
                            fp64_residual_mse=mse,shared_bias_mse=float(mean_error.square().mean()),
                            centered_residual_mse=float((error-mean_error).square().mean()),
                            local_carry_mse=copy,pred_over_local_carry=mse/max(copy,1e-30),
                            observed_rms=float(emb.float().square().mean().sqrt()),
                            observed_mean_rms=float(emb.float().mean((0,1)).square().mean().sqrt()),
                            observed_std_mean=float(emb.float().std((0,1)).mean()),
                            predicted_mean_rms=float(pred.float().mean((0,1)).square().mean().sqrt()),
                            raw_cls_std_mean=float(cls.float().std((0,1)).mean()),batchnorm=bn_report)
                        if split=='validation'and policy=='eval_fp32'and mode=='running':
                            with np.load(audit_dir/'state-audit.npz')as z:reference=z['observed']
                            actual=emb.float().cpu().numpy()
                            np.testing.assert_allclose(actual,reference,rtol=2e-4,atol=2e-4)
                            row['existing_encoding_max_abs_error']=float(np.max(np.abs(actual-reference)))
                        reports.append(row)
                        if mode in ['running','both_batch']:
                            cached[split+'__'+policy+'__'+mode+'__emb']=emb.float().cpu().numpy()
                            cached[split+'__'+policy+'__'+mode+'__pred']=pred.float().cpu().numpy()
                    finally:
                        for hook in hooks:hook.remove()
                        for k,buffer in model.named_buffers():buffer.copy_(original_buffers[k])
                        model.eval()
    assert all(torch.equal(v,original_buffers[k])for k,v in model.named_buffers())
    assert not {w['episode']for w in manifests['train']['windows']}.intersection(w['episode']for w in manifests['validation']['windows'])
    np.savez_compressed(out/'cached-features.npz',**cached)
    result=dict(run=str(root),source_sha256=digest(__file__),weights_sha256=digest(weights),config_sha256=digest(root/'config.json'),
        actual_precision=cfg['precision'],actual_matmul_precision=cfg['float32_matmul_precision'],rows=reports,manifests=manifests,
        buffers_restored=True,seconds=time.time()-started,
        scope='Frozen diagnostic; no optimizer, parameter update, saved calibration or planning change. Same last production training batch and common heldout64windows. BN batch modes use additional simultaneous observations and sometimes change target codes; they are attribution controls, not deployment results. Native train-dropout row approximates training at checkpoint600; logged step600 loss was before its final parameter update. Train128 vsval64 SIGReg totals are not directly comparable.')
    (out/'bn-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),buffers_restored=True,seconds=result['seconds'],rows=[{k:r[k]for k in ['split','precision_policy','bn_mode','native_pred_loss','observed_rms']}for r in reports])),flush=True)


if __name__=='__main__':main()
