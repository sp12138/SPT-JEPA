"""One fixed, training-only precise-BN control; never changes original artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset


def read(path):return json.loads(Path(path).read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Moments:
    """Merge actual vector counts, including between-chunk mean variation."""
    def __init__(self):self.n=0;self.mean=None;self.m2=None
    def update(self,x):
        x=np.asarray(x,dtype=np.float64)
        assert x.ndim==2 and len(x)>0 and np.isfinite(x).all()
        n=len(x);mean=x.mean(0);m2=np.square(x-mean).sum(0)
        if not self.n:self.n=n;self.mean=mean;self.m2=m2;return
        delta=mean-self.mean;total=self.n+n
        self.m2+=m2+np.square(delta)*(self.n*n/total)
        self.mean+=delta*(n/total);self.n=total
    @property
    def variance(self):
        assert self.n>1
        return self.m2/(self.n-1)


def contracts(output):
    rng=np.random.default_rng(1042);x=rng.normal(size=(37,11));x[:5]+=7
    moment=Moments()
    for block in [x[:5],x[5:19],x[19:]]:moment.update(block)
    np.testing.assert_allclose(moment.mean,x.mean(0),atol=1e-14,rtol=1e-14)
    np.testing.assert_allclose(moment.variance,x.var(0,ddof=1),atol=1e-13,rtol=1e-13)
    bn=torch.nn.BatchNorm1d(11).double().eval();tx=torch.from_numpy(x)
    with torch.no_grad():bn.running_mean.copy_(torch.from_numpy(moment.mean));bn.running_var.copy_(torch.from_numpy(moment.variance))
    fixed=bn(tx);batched=torch.cat([bn(tx[:5]),bn(tx[5:19]),bn(tx[19:])])
    assert torch.equal(fixed,batched)
    batch=F.batch_norm(tx,None,None,bn.weight,bn.bias,training=True,momentum=0.,eps=bn.eps)
    ratio=torch.sqrt((tx.var(0,unbiased=False)+bn.eps)/(tx.var(0,unbiased=True)+bn.eps))
    corrected=(batch-bn.bias)*ratio+bn.bias
    torch.testing.assert_close(fixed,corrected,atol=1e-13,rtol=1e-13)
    result=dict(passed=True,count=moment.n,unequal_chunks=[5,14,18],global_mean_max_error=float(np.max(abs(moment.mean-x.mean(0)))),
        global_unbiased_variance_max_error=float(np.max(abs(moment.variance-x.var(0,ddof=1)))),
        deterministic_fixed_bn_batch_independent=True,finite_sample_batch_bn_correction_max_error=float((fixed-corrected).abs().max()),
        source_sha256=digest(__file__),scope='CPU algebra only. Standard pooled first/second moments; unbiased running variance, no tuned momentum or dataset fitting.')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if(out/'contracts.json').exists():raise FileExistsError(str(out))
    (out/'contracts.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def main(args):
    from as_jepa_lewm.factory import build_model
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    root=Path(args.run);out=Path(args.output)
    if out.exists()and any(out.iterdir()):raise FileExistsError('Use a fresh calibration output')
    out.mkdir(parents=True,exist_ok=True)
    cfg=read(root/'config.json');manifest=read(args.manifest)
    assert cfg['model_class']=='native_lewm'and manifest['split']=='train'and len(manifest['windows'])==128
    data=Path(cfg['data'])
    assert digest(data/'dataset_card.json')==manifest['dataset_card_sha256']
    for name,sha in manifest['metadata_sha256'].items():assert digest(data/name)==sha
    allowed=set(np.load(data/'train_episodes.npy').tolist());forbidden=set(np.load(data/'validation_episodes.npy').tolist())|set(np.load(data/'test_episodes.npy').tolist())
    eps=[w['episode']for w in manifest['windows']]
    assert len(set(eps))==128 and set(eps)<=allowed and not set(eps)&forbidden
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(1042)
    device=torch.device(args.device);model=build_model(cfg).to(device).eval();model.requires_grad_(False)
    source=root/f'model-step{args.step:07d}';original_sha=digest(source/'weights.pt')
    original=torch.load(source/'weights.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(original,strict=True)
    ds=BenchmarkWindows(data,window=8,split='train');classes=[];actions=[];offset=0;started=time.time()
    with torch.no_grad():
        for batch in DataLoader(Subset(ds,[r['index']for r in manifest['windows']]),batch_size=4,shuffle=False,num_workers=0):
            for j,rows in enumerate(batch['rows'].tolist()):
                assert rows==manifest['windows'][offset+j]['rows']and int(batch['episode'][j])==manifest['windows'][offset+j]['episode']
            pixel=model.normalize(batch['pixels'].to(device));n=len(pixel)
            classes.append(model.encoder(pixel.flatten(0,1),interpolate_pos_encoding=True).last_hidden_state[:,0].reshape(n,8,192))
            actions.append(batch['action'].to(device));offset+=n
        if ds.file is not None:ds.file.close()
        cls=torch.cat(classes);act=torch.cat(actions)
        def set_statistics(name,inputs):
            bn=dict(model.named_modules())[name];m=Moments()
            flat=inputs.float().cpu().numpy()
            for start in range(0,len(flat),32):m.update(flat[start:start+32])
            # Independent full-array checks protect count and dependency-order plumbing.
            np.testing.assert_allclose(m.mean,flat.astype(np.float64).mean(0),rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(m.variance,flat.astype(np.float64).var(0,ddof=1),rtol=1e-9,atol=1e-12)
            bn.running_mean.copy_(torch.from_numpy(m.mean).to(device=device,dtype=bn.running_mean.dtype))
            bn.running_var.copy_(torch.from_numpy(m.variance).to(device=device,dtype=bn.running_var.dtype))
            fixed=bn(inputs)
            batch=F.batch_norm(inputs,None,None,bn.weight,bn.bias,training=True,momentum=0.,eps=bn.eps)
            ratio=torch.sqrt((torch.from_numpy(m.m2/m.n).to(device)+bn.eps)/(torch.from_numpy(m.variance).to(device)+bn.eps)).float()
            corrected=(batch-bn.bias)*ratio+bn.bias
            replay=float((fixed-corrected).square().mean())
            assert torch.isfinite(fixed).all()and replay<1e-6
            return dict(name=name,vectors=m.n,default_momentum=bn.momentum,num_batches_tracked=int(bn.num_batches_tracked),
                old_mean_rms=float(original[name+'.running_mean'].square().mean().sqrt()),new_mean_rms=float(bn.running_mean.square().mean().sqrt()),
                old_variance_median=float(original[name+'.running_var'].median()),new_variance_median=float(bn.running_var.median()),
                fixed_vs_batch_bn_mse=float((fixed-batch).square().mean()),finite_sample_corrected_mse=replay,
                mean=m.mean.tolist(),unbiased_variance=m.variance.tolist())
        first_input=model.projector.net[0](cls.flatten(0,1))
        first=set_statistics('projector.net.1',first_input);assert first['vectors']==1024
        observed=model.projector(cls.flatten(0,1)).reshape(128,8,192)
        embedded=model.action_encoder(act)
        raw_prediction=model.predictor(observed[:,:7],embedded[:,:7])
        second_input=model.pred_proj.net[0](raw_prediction.flatten(0,1))
        second=set_statistics('pred_proj.net.1',second_input);assert second['vectors']==896
        predicted=model.predict(observed[:,:7],embedded[:,:7])
        fixed_mse=float((predicted-observed[:,1:]).square().mean())
        fixed_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        allowed_buffers={prefix+'.'+key for prefix in ['projector.net.1','pred_proj.net.1']for key in ['running_mean','running_var']}
        changed=[k for k,v in fixed_state.items()if not torch.equal(v,original[k])]
        assert set(changed)==allowed_buffers
        # Additional real observations used by batch-BN control are not used at deployment.
        model.projector.train();model.pred_proj.train();torch.manual_seed(1042)
        batch_observed=model.projector(cls.flatten(0,1)).reshape(128,8,192)
        batch_prediction=model.predict(batch_observed[:,:7],embedded[:,:7])
        batch_mse=float((batch_prediction-batch_observed[:,1:]).square().mean())
        model.load_state_dict(fixed_state,strict=True);model.eval()
        single=[]
        for i in range(2):
            z=model.projector(cls[i].reshape(8,192)).reshape(1,8,192)
            single.append(model.predict(z[:,:7],embedded[i:i+1,:7]))
        sample_error=float((torch.cat(single)-predicted[:2]).square().mean());assert sample_error<1e-7
        assert all(torch.equal(v.cpu(),fixed_state[k])for k,v in model.state_dict().items())
    export=out/'model';export.mkdir()
    torch.save(fixed_state,export/'weights.pt');shutil.copyfile(source/'config.json',export/'config.json')
    assert digest(source/'config.json')==digest(export/'config.json')
    assert all(torch.equal(v,original[k])for k,v in fixed_state.items()if k not in allowed_buffers)
    assert digest(source/'weights.pt')==original_sha
    np.savez_compressed(out/'train-replay.npz',observed=observed.cpu().numpy(),prediction=predicted.cpu().numpy(),
        batch_observed=batch_observed.cpu().numpy(),batch_prediction=batch_prediction.cpu().numpy())
    result=dict(run=str(root),step=args.step,source_sha256=digest(__file__),original_weights_sha256=digest(source/'weights.pt'),
        export_weights_sha256=digest(export/'weights.pt'),export=str(export),manifest=str(args.manifest),manifest_sha256=digest(args.manifest),
        source_config_sha256=digest(source/'config.json'),statistics=[first,second],changed_buffers=changed,
        all_other_state_bitexact=True,training_only=True,precision='FP32/highest deployment precision; dropout off',
        replay=dict(fixed_one_step_mse=fixed_mse,batch_bn_one_step_mse=batch_mse,
            fixed_vs_batch_state_mse=float((observed-batch_observed).square().mean()),
            fixed_vs_batch_prediction_mse=float((predicted-batch_prediction).square().mean()),
            fixed_single_clip_vs_batch_prediction_mse=sample_error),seconds=time.time()-started,
        scope='Posthoc training-only precise-BN baseline variant, not literal native LeWM or a new method. Uniform global moments with unbiased variance; projector first, then prediction BN on calibrated observed states. No gradients, optimizer, physical/validation labels, momentum tuning or original checkpoint/config edit. Preserve original uncalibrated results separately.')
    (out/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out),export=str(export),replay=result['replay'],seconds=result['seconds'],other_state_exact=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['contracts','calibrate']);p.add_argument('--run');p.add_argument('--manifest');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--step',type=int,default=600);a=p.parse_args()
    if a.phase=='contracts':contracts(a.output)
    else:
        if not a.run or not a.manifest:p.error('calibrate requires --run and --manifest')
        main(a)
