"""Read-only separation of fixed-carrier precision and train-window distribution."""
import argparse,hashlib,itertools,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from as_jepa_lewm.spatial_carrier import SpatialCarrierLeWM,projection_matrix
from as_jepa_lewm.train import WindowSampler
from as_jepa_tools.benchmark_data import BenchmarkWindows
from experiments.ideas_validate_20261031.carrier.calibrate import tensor_hash
from vforge.ops import OpsConfig,initialize_ops


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def precision(value):
    torch.set_float32_matmul_precision(value)
    torch.backends.cuda.matmul.allow_tf32=(value=='high')


def project(patches,local,dense):
    grid=patches.reshape(-1,8,2,8,2,192).mean(dim=(2,4))
    return {'local':(grid@local).reshape(*patches.shape[:2],1024),
            'dense':(patches.flatten(2)@dense)}


def summarize(z):
    flat=z.reshape(-1,1024).double();std=flat.std(0,unbiased=False);energy=flat.square().mean(0)
    top=torch.argsort(energy,descending=True)
    return dict(frames=len(flat),mean_coordinate_std=float(std.mean()),rms=float(flat.square().mean().sqrt()),
        abs_max=float(flat.abs().max()),mean_abs_coordinate_bias=float(flat.mean(0).abs().mean()),
        coordinate_std_quantiles=torch.quantile(std,torch.tensor([0.,.01,.1,.5,.9,.99,1.],dtype=torch.double)).tolist(),
        top1_energy_fraction=float(energy[top[:1]].sum()/energy.sum()),
        top16_energy_fraction=float(energy[top[:16]].sum()/energy.sum()),
        top16_coordinates=top[:16].tolist(),top16_coordinate_rms=energy[top[:16]].sqrt().tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['tworoom','cube'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--root',required=True)
    a=p.parse_args();base=Path('/efs/prescottsun/SPT-JEPA');out=Path(a.root);out.mkdir(parents=True,exist_ok=False)
    start=time.time();torch.set_num_threads(2);torch.manual_seed(a.seed)
    source=read(base/'runs/local_carrier_preflight_20261031/source-record.json')
    cfgpath=base/f'runs/local_carrier_preflight_20261031/{a.task}-spatial-local-seed{a.seed}/config.json';cfg=read(cfgpath)
    calpath=Path(cfg['spatial_calibration']['path']);cal=read(calpath)
    assert sha(calpath)==cfg['spatial_calibration']['sha256'] and cal['model_seed']==a.seed
    initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=torch.device('cuda'),parallel_state=None)
    model=SpatialCarrierLeWM(**cfg['model']).cuda().eval()
    initial=Path(cfg['output'])/'initial-model/weights.pt';weights=torch.load(initial,map_location='cpu',weights_only=True)
    model.load_state_dict(weights,strict=True);del weights
    assert tensor_hash(model.encoder.state_dict())==cal['encoder_initial_sha256']
    dense=torch.from_numpy(projection_matrix('dense')).cuda();local=model.spatial_projection
    stats={arm:{k:torch.tensor(cal['arms'][arm]['model_kwargs']['spatial_'+k]) for k in ('mean','scale')} for arm in ('local','dense')}
    ds=BenchmarkWindows(cfg['data'],window=8,seed=a.seed,split='train')
    indices=dict(calibration=cal['selected_window_indices'],native_first_batch=list(itertools.islice(iter(
        WindowSampler(ds,start=0,seed=a.seed,batch_size=128)),128)))
    values={};manifests={}
    with torch.no_grad():
        for dataset,ids in indices.items():
            pieces={arm:{path:[] for path in ('HH','HL','LH','LL')} for arm in ('local','dense')}
            manifests[dataset]=[]
            for batch in DataLoader(Subset(ds,ids),batch_size=8,num_workers=0):
                pixels=model.normalize(batch['pixels'].cuda())
                for enc in ('H','L'):
                    precision('highest' if enc=='H' else 'high');patches=model.patch_features(pixels)
                    for proj in ('H','L'):
                        precision('highest' if proj=='H' else 'high')
                        for arm,value in project(patches,local,dense).items():pieces[arm][enc+proj].append(value.cpu())
                manifest_start=len(manifests[dataset])
                manifests[dataset].extend(dict(index=int(ids[manifest_start+j]),episode=int(batch['episode'][j]),
                    start=int(batch['start'][j]),rows=batch['rows'][j].tolist()) for j in range(len(batch['episode'])))
            values[dataset]={arm:{path:torch.cat(v) for path,v in paths.items()} for arm,paths in pieces.items()}
            print(json.dumps(dict(pid=os.getpid(),task=a.task,seed=a.seed,completed=dataset,seconds=time.time()-start)),flush=True)
    # All aggregates use CPU float64; no labels, optimizer or validation involved.
    report=dict(task=a.task,seed=a.seed,pid=os.getpid(),source=source,source_record_sha256=sha(base/'runs/local_carrier_preflight_20261031/source-record.json'),
        config=str(cfgpath),config_sha256=sha(cfgpath),calibration=str(calpath),calibration_sha256=sha(calpath),
        helper_sha256=sha(__file__),encoder_initial_sha256=cal['encoder_initial_sha256'],initial_weights_sha256=sha(initial),
        path_key='First letter encoder, second projection; H=highest/TF32 false, L=high/TF32 true. Same pixels and frozen tensors.',
        datasets={},scope='Read-only frozen measurement. Exact native first128 training indices; no attempt to replay renderer/optimizer/loss RNG.')
    arrays={}
    for dataset,arms in values.items():
        report['datasets'][dataset]={}
        for arm,paths in arms.items():
            mean,scale=stats[arm]['mean'],stats[arm]['scale'];z={k:(v-mean)/scale for k,v in paths.items()}
            entry=dict(scale_min=float(scale.min()),scale_median=float(scale.median()),paths={},precision_errors={})
            for path,v in z.items():
                entry['paths'][path]=dict(initial=summarize(v[:,:1]),future=summarize(v[:,1:]),all=summarize(v))
                arrays[f'{dataset}__{arm}__{path}']=paths[path].numpy()
            for name,path in [('projection_only','HL'),('encoder_only','LH'),('combined','LL')]:
                dz=z[path]-z['HH'];raw=paths[path]-paths['HH']
                entry['precision_errors'][name]=dict(normalized=summarize(dz),raw_rms=float(raw.double().square().mean().sqrt()),
                    rms_in_sigma_units=float(dz.double().square().mean().sqrt()/.1))
            rawhighest=paths['HH'].double();calraw=values['calibration'][arm]['HH'][:,0].double()
            # Alternative fixed training-only covariances are diagnostics, not model/config updates.
            future_scale=values['calibration'][arm]['HH'].double().flatten(0,1).std(0,unbiased=False)
            pooled_scale=calraw.std(0,unbiased=False).square().mean().sqrt().repeat(1024)
            entry['diagnostic_statistics_only']={}
            for label,den in [('all8_calibration_frames',future_scale),('shared_global_initial_rms',pooled_scale)]:
                entry['diagnostic_statistics_only'][label]={path:summarize((v.double()-mean.double())/den.clamp_min(1e-6)) for path,v in paths.items()}
            highcal=values['calibration'][arm]['LL'][:,0].double()
            entry['diagnostic_statistics_only']['same_high_function_initial']={path:summarize((v.double()-highcal.mean(0))/highcal.std(0,unbiased=False).clamp_min(1e-6)) for path,v in paths.items()}
            entry['small_scale_counts']={str(t):int((scale<t).sum()) for t in (.001,.003,.01,.03,.1)}
            report['datasets'][dataset][arm]=entry
    report.update(seconds=time.time()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'windows.json').write_text(json.dumps(manifests,indent=2)+'\n')
    np.savez_compressed(out/'measurements.npz',**arrays)
    print(json.dumps(dict(done=True,output=str(out),pid=os.getpid(),seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
