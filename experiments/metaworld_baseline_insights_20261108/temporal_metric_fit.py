"""T2G-adapted MetaWorld diagnostic;public algorithm,not a novel candidate.

Only own scratch frozen RGB features and trajectory chronology enter the head.
No state/action/reward/task labels or external pretrained weights are used.
Ragged clips retain short trajectories without padded target observations.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

import numpy as np

BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_t2g_adapted_20261108'
SCENE=BASE/'runs/metaworld_baseline_insights_20261108/scene-probe-v1'
PUBLIC=BASE/'reports/metaworld_baseline_insights_20261108/t2g-source-572784c/temporal_metric.py'
CONFIG=dict(hidden_dim=256,depth=2,dropout=0.,max_steps=15,min_steps=1,gap=2,margin=1.,negative_weight=0.,
    lr=5e-5,weight_decay=.001,batch_size=128,epochs=5,seed=20261108)


def read(p):return json.loads(Path(p).read_text())


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)


def encode(name):
    import h5py
    import hdf5plugin
    import torch
    from experiments.metaworld_baselines_20261107.models import PublicBaseline
    from experiments.metaworld_baselines_20261107.train_native import digest
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');assert not torch.cuda.is_initialized()
    out=ROOT/name;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    job=next(j for j in read(BASE/'runs/metaworld_baselines_native_20261108/plan.json')['jobs'] if j['name']==name)
    cfg=read(job['config']);data=Path(cfg['data']);phases=[p for p in read(data/'pack-index.json')['phases'] if p['phase']==0]
    examples=read(SCENE/'selection.json')['examples'];splits={}
    for row in examples:
        old=splits.setdefault(row['original_episode'],row['split']);assert old==row['split']
    assert len(splits)==2500
    phases.sort(key=lambda p:p['original_episode']);indices=[];episodes=[]
    for p in phases:
        start=len(indices);indices.extend(range(p['offset'],p['offset']+p['length']))
        episodes.append(dict(original_episode=p['original_episode'],split=splits[p['original_episode']],offset=start,length=p['length']))
    indices=np.asarray(indices);assert np.all(np.diff(indices)>0)
    model=PublicBaseline(cfg['method'],cfg['sources']);weights=Path(job['output'])/f"model-step{job['epoch_steps']['100']:07d}/weights.pt"
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=digest(model)
    old=read(SCENE/name/'FROZEN.json');assert old['passed'] and old['model_state_before']==before and old['weights_sha256']==sha(weights)
    # Reuse RGB-only embedding field;no privileged scene-readout labels/coefs.
    old_features=np.load(SCENE/name/'features.npz')['embedding'][:len(examples)]
    old_indices=np.asarray([e['packed_index'] for e in examples]);positions=np.searchsorted(indices,old_indices)
    assert np.array_equal(indices[positions],old_indices)
    features=np.full((len(indices),old_features.shape[1]),np.nan,np.float32);features[positions]=old_features
    missing=np.flatnonzero(~np.isfinite(features[:,0]));card=read(data/'dataset_card.json')
    def one(images):
        x=torch.from_numpy(np.asarray(images))
        if x.shape[-1]==3:x=x.permute(0,3,1,2)
        assert x.dtype==torch.uint8 and x.shape[1:]==(3,224,224)
        with torch.inference_mode():return model.encode(dict(pixels=model.normalize(x[:,None])))['emb'][:,0].float().numpy()
    with h5py.File(card['h5_path'],'r') as f:
        ds=f['pixels'];check=positions[np.linspace(0,len(positions)-1,16,dtype=np.int64)]
        restored=one(ds[indices[check]])
        parity=float(np.max(np.abs(restored-features[check])));assert np.allclose(restored,features[check],atol=1e-5,rtol=1e-5)
        for start in range(0,len(missing),16):
            chosen=missing[start:start+16];features[chosen]=one(ds[indices[chosen]])
            if start%512==0:save(out/'progress.json',dict(stage='frozen_rgb_features',new_frames=min(start+16,len(missing)),new_frames_total=len(missing),seconds=time.monotonic()-started))
    assert np.isfinite(features).all() and digest(model)==before and not torch.cuda.is_initialized()
    np.savez_compressed(out/'rgb_features.npz',embedding=features,packed_indices=indices)
    save(out/'episodes.json',dict(episodes=episodes,split_counts={str(s):sum(e['split']==s for e in episodes) for s in range(3)},
        scope='Originalsceneprobe40fit/5validation/5heldouttrajectoriespertask. IDs/splitsmetadataonly,notheadinputs/targets. Allphase0frames,notall5phases.'))
    save(out/'FROZEN.json',dict(passed=True,name=name,worldmodel_updates=0,model_state_before=before,model_state_after=digest(model),
        weights_sha256=sha(weights),feature_sha256=sha(out/'rgb_features.npz'),episodes_sha256=sha(out/'episodes.json'),
        frames=len(indices),reused_rgb_frames=len(old_indices),new_rgb_frames=len(missing),reused_feature_max_abs_difference=parity,
        parity_rgb_frames=len(check),actual_new_nn_encoded_frames=len(missing)+len(check),
        gpu_context_created=False,new_physical_episodes=0,seconds=time.monotonic()-started,feature_source='RGBONLY',
        state_action_reward_task_targets=False,external_pretrained_weights=False,source_sha256=sha(__file__)))


def fit(name):
    import torch
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.manual_seed(CONFIG['seed']);assert not torch.cuda.is_initialized()
    out=ROOT/name;frozen=read(out/'FROZEN.json');assert frozen['passed'] and sha(out/'rgb_features.npz')==frozen['feature_sha256']
    assert sha(out/'episodes.json')==frozen['episodes_sha256']
    spec=importlib.util.spec_from_file_location('audited_t2g',PUBLIC);public=importlib.util.module_from_spec(spec);spec.loader.exec_module(public)
    values=torch.from_numpy(np.load(out/'rgb_features.npz')['embedding']);episodes=read(out/'episodes.json')['episodes']
    clips={s:[] for s in range(3)};ids={s:set() for s in range(3)};excluded=[]
    for e in episodes:
        ids[e['split']].add(e['original_episode'])
        if e['length']<4:excluded.append(e['original_episode']);continue
        for start in range(e['length']-3):
            length=min(16,e['length']-start);assert length>=4
            clips[e['split']].append((e['offset']+start,length))
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    metric=public.TemporalMetric(values.shape[-1],hidden_dim=256,depth=2,dropout=0.).cpu()
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in metric.modules())
    optimizer=torch.optim.AdamW(metric.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    def triples(rows,generator):
        groups={}
        for offset,length in rows:groups.setdefault(length,[]).append(offset)
        result={k:[] for k in ['anchor','near','far']}
        for length,offsets in groups.items():
            index=torch.tensor(offsets)[:,None]+torch.arange(length)[None,:];emb=values[index]
            p=public.make_temporal_pairs(emb,max_steps=15,min_steps=1,gap=2,generator=generator,sampling='legacy')
            for k in result:result[k].append(p[k])
        return {k:torch.cat(v) for k,v in result.items()}
    def evaluate(split,seed):
        generator=torch.Generator().manual_seed(seed);correct=raw_correct=total=0;loss=0.
        with torch.inference_mode():
            for start in range(0,len(clips[split]),128):
                p=triples(clips[split][start:start+128],generator);near=metric(p['anchor'],p['near']);far=metric(p['anchor'],p['far'])
                n=len(near);correct+=int((far>near).sum());raw_correct+=int(((p['far']-p['anchor']).norm(dim=-1)>(p['near']-p['anchor']).norm(dim=-1)).sum())
                total+=n;loss+=float(torch.relu(1+near-far).sum())
        return dict(triples=total,rank_accuracy=correct/total,raw_latent_rank_accuracy=raw_correct/total,margin_loss=loss/total)
    # Check that the public head/loss remain sample-level at actual dimensions.
    g=torch.Generator().manual_seed(1);p=triples(clips[0][:8],g)
    batched=public.temporal_metric_loss(metric(p['anchor'],p['near']),metric(p['anchor'],p['far']),margin=1.,negative_weight=0.)['loss']
    individual=torch.stack([public.temporal_metric_loss(metric(p['anchor'][i:i+1],p['near'][i:i+1]),metric(p['anchor'][i:i+1],p['far'][i:i+1]),margin=1.,negative_weight=0.)['loss'] for i in range(8)]).mean()
    assert torch.allclose(batched,individual,atol=2e-5,rtol=2e-5)
    before=evaluate(1,1001);rows=[];generator=torch.Generator().manual_seed(CONFIG['seed']);steps=0
    for epoch in range(5):
        order=torch.randperm(len(clips[0]),generator=generator);total=0.;count=0
        for start in range(0,len(order)-127,128):
            p=triples([clips[0][i] for i in order[start:start+128].tolist()],generator)
            optimizer.zero_grad(set_to_none=True)
            loss=public.temporal_metric_loss(metric(p['anchor'],p['near']),metric(p['anchor'],p['far']),margin=1.,negative_weight=0.)['loss']
            assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in metric.parameters())
            optimizer.step();steps+=1;total+=float(loss.detach())*128;count+=128
        row=dict(epoch=epoch+1,train_margin_loss=total/count,validation=evaluate(1,1001),optimizer_steps=steps);rows.append(row)
        save(out/'training.json',dict(config=CONFIG,epochs=rows));print(json.dumps(row),flush=True)
    metric.eval();test=evaluate(2,1002)
    # This checkpoint contains own randomly initialized/trained head weights only.
    state={k:v.detach().cpu() for k,v in metric.state_dict().items()};torch.save(state,out/'metric-state.pt')
    torch.save(dict(model=state,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        sampler_rng=generator.get_state(),config=CONFIG,completed_epochs=5,optimizer_steps=steps),out/'metric-training-state.pt')
    check=public.TemporalMetric(values.shape[-1],hidden_dim=256,depth=2,dropout=0.);check.load_state_dict(torch.load(out/'metric-state.pt',weights_only=True,map_location='cpu'),strict=True)
    assert all(torch.equal(state[k],check.state_dict()[k]) for k in state) and not torch.cuda.is_initialized()
    save(out/'summary.json',dict(passed=True,name=name,config=CONFIG,public_commit='572784cd85a3e4af802be88063449f7bbd0580c9',
        public_source_sha256=sha(PUBLIC),adapter_source_sha256=sha(__file__),feature_proof=frozen,
        clips={str(k):len(v) for k,v in clips.items()},excluded_short_original_episodes=excluded,
        original_episode_counts={str(k):len(v) for k,v in ids.items()},validation_before=before,validation=rows[-1]['validation'],heldout=test,
        sample_level_batch_independence_check=True,head_optimizer_updates=steps,worldmodel_optimizer_updates=0,
        metric_state_sha256=sha(out/'metric-state.pt'),metric_checkpoint_restore_passed=True,gpu_context_created=False,
        new_physical_episodes=0,state_reward_action_task_labels=False,external_pretrained_weights=False,
        scope='T2G-adaptedbaselinecost,NOTnovel/officialMetaWorldreplication/performanceclaim. PublicCube5epoch/lr5e-5/batch128/depth2hidden256/margin1/gap2/max15/negative0. '
            'Fixedown100epochscratchworldmodelfrozen;nativebodyretainsbatchregs,NOTentiresamplelevelcandidate. '
            'Allphase0RGBframes,same40fit/5train-validation/5testoriginaltrajectoriespertask. '
            'Raggedclips4..16retainshortepisodeswithoutpadding;length<4explicitlyexcluded. Samplingpubliclegacywithinclip;notfixed16framepublicdataset. '
            'Nostate/action/reward/taskIDtargets/inputs,unknownfutureactiontargets,strongteacherweights,privilegedreadoutsornewphysicaldata. '
            'Temporalrankdiagnosticonly;noMPC/controlsuccessyet.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['encode','fit'],required=True)
    p.add_argument('--name',choices=['smwm-seed1','h-jepa-seed2'],required=True);a=p.parse_args()
    try:(encode if a.mode=='encode' else fit)(a.name)
    except BaseException:
        out=ROOT/a.name;out.mkdir(parents=True,exist_ok=True)
        save(out/f'ERROR-{a.mode}.json',dict(traceback=traceback.format_exc()));raise
