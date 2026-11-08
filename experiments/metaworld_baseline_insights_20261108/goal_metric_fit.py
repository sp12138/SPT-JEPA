"""Paired reference vs future-goal RGB ordering;no action/policy targets.

Public T2G metric architecture and numerical training loop are retained.
The reference replay tests exact equality to the archived public-head run.
The goal arm loads the exact same initial head tensors and reverses only
observed RGB time in the metric loss. This is an objective-alignment ablation,
not a claim of an original algorithm or an entirely sample-level world model.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import numpy as np
BASE=Path('/efs/prescottsun/SPT-JEPA')
ROOT=BASE/'runs/metaworld_goal_rank_20261108'
PARENT=BASE/'runs/metaworld_t2g_adapted_20261108'
PUBLIC=BASE/'reports/metaworld_baseline_insights_20261108/t2g-source-572784c/temporal_metric.py'
CONFIG=dict(hidden_dim=256,depth=2,dropout=0.,max_steps=15,min_steps=1,gap=2,margin=1.,negative_weight=0.,lr=5e-5,weight_decay=.001,batch_size=128,epochs=5,seed=20261108)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def fit(name,mode):
    import torch
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.manual_seed(CONFIG['seed']);assert not torch.cuda.is_initialized()
    out=ROOT/mode/name;out.mkdir(parents=True,exist_ok=False)
    parent=PARENT/name
    for filename in ['FROZEN.json','rgb_features.npz','episodes.json']:(out/filename).symlink_to(parent/filename)
    frozen=read(out/'FROZEN.json');assert frozen['passed'] and sha(out/'rgb_features.npz')==frozen['feature_sha256']
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
    if mode=='goal':
        initial=torch.load(ROOT/'reference'/name/'metric-init.pt',map_location='cpu',weights_only=True)
        metric.load_state_dict(initial,strict=True)
    torch.save({k:v.detach().clone() for k,v in metric.state_dict().items()},out/'metric-init.pt')
    optimizer=torch.optim.AdamW(metric.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    def triples(rows,generator):
        groups={}
        for offset,length in rows:groups.setdefault(length,[]).append(offset)
        result={k:[] for k in ['anchor','near','far']}
        for length,offsets in groups.items():
            index=torch.tensor(offsets)[:,None]+torch.arange(length)[None,:];emb=values[index]
            if mode=='goal':emb=emb.flip(1) # Anchor is a future RGB goal;near/far are earlier observed frames.
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
        scope='Goal-anchorobjectivealignmentablation,NOTnovelmethod/publishedreplication/controlperformanceclaim. PublicT2Gclass/configretained. '
            'Fixedown100epochscratchworldmodelfrozen;nativebodyretainsbatchregs,NOTentiresamplelevelcandidate. '
            'Allphase0RGBframes,same40fit/5train-validation/5testoriginaltrajectoriespertask. '
            'Raggedclips4..16retainshortepisodeswithoutpadding;length<4explicitlyexcluded. Samplingpubliclegacywithinclip;notfixed16framepublicdataset. '
            'Nostate/action/reward/taskIDtargets/inputs,unknownfutureactiontargets,strongteacherweights,privilegedreadoutsornewphysicaldata. '
            'Temporalrankdiagnosticonly;noMPC/controlsuccessyet.'))

    result=read(out/'summary.json');result.update(objective_direction=mode,initial_state_sha256=sha(out/'metric-init.pt'),
        frozen_feature_cache_reused=True,worldmodels_loaded_during_fit=0,
        objective='d(goal,later)<d(goal,earlier)' if mode=='goal' else 'd(start,near)<d(start,far)',
        direction_adapter='Reverse ONLY the RGB embedding clip before public triplet sampling;no action sequence or physical dynamics is reversed.')
    if mode=='reference':
        original=torch.load(PARENT/name/'metric-state.pt',map_location='cpu',weights_only=True)
        result['archived_public_reference_bitexact']=all(torch.equal(v,state[k]) for k,v in original.items())
        result['archived_public_max_parameter_difference']=max(float((v-state[k]).abs().max()) for k,v in original.items())
    else:
        initial=torch.load(out/'metric-init.pt',map_location='cpu',weights_only=True)
        reference=torch.load(ROOT/'reference'/name/'metric-init.pt',map_location='cpu',weights_only=True)
        result['reference_goal_initial_tensors_exact']=all(torch.equal(v,initial[k]) for k,v in reference.items())
        assert result['reference_goal_initial_tensors_exact']
    save(out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['name','objective_direction','head_optimizer_updates','validation','heldout']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['smwm-seed1','h-jepa-seed2'],required=True)
    p.add_argument('--direction',choices=['reference','goal'],required=True);a=p.parse_args()
    try:fit(a.name,a.direction)
    except BaseException:
        out=ROOT/a.direction/a.name;out.mkdir(parents=True,exist_ok=True)
        save(out/'ERROR.json',dict(traceback=traceback.format_exc()));raise
