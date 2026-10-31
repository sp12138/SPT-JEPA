"""Twelve held Cube models, one old executed pool; no simulation or fitting."""
import argparse
import ast
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

COMMON = Path(__file__).resolve().parents[1]/'planner_context_20261031/assay.py'
spec = importlib.util.spec_from_file_location('context_common', COMMON)
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
B, PY, read, write, sha = C.B, C.PY, C.read, C.write, C.sha
INVENTORY = B/'reports/cube-pool-reuse-review-20261031/inventory.json'
ENCODER = B/'reports/cube-pool-reuse-review-20261031/encoder-contract.json'


def tensor_digest(weights):
    import torch, hashlib
    h = hashlib.sha256()
    keys = sorted(k for k in weights if k.startswith('encoder.') or k in ['spatial_projection','spatial_mean','spatial_scale'])
    for k in keys:
        t = weights[k].contiguous(); h.update(str((k,tuple(t.shape),t.dtype)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest(), len(keys)


def geometry_contract(cfg, weights, expected):
    assert cfg['model']['spatial_mode']=='local'
    assert cfg['model'].get('spatial_basis','identity')=='identity' and cfg['model'].get('spatial_chart','none')=='none'
    assert tensor_digest(weights)==(expected,201)


def prepare(a):
    import numpy as np, torch
    torch.set_num_threads(2); root=Path(a.root); assert not root.exists()
    inv=read(INVENTORY); enc=read(ENCODER); helper=read(a.source_record) if a.source_record else None
    script=Path(helper['source'])/'experiments/cube_pool_reuse_20261031/assay.py' if helper else Path(__file__).resolve()
    if helper: C.source_check(helper); assert sha(script)==sha(__file__)
    assert inv['cases']==50 and inv['candidates_per_case']==39 and inv['all12_current_primary_pairs_identical']
    for case in inv['cases_provenance']: assert sha(case['path'])==case['sha256']
    assert enc['all_local_four_per_seed_encoder_projection_normalization_tensors_exact']
    sourcechecks=set(); jobs=[]; specs=[]
    # Source equivalence: the three actual encoder primitives are identical ASTs;
    # added dynamic basis/chart functions are identity branches in these configs.
    f0,f1=[x['functions'] for x in enc['source_comparison']]
    for name in ['patch_features','project_patches','raw_carrier']: assert f0[name]['sha256_AST']==f1[name]['sha256_AST']
    for entry in enc['source_comparison']:
        text=Path(entry['path']).read_text(); tree=ast.parse(text); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='SpatialCarrierLeWM')
        fs={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        for name,x in entry['functions'].items(): assert ast.get_source_segment(text,fs[name])==x['source']
        if 'transform_spatial_basis' in fs:
            for name,condition in [('transform_spatial_basis',"if self.spatial_basis=='identity':return state"),('transform_spatial_chart',"if self.spatial_chart=='none':return state")]:
                assert condition in ast.get_source_segment(text,fs[name])
    ordered=[]
    for seed in [1,2]:
        for kind in ['identity','static','state','local6000']:
            m=next(x for x in inv['models'] if x['seed']==seed and (('local_isotropic2000' in x['run']) if kind=='identity' else ('readout-'+kind in x['run']) if kind in ['static','state'] else ('local_isotropic6000' in x['run'])))
            ordered.append((kind,m))
    for kind in ['native12000','tf12000']:
        for seed in [1,2]: ordered.append((kind,next(x for x in inv['models'] if x['seed']==seed and ('native_cube12000' if kind=='native12000' else 'cube_tf12000') in x['run'])))
    for i,(kind,m) in enumerate(ordered):
        run=Path(m['run']); parent=run.parent; source=read(parent/'source-record.json')
        if source['source_commit'] not in sourcechecks:C.source_check(source);sourcechecks.add(source['source_commit'])
        cfg=read(run/'config.json'); weights=Path(m['checkpoint'])/'weights.pt'; weightsha=sha(weights)
        dim=1024 if i<8 else 192 if kind=='native12000' else 64; name=f'{kind}-seed{m["seed"]}'
        geometry=f'local-seed{m["seed"]}' if i<8 else name; owner=kind in ['identity','native12000','tf12000']; expected=None
        if i<8:
            expected=next(x['encoder_geometry_sha256']for x in enc['rows']if x['checkpoint']==m['checkpoint'])
            w=torch.load(weights,map_location='cpu',weights_only=True,mmap=True);geometry_contract(cfg,w,expected);del w
        assert cfg['model']['window']==8 and cfg['model']['action_dim']==25
        s=dict(id=name,kind=kind,seed=m['seed'],dimension=dim,model_source=source,helper_source=helper,helper_sha256=sha(__file__),common_sha256=sha(COMMON),
            config=str(run/'config.json'),config_sha256=sha(run/'config.json'),weights=str(weights),weights_sha256=weightsha,
            encoder_geometry_sha256=expected,encoder_owner=owner,encoded_root=str(root/'encoded'/geometry),
            owner_id=f'identity-seed{m["seed"]}' if i<8 else name,inventory=str(INVENTORY),inventory_sha256=sha(INVENTORY),output=str(root/name),
            budget=dict(rgb_frames=2050 if owner else 0,encoder=300 if owner else 0,predict=250,action_encoder=50,renderer=0))
        sp=root/'specs'/(name+'.json');write(sp,s);specs.append(s)
        host,gpu=('dev2',i)if i<4 else ('dev13',i)if i<8 else ('dev15',i-8)
        jobs.append(dict(id=name,host=host,gpu=gpu,hold=True,output=s['output'],source=helper,depends_on=[]if owner else[s['owner_id']],
                         command=[PY,'-B','-u',str(script),'run','--spec',str(sp)]))
    world=B/'sources/stable-worldmodel/stable_worldmodel/world/world.py';text=world.read_text().splitlines()
    assert 'for rows in (init_rows, goal_rows):' in '\n'.join(text[580:605]) and 'self.infos[key] = np.broadcast_to(' in '\n'.join(text[580:605])
    sourcefiles=[world,B/'sources/stable-worldmodel/stable_worldmodel/wm/lewm/lewm.py',B/'sources/clear-lewm-20261030/clear_lewm/runner.py']
    protocol=dict(specs=specs,inventory_sha256=sha(INVENTORY),encoder_contract_sha256=sha(ENCODER),source_hashes={str(p):sha(p)for p in sourcefiles},
        initial_observation='Original dataset start_pixels, as actual World dataset infos injection; NOT environment-rendered pool pixels[:,0].',
        initial_source_excerpt=[dict(line=i+1,text=x)for i,x in enumerate(text)if 575<=i+1<=626],
        totals={k:sum(s['budget'][k]for s in specs)for k in specs[0]['budget']},
        scope='50 old cases,39 unchanged raw-command candidates,1 real+5 predicted. Zero simulation/fit/renderer/training. Original-source raw BN. Old bank lacks warm3+5/fresh/second-plan outcomes; candidate proposal sources are old models. All groups/seed/cases retained.')
    assert protocol['totals']==dict(rgb_frames=12300,encoder=1800,predict=3000,action_encoder=600,renderer=0)
    write(root/'protocol.json',protocol);write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment={'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},jobs=jobs))
    (root/'HOLD_DISPATCH').write_text('Candidate; preserve until frozen helper and actual GPU availability verified.\n')
    # Algebra/data-only contract; does not execute a world model.
    z=np.arange(39*7,dtype=np.float64).reshape(39,7)/100;g=np.arange(7)/13
    cost=np.square(z-g).sum(-1);assert cost.shape==(39,)and np.all(cost>=0)
    assert np.array_equal(np.argsort(cost),np.argsort(cost*7))
    write(root/'contracts.json',dict(passed=True,models=12,encoder_maps=6,local_selected_tensors_per_model=201,model_forward_calls=0,
        native_H1_T5='same exact pinned LeWM.rollout as passing context assay62a8022; all checkpoints max prefix7 and actiondim25',shape_only_pixels_with_prefilled_emb=True))
    print(json.dumps(dict(root=str(root),passed=True,totals=protocol['totals'])),flush=True)


def enter(a):
    s=read(a.spec);env={**os.environ,'PYTHONPATH':s['model_source']['pythonpath'],'PYTHONDONTWRITEBYTECODE':'1'}
    os.chdir(s['model_source']['source']);os.execve(PY,[PY,'-B','-u',str(Path(__file__).resolve()),'inner','--spec',a.spec],env)


def inner(a):
    import numpy as np, torch
    from as_jepa_lewm.factory import build_model
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from clear_lewm.runner import _image_transform
    from vforge.ops import OpsConfig,initialize_ops
    s=read(a.spec);out=Path(s['output']);out.mkdir(exist_ok=True);assert not(out/'result.json').exists()
    assert sha(__file__)==s['helper_sha256']and sha(COMMON)==s['common_sha256'] and sha(s['inventory'])==s['inventory_sha256']
    C.source_check(s['model_source']);cfg=read(s['config']);assert sha(s['config'])==s['config_sha256']and sha(s['weights'])==s['weights_sha256']
    inv=read(s['inventory']);torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(1042)
    device=torch.device('cuda');initialize_ops(OpsConfig(backend='sdpa',allow_fallback=False),device=device,parallel_state=None)
    w=torch.load(s['weights'],map_location='cpu',weights_only=True)
    if s['encoder_geometry_sha256']:geometry_contract(cfg,w,s['encoder_geometry_sha256'])
    model=build_model(cfg).to(device).eval();model.requires_grad_(False);model.load_state_dict(w,strict=True);del w
    assert model.rollout.__func__ is LeWM.rollout and model.predictor.num_frames==7
    if s['encoder_geometry_sha256']:
        assert getattr(model,'spatial_basis','identity')=='identity'and getattr(model,'spatial_chart','none')=='none'
    cls=Path(inspect.getsourcefile(type(model))).resolve();assert cls.is_relative_to(Path(s['model_source']['source']))
    imports={str(p):sha(p)for p in [cls,Path(inspect.getsourcefile(LeWM)),Path(inspect.getsourcefile(_image_transform))]}
    card=read(Path(cfg['data'])/'dataset_card.json');mean=np.asarray(card['action_mean']);std=np.asarray(card['action_std'])
    transform=_image_transform(224);counts={k:0 for k in s['budget']};lengths=[]
    def forbid(*a,**kw):raise AssertionError('Renderer forbidden')
    def encoder_hook(module,args):
        counts['encoder']+=1;counts['rgb_frames']+=len(args[0]);assert counts['encoder']<=s['budget']['encoder']and counts['rgb_frames']<=s['budget']['rgb_frames']
    def action_hook(module,args):counts['action_encoder']+=1;assert counts['action_encoder']<=50
    handles=[model.encoder.register_forward_pre_hook(encoder_hook),model.action_encoder.register_forward_pre_hook(action_hook)]
    if getattr(model,'renderer',None)is not None:handles.append(model.renderer.register_forward_pre_hook(forbid))
    original=model.predict
    def predict(z,actions):
        counts['predict']+=1;assert counts['predict']<=250;lengths.append(z.shape[1]);assert z.shape[:2]==actions.shape[:2];return original(z,actions)
    model.predict=predict
    def encode(images):
        t=torch.from_numpy(np.array(images,copy=True)).permute(0,3,1,2);pixels=transform(t).to(device)[:,None]
        return model.encode({'pixels':pixels})['emb'][:,0]
    torch.cuda.synchronize();before=C.state_hash(model);rngbefore=C.rng_hash();start=time.time();torch.cuda.reset_peak_memory_stats()
    cases=[];encoded=Path(s['encoded_root']);encoded.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for record in inv['cases_provenance']:
            p=Path(record['path']);assert sha(p)==record['sha256'];case=record['case'];dest=encoded/f'case{case:04d}.npz'
            with np.load(p)as f:
                raw=f['actions'];families=f['family'];assert raw.shape==(39,25,5)
                if s['encoder_owner']:
                    assert not dest.exists();pixels=f['pixels'];sg=encode(np.stack([f['start_pixels'],f['goal_pixels']]))
                    true=torch.cat([encode(pixels[lo:lo+8,-1])for lo in range(0,39,8)])
                    initial_mse=np.square(pixels[:,0].astype(np.float64)-f['start_pixels'].astype(np.float64)).mean((1,2,3))
                    np.savez_compressed(dest,start=sg[0].cpu().numpy(),goal=sg[1].cpu().numpy(),true=true.cpu().numpy(),initial_rgb_mse=initial_mse,
                        pool_sha256=record['sha256'],owner=s['id'],owner_weights_sha256=s['weights_sha256'],encoder_geometry_sha256=s['encoder_geometry_sha256']or'not_shared')
                with np.load(dest)as e:
                    assert str(e['pool_sha256'].item())==record['sha256']and str(e['owner'].item())==s['owner_id']
                    if s['encoder_geometry_sha256']:assert str(e['encoder_geometry_sha256'].item())==s['encoder_geometry_sha256']
                    initial=torch.from_numpy(e['start'].copy()).to(device);goal=torch.from_numpy(e['goal'].copy()).to(device);true=torch.from_numpy(e['true'].copy()).to(device);initial_mse=e['initial_rgb_mse'].copy()
                assert initial.shape==goal.shape==(s['dimension'],)and true.shape==(39,s['dimension'])
                # Exact train-card StandardScaler float32 transform order;
                # preserve original raw commands, with no Box projection.
                normalized=raw.copy();normalized-=mean;normalized/=std
                actions=torch.from_numpy(normalized.reshape(39,5,25)).to(device)
                # One case,39 candidate samples; prefilled dataset initial encoding.
                info={'pixels':initial.new_empty(1,39,1,0),'emb':initial[None,None,None].expand(1,39,1,-1)}
                pred=model.rollout(info,actions[None])['predicted_emb'][0]
                assert pred.shape==(39,6,s['dimension'])and torch.isfinite(pred).all()and torch.equal(pred[:,0],initial[None].expand(39,-1))
                predicted=(pred[:,-1]-goal).square().sum(-1).cpu().numpy();actual=(true-goal).square().sum(-1).cpu().numpy()
                target=out/f'case{case:04d}.npz';assert not target.exists()
                np.savez_compressed(target,predicted_cost=predicted,encoded_cost=actual,physical_cost=f['physical_cost'],position_error_m=f['position_error_m'],
                    orientation_error_deg=f['orientation_error_deg'],success=f['success'],hold_count=f['hold_count'],family=families,raw_actions=raw,
                    predicted_endpoint=pred[:,-1].cpu().numpy(),true_endpoint=true.cpu().numpy(),goal=goal.cpu().numpy(),initial=initial.cpu().numpy(),initial_rgb_mse=initial_mse)
                cases.append(dict(case=case,file=str(target),sha256=sha(target),encoded_sha256=sha(dest),pool_sha256=record['sha256']))
    torch.cuda.synchronize();seconds=time.time()-start;after=C.state_hash(model);rngafter=C.rng_hash()
    assert before==after and rngbefore==rngafter and counts==s['budget'] and lengths==list(range(1,6))*50
    assert all(not m.training for m in model.modules())and sha(s['weights'])==s['weights_sha256']
    assert all(sha(p)==h for p,h in imports.items())
    write(out/'result.json',dict(passed=True,spec=s,cases=cases,actual_calls=counts,state_before=before,state_after=after,rng_before=rngbefore,rng_after=rngafter,
        seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30,actual_imports=imports,raw_BN=True,source_unchanged=True))
    print(json.dumps(dict(passed=True,id=s['id'],cases=50,counts=counts,seconds=seconds)),flush=True)


def collect(a):
    import numpy as np
    from scipy.stats import spearmanr
    root=Path(a.root);plan=read(root/'plan.json');assert len(plan['jobs'])==12;rows=[];totals=None
    def corr(x,y):
        return None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic)
    def interval(x):
        x=np.asarray([v for v in x if v is not None],float)
        if not len(x):return dict(mean=None,ci95=None,n=0)
        draws=x[np.random.default_rng(1042).integers(len(x),size=(5000,len(x)))].mean(1)
        return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n=len(x))
    for job in plan['jobs']:
        assert read(root/'exits'/(job['id']+'.json'))['returncode']==0
        report=read(Path(job['output'])/'result.json');assert report['passed']and report['state_before']==report['state_after']and report['rng_before']==report['rng_after']
        s=report['spec'];conditions={k:[]for k in ['nonoracle','oracle_neighborhood','all']};zero=[]
        if totals is None:totals={k:0 for k in report['actual_calls']}
        for k,v in report['actual_calls'].items():totals[k]+=v
        for rec in report['cases']:
            assert sha(rec['file'])==rec['sha256']
            with np.load(rec['file'])as f:
                pc,ec,ph=f['predicted_cost'],f['encoded_cost'],f['physical_cost'][:,-1];pos=f['position_error_m'][:,-1];ang=f['orientation_error_deg'][:,-1];success=f['success'].any(1);family=f['family']
                assert len(pc)==len(ec)==len(ph)==39 and np.isfinite(pc).all()and np.isfinite(ec).all()
                def truth(i):return dict(index=int(i),family=str(family[i]),pose_cost=float(ph[i]),position_m=float(pos[i]),angle_deg=float(ang[i]),position_pass=float(pos[i]<=.03),angle_pass=float(ang[i]<=15),joint_pose_pass=float(pos[i]<=.03 and ang[i]<=15),hold_success=float(success[i]))
                zero.append(dict(case=rec['case'],**truth(int(np.flatnonzero(family=='zero')[0]))))
                groups=dict(nonoracle=np.flatnonzero(np.isin(family,['zero','gaussian'])|np.char.startswith(family,'cem_')),oracle_neighborhood=np.flatnonzero(np.char.startswith(family,'recorded')),all=np.arange(39))
                assert {k:len(v)for k,v in groups.items()}==dict(nonoracle=22,oracle_neighborhood=17,all=39)
                for name,ids in groups.items():
                    pi,ei,hi=[int(ids[np.argmin(v[ids])])for v in [pc,ec,ph]]
                    selections={name:truth(i)for name,i in [('predicted',pi),('encoded_oracle',ei),('physical_oracle',hi)]}
                    v=dict(case=rec['case'],count=len(ids),physical_encoded_rank=corr(ph[ids],ec[ids]),encoded_predicted_rank=corr(ec[ids],pc[ids]),physical_predicted_rank=corr(ph[ids],pc[ids]),position_encoded_rank=corr(pos[ids],ec[ids]),angle_encoded_rank=corr(ang[ids],ec[ids]),
                        coverage_hold=float(success[ids].any()),coverage_endpoint_pose=float(((pos[ids]<=.03)&(ang[ids]<=15)).any()),metric_regret=float(ph[ei]-ph[hi]),signed_dynamics_regret=float(ph[pi]-ph[ei]),total_pose_regret=float(ph[pi]-ph[hi]),
                        encoded_selection_regret_relative_range=float((ec[pi]-ec[ei])/np.ptp(ec[ids])) if np.ptp(ec[ids]) else 0.,selections=selections)
                    for selector,item in selections.items():
                        for key in ['pose_cost','position_m','angle_deg','position_pass','angle_pass','joint_pose_pass','hold_success']:v[selector+'_'+key]=item[key]
                    conditions[name].append(v)
        summaries={g:{k:interval([r[k]for r in rs])for k in rs[0]if k not in ['case','count','selections']}for g,rs in conditions.items()}
        rows.append(dict(id=s['id'],kind=s['kind'],seed=s['seed'],summary=summaries,cases=conditions,zero=zero,result=str(Path(job['output'])/'result.json'),result_sha256=sha(Path(job['output'])/'result.json')))
    assert totals==dict(rgb_frames=12300,encoder=1800,predict=3000,action_encoder=600,renderer=0)
    paired=[]
    for seed in [1,2]:
        ref=next(r for r in rows if r['kind']=='identity'and r['seed']==seed)
        for r in rows:
            if r['seed']!=seed or r['kind']=='identity':continue
            for group in ['nonoracle','oracle_neighborhood','all']:
                ar,br=r['cases'][group],ref['cases'][group];assert [v['case']for v in ar]==[v['case']for v in br]
                for metric in ['predicted_hold_success','predicted_pose_cost','physical_encoded_rank','encoded_predicted_rank','metric_regret','signed_dynamics_regret','total_pose_regret']:
                    diffs=[x[metric]-y[metric]for x,y in zip(ar,br)if x[metric]is not None and y[metric]is not None]
                    paired.append(dict(candidate=r['id'],reference=ref['id'],group=group,metric=metric,**interval(diffs)))
    # Encoded selector is exactly shared across the four local checkpoints/seed.
    for seed in [1,2]:
        ids=[r for r in rows if r['seed']==seed and r['kind']in ['identity','static','state','local6000']]
        for g in ['nonoracle','oracle_neighborhood','all']:
            assert all([v['selections']['encoded_oracle']for v in r['cases'][g]]==[v['selections']['encoded_oracle']for v in ids[0]['cases'][g]]for r in ids)
    result=dict(rows=rows,paired=paired,actual_calls=totals,all12_pass=True,local_encoded_oracle_invariant=True,
        scope='Fixed old39 candidates,50 cases,25 raw steps,raw command actions,first cold plan only. Bootstrap resamples cases, not candidates or training seeds. Old-model candidate bank is not new-model CEM coverage. No training/fitting/new simulator trajectories.')
    write(Path(a.output)/'results.json',result);write(Path(a.output)/'summary.json',{**result,'rows':[{k:v for k,v in r.items()if k not in ['cases','zero']}for r in rows]})
    print(json.dumps([dict(id=r['id'],nonoracle={k:r['summary']['nonoracle'][k]['mean']for k in ['physical_encoded_rank','encoded_predicted_rank','predicted_hold_success','encoded_oracle_hold_success','physical_oracle_hold_success','metric_regret','signed_dynamics_regret']})for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','inner','collect']);p.add_argument('--root');p.add_argument('--source-record');p.add_argument('--spec');p.add_argument('--output');a=p.parse_args()
    {'prepare':prepare,'run':enter,'inner':inner,'collect':collect}[a.mode](a)
