"""Prepare exactly8 native two-update runs+8 strict audits; no600 route."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

B=Path('/efs/prescottsun/SPT-JEPA')
OLD=B/'runs/local_isotropic600_20261031'
PY='/efs/prescottsun/project_jepa/envs/lewm/bin/python'
PREFIX='experiments/patch_token_20261031/'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def allocations():
    slots=[('dev2',0),('dev2',1),('dev2',2),('dev13',0),('dev13',1),('dev13',2),('dev15',0),('dev15',1)]
    return [dict(id=f'{task}-patch-{basis}-seed{seed}',task=task,training_seed=seed,arm=basis,host=host,gpu=gpu)
        for (task,seed,basis),(host,gpu) in zip(itertools.product(('tworoom','cube'),(1,2),('identity','orthogonal')),slots)]
def configure(old,a,output):
    cfg=copy.deepcopy(old)
    assert cfg['model_class']=='spatial_carrier' and cfg['model']['spatial_mode']=='local'
    assert cfg['seed']==a['training_seed'] and cfg['batch_size']==128 and cfg['max_steps']==124910
    m=cfg['model'];assert m['window']==8 and m['beta']==.25 and m['sigma0']==.1 and m['rho']==.8 and m['forecast_fraction']==.5
    assert m.get('carry',True) is True and m.get('spatial_basis','identity')=='identity'
    assert m.get('spatial_chart','none')=='none' and m.get('spatial_readout_mode','none')=='none'
    assert cfg.get('freeze_encoder_steps',0)==0 and cfg.get('calibrate_state_std') is None
    assert not cfg.get('initial_weights') and not cfg.get('initial_encoder_checkpoint')
    assert len(set(m['spatial_scale']))==1
    cfg.update(output=str(output),model_class='patch_carrier',checkpoint_every=2,retain_steps=[2],record_initial_weights=True)
    cfg['model'].update(patch_basis=a['arm'],patch_predictor_seed=a['training_seed'])
    cfg['audit_expectations']=dict(gradients=dict(action_encoder='active'),
        initial_unchanged_prefixes=['spatial_projection','spatial_mean','spatial_scale']+
            (['predictor.coordinate_basis'] if a['arm']=='orthogonal' else []))
    return cfg
def jobs(root,source,configs):
    result=[]
    for a in allocations():
        name=a['id'];out=root/name;old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}'
        peer=f'{a["task"]}-patch-{"orthogonal" if a["arm"]=="identity" else "identity"}-seed{a["training_seed"]}'
        common={k:v for k,v in a.items() if k!='id'};common['source']=source
        result.append(dict(**common,id='train2-'+name,phase='train2',output=str(out),config=str(out/'config.json'),
            command=[PY,'-B','-u','-m','as_jepa_tools.train_stage','--config',str(out/'config.json'),'--stop-step','2']))
        dest=root/('fullstate-'+name)
        result.append(dict(**common,id=dest.name,phase='fullstate',output=str(dest),
            depends_on=['train2-'+name,'train2-'+peer],
            command=[PY,'-B','-u',str(Path(source['source'])/PREFIX/'preflight.py'),'--run',str(out),
                '--output',str(dest),'--reference-initial',str(old/'initial-model/weights.pt'),
                '--paired-initial',str(root/peer/'initial-model/weights.pt'),'--expected-step','2']))
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--source-record',required=True)
    p.add_argument('--root',default=str(B/'runs/patch_token_preflight_20261031'));args=p.parse_args()
    root=Path(args.root);assert not root.exists()
    source=read(args.source_record);manifest=read(source['manifest'])
    assert source['source_commit']==manifest['source_commit']
    for name,digest in manifest['files'].items():assert sha(Path(source['source'])/name)==digest,name
    assert sha(__file__)==manifest['files'][PREFIX+'prepare_preflight.py']
    cpu=read(Path(source['source'])/PREFIX/'cpu/production.json')
    assert cpu['passed'] and not cpu['CUDA_initialized'] and cpu['default_model_source_exact']
    for name,digest in cpu['source_sha256'].items():assert manifest['files'][name]==digest,name
    assert cpu['no_CUDA_reseed_called'] and cpu['actual_production_mask_contracts']['passed']
    prep=read(Path(source['source'])/PREFIX/'cpu/preparation.json')
    assert prep['passed'] and prep['endpoints']==16 and prep['only_native_two_updates']
    for name,digest in prep['source_sha256'].items():assert manifest['files'][PREFIX+name]==digest,name
    configs={};records=[]
    for a in allocations():
        old=OLD/f'{a["task"]}-spatial-local-seed{a["training_seed"]}';original=read(old/'config.json')
        cal=original['spatial_calibration'];assert sha(cal['path'])==cal['sha256']
        assert read(cal['path'])['normalization']=='global_rms' and read(cal['path'])['measured_frames']==128
        cfg=configure(original,a,root/a['id']);configs[a['id']]=cfg
        records.append(dict(**a,old_config=str(old/'config.json'),old_config_sha256=sha(old/'config.json'),
            old_initial=str(old/'initial-model/weights.pt'),old_initial_sha256=sha(old/'initial-model/weights.pt'),calibration=cal,
            data_card_sha256=sha(Path(cfg['data'])/'dataset_card.json')))
    prepared=jobs(root,source,configs);ids={j['id'] for j in prepared}
    assert len(prepared)==len(ids)==16
    assert all(set(j.get('depends_on',[]))<=ids for j in prepared)
    for a in allocations():
        left=copy.deepcopy(configs[a['id']]);left.pop('output');left['model'].pop('patch_basis');left.pop('audit_expectations')
        peer=f'{a["task"]}-patch-{"orthogonal" if a["arm"]=="identity" else "identity"}-seed{a["training_seed"]}'
        right=copy.deepcopy(configs[peer]);right.pop('output');right['model'].pop('patch_basis');right.pop('audit_expectations')
        assert left==right
    for name,cfg in configs.items():write(root/name/'config.json',cfg)
    for job in prepared:
        if job['phase']=='train2':job['config_sha256']=sha(job['config'])
    write(root/'source-record.json',source)
    write(root/'candidate-plan.json',dict(root=str(root),deadline=None,environment=read(OLD/'plan.json')['environment'],jobs=prepared))
    write(root/'protocol.json',dict(source=source,runs=records,endpoints=16,steps=2,batch=128,
        purpose='Actual native two-update/memory/fullstate/gradient gate only; no600 plan or authorization.',
        unchanged='Original raw local1024 encoder/calibration/action/DiT/loss/noise/data/goal metric/nativeTrainer/schedule.',
        difference='Replace only predictor; spatial token vs fixed orthogonal internal state coordinates. No innovation claim.',
        predictor_seed='Explicit training seed; paired arms same, seeds1/2 different. All original nonpredictor tensors and initialization outer RNG exact.',
        runtime_noise='Native flat vs token dropout consumes different amounts of randomness; distributions unchanged, no exact per-draw claim.',
        failure='Preserve actual memory/numerical/gradient failures; no batch/width/clip/planner reduction or automatic continuation.'))
    (root/'HOLD').write_text('Prepared two-step candidates; activate only after frozen-source and actual-free audit.\n')
    print(json.dumps(dict(root=str(root),endpoints=16,activated=False)))
if __name__=='__main__':main()
