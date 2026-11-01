"""Prepare18 bounded two-update native preflights;600 requires later review."""
import argparse
import copy
from pathlib import Path

from common import B, allocations, dataset, read, reference, sha, source_check, windows, write


def configure(a, root):
    original=reference(a['task'],a['seed'],'tf' if a['arm']=='tf' else 'flat')
    cfg=copy.deepcopy(read(original));assert cfg['seed']==a['seed'] and cfg['batch_size']==128
    for key in ['stop_after','initial_weights','initial_encoder_checkpoint','is_resuming']:
        cfg.pop(key,None)
    cfg.update(output=str(root/a['id']),record_initial_weights=True,checkpoint_every=200,
               retain_steps=[2,200,400,600],max_seconds=36000)
    if a['arm']=='tf':
        cfg['model_class']='robust'
    else:
        if a['task']=='pusht':
            cal=root/'calibration'/f'pusht-seed{a["seed"]}.json';v=read(cal);assert v['passed']
            cfg['model'].update(v['model_kwargs'])
            cfg['spatial_calibration']=dict(path=str(cal),sha256=sha(cal))
        cfg['model_class']='local_readout' if a['arm']=='field' else 'spatial_carrier'
        if a['arm']=='field':cfg['model']['local_readout_seed']=a['seed']
        assert len(set(cfg['model']['spatial_scale']))==1
    cfg['audit_expectations']=dict(gradients=dict(action_encoder='active'))
    if a['arm']!='tf':
        cfg['audit_expectations']['initial_unchanged_prefixes']=['spatial_projection','spatial_mean','spatial_scale']
    assert cfg['max_steps']==124910
    m=cfg['model'];assert m['rho']==.8 and m['forecast_fraction']==.5 and m['beta']==.25 and m['sigma0']==.1
    assert m.get('flow_logit_shift',0.)==0.
    return cfg,original


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    root=Path(a.root);assert not(root/'protocol.json').exists()
    source=read(a.source);source_check(source)
    cpu=read(root/'cpu-contracts.json');assert cpu['passed'] and not cpu['CUDA_initialized']
    runs=[];cohorts={}
    for row in allocations():
        cfg,original=configure(row,root)
        if row['task'] not in cohorts:
            ds=dataset(cfg,'validation');cohorts[row['task']]=windows(ds,64)
        path=root/row['id']/'config.json';write(path,cfg)
        runs.append(dict(**row,run=str(path.parent),config_sha256=sha(path),
                         reference_config=str(original),reference_config_sha256=sha(original),
                         data_card_sha256=sha(Path(cfg['data'])/'dataset_card.json')))
    for task,cohort in cohorts.items():write(root/'cohorts'/f'{task}.json',cohort)
    environment=read(B/'runs/local_isotropic600_20261031/plan.json')['environment']
    write(root/'protocol.json',dict(source=source,runs=runs,environment=environment,preflight_stop=2,short_gate_stop=600,
        scheduler_max_steps=124910,batch_size=128,loss='Unchanged sample-level Robust forecast FM+rate',
        controllers='Native GoalMSE/CEM300x30,top30,H5,history3,receding5,50rawsteps',
        evaluation='PushT original P0 heldout;TwoRoom/Cube existing custom CLEAR strict heldout;not canonical CLEAR or a uniform success predicate.',
        initialization='Flat/field share encoder/action/renderer/transformer body and outer RNG;only output head differs.',
        no_optimal_actions=True,no_auxiliary_labels=True,automatic_long_training=False,
        note='Native dataset/anchor rules preserved;PushT distant anchor differs from benchmark initial anchor. Three models have different state widths/capacity;flat vsfield is close matched capacity.',
        review_required_before600=True,review_required_before_long=True))
    print('Prepared18 native two-update gates',root)


if __name__=='__main__':main()
