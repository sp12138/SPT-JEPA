"""Focused tests of user-selected branches,not their exhaustive specifications."""
import argparse
import copy
import inspect
from pathlib import Path

from common import B,read,write,sha,source_check


def forecast_kwargs(cfg):
    from as_jepa_lewm.model import ASJepaLeWM
    allowed=set(inspect.signature(ASJepaLeWM.__init__).parameters)-{'self'}
    return {k:v for k,v in cfg['model'].items() if k in allowed}


def fixed_kwargs(cfg):
    result=dict(cfg['model'])
    for key in list(result):
        if key.startswith('renderer_') or key in ['beta','sigma0','rho','forecast_fraction','flow_logit_shift']:
            result.pop(key)
    return result


def main(root,source_record):
    root=Path(root);assert not root.exists();source=read(source_record);source_check(source)
    old=B/'runs/unified_three_20261101';parent=read(old/'protocol.json')
    from as_jepa_tools.benchmark_gap import GapBenchmarkWindows
    gaps={};tables={};manifests={}
    prior_plan=read(B/'runs/trajectory_search_local_20261101/candidate-plan.json')
    prior_specs=[read(j['spec']) for j in prior_plan['jobs']]
    for task in ['tworoom','cube']:
        ds=GapBenchmarkWindows(B/f'data/hardtasks-v1/{task}-metadata')
        table={gap:float(__import__('numpy').minimum(ds.counts,gap*ds.stride).sum()/ds.natural_length) for gap in [1,2,4,8,16]}
        gap=max(k for k,v in table.items() if v<=.2)
        gaps[task]=gap;tables[task]=table
        data=str(B/f'data/hardtasks-v1/{task}-metadata')
        selected={s['manifest'] for s in prior_specs if s['data']==data and s['case_set'] in ['fresh',task+'-fresh']}
        assert len(selected)==1
        manifests[task]=dict(primary=str(Path(data)/'manifests/validation-strict.json'),fresh=selected.pop())
        for path in manifests[task].values():assert len(read(path)['pairs'])==50
    runs=[]
    def add(task,seed,family,arm,host,changes,model):
        base=copy.deepcopy(read(old/f'{task}-{"flat" if family=="query" else "tf"}-seed{seed}/config.json'))
        for key in ['robust_variant','research_variant','frontier_variant','audit_expectations','stop_after','freeze_encoder_steps','calibrate_state_std']:base.pop(key,None)
        ident=f'{family}-{task}-{arm}-seed{seed}';run=root/ident
        base.update(output=str(run),model=model,record_initial_weights=True,export_initial=True,
                    retain_steps=[2,2000],checkpoint_every=1000,max_seconds=21600,ramp_steps=0,
                    trace_batches=True,workers=4,record_runtime_source=True)
        base.update(changes)
        encoder=old/f'{task}-flat-seed{seed}'/'initial-model'
        base['initial_encoder_checkpoint']=str(encoder)
        base['initial_encoder_scope']='Exact original RANDOM encoder initialization;no learned/pretrained encoder data'
        base['frozen_encoder_sha256']=sha(encoder/'weights.pt')
        if family!='query':base['audit_expectations']=dict(gradients=dict(encoder='active',pool='active',predictor='active',action_encoder='active',**({'renderer':'active'} if family=='round2' else {})))
        write(run/'config.json',base)
        runs.append(dict(id=ident,task=task,seed=seed,family=family,arm=arm,host=host,run=str(run),config_sha256=sha(run/'config.json')))
    # Machine1: learned future-query function versus the fixed direct forecast.
    for task in ['pusht','tworoom','cube']:
        for seed in [1,2]:
            c=read(old/f'{task}-flat-seed{seed}/config.json');m=fixed_kwargs(c)
            add(task,seed,'query','function','dev2',dict(model_class='goal_query',goal_query_reference=True),dict(m,query_seed=seed))
            add(task,seed,'query','forecast','dev2',dict(model_class='fixed_forecast'),dict(m,predictor_kind='flat'))
    # Machine2: six sample-level runs plus two explicitly non-sample controls.
    for task in ['pusht','tworoom','cube']:
        for seed in [1,2]:
            c=read(old/f'{task}-tf-seed{seed}/config.json');m=forecast_kwargs(c)
            m.update(beta=.3,lam=1.,horizon='one_step',forecast_noise=False,prior_input='sample',estimator='logdet_window',conditioning='anchor',forecast_seed=seed)
            extra=dict(model_class='decoder_free',calibrate_state_std=.1)
            if task!='pusht':extra.update(benchmark_anchor_rule='past_gap',gap=gaps[task]);m['anchor_gap']=gaps[task]
            add(task,seed,'decoder','window','dev13',extra,m)
            if task=='tworoom':add(task,seed,'decoder','pooled','dev13',extra,dict(m,estimator='logdet'))
    # Machine3: separate scale from anchor through a complete2x2 on TwoRoom.
    for seed in [1,2]:
        c=read(old/f'tworoom-tf-seed{seed}/config.json')
        for scale in ['original','calibrated']:
            for anchor in ['near','past']:
                m=forecast_kwargs(c);m.update(beta=.25,lam=1.,horizon='one_step',forecast_noise='sample',rho=0.,forecast_grad='joint',forecast_seed=seed)
                gap=0 if anchor=='near' else gaps['tworoom'];m['anchor_gap']=gap
                extra=dict(model_class='forecast_scored',benchmark_anchor_rule='past_gap',gap=gap)
                if scale=='calibrated':extra['calibrate_state_std']=.1
                add('tworoom',seed,'round2',scale+'-'+anchor,'dev15',extra,m)
    assert len(runs)==28
    order=[('pusht',1),('pusht',2),('tworoom',1),('cube',1),('tworoom',2),('cube',2)]
    query=[r for pair in order for r in runs if r['family']=='query' and (r['task'],r['seed'])==pair]
    runs=query+[r for r in runs if r['family']!='query']
    for task in ['pusht','tworoom','cube']:write(root/'cohorts'/f'{task}.json',read(old/'cohorts'/f'{task}.json'))
    env=dict(parent['environment']);env['TMPDIR']='/tmp/spt-jepa-branch-trials-20261102'
    protocol=dict(runs=runs,source=source,environment=env,stop_step=2000,preflight_step=2,
                  source_branches=dict(decoder_free='fbb3efa6a44fce8ca4eaa53b9c36eb2b962ffcf2',round2='43962c526f825a8bbba28f424feddfada330910b'),
                  gap_selection=dict(rule='Largest1/2/4/8/16 with exact train fallback share<=.2;no physical-label/performance selection',gaps=gaps,tables=tables),
                  host_families=dict(dev2='query',dev13='decoder',dev15='round2'),
                  manifests=manifests,manifest_sha256={path:sha(path) for task in manifests.values() for path in task.values()},
                  scope='Focused mechanism trials under renewed user authorization,not wholesale reproduction of uploadedspec grids. '
                        'Two training seeds,same task cases/nativeCEM budgets;query cost is learned,classifier references independent peritem. '
                        'PooledTwoRoom controls are batch-coupled,not sample-level candidates. '
                        'All new encoder starts load original randominitial weights,not trained/pretrainedencoders. '
                        '2k is boundedscreen,not convergence;maturePushT comparisons requirelargerbudget.',
                  original_goal='paused;notresumed',continuous_monitor=False,automatic_long_training=False)
    write(root/'protocol.json',protocol)
    print(dict(runs=len(runs),gaps=gaps,counts={h:sum(x['host']==h for x in runs) for h in ['dev2','dev13','dev15']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args();main(a.root,a.source)
