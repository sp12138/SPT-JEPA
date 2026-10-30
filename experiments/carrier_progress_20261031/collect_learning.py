"""Complete-only CPU collection of the12 Carrier continuations and prior controls.

Reads saved artifacts once. Never polls, trains, simulates, or launches a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Analysis helpers were added after the immutable training snapshots. Running
# this file from the checkout must not accidentally resolve an older snapshot.
_repo = next((p for p in Path(__file__).resolve().parents
              if (p/'experiments/ideas_validate_20261031/collect.py').is_file()), None)
if _repo is not None:
    sys.path.insert(0, str(_repo))

from experiments.ideas_validate_20261031.collect import (
    canonical, digest, paired_outcomes, probe_label_summaries, read, reduce_state_audit,
)
from experiments.carrier_progress_20261031.prepare_learning import ARMS, assignments, semantic


B = Path('/efs/prescottsun/SPT-JEPA')
PHASES = ('learn1200', 'fullstate1200', 'learn2000', 'fullstate2000',
          'health', 'info', 'transition', 'planning')
CONTROLS = ('temporal-forecast', 'original', 'lower-beta')


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size)


def planning_signature(planning):
    return dict(planning_manifest_sha256=planning['manifest_sha256'], planning_pairs=planning['pairs'],
        **{key: planning[key] for key in ('solver', 'plan', 'protocol', 'split', 'normalization')})


def signature_from_reference(reference):
    return {key: reference[key] for key in ('planning_manifest_sha256', 'planning_pairs',
                                          'solver', 'plan', 'protocol', 'split', 'normalization')}


def successes(planning):
    values = planning['metrics']['episode_successes']
    assert len(values) == len(planning['pairs']) == 50 and all(v in (0, 1, False, True) for v in values)
    assert len({p['episode_id'] for p in planning['pairs']}) == 50
    rate = sum(values) * 2.
    assert abs(rate - planning['metrics']['success_rate']) < 1e-8
    assert planning['checkpoint_audit']['strict_equal']
    return [bool(v) for v in values], rate


def verify_manifest(source):
    manifest = read(source['manifest'])
    assert manifest['source_commit'] == source['source_commit']
    for name, sha in manifest['files'].items():
        assert digest(Path(manifest['source']) / name) == sha, name
    return artifact(source['manifest'])


def load_baselines(root, task_references):
    """AS-JEPA controls, not native LeWM training-budget replications."""
    root = Path(root); plan, protocol = read(root/'plan.json'), read(root/'protocol.json')
    assert protocol['stop_step'] == 2000 and protocol['scheduler_horizon'] == 124910
    rows, sources = [], {}
    for job in plan['jobs']:
        if job.get('phase') != 'learn2000' or job['method'] not in CONTROLS:
            continue
        name, task, seed = job['id'], job['task'], job['training_seed']
        for prefix in ('', 'health-', 'planning-'):
            assert read(root/'exits'/(prefix+name+'.json'))['returncode'] == 0
        run = Path(job['output']); cfg = read(run/'config.json')
        assert digest(run/'config.json') == job['config_sha256']
        stage = read(run/'stage-complete.json'); status = read(run/'status.json')
        health = read(root/('health-'+name)/'health.json')
        planning_path = root/('planning-'+name)/'metrics.json'; planning = read(planning_path)
        assert stage['stop_step'] == status['step'] == health['audit']['step'] == 2000
        assert status['state'] == 'finished' and all(v for v in health['audit'].values() if isinstance(v, bool))
        assert Path(planning['checkpoint']) == run/'model-step0002000'
        assert canonical(planning_signature(planning)) == canonical(signature_from_reference(task_references[task]))
        outcomes, rate = successes(planning)
        size = read(run/'model_size.json')
        assert size['total'] == health['parameters'] == planning['checkpoint_audit']['parameters']
        source = job['source']; commit = source['source_commit']
        if commit not in sources:
            sources[commit] = dict(source=source, manifest=verify_manifest(source))
        rows.append(dict(id=name, task=task, seed=seed, method=job['method'], step=2000,
            config=cfg, success_rate=rate, episode_successes=outcomes, model_size=size,
            source=source, planning_signature=planning_signature(planning),
            artifacts={key:artifact(path) for key,path in dict(config=run/'config.json',
                status=run/'status.json', health=root/('health-'+name)/'health.json',
                planning=planning_path, stage=run/'stage-complete.json').items()}))
    assert len(rows) == 12 and {(x['task'],x['seed'],x['method']) for x in rows} == {
        (t,s,m) for t in ('tworoom','cube') for s in (1,2) for m in CONTROLS}
    return rows, sources


def paired_summary(rows, bootstrap):
    """Same50 cases shared across two fixed seeds, reusing the prior paired logic."""
    assert sorted(r['seed'] for r in rows) == [1, 2]
    rows = sorted(rows, key=lambda r:r['seed'])
    matrix = np.asarray([r['per_case_difference'] for r in rows], dtype=float)
    assert matrix.shape == (2, 50)
    draws = matrix.mean(0)[np.random.default_rng(1042).integers(50, size=(bootstrap,50))].mean(1)*100
    return dict(seeds=[1,2], differences_by_seed_pp=[r['difference_pp'] for r in rows],
        mean_difference_pp=float(matrix.mean()*100),
        shared_case_bootstrap_ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
        scope='Conditional on these two trained seeds. Same case indices resampled jointly; '
              'not training-seed uncertainty or multiple-comparison-adjusted inference.')


def collect(root, initial_results, baseline_root, bootstrap=10000):
    root=Path(root); plan=read(root/'plan.json'); protocol=read(root/'protocol.json')
    src=read(root/'source-record.json'); compatibility=read(root/'source-compatibility.json')
    parent=Path(protocol['parent']); previous=read(initial_results); review=read(parent/'review600.json')
    assert digest(initial_results) == review['results_sha256']
    assert previous['completed_endpoint_count'] == 120 and previous['initial_tensor_identity']['passed']
    assert protocol['stop_step'] == 2000 and protocol['initial_step'] == 600 and protocol['runs'] == 12
    records=list(assignments()); assert protocol['allocations'] == records
    expected={phase+'-'+r['id'] for r in records for phase in PHASES}
    assert len(plan['jobs']) == 96 and {j['id'] for j in plan['jobs']} == expected
    assert src['source_commit'] == protocol['source_commit'] == compatibility['new_source']['source_commit']
    source_manifest=verify_manifest(src)
    assert source_manifest['sha256'] == compatibility['new_manifest_sha256']
    jobs={j['id']:j for j in plan['jobs']}; exits=[]
    for job in plan['jobs']:
        exit_record=read(root/'exits'/(job['id']+'.json'))
        assert exit_record['returncode'] == 0, job['id']
        assert exit_record['id'] == job['id']
        assert canonical(job['source']) == canonical(src)
        exits.append(dict(exit_record,phase=job['phase']))
    oldrows={r['id']:r for r in previous['rows'] if r['family']=='carrier'}
    assert len(oldrows)==12
    task_references=previous['task_references']; rows=[]; trajectories=[]
    for rec in records:
        name=rec['id']; run=root/name; old=oldrows[name]
        cfg=read(run/'config.json'); oldcfg=read(parent/name/'config.json')
        assert semantic(cfg)==semantic(oldcfg)
        for phase in ('learn1200','learn2000'):
            assert digest(run/'config.json')==jobs[phase+'-'+name]['config_sha256']
        status=read(run/'status.json'); stage=read(run/'stage-complete.json'); size=read(run/'model_size.json')
        assert status['state']=='finished' and status['step']==stage['stop_step']==2000 and stage['initial_step']==1200
        assert size==old['model_size'] and size['total']==size['trainable']+size['frozen']
        pre=read(run/'pre-resume-fullstate.json')
        assert pre['step']==600 and pre['passed'] and pre['copy_byte_identical'] and all(pre['checks'].values())
        full=[]
        for step in (1200,2000):
            p=root/f'fullstate{step}-{name}'/'fullstate.json'; audit=read(p)
            assert audit['step']==step and audit['passed'] and all(audit['checks'].values())
            assert audit['frozen_tensors']['equal'] and audit['source_commit']==src['source_commit']
            assert audit['config_sha256']==digest(run/'config.json')
            assert audit['campaign_samples']==step*cfg['batch_size']
            assert audit['native_stage_complete']['initial_step']==(600 if step==1200 else 1200)
            assert audit['native_stage_complete']['stop_step']==step
            assert Path(audit['archived_checkpoint'])==run/f'checkpoint-step{step:07d}.ckpt'
            assert Path(audit['archived_checkpoint']).is_file()
            full.append(dict(step=step,audit=audit,artifact=artifact(p),
                             archive_hash_scope='Hash verified by archival endpoint; collector does not reread full model files.'))
        paths={key:root/(prefix+name)/filename for key,prefix,filename in (
            ('health','health-','health.json'),('info','info-','information-audit.json'),
            ('state','transition-','state-audit.json'),('windows','transition-','windows.json'),
            ('planning','planning-','metrics.json'))}
        health,info,state,windows,planning=[read(paths[k]) for k in paths]
        assert health['audit']['step']==info['step']==state['step']==2000
        assert all(v for v in health['audit'].values() if isinstance(v,bool))
        assert state['config_sha256']==digest(run/'config.json')
        assert state['weights_sha256']==full[-1]['audit']['portable_sha256']
        assert state['manifest_sha256']==digest(paths['windows'])==old['state_manifest_sha256']
        assert state['cases']==windows['count']==64 and 'carrier_chart_certificate' in state
        assert canonical(windows)==canonical(task_references[rec['task']]['state_windows'])
        assert size['total']==health['parameters']==planning['checkpoint_audit']['parameters']
        assert Path(planning['checkpoint'])==run/'model-step0002000'
        assert canonical(planning_signature(planning))==canonical(signature_from_reference(task_references[rec['task']]))
        assert not set(info['probes']['train_episodes']) & set(info['probes']['validation_episodes'])
        for key in ('train_episodes','validation_episodes','label_keys'):
            assert info['probes'][key]==old['information']['probes'][key]
        assert [{k:b[k] for k in ('episodes','starts')} for b in info['batches']]==task_references[rec['task']]['information_windows']
        assert [{k:b[k] for k in ('episode','start')} for b in health['cases']]==task_references[rec['task']]['health_windows']
        outcomes,rate=successes(planning)
        paths.update(config=run/'config.json',pre_resume=run/'pre-resume-fullstate.json',status=run/'status.json')
        trace=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        rows.append(dict(**rec,seed=rec['training_seed'],step=2000,config=cfg,success_rate=rate,
            episode_successes=outcomes,model_size=size,pre_resume=pre,fullstate=full,
            health=health,information=dict(batches=info['batches'],probes=info['probes'],
                probe_summary=probe_label_summaries(info['probes'])),state=reduce_state_audit(state),
            training=dict(status=status,trace=trace),planning_signature=planning_signature(planning),
            source_artifacts={k:artifact(p) for k,p in paths.items()}))
        trajectories.append(dict(id=name,task=rec['task'],seed=rec['training_seed'],arm=rec['arm'],
            steps=[600,2000],success_rates=[old['success_rate'],rate],
            outcomes=[old['episode_successes'],outcomes],
            state_conditions=[old['state']['conditions'],state['conditions']],
            chart_certificates=[old['state']['carrier_chart_certificate'],state['carrier_chart_certificate']],
            **paired_outcomes(outcomes,old['episode_successes'],bootstrap),
            scope='More training plus the common native resume boundaries; not a one-component causal effect.'))
    baselines,baseline_sources=load_baselines(baseline_root,task_references)
    native_evaluator='as_jepa_tools/benchmark_evaluate.py'
    for source in baseline_sources.values():
        assert digest(Path(source['source']['source'])/native_evaluator)==digest(Path(src['source'])/native_evaluator)
    lookup={(r['task'],r['seed'],r['arm']):r for r in rows}
    controls={(r['task'],r['seed'],r['method']):r for r in baselines}
    comparisons=[]; summaries=[]; family_checks=[]
    for task in ('tworoom','cube'):
        pairs=[('unconstrained','identity'),('bounded','identity'),('bounded','unconstrained')]
        pairs += [(arm,method) for arm in ARMS for method in CONTROLS]
        for candidate,reference in pairs:
            paired=[]
            for seed in (1,2):
                a=lookup[task,seed,candidate]
                b=lookup[task,seed,reference] if reference in ARMS else controls[task,seed,reference]
                kind='within_carrier' if reference in ARMS else 'historical_training_control'
                if kind=='historical_training_control':
                    for key in ('data','dataset_kind','seed','batch_size','workers','max_steps'):
                        assert a['config'][key]==b['config'][key], (task,seed,key)
                    for key in ('mode','sigma0','window','renderer_width','renderer_depth','renderer_heads','forecast_seed','action_dim'):
                        assert a['config']['model'][key]==b['config']['model'][key], key
                    if reference=='temporal-forecast':
                        assert all(a['config']['model'][k]==v for k,v in b['config']['model'].items())
                item=dict(task=task,seed=seed,candidate=candidate,reference=reference,kind=kind,
                    **paired_outcomes(a['episode_successes'],b['episode_successes'],bootstrap))
                paired.append(item);comparisons.append(item)
            summaries.append(dict(task=task,candidate=candidate,reference=reference,kind=kind,
                                  **paired_summary(paired,bootstrap)))
        for seed in (1,2):
            group=[lookup[task,seed,a] for a in ARMS]
            assert len({r['model_size']['total'] for r in group})==1
            assert len({r['config']['carrier_calibration']['sha256'] for r in group})==1
            family_checks.append(dict(task=task,seed=seed,total_parameters=group[0]['model_size']['total'],
                trainable_parameters={r['arm']:r['model_size']['trainable'] for r in group},
                calibration=group[0]['config']['carrier_calibration']))
    return dict(root=str(root),source_commit=src['source_commit'],protocol=protocol,rows=rows,
        completed_endpoint_count=len(exits),completed_endpoints=exits,source_manifest=source_manifest,
        source_compatibility=compatibility,initial600_results=artifact(initial_results),
        baselines=baselines,baseline_sources=baseline_sources,within_family_checks=family_checks,
        trajectories=trajectories,comparisons=comparisons,comparison_summaries=summaries,
        collector_sha256=digest(__file__),bootstrap=dict(repetitions=bootstrap,seed=1042),
        analysis_helpers={name:artifact(path) for name,path in {
            'paired_and_artifact_helpers':paired_outcomes.__code__.co_filename,
            'allocation_and_semantics':assignments.__code__.co_filename}.items()},
        limitations=['Both seeds retained; case CIs condition on these seeds and ignore training-seed uncertainty.',
            'AS-JEPA original/lower/TF controls have matched data, update counts and native evaluator, but '
            'Carrier adds8320 total chart parameters, freezes encoder/pool, and uses128 train-only calibration frames. '
            'Optimizer-state size/trainable capacity/compute are not equal; no single-component attribution.',
            'Common1200 resume boundary is additional to the historical2000 controls.',
            'Native chart-coordinate MSE is not directly comparable across learned charts. '
            'Use common-coordinate inversion only when its certificate and numerical residual support it.',
            'Training uses high/TF32 whereas evaluation uses highest; previously measured shift remains.',
            'These are AS-JEPA training controls, not matched-budget native LeWM replications; no published-score transfer.',
            'Exploratory50-case intervals can be degenerate and are not multiple-comparison adjusted.'])


def write_report(result,path):
    lookup={(r['task'],r['seed'],r['arm']):r for r in result['rows']}
    old={r['id']:r for r in result['trajectories']}
    lines=['# Carrier2000 complete review','',f"All96 endpoints exit0; source `{result['source_commit']}`. "
        'Twelve continuations, both tasks and both seeds retained. Full native1200/2000 archives and frozen-state audits passed.','',
        '| Task | Arm | Seed1:600→2000 | Seed2:600→2000 | Trainable / total |',
        '|---|---|---:|---:|---:|']
    for task in ('tworoom','cube'):
        for arm in ARMS:
            a,b=[lookup[task,s,arm] for s in (1,2)]; av=old[a['id']]['success_rates'];bv=old[b['id']]['success_rates']
            lines.append(f"| {task} | {arm} | {av[0]:g}→{av[1]:g}% | {bv[0]:g}→{bv[1]:g}% | {a['model_size']['trainable']:,}/{a['model_size']['total']:,} |")
    lines+=['','Historical2000 AS-JEPA controls share the exact50 ordered cases, planner, data, seeds and update horizon. '
             'They differ in frozen parameters, chart capacity and calibration; improvements are whole-design comparisons.','',
             '| Task | Historical method | Seed1 | Seed2 |','|---|---|---:|---:|']
    for task in ('tworoom','cube'):
        for method in CONTROLS:
            r=sorted([x for x in result['baselines'] if x['task']==task and x['method']==method],key=lambda x:x['seed'])
            lines.append(f"| {task} | {method} | {r[0]['success_rate']:g}% | {r[1]['success_rate']:g}% |")
    lines+=['','| Task | Comparison | Seed1/2 difference | Mean [shared-case95% interval] |',
            '|---|---|---:|---:|']
    for r in result['comparison_summaries']:
        if r['kind']!='within_carrier' and r['reference']!='temporal-forecast':
            continue
        a,b=r['differences_by_seed_pp'];lo,hi=r['shared_case_bootstrap_ci95_pp']
        lines.append(f"| {r['task']} | {r['candidate']}−{r['reference']} | {a:+g}/{b:+g}pp | {r['mean_difference_pp']:+g} [{lo:+g},{hi:+g}]pp |")
    lines+=['','All per-case outcomes, paired comparisons to original/lower controls, health/state/probe results, '
            'chart certificates, source hashes and full-state audit records are in `results.json`. '
            'Native latent MSE across different charts requires a common-coordinate check. '
            'Planning gains and factual dynamics must be assessed separately.','',*['- '+s for s in result['limitations']],'']
    Path(path).write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(B/'runs/carrier_learn2000_20261031'))
    parser.add_argument('--initial-results',default=str(B/'reports/ideas-validate600-20261031/results.json'))
    parser.add_argument('--baseline-root',default=str(B/'runs/hardtasks_learn2000_20261030'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--bootstrap',type=int,default=10000)
    args=parser.parse_args(); assert args.bootstrap>=100
    result=collect(args.root,args.initial_results,args.baseline_root,args.bootstrap)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_report(result,out/'RESULTS.md')
    print(json.dumps(dict(completed=result['completed_endpoint_count'],rows=len(result['rows']),
        scores=[{k:r[k] for k in ('task','seed','arm','success_rate')} for r in result['rows']]),indent=2))


if __name__=='__main__':
    main()
