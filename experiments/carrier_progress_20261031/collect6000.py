"""Complete-only CPU collector for the reviewed8 Carrier +4 TF6000 campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p/'experiments/ideas_validate_20261031/collect.py').is_file())
sys.path.insert(0, str(REPO))
from experiments.ideas_validate_20261031.collect import canonical, digest, paired_outcomes, read
from experiments.carrier_progress_20261031.collect_learning import artifact, paired_summary, planning_signature, successes, verify_manifest
from experiments.carrier_progress_20261031.prepare_learning import semantic
from experiments.carrier_progress_20261031.prepare_next import allocations

B = Path('/efs/prescottsun/SPT-JEPA')


def compact_state(state):
    if state is None:
        return None
    c = state['conditions']
    return dict(step=state['step'], clean_mse=c['clean_rollout']['mse'],
        initial_carry_mse=c['initial_carry']['mse'],
        clean_over_initial_carry=c['clean_rollout']['mse']/c['initial_carry']['mse'],
        teacher_mse=c['teacher_forced_prefix']['mse'],
        shifted_observed_carry_mse=c['shifted_observed_carry']['mse'],
        teacher_over_shifted_carry=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
        other_initial_mse=c['other_initial']['mse'], other_actions_mse=c['other_episode_actions']['mse'],
        other_initial_effect=c['other_initial']['intervention_effect_mse'],
        other_actions_effect=c['other_episode_actions']['intervention_effect_mse'],
        initial_temporal_variation=state['actual_temporal_variation']['initial_mean'],
        state_std=state['observed_state_std_mean'],
        chart_certificate=state['carrier_chart_certificate'])


def trace_summary(trace):
    unique = {r['step']:r for r in trace if 'loss' in r}
    result = dict(first_step=min(unique), last_step=max(unique), recorded_unique_steps=len(unique))
    for key, lo, hi in [('early',2500,3000),('recent',4500,5000),('late',5500,6000)]:
        rows=[r for step,r in unique.items() if lo<=step<=hi]
        assert len(rows)>=2
        result[key]=dict(step_range=[min(r['step'] for r in rows),max(r['step'] for r in rows)],
            records=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in
            ['loss','fm','rate','entry','surprise','feature_std']})
    result['late_vs_early_change_percent']={k:100*(result['late'][k]/result['early'][k]-1)
        for k in ['loss','fm','rate'] if result['early'][k]!=0}
    result['late_vs_recent_change_percent']={k:100*(result['late'][k]/result['recent'][k]-1)
        for k in ['loss','fm','rate'] if result['recent'][k]!=0}
    return result


def collect(base, bootstrap):
    root=base/'runs/carrier_learn6000_20261031'
    plan,protocol,source,compat=[read(root/f) for f in
        ['plan.json','protocol.json','source-record.json','source-compatibility.json']]
    records=allocations(); assert protocol['allocations']==records
    assert protocol['initial_step']==2000 and protocol['stop_step']==6000
    assert protocol['runs']==12 and protocol['jobs']==len(plan['jobs'])==68
    jobs={j['id']:j for j in plan['jobs']}
    expected={f'{phase}-{r["id"]}' for r in records for phase in
        ['learn6000','fullstate6000','health','planning','fresh-planning']}
    expected|={f'transition-{r["id"]}' for r in records if r['family']=='carrier'}
    assert set(jobs)==expected
    missing=[j for j in jobs if not (root/'exits'/(j+'.json')).exists()]
    if missing:
        raise RuntimeError('Campaign incomplete; no result collection: '+', '.join(missing))
    exits=[]
    for job in jobs.values():
        x=read(root/'exits'/(job['id']+'.json'))
        assert x['id']==job['id'] and x['returncode']==0
        assert canonical(job['source'])==canonical(source)
        exits.append(dict(x,phase=job['phase']))
    assert source['source_commit']==protocol['source_commit']==compat['new_source']['source_commit']
    source_artifact=verify_manifest(source)
    assert source_artifact['sha256']==compat['new_manifest_sha256']
    assert digest(source['runtime_manifest'])==compat['shared_runtime_manifest_sha256']
    for name,sha in compat['identical_files'].items():
        assert digest(Path(source['source'])/name)==digest(Path(compat['old_source']['source'])/name)==sha
    assert digest(protocol['tf_source_review']['path'])==protocol['tf_source_review']['sha256']
    old_path=base/'reports/carrier-learn2000-20261031/results.json'
    fresh_old_plan_path=base/'runs/carrier_fresh2000_20261031/plan.json'
    diagnostic_path=base/'reports/carrier-diagnostics-20261031/results.json'
    old,old_fresh_plan,diagnostic=read(old_path),read(fresh_old_plan_path),read(diagnostic_path)
    old_rows={r['id']:r for r in old['rows']+old['baselines']}
    old_fresh_jobs={j['id']:j for j in old_fresh_plan['jobs']}
    parent_specs={r['id']:r for r in protocol['parents']}
    rows=[]
    for rec in records:
        name=rec['id'];run=root/name;carrier=rec['family']=='carrier'
        parent=Path(parent_specs[name]['parent']);old_name=parent.name;prior=old_rows[old_name]
        cfg,old_cfg=read(run/'config.json'),read(parent/'config.json')
        assert semantic(cfg)==semantic(old_cfg)
        config_hash=digest(run/'config.json');assert config_hash==jobs['learn6000-'+name]['config_sha256']
        status,stage,size,pre=[read(run/f) for f in ['status.json','stage-complete.json','model_size.json','pre-resume-fullstate.json']]
        assert status['state']=='finished' and status['step']==stage['stop_step']==6000
        assert stage['initial_step']==2000 and stage['stage_complete']
        assert pre['step']==2000 and pre['passed'] and pre['copy_byte_identical'] and all(pre['checks'].values())
        assert pre['config_sha256']==digest(parent/'config.json')
        assert Path(pre['preserved_parent_checkpoint'])==parent/'last.ckpt'
        full_path=root/('fullstate6000-'+name)/'fullstate.json';full=read(full_path)
        assert full['step']==6000 and full['passed'] and all(full['checks'].values())
        assert full['config_sha256']==config_hash and full['source_commit']==source['source_commit']
        assert full['campaign_samples']==6000*cfg['batch_size']
        assert full['native_stage_complete']==stage and full['frozen_tensors']['equal']
        assert Path(full['archived_checkpoint'])==run/'checkpoint-step0006000.ckpt'
        assert Path(full['archived_checkpoint']).is_file()
        assert (run/'model-step0004000'/'weights.pt').is_file()
        portable=run/'model-step0006000'/'weights.pt'
        assert digest(portable)==full['portable_sha256']
        exported_config_path=portable.parent/'config.json'
        exported_config=read(exported_config_path)
        assert all(exported_config[k]==v for k,v in cfg['model'].items())
        health_path=root/('health-'+name)/'health.json';health=read(health_path)
        assert health['audit']['step']==6000 and all(v for v in health['audit'].values() if isinstance(v,bool))
        assert size['total']==health['parameters']==prior['model_size']['total']
        if carrier:
            assert [{k:r[k] for k in ['episode','start']} for r in health['cases']]==[
                {k:r[k] for k in ['episode','start']} for r in prior['health']['cases']]
        else:
            old_health=read(prior['artifacts']['health']['path'])
            assert [{k:r[k] for k in ['episode','start']} for r in health['cases']]==[
                {k:r[k] for k in ['episode','start']} for r in old_health['cases']]
        state=None;state_art=None
        if carrier:
            state_path=root/('transition-'+name)/'state-audit.json';state=read(state_path)
            windows_path=root/('transition-'+name)/'windows.json'
            assert state['step']==6000 and state['config_sha256']==config_hash
            assert state['weights_sha256']==full['portable_sha256']
            assert state['manifest_sha256']==digest(windows_path)==prior['state']['manifest_sha256']
            assert state['cases']==64
            assert read(windows_path)==read(prior['source_artifacts']['windows']['path'])
            state_art=artifact(state_path)
        planning={};parents_planning={}
        for case_set,phase in [('primary','planning'),('fresh','fresh-planning')]:
            p=root/(phase+'-'+name)/'metrics.json';x=read(p);job=jobs[phase+'-'+name]
            manifest=Path(job['command'][job['command'].index('--manifest')+1])
            assert digest(manifest)==job['manifest_sha256']==x['manifest_sha256']
            assert read(manifest)['pairs']==x['pairs']
            assert Path(x['checkpoint'])==portable.parent
            assert x['checkpoint_audit']['parameters']==size['total']
            out,rate=successes(x)
            if case_set=='primary':
                desc=prior.get('source_artifacts',prior.get('artifacts'))['planning']
                old_p=Path(desc['path']);assert digest(old_p)==desc['sha256']
            else:
                old_job=old_fresh_jobs[old_name]
                old_p=Path(old_job['output'])/'metrics.json'
                assert read(base/'runs/carrier_fresh2000_20261031/exits'/(old_name+'.json'))['returncode']==0
            old_eval=read(old_p);old_out,old_rate=successes(old_eval)
            assert canonical(planning_signature(x))==canonical(planning_signature(old_eval))
            assert Path(old_eval['checkpoint'])==parent/'model-step0002000'
            planning[case_set]=dict(success_rate=rate,episode_successes=out,signature=planning_signature(x),source=artifact(p))
            parents_planning[case_set]=dict(success_rate=old_rate,episode_successes=old_out,source=artifact(old_p))
        trace=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
        train_summary=trace_summary(trace)
        rows.append(dict(**rec,seed=rec['training_seed'],step=6000,config=cfg,model_size=size,
            pre_resume=pre,fullstate=full,health=health,planning=planning,parent2000=parents_planning,
            state=state,parent2000_state=prior.get('state'),state_summary=compact_state(state),
            parent2000_state_summary=compact_state(prior.get('state')),
            training=dict(status=status,trace=trace,summary=train_summary),
            artifacts=dict(config=artifact(run/'config.json'),fullstate=artifact(full_path),health=artifact(health_path),
                exported_config=artifact(exported_config_path),
                state=state_art,portable_weights=dict(path=str(portable),sha256=full['portable_sha256']),
                archive=dict(path=full['archived_checkpoint'],sha256=full['checkpoint_sha256'],
                    scope='Archival endpoint verified byte identity; collector checks existence without reloading full state.'))))
    lookup={(r['task'],r['seed'],r['arm']):r for r in rows}
    comparisons=[];summaries=[]
    for task in ['tworoom','cube']:
        arms=['identity','unconstrained','bounded'] if task=='tworoom' else ['identity']
        for case in ['primary','fresh']:
            pairs=[(a,'temporal-forecast') for a in arms]
            if task=='tworoom':pairs += [('unconstrained','identity'),('bounded','identity'),('bounded','unconstrained')]
            for a,b in pairs:
                group=[]
                for seed in [1,2]:
                    aa,bb=lookup[task,seed,a],lookup[task,seed,b]
                    assert canonical(aa['planning'][case]['signature'])==canonical(bb['planning'][case]['signature'])
                    for key in ['data','dataset_kind','batch_size','seed','max_steps']:
                        assert aa['config'][key]==bb['config'][key]
                    item=dict(task=task,case_set=case,seed=seed,candidate=a,reference=b,kind='same6000',
                        **paired_outcomes(aa['planning'][case]['episode_successes'],bb['planning'][case]['episode_successes'],bootstrap))
                    group.append(item);comparisons.append(item)
                summaries.append(dict(task=task,case_set=case,candidate=a,reference=b,kind='same6000',**paired_summary(group,bootstrap)))
            for arm in arms+['temporal-forecast']:
                group=[]
                for seed in [1,2]:
                    r=lookup[task,seed,arm]
                    item=dict(task=task,case_set=case,seed=seed,candidate=arm,reference=arm,kind='6000_minus2000',
                        **paired_outcomes(r['planning'][case]['episode_successes'],r['parent2000'][case]['episode_successes'],bootstrap))
                    group.append(item);comparisons.append(item)
                summaries.append(dict(task=task,case_set=case,candidate=arm,reference=arm,kind='6000_minus2000',**paired_summary(group,bootstrap)))
    released=[]
    for task in ['tworoom','cube']:
        p=base/f'runs/hardtasks_gate600_20261030/reference-{task}-released-lewm-strict-validation/metrics.json'
        for case,path in [('primary',p),('fresh',Path(next(r['output'] for r in diagnostic['rows'] if r['task']==task and r['case_set']=='fresh' and r['family']=='released-lewm'))/'metrics.json')]:
            x=read(path);reference=lookup[task,1,'identity']['planning'][case]['signature']
            sig=planning_signature(x)
            assert {k:v for k,v in sig.items() if k!='normalization'}=={k:v for k,v in reference.items() if k!='normalization'}
            released.append(dict(task=task,case_set=case,success_rate=x['metrics']['success_rate'],source=artifact(path),
                scope='Unmatched released checkpoint: training exposure/capacity/normalization differ; not a matched-budget replication.'))
    return dict(root=str(root),source_commit=source['source_commit'],protocol=protocol,
        completed_endpoint_count=len(exits),completed_endpoints=exits,source_manifest=source_artifact,
        source_compatibility=compat,rows=rows,comparisons=comparisons,comparison_summaries=summaries,
        released_reference=released,bootstrap=dict(repetitions=bootstrap,seed=1042),
        sources=dict(parent2000=artifact(old_path),parent_fresh2000_plan=artifact(fresh_old_plan_path),
                     diagnostics600=artifact(diagnostic_path)),collector=artifact(__file__),
        scope='Both seeds retained. Fixed50 primary and50 fresh validation cases; reused validation is not independent test. '
              'Paired case intervals condition on two fixed training seeds, without seed-level uncertainty or multiplicity correction. '
              'F/TF share6000updates/data/planner; capacity, frozen parameters, calibration and training compute differ. '
              'No new physical-label training. No TF transition audit was scheduled.')


def small(result, result_path):
    rows=[]
    for r in result['rows']:
        rows.append({**{k:r[k] for k in ['id','task','seed','arm','family','model_size','state_summary','parent2000_state_summary']},
            'success6000':{c:v['success_rate'] for c,v in r['planning'].items()},
            'success2000':{c:v['success_rate'] for c,v in r['parent2000'].items()},
            'training':r['training']['summary'],
            'checks':dict(pre_resume=r['pre_resume']['passed'],fullstate=r['fullstate']['passed'],
                fullstate_checks=r['fullstate']['checks'],health=r['health']['audit'])})
    return dict(full_results=artifact(result_path),source_commit=result['source_commit'],
        completed_endpoint_count=result['completed_endpoint_count'],rows=rows,
        comparison_summaries=result['comparison_summaries'],released_reference=result['released_reference'],
        sources=result['sources'],scope=result['scope'])


def document(s):
    lines=['# Carrier/TF6000 complete review','',f"All{s['completed_endpoint_count']} endpoints exit0; source `{s['source_commit']}`. All12 resume, full-state, strict-load and native evaluator contracts passed.", '',
        '| Task | Arm | Primary2000→6000 seed1 /seed2 | Fresh2000→6000 seed1 /seed2 |','|---|---|---:|---:|']
    for task in ['tworoom','cube']:
        for arm in (['identity','unconstrained','bounded','temporal-forecast'] if task=='tworoom' else ['identity','temporal-forecast']):
            rr=sorted([r for r in s['rows'] if r['task']==task and r['arm']==arm],key=lambda r:r['seed'])
            v=[' / '.join(f"{r['success2000'][c]:g}→{r['success6000'][c]:g}%" for r in rr) for c in ['primary','fresh']]
            lines.append(f"| {task} | {arm} | {v[0]} | {v[1]} |")
    lines+=['','| Task/cases |6000 comparison| Seed1/2 Δpp | Mean [paired-case95% interval] |','|---|---|---:|---:|']
    for x in s['comparison_summaries']:
        if x['kind']!='same6000':continue
        lo,hi=x['shared_case_bootstrap_ci95_pp'];ds=x['differences_by_seed_pp']
        lines.append(f"| {x['task']}/{x['case_set']} | {x['candidate']}−{x['reference']} | {ds[0]:+g}/{ds[1]:+g} | {x['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines+=['','| Task/arm/seed | Clean/copy2000→6000 | Teacher/local-copy2000→6000 | Residual global bound6000 | late/recent loss change |','|---|---:|---:|---:|---:|']
    for r in s['rows']:
        new,old=r['state_summary'],r['parent2000_state_summary']
        loss=r['training']['late_vs_recent_change_percent']['loss']
        if new:
            a=f"{old['clean_over_initial_carry']:.4f}→{new['clean_over_initial_carry']:.4f}"
            b=f"{old['teacher_over_shifted_carry']:.4f}→{new['teacher_over_shifted_carry']:.4f}"
            lip=new['chart_certificate']['residual_lipschitz_upper']
            lines.append(f"| {r['task']}/{r['arm']}/{r['seed']} | {a} | {b} | {lip:.4f} | {loss:+.1f}% |")
        else:lines.append(f"| {r['task']}/{r['arm']}/{r['seed']} | not scheduled | not scheduled | — | {loss:+.1f}% |")
    lines+=['','Clean/copy compares causal forecast to initial-state persistence; teacher/local-copy has extra real past observations. '
        'Identity-carrier coordinates are fixed across checkpoints. Learned-chart normalized ratios still change with chart geometry; '
        'they do not by themselves establish common-coordinate dynamics improvement. Residual Lipschitz bound≥1 loses a sufficient '
        'global invertibility certificate; it does not establish noninvertibility.','',
        'Late/recent losses compare recorded training minibatches at steps5500–6000 vs4500–5000 (six recorded minibatches per interval). Decreasing loss is evidence against a '
        'clear training-loss plateau, not proof that planning will improve with more updates. No transition assay was scheduled for TF.','',
        'Released LeWM references: TwoRoom primary72/fresh68%, Cube primary30/fresh24%; training budget, exposure, capacity and normalization '
        'differ. Internal F/TF have equal updates/data/native planner but differ in frozen/trainable capacity,8320 chart parameters and '
        '128-frame train-only calibration. All parents and source snapshots remain preserved.','',s['scope'],'',
        '判断与下一步：TwoRoom的三种F在primary/fresh均继续改善，有依据再进行一个有上限的训练阶段；free/bound没有跨病例集一致胜者，仍需保留配对控制。'
        'Cube固定RP64载体的clean预测确实改善，但规划未一致提升，且全部四个seed×病例条件都低于同6000步TF。'
        '建议原Cube F保持6000，优先等待空间机制短gate；Cube TF两病例集均较2000平均+10pp，作为继续训练的对照有实际任务依据。'
        '这不是证明F已经收敛，而是继续优化当前目标的任务回报不足。TwoRoom TF仍失败，不能仅因renderer loss下降宣称世界模型持续改善。'
        '更长F与更短TF对比需明确预算差异，或保留必要的同预算TF控制；本报告不启动下一阶段。','',
        'Full per-case outcomes, audit records and traces remain in EFS `results.json`. Small `summary.json` records its SHA. '
        'No new experiment, GPU job, model modification, baseline BN calibration or local1024 gate is part of this collector.','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=B);p.add_argument('--bootstrap',type=int,default=10000)
    a=p.parse_args();result=collect(a.base,a.bootstrap);out=a.base/'reports/carrier6000-20261031';out.mkdir(parents=True,exist_ok=True)
    path=out/'results.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary=small(result,path);(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (out/'RESULTS.md').write_text(document(summary))
    print(json.dumps({'output':str(out),'rows':[{k:r[k] for k in ['task','seed','arm','success6000']} for r in summary['rows']]},indent=2))
