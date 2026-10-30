"""CPU-only complete fresh2000 review; no model loading, updates or launches."""
import json
from pathlib import Path
import sys
import numpy as np

REPO=next(p for p in Path(__file__).resolve().parents if (p/'experiments/ideas_validate_20261031/collect.py').is_file())
sys.path.insert(0,str(REPO))
from experiments.ideas_validate_20261031.collect import read,digest,canonical,paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import artifact,paired_summary,planning_signature,successes

B=Path('/efs/prescottsun/SPT-JEPA')
ROOT=B/'runs/followup_fresh2000_20261031'
OUT=B/'reports/followup-fresh2000-20261031'


def collect():
    plan,protocol=read(ROOT/'plan.json'),read(ROOT/'protocol.json')
    assert len(plan['jobs'])==protocol['jobs']==12 and protocol['model_updates']==0
    parent_refs={r['id']:r for r in protocol['references']}
    old_fresh_root=B/'runs/carrier_fresh2000_20261031'
    assert digest(old_fresh_root/'plan.json')==protocol['parent_fresh_plan_sha256']
    old_plan=read(old_fresh_root/'plan.json');old2000_path=B/'reports/carrier-learn2000-20261031/results.json'
    old2000=read(old2000_path);oldrows={r['id']:r for r in old2000['rows']+old2000['baselines']}
    local_parent_path=B/'reports/local-isotropic2000-20261031/results.json'
    native_parent_path=B/'reports/native-lewm2000-20261031/results.json'
    parents=dict(local=read(local_parent_path),native=read(native_parent_path))
    signatures={};rows=[];sources={};exits=[]
    for job in old_plan['jobs']:
        assert read(old_fresh_root/'exits'/(job['id']+'.json'))['returncode']==0
        fp=Path(job['output'])/'metrics.json';fresh=read(fp)
        prior=oldrows[job['id']];arts=prior.get('source_artifacts',prior.get('artifacts'))
        pp=Path(arts['planning']['path']);assert digest(pp)==arts['planning']['sha256'];primary=read(pp)
        arm='F-'+job['arm'] if job['family']=='carrier' else 'TF'
        values={}
        for cases,x in [('primary',primary),('fresh',fresh)]:
            outcomes,rate=successes(x);sig=planning_signature(x)
            key=job['task'],cases
            if key in signatures:assert canonical(signatures[key])==canonical(sig)
            else:signatures[key]=sig
            values[cases]=dict(success_rate=rate,episode_successes=outcomes)
        rows.append(dict(task=job['task'],seed=job['training_seed'],arm=arm,role='existing2000_control',
            values=values,model_parameters=prior['model_size']['total'],
            artifacts=dict(primary=artifact(pp),fresh=artifact(fp))))
    for task in ['tworoom','cube']:
        spec=protocol['manifests'][task];assert digest(spec['path'])==spec['sha256']
        m=read(spec['path']);assert m['campaign_split']=='validation' and len(m['pairs'])==50
        assert m['pairs']==signatures[task,'fresh']['planning_pairs']
        assert spec['sha256']==signatures[task,'fresh']['planning_manifest_sha256']
        assert m['fresh_selection']['exact_pair_overlap_count']==0
    for job in plan['jobs']:
        ex=read(ROOT/'exits'/(job['id']+'.json'));assert ex['returncode']==0 and ex['id']==job['id'];exits.append(ex)
        ref=parent_refs[job['id']];pp=Path(ref['parent_evaluation'])/'metrics.json';fp=Path(job['output'])/'metrics.json'
        primary,fresh=read(pp),read(fp);checkpoint=Path(ref['checkpoint'])
        assert fresh['checkpoint']==primary['checkpoint']==str(checkpoint)
        cmd=job['command'];assert cmd[cmd.index('--checkpoint')+1]==str(checkpoint)
        assert digest(cmd[cmd.index('--manifest')+1])==protocol['manifests'][job['task']]['sha256']
        parent_root=Path(ref['parent_evaluation']).parent
        parent_plan=read(parent_root/'plan.json')
        parent_job=next(j for j in parent_plan['jobs'] if Path(j['output'])==Path(ref['parent_evaluation']))
        assert canonical(parent_job['source'])==canonical(job['source'])
        assert read(parent_root/'exits'/(parent_job['id']+'.json'))['returncode']==0
        source=job['source'];commit=source['source_commit']
        if commit not in sources:
            manifest=read(source['manifest']);assert manifest['source_commit']==commit
            for name,sha in manifest['files'].items():assert digest(Path(source['source'])/name)==sha
            sources[commit]=dict(source=source,manifest=artifact(source['manifest']),all_manifest_files_verified=True)
        weights_sha=digest(checkpoint/'weights.pt');export_config=read(checkpoint/'config.json')
        arm='native-BN' if job['family']=='native-lewm' else job['arm']
        if arm=='native-BN':
            calibration_path=checkpoint.parent/'calibration.json';cal=read(calibration_path)
            assert cal['step']==2000 and cal['training_only'] and cal['all_other_state_bitexact']
            assert weights_sha==cal['export_weights_sha256'] and cal['export']==str(checkpoint)
            assert digest(cal['manifest'])==cal['manifest_sha256']
            assert read(cal['manifest'])['split']=='train'
            original=Path(cal['run'])
            assert digest(original/'model-step0002000/config.json')==cal['source_config_sha256']==digest(checkpoint/'config.json')
            assert digest(original/'model-step0002000/weights.pt')==cal['original_weights_sha256']
            assert read(original/'stage-complete.json')['stop_step']==2000
            prior=next(r for r in parents['native']['rows'] if r['task']==job['task'] and r['seed']==job['training_seed'])
            assert prior['fullstate_passed'] and prior['direct_buffer_only_check']
            assert abs(prior['calibrated_success']-primary['metrics']['success_rate'])<1e-10
            assert prior['calibrated_outcomes']==primary['metrics']['episode_successes']
            integrity=dict(calibration=artifact(calibration_path),train_only_BN=True,
                changed_buffers=cal['changed_buffers'],all_other_state_bitexact=True)
        else:
            original=checkpoint.parent;cfg=read(original/'config.json')
            fullpath=parent_root/('fullstate2000-'+original.name)/'fullstate.json';full=read(fullpath)
            assert full['step']==2000 and full['passed'] and all(full['checks'].values())
            assert weights_sha==full['portable_sha256']
            assert digest(original/'config.json')==full['config_sha256']
            assert all(export_config[k]==v for k,v in cfg['model'].items())
            assert read(original/'stage-complete.json')['stop_step']==2000
            prior=next(r for r in parents['local']['rows'] if r['id']==original.name)
            assert abs(prior['success_rate']-primary['metrics']['success_rate'])<1e-10
            assert prior['episode_successes']==primary['metrics']['episode_successes']
            integrity=dict(fullstate=artifact(fullpath),passed=True,portable_equals_fullstate=True)
        values={}
        for cases,x in [('primary',primary),('fresh',fresh)]:
            outcomes,rate=successes(x)
            assert canonical(planning_signature(x))==canonical(signatures[job['task'],cases])
            values[cases]=dict(success_rate=rate,episode_successes=outcomes)
        rows.append(dict(task=job['task'],seed=job['training_seed'],arm=arm,role='fresh_confirmation',
            values=values,checkpoint=str(checkpoint),weights_sha256=weights_sha,source_commit=commit,
            model_parameters=fresh['checkpoint_audit']['parameters'],integrity=integrity,
            artifacts=dict(primary=artifact(pp),fresh=artifact(fp),export_config=artifact(checkpoint/'config.json'))))
    assert len(rows)==28
    lookup={(r['task'],r['seed'],r['arm']):r for r in rows}
    pairs=[('local',b) for b in ['dense','native-BN','F-identity','F-unconstrained','F-bounded','TF']]
    pairs += [('native-BN',b) for b in ['F-identity','F-unconstrained','F-bounded','TF']]
    comparisons=[];summaries=[]
    for task in ['tworoom','cube']:
        for cases in ['primary','fresh']:
            for a,b in pairs:
                group=[]
                for seed in [1,2]:
                    aa,bb=lookup[task,seed,a]['values'][cases],lookup[task,seed,b]['values'][cases]
                    r=dict(task=task,case_set=cases,seed=seed,candidate=a,reference=b,
                        **paired_outcomes(aa['episode_successes'],bb['episode_successes'],10000))
                    comparisons.append(r);group.append(r)
                summaries.append(dict(task=task,case_set=cases,candidate=a,reference=b,**paired_summary(group,10000)))
    return dict(root=str(ROOT),completed_endpoint_count=12,exits=exits,rows=rows,
        sources=sources,protocol=protocol,planning_signatures={f'{t}-{c}':s for (t,c),s in signatures.items()},
        comparisons=comparisons,comparison_summaries=summaries,
        parent_reports={k:artifact(p) for k,p in dict(local=local_parent_path,native=native_parent_path,old_F_TF=old2000_path).items()},
        collector=artifact(__file__),bootstrap=dict(repetitions=10000,seed=1042,unit='50 paired cases jointly resampled over two fixed seeds'),
        scope='Both seeds and both validation case sets retained. Conditional case CIs exclude seed-population uncertainty and multiplicity correction. '
              'Fresh and primary are different cases, so their success changes are descriptive, not paired case effects. '
              'Native-BN is the existing training-only posthoc BN variant, not literal native inference. No calibration or model updates in this evaluation.')


def compact(result,path):
    return dict(full_results=artifact(path),completed_endpoint_count=12,
        rows=[dict(task=r['task'],seed=r['seed'],arm=r['arm'],model_parameters=r['model_parameters'],
                   primary=r['values']['primary']['success_rate'],fresh=r['values']['fresh']['success_rate']) for r in result['rows']],
        comparison_summaries=result['comparison_summaries'],source_commits=list(result['sources']),
        parent_reports=result['parent_reports'],scope=result['scope'],bootstrap=result['bootstrap'])


def report(s):
    lines=['# 2000步fresh确认：Cube local优势尚未稳定复现','',
        '全部12项fresh评测exit0，使用原先固定的50 fresh病例及未更新的2000步checkpoint。Cube local从primary24/20%变为fresh16/10%；fresh与dense、训练集BN校准native均无胜出证据。此前primary上的候选优势应保留为探索结果，不能外推为稳定提升。','',
        '| 任务 | 方法 | primary seed1/2 | fresh seed1/2 |','|---|---|---:|---:|']
    for task in ['tworoom','cube']:
        for arm in ['local','dense','native-BN','F-identity','F-unconstrained','F-bounded','TF']:
            rows=sorted([r for r in s['rows'] if r['task']==task and r['arm']==arm],key=lambda r:r['seed'])
            vals=['/'.join(f"{r[k]:g}" for r in rows) for k in ['primary','fresh']]
            lines.append(f"| {task} | {arm} | {vals[0]} | {vals[1]} |")
    lines+=['','数值为成功率%。所有模型均2000更新；旧F/TF行复用已完成同fresh评测，不是重新运行。','',
        '| 任务 | fresh差值 | seed1/2 Δpp | 均值 [配对case95%区间] |','|---|---|---:|---:|']
    for x in s['comparison_summaries']:
        if x['case_set']!='fresh' or x['candidate']!='local' or x['reference'] not in ['dense','native-BN','F-identity','TF']:continue
        lo,hi=x['shared_case_bootstrap_ci95_pp'];a,b=x['differences_by_seed_pp']
        lines.append(f"| {x['task']} | local−{x['reference']} | {a:+g}/{b:+g} | {x['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines+=['','区间按50配对cases重采样10000次，同一case索引同时用于两个固定训练seed；未涵盖训练seed总体不确定性、多重比较。primary/fresh是不同病例，不能把两集合的成功率差当作配对模型改进。fresh来自相同validation pool，与primary精确pair重合0、episode重合TwoRoom34/Cube1，不是独立test。','',
        '完整核验通过：原fresh manifest哈希/有序病例、CEM配置、horizon/budget、strict判据和train-episode动作归一化，与父primary及旧F/TF各自病例集一致；checkpoint路径与父primary相同，权重SHA对local完整归档或native校准导出记录匹配，冻结源全部manifest文件通过哈希核验。CEM seed仍42，fresh manifest的policy/selection seed为2026103101，均未变化。','',
        'native-BN保留预先固定的train-only precise-BN处理：只改4个running_mean/variance buffers，其他状态bitexact；它不是literal native推理。native约18.04M参数/192维状态，local/dense约55.9M/1024维，旧F/TF约51.1M/64维，容量、冻结参数、校准暴露和计算不匹配。不能把跨家族差值单独归因于局部结构或某个loss。','',
        '本次只确认现成checkpoint，不新增训练、校准、GPU assay或自动续跑。完整逐case结果、来源与所有对比保留EFS，summary.json含其SHA；本报告不改写历史primary报告。','']
    return '\n'.join(lines)


if __name__=='__main__':
    r=collect();OUT.mkdir(parents=True,exist_ok=True);path=OUT/'results.json'
    path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');s=compact(r,path)
    (OUT/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(s));print(report(s))
