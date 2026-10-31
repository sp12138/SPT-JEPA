"""Completed-only CPU review of the bounded dynamic-readout2000 continuation."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from collect600 import artifact, digest, paired, read, signature
from head_mechanism import NAMES, PREFIX, measure

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B/'runs/dynamic_readout2000_20261031'
PARENT = B/'runs/dynamic_readout600_20261031'
OLD = B/'runs/local_isotropic2000_20261031'
FRESH = B/'runs/followup_fresh2000_20261031'


def check_source(record):
    manifest = read(record['manifest'])
    assert manifest['source_commit'] == record['source_commit']
    for name, expected in manifest['files'].items():
        assert digest(Path(record['source'])/name) == expected
    return dict(record=record, manifest=artifact(record['manifest']))


def null_components(observed, clean, state):
    y, p = observed[:, 1:].astype(np.float64), clean.astype(np.float64)
    zero, energy, alignment = np.square(y).mean((0, 2)), np.square(p).mean((0, 2)), (2*p*y).mean((0, 2))
    mse = np.square(p-y).mean((0, 2))
    carry = np.square(y-observed[:, :1]).mean((0, 2))
    assert np.max(np.abs(mse-(zero+energy-alignment))) < 1e-12
    for key, value in [('zero_prediction',zero), ('clean_rollout',mse), ('initial_carry',carry)]:
        assert np.isclose(value.mean(), state['conditions'][key]['mse'], rtol=1e-10, atol=1e-12)
    return dict(zero_mse=float(zero.mean()), clean_mse=float(mse.mean()), carry_mse=float(carry.mean()),
                prediction_energy=float(energy.mean()), twice_alignment=float(alignment.mean()),
                clean_over_zero=float(mse.mean()/zero.mean()), clean_over_copy=float(mse.mean()/carry.mean()),
                per_horizon=dict(zero_mse=zero.tolist(),clean_mse=mse.tolist(),copy_mse=carry.tolist(),
                                 clean_over_zero=(mse/zero).tolist(),prediction_energy=energy.tolist(),twice_alignment=alignment.tolist()))


def collect():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert (ROOT/'HOLD_DISPATCH').exists() and read(ROOT/'closure.json')['all72_actual_exit0']
    plan, protocol = read(ROOT/'plan.json'), read(ROOT/'protocol.json')
    assert len(plan['jobs']) == 72
    for job in plan['jobs']:
        assert read(ROOT/'exits'/(job['id']+'.json'))['returncode'] == 0
        assert job['source'] == protocol['model_source']
    assert protocol['model_source']['source_commit'] == 'd6d0a9f85127a8af3e65bd31b29c161c3584867c'
    assert protocol['helper_source']['source_commit'] == 'daad1e825d2e300065a4e5167d8fe0f64dd1645a'
    sources = [check_source(s) for s in [protocol['model_source'], protocol['helper_source'],
        read(B/'runs/dynamic_readout_recovery_source_20261031/source-record.json'), read(OLD/'source-record.json')]]
    prepost = read(ROOT/'prepost-review.json')
    assert prepost['passed'] and prepost['audits'] == 16
    parent_report = B/'reports/dynamic-readout600-20261031/results.json'
    parent_summary = B/'reports/dynamic-readout600-20261031/summary.json'
    assert digest(parent_summary) == protocol['parent_review_sha256']
    assert digest(parent_report) == read(parent_summary)['full_results']['sha256']
    for reference in protocol['references']:
        for field in ['primary_metrics','fresh_metrics','primary_manifest','fresh_manifest']:
            assert digest(reference[field]) == reference[field+'_sha256']
    parent_rows = {r['id']:r for r in read(parent_report)['rows']}
    rows, checks, reference_head = [], [], None
    for task in ['tworoom','cube']:
        for seed in [1,2]:
            old = OLD/f'{task}-spatial-local-seed{seed}'
            oldcfg = read(old/'config.json')
            oldstate = OLD/('transition-'+old.name)
            with np.load(oldstate/'state-audit.npz') as cache:
                refobs, refactions = cache['observed'].copy(), cache['actions'].copy()
            for arm in ['identity','static','state']:
                run = old if arm == 'identity' else ROOT/f'{task}-readout-{arm}-seed{seed}'
                root = OLD if arm == 'identity' else ROOT
                cfg = read(run/'config.json')
                expected = copy.deepcopy(oldcfg)
                expected['output'] = str(run)
                if arm != 'identity':
                    expected['model']['spatial_readout_mode'] = arm
                    for suffix in NAMES:
                        expected['audit_expectations']['gradients'][PREFIX+suffix] = 'active'
                assert cfg == expected
                cal = cfg['spatial_calibration']
                assert digest(cal['path']) == cal['sha256'] and read(cal['path'])['normalization'] == 'global_rms'
                assert len(set(cfg['model']['spatial_scale'])) == 1
                folder = root/('transition-'+run.name)
                spath, hpath = folder/'state-audit.json', root/('health-'+run.name)/'health.json'
                state, health = read(spath), read(hpath)
                windows = read(folder/'windows.json')
                assert windows == read(oldstate/'windows.json') and state['manifest_sha256'] == digest(folder/'windows.json')
                assert state['step'] == health['audit']['step'] == 2000 and state['cases'] == 64
                assert state['config_sha256'] == digest(run/'config.json')
                weights_path = run/'model-step0002000/weights.pt'
                weights_hash = digest(weights_path)
                assert state['weights_sha256'] == weights_hash
                assert all(v for v in health['audit'].values() if isinstance(v,bool))
                reference_health = read(OLD/('health-'+old.name)/'health.json')
                assert [(c['episode'],c['start']) for c in health['cases']] == [(c['episode'],c['start']) for c in reference_health['cases']]
                assert len(health['cases']) == 16
                with np.load(folder/'state-audit.npz') as cache:
                    observed, actions, clean = cache['observed'].copy(), cache['actions'].copy(), cache['clean_rollout'].copy()
                assert np.array_equal(observed,refobs) and np.array_equal(actions,refactions)
                null = null_components(observed,clean,state)
                planning = {}
                for cohort, phase in [('primary','planning'),('fresh','fresh-planning')]:
                    planning_root = FRESH if arm == 'identity' and cohort == 'fresh' else root
                    ppath = planning_root/(phase+'-'+run.name)/'metrics.json'
                    p = read(ppath)
                    refpath = (FRESH if cohort == 'fresh' else OLD)/(phase+'-'+old.name)/'metrics.json'
                    assert signature(p) == signature(read(refpath))
                    assert Path(p['checkpoint']) == run/'model-step0002000' and p['checkpoint_audit']['strict_equal']
                    successes = p['metrics']['episode_successes']
                    assert len(successes) == len(p['pairs']) == 50 and all(x in [0,1] for x in successes)
                    assert abs(2*sum(successes)-p['metrics']['success_rate']) < 1e-8
                    planning[cohort] = dict(success=2*sum(successes),outcomes=successes,signature=signature(p),artifact=artifact(ppath))
                c = state['conditions']
                row = dict(task=task,seed=seed,arm=arm,id=run.name,planning=planning,null=null,
                    other_initial_ratio=c['other_initial']['mse']/c['clean_rollout']['mse'],
                    other_actions_ratio=c['other_episode_actions']['mse']/c['clean_rollout']['mse'],
                    wrong_order_ratio=c['wrong_action_order']['mse']/c['clean_rollout']['mse'],
                    one_step_over_copy=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                    renderer={key:health['summary']['fm_t0.95_'+key]/health['summary']['fm_t0.95_forecast'] for key in ['wrong_state','wrong_actions','carry']},
                    state=state,health=health,artifacts={k:artifact(p) for k,p in dict(config=run/'config.json',weights=weights_path,state=spath,health=hpath,cache=folder/'state-audit.npz').items()})
                if arm != 'identity':
                    parent = PARENT/run.name
                    for phase in ['pre-resume-fullstate','post-copy-fullstate']:
                        a = read(run/(phase+'.json')); assert a['passed'] and all(a['checks'].values()) and a['step'] == 600
                        assert a['checkpoint_sha256'] == digest(parent/'last.ckpt')
                    archives = []
                    for step in [1200,2000]:
                        a = read(ROOT/('archive'+str(step)+'-'+run.name)/'fullstate.json')
                        assert a['passed'] and all(a['checks'].values()) and a['step'] == step
                        assert a['source_commit'] == protocol['model_source']['source_commit']
                        assert a['config_sha256'] == digest(run/'config.json')
                        assert a['checkpoint_sha256'] == digest(run/f'checkpoint-step{step:07d}.ckpt')
                        assert a['portable_sha256'] == digest(run/f'model-step{step:07d}/weights.pt')
                        assert a['frozen_tensors']['equal'] and a['frozen_tensors']['compared_tensors'] == 210
                        stage = a['native_stage_complete']
                        assert stage['initial_step'] == (600 if step == 1200 else 1200) and stage['stop_step'] == step
                        archives.append(a)
                    native_path = ROOT/('fullstate2000-'+run.name)
                    native, extra = read(native_path/'preflight.json'), read(native_path/'readout-audit.json')
                    assert native['passed'] and all(native['checkpoint']['checks'].values()) and extra['passed']
                    assert extra['step'] == 2000 and extra['original_initial_tensors_exact'] == 553 and extra['frozen_tensors_unchanged'] == 210
                    assert extra['new_parameters'] == 5376 and extra['new_tensors'] == 4
                    assert native['weights_sha256'] == extra['weights_sha256'] == weights_hash
                    assert extra['checkpoint_sha256'] == archives[-1]['checkpoint_sha256'] == digest(run/'last.ckpt')
                    assert extra['initial_sha256'] == digest(run/'initial-model/weights.pt') == digest(parent/'initial-model/weights.pt')
                    # The old native2000 resume references its preserved600
                    # initial export; it did not duplicate initial-model locally.
                    old_initial = B/'runs/local_isotropic600_20261031'/old.name/'initial-model/weights.pt'
                    assert extra['reference_initial_sha256'] == digest(old_initial)
                    assert extra['head_gradient']['passed'] and all(v['finite'] and v['max_abs'] > 0 for v in extra['head_gradient']['per_tensor'].values())
                    side = read(root/('health-'+run.name)/'exact-parameter-evidence.json')
                    assert side['passed'] and side['expected_step'] == 2000
                    assert side['evidence']['hashes']['weights'] == weights_hash
                    assert side['evidence']['fullstate_report_sha256'] == digest(native_path/'preflight.json')
                    assert side['evidence']['readout_report_sha256'] == digest(native_path/'readout-audit.json')
                    for name, value in side['evidence']['parameters'].items():
                        assert value['actual_fp32_gradient_max'] == extra['head_gradient']['per_tensor'][name[len(PREFIX):]]['max_abs']
                    initial = torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
                    initial_head = {k:initial[PREFIX+k].clone() for k in NAMES}
                    if reference_head is None: reference_head = initial_head
                    else: assert all(torch.equal(v,reference_head[k]) for k,v in initial_head.items())
                    weights = torch.load(weights_path,map_location='cpu',weights_only=True,mmap=True)
                    head = {k:weights[PREFIX+k].float() for k in NAMES}
                    with torch.no_grad():
                        row['head'] = dict(observed=measure(torch.from_numpy(observed),head,arm),
                            clean_prefix=measure(torch.from_numpy(np.concatenate([observed[:,:1],clean[:,:-1]],axis=1)),head,arm),
                            correction_over_original_increment=None)
                    checks.append(dict(id=run.name,archives=archives,native=artifact(native_path/'preflight.json'),
                        direct_gradient=extra['head_gradient'],health_sidecar=artifact(root/('health-'+run.name)/'exact-parameter-evidence.json'),
                        original_initial553=True,frozen210=True,matched_observed_actions=True,matched_primary_fresh_cases_solver_normalization=True))
                    row['parent600'] = {k:parent_rows[run.name][k] for k in ['success','rollout_over_copy','other_initial_ratio','other_actions_ratio']}
                rows.append(row)
    comparisons, aggregate = [], []
    for task in ['tworoom','cube']:
        for cohort in ['primary','fresh']:
            for candidate,reference in [('static','identity'),('state','identity'),('state','static')]:
                results = []
                for seed in [1,2]:
                    a,b = [next(x for x in rows if (x['task'],x['seed'],x['arm']) == (task,seed,arm)) for arm in [candidate,reference]]
                    result = dict(task=task,cohort=cohort,seed=seed,candidate=candidate,reference=reference,
                                  **paired(a['planning'][cohort]['outcomes'],b['planning'][cohort]['outcomes']))
                    comparisons.append(result); results.append(result)
                matrix = np.asarray([x['per_case_difference'] for x in results],dtype=float)
                draws = matrix.mean(0)[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
                aggregate.append(dict(task=task,cohort=cohort,candidate=candidate,reference=reference,
                    difference_pp=float(matrix.mean()*100),per_seed_difference_pp=[x['difference_pp'] for x in results],
                    ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    assert len(rows) == 12 and len(checks) == 8 and not torch.cuda.is_initialized()
    return dict(rows=rows,checks=checks,comparisons=comparisons,aggregate_comparisons=aggregate,sources=sources,
        protocol=artifact(ROOT/'protocol.json'),plan=artifact(ROOT/'plan.json'),closure=read(ROOT/'closure.json'),
        completion=dict(actual_exits0=72,native_training_stages=16,archive_checks=16,prepost_resume_checks=16,final_fullstate_checks=8),
        parent600_contract='Original8 training OS statuses unknown;8 fullstate exit0;8 preserved originalhealth exit1;final24 evaluation exits0. The new2000 root has72 actual exit0, not a relabeling of600.',
        helper=artifact(__file__),scope='Completed same-seed/same-case CPU review. No model/encoder/predictor execution, new cohort, fitting, GPU or budget extension. Head-only FP32 formula uses saved weights and states; missing pre-correction delta is not inferred. Paired CIs resample50 common cases jointly over two fixed trained seeds,10000 draws seed1042; not training-seed uncertainty or multiplicity-adjusted. Cube original evaluation has documented repeat variance.')


def report(s):
    lines=['# Dynamic readout2000：完整审阅','','72个端点均实际exit0；原600阶段的未知OS退出码和失败health记录不变。8组均在2000停止，16次完整归档、16次600恢复前后审计、8次最终完整状态/实际梯度通过；原553初始化/210冻结张量、两个病例集、规划器和归一化匹配。已关闭空闲调度器并HOLD，无后续训练。','','|任务/病例集|identity seed1/2|STATIC seed1/2|STATE seed1/2|STATE−STATIC pp [条件95%CI]|','|---|---:|---:|---:|---:|']
    for task in ['tworoom','cube']:
        for cohort in ['primary','fresh']:
            values=['/'.join(str(x['planning'][cohort]['success']) for x in s['rows'] if x['task']==task and x['arm']==arm) for arm in ['identity','static','state']]
            q=next(x for x in s['aggregate_comparisons'] if(x['task'],x['cohort'],x['candidate'],x['reference'])==(task,cohort,'state','static'))
            lines.append('|{}/{}|{}|{}|{}|{:+.1f} [{:+.1f},{:+.1f}]|'.format(task,cohort,*values,q['difference_pp'],*q['ci95_pp']))
    lines+=['','|任务/seed|臂|clean/zero|clean/copy|错初态/错动作比|energy / 2Epy|h1→h7 clean/zero|','|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        n=r['null'];h=n['per_horizon']['clean_over_zero']
        lines.append('|{}/{}|{}|{:.3f}|{:.3f}|{:.3f}/{:.3f}|{:.3f}/{:.3f}|{:.3f}→{:.3f}|'.format(r['task'],r['seed'],r['arm'],n['clean_over_zero'],n['clean_over_copy'],r['other_initial_ratio'],r['other_actions_ratio'],n['prediction_energy'],n['twice_alignment'],h[0],h[-1]))
    lines+=['','CI条件于这两个训练seed及共同病例，不是训练seed总体区间，也未校正多重比较。Cube原协议有重复波动。zero为固定训练中心，MSE=zero+energy−2Epy；不拟合验证均值或alpha，null级平均误差不等于零输出/无信息。所有7个horizon及renderer干预、head变化保留在summary。缺少修正前δ，仍不报告Mδ/δ。逐参数logger伪0保留，实际梯度检查见fullstate与health sidecar。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    assert not (a.output/'results.json').exists() and not (a.output/'summary.json').exists()
    full=collect();(a.output/'results.json').write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
    summary={k:v for k,v in full.items() if k not in ['rows','checks','comparisons']}
    summary['full_results']=artifact(a.output/'results.json')
    summary['rows']=[{k:v for k,v in r.items() if k not in ['state','health','planning']} | {'planning':{c:{'success':v['success'],'artifact':v['artifact']} for c,v in r['planning'].items()}} for r in full['rows']]
    summary['checks']=[{k:v for k,v in c.items() if k!='archives'} | {'archives':[{'step':v['step'],'checkpoint_sha256':v['checkpoint_sha256'],'checks':v['checks']} for v in c['archives']]} for c in full['checks']]
    summary['comparisons']=[{k:v for k,v in c.items() if k!='per_case_difference'} for c in full['comparisons']]
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (a.output/'RESULTS.md').write_text(report(full))
    print(json.dumps(dict(output=str(a.output),comparisons=summary['aggregate_comparisons']),indent=2))
