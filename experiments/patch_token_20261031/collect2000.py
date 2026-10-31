"""Complete-only CPU collection of the immutable eight-run patch-token2000 gate.

No model import/forward, environment rollout, GPU use, fit, or training.
"""
import argparse
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

B = Path('/efs/prescottsun/SPT-JEPA')
OLD = B/'runs/local_isotropic2000_20261031'
PRE = B/'runs/patch_token600_20261031'
MODEL = '1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'
HELPER = '3a3b249074b0cd59c462092d87a3840a542fe55a'


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(None)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    return dict(path=str(path), sha256=sha(str(path)), bytes=Path(path).stat().st_size)


def signature(p):
    return {k: p[k] for k in ['manifest_sha256', 'pairs', 'solver', 'plan', 'protocol', 'split', 'normalization']}


def paired(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    assert a.shape == b.shape == (50,)
    delta = a-b
    draws = delta[np.random.default_rng(1042).integers(50, size=(10000, 50))].mean(1)*100
    return dict(difference_pp=float(delta.mean()*100), gained=int(((a == 1) & (b == 0)).sum()),
                lost=int(((a == 0) & (b == 1)).sum()), ci95_pp=np.quantile(draws, [.025, .975]).tolist(),
                per_case_difference=delta.tolist())


def moments(pred, target):
    pred, target = pred.astype(np.float64), target.astype(np.float64)
    zero = np.square(target).mean((0, 2))
    energy = np.square(pred).mean((0, 2))
    alignment = 2*(pred*target).mean((0, 2))
    mse = np.square(pred-target).mean((0, 2))
    assert np.allclose(mse, zero+energy-alignment, rtol=1e-12, atol=1e-12)
    return dict(mse=float(mse.mean()), zero_mse=float(zero.mean()), prediction_energy=float(energy.mean()),
                alignment_twice=float(alignment.mean()), mse_over_zero=float(mse.mean()/zero.mean()),
                per_horizon=dict(mse=mse.tolist(), zero_mse=zero.tolist(), prediction_energy=energy.tolist(),
                                 alignment_twice=alignment.tolist(), mse_over_zero=(mse/zero).tolist()))


def predictor_hash(weights):
    h = hashlib.sha256()
    for key in sorted(k for k in weights if k.startswith('predictor.') and k != 'predictor.coordinate_basis'):
        h.update(key[len('predictor.'):].encode())
        h.update(weights[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def collect(root):
    torch.set_num_threads(2)
    plan = read(root/'plan.json')
    assert len(plan['jobs']) == 72
    exits = [read(root/'exits'/(j['id']+'.json')) for j in plan['jobs']]
    assert all(e['returncode'] == 0 for e in exits)
    assert (root/'HOLD').exists() and (root/'HOLD_DISPATCH').exists()
    closed = read(root/'closed.json')
    assert closed['all72_actual_exit0'] and closed['owned_gpus_no_compute']
    sources = []
    for name, expected in [('source-record.json', MODEL), ('helper-source-record.json', HELPER)]:
        source = read(root/name)
        assert source['source_commit'] == expected
        manifest = read(source['manifest'])
        assert manifest['source_commit'] == expected
        for path, digest in manifest['files'].items():
            assert sha(str(Path(source['source'])/path)) == digest, path
        sources.append(dict(record=source, manifest=artifact(source['manifest'])))
    model = sources[0]['record']
    runtime = read(model['runtime_manifest'])
    for path, digest in runtime['files'].items():
        assert sha(str(Path(runtime['source'])/path)) == digest, path
    cpu = read(Path(model['source'])/'experiments/patch_token_20261031/cpu/production.json')
    initial_hash = {(r['task'], r['seed']): r['predictor_sha256'] for r in cpu['initial']}
    jobs = {j['output']: j for j in plan['jobs'] if j['phase'] == 'learn1200'}
    assert len(jobs) == 8 and all(j['source'] == model for j in plan['jobs'])
    parent_summary = read(B/'reports/patch-token600-20261031/summary.json')
    assert sha(parent_summary['full_results']['path']) == parent_summary['full_results']['sha256']
    parent_full = read(parent_summary['full_results']['path'])
    parent_rows = {(r['task'], r['seed'], r['arm']): r for r in parent_full['rows']}
    references = {r['id']: r for r in read(root/'protocol.json')['references']}
    rows, checks = [], []
    oldplan = read(OLD/'plan.json')
    for task in ['tworoom', 'cube']:
        for seed in [1, 2]:
            old = OLD/f'{task}-spatial-local-seed{seed}'
            oldp = read(OLD/('planning-'+old.name)/'metrics.json')
            oldfp = B/'runs/followup_fresh2000_20261031'/('fresh-planning-'+old.name)/'metrics.json'
            oldfresh = read(oldfp)
            oldh = read(OLD/('health-'+old.name)/'health.json')
            oldw = read(OLD/('transition-'+old.name)/'windows.json')
            with np.load(OLD/('transition-'+old.name)/'state-audit.npz') as a:
                observed, actions = a['observed'].copy(), a['actions'].copy()
            for arm in ['flat_reference', 'identity', 'orthogonal']:
                is_old = arm == 'flat_reference'
                run = old if is_old else root/f'{task}-patch-{arm}-seed{seed}'
                owner = OLD if is_old else root
                name = run.name
                cfg = read(run/'config.json')
                stage = read(run/'stage-complete.json')
                assert stage['stage_complete'] and stage['stop_step'] == 2000 and stage['initial_step'] == 1200
                assert stage['config_sha256'] == sha(str(run/'config.json'))
                assert stage['scheduler_total_steps'] == cfg['max_steps'] == 124910 and cfg['batch_size'] == 128
                cal = cfg['spatial_calibration']
                assert sha(cal['path']) == cal['sha256'] and read(cal['path'])['normalization'] == 'global_rms'
                if is_old:
                    oldjobs = [j for j in oldplan['jobs'] if j['id'].endswith('-'+name) or j['output'] == str(run)]
                    assert len(oldjobs) == 8 and all(read(OLD/'exits'/(j['id']+'.json'))['returncode'] == 0 for j in oldjobs)
                else:
                    expected = copy.deepcopy(read(PRE/name/'config.json'))
                    expected.update(output=str(run), checkpoint_every=1200, retain_steps=[1200, 2000])
                    assert expected == cfg
                    expected_old = copy.deepcopy(read(old/'config.json'))
                    expected_old.update(output=str(run), model_class='patch_carrier', checkpoint_every=1200,
                                        retain_steps=[1200,2000], record_initial_weights=True)
                    expected_old['model'].update(patch_basis=arm, patch_predictor_seed=seed)
                    expected_old['audit_expectations'] = dict(gradients=dict(action_encoder='active'),
                        initial_unchanged_prefixes=['spatial_projection','spatial_mean','spatial_scale']+
                            (['predictor.coordinate_basis'] if arm=='orthogonal' else []))
                    assert cfg == expected_old, 'Only predictor/operational-audit config changes allowed'
                    assert jobs[str(run)]['config_sha256'] == sha(str(run/'config.json'))
                pp, sp, hp = [owner/(prefix+name)/fn for prefix, fn in [
                    ('planning-', 'metrics.json'), ('transition-', 'state-audit.json'), ('health-', 'health.json')]]
                p, s, h = map(read, [pp, sp, hp])
                fp = oldfp if is_old else owner/('fresh-planning-'+name)/'metrics.json'
                fresh = read(fp)
                assert signature(fresh) == signature(oldfresh)
                assert sha(fresh['manifest']) == fresh['manifest_sha256']
                assert fresh['checkpoint_audit']['strict_equal'] and Path(fresh['checkpoint']) == run/'model-step0002000'
                if not is_old:
                    ref = references[name]
                    assert sha(ref['primary_metrics']) == ref['primary_metrics_sha256']
                    assert sha(ref['fresh_metrics']) == ref['fresh_metrics_sha256']
                    assert p['manifest_sha256'] == ref['primary_manifest_sha256']
                    assert fresh['manifest_sha256'] == ref['fresh_manifest_sha256']
                wpath = owner/('transition-'+name)/'windows.json'
                assert signature(p) == signature(oldp)
                assert sha(p['manifest']) == p['manifest_sha256']
                assert p['checkpoint_audit']['strict_equal'] and Path(p['checkpoint']) == run/'model-step0002000'
                assert read(wpath) == oldw and sha(str(wpath)) == s['manifest_sha256'] and s['cases'] == 64
                assert s['step'] == h['audit']['step'] == 2000 and s['config_sha256'] == sha(str(run/'config.json'))
                assert all(v for v in h['audit'].values() if isinstance(v, bool))
                assert len(h['cases']) == 16 and [(x['episode'], x['start']) for x in h['cases']] == [(x['episode'], x['start']) for x in oldh['cases']]
                weights = run/'model-step0002000/weights.pt'
                assert sha(str(weights)) == s['weights_sha256']
                npz = owner/('transition-'+name)/'state-audit.npz'
                with np.load(npz) as a:
                    assert np.array_equal(observed, a['observed']) and np.array_equal(actions, a['actions'])
                    assert np.array_equal(a['observed_future'], observed[:, 1:])
                    assert not np.count_nonzero(a['zero_prediction'])
                    energy = {k: moments(a[k], observed[:, 1:]) for k in s['conditions']}
                for k, e in energy.items():
                    assert np.isclose(e['mse'], s['conditions'][k]['mse'], rtol=2e-6, atol=1e-8), (name, k)
                c = s['conditions']
                clean = c['clean_rollout']['mse']
                outcomes = p['metrics']['episode_successes']
                assert len(outcomes) == len(p['pairs']) == 50 and all(x in [0, 1, False, True] for x in outcomes)
                success = sum(outcomes)*2
                assert np.isclose(success, p['metrics']['success_rate'])
                fresh_outcomes = fresh['metrics']['episode_successes']
                assert len(fresh_outcomes) == len(fresh['pairs']) == 50
                assert all(x in [0,1,False,True] for x in fresh_outcomes)
                assert np.isclose(sum(fresh_outcomes)*2, fresh['metrics']['success_rate'])
                status = read(run/'status.json')
                assert status['step'] == 2000 and status['state'] == 'finished'
                row = dict(task=task, seed=seed, arm=arm, id=name, success=success, outcomes=outcomes,
                    fresh_success=sum(fresh_outcomes)*2, fresh_outcomes=fresh_outcomes,
                    loss=status['loss'], peak_gib=status['peak_gib'],
                    rollout_over_copy=clean/c['initial_carry']['mse'], rollout_over_zero=clean/c['zero_prediction']['mse'],
                    one_step_over_copy=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                    other_initial_ratio=c['other_initial']['mse']/clean, other_actions_ratio=c['other_episode_actions']['mse']/clean,
                    wrong_order_ratio=c['wrong_action_order']['mse']/clean,
                    renderer_ratios={key: h['summary']['fm_t0.95_'+key]/h['summary']['fm_t0.95_forecast'] for key in ['wrong_state','wrong_actions','carry','zero']},
                    energy=energy, conditions=c, health_summary=h['summary'], model_size=read(run/'model_size.json'),
                    trace=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()],
                    artifacts={k: artifact(v) for k,v in dict(config=run/'config.json', planning=pp, fresh_planning=fp, state=sp, cache=npz,
                        health=hp, windows=wpath, weights=weights, calibration=Path(cal['path'])).items()},
                    common_observations_actions_exact=True, planning_signature=signature(p))
                parent_row = parent_rows[(task,seed,arm)]
                assert signature(p) == parent_row['planning_signature']
                with np.load(parent_row['artifacts']['cache']['path']) as a:
                    assert np.array_equal(a['observed'], observed) and np.array_equal(a['actions'], actions)
                row['parent600'] = {k:parent_row[k] for k in ['success','outcomes','rollout_over_copy','rollout_over_zero',
                    'one_step_over_copy','other_initial_ratio','other_actions_ratio','wrong_order_ratio','energy']}
                row['change600_to2000'] = paired(outcomes,parent_row['outcomes'])
                if not is_old:
                    npth = root/('fullstate2000-'+name)/'preflight.json'
                    epth = npth.with_name('patch-audit.json')
                    native, extra = read(npth), read(epth)
                    assert native['passed'] and extra['passed'] and all(native['checkpoint']['checks'].values())
                    assert native['config_sha256'] == sha(str(run/'config.json'))
                    assert native['weights_sha256'] == sha(str(weights))
                    assert extra['expected_step'] == native['checkpoint']['step'] == 2000
                    assert extra['native_report_sha256'] == sha(str(npth))
                    for path, digest in extra['binding'].items():
                        assert sha(path) == digest, path
                    assert extra['nonpredictor_initial_tensors_exact'] == 466 and extra['original_initial_tensors'] == 553
                    assert extra['paired_parameter_initial_tensors_exact'] == 542 and extra['paired_initial_all_common_buffers_exact']
                    assert extra['original_native_checks_unmodified'] and extra['additional_backward_calls'] == 0
                    assert extra['optimizer']['native_full_state_roundtrip'] and extra['optimizer']['sampler_cursor'] == 256000
                    assert extra['optimizer']['all_trainable_parameters_have_state'] and extra['optimizer']['frozen_encoder_pool_have_no_state']
                    assert native['frozen_tensors_unchanged']['equal'] and native['frozen_tensors_unchanged']['compared_tensors'] == 210+(arm == 'orthogonal')
                    groups = extra['gradient']['patch_groups']
                    assert len(groups) == 22 and all(v['passed'] for v in groups.values())
                    assert all(v['gradient_present'] and v['finite'] for v in extra['gradient']['predictor_per_tensor'].values())
                    assert sha(str(run/'checkpoint-step0000600.ckpt')) == sha(str(PRE/name/'last.ckpt'))
                    for phase in ['pre-resume-fullstate','post-copy-fullstate']:
                        a = read(run/(phase+'.json'))
                        assert a['passed'] and all(a['checks'].values())
                        assert a['checkpoint_sha256'] == sha(str(PRE/name/'last.ckpt'))
                        assert a['source_commit'] == MODEL and a['sampler']['passed']
                    archives = []
                    for step in [1200,2000]:
                        ap = root/('archive'+str(step)+'-'+name)/'fullstate.json'
                        a = read(ap)
                        assert a['passed'] and all(a['checks'].values()) and a['step'] == step
                        assert a['source_commit'] == MODEL and a['sources']['helper']['source_commit'] == HELPER
                        assert sha(a['archived_checkpoint']) == a['checkpoint_sha256']
                        assert sha(a['portable_path']) == a['portable_sha256']
                        assert a['native_stage_complete']['initial_step'] == (600 if step==1200 else 1200)
                        if step == 2000:assert a['checkpoint_sha256'] == native['checkpoint_sha256']
                        archives.append(dict(step=step,record=artifact(ap),checkpoint_sha256=a['checkpoint_sha256'],checks=a['checks'],
                            frozen_tensors=a['frozen_tensors']['compared_tensors'],sampler=a['sampler']))
                    initial = torch.load(run/'initial-model/weights.pt', map_location='cpu', weights_only=True, mmap=True)
                    ihash = predictor_hash(initial)
                    assert ihash == initial_hash[(task, seed)]
                    del initial
                    portable = []
                    for step in [1200, 2000]:
                        wp = run/f'model-step{step:07d}/weights.pt'
                        w = torch.load(wp, map_location='cpu', weights_only=True, mmap=True)
                        assert len(w) == 553+(arm == 'orthogonal')
                        assert all(bool(torch.isfinite(t).all()) for t in w.values())
                        portable.append(dict(step=step, finite=True, **artifact(wp)))
                        del w
                    row.update(native_gradients=native['gradients'], patch_gradients=groups,
                        zero_norm_predictor_tensors=[k for k,v in extra['gradient']['predictor_per_tensor'].items() if v['norm'] == 0],
                        initial_predictor_sha256=ihash, optimizer=extra['optimizer'], portable=portable,
                        native_fullstate=artifact(npth), patch_audit=artifact(epth), archives=archives)
                    checks.append(dict(id=name, shared_nonpredictor466_exact=True, paired542_parameters_exact=True,
                        common_buffers_exact=True, native_restore=True, original_native_checks_unmodified=True,
                        frozen_tensors=210+(arm=='orthogonal'), per_layer_groups_positive_finite=22,
                        same_cases_solver_normalization=True, observations_actions_exact=True,
                        old_flat_config_exact_except_predictor_and_operational_audit=True))
                rows.append(row)
    comparisons, averages, growth = [], [], []
    for split, outcome_key in [('primary','outcomes'),('fresh','fresh_outcomes')]:
        for task in ['tworoom','cube']:
            for candidate,reference in [('identity','orthogonal'),('identity','flat_reference'),('orthogonal','flat_reference')]:
                pairs=[]
                for seed in [1,2]:
                    a=next(r for r in rows if(r['task'],r['seed'],r['arm'])==(task,seed,candidate))
                    b=next(r for r in rows if(r['task'],r['seed'],r['arm'])==(task,seed,reference))
                    pairs.append(dict(split=split,task=task,seed=seed,candidate=candidate,reference=reference,**paired(a[outcome_key],b[outcome_key])))
                comparisons.extend(pairs)
                delta=np.asarray([p['per_case_difference']for p in pairs]).mean(0)
                draws=delta[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
                averages.append(dict(split=split,task=task,candidate=candidate,reference=reference,mean_difference_pp=float(delta.mean()*100),
                    by_seed_pp=[p['difference_pp']for p in pairs],ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    for task in ['tworoom','cube']:
        for arm in ['flat_reference','identity','orthogonal']:
            rr=[r for r in rows if r['task']==task and r['arm']==arm]
            delta=np.asarray([r['change600_to2000']['per_case_difference']for r in rr]).mean(0)
            draws=delta[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
            growth.append(dict(task=task,arm=arm,mean_difference_pp=float(delta.mean()*100),ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    assert not torch.cuda.is_initialized()
    return dict(passed=True,root=str(root),all72_actual_exit0=True,closed=closed,rows=rows,checks=checks,
        comparisons=comparisons,comparison_summaries=averages,growth_summaries=growth,parent600_results=parent_summary["full_results"],sources=sources,collector=artifact(__file__),
        CPU_only=True,GPU_initialized=False,
        scope='All8 task/seed/arm runs retained.64 existing validation windows;50 original strict primary and50 pre-existing fresh cases. Fixed raw training-mean zero, no validation fit. Historical flat predictor differs in size/compute.10k paired empirical-case bootstrap draws seed1042; shared cases resampled jointly across two fixed training seeds, not seed-population uncertainty or multiplicity adjustment. Cube rendering has known small repeat variance.2000 reached after preserved1200 archive; no further continuation authorized.')

def report(s):
    lines=['# Patch-token2000：完整成熟门结果','',
        '72/72实际exit0；8组从原600原生恢复，1200/2000共16份完整归档、全部最终fullstate/22分层梯度组/health通过。精确owned dispatcher/monitor关闭并HOLD；无自动>2000。','',
        '|任务|臂|primary600→2000，seed1/2|fresh2000，seed1/2|', '|---|---|---:|---:|']
    for task in ['tworoom','cube']:
        for arm in ['flat_reference','identity','orthogonal']:
            rr=[r for r in s['rows']if r['task']==task and r['arm']==arm]
            before='/'.join(str(r['parent600']['success'])for r in rr)
            after='/'.join(str(r['success'])for r in rr)
            fresh='/'.join(str(r['fresh_success'])for r in rr)
            lines.append(f'|{task}|{arm}|{before} → {after}|{fresh}|')
    lines+=['','|病例/任务|对比|均值差pp [条件95%CI]|','|---|---|---:|']
    for q in s['comparison_summaries']:
        lines.append('|{}/{}|{}−{}|{:+.1f} [{:+.1f},{:+.1f}]|'.format(q['split'],q['task'],q['candidate'],q['reference'],q['mean_difference_pp'],*q['ci95_pp']))
    lines+=['','|任务/seed|臂|clean/copy600→2000|clean/zero600→2000|别段动作/clean600→2000|teacher/邻帧copy2000|','|---|---|---:|---:|---:|---:|']
    for r in s['rows']:
        p=r['parent600'];lines.append('|{}/{}|{}|{:.3f}→{:.3f}|{:.3f}→{:.3f}|{:.4f}→{:.4f}|{:.3f}|'.format(r['task'],r['seed'],r['arm'],
            p['rollout_over_copy'],r['rollout_over_copy'],p['rollout_over_zero'],r['rollout_over_zero'],p['other_actions_ratio'],r['other_actions_ratio'],r['one_step_over_copy']))
    lines+=['',
        '全部raw observed/actions跨两token、旧flat以及600/2000缓存逐元素一致；病例/solver/归一化/hash匹配。两token共享初值542参数及共同buffers，466非predictor初值与flat相同，冻结210/211张量不变。准备16次实际模型/AdamW/scheduler恢复及归档审计保留，RNG仅保存/复制值核验，sampler next32与start0序列参考相同；不声称不中断GPU轨迹逐值复现。',
        '', '所有7horizon的raw zero/copy/teacher/clean、预测能量与对齐项保留在summary，核MSE=zero+energy−alignment。zero是固定训练中心，无验证拟合；平均误差接近zero并不证明常数或无信息，teacher额外接收真实前缀。动作干预与任务收益分别报告。',
        '', '两token参数/初始化/损失一致，orth仅在predictor内部固定变基并逆回raw goal坐标。旧flat predictor11,389,184参数，token523,536；token训练predictor forward约3.17倍MAC，orth另计两次FP64变基，不能作匹配计算或单独归因于rank的对照。固定普通动作作为输入，没有专家最优动作标签，损失保持sample-level。',
        '', 'CI对共同50病例进行10000次配对重采样并在两固定seed间共享病例索引；它不覆盖训练seed总体不确定性、多重比较或Cube原协议重复波动。fresh同属验证集、复用既有生成规则，不能称独立测试集。报告不授权继续训练或宣称原创。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(B/'runs/patch_token2000_20261031'))
    p.add_argument('--output',default=str(B/'reports/patch-token2000-20261031'));args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=collect(Path(args.root))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    small=copy.deepcopy(result)
    for r in small['rows']:
        for k in ['trace','planning_signature','conditions']:r.pop(k,None)
    small['full_results']=artifact(out/'results.json')
    (out/'summary.json').write_text(json.dumps(small,indent=2,allow_nan=False)+'\n')
    (out/'RESULTS.md').write_text(report(result))
    print(json.dumps(dict(passed=True,endpoints=72,comparisons=result['comparison_summaries'],rows=[{k:r[k]for k in ['id','success','fresh_success','rollout_over_zero','other_actions_ratio']}for r in result['rows']])))


if __name__=='__main__':main()
