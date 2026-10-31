"""Complete-only CPU collection of the immutable eight-run patch-token gate.

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
OLD = B/'runs/local_isotropic600_20261031'
PRE = B/'runs/patch_token_preflight_20261031'
MODEL = '1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'
HELPER = 'd26df118325cf656e15b3d36fe0089c99f4b4dc2'


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
    assert len(plan['jobs']) == 40
    exits = [read(root/'exits'/(j['id']+'.json')) for j in plan['jobs']]
    assert all(e['returncode'] == 0 for e in exits)
    assert (root/'HOLD').exists() and (root/'HOLD_DISPATCH').exists()
    closed = read(root/'closed.json')
    assert closed['all40_actual_exit0'] and closed['owned_gpus_no_compute']
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
    jobs = {j['output']: j for j in plan['jobs'] if j['phase'] == 'train600'}
    assert len(jobs) == 8 and all(j['source'] == model for j in plan['jobs'])
    rows, checks = [], []
    oldplan = read(OLD/'plan.json')
    for task in ['tworoom', 'cube']:
        for seed in [1, 2]:
            old = OLD/f'{task}-spatial-local-seed{seed}'
            oldp = read(OLD/('planning-'+old.name)/'metrics.json')
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
                assert stage['stage_complete'] and stage['stop_step'] == 600 and stage['initial_step'] == 0
                assert stage['config_sha256'] == sha(str(run/'config.json'))
                assert stage['scheduler_total_steps'] == cfg['max_steps'] == 124910 and cfg['batch_size'] == 128
                cal = cfg['spatial_calibration']
                assert sha(cal['path']) == cal['sha256'] and read(cal['path'])['normalization'] == 'global_rms'
                if is_old:
                    oldjobs = [j for j in oldplan['jobs'] if j['id'].endswith('-'+name) or j['output'] == str(run)]
                    assert len(oldjobs) == 5 and all(read(OLD/'exits'/(j['id']+'.json'))['returncode'] == 0 for j in oldjobs)
                else:
                    expected = copy.deepcopy(read(PRE/name/'config.json'))
                    expected.update(output=str(run), checkpoint_every=200, retain_steps=[200, 400, 600])
                    assert expected == cfg
                    expected_old = copy.deepcopy(read(old/'config.json'))
                    expected_old.update(output=str(run), model_class='patch_carrier', checkpoint_every=200,
                                        retain_steps=[200,400,600], record_initial_weights=True)
                    expected_old['model'].update(patch_basis=arm, patch_predictor_seed=seed)
                    expected_old['audit_expectations'] = dict(gradients=dict(action_encoder='active'),
                        initial_unchanged_prefixes=['spatial_projection','spatial_mean','spatial_scale']+
                            (['predictor.coordinate_basis'] if arm=='orthogonal' else []))
                    assert cfg == expected_old, 'Only predictor/operational-audit config changes allowed'
                    assert jobs[str(run)]['config_sha256'] == sha(str(run/'config.json'))
                pp, sp, hp = [owner/(prefix+name)/fn for prefix, fn in [
                    ('planning-', 'metrics.json'), ('transition-', 'state-audit.json'), ('health-', 'health.json')]]
                p, s, h = map(read, [pp, sp, hp])
                wpath = owner/('transition-'+name)/'windows.json'
                assert signature(p) == signature(oldp)
                assert sha(p['manifest']) == p['manifest_sha256']
                assert p['checkpoint_audit']['strict_equal'] and Path(p['checkpoint']) == run/'model-step0000600'
                assert read(wpath) == oldw and sha(str(wpath)) == s['manifest_sha256'] and s['cases'] == 64
                assert s['step'] == h['audit']['step'] == 600 and s['config_sha256'] == sha(str(run/'config.json'))
                assert all(v for v in h['audit'].values() if isinstance(v, bool))
                assert len(h['cases']) == 16 and [(x['episode'], x['start']) for x in h['cases']] == [(x['episode'], x['start']) for x in oldh['cases']]
                weights = run/'model-step0000600/weights.pt'
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
                status = read(run/'status.json')
                assert status['step'] == 600 and status['state'] == 'finished'
                row = dict(task=task, seed=seed, arm=arm, id=name, success=success, outcomes=outcomes,
                    loss=status['loss'], peak_gib=status['peak_gib'],
                    rollout_over_copy=clean/c['initial_carry']['mse'], rollout_over_zero=clean/c['zero_prediction']['mse'],
                    one_step_over_copy=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                    other_initial_ratio=c['other_initial']['mse']/clean, other_actions_ratio=c['other_episode_actions']['mse']/clean,
                    wrong_order_ratio=c['wrong_action_order']['mse']/clean,
                    renderer_ratios={key: h['summary']['fm_t0.95_'+key]/h['summary']['fm_t0.95_forecast'] for key in ['wrong_state','wrong_actions','carry','zero']},
                    energy=energy, conditions=c, health_summary=h['summary'], model_size=read(run/'model_size.json'),
                    trace=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()],
                    artifacts={k: artifact(v) for k,v in dict(config=run/'config.json', planning=pp, state=sp, cache=npz,
                        health=hp, windows=wpath, weights=weights, calibration=Path(cal['path'])).items()},
                    common_observations_actions_exact=True, planning_signature=signature(p))
                if not is_old:
                    npth = root/('fullstate-'+name)/'preflight.json'
                    epth = npth.with_name('patch-audit.json')
                    native, extra = read(npth), read(epth)
                    assert native['passed'] and extra['passed'] and all(native['checkpoint']['checks'].values())
                    assert native['config_sha256'] == sha(str(run/'config.json'))
                    assert native['weights_sha256'] == sha(str(weights))
                    assert extra['expected_step'] == native['checkpoint']['step'] == 600
                    assert extra['native_report_sha256'] == sha(str(npth))
                    for path, digest in extra['binding'].items():
                        assert sha(path) == digest, path
                    assert extra['nonpredictor_initial_tensors_exact'] == 466 and extra['original_initial_tensors'] == 553
                    assert extra['paired_parameter_initial_tensors_exact'] == 542 and extra['paired_initial_all_common_buffers_exact']
                    assert extra['original_native_checks_unmodified'] and extra['additional_backward_calls'] == 0
                    assert extra['optimizer']['native_full_state_roundtrip'] and extra['optimizer']['sampler_cursor'] == 76800
                    assert extra['optimizer']['all_trainable_parameters_have_state'] and extra['optimizer']['frozen_encoder_pool_have_no_state']
                    assert native['frozen_tensors_unchanged']['equal'] and native['frozen_tensors_unchanged']['compared_tensors'] == 210+(arm == 'orthogonal')
                    groups = extra['gradient']['patch_groups']
                    assert len(groups) == 22 and all(v['passed'] for v in groups.values())
                    assert all(v['gradient_present'] and v['finite'] for v in extra['gradient']['predictor_per_tensor'].values())
                    initial = torch.load(run/'initial-model/weights.pt', map_location='cpu', weights_only=True, mmap=True)
                    ihash = predictor_hash(initial)
                    assert ihash == initial_hash[(task, seed)]
                    del initial
                    portable = []
                    for step in [200, 400, 600]:
                        wp = run/f'model-step{step:07d}/weights.pt'
                        w = torch.load(wp, map_location='cpu', weights_only=True, mmap=True)
                        assert len(w) == 553+(arm == 'orthogonal')
                        assert all(bool(torch.isfinite(t).all()) for t in w.values())
                        portable.append(dict(step=step, finite=True, **artifact(wp)))
                        del w
                    row.update(native_gradients=native['gradients'], patch_gradients=groups,
                        zero_norm_predictor_tensors=[k for k,v in extra['gradient']['predictor_per_tensor'].items() if v['norm'] == 0],
                        initial_predictor_sha256=ihash, optimizer=extra['optimizer'], portable=portable,
                        native_fullstate=artifact(npth), patch_audit=artifact(epth))
                    checks.append(dict(id=name, shared_nonpredictor466_exact=True, paired542_parameters_exact=True,
                        common_buffers_exact=True, native_restore=True, original_native_checks_unmodified=True,
                        frozen_tensors=210+(arm=='orthogonal'), per_layer_groups_positive_finite=22,
                        same_cases_solver_normalization=True, observations_actions_exact=True,
                        old_flat_config_exact_except_predictor_and_operational_audit=True))
                rows.append(row)
    comparisons, averages = [], []
    for task in ['tworoom', 'cube']:
        for candidate, reference in [('identity','orthogonal'), ('identity','flat_reference'), ('orthogonal','flat_reference')]:
            pairs = []
            for seed in [1, 2]:
                a = next(r for r in rows if (r['task'],r['seed'],r['arm']) == (task,seed,candidate))
                b = next(r for r in rows if (r['task'],r['seed'],r['arm']) == (task,seed,reference))
                q = dict(task=task, seed=seed, candidate=candidate, reference=reference, **paired(a['outcomes'],b['outcomes']))
                pairs.append(q)
            comparisons.extend(pairs)
            delta = np.asarray([p['per_case_difference'] for p in pairs]).mean(0)
            draws = delta[np.random.default_rng(1042).integers(50,size=(10000,50))].mean(1)*100
            averages.append(dict(task=task,candidate=candidate,reference=reference,mean_difference_pp=float(delta.mean()*100),
                                 by_seed_pp=[p['difference_pp'] for p in pairs],ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    assert not torch.cuda.is_initialized()
    return dict(passed=True,root=str(root),all40_actual_exit0=True,closed=closed,rows=rows,checks=checks,
        comparisons=comparisons,comparison_summaries=averages,sources=sources,collector=artifact(__file__),
        CPU_only=True,GPU_initialized=False,
        scope='All8 task/seed/arm runs retained.64 existing validation windows;50 original strict cases. Fixed raw training-mean zero, no validation fit. Historical flat predictor differs in size/compute.10k paired empirical-case bootstrap draws seed1042; shared cases resampled jointly across two fixed training seeds, not seed-population uncertainty or multiplicity adjustment. Cube rendering has known small repeat variance.600 remains below1249warmup; no continuation authorized.')


def report(s):
    lines=['# Patch-token 600：完整短门结果','',
        '40/40实际exit0；8组均fresh600，全部原生完整状态、分层梯度、初始/冻结张量及原病例协议核验通过。8槽空闲，三个dispatcher已关闭并HOLD；monitor末条记录全部完成后已退出。未续训。','',
        '|任务|旧flat1024 seed1/2|token identity seed1/2|token orthogonal seed1/2|identity−orthogonal pp [条件95%CI]|',
        '|---|---:|---:|---:|---:|']
    for task in ['tworoom','cube']:
        cells=['/'.join(str(r['success']) for r in s['rows'] if r['task']==task and r['arm']==arm) for arm in ['flat_reference','identity','orthogonal']]
        q=next(c for c in s['comparison_summaries'] if (c['task'],c['candidate'],c['reference'])==(task,'identity','orthogonal'))
        lines.append('|{}|{}|{}|{}|{:+.1f} [{:+.1f},{:+.1f}]|'.format(task,*cells,q['mean_difference_pp'],*q['ci95_pp']))
    lines += ['','|对比|TwoRoom均值差pp [条件95%CI]|Cube均值差pp [条件95%CI]|','|---|---:|---:|']
    for candidate in ['identity','orthogonal']:
        vals=[]
        for task in ['tworoom','cube']:
            q=next(c for c in s['comparison_summaries'] if (c['task'],c['candidate'],c['reference'])==(task,candidate,'flat_reference'))
            vals.append('{:+.1f} [{:+.1f},{:+.1f}]'.format(q['mean_difference_pp'],*q['ci95_pp']))
        lines.append('|{}−旧flat|{}|{}|'.format(candidate,*vals))
    lines += ['','|任务/seed|臂|clean/copy|clean/zero|teacher/相邻copy|错初态/clean|别段动作/clean|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        lines.append('|{task}/{seed}|{arm}|{rollout_over_copy:.3f}|{rollout_over_zero:.3f}|{one_step_over_copy:.3f}|{other_initial_ratio:.3f}|{other_actions_ratio:.5f}|'.format(**r))
    lines += ['',
        '空间identity在两任务两seed的动作交换效应均强于orthogonal，但均弱于旧flat（TR1.041/1.045、Cube1.156/1.151）。identity的clean/copy也没有超过旧flat。identity规划点估计相对两种对照均更高，但共享50病例的配对区间仍需完整保留，不能据此宣称稳定优势。orthogonal平均clean MSE更低；这与任务排序相反。现有结果支持继续区分动作条件信号与总体重构误差，尚未证实移除global输出秩限制带来大幅控制收益。',
        '', '所有token的clean误差仍高于固定训练中心zero；这不等于输出为常数或没有信息。summary保留7horizon的预测能量、对齐项和恒等式MSE=zero+energy−alignment，并记录真实初态/动作干预。teacher接收真实前缀，不能与纯因果rollout混同。',
        '', '两token臂共享全部542初始参数张量及共同buffers；466非predictor初值与旧flat一致。24份200/400/600 portable均有限，8份600完整checkpoint逐hash绑定，模型/AdamW/scheduler原生恢复通过，RNG保存字段齐备，sampler cursor76800及后续索引核验通过。22个predictor分层梯度组/每run全部有限非零；单参数零值另列，不能由模块非零推断每个参数均非零。',
        '', '原raw局部观测和动作缓存跨三臂逐元素一致；数据/校准/噪声/rate/renderer/goal L2/原CEM300×30,h5,history3、50rawsteps不变。token predictor523,536参数 vs flat11,389,184；TF predictor forward约3.17倍MAC，orth另有双FP64变基，故旧flat只作容量/计算不同的历史上下文。峰值显存112.86–112.97GiB。',
        '', '600仍在1249步warmup内。建议有限2000成熟检验有研究价值：identity相对旧flat四个task/seed的任务点估计均提高，Cube条件区间为正；但动作效应和平均MSE不优于flat，尚不足以把收益归因为更好的条件动力学。此为预算建议，但本报告不授权2000、不声称方法创新，也不把短门当成熟效果。置信区间仅条件于这两个已训练seed，未覆盖seed不确定性、多重比较或Cube渲染重复差异。']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(B/'runs/patch_token600_20261031'))
    p.add_argument('--output',default=str(B/'reports/patch-token600-20261031'));args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=collect(Path(args.root))
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    small=copy.deepcopy(result)
    for r in small['rows']:
        for k in ['trace','planning_signature','conditions']:r.pop(k,None)
    small['full_results']=artifact(out/'results.json')
    (out/'summary.json').write_text(json.dumps(small,indent=2,allow_nan=False)+'\n')
    (out/'RESULTS.md').write_text(report(result))
    print(json.dumps(dict(passed=True,endpoints=40,comparisons=result['comparison_summaries'],
        rows=[{k:r[k] for k in ['task','seed','arm','success','rollout_over_copy','rollout_over_zero','other_actions_ratio']} for r in result['rows']])))


if __name__=='__main__':main()
