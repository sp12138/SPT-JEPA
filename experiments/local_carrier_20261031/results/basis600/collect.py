"""One-pass, CPU-only collection of the completed orthogonal-basis control."""
import copy
import json
import math
from pathlib import Path
import time

import torch
from experiments.ideas_validate_20261031.collect import read, digest, canonical, paired_outcomes
from experiments.carrier_progress_20261031.collect_learning import (
    artifact, paired_summary, planning_signature, successes, verify_manifest,
)

B = Path('/efs/prescottsun/SPT-JEPA')
ROOT = B/'runs/local_basis600_20261031'
OLD = B/'runs/local_isotropic600_20261031'
PRE = B/'runs/local_basis_preflight_20261031'
OUT = B/'reports/local-basis600-20261031'


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def collect():
    torch.set_num_threads(2)
    plan, proto, src = read(ROOT/'plan.json'), read(ROOT/'protocol.json'), read(ROOT/'source-record.json')
    oldsrc = read(OLD/'source-record.json')
    assert len(plan['jobs']) == 20 and proto['step'] == 600 and len(proto['runs']) == 4
    assert (ROOT/'HOLD_DISPATCH').exists()
    assert read(PRE/'review.json')['passed']
    sources = [dict(source=s, manifest=verify_manifest(s)) for s in [src, oldsrc]]
    exits = []
    for j in plan['jobs']:
        e = read(ROOT/'exits'/(j['id']+'.json'))
        assert e['returncode'] == 0 and e['id'] == j['id'] and j['source'] == src
        exits.append(e)
    oldplan = read(OLD/'plan.json')
    rows, pairs, checks = [], [], []
    for rec in proto['runs']:
        name, task, seed = rec['id'], rec['task'], rec['training_seed']
        run = ROOT/name
        ref = OLD/f'{task}-spatial-local-seed{seed}'
        cfg, oldcfg = read(run/'config.json'), read(ref/'config.json')
        expected = copy.deepcopy(oldcfg)
        expected['output'] = str(run)
        expected['model']['spatial_basis'] = 'orthogonal'
        expected['audit_expectations']['initial_unchanged_prefixes'].append('spatial_orthogonal_basis')
        assert cfg == expected
        cal = cfg['spatial_calibration']
        assert digest(cal['path']) == cal['sha256'] and read(cal['path'])['normalization'] == 'global_rms'
        assert len(set(cfg['model']['spatial_scale'])) == 1
        fullpath = ROOT/('fullstate-'+name)/'preflight.json'
        full = read(fullpath)
        assert full['passed'] and all(full['checkpoint']['checks'].values())
        assert full['checkpoint']['step'] == 600 and full['frozen_tensors_unchanged']['equal']
        assert full['frozen_tensors_unchanged']['compared_tensors'] == 211
        assert all(v['passed'] for v in full['gradients'].values())
        initial = torch.load(run/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        reference = torch.load(ref/'initial-model/weights.pt', map_location='cpu', weights_only=True)
        finalpath = run/'model-step0000600/weights.pt'
        final = torch.load(finalpath, map_location='cpu', weights_only=True)
        qkey = 'spatial_orthogonal_basis'
        assert len(reference) == 553 and set(initial) == set(reference) | {qkey}
        assert all(torch.equal(v, initial[k]) for k,v in reference.items())
        frozen = [k for k in initial if k.startswith(('encoder.', 'pool.', 'spatial_'))]
        assert len(frozen) == 211 and all(torch.equal(initial[k], final[k]) for k in frozen)
        q = initial[qkey].double()
        assert torch.equal(q.T @ q, torch.eye(1024, dtype=torch.double))
        assert all(bool(torch.isfinite(v).all()) for v in final.values())
        assert digest(finalpath) == full['weights_sha256']
        checks.append(dict(task=task, seed=seed, original_initial_tensors_exact=553,
            extra_frozen_tensor=qkey, frozen600_exact=211, orthogonality_exact_fp64=True,
            fullstate=artifact(fullpath), initial=artifact(run/'initial-model/weights.pt'),
            reference_initial=artifact(ref/'initial-model/weights.pt'), final_weights=artifact(finalpath),
            normalization=cal, config_except_basis_and_output_exact=True))
        del initial, reference, final, q
        pairrows = []
        for arm, root, rr in [('identity', OLD, ref), ('orthogonal', ROOT, run)]:
            rn = rr.name
            relevant = [j for j in (oldplan if arm == 'identity' else plan)['jobs']
                        if j.get('output') == str(rr) or j['id'].endswith('-'+rn)]
            assert len(relevant) == 5
            for j in relevant:
                assert read(root/'exits'/(j['id']+'.json'))['returncode'] == 0
            paths = dict(config=rr/'config.json', stage=rr/'stage-complete.json', status=rr/'status.json',
                size=rr/'model_size.json', health=root/('health-'+rn)/'health.json',
                transition=root/('transition-'+rn)/'state-audit.json',
                windows=root/('transition-'+rn)/'windows.json', planning=root/('planning-'+rn)/'metrics.json',
                training=rr/'metrics.jsonl')
            stage, status, size, h, s, w, p = [read(paths[k]) for k in
                ['stage', 'status', 'size', 'health', 'transition', 'windows', 'planning']]
            assert stage['initial_step'] == 0 and stage['stage_complete']
            assert stage['stop_step'] == status['step'] == h['audit']['step'] == s['step'] == 600
            assert status['state'] == 'finished' and all(v for v in h['audit'].values() if isinstance(v, bool))
            assert s['cases'] == w['count'] == 64 and s['manifest_sha256'] == digest(paths['windows'])
            assert s['config_sha256'] == digest(paths['config'])
            assert size['total'] == h['parameters'] == p['checkpoint_audit']['parameters']
            assert Path(p['checkpoint']) == rr/'model-step0000600'
            oo, rate = successes(p)
            trace = [json.loads(line) for line in paths['training'].read_text().splitlines() if line.strip()]
            assert finite(trace) and finite(s) and finite(h)
            c = s['conditions']; clean = c['clean_rollout']['mse']
            row = dict(task=task, seed=seed, arm=arm, success=rate, episode_successes=oo,
                rollout=clean/c['initial_carry']['mse'],
                one_step=c['teacher_forced_prefix']['mse']/c['shifted_observed_carry']['mse'],
                wrongstate_ratio=c['other_initial']['mse']/clean,
                otheractions_ratio=c['other_episode_actions']['mse']/clean,
                wrongorder_ratio=c['wrong_action_order']['mse']/clean,
                renderer={k:h['summary']['fm_t0.95_'+k]/h['summary']['fm_t0.95_forecast']
                          for k in ['wrong_state', 'wrong_actions', 'carry']},
                rms=s['observed_state_rms'], coordinate_std_mean=s['observed_state_std_mean'],
                model_size=size, planning_seconds=p['seconds'], state=s, health=h, training=trace,
                planning_signature=planning_signature(p), windows=w,
                artifacts={k:artifact(v) for k,v in paths.items()})
            rows.append(row); pairrows.append(row)
        a, b = pairrows
        assert canonical(a['planning_signature']) == canonical(b['planning_signature'])
        assert a['windows'] == b['windows'] and a['model_size'] == b['model_size']
        assert [{k:v[k] for k in ['episode','start']} for v in a['health']['cases']] == [
            {k:v[k] for k in ['episode','start']} for v in b['health']['cases']]
        assert abs(a['rms']-b['rms']) < 1e-6
        for k in ['initial_carry', 'shifted_observed_carry', 'zero_prediction']:
            assert abs(a['state']['conditions'][k]['mse']-b['state']['conditions'][k]['mse']) < 1e-6
        pairs.append(dict(task=task, seed=seed, **paired_outcomes(b['episode_successes'],a['episode_successes'],10000)))
    summaries = [dict(task=task, **paired_summary([p for p in pairs if p['task']==task],10000))
                 for task in ['tworoom','cube']]
    return dict(created_at=time.time(), completed_endpoint_count=20, exits=exits, sources=sources,
        checks=checks, rows=rows, comparisons=pairs, comparison_summaries=summaries,
        collector=artifact(__file__), preflight_review=artifact(PRE/'review.json'),
        scope='Completed CPU artifact/tensor checks only. Both trained seeds retained. No model forward, probe fitting, GPU, training, or new simulation.')


def report(s):
    lines=['600步旋转短门没有一致规划收益或损害。20/20端点exit0；同病例、CEM、归一化和预算，'
        '4对全部553个原始初始张量逐位相同，仅新增固定Q；211个冻结张量含Q至600步不变，健康及完整状态检查通过。', '',
        '| 任务 | 原local成功率，seed1/2 | 旋转成功率，seed1/2 | 旋转−原local均值pp [配对95%区间] |',
        '|---|---:|---:|---:|']
    def rr(task,arm):return sorted([r for r in s['rows'] if r['task']==task and r['arm']==arm],key=lambda r:r['seed'])
    for task in ['tworoom','cube']:
        q=next(x for x in s['comparison_summaries'] if x['task']==task);lo,hi=q['shared_case_bootstrap_ci95_pp']
        vals=['/'.join(f"{r['success']:g}" for r in rr(task,arm)) for arm in ['identity','orthogonal']]
        lines.append(f"| {task} | {vals[0]} | {vals[1]} | {q['mean_difference_pp']:+g} [{lo:+g},{hi:+g}] |")
    lines += ['', '| 任务/seed | 基底 | rollout/初态复制 | 单步/前帧复制 | 错初态/正确 | 别段动作/正确 | DiT错初态/错动作 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in s['rows']:
        lines.append(f"| {r['task']}/{r['seed']} | {r['arm']} | {r['rollout']:.3f} | {r['one_step']:.3f} | "
            f"{r['wrongstate_ratio']:.3f} | {r['otheractions_ratio']:.3f} | "
            f"{r['renderer']['wrong_state']:.4f}/{r['renderer']['wrong_actions']:.4f} |")
    lines += ['', '比值均为误差之比；DiT在t=.95相对正确forecast。旋转后两任务/两seed的预测比复制更好，'
        '初态和动作干预也有影响；DiT错条件比仍近1，未形成成熟条件生成证据。600步仍处1249步warmup内，'
        '不能据此认定成熟架构的坐标依赖或独立于坐标。区间按50病例共同重采样10000次，仅条件于这两个训练seed。'
        'Q保留观测信息、RMS及欧氏距离至舍入误差；逐坐标std均值不具有旋转不变性。'
        '该控制同时改变predictor与renderer的坐标因子化，同seed高斯噪声仅同分布而非逐样本共轭；没有续跑。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    result=collect();OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    keep=['task','seed','arm','success','rollout','one_step','wrongstate_ratio','otheractions_ratio','wrongorder_ratio',
          'renderer','rms','coordinate_std_mean','model_size','planning_seconds']
    summary=dict(full_results=artifact(OUT/'results.json'),completed_endpoint_count=20,
        rows=[{k:r[k] for k in keep} for r in result['rows']],checks=result['checks'],
        comparison_summaries=result['comparison_summaries'],sources=result['sources'],collector=result['collector'],scope=result['scope'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (OUT/'RESULTS.md').write_text(report(summary))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
