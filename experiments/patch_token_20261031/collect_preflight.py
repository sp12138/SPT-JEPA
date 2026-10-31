"""Completed-only CPU collection; no forward, training, launch, or GPU call."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def predictor_hash(weights):
    h=hashlib.sha256()
    for key in sorted(k for k in weights if k.startswith('predictor.') and k!='predictor.coordinate_basis'):
        h.update(key[len('predictor.'):].encode());h.update(weights[key].contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);root=Path(args.root);out=Path(args.output)
    plan=read(root/'plan.json');assert len(plan['jobs'])==16
    exits=[read(root/'exits'/(j['id']+'.json')) for j in plan['jobs']]
    assert all(e['returncode']==0 for e in exits)
    assert (root/'HOLD').exists() and (root/'HOLD_DISPATCH').exists()
    closed=read(root/'closed.json');assert closed['all16_actual_exit0'] and closed['owned_gpus_empty']
    source=read(root/'source-record.json');assert source['source_commit']=='1018ebbe9beb3e5f9b7c91018f03ff8a4f17884d'
    for kind in ['manifest','runtime_manifest']:
        manifest=read(source[kind])
        for name,digest in manifest['files'].items():assert sha(Path(manifest['source'])/name)==digest
    cpu=read(Path(source['source'])/'experiments/patch_token_20261031/cpu/production.json')
    initial_hash={(r['task'],r['seed']):r['predictor_sha256'] for r in cpu['initial']}
    rows=[]
    for job in plan['jobs']:
        if job['phase']!='train2':continue
        run=Path(job['output']);cfg=read(run/'config.json');status=read(run/'status.json')
        assert sha(run/'config.json')==job['config_sha256']
        assert status['state']=='finished' and status['step']==2
        native_path=root/('fullstate-'+run.name)/'preflight.json'
        extra_path=native_path.with_name('patch-audit.json');native=read(native_path);extra=read(extra_path)
        assert native['passed'] and extra['passed'] and all(native['checkpoint']['checks'].values())
        assert extra['nonpredictor_initial_tensors_exact']==466 and extra['original_initial_tensors']==553
        assert extra['paired_parameter_initial_tensors_exact']==542
        assert extra['additional_backward_calls']==0 and extra['optimizer']['sampler_cursor']==256
        assert extra['native_report_sha256']==sha(native_path)
        for path,digest in extra['binding'].items():assert sha(path)==digest,path
        assert extra['optimizer']['native_full_state_roundtrip']
        assert extra['parameter_counts']['predictor']==523536
        weights=torch.load(run/'initial-model/weights.pt',map_location='cpu',weights_only=True,mmap=True)
        digest=predictor_hash(weights);assert digest==initial_hash[(job['task'],job['training_seed'])]
        predictor_parameter_tensors=sum(n.startswith('predictor.') for n in extra['gradient']['parameter_names'])
        assert predictor_parameter_tensors==87
        assert len(weights)==466+predictor_parameter_tensors+(job['arm']=='orthogonal')
        del weights
        gradient=extra['gradient']['patch_groups'];assert all(v['passed'] for v in gradient.values())
        parameters=extra['gradient']['predictor_per_tensor']
        assert all(v['gradient_present'] and v['finite'] for v in parameters.values())
        rows.append(dict(id=run.name,task=job['task'],seed=job['training_seed'],arm=job['arm'],
            actual_step=2,loss=status['loss'],peak_gib=status['peak_gib'],
            native_gradient_norms={k:v['gradient_norm'] for k,v in native['gradients'].items()},
            patch_gradient_norms={k:v['gradient_norm'] for k,v in gradient.items()},
            zero_norm_predictor_tensors=[k for k,v in parameters.items() if v['norm']==0],
            frozen_tensors_exact=native['frozen_tensors_unchanged']['compared_tensors'],
            nonpredictor_initial_tensors_exact=466,paired_parameter_tensors_exact=542,
            predictor_initial_sha256=digest,optimizer=extra['optimizer'],
            portable_sha256=native['weights_sha256'],fullstate_sha256=native['checkpoint_sha256'],
            native_report=str(native_path),native_report_sha256=sha(native_path),
            patch_report=str(extra_path),patch_report_sha256=sha(extra_path)))
    assert len(rows)==8
    result=dict(passed=True,source=source,root=str(root),completed=16,failed=0,real_updates_per_run=2,
        all_8_native_optimizers_restored=True,all_8_production_initial_predictors_match_CPU=True,
        all_nonpredictor_initial_tensors_exact=True,all_seeds_preserved=True,rows=rows,closed=closed,
        collector_sha256=sha(__file__),CPU_only=True,GPU_initialized=torch.cuda.is_initialized(),
        scope='Technical memory/gradient/checkpoint success only. No600 authorization, no planning/learning/novelty claim.',
        runtime_RNG='Flat vs token dropout draw counts differ; no runtime per-draw equivalence claimed. Paired token initial loss/common grads validated separately.')
    assert not result['GPU_initialized']
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Patch token：两步技术预检','',
        '16/16实际exit0；8组各执行原生两步，随后严格fullstate/FP32梯度审计全部通过。调度器空闲后关闭并HOLD。未启动600训练。','',
        '|task / seed|arm|native loss at2|peak GiB|predictor grad|action encoder grad|',
        '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["task"]} / {r["seed"]}|{r["arm"]}|{r["loss"]:.4g}|{r["peak_gib"]:.2f}|{r["native_gradient_norms"]["predictor"]:.4g}|{r["native_gradient_norms"]["action_encoder"]:.4g}|')
    lines+=['','466个原nonpredictor初始张量、542个成对parameter张量（含冻结参数）、原校准/source/config及portable/fullstate严格核验；原生模型/AdamW/scheduler恢复和sampler cursor256通过。predictor全部分层条件/注意力/MLP梯度组实际非零且有限，未添加额外backward。',
        '', '每训练seed的predictor初值与CPU证据逐字节一致，seed1/2不同。head零初始化的早期梯度延迟不是失败；两步通过也不是功能性预测/规划成功。保留原日志的NFS finalizer警告与真实exit，不以日志stage_complete代替OS exit。',
        '', '唯一批准范围到此结束。完整逐seed/分层梯度、参数状态、hash和关闭记录见summary.json；原始checkpoint与报告留在EFS。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,endpoints=16,peak_gib=[r['peak_gib'] for r in rows])))


if __name__=='__main__':main()
