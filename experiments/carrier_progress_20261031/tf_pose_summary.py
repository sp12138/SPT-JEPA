"""Summarize completed TF/F pose probes; bootstrap saved errors only, no refitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

B=Path('/efs/prescottsun/SPT-JEPA')
R=B/'reports/tf6000-pose-20261031'
MODES=('observed_future','clean_rollout','teacher_forced_prefix','initial_carry','shifted_observed_carry')
NBOOT=10000


def read(p):return json.loads(Path(p).read_text())


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def artifact(p):return dict(path=str(p),sha256=digest(p),bytes=Path(p).stat().st_size)


def verified(record):
    assert digest(record['path'])==record['sha256'],record['path']
    return record


def errors(scores):
    xyz=np.asarray(scores['xyz_euclidean']['per_frame_cm']).reshape(64,7)
    pose=np.asarray(scores['orientation_angles']['per_frame_deg']).reshape(64,7)
    arrays=dict(xyz_error_cm=xyz,pose_error_deg=pose,within3cm_pp=100*(xyz<=3),
        within15deg_pp=100*(pose<=15),joint_frame_pp=100*((xyz<=3)&(pose<=15)))
    assert np.isfinite(xyz).all() and np.isfinite(pose).all()
    expected=[scores['xyz_euclidean']['mean_cm'],scores['orientation_angles']['mean_deg'],
        scores['xyz_euclidean']['fraction_within3cm']*100,
        scores['orientation_angles']['fraction_within15deg']*100,
        scores['joint_readout_within3cm_and15deg_fraction']*100]
    for value,target in zip(arrays.values(),expected):assert abs(value.mean()-target)<1e-10
    return arrays


def interval(window_delta,draws):
    values=np.asarray(window_delta)
    assert values.shape==(64,)
    samples=values[draws].mean(1)
    return dict(difference=float(values.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())


def collect():
    run=B/'runs/tf6000_pose_20261031';plan=read(run/'plan.json');source=read(run/'source-record.json')
    assert len(plan['jobs'])==2 and source['source_commit'].startswith('6f6d513')
    assert (run/'HOLD_DISPATCH').exists()
    exits=[]
    for job in plan['jobs']:
        x=read(run/'exits'/(job['id']+'.json'));assert x['returncode']==0
        assert job['source']['source_commit']==source['source_commit'];exits.append(x)
    manifest=read(source['manifest']);summaries=[];full={};proof=[];cohorts=[]
    for seed in [1,2]:
        out=R/f'seed{seed}-readouts';s=read(out/'summary.json');d=read(out/'results.json')
        verified(s['full_results']);verified(d['extraction']);e=read(d['extraction']['path'])
        for rec in [e['source'],e['weights'],e['config'],e['features'],d['source'],d['pinned_task_source']]:verified(rec)
        assert e['source']['sha256']==manifest['files']['experiments/carrier_progress_20261031/tf_pose.py']
        assert e['model_state_unchanged'] and e['old_F_encoder_calls']==d['old_F_model_calls']==0
        assert e['frames_encoded']==1536 and d['training_frames']==1024 and d['validation_frames']==448
        assert d['no_validation_tuning'] and e['source_contract']['source_commit'].startswith('8e0f821')
        original_source=read(e['source_contract']['source_record']['path'])
        for name,item in e['source_contract']['verified_files'].items():
            sha=item if isinstance(item,str) else item['sha256']
            assert digest(Path(original_source['source'])/name)==sha
            if isinstance(item,dict):assert digest(item['loaded_path'])==sha
        for rec in e['reference_provenance'].values():verified(rec)
        cohorts.append((e['reference_provenance']['training_manifest']['sha256'],e['reference_provenance']['validation_manifest']['sha256']))
        assert max(abs(delta) for row in d['old_F_observation_reproduction'].values() for delta in row.values())<2e-5
        summaries.append(s);full[seed]=d
        proof.append(dict(seed=seed,full_results=s['full_results'],extraction=d['extraction'],
            source=e['source'],weights=e['weights'],config=e['config'],source_contract=e['source_contract'],
            reference_provenance=e['reference_provenance'],pinned_task_source=d['pinned_task_source'],
            model_weights_and_buffers_unchanged_during_extraction=True,current_weight_file_sha_matches=True,
            old_F_model_calls=0,F_observed600_equals6000=True,
            F_observed_readout_reproduction=d['old_F_observation_reproduction']))
    assert cohorts[0]==cohorts[1]
    draws=np.random.default_rng(1042).integers(64,size=(NBOOT,64))
    paired=[];windows={};within=[]
    for seed in [1,2]:
        lookup={(r['model'],r['readout']):r for r in full[seed]['rows']}
        for kind in ['linear','rbf']:
            for mode in MODES:
                a=errors(lookup['TF6000',kind]['conditions'][mode]);b=errors(lookup['Fidentity6000',kind]['conditions'][mode])
                delta={key:(a[key]-b[key]).mean(1) for key in a}
                windows[seed,kind,mode]=delta
                paired.append(dict(seed=seed,readout=kind,condition=mode,contrast='TF minus F',
                    metrics={key:interval(value,draws) for key,value in delta.items()}))
            for model in ['TF6000','Fidentity6000']:
                for reference in ['observed_future','initial_carry']:
                    a=errors(lookup[model,kind]['conditions']['clean_rollout']);b=errors(lookup[model,kind]['conditions'][reference])
                    within.append(dict(seed=seed,model=model,readout=kind,contrast='clean_rollout minus '+reference,
                        metrics={key:interval((a[key]-b[key]).mean(1),draws) for key in a}))
    aggregate=[]
    for kind in ['linear','rbf']:
        for mode in MODES:
            aggregate.append(dict(readout=kind,condition=mode,contrast='TF minus F',
                metrics={key:interval((windows[1,kind,mode][key]+windows[2,kind,mode][key])/2,draws)
                         for key in windows[1,kind,mode]}))
    return dict(source_commit=source['source_commit'],completed_jobs=2,exits=exits,
        closed_dispatcher=artifact(run/'closed-dev2.json'),held=True,proof=proof,
        results=[dict(seed=s['seed'],rows=s['rows'],constant=s['constant'],codebook=s['codebook']) for s in summaries],
        paired_seed_intervals=paired,paired_two_fixed_seed_intervals=aggregate,within_model_intervals=within,
        bootstrap=dict(repetitions=NBOOT,seed=1042,resampling_unit='64 paired clips; all7 correlated future frames stay together',
            aggregate='Same clip indices jointly resampled across two fixed training seeds.',
            limitations='Conditional on trained models and fitted readouts; no training/probe-refit uncertainty, seed-population inference or multiplicity correction.'),
        interpretation='The TF position-accessibility advantage is already present in observed states. Both pose readouts remain weak under these fixed probes. '
            'No causal mediation, information-absence or chance-ceiling claim; framewise joint threshold is not sustained task success.',
        collector=artifact(__file__))


def document(s):
    def row(seed,model,kind):
        return next(r for v in s['results'] if v['seed']==seed for r in v['rows'] if r['model']==model and r['readout']==kind)
    lines=['# Cube TF6000：位置可读性优势已存在于观测状态','',
        '此前将重点放在姿态不足，解释不完整。本次同帧、同64维、同读出检验表明：TF相对固定F的优势已经体现在**观测位置可读性**，不只发生在未来预测阶段；两者的姿态读出在固定linear/RBF探针下都仍弱。该结果不证明位置可读性因果中介了规划增益。','',
        '| 条件 / 读出 | TF xyz cm | F xyz cm | TF ≤3cm % | F ≤3cm % | TF姿态° | F姿态° |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ['observed_future','clean_rollout']:
        for kind in ['linear','rbf']:
            values=[]
            for model,key,scale in [('TF6000','xyz_cm',1),('Fidentity6000','xyz_cm',1),
                ('TF6000','xyz_within3cm',100),('Fidentity6000','xyz_within3cm',100),
                ('TF6000','pose_deg',1),('Fidentity6000','pose_deg',1)]:
                values.append('/'.join(f"{row(seed,model,kind)['conditions'][mode][key]*scale:.2f}" for seed in [1,2]))
            lines.append('| '+mode+' / '+kind+' | '+' | '.join(values)+' |')
    lines+=['','所有斜线均为seed1/2。训练探针使用原128clips×8帧，验证为episode互斥的64clips×7未来帧；无验证调参。linear与RBF均固定λ=.001，RBF带宽仅由训练距离中位数确定，使用原训练quat codebook和固定cube24角度规则。','',
        '| TF−F，均值配对差 | xyz cm [95%区间] | ≤3cm pp [95%区间] | 姿态° [95%区间] |',
        '|---|---:|---:|---:|']
    for x in s['paired_two_fixed_seed_intervals']:
        if x['condition'] not in ['observed_future','clean_rollout']:continue
        values=[]
        for key in ['xyz_error_cm','within3cm_pp','pose_error_deg']:
            a=x['metrics'][key];lo,hi=a['ci95'];values.append(f"{a['difference']:+.2f} [{lo:+.2f},{hi:+.2f}]")
        lines.append('| '+x['condition']+' / '+x['readout']+' | '+' | '.join(values)+' |')
    lines+=['','负xyz/姿态差表示TF误差更小；正≤3cm差表示TF更多帧进入位置阈值。区间按64个配对clip重采样10000次、保留每clip七帧相关性，两固定训练seed使用同一抽样索引。逐seed区间及全部五条件、两种读出保留于summary.json；未涵盖模型/探针重训或多重比较不确定性。','',
        '姿态常量参考为23.13°、≤15°32.81%，这是观测到的参考而非“chance ceiling”；两模型缺乏稳定姿态优势，不等于没有姿态信息。部分clean预测的位置读出误差低于observed，可能涉及平滑/条件均值与读出几何，不能把该差直接当作物理动力学精度或因果分解。','',
        '两条job均exit0，冻结分析源6f6d513；模型/数据/clean-forecast字节核对原8e。每个TF仅编码1536帧，提取前后全部序列化weights/buffers逐tensor相等，权重文件SHA复核一致，无BN校准或优化器更新。F600训练/观测与保存F6000预测完全复用，0次F模型调用；旧F观测分数再现的最大标量差<5e−9。来源、权重、原报告SHA均见summary.json。','',
        '物理标签只训练诊断readout。帧级位置≤3cm且姿态≤15°的联合比例**不等于**实际规划成功或连续3物理步保持。完整误差数组保留EFS原报告；本总结只做既有误差CPU重采样，未新增assay、GPU、图或文献搜索。','']
    return '\n'.join(lines)


if __name__=='__main__':
    s=collect();out=R/'review';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
    (out/'RESULTS.md').write_text(document(s))
    print(document(s));print('Artifacts:',[(p.name,p.stat().st_size) for p in out.iterdir()])
