"""Summarize completed span results only; no model or cache computation."""
import hashlib,json
from pathlib import Path
B=Path('/efs/prescottsun/SPT-JEPA')
root=B/'runs/predictor_span_20261031';out=B/'reports/predictor-span-20261031'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=read(out/'results.json');source=read(root/'source-record.json');finish=read(root/'exit.json')
rows=[]
for x in r['rows']:
    row={k:v for k,v in x.items()if k not in ['clean','one_step']}
    for mode in ['clean','one_step']:
        if mode in x:row[mode]={k:v for k,v in x[mode].items()if not k.startswith('per_window')}
    rows.append(row)
summary=dict(passed=r['passed'],full_results=str(out/'results.json'),full_results_sha256=sha(out/'results.json'),
    helper_commit=source['source_commit'],helper_sha256=r['helper_sha256'],protocol_sha256=r['protocol_sha256'],
    plan_sha256=r['plan_sha256'],exit=finish,model_calls=0,cuda_initialized=False,
    cohort='Same64 validation windows×8frames per task,18 existing caches/20checkpoints;bothseeds retained',
    source_commits=sorted(set(v['source']['source_commit']for v in r['source_provenance'].values())),
    observation_checks=r['observation_checks'],rows=rows,scope=r['scope'],
    conclusions=['Current local/basis1024 output has effective affine rank191, verified with cached forecasts.',
        'Local6000 affine oracle explains50.1–51.0% of Cube clean MSE, versus11.4–14.2% forTwoRoom.',
        'True endpoint change outside the affine subspace is11.1–11.6% forCube,1.09–1.21% forTwoRoom.',
        'Basis2000 lowers totalMSE but increases the affine oracle in both tasks/seeds.',
        '64D controls have full effective rank64; full rank does not ensure task success.',
        'Bounds freeze current output weights, not all retrained models; no physical or planning mediation claim.'])
with(out/'summary.json').open('x')as f:json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
index={x['id']:x for x in r['rows']}
lines=['# Predictor输出子空间：CPU机制审计','',
    '限制真实存在，并构成Cube当前1024维预测误差的重要部分；它不能独立解释规划成败。全部20个旧checkpoint、18份现有缓存核验通过；两任务各使用相同64个validation窗口×8帧，保留全部两seed。没有新编码、网络forward、物理标签拟合、GPU或训练。','',
    '冻结32fad/da/8e的模型使用carry，native pred_proj为Identity，最终输出是LayerNorm(192)→Linear(192,D)，之后无非线性；selected Robust/spatial/carrier类无predict覆盖。设W、b是输出参数，γ、β是LayerNorm参数，H=I−11ᵀ/192，则V=W diag(γ)H、c=Wβ+b。每次增量属于c+range(V)，k步预测属于u₀+k c+range(V)，有效秩≤191，比宽松span(W,b)≤193更紧。实测全部12个local/basis的秩W、[W,b]、V分别192、193、191。FP32缓存离精确仿射空间max残差仅2.19e−6–6.25e−6；误差正交分解相对偏差<2.9e−9。','',
    '当前输出层固定时，真实终点的不可达oracle为 ‖(I−P_V)(u_k−u₀−k c)‖²/D；one-step替换为当前真实u_(k−1)和c。下表均为seed1/seed2；比例由总能量相除，不平均不稳定的逐窗口比率。','',
    '| 任务 / checkpoint | Clean MSE | Oracle MSE | Oracle / clean误差 | Oracle / 真实变化 | Oracle / one-step误差 |',
    '|---|---:|---:|---:|---:|---:|']
for task in ['tworoom','cube']:
    for family,step in [('local',2000),('local',6000),('basis',2000)]:
        pair=[index[f'{family}-{step}-{task}-seed{s}']for s in [1,2]];vals=[]
        for mode,key,pct in [('clean','prediction_mse',False),('clean','oracle_mse',False),
            ('clean','oracle_over_error',True),('clean','oracle_over_true_change',True),('one_step','oracle_over_error',True)]:
            vals.append(' / '.join(('%0.2f%%'%(100*x[mode][key])if pct else '%0.5f'%x[mode][key])for x in pair))
        lines.append(f'| {task} {family}{step} | '+' | '.join(vals)+' |')
lines += ['',
    'Cube local6000约一半clean误差处在当前子空间外，但占真实变化仅11.59/11.09%；不能以1024−191维声称丢失81%的任务信息。TwoRoom6000对应真实变化仅1.21/1.09%，第7步oracle只占预测误差8.81/6.79%；其主要剩余误差仍在可输出子空间内。','',
    '实证反例：local从2000→6000秩完全不变，TwoRoom oracle由.0573/.0676降到.0195/.0176，Cube由.2150/.2059降到.1503/.1427，说明子空间会学习对齐。basis2000降低总MSE，却提高oracle：更好的子空间内拟合可能掩盖更差的输出方向覆盖；但这仍不证明basis的任务下降由此造成。','',
    '| 64维参考6000 | 有效秩seed1/2 | Oracle / clean误差seed1/2 |','|---|---:|---:|']
for task in ['tworoom','cube']:
    for family in ['F64','TF64']:
        pair=[index[f'{family}-6000-{task}-seed{s}']for s in [1,2]]
        values=' / '.join('%.2e'%x['clean']['oracle_over_error']if 'clean'in x else '无既存cache'for x in pair)
        lines.append(f'| {task} {family} | 64 / 64 | {values} |')
lines += ['',
    '64维参考没有这一固定线性输出秩障碍，但表征、训练自由度和原始MSE尺度不同，不能跨家族直接比较MSE。TwoRoom TF本身规划弱，满秩显然不是充分条件。','',
    '固定初态、目标g和horizon时，native欧氏goal cost可分为可输出方向的代价，加上 ‖(I−P_V)(u₀+k c−g)‖²。后者与候选动作无关，这些目标方向无法经模型cost指导动作选择；但它不直接证明排序错误，也可能主要对应任务无关变化。没有可复用的local1024固定物理probe系数，本次未新增拟合或声称姿态/接触分量丢失。','',
    '边界：这是**当前checkpoint**的必要可达下界，允许任意隐藏向量，忽略真实动作、attention与LayerNorm半径的额外限制；不是充分动力学模型，也不是跨训练所有W的架构不可解定理。local2000/6000观测逐元素相同；basis与固定Q旋转max误差≤2.39e−7。独立CPU toy验证低秩但oracle为零、正交目标有正下界及Pythagorean分解。没有实施新预测器或作创新/规划因果声明。','',
    '源码定位：冻结local32fad和basis da的as_jepa_lewm/model.py:81、84、120；native stable_worldmodel/wm/lewm/lewm.py:21、43；module.py:139、154、173、285。Dropout只在输出前，原audit eval关闭。','',
    'Helper `%s`；计划B/runs/predictor_span_20261031/plan.json；完整结果B/reports/predictor-span-20261031/results.json（SHA `%s`）。exit0，wall11.86秒，矩阵分析5.50秒，CUDA未初始化。summary.json保留全部seed、horizon与结果hash，完整输入/源hash留在EFS。'%(source['source_commit'],sha(out/'results.json'))]
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(summary_bytes=(out/'summary.json').stat().st_size,report_bytes=(out/'RESULTS.md').stat().st_size,
    full_results_sha256=sha(out/'results.json'))))
