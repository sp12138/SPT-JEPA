"""CPU-only summary of completed fresh2000 and frozen spatial600 diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(p):
    return dict(path=str(p), sha256=digest(p), bytes=Path(p).stat().st_size)


def same_evaluation(a, b):
    for k in ['manifest_sha256', 'pairs', 'solver', 'plan', 'protocol', 'split']:
        assert a[k] == b[k], k


def compact_scores(x):
    return dict(xyz={k: v for k, v in x['xyz_euclidean'].items() if not isinstance(v, list)},
        orientation={k: v for k, v in x['orientation_angles'].items() if not isinstance(v, list)},
        xyz_r2=x['xyz_features']['variance_weighted_r2'],
        orientation_r2=x['orientation_features']['variance_weighted_r2'],
        joint_fraction=x['joint_readout_within3cm_and15deg_fraction'])


def collect(b):
    root = b/'runs/carrier_fresh2000_20261031'
    plan, protocol = read(root/'plan.json'), read(root/'protocol.json')
    old_path = b/'reports/carrier-diagnostics-20261031/results.json'
    learn_path = b/'reports/carrier-learn2000-20261031/results.json'
    old, learn = read(old_path), read(learn_path)
    assert protocol['input_result_sha256'] == digest(learn_path)
    previous = [r for r in old['rows'] if r['case_set'] == 'fresh' and r['metric'] == 'identity']
    old_metrics = {}
    for r in previous:
        p = Path(r['output'])/'metrics.json'
        assert digest(p) == r['metrics_sha256']
        old_metrics[r['id']] = read(p)
    fresh = []
    for j in plan['jobs']:
        exit_record = read(root/'exits'/(j['id']+'.json'))
        assert exit_record['returncode'] == 0
        path = Path(j['output'])/'metrics.json'
        x = read(path)
        cmd = j['command']
        checkpoint = Path(cmd[cmd.index('--checkpoint')+1])
        manifest = Path(cmd[cmd.index('--manifest')+1])
        assert x['checkpoint'] == str(checkpoint)
        assert checkpoint.name == 'model-step0002000'
        assert digest(checkpoint/'weights.pt') == j['checkpoint_weights_sha256']
        assert digest(checkpoint/'config.json') == j['checkpoint_config_sha256']
        assert digest(manifest) == j['manifest_sha256'] == x['manifest_sha256']
        assert read(manifest)['pairs'] == x['pairs']
        assert x['checkpoint_audit']['strict_equal']
        assert len(x['metrics']['episode_successes']) == 50
        assert abs(x['metrics']['success_rate']-2*sum(x['metrics']['episode_successes'])) < 1e-8
        old_task = [r for r in previous if r['task'] == j['task']]
        for prior in old_task:
            same_evaluation(x, old_metrics[prior['id']])
            if prior['family'] != 'released-lewm':
                assert x['normalization'] == old_metrics[prior['id']]['normalization']
        prior = next(r for r in learn['rows']+learn['baselines'] if r['id'] == j['id'])
        artifacts = prior.get('source_artifacts', prior.get('artifacts'))
        primary = read(artifacts['planning']['path'])
        assert digest(artifacts['planning']['path']) == artifacts['planning']['sha256']
        assert primary['checkpoint'] == x['checkpoint']
        for k in ['plan', 'protocol', 'split', 'normalization']:
            assert primary[k] == x[k], k
        assert {k:v for k,v in primary['solver'].items() if k!='seed'} == {
            k:v for k,v in x['solver'].items() if k!='seed'}
        fresh.append(dict(task=j['task'], family=j['family'], arm=j['arm'], seed=j['training_seed'],
            success_rate=x['metrics']['success_rate'], source=artifact(path),
            checkpoint=str(checkpoint), checkpoint_weights_sha256=j['checkpoint_weights_sha256'],
            checkpoint_config_sha256=j['checkpoint_config_sha256'],
            manifest_sha256=x['manifest_sha256'], policy_seed=x['solver']['seed'],
            compatibility=dict(same_cases_solver_manifest_as_all_previous_fresh_references=True,
                exact_checkpoint_as_primary2000=True, strict_checkpoint_load=True,
                weights_and_config_match_predeclared_hashes=True)))
    assert len(fresh) == 16
    spatial = []
    cohorts = []
    spatial_root = b/'runs/carrier_spatial600_20261031'
    for name in ['extract-seed1','extract-seed2','analyze-seed1','analyze-seed2']:
        assert read(spatial_root/'exits'/(name+'.json'))['returncode'] == 0
    for seed in [1,2]:
        path = b/f'reports/carrier-spatial600-20261031/seed{seed}-readouts/results.json'
        x = read(path)
        extraction_path = Path(x['extraction'])/'extraction.json'
        assert digest(extraction_path) == x['extraction_sha256']
        ext = read(extraction_path)
        assert ext['checkpoint_step'] == 600
        assert [x[k] for k in ['training_clips','training_frames','validation_clips','validation_future_frames']] == [128,1024,64,448]
        assert x['no_validation_tuning'] and x['ridge'] == .001
        train_ep = {w['episode'] for w in ext['manifests']['train']['windows']}
        val_ep = {w['episode'] for w in ext['manifests']['validation']['windows']}
        assert not train_ep & val_ep
        cohorts.append((ext['training_manifest_sha256'],ext['validation_manifest_sha256']))
        ref = next(r for r in previous if r['task']=='cube' and r['arm']=='identity' and r['seed']==seed)
        assert ext['weights_sha256'] == ref['checkpoint_weights_sha256']
        # Extraction records the training-run config; evaluator records the exported model config.
        run_config_path = Path(ext['run'])/'config.json'
        checkpoint_config_path = Path(ref['checkpoint'])/'config.json'
        assert ext['config_sha256'] == digest(run_config_path)
        assert ref['checkpoint_config_sha256'] == digest(checkpoint_config_path)
        run_config, model_config = read(run_config_path), read(checkpoint_config_path)
        assert all(model_config[k] == v for k,v in run_config['model'].items())
        for delta in x['rp64_existing_probe_differences'].values():
            assert abs(delta['invariant_r2_difference']) < 2e-8
            assert abs(delta['mean_angle_difference_deg']) < 1e-8
        rows = []
        for r in x['rows']:
            row = {k:r[k] for k in ['space','feature','dimensions','readout','configuration']}
            row.update(train=compact_scores(r['train']), validation=compact_scores(r['validation']))
            rows.append(row)
        assert len(rows) == 8
        spatial.append(dict(seed=seed, source=artifact(path), extraction=artifact(extraction_path),
            cohort=dict(training_clips=128, training_frames=1024, validation_clips=64,
                        validation_future_frames=448, disjoint_episodes=True),
            rows=rows, constant=compact_scores(x['constant_reference']),
            true_label_decoder=compact_scores(x['true_label_decoder_reference']),
            config_checks=dict(training_run_config=artifact(run_config_path),
                exported_model_config=artifact(checkpoint_config_path), model_parameters_equal=True),
            rp64_existing_probe_differences=x['rp64_existing_probe_differences'], scope=x['scope']))
    assert cohorts[0] == cohorts[1]
    return dict(inputs=dict(fresh_plan=artifact(root/'plan.json'), fresh_protocol=artifact(root/'protocol.json'),
        diagnostics600=artifact(old_path), learn2000=artifact(learn_path)),
        exits=dict(fresh2000=16, spatial600=4, all_returncodes_zero=True),
        fresh2000=fresh, fresh600=[{k:r[k] for k in ['task','family','arm','seed','success_rate']}
                                 for r in previous],
        fresh_scope='Same previously declared50 validation pairs and policy seed2026103101; zero exact-pair overlap with primary, but same validation pool (episode overlap TwoRoom34/Cube1). Released LeWM budget/exposure/capacity/normalization unmatched.',
        spatial600=spatial,
        next_action='Parent reports8F+4TF6000 launched2026-10-31 19:53 (as reported by parent); this summary does not inspect their active results.',
        report_script=artifact(__file__))


def plot(s, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['4×4 · RP64 · linear','4×4 · RP64 · RBF','4×4 · RP192 · linear',
            '4×4 · full3072 · linear','4×4 · full3072 · RBF','16×16 · RP64 · linear',
            '16×16 · RP192 · linear','16×16 · full49152 · linear']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.4),sharey=True)
    for seed_result,color,marker,offset in zip(s['spatial600'],['#0072B2','#D17600'],['o','^'],[-.09,.09]):
        y=[i+offset for i in range(8)]
        scores=[r['validation']['xyz'] for r in seed_result['rows']]
        axes[0].scatter([r['mean_cm'] for r in scores],y,c=color,marker=marker,label=f"Seed {seed_result['seed']}",s=42)
        axes[1].scatter([r['fraction_within3cm']*100 for r in scores],y,c=color,marker=marker,s=42)
    axes[0].set_yticks(range(8),labels);axes[0].invert_yaxis()
    axes[0].axvline(3,color='#8B8793',linestyle='--',lw=1)
    axes[0].text(3.15,-.27,'3 cm per-frame criterion',fontsize=8,color='#746E7B')
    axes[0].set_xlim(0,13);axes[1].set_xlim(0,35)
    axes[0].set_xlabel('Mean xyz error (cm; lower is better)')
    axes[1].set_xlabel('Frames with xyz error ≤3 cm (%)')
    for ax in axes:ax.grid(axis='x',color='#E3E5E8');ax.set_axisbelow(True)
    axes[0].legend(loc='lower right',frameon=False)
    fig.suptitle('Cube: spatial compression and readout both limit position accuracy',fontsize=13,x=.03,ha='left')
    fig.subplots_adjust(left=.26,right=.98,top=.88,bottom=.2,wspace=.12)
    fig.text(.03,.045,'Frozen identity-carrier600; identical episode-disjoint 1024 training / 448 validation frames. No validation tuning.\n'
             'Physical labels train diagnostic readouts only. These frame-level scores are not planning success or the required 3-step hold.',fontsize=9,color='#4F5660')
    for extension in ['png','svg']:
        fig.savefig(out/f'spatial-position.{extension}',dpi=170,facecolor='white')
    plt.close(fig)


def document(s):
    lines=['# Fresh2000与空间读出：完成复核','',
        'TwoRoom的进展在预先固定的fresh病例上保留；Cube仍弱。两项诊断共20个端点全部exit0，未增加训练或查询活动结果。','',
        '| 任务 | F identity | F free | F bounded | TF | released LeWM |','|---|---:|---:|---:|---:|---:|']
    for task in ['tworoom','cube']:
        vals=[]
        for arm in ['identity','unconstrained','bounded',None]:
            rows=sorted([r for r in s['fresh2000'] if r['task']==task and r['arm']==arm],key=lambda r:r['seed'])
            vals.append('/'.join(f"{r['success_rate']:g}" for r in rows))
        ref=next(r['success_rate'] for r in s['fresh600'] if r['task']==task and r['family']=='released-lewm')
        lines.append(f"| {task} | "+' | '.join(vals)+f' | {ref:g} |')
    lines += ['', '数值为成功率%，每格seed1/2；F/TF均2000步，公开LeWM为之前同fresh病例的单checkpoint参考、训练预算不匹配。16项已逐一校验相同有序50病例、manifest、solver（含policy seed）、plan、protocol，以及checkpoint路径、权重/config预声明SHA、严格加载。相对primary2000仅换预声明病例与policy seed；相对旧fresh600参考保持评测协议。内部归一化一致；released保留其full-data归一化。', '',
        'fresh仍来自相同validation pool（与primary精确pair重合0，episode重合TwoRoom34/Cube1），不是独立test。TwoRoom bound−free为−6/+12pp，不能据seed2的90%判定约束胜出。Cube各F为6–12%，TF6/6%，released24%，尚无稳定优势。','',
        '![空间读出位置误差](spatial-position.png)','',
        '| 空间特征 / 读出 | xyz平均误差cm | ≤3cm比例% | 姿态平均误差° | ≤15°比例% |','|---|---:|---:|---:|---:|']
    for i in range(8):
        rows=[x['rows'][i] for x in s['spatial600']]
        name=f"{rows[0]['space']} / {rows[0]['feature']} / {rows[0]['readout']}"
        values=[]
        for group,key,scale in [('xyz','mean_cm',1),('xyz','fraction_within3cm',100),('orientation','mean_deg',1),('orientation','fraction_within15deg',100)]:
            values.append('/'.join(f"{r['validation'][group][key]*scale:.2f}" for r in rows))
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '同RBF读出下，RP64→完整4×4把位置误差7.84/7.88降到5.64/5.65cm；同RP64下，linear→RBF把9.26/9.51降到7.84/7.88cm。两种因素均有作用。RP192线性位置也改善至7.74/7.52cm，故“RP192无改善”仅适用于没有姿态收益，不能泛指所有信息。', '',
        '姿态仍约21–23°，常量参考23.13°、≤15°32.81%；最佳已测读出也仅约39%进入15°。完整patch线性读出位置6.72/6.74cm，但姿态实测21.01/20.87°，仍无明显突破。这些结果支持当前压缩/读出限制了可读位置精度，却不能证明姿态信息已丢失，更不能单独解释规划失败。', '',
        '所有读出固定λ=.001，无验证调参；两seed同128训练clips×8帧、64验证clips×7未来帧，episode互斥。1024训练帧高度相关，维度/统计容量不同；高维失败仍可能涉及样本量、正则或归纳偏好。物理标签仅拟合诊断readout，未进入世界模型训练。阈值比例不是规划成功率，也未验证连续3物理步保持。原RP64线性/RBF姿态结果复现，最大R²差<2e−8、角度差0。','',
        '后续8F+4TF到6000步已由root于19:53启动；此报告未轮询这些运行。完整逐帧/逐病例数组保留EFS，[summary.json](summary.json)仅含标量、兼容性检查和原报告SHA。','']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,default=Path('/efs/prescottsun/SPT-JEPA'))
    a=p.parse_args();out=a.base/'reports/carrier-followup-review-20261031';out.mkdir(parents=True,exist_ok=True)
    summary=collect(a.base)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    (out/'RESULTS.md').write_text(document(summary));plot(summary,out)
    print(json.dumps({'output':str(out),'files':[artifact(x) for x in sorted(out.iterdir())]},indent=2))
