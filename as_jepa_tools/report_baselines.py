"""Publish completed matched25k endpoints with checked source/result identities."""

def main():
    from pathlib import Path
    import json, hashlib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    c = Path(args.campaign)
    out = Path(args.output)
    groups = [('L1', 0, 'matched-L1-standard-mlpbn-seed0'), ('L1', 1, 'L1-mlpbn-25k-seed1'), ('L1', 2, 'L1-mlpbn-25k-seed2'), ('S0', 0, 'endpoint-source-S0-25k-seed0'), ('S0', 1, 'S0-25k-seed1'), ('S0', 2, 'S0-25k-seed2'), ('S1', 0, 'endpoint-source-S1-25k-seed0'), ('S1', 1, 'S1-25k-seed1'), ('S1', 2, 'S1-25k-seed2')]
    rows = []
    for family, seed, parent in groups:
        name = 'eval-' + parent
        run = c / name / 'results'
        exit = json.loads((c / 'exits' / (name + '.json')).read_text())
        status = json.loads((run / 'status.json').read_text())
        assert exit['returncode'] == 0 and status['state'] == 'finished'
        quality = json.loads((run / 'quality/metrics.json').read_text())
        probe = json.loads((run / 'probes/mean/probes.json').read_text())
        cf = json.loads((run / 'counterfactual/counterfactual.json').read_text())
        p0 = []
        for eval_seed in [42, 43, 44]:
            path = run / 'planning' / ('seed' + str(eval_seed)) / 'metrics.json'
            if path.exists():
                p0.append(json.loads(path.read_text()))
        row = dict(family=family, seed=seed, updates=25000, checkpoint_manifest=json.loads((run / 'checkpoint_manifest.json').read_text()), quality=quality, probes=probe, counterfactual=cf, planning=p0, quality_cases=json.loads((run / 'quality/cases.json').read_text()), exit=exit)
        rows.append(row)
        arrays = np.load(run / 'quality/per-case-metrics.npz')
        np.savez_compressed(out / f'baseline-{family}-seed{seed}-quality.npz', **{k: arrays[k] for k in arrays.files})
    l0=[]
    for seed,name in [(0,"eval-matched-L0-standard-cursorfix-seed0"),(1,"eval-matched-L0-standard-seed1"),(2,"eval-L0-25k-seed2")]:
        folder=c/name/"results"
        completed=json.loads((c/"exits"/(name+".json")).read_text())
        assert completed["returncode"]==0 and json.loads((folder/"status.json").read_text())["state"]=="finished"
        l0.append(dict(seed=seed,checkpoint_manifest=json.loads((folder/"checkpoint_manifest.json").read_text()),
                       probes=json.loads((folder/"probes/mean/probes.json").read_text()),
                       planning=[json.loads((folder/"planning"/("seed"+str(value))/"metrics.json").read_text()) for value in [42,43,44]]))
    payload = dict(scope='Nine completed exploratory25k endpoints,same256casegeneration protocol; objectives and architectures differ. Not the original10epochcohort or a causal architecture comparison.', models=rows, l0=l0)
    (out / 'baselines-25k.json').write_text(json.dumps(payload, indent=2))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.bbox': 'tight'})
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {'L1': ('#1976b9', '^', 'L1 MLP-BN'), 'S0': ('#db7b25', 'o', 'Native S0'), 'S1': ('#60519b', 's', 'Native S1')}
    for family, (color, marker, label) in styles.items():
        subset = [x for x in rows if x['family'] == family]
        ax.scatter([x['probes']['agent_r2'] for x in subset], [x['quality']['h7_agent_ratio'] for x in subset], s=65, c=color, marker=marker, label=label)
        for x in subset:
            ax.annotate(str(x['seed']), (x['probes']['agent_r2'], x['quality']['h7_agent_ratio']), xytext=(5, 7 if x['seed'] == 1 else -13), textcoords='offset points', fontsize=9)
    ax.axhline(1, color='#555555', linestyle='--', label='Copy baseline')
    ax.set(xlabel='Agent position probe R2', ylabel='h7 agent position error / copy', xlim=(0.4, 0.9), ylim=(0, 6.7), title='Readable state does not guarantee a useful closed loop\n25k updates; 256 fixed cases; labels are training seed IDs')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(out / ('probe-versus-rollout-25k.' + ext), dpi=180)
    plt.close(fig)
    table = []
    for x in rows:
        q = x['quality']
        p = x['probes']
        rates = [z['metrics']['success_rate'] for z in x['planning']]
        table.append(f'| {x['family']} seed{x['seed']} | {p['agent_r2']:.3f} / {p['block_position_r2']:.3f} | {q['h7_agent_ratio']:.3f} / {q['h7_moving_block_ratio']:.3f} | {q['fvd']['fvd']:.1f} | {x['counterfactual']['action_identification_accuracy'] * 100:.1f} | ' + (f'{np.mean(rates):.1f}' if rates else '—（原规格未要求）') + ' |')
    text = '# 25k匹配终点：probe、生成与控制分别报告\n\n这九个终点都已跑完完整评估，训练为25k更新、生成用同一256例和同一pose/I3D协议。架构、训练目标和每窗口帧数不同，这张表不是最终完整预算下的架构排名。\n\n| 模型 | agent/block位置probe R2 | h7 agent / moving-block相对copy误差 | FVD | 动作识别(%) | P0均值(%) |\n| --- | --- | --- | --- | --- | --- |\n' + '\n'.join(table) + '\n\n![Probe与闭环误差](probe-versus-rollout-25k.png)\n\nNativeS1的两个新增种子的agent probe约.75/.80，但h7误差约为copy的5.97/4.94倍；可线性读取物理位置不保证自由闭环有效。这个结果支持继续验证语义记忆/图像反馈对照，不能单凭它认定某个反馈修复已经成功。\n\nNativeS0 seed2的FVD约1243，优于copy的1629，但h7 agent/moving-block误差比约1.12/1.31，均比copy差。更好的FVD本身也不足以证明动力学或动作控制。L1两个种子的物块生成优于copy，但agent仍近copy，P0分别约27.3%和16.0%。\n\n[完整逐模型数值和每例动作/规划记录](baselines-25k.json)包含模型配置、checkpoint哈希、全部horizon的质量指标、50k帧训练probe、5k帧测试probe及case身份。baseline-*-quality.npz保留逐例数值误差；像素、权重、视频和原始数据均留在EFS。L2的新增训练种子和完整10epoch结果仍在继续。\n'
    summary=["\n三训练种子的描述统计（样本标准差，非完整10epoch结论）：\n",
             "| 模型 | h7 agent/copy mean±SD | moving-block/copy mean±SD | P0 mean±SD across training seeds |",
             "| --- | --- | --- | --- |"]
    for family in ["L1","S0","S1"]:
        subset=[x for x in rows if x["family"]==family]
        agent=np.array([x["quality"]["h7_agent_ratio"] for x in subset])
        block=np.array([x["quality"]["h7_moving_block_ratio"] for x in subset])
        p0=[np.mean([z["metrics"]["success_rate"] for z in x["planning"]]) for x in subset if x["planning"]]
        summary.append(f"| {family} | {agent.mean():.3f}±{agent.std(ddof=1):.3f} | {block.mean():.3f}±{block.std(ddof=1):.3f} | "+(f"{np.mean(p0):.2f}±{np.std(p0,ddof=1):.2f}%" if p0 else "原规格未要求")+" |")
    p0=np.array([np.mean([x["metrics"]["success_rate"] for x in row["planning"]]) for row in l0])
    summary.append(f"| L0 | 无renderer | 无renderer | {p0.mean():.2f}±{p0.std(ddof=1):.2f}% |")
    text+="\n"+"\n".join(summary)+"\n\nL0三训练种子分别先平均各自P042/43/44；表中标准差在训练种子之间计算，没有把9次规划当成9个训练种子。原要求差异接近时的额外seed3/4及L1 SIGReg筛选，留给完整预算比较后按规则处理。\n"
    (out / 'BASELINES_25K.md').write_text(text)
    readme = out / 'README.md'
    s = readme.read_text()
    s += '\n补充：[九个25k匹配终点的完整结果与probe/闭环图](BASELINES_25K.md)。\n'
    readme.write_text(s)
    manifest = json.loads((out / 'manifest.json').read_text())
    manifest['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
    manifest['supplement_scope'] = 'All nine specified25k baseline endpoints exit0 and stage-finished before inclusion.'
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('BASELINES_PACKAGED', len(rows))
if __name__ == '__main__':
    main()
