"""All completed Cube endpoints, fixed historical controls and current first-epoch diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(Path(path).read_text())


def main(review, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    review, output = Path(review), Path(output); output.mkdir(exist_ok=False)
    base = Path('/efs/prescottsun/SPT-JEPA'); root = base / 'runs/lingjing_fullspec_recovery_v4_20261102'
    original = base / 'runs/lingjing_fullspec_main_v3_20261102'
    controls = read(review / 'archived-cube-controls.json'); floor = read(review / 'pose-floor-v1.json')
    pose_controls_path = review / 'pose-controls-v6/summary.json'
    pose_controls = read(pose_controls_path) if pose_controls_path.exists() else None
    if pose_controls:
        for row in pose_controls['rows']:
            path = Path(row['command'][row['command'].index('--output') + 1]) / 'pose.json'
            value = read(path); assert value['state_before'] == value['state_after']
            row['frozen_proof'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_sha256=value['checkpoint_sha256'], state_before=value['state_before'], state_after=value['state_after'])
    records = []; pose_rows = []
    for arm in range(1, 7):
        for seed in [1, 2]:
            ident = f'round2-K-K{arm}-s{seed}'; terminal = read(original / 'terminals' / (ident + '.json'))
            assert terminal['passed'] and terminal['nominal_budget_complete']
            audit = read(original / 'audits' / (ident + '-step12000.json'))
            assert audit['passed'] and audit['learning_paths_active']
            folder = original / 'evaluations' / ident / '12000'
            pose = read(folder / 'pose/pose.json'); rbf = next(r for r in pose['rows'] if r['readout'] == 'rbf')
            pose_rows.append(dict(id=ident, arm='K' + str(arm), seed=seed, conditions=rbf['conditions']))
            for controller in ['white', 'correlated']:
                for case in ['primary', 'fresh']:
                    path = folder / (controller + '-' + case) / 'metrics.json'; m = read(path)
                    ref = next(r for r in controls['rows'] if r['family'] == 'native-raw' and r['seed'] == seed and r['case_set'] == case and r['controller'] == controller)
                    assert m['manifest_sha256'] == ref['manifest_sha256'] and len(m['metrics']['episode_successes']) == 50
                    records.append(dict(arm='K' + str(arm), seed=seed, controller=controller, case_set=case,
                        percent=m['metrics']['success_rate'], successes=m['metrics']['episode_successes'],
                        manifest_sha256=m['manifest_sha256'], parameters=m['checkpoint_audit']['parameters'],
                        source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(records) == 48
    means = []
    for arm in ['K' + str(k) for k in range(1, 7)]:
        values = {c + '-' + s: float(np.mean([r['percent'] for r in records if r['arm'] == arm and r['controller'] == c and r['case_set'] == s])) for c in ['white', 'correlated'] for s in ['primary', 'fresh']}
        values.update(arm=arm, xyz_cm=float(np.mean([r['conditions']['observed_future']['xyz_cm'] for r in pose_rows if r['arm'] == arm])),
            pose_deg=float(np.mean([r['conditions']['observed_future']['pose_deg'] for r in pose_rows if r['arm'] == arm])))
        means.append(values)
    contrasts = []
    for a, b in [('K1', 'K2'), ('K3', 'K2'), ('K1', 'K4'), ('K5', 'K1'), ('K6', 'K1')]:
        for controller in ['white', 'correlated']:
            for case in ['primary', 'fresh']:
                def get(arm, seed):return np.asarray(next(r['successes'] for r in records if r['arm'] == arm and r['seed'] == seed and r['controller'] == controller and r['case_set'] == case), dtype=float)
                delta = np.stack([get(a, seed) - get(b, seed) for seed in [1, 2]])
                bank = np.random.default_rng(20261102).integers(50, size=(10000, 50))
                distribution = 100 * delta.mean(0)[bank].mean(1)
                contrasts.append(dict(a=a, b=b, controller=controller, case_set=case, delta_pp=100 * float(delta.mean()),
                    per_seed_pp=(100 * delta.mean(1)).tolist(), conditional_case95ci=np.quantile(distribution, [.025, .975]).tolist()))
    gates = []
    for path in sorted((root / 'evaluations').glob('*/12491/stage-evidence.json')):
        value = read(path); ident = path.parents[1].name; latent = path.parent / 'latent/latent-eval.json'
        if not latent.exists():continue
        l = read(latent)
        gates.append(dict(id=ident, gate=value['gate'], mean_probes=value['probes'].get('mean'), noisy_probes=value['probes'].get('noisy'),
            rollout=l['rollout']['errors'], moving_block_h7_ratio=l['rollout']['h7_moving_block_ratio'],
            counterfactual_accuracy=l['counterfactual']['accuracy'] if l.get('counterfactual') else None,
            information={k:v for k,v in l['information'].items() if k not in ['records', 'eigenvalues_over_sigma2', 'nats_per_direction']},
            noisy_pose_r2=l['probes']['noisy_r2'], source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(cube_complete48=True, cube=records, cube_means=means, cube_pose=pose_rows, prespecified_contrasts=contrasts,
        archived_controls=controls, pose_floor=floor, pose_controls=pose_controls, first_epoch_gates=gates,
        scope='All seeds/controllers/cases retained. Historical controls are not contemporary simulator repeats or equal model compute. Conditional intervals exclude training-seed and environment-repeat uncertainty;no multiplicity correction. Partial first-epoch gates are not3epoch winners or failed nominal budgets.')
    (output / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    labels = ['Native', 'TF'] + ['K' + str(k) for k in range(1, 7)]
    for ax, controller in zip(axes[:2], ['white', 'correlated']):
        for case, offset, colour in [('primary', -.18, '#28689d'), ('fresh', .18, '#d88039')]:
            values = []; vectors = []
            for label in labels:
                vec = [r['percent'] for r in controls['rows'] if r['family'] == ('native-raw' if label == 'Native' else 'TF64') and r['controller'] == controller and r['case_set'] == case] if label in ['Native', 'TF'] else [r['percent'] for r in records if r['arm'] == label and r['controller'] == controller and r['case_set'] == case]
                assert len(vec) == 2; values.append(float(np.mean(vec))); vectors.append(vec)
            x = np.arange(8) + offset; ax.bar(x, values, width=.32, color=colour, label=case, alpha=.85)
            for seed, marker in [(0, 'o'), (1, 's')]:ax.scatter(x + (-.035 if seed == 0 else .035), [v[seed] for v in vectors], color='#202020', marker=marker, s=15, zorder=3)
        ax.set(xticks=np.arange(8), xticklabels=labels, ylim=(0, 48), ylabel='Strict task success (%)', title='Cube12000: ' + controller + ' CEM')
        ax.grid(axis='y', alpha=.16); ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    pose_labels = labels if pose_controls else labels[2:]
    for seed, marker, colour in [(1, 'o', '#28689d'), (2, 's', '#d88039')]:
        values = []
        if pose_controls:
            for family in ['native-raw', 'TF64']:
                row = next(r for r in pose_controls['rows'] if r['family'] == family and r['seed'] == seed)
                values.append(next(x['conditions']['observed_future']['pose_deg'] for x in row['readouts'] if x['readout'] == 'rbf'))
        values += [next(r['conditions']['observed_future']['pose_deg'] for r in pose_rows if r['arm'] == 'K' + str(k) and r['seed'] == seed) for k in range(1, 7)]
        ax.scatter(np.arange(len(pose_labels)) + (-.08 if seed == 1 else .08), values, marker=marker, color=colour, label='training seed ' + str(seed))
    ax.axhline(floor['rows'][0]['constant_training_mean']['mean_deg'], linestyle='--', color='#666666', label='constant readout')
    ax.axhline(floor['rows'][0]['oracle_true_invariant']['mean_deg'], linestyle=':', color='#4b906a', label='true-feature codebook floor')
    ax.set(xticks=np.arange(len(pose_labels)), xticklabels=pose_labels, ylim=(0, 27), ylabel='Cube-symmetry angle error (deg)', title='Fixed RBF pose readout: observed state')
    ax.legend(frameon=False, fontsize=8, loc='center right'); ax.grid(axis='y', alpha=.16)
    fig.suptitle('State use survived; goal precision and control did not follow', fontsize=13)
    for suffix in ['png', 'svg', 'pdf']:fig.savefig(output / ('cube12000-insights.' + suffix), dpi=180)
    plt.close(fig)
    path = output / 'cube12000-insights.svg';path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    print(json.dumps(dict(cube_complete48=True, means=means, first_epoch_gates=len(gates), output=str(output))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--review', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); main(a.review, a.output)
