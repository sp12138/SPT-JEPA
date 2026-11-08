"""One read-only experiment inspection; no timer,launcher,or NN import."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path('/efs/prescottsun/SPT-JEPA')
HARD = ['bin-picking-v3', 'door-open-v3', 'drawer-open-v3', 'peg-insert-side-v3', 'pick-place-v3', 'push-v3']


def read(path):
    return json.loads(path.read_text())


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    result = dict(checked_utc=now.isoformat())
    for key, name in [('original', 'metaworld_baselines_native_20261108'), ('scaling', 'metaworld_baseline_scaling_20261108')]:
        root = BASE / 'runs' / name
        summary = read(root / 'results/summary.json')
        jobs = {j['name']: j for j in read(root / 'plan.json')['jobs']}
        rows = []
        for item in summary['records']:
            health = item.get('health', {})
            epochs = {}
            for epoch, score in item.get('epochs', {}).items():
                groups = score['groups']
                epochs[epoch] = dict(macro50=score['macro50'], episodes=score['episodes'],
                    hard6success=sum(groups.get(t, {}).get('successes', 0) for t in HARD),
                    hard6episodes=sum(groups.get(t, {}).get('episodes', 0) for t in HARD))
            out = Path(jobs[item['name']]['output'])
            proofs = {n: read(out / n) for n in ['PREFLIGHT_PASSED.json', 'error.json'] if (out / n).exists()}
            rows.append(dict(name=item['name'], complete=item['complete'], error=item['error'],
                health_step=health.get('step'), health_utc=health.get('utc'), loss=health.get('loss'),
                lrs=health.get('lrs'), gradients=health.get('gradients'), epochs=epochs, proofs=proofs))
        result[key] = dict(summary_utc=summary['utc'], complete=summary['complete'], errors=summary['errors'],
            claims=len(list((root / 'claims').glob('*.json'))), rows=rows)
    native = read(BASE / 'runs/metaworld_baselines_20261107/native-controller/summary.json')
    result['native'] = {k: native.get(k) for k in ['utc', 'completed_shards', 'errors', 'episodes', 'models']}
    pilot = BASE / 'runs/metaworld_native100_pilot_20261108'
    if pilot.exists():
        result['native_pilot'] = {n: read(pilot / n) for n in ['OWNERSHIP.json', 'progress.json', 'summary.json', 'ERROR.json', 'CLOSED.json'] if (pilot / n).exists()}
        result['native_pilot']['formal_trace_files'] = len(list(pilot.glob('*/formal-*/*.npz')))
        result['native_pilot']['technical_completed_reports'] = len(list(pilot.glob('*/technical-*/metrics.json')))
    probe = BASE / 'runs/metaworld_baseline_insights_20261108/scene-probe-v1'
    if probe.exists():
        result['scene_probe'] = {}
        for name in ['smwm-seed1', 'h-jepa-seed2', 'smwm-seed1-epoch200', 'h-jepa-seed2-epoch200']:
            out = probe / name
            p = {n: read(out / n) for n in ['progress.json', 'FROZEN.json', 'ERROR.json'] if (out / n).exists()}
            source = out / ('summary-v2.json' if (out / 'summary-v2.json').exists() else 'summary.json')
            if source.exists():
                summary = read(source)
                compact = {f: {s: {g: {k: v for k, v in m.items() if k != 'task_rmse_native_units'}
                    for g, m in r[s].items()} for s in ['offline_test', 'online_initial']} for f, r in summary['results'].items()}
                p['summary.json'] = dict(passed=summary['passed'], effective_source=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), results=compact)
            result['scene_probe'][name] = p
    for label, dirname in [('support_loan','metaworld_control_support_20261108'),('geometry_loan','metaworld_geometry_cost_20261108')]:
        root = BASE / 'runs' / dirname
        if root.exists():
            result[label] = {n: read(root / n) for n in ['OWNERSHIP.json','GPU_LOAN_READY.json','progress.json','summary.json','ERROR.json','CLOSED.json'] if (root / n).exists()}
            result[label]['formal_trace_files'] = len(list(root.glob('*/*/formal-*/*.npz')))
            result[label]['technical_completed_reports'] = len(list(root.glob('*/*/technical-*/metrics.json')))
    t2g = BASE / 'runs/metaworld_t2g_adapted_20261108'
    if t2g.exists():
        result['t2g_adapted'] = {}
        for name in ['smwm-seed1', 'h-jepa-seed2']:
            out = t2g / name
            status = {n: read(out / n) for n in ['progress.json', 'FROZEN.json', 'ERROR-encode.json', 'ERROR-fit.json'] if (out / n).exists()}
            if (out / 'training.json').exists():
                status['last_training_epoch'] = read(out / 'training.json')['epochs'][-1]
            if (out / 'summary.json').exists():
                complete = read(out / 'summary.json')
                status['summary'] = {k: complete[k] for k in ['passed', 'validation_before', 'validation', 'heldout', 'head_optimizer_updates', 'worldmodel_optimizer_updates']}
            result['t2g_adapted'][name] = status
    dest = BASE / 'reports/metaworld_baseline_scaling_20261108/hourly' / (now.strftime('%Y%m%dT%H%M%S') + '.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')
    print('snapshot', str(dest))
    for key in ['original', 'scaling']:
        data = result[key]
        print(key, {k: data[k] for k in ['complete', 'errors', 'claims']})
        for row in data['rows']:
            if key == 'original' and row['complete'] and not row['error']:
                continue
            if key == 'scaling' and row['health_step'] is None and not row['proofs']:
                continue
            print(row['name'], 'step', row['health_step'], 'scores', {e: round(s['macro50'], 3) for e, s in row['epochs'].items()},
                  'hard', {e: s['hard6success'] for e, s in row['epochs'].items()}, 'error', row['proofs'].get('error.json'))
    print('native', {k: result['native'][k] for k in ['completed_shards', 'errors', 'episodes']})
    if 'native_pilot' in result:
        p = result['native_pilot']
        print('early_native_pilot', {k: p[k] for k in ['formal_trace_files', 'technical_completed_reports']},
              'error', p.get('ERROR.json'), 'closed', p.get('CLOSED.json'))
        print('pilot_model_results', [{k: r[k] for k in ['name', 'episodes', 'native_successes', 'cem_successes']}
              for r in p.get('progress.json', {}).get('models', [])])
    for name, p in result.get('scene_probe', {}).items():
        print('scene_probe', name, 'progress', p.get('progress.json'), 'frozen', p.get('FROZEN.json', {}).get('passed'),
              'complete', p.get('summary.json', {}).get('passed'), 'error', p.get('ERROR.json'))
    for label in ['support_loan','geometry_loan']:
        if label in result:
            p = result[label]
            print(label, 'formal_traces', p['formal_trace_files'], 'technical_reports',p['technical_completed_reports'],
                  'gpu_loan_ready','GPU_LOAN_READY.json' in p,'closed',p.get('CLOSED.json'),
                  'error',str(p.get('ERROR.json'))[:1400],'summary',p.get('summary.json'))
    for name, status in result.get('t2g_adapted', {}).items():
        print('t2g_adapted', name, 'progress', status.get('progress.json'), 'complete', status.get('summary'),
              'last_training_epoch', status.get('last_training_epoch'),
              'errors', {k: v for k, v in status.items() if k.startswith('ERROR-')})


if __name__ == '__main__':
    main()
