"""Bounded native training/health sequence; no training implementation or monitor."""
import argparse
import copy
import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from common import PY, read, sha, source_check, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--phase', choices=['gate', 'long'], required=True)
    a = p.parse_args()
    root = Path(a.root)
    protocol = read(root / 'protocol.json')
    row = next(r for r in protocol['runs'] if r['id'] == a.run_id)
    run = Path(row['run'])
    assert sha(run / 'config.json') == row['config_sha256']
    source_check(row['source'])
    assert read(row['actual_postcopy_CPU_restore'])['passed']
    assert not (run / 'STOP').exists()
    gpu = row['gpu']
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu)], text=True)
    assert int(free.strip()) == 0, free
    if a.phase == 'long':
        approval = read(root / 'long-approval.json')
        assert approval['approved'] and approval['stop_step'] == protocol['stop_step']
        assert a.run_id in approval['run_ids'] and read(root / 'gate-health-records' / (a.run_id + '.json'))['passed']
    folder = root / 'worker-records'; folder.mkdir(exist_ok=True)
    record_path = folder / f'{a.run_id}-{a.phase}.json'
    assert not record_path.exists()
    env = dict(os.environ, **protocol['old_environment'])
    env.update(PYTHONPATH=row['source']['pythonpath'], PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
    started = time.monotonic()
    results = []
    write(folder / f'{a.run_id}-{a.phase}-start.json', dict(pid=os.getpid(), gpu=gpu,
          started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          phase=a.phase, config_sha256=row['config_sha256'], code_sha256=sha(__file__), monitor=False))

    def execute(label, command, seconds):
        logs = root / 'worker-logs'; logs.mkdir(exist_ok=True)
        logpath = logs / f'{a.run_id}-{a.phase}-{label}.log'
        with logpath.open('x') as log:
            child = subprocess.Popen(command, env=env, cwd=row['source']['source'],
                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(folder / f'{a.run_id}-{a.phase}-{label}-process.json', dict(pid=child.pid, pgid=child.pid,
                  command=command, native_source=row['source'], parent_worker_pid=os.getpid()))
            try:
                rc = child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                rc = 124
        results.append(dict(phase=label, returncode=rc, log=str(logpath)))
        assert rc == 0, str(logpath)

    ok, error = False, None
    try:
        command = row['gate_command'] if a.phase == 'gate' else row['long_command']
        execute('native-training', command, 2400 if a.phase == 'gate' else 36000)
        step = protocol['health_gate_step'] if a.phase == 'gate' else protocol['stop_step']
        status = read(run / 'status.json')
        assert status['state'] == 'finished' and status['step'] == step
        archive = run / 'full-archives'; archive.mkdir(exist_ok=True)
        destination = archive / f'step{step:07d}.ckpt'
        assert not destination.exists()
        shutil.copy2(run / 'last.ckpt', destination)
        assert sha(destination) == sha(run / 'last.ckpt')
        health = row['gate_health_command'].copy()
        if a.phase == 'long':
            health[health.index('--expected-step')+1] = str(step)
            health[health.index('--output')+1] = str(root / ('final-health-'+a.run_id))
        execute('native-backward-health', health, 600)
        health_dir = Path(health[health.index('--output')+1])
        report = read(health_dir / 'preflight.json')
        assert report['passed']
        if a.phase == 'gate':
            write(root / 'gate-health-records' / (a.run_id+'.json'), dict(passed=True, step=step,
                  report=str(health_dir / 'preflight.json'), report_sha256=sha(health_dir / 'preflight.json'),
                  complete_checkpoint=str(destination), checkpoint_sha256=sha(destination), status=status))
        else:
            # Final evaluation is part of this finite worker, not a periodic
            # watcher. Each original/random/retrieval result is retained.
            helper = approval['endpoint_helper_source']
            source_check(helper)
            original_plan = read(Path(approval['reference_spec_root']) / 'candidate-plan.json')
            checkpoint = run / f'model-step{step:07d}'
            for case in ('primary', 'fresh', 'newcases'):
                reference_case = 'primary' if case == 'newcases' else case
                reference_id = f'{row["family"]}-seed{row["seed"]}-{reference_case}-original'
                original_job = next(j for j in original_plan['jobs'] if j['id'] == reference_id)
                base = read(original_job['spec'])
                for mode in ('original', 'random', 'retrieval'):
                    spec = copy.deepcopy(base)
                    ident = f'{a.run_id}-{case}-{mode}'
                    spec.update(id=ident, helper_source=helper, checkpoint=str(checkpoint), mode=mode,
                                case_set=case, output=str(root / 'final-evaluations' / ident))
                    if case == 'newcases':
                        spec['manifest'] = approval['newcase_manifest']
                    inputs = [checkpoint/'weights.pt', checkpoint/'config.json', Path(spec['manifest']),
                              Path(spec['data'])/'dataset_card.json', Path(spec['bank_report']),
                              Path(read(spec['bank_report'])['bank']), Path(spec['reference_metrics']), Path(spec['native_evaluator'])]
                    spec['immutable_inputs'] = {str(path): sha(path) for path in inputs}
                    path = root / 'final-specs' / (ident+'.json'); write(path, spec)
                    execute('evaluation-'+case+'-'+mode, [PY, '-B', '-u',
                            str(Path(helper['source'])/'experiments/trajectory_search_20261101/evaluate.py'),
                            '--spec', str(path)], 900)
        ok = True
    except Exception as exc:
        error = repr(exc)
    write(record_path, dict(passed=ok, run_id=a.run_id, phase=a.phase, results=results,
          seconds=time.monotonic()-started, error=error, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          continuous_monitor=False))
    if not ok:
        raise RuntimeError(error)
    print(f'{a.run_id}:{a.phase} finished with native checkpoint/backward health.', flush=True)


if __name__ == '__main__':
    main()
