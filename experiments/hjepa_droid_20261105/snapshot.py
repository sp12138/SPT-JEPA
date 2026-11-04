"""One read-only resource/progress snapshot; never a recurring LLM monitor."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main(args):
    root = Path(args.root)
    assert args.host in ['dev13', 'dev15']
    ownership = json.loads((root / f'queue-adoption-{args.host}.json').read_text())
    rows = []
    for job in ownership['jobs']:
        native = job['native_pid']
        command = Path(f'/proc/{native}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert '/helper-v8/train.py' in command
        arm = ['temporal-forecast', 'innovation', 'lewm', 'h-jepa'][job['pair']]
        seed = 1 if args.host == 'dev13' else 1000
        run = root / 'training' / f'{arm}-seed{seed}'
        health = [json.loads((run / f'health-rank{rank}.json').read_text()) for rank in [0, 1]]
        assert all(h['step'] >= 100 and all(v > 0 for v in h['gradients'].values()) for h in health)
        assert not (run / 'error.json').exists()
        assert Path(f"/proc/{job['pid']}/cmdline").exists()
        rows.append(dict(arm=arm, seed=seed, native_pid=native, controller_pid=job['pid'],
                         gpus=job['gpus'], native_command=command, health=health))
    query = ['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader']
    result = dict(passed=True, host=args.host, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  runs=rows, gpu_snapshot=subprocess.run(query, capture_output=True, text=True, check=True).stdout,
                  instantaneous_not_continuous=True)
    output = root / 'results/startup' / (args.host + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, host=args.host,
                         steps={row['arm']: [h['step'] for h in row['health']] for row in rows},
                         gpu_snapshot=result['gpu_snapshot'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--host', required=True)
    main(parser.parse_args())
