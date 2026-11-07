"""User-authorized hourly message delivery to the SAME Codex session.

No new agent, model override, or independent goal is created. launchd supplies
the cadence; this process does not run a polling LLM or inspect credentials.
"""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess


def run(args):
    state = args.repo / 'artifacts/metaworld_hourly_20261108'
    state.mkdir(parents=True, exist_ok=True)
    if (state / 'STOP').exists(): return
    with (state / 'delivery.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        if args.probe:
            message = ('【每小时监控连通性验证】这是用户刚授权的同会话定时检查投递测试。'
                       '收到后记录连通即可，继续当前300epoch续训与定时检查配置；不要启动重复实验或第二套监控。')
        else:
            message = (args.repo / 'experiments/metaworld_baseline_scaling_20261108/hourly-prompt.txt').read_text()
        result = subprocess.run([str(args.codex), 'queue', '--thread', args.thread, '--message', message, '--resume'],
                                cwd=args.repo, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=45)
        report = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), thread=args.thread,
                      probe=args.probe, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      scope='CLI delivery result; an accepted queued message is not a completed research check.')
        with (state / 'deliveries.jsonl').open('a') as stream: stream.write(json.dumps(report)+'\n')
        (state / 'latest-delivery.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))
        if result.returncode: raise SystemExit(result.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--thread', required=True); p.add_argument('--codex', type=Path, required=True)
    p.add_argument('--probe', action='store_true'); run(p.parse_args())
