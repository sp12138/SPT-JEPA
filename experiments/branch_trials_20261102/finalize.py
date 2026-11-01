"""Finite completion trigger: archive all terminal evidence and exit; no restart."""
import argparse
import datetime
from pathlib import Path
import time

from common import read,write,sha


def main(root):
    root=Path(root);protocol=read(root/'protocol.json');deadline=time.monotonic()+24*3600
    while True:
        paths=[root/'dispatchers'/f'{host}-gate-finished.json' for host in protocol['host_families']]
        if all(p.exists() for p in paths):break
        if time.monotonic()>deadline:
            write(root/'completion-timeout.json',dict(reason='24h finite completion deadline;no model restart or stop',continuous_monitor=False));return
        time.sleep(30)
    terminals=[read(p) for p in paths]
    collection_error=None
    if all(x['passed'] for x in terminals):
        try:
            from collect import main as collect
            collect(root)
        except Exception as exc:
            collection_error=repr(exc)
    if not all(x['passed'] for x in terminals) or collection_error:
        write(root/'completion-failures.json',dict(dispatchers=terminals,collection_error=collection_error,automatic_restart=False))
    write(root/'HOLD',dict(reason='Finite stage completed;no automatic extension',
                          all_dispatchers_passed=all(x['passed'] for x in terminals),
                          collection_error=collection_error,
                          completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          helper_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
