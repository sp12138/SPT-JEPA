"""Bounded fresh-open retries for observed EFS ESTALE in the read-only collector."""
import argparse
import errno
import json
from pathlib import Path
import time


def read(path):
    for attempt in range(3):
        try:return json.loads(Path(path).read_text())
        except (OSError,json.JSONDecodeError) as error:
            if isinstance(error,OSError) and error.errno not in [errno.EIO,errno.ESTALE,errno.ENOENT]:raise
            if attempt==2:raise
            # Reopen the exact same named record; never substitute/skip it.
            time.sleep([.1,.5][attempt])


def test(output):
    from unittest.mock import patch
    with patch.object(Path,'read_text',side_effect=[OSError(errno.ESTALE,'injected stale handle'),'{"step":100}']) as mock:
        assert read('/unused')==dict(step=100) and mock.call_count==2
    with patch.object(Path,'read_text',side_effect=OSError(errno.ESTALE,'persistent')) as mock:
        try:read('/unused')
        except OSError as error:assert error.errno==errno.ESTALE and mock.call_count==3
        else:raise AssertionError('Persistent error was hidden')
    with patch.object(Path,'read_text',side_effect=PermissionError(errno.EACCES,'denied')) as mock:
        try:read('/unused')
        except PermissionError:assert mock.call_count==1
        else:raise AssertionError('Unexpected access error was hidden')
    with patch.object(Path,'read_text',side_effect=['{','{"passed":true}']):assert read('/unused')['passed']
    Path(output).write_text(json.dumps(dict(passed=True,transient_estale_reopened=True,persistent_estale_stops_after3=True,
        access_errors_propagated_immediately=True,partial_json_reopened=True,model_updates=0),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--test-output');args=p.parse_args()
    if args.test_output:test(args.test_output)
    else:
        from experiments.metaworld_baselines_20261107 import campaign
        campaign.read=read
        campaign.collect()
