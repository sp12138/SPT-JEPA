"""Operational local-cache binding around the unchanged native stage/audit."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def install_cache():
    from as_jepa_tools.benchmark_data import BenchmarkWindows
    cache = Path('/tmp/prescottsun-mw-input-inverse-20261107-v1')
    proof = json.loads((cache/'COMPLETE.json').read_text())
    assert proof['passed'] and len(proof['shards']) == 250
    assert len(proof['actual_equal_windows']) == 64
    vds_sha = hashlib.sha256((cache/'dataset.h5').read_bytes()).hexdigest()
    assert proof['cache_vds_sha256'] == vds_sha
    original_open = BenchmarkWindows.open
    def open_cached(self):
        expected = Path(proof['original'])/'dataset.h5'
        actual = Path(self.card['h5_path'])
        if actual == expected:
            assert hashlib.sha256(expected.read_bytes()).hexdigest() == proof['original_vds_sha256']
            self.card = dict(self.card, h5_path=str(cache/'dataset.h5'))
        return original_open(self)
    BenchmarkWindows.open = open_cached
    return proof


if __name__ == '__main__':
    import torch  # Same installed native runtime, no new learner/optimizer.
    install_cache()
    mode = sys.argv.pop(1)
    source = Path(os.environ['SPT_INPUT_INVERSE_NN_SOURCE'])
    if mode == 'train':
        runpy.run_path(str(source/'experiments/lingjing_fullspec_20261102/safe_train_stage.py'), run_name='__main__')
    elif mode == 'audit':
        runpy.run_path(str(source/'experiments/metaworld_input_inverse_20261107/audit.py'), run_name='__main__')
    else:
        raise ValueError(mode)
