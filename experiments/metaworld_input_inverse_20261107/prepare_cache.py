"""Byte-verified local HDF5 cache; no new samples or preprocessing."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(original, cache):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    cache.mkdir(parents=True, exist_ok=False)
    (cache/'shards').mkdir()
    def copy(source):
        target = cache/'shards'/source.name
        shutil.copyfile(source, target)
        expected, actual = sha(source), sha(target)
        assert expected == actual, source
        return dict(source=str(source), target=str(target), sha256=actual, bytes=target.stat().st_size)
    sources = sorted((original/'shards').glob('*.h5'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(copy, sources))
    with h5py.File(original/'dataset.h5', 'r') as src, h5py.File(cache/'dataset.h5', 'w', libver='latest') as dst:
        for key in src:
            data = src[key]
            layout = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            for item in data.virtual_sources():
                lower, upper = item.vspace.get_select_bounds()
                local = cache/'shards'/Path(item.file_name).name
                with h5py.File(local, 'r') as shard:
                    shape = shard[item.dset_name].shape
                assert lower[1:] == tuple(0 for _ in shape[1:])
                assert upper[1:] == tuple(x-1 for x in shape[1:])
                assert upper[0]-lower[0]+1 == shape[0]
                layout[lower[0]:upper[0]+1] = h5py.VirtualSource(str(local), item.dset_name, shape=shape)
            dst.create_virtual_dataset(key, layout)
    # Read actual RGB/control/raw-index windows through both VDSs, all byte equal.
    checks = []
    with h5py.File(original/'dataset.h5', 'r') as a, h5py.File(cache/'dataset.h5', 'r') as b:
        starts = np.random.default_rng(20261107).integers(0, len(a['pixels'])-8, 64)
        for start in starts:
            for key in a:
                left, right = a[key][start:start+8], b[key][start:start+8]
                assert np.array_equal(left, right), (key, int(start))
            checks.append(int(start))
    report = dict(passed=True, original=str(original), original_vds_sha256=sha(original/'dataset.h5'),
        cache=str(cache), cache_vds_sha256=sha(cache/'dataset.h5'), shards=records,
        actual_equal_windows=checks, fields=['pixels', 'action', 'raw_row'],
        scope='Only VDS filenames rebound to identical local shard bytes; all rows, controls and RGB unchanged.')
    (cache/'COMPLETE.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cache=str(cache), files=len(records), bytes=sum(r['bytes'] for r in records))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    prepare(a.original, a.cache)
