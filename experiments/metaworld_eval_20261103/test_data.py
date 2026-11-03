"""Causal alignment regression for phase packing and cross-file episodes."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import h5py

from experiments.metaworld_eval_20261103.prepare_data import initialize_worker, prepare_bundle


class PhasePackingTest(unittest.TestCase):
    def test_cross_file_episode_retains_every_phase_and_causal_action(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / 'shards').mkdir()
            rows = []
            for i in range(43):
                image = Image.fromarray(np.full((480, 480, 3), i, dtype=np.uint8))
                buffer = io.BytesIO(); image.save(buffer, format='PNG')
                rows.append(dict(index=i, action=[i / 10, -i / 10, i / 100, -.5],
                                 **{'observation.image': dict(bytes=buffer.getvalue(), path=None)}))
            for index, (start, stop) in enumerate([(0, 17), (17, 43)]):
                pq.write_table(pa.Table.from_pylist(rows[start:stop]), root / f'raw{index}.parquet')
            ranges = [dict(path='raw0.parquet', start=0, end=17), dict(path='raw1.parquet', start=17, end=43)]
            initialize_worker(root, root, ranges, [])
            result = prepare_bundle((0, [dict(episode_index=0, dataset_from_index=0, dataset_to_index=43,
                                              task_id=0, length=43)]))
            self.assertEqual(sum(max(0, p['length'] - 7) for p in result['phases']), 43 - 35)
            with h5py.File(result['path'], 'r') as file:
                self.assertEqual(set(file), {'pixels', 'action', 'raw_row'})
                all_rows = []
                for phase in result['phases']:
                    offset, n = phase['offset'], phase['length']
                    actual = file['raw_row'][offset:offset + n]
                    expected = np.arange(phase['phase'], 43, 5)
                    np.testing.assert_array_equal(actual, expected)
                    all_rows.extend(actual.tolist())
                    np.testing.assert_array_equal(file['pixels'][offset:offset + n, 0, 0, 0], expected)
                    for k, raw in enumerate(expected[:-1]):
                        actions = np.clip(np.array([r['action'] for r in rows[raw:raw + 5]], dtype=np.float32), -1, 1)
                        np.testing.assert_array_equal(file['action'][offset + k], actions.reshape(-1))
                    np.testing.assert_array_equal(file['action'][offset + n - 1], np.zeros(20))
                self.assertEqual(sorted(all_rows), list(range(43)))
                self.assertGreater(result['clipped_components'], 0)
                self.assertEqual(result['train_controls'], 43 - 5)


if __name__ == '__main__':
    unittest.main()
