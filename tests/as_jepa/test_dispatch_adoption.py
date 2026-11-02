"""A dispatcher restart must neither duplicate SGD nor discard recorded failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/lingjing_fullspec_20261102'))
from repair_dispatch import resume_partition


class DispatchAdoptionTest(unittest.TestCase):
    def test_existing_workers_and_terminals_never_return_to_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'terminals').mkdir(); (root / 'processes').mkdir()
            rows = [dict(id=name) for name in ['pending', 'running', 'passed', 'failed']]
            (root / 'processes/running.json').write_text(json.dumps(dict(pid=123, gpu=0)))
            for name, passed in [('passed', True), ('failed', False)]:
                (root / 'terminals' / (name + '.json')).write_text(json.dumps(dict(passed=passed)))
            pending, adopted, results = resume_partition(root, rows)
            self.assertEqual([r['id'] for r in pending], ['pending'])
            self.assertEqual([r['id'] for r, _ in adopted], ['running'])
            self.assertEqual({r['id']: r['returncode'] for r in results}, {'passed': 0, 'failed': 1})


if __name__ == '__main__':unittest.main()
