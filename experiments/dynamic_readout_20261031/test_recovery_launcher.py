import contextlib
import errno
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

from experiments.dynamic_readout_20261031 import recovery_launcher as launcher


class AtomicContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.write = launcher.atomic_writer(self.root, 'dev15')

    def test_one_disappearance_recreates_and_writes_valid_json(self):
        replace = launcher.os.replace
        attempts = []
        def disappear_once(source, target):
            attempts.append(str(source))
            if len(attempts) == 1:
                Path(source).unlink()
            return replace(source, target)
        target = self.root / 'exits/job.json'
        with mock.patch.object(launcher.os, 'replace', side_effect=disappear_once):
            self.write(target, {'returncode': 0, 'id': 'actual-child'})
        self.assertEqual(json.loads(target.read_text()), {'returncode': 0, 'id': 'actual-child'})
        self.assertEqual(len(attempts), 2)
        self.assertNotEqual(*attempts)
        self.assertEqual(list(target.parent.glob('*.tmp')), [])

    def test_authoritative_permanent_failure_is_not_success(self):
        for relative in ['exits/job.json', 'job/dispatch.json', 'other/workers/dev15.json']:
            target = self.root / relative
            with mock.patch.object(launcher.os, 'replace', side_effect=FileNotFoundError(errno.ENOENT, 'injected')) as replace:
                with self.assertRaises(FileNotFoundError):
                    self.write(target, {'returncode': 0})
                self.assertEqual(replace.call_count, 3)
            self.assertFalse(target.exists())

    def test_only_exact_worker_snapshot_failure_is_nonfatal(self):
        target = self.root / 'workers/dev15.json'
        stderr = io.StringIO()
        with mock.patch.object(launcher.os, 'replace', side_effect=FileNotFoundError(errno.ENOENT, 'injected')) as replace:
            with contextlib.redirect_stderr(stderr):
                self.write(target, {'running': {}})
            self.assertEqual(replace.call_count, 3)
        self.assertIn('Non-authoritative worker snapshot', stderr.getvalue())
        self.assertFalse(target.exists())
        self.write(target, {'running': {}, 'next_loop': True})
        self.assertTrue(json.loads(target.read_text())['next_loop'])

    def test_other_errors_are_not_retried_or_silenced(self):
        with mock.patch.object(launcher.os, 'replace', side_effect=PermissionError(errno.EACCES, 'injected')) as replace:
            with self.assertRaises(PermissionError):
                self.write(self.root / 'workers/dev15.json', {})
            self.assertEqual(replace.call_count, 1)

    def test_same_input_concurrent_writes_are_complete_json(self):
        target = self.root / 'workers/dev15.json'
        value = {'running': {'0': {'id': 'a', 'pid': 123}}, 'padding': list(range(100))}
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: self.write(target, value), range(64)))
        self.assertEqual(json.loads(target.read_text()), value)
        self.assertEqual(list(target.parent.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
