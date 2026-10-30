"""CPU operational contracts; synthetic checkpoint bytes are not model evidence."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.carrier_progress_20261031 import prepare_learning as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PreparationContracts(unittest.TestCase):
    def test_assignment_and_semantics(self):
        rows = list(p.assignments())
        self.assertEqual(len(rows), 12)
        self.assertEqual([(r['host'], r['gpu']) for r in rows],
                         [('dev2', i) for i in range(8)] + [('dev13', i) for i in range(4)])
        self.assertEqual(rows[8]['id'], 'cube-carrier-bounded-seed1')
        cfg = dict(model_class='carrier', max_steps=124910, batch_size=128, workers=4,
                   model={'carrier_chart': 'identity', 'carrier_mean': [1.]}, seed=1)
        before = copy.deepcopy(cfg)
        result = p.continuation_config(cfg, '/new')
        self.assertEqual(cfg, before)
        self.assertEqual(p.semantic(cfg), p.semantic(result))
        self.assertEqual(result['retain_steps'], [1200, 2000])
        saved = dict(cfg, **{'slurm.job_id': 'N/A', 'slurm.task_id': 'N/A', 'is_resuming': False})
        self.assertEqual(p.checkpoint_semantic(saved), p.semantic(cfg))
        self.assertNotEqual(p.checkpoint_semantic(dict(saved, seed=2)), p.semantic(cfg))

    def test_full_dependency_chain_and_history_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); parent = base / 'old'; root = base / 'new'
            src = dict(source=str(base / 'source'), source_commit='synthetic', pythonpath='synthetic')
            source_record = base / 'source-record.json'; write(source_record, src)
            write(parent / 'source-record.json', src)
            write(parent / 'review600.json', dict(all_exit_zero=True,
                all_initial_original550_equal=True, all_full_state_health_pass=True))
            (parent / 'HOLD_DISPATCH').write_text('preserve')
            jobs = []
            for row in p.assignments():
                out = parent / row['id']; out.mkdir()
                (out / 'last.ckpt').write_bytes(('synthetic-' + row['id']).encode())
                (out / 'STOP').write_text('preserve')
                cal = out / 'calibration.json'; write(cal, dict(synthetic=True))
                cfg = dict(model_class='carrier', max_steps=124910, batch_size=128, workers=4,
                    model={'carrier_chart': row['arm']}, seed=row['training_seed'], data='/data',
                    carrier_calibration=dict(path=str(cal), sha256=p.digest(cal)))
                write(out / 'config.json', cfg)
                write(out / 'stage-complete.json', dict(stage_complete=True, stop_step=600))
                for prefix in ('', 'health-', 'info-', 'transition-', 'planning-'):
                    write(parent / 'exits' / (prefix + row['id'] + '.json'), dict(returncode=0))
                jobs.append(dict(row, output=str(out), phase='train600', config_sha256=p.digest(out/'config.json')))
            write(parent / 'plan.json', dict(environment={'SHARED': 'same'}, jobs=jobs))
            args = SimpleNamespace(root=str(root), parent=str(parent), source_record=str(source_record), dry_run=False)
            audit = lambda run, step, old: dict(passed=True, checkpoint_sha256=p.digest(Path(run)/'last.ckpt'))
            with patch.object(p, 'verify_sources', return_value={'identical_files': {}}), \
                 patch.object(p, 'activate_source'), patch.object(p, 'state_audit', side_effect=audit):
                p.prepare(args)
                with self.assertRaises(AssertionError):
                    p.prepare(args)
            plan = p.read(root / 'plan.json'); self.assertEqual(len(plan['jobs']), 96)
            by_id = {j['id']: j for j in plan['jobs']}
            for row in p.assignments():
                name = row['id']; previous = []
                for prefix in ('learn1200-', 'fullstate1200-', 'learn2000-', 'fullstate2000-',
                               'health-', 'info-', 'transition-', 'planning-'):
                    job = by_id[prefix + name]
                    self.assertEqual(job['depends_on'], previous)
                    previous = [job['id']]
                self.assertEqual(p.digest(root/name/'last.ckpt'), p.digest(parent/name/'last.ckpt'))
                self.assertEqual((parent/name/'STOP').read_text(), 'preserve')
                self.assertFalse((root/name/'STOP').exists())
            self.assertEqual((parent/'HOLD_DISPATCH').read_text(), 'preserve')
            self.assertEqual(plan['environment'], {'SHARED': 'same'})

    def test_archive_accepts_dispatch_directory_and_requires_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir(); out=root/'audit'; out.mkdir()
            (out/'dispatch.log').write_text('dispatcher owns this')
            (run/'last.ckpt').write_bytes(b'synthetic checkpoint')
            write(run/'stage-complete.json', dict(stage_complete=True, stop_step=1200))
            source=root/'source.json'; write(source, dict(source_commit='synthetic'))
            args=SimpleNamespace(source_record=str(source), output=str(out), audit_run=str(run),
                                 expected_step=1200, parent_run=str(run))
            with patch.object(p, 'activate_source'), patch.object(p, 'state_audit', return_value=dict(
                    passed=True, checkpoint_sha256=p.digest(run/'last.ckpt'))):
                p.archive_state(args)
            self.assertEqual((run/'checkpoint-step0001200.ckpt').read_bytes(), b'synthetic checkpoint')
            self.assertTrue(p.read(out/'fullstate.json')['passed'])


if __name__ == '__main__':
    unittest.main()
