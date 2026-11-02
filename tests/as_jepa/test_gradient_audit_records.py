"""Regression for a real first-epoch audit false alarm; actual zero/stale/missing remain failures."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/lingjing_fullspec_20261102'))
from hard_worker import audit_gradients


class GradientRecordsTest(unittest.TestCase):
    def test_unlogged_nonround_finished_row_is_not_zero(self):
        trace = [dict(step=s, state='running', grad_norm_encoder=.4) for s in [12200, 12300, 12400]]
        trace.append(dict(step=12491, state='finished'))
        report = audit_gradients(trace, {'encoder': 'active'}, 12491)['encoder']
        self.assertTrue(report['contract_passed'])
        self.assertEqual(report['last_measured_step'], 12400)
        self.assertEqual(report['recent_minimum'], .4)

    def test_measured_zero_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit_gradients([dict(step=12400, grad_norm_encoder=0.)], {'encoder': 'active'}, 12491)

    def test_missing_scheduled_measurement_is_rejected(self):
        with self.assertRaises(AssertionError):
            audit_gradients([dict(step=12300, grad_norm_encoder=1.), dict(step=12400)], {'encoder': 'active'}, 12491)

    def test_stale_or_absent_measurement_is_rejected(self):
        for trace in [[dict(step=12200, grad_norm_encoder=1.)], [dict(step=12491, state='finished')]]:
            with self.assertRaises(AssertionError):audit_gradients(trace, {'encoder': 'active'}, 12491)

    def test_frozen_path_cannot_hide_historical_gradient(self):
        with self.assertRaises(AssertionError):
            audit_gradients([dict(step=100, grad_norm_encoder=1.), dict(step=12400, grad_norm_encoder=0.)], {'encoder': 'frozen'}, 12491)


if __name__ == '__main__':unittest.main()
