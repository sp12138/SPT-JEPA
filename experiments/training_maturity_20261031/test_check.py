"""CPU regression for the observed finished-status false alarm; no remote IO."""
import json
from pathlib import Path
import unittest

from check import gradient_evidence


class GradientEvidenceTest(unittest.TestCase):
    def test_actual_finished_native_records(self):
        evidence = json.loads((Path(__file__).parent / 'results/review130026/gradient-status-clarification.json').read_text())
        for run in evidence['runs']:
            with self.subTest(run=run['id']):
                result = gradient_evidence(dict(state=run['snapshot_state'], step=run['snapshot_step']),
                                           [run['last_actual_training_gradient_record']], run['family'])
                self.assertEqual(result, run['corrected_gradient_evidence'])
                self.assertTrue(result['active_gradient_positive'])
                self.assertEqual(result['gradient_evidence_step'], 23300)

    def test_missing_running_gradients_do_not_use_history(self):
        result = gradient_evidence(dict(state='running', step=200),
                                   [dict(state='running', step=100, grad_norm_predictor=1.0)], 'native')
        self.assertFalse(result['active_gradient_positive'])
        self.assertEqual(result['gradient_source'], 'status')

    def test_missing_finished_history_fails(self):
        result = gradient_evidence(dict(state='finished', step=200), [], 'native')
        self.assertFalse(result['active_gradient_positive'])
        self.assertEqual(result['gradient_source'], 'missing')

    def test_bad_active_values_are_not_hidden(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.subTest(value=value):
                result = gradient_evidence(dict(state='finished', step=200, grad_norm_predictor=value),
                                           [dict(state='running', step=100, grad_norm_predictor=1.0)], 'native')
                self.assertFalse(result['active_gradient_positive'])
                self.assertEqual(result['gradient_source'], 'status')

    def test_latest_history_failure_is_not_skipped(self):
        result = gradient_evidence(dict(state='finished', step=200),
                                   [dict(state='running', step=100, grad_norm_predictor=1.0),
                                    dict(state='running', step=150, grad_norm_predictor=0.0)], 'native')
        self.assertFalse(result['active_gradient_positive'])
        self.assertEqual(result['gradient_evidence_step'], 150)

    def test_local_frozen_modules_and_finite_active_modules(self):
        status = dict(state='running', step=200, grad_norm_encoder=0.0,
                      grad_norm_pool=0.0, grad_norm_predictor=0.1, grad_norm_renderer=0.2)
        self.assertTrue(gradient_evidence(status, [], 'local')['active_gradient_positive'])
        self.assertFalse(gradient_evidence(status, [], 'native')['active_gradient_positive'])
        status['grad_norm_renderer'] = 0.0
        self.assertFalse(gradient_evidence(status, [], 'local')['active_gradient_positive'])


if __name__ == '__main__':
    unittest.main()
