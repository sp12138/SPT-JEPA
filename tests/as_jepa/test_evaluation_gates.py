import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from as_jepa_tools.collapse_audit import summarize_errors
from as_jepa_tools.completion_evaluate import verify_probes


class EvaluationGateContracts(unittest.TestCase):
    def test_collapse_uses_only_moved_frames_and_both_conditions(self):
        anchor = np.array([[10., 100.], [10., 100.]])
        posterior = np.array([[8., 200.], [8., 200.]])
        moved = np.array([[True, False], [True, False]])
        self.assertFalse(summarize_errors(posterior, anchor, moved, [.2, .2])["collapsed"])
        self.assertTrue(summarize_errors(posterior, anchor, moved, [.2, .01])["collapsed"])
        self.assertTrue(summarize_errors(anchor * .9, anchor, moved, [.2, .2])["collapsed"])
        with self.assertRaises(ValueError):
            summarize_errors(posterior, anchor, np.zeros_like(moved), [.2, .2])

    def test_successful_outer_probe_exit_cannot_hide_inner_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = root / "status.json"
            status.write_text(json.dumps(dict(state="finished", results=[dict(error="inner failed")])))
            with self.assertRaises(RuntimeError):
                verify_probes(status)
            status.write_text(json.dumps(dict(state="finished", results=[dict(error=None, output=str(root))])))
            with self.assertRaises(RuntimeError):
                verify_probes(status)
            (root / "probes.json").write_text("{}")
            verify_probes(status)


if __name__ == "__main__":
    unittest.main()
