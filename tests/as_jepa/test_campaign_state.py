import json
import tempfile
import unittest
from pathlib import Path
from as_jepa_tools.campaign_state import update


class CampaignGateContracts(unittest.TestCase):
    def run_fixture(self, loss=1., batch=128, diagnostics=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name) / "project"
        root = project / "runs/campaign"
        out = root / "full-S2-smoke"
        out.mkdir(parents=True)
        (root / "exits").mkdir()
        (root / "exits/full-S2-smoke.json").write_text(json.dumps(dict(returncode=0)))
        job = dict(id="full-S2-smoke", output=str(out), source={}, host="dev15", gpu=0)
        plan = root / "plan.json"
        plan.write_text(json.dumps(dict(root=str(root), jobs=[job])))
        rows = []
        for step, lr in [(19, 2e-5), (99, 1e-4), (199, 1e-4)]:
            rows.append(dict(step=step, loss=loss, perf=dict(global_batch=batch),
                             learning_rate_after_step=lr,
                             metrics=dict(grad_norm_encoder_post_clip=.1) if diagnostics else {}))
        (out / "train_steps.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
        update(plan)
        return root / "gates/full-S2-smoke-PASS.json"

    def test_exit_zero_does_not_unlock_nonfinite_or_wrong_budget(self):
        self.assertFalse(self.run_fixture(loss=float("nan")).exists())
        self.assertFalse(self.run_fixture(batch=64).exists())
        self.assertFalse(self.run_fixture(diagnostics=False).exists())

    def test_real_execution_contract_unlocks_only_valid_smoke(self):
        self.assertTrue(self.run_fixture().exists())


if __name__ == "__main__":
    unittest.main()
