"""Exercise the real scheduler startup without training or simulation imports."""
import ast
import json
from pathlib import Path
import tempfile
import time
import unittest


class ScalingStartupTest(unittest.TestCase):
    def invoke(self, root, first_hour, ancestry, elapsed):
        run = root/"training/example"
        run.mkdir(parents=True)
        config = dict(host="dev6", variant="spt-control-affine", gpus="0,1,2,3",
            world_size=4, growing_data=False, ancestry_updates=ancestry)
        if first_hour is not None:
            config["first_readout_hour"] = first_hour
        (run/"config.json").write_text(json.dumps(config))
        (root/"PRETRAIN_READY.json").touch()
        calls = []
        state = dict(step=ancestry, elapsed=elapsed)

        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))

        def call(unused_root, arguments, unused_log, unused_gpus):
            label = arguments[arguments.index("--label")+1]
            target = float(arguments[arguments.index("--target-seconds")+1])
            limit = int(arguments[arguments.index("--max-updates")+1])
            calls.append((label, target, limit))
            if target > state["elapsed"]:
                state["step"] += limit
                state["elapsed"] += 1
            save(run/label/"ready.json", dict(passed=True, step=state["step"]))
            if label == "restore-probe":
                (root/"STOP").touch()

        source = Path(__file__).resolve().parents[1]/"experiments/robocasa_stable_scaling/campaign.py"
        node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "worker")
        namespace = dict(json=json, time=time, save=save, call=call, PYTHON="python", MODULE="learner",
            gpu_startup_ready=lambda *args: True, corpus=lambda *args: root/"manifest.json",
            stage=lambda *args: None)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        return namespace["worker"], calls, state

    def test_resumed_startup_uses_cumulative_budget_and_retains_updates(self):
        for first_hour, ancestry, elapsed, target in [(2, 16136, 3600.17, 7200),
                (None, 0, 0, 3600)]:
            with self.subTest(first_hour=first_hour), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                worker, calls, state = self.invoke(root, first_hour, ancestry, elapsed)
                worker(root, "dev6", "example")
                self.assertEqual(calls, [("entry", target, 8), ("restore-probe", target, 8)])
                self.assertEqual(state["step"], ancestry+16)

    def test_rejects_zero_update_startup_before_readout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            worker, calls, state = self.invoke(root, 1, 16136, 3600.17)
            with self.assertRaises(AssertionError):
                worker(root, "dev6", "example")
            self.assertEqual(len(calls), 1)
            self.assertEqual(state["step"], 16136)
            self.assertTrue((root/"training/example/ERROR.json").exists())


if __name__ == "__main__":
    unittest.main()
