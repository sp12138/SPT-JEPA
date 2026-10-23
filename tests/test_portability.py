"""Check source closure, unchanged model semantics and real run preparation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def test_runtime_source_matches_frozen_baseline(self):
        hashes = json.loads((ROOT / "provenance/upstream_runtime_hashes.json").read_text())
        for name, expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / "runtime" / name).read_bytes()).hexdigest(), expected, name)

    def test_model_and_training_semantics_are_preserved(self):
        old = OmegaConf.to_container(OmegaConf.load(ROOT / "provenance/upstream_pretrain_50k.yaml"), resolve=True)
        new = OmegaConf.to_container(OmegaConf.load(ROOT / "jobs/configs/pretrain_50k.yaml"), resolve=True)
        for section in ["model", "training", "latent_cache", "optimizer", "optimizers", "scheduler", "parallel", "attention", "attention_offload"]:
            a, b = old[section], new[section]
            if section == "model":
                self.assertFalse(a["extras"]["repa"]["enabled"])
                self.assertFalse(b["extras"]["repa"]["enabled"])
                a["extras"]["repa"]["teacher_checkpoint"] = b["extras"]["repa"]["teacher_checkpoint"]
            self.assertEqual(a, b, section)
        for field in ["seed", "deterministic", "precision", "autocast", "gradient_clip_val", "gradient_accumulation_steps", "gradient_norm_dtype"]:
            self.assertEqual(old["runtime"][field], new["runtime"][field], field)

    def test_active_jobs_do_not_import_shared_experiment_code(self):
        for path in (ROOT / "jobs").rglob("*.py"):
            self.assertNotIn("/efs/stormyxie", path.read_text(), str(path))
            self.assertNotIn("/threed-code/", path.read_text(), str(path))

    def test_detector_is_the_pinned_source_and_distance_is_correct(self):
        path = ROOT / "jobs/fvd_reference/detector/fvd.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "d0229a9055084132c93628edd98157fbbd5a8818f11834c27bd1a80b7e220f33")
        spec = importlib.util.spec_from_file_location("spt_test_fvd", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        features = np.random.default_rng(7).normal(size=(100, 3))
        self.assertAlmostEqual(module.frechet_distance(features, features)["fvd"], 0.0, places=10)
        self.assertAlmostEqual(module.frechet_distance(features, features + 2)["fvd"], 12.0, places=10)

    def test_native_train_and_eval_resolve_from_repository(self):
        sys.path.insert(0, str(ROOT / "runtime"))
        from vforge.tasks.registry import resolve_builder
        import vforge.tasks.train
        import vforge.tasks.eval

        for verb in ["train", "eval"]:
            function = resolve_builder("rep_world", verb)
            module = sys.modules[function.__module__]
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT / "runtime"))
        for name, module in list(sys.modules.items()):
            if name == "vforge" or name.startswith("vforge."):
                if getattr(module, "__file__", None):
                    self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT / "runtime"), name)

    def test_prepare_run_freezes_a_commit_with_explicit_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, val = root / "train.jsonl", root / "val.jsonl"
            row = dict(sample_id="test", rgb_path=str(root / "video.mp4"), start_frame=0)
            for path in [train, val]:
                path.write_text(json.dumps(row) + "\n")
            subprocess.run([sys.executable, "-B", str(ROOT / "jobs/prepare_run.py"), "portable",
                            "--data-root", str(root), "--train-manifest", str(train),
                            "--val-manifest", str(val)], check=True, capture_output=True, text=True)
            run = root / "runs/portable"
            plan = json.loads((run / "plan.json").read_text())
            cfg = OmegaConf.load(plan["pretrain_config"])
            self.assertEqual(cfg.data.train_manifest, str(train))
            self.assertEqual(cfg.data.val_manifest, str(val))
            self.assertEqual(cfg.runtime.output_dir, str(run / "output"))
            self.assertTrue((run / "source/jobs/fvd_reference/detector/fvd.py").is_file())
            for name, expected in json.loads((run / "source/jobs/job_hashes.json").read_text()).items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
