"""Cap an existing Lightning run at a staged update without changing its LR horizon."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--stop-step", required=True, type=int)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    config_path = Path(a.config)
    cfg = json.loads(config_path.read_text())
    if not 0 < a.stop_step <= cfg["max_steps"]:
        raise ValueError("Stage limit must fall within the unchanged scheduler horizon")
    root = Path(cfg["output"])
    if (root / "STOP").exists():
        raise RuntimeError("Use a fresh stage output; do not clear an old user's stop marker")
    initial_step = 0
    if a.resume:
        state = torch.load(root / "last.ckpt", map_location="cpu", weights_only=False)
        initial_step = int(state["global_step"])
        if not initial_step < a.stop_step:
            raise ValueError("Resume checkpoint must precede the stage limit")
        if not state.get("optimizer_states") or not state.get("lr_schedulers") or not state.get("campaign_rng"):
            raise ValueError("Full optimizer, scheduler and RNG checkpoint required")
        if state["campaign_samples"] != initial_step * cfg["batch_size"]:
            raise ValueError("Resume sample cursor is not batch aligned")
        del state
    import as_jepa_lewm.train as entry
    original = entry.PersistentProgress.on_train_batch_end
    def capped(progress, trainer, module, outputs, batch, batch_idx):
        original(progress, trainer, module, outputs, batch, batch_idx)
        if int(trainer.global_step) >= a.stop_step:
            trainer.should_stop = True
    entry.PersistentProgress.on_train_batch_end = capped
    argv = sys.argv
    try:
        sys.argv = [str(config_path), str(config_path)] + (["--resume"] if a.resume else [])
        entry.main()
    finally:
        sys.argv = argv
        entry.PersistentProgress.on_train_batch_end = original
    status = json.loads((root / "status.json").read_text())
    if status["state"] != "finished" or status["step"] != a.stop_step:
        raise RuntimeError("Stage failed to reach its exact update budget: " + str(status))
    record = dict(stage_complete=True, initial_step=initial_step, stop_step=a.stop_step,
                  scheduler_total_steps=cfg["max_steps"], config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  scope="Existing Lightning Trainer and checkpoint code; only an operational exact-step stop callback is added.")
    (root / "stage-complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
