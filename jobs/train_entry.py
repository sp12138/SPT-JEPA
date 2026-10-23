"""Load a frozen campaign config and use the Native training entry point."""
import sys
from omegaconf import OmegaConf
from vforge.tasks.train import run_train

if len(sys.argv) != 2:
    raise SystemExit('Usage: train_entry.py CONFIG')
run_train(OmegaConf.load(sys.argv[1]))
