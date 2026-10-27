import sys
from omegaconf import OmegaConf
from vforge.tasks.registry import ModelTasks,register
register("as_jepa_spt",ModelTasks(train="vforge.tasks.as_jepa.train:build_train_components"))
from vforge.tasks.train import run_train
if __name__=="__main__":run_train(OmegaConf.load(sys.argv[1]))
