"""Same learned operations and parameters; only the factual loss horizon changes."""
from experiments.metaworld_transition_only_20261109.model import FactualRecurrent


class HorizonMatched(FactualRecurrent):
    def __init__(self,horizon=15,width=128,inverse_weight=0.):
        assert horizon in [15,60]
        super().__init__(horizon=15,width=width,inverse_weight=inverse_weight)
        self.horizon=horizon


def build(cfg):
    return HorizonMatched(cfg['horizon'],cfg['width'],cfg['inverse_weight'])
