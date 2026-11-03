"""Predict observation innovations relative to the observed anchor,per sample."""
import torch

from as_jepa_lewm.robust import RobustLeWM


class InnovationLeWM(RobustLeWM):
    def prepare(self,batch):
        pixels,render,anchor=super().prepare(batch)
        # A reversible translation conditional on an already observed frame.
        # No motion mask,object annotation,batch statistics or extra head.
        return pixels,render-anchor[:,None],anchor

    @torch.no_grad()
    def render_states(self,states,anchor,steps=50,seed=0):
        return super().render_states(states,anchor,steps,seed)+anchor
