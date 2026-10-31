"""Optional state-conditioned output-basis diagnostic, not a new objective.

The static arm has exactly the same parameters and operations, but its sixteen
state-input columns receive fixed zeros. Both add zero at initialization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

READOUT_SEED = 20261031


class DynamicReadout(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in {'static', 'state'}:
            raise ValueError('Readout requires static or state mode')
        self.mode = mode
        # Constructors use CPU RNG internally; no global/CUDA stream is changed.
        with torch.random.fork_rng(devices=[]):
            self.input = nn.Linear(18, 64)
            self.output = nn.Linear(64, 64)
        generator = torch.Generator(device='cpu').manual_seed(READOUT_SEED)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.input.weight, a=math.sqrt(5), generator=generator)
            nn.init.uniform_(self.input.bias, -1/math.sqrt(18), 1/math.sqrt(18), generator=generator)
            self.output.weight.zero_()
            self.output.bias.zero_()
        axis = torch.linspace(-1., 1., 8)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        # Fixed row-major (x,y), reconstructed by every loader, no new state key.
        self.register_buffer('positions', torch.stack([x, y], -1).reshape(64, 2), persistent=False)

    def mixing_matrix(self, state):
        if state.shape[-1] != 1024:
            raise ValueError('Readout requires64 slots×16 channels')
        with torch.autocast(device_type=state.device.type, enabled=False):
            slots = state.float().reshape(*state.shape[:-1], 64, 16)
            if self.mode == 'static':
                slots = torch.zeros_like(slots)
            positions = self.positions.float().expand(*slots.shape[:-2], 64, 2)
            hidden = torch.tanh(F.linear(torch.cat([slots, positions], -1), self.input.weight.float(), self.input.bias.float()))
            return F.linear(hidden, self.output.weight.float(), self.output.bias.float())

    def forward(self, state, prediction):
        if state.shape != prediction.shape or state.shape[-1] != 1024:
            raise ValueError('State and native carry prediction must have matching1024-D shapes')
        with torch.autocast(device_type=state.device.type, enabled=False):
            # Subtract only after FP32 conversion. The original predictor has
            # already added carry; this does not recover sub-ULP increments.
            delta = (prediction.float()-state.float()).reshape(*state.shape[:-1], 64, 16)
            correction = (self.mixing_matrix(state) @ delta).reshape_as(prediction)
            return prediction + correction.to(prediction.dtype)
