"""Learn goal patch correspondence before deciding the current action.

Only image coordinates and scratch visual features enter the matcher. The
current-control flow objective, data and deployment clocks are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .temporal_bridge import TemporalBridge


class CorrespondenceBridge(TemporalBridge):
    def __init__(self, **kwargs):
        assert kwargs.get('visual_source', 'local') == 'local'
        assert kwargs.get('goal_representation') == 'directed'
        assert not kwargs.get('history_context', False)
        assert kwargs.get('goal_information', 'provided') == 'provided'
        super().__init__(**kwargs)
        self.match_query = nn.Linear(self.width, self.width, bias=False)
        self.match_key = nn.Linear(self.width, self.width, bias=False)
        previous = self.goal_fusion[0]
        self.goal_fusion[0] = nn.Linear(self.width*2+2, self.width)
        with torch.no_grad():
            self.goal_fusion[0].weight[:, :self.width*2].copy_(previous.weight)
            self.goal_fusion[0].bias.copy_(previous.bias)
        grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 14),
                                          torch.linspace(-1, 1, 14), indexing='ij'), -1).reshape(196, 2)
        self.register_buffer('image_coordinates', grid, persistent=False)

    def memory(self, batch):
        images = batch['pixels'].float()/127.5-1.
        b, t = images.shape[:2]
        assert t == 2
        current, goal = self.encoder(images.flatten(0, 1)).unflatten(0, (b, t)).unbind(1)
        q = F.normalize(self.match_query(current).float(), dim=-1)
        k = F.normalize(self.match_key(goal).float(), dim=-1)
        correspondence = (q@k.transpose(-1, -2)*self.width**.5).softmax(-1)
        aligned = correspondence@goal.float()
        displacement = correspondence@self.image_coordinates-self.image_coordinates
        value = torch.cat([current, aligned-current, displacement], -1)
        return self.goal_fusion(value)+self.position+self.kind[0]

    def audit_expected_gradients(self):
        return dict(super().audit_expected_gradients(), match_query='active', match_key='active')
