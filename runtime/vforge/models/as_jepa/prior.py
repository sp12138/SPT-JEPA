import torch
from torch import nn
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.conditioner import CausalViTBlock
from vforge.models.rep_world.sparsity import state_causal_sparsity, use_segmented_attention
from vforge.models.rep_world.split_conditioner import _initialize

class StatePrior(nn.Module):
    def __init__(self, K, d, action_dim, width=384, depth=6, heads=6):
        super().__init__()
        cfg = ConditionerConfig(width=width, heads=heads, mlp_ratio=4.0)
        self.K = K
        self.state_in = nn.Linear(d, width)
        self.action_in = nn.Linear(action_dim, width)
        self.slot = nn.Parameter(torch.zeros(K + 1, width))          # K state slots plus the action slot
        self.blocks = nn.ModuleList(CausalViTBlock(cfg) for _ in range(depth))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head = nn.Linear(width, d)

    def forward(self, s, a):                     # s: [B, T, K, d] past states; a: [B, T, action_dim] actions leaving them
        B, T, K, d = s.shape
        tokens = torch.cat([self.state_in(s), self.action_in(a)[:, :, None]], dim=2) + self.slot   # [B, T, K+1, width]
        h = tokens.reshape(B, T * (K + 1), -1)
        L = T * (K + 1)
        step = torch.arange(L, device=s.device) // (K + 1)
        mask = state_causal_sparsity(L, K + 1) if use_segmented_attention() else step[:, None] >= step[None, :]
        positions = (step.float(), (torch.arange(L, device=s.device) % (K + 1)).float(), torch.zeros(L, device=s.device))
        for block in self.blocks:
            h = block(h, mask=mask, positions=positions)
        out = self.norm(h).reshape(B, T, K + 1, -1)[:, :, :K]        # state-slot outputs of each step
        return s + self.head(out)                # identity carry: means of s_1 .. s_T as changes from s_0 .. s_{T-1}

    def initialize_weights(self):
        _initialize(self, self.blocks)
        nn.init.normal_(self.slot, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

