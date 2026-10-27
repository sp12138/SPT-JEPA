import torch
from torch import nn

class QueryPool(nn.Module):                      # K learned queries cross-attend to one frame's tokens
    def __init__(self, width, heads, K, d):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(K, width) * 0.02)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, d)          # plain linear head: no normalization after it
    def forward(self, tokens):                   # [N, L, width] -> mu: [N, K, d]
        q = self.queries.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(q, tokens, tokens, need_weights=False)
        return self.head(self.norm(pooled))
    def initialize(self):                        # must run after meta-device materialization (branch S)
        nn.init.normal_(self.queries, std=0.02)
        self.attn.out_proj.reset_parameters()    # not covered by _reset_parameters()
        self.attn._reset_parameters()            # input projection xavier; both biases zero
        nn.init.ones_(self.norm.weight); nn.init.zeros_(self.norm.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)
