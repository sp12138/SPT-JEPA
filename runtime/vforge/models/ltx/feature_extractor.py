"""LTX-2.3 dual Gemma projection, matching the reference V2 normalization."""

import math

import torch
from torch import nn


class FeatureExtractorV2(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int, video_dim: int, audio_dim: int):
        super().__init__()
        self.embedding_dim = hidden_size
        flat_dim = hidden_size * (hidden_layers + 1)
        self.video_aggregate_embed = nn.Linear(flat_dim, video_dim)
        self.audio_aggregate_embed = nn.Linear(flat_dim, audio_dim)

    def forward(self, hidden_states, attention_mask, padding_side="left"):
        del padding_side
        encoded = torch.stack(hidden_states, dim=-1)
        variance = torch.mean(encoded**2, dim=2, keepdim=True)
        normed = (encoded * torch.rsqrt(variance + 1e-6)).flatten(2)
        normed = torch.where(attention_mask.bool().unsqueeze(-1), normed, 0.0)
        video = self.video_aggregate_embed(
            normed * math.sqrt(self.video_aggregate_embed.out_features / self.embedding_dim)
        )
        audio = self.audio_aggregate_embed(
            normed * math.sqrt(self.audio_aggregate_embed.out_features / self.embedding_dim)
        )
        return video, audio
