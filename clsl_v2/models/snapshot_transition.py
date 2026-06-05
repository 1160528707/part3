from __future__ import annotations

import torch
from torch import nn
from .components import MLP


class SnapshotToTrajectoryTransition(nn.Module):
    """Latent transition operator from a single snapshot to future latent state."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = MLP(latent_dim + 1, hidden_dim, latent_dim, dropout=dropout, layers=3)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, delta_t_days: torch.Tensor) -> torch.Tensor:
        dt = torch.log1p(delta_t_days.float().clamp_min(0.0)).view(-1, 1) / 6.0
        dz = self.net(torch.cat([z, dt], dim=-1))
        return self.norm(z + dz)
