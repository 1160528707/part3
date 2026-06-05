from __future__ import annotations

"""Time-consistent latent transition operator.

Drop-in replacement for the original one-step MLP transition.  This version uses
a small neural ODE-style residual flow with Heun integration and exposes a
semigroup consistency loss:

    T_{t2}(T_{t1}(z)) ~= T_{t1+t2}(z).

That property is important for making the snapshot-to-trajectory module look
like a real latent disease dynamics model rather than a generic MLP head.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP


class TimeFourierEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 8, max_period: float = 16.0) -> None:
        super().__init__()
        freqs = torch.logspace(0.0, math.log10(max_period), n_frequencies)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return int(self.freqs.numel() * 2 + 1)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.view(-1, 1)
        angles = tau / self.freqs.view(1, -1)
        return torch.cat([tau, torch.sin(angles), torch.cos(angles)], dim=-1)


class SnapshotToTrajectoryTransition(nn.Module):
    """Neural residual flow from a snapshot latent state to a future state."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        ode_steps: int = 4,
        time_scale: float = 180.0,
        n_time_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.ode_steps = int(max(1, ode_steps))
        self.time_scale = float(time_scale)
        self.time_emb = TimeFourierEmbedding(n_frequencies=n_time_frequencies)
        self.field = MLP(latent_dim + self.time_emb.out_dim, hidden_dim, latent_dim, dropout=dropout, layers=3)
        self.gate = nn.Sequential(nn.Linear(latent_dim + self.time_emb.out_dim, latent_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(latent_dim)

    def _tau(self, delta_t_days: torch.Tensor) -> torch.Tensor:
        # Smoothly maps days to a compact positive integration time.
        return torch.log1p(delta_t_days.float().clamp_min(0.0) / self.time_scale)

    def vector_field(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(tau).to(z.device, z.dtype)
        inp = torch.cat([z, te], dim=-1)
        return self.field(inp) * self.gate(inp)

    def forward(self, z: torch.Tensor, delta_t_days: torch.Tensor) -> torch.Tensor:
        tau_total = self._tau(delta_t_days).to(z.device, z.dtype).view(-1, 1)
        h = z
        for step in range(self.ode_steps):
            t0 = tau_total * (step / self.ode_steps)
            dt = tau_total / self.ode_steps
            f0 = self.vector_field(h, t0)
            h_euler = h + dt * f0
            f1 = self.vector_field(h_euler, t0 + dt)
            h = h + 0.5 * dt * (f0 + f1)
        return self.norm(h)

    def semigroup_loss(self, z: torch.Tensor, delta_t_days: torch.Tensor) -> torch.Tensor:
        """Self-supervised temporal consistency loss for unlabeled visits."""
        dt = delta_t_days.float().clamp_min(0.0)
        dt1 = dt * 0.5
        dt2 = dt - dt1
        direct = self.forward(z, dt)
        composed = self.forward(self.forward(z, dt1), dt2)
        return F.mse_loss(composed, direct.detach())
