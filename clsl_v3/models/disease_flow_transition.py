from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP


@dataclass
class DiseaseFlowDiagnostics:
    flow_loss: torch.Tensor
    gate_mean: torch.Tensor
    semigroup_loss: torch.Tensor


class DiseaseFlowTransition(nn.Module):
    """Flow-matched disease-state transition with selective disease updates.

    The module learns a conditional vector field that transports current
    disease-specific latent states toward future disease-state posteriors. Unlike a
    deterministic one-step MLP/ODE, it supports flow-matching supervision when a
    weak future latent target is available, and it learns disease-specific update
    gates so stable conditions do not over-update.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_diseases: int,
        dropout: float = 0.1,
        time_scale: float = 180.0,
        n_time_frequencies: int = 8,
        ode_steps: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(num_diseases)
        self.time_scale = float(time_scale)
        self.n_freq = int(max(1, n_time_frequencies))
        self.ode_steps = int(max(1, ode_steps))
        self.disease_emb = nn.Parameter(torch.randn(self.k, hidden_dim) * 0.02)
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        time_dim = self.n_freq * 2 + 1
        self.graph_msg = nn.Linear(latent_dim, hidden_dim)
        self.vector_field = MLP(hidden_dim * 3 + time_dim, hidden_dim, latent_dim, dropout=dropout, layers=3)
        self.update_gate = MLP(hidden_dim * 3 + time_dim, hidden_dim, 1, dropout=dropout, layers=2)
        self.norm = nn.LayerNorm(latent_dim)

    def time_embedding(self, delta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # delta_t: [B], t: [B]
        dt = (delta_t.float() / max(self.time_scale, 1e-6)).view(-1, 1)
        tt = t.float().view(-1, 1)
        freqs = torch.exp(torch.linspace(0, 4, self.n_freq, device=delta_t.device, dtype=torch.float32)).view(1, -1)
        x = dt * tt * freqs
        return torch.cat([dt.to(x.dtype), torch.sin(x), torch.cos(x)], dim=-1).to(delta_t.dtype)

    def _adjacency(self, patient_adj: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        b, k, _ = z.shape
        if patient_adj is None:
            eye = torch.eye(k, device=z.device, dtype=z.dtype)
            return eye.unsqueeze(0).expand(b, -1, -1)
        if patient_adj.dim() == 2:
            return patient_adj.to(z.device, z.dtype).unsqueeze(0).expand(b, -1, -1)
        return patient_adj.to(z.device, z.dtype)

    def velocity(self, z_t: torch.Tensor, delta_t: torch.Tensor, t: torch.Tensor, patient_adj: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, _ = z_t.shape
        adj = self._adjacency(patient_adj, z_t)
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh_z = torch.bmm(adj / denom, z_t)
        z_h = self.z_proj(z_t)
        msg_h = self.graph_msg(neigh_z)
        disease_h = self.disease_emb.unsqueeze(0).expand(b, k, -1)
        time_h = self.time_embedding(delta_t, t).unsqueeze(1).expand(b, k, -1)
        inp = torch.cat([z_h, msg_h, disease_h, time_h], dim=-1)
        raw_v = self.vector_field(inp)
        gate = torch.sigmoid(self.update_gate(inp))
        return gate * raw_v, gate.squeeze(-1)

    def forward(self, z_current: torch.Tensor, delta_t: torch.Tensor, patient_adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = z_current
        b = z.size(0)
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t = z.new_full((b,), (i + 0.5) * dt)
            v, _ = self.velocity(z, delta_t, t, patient_adj=patient_adj)
            z = self.norm(z + dt * v)
        return z

    def flow_matching_loss(
        self,
        z_current: torch.Tensor,
        z_future_target: torch.Tensor,
        delta_t: torch.Tensor,
        patient_adj: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DiseaseFlowDiagnostics]:
        b = z_current.size(0)
        t = torch.rand(b, device=z_current.device, dtype=z_current.dtype)
        noise = torch.randn_like(z_current) * 0.01
        z_t = (1 - t.view(b, 1, 1)) * z_current.detach() + t.view(b, 1, 1) * z_future_target.detach() + noise
        target_v = z_future_target.detach() - z_current.detach()
        pred_v, gate = self.velocity(z_t, delta_t, t, patient_adj=patient_adj)
        flow_loss = F.mse_loss(pred_v, target_v)
        # Semigroup weak consistency: T(dt) ~= T(dt/2) o T(dt/2)
        half_dt = delta_t * 0.5
        z_half = self.forward(z_current, half_dt, patient_adj=patient_adj)
        z_two_half = self.forward(z_half, half_dt, patient_adj=patient_adj)
        z_full = self.forward(z_current, delta_t, patient_adj=patient_adj)
        semigroup = F.mse_loss(z_two_half, z_full.detach())
        loss = flow_loss + 0.1 * semigroup
        if not return_diagnostics:
            return loss
        return loss, DiseaseFlowDiagnostics(flow_loss=flow_loss.detach(), gate_mean=gate.mean().detach(), semigroup_loss=semigroup.detach())


class FutureLabelPosteriorEncoder(nn.Module):
    """Weak future latent target from partially observed future labels."""

    def __init__(self, num_diseases: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.k = int(num_diseases)
        self.latent_dim = int(latent_dim)
        self.disease_emb = nn.Parameter(torch.randn(self.k, hidden_dim) * 0.02)
        self.net = MLP(3 + hidden_dim, hidden_dim, latent_dim, dropout=dropout, layers=3)

    def forward(self, y_future: torch.Tensor, y_future_mask: torch.Tensor, base_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, k = y_future.shape
        y = torch.nan_to_num(y_future.float(), nan=0.0)
        m = y_future_mask.float()
        unknown = 1.0 - m
        emb = self.disease_emb.unsqueeze(0).expand(b, k, -1)
        inp = torch.cat([y.unsqueeze(-1), m.unsqueeze(-1), unknown.unsqueeze(-1), emb], dim=-1)
        z = self.net(inp)
        if base_z is not None:
            # Only partially override base latent; labels are a weak target, not a full state.
            z = 0.5 * z + 0.5 * base_z.detach()
        return z
