from __future__ import annotations

"""Disease-specific graph-aware latent transition for CLSL-v2.

This module replaces a global one-step MLP transition with a disease-level
neural residual flow. It supports z with shape [B, K, latent_dim], optional
current disease states, and a sparse learnable transition graph initialized from
clinical priors.
"""

import math
from typing import Dict, Optional

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
        angles = tau / self.freqs.to(tau.device, tau.dtype).view(1, -1)
        return torch.cat([tau, torch.sin(angles), torch.cos(angles)], dim=-1)


class SnapshotToTrajectoryTransition(nn.Module):
    """Graph-aware neural residual flow from current disease states to future states.

    Parameters
    ----------
    latent_dim:
        Per-disease latent dimension.
    hidden_dim:
        Hidden dimension for the vector field.
    num_diseases:
        If supplied, enables [B,K,L] disease-specific transition. If omitted,
        the module remains backward compatible with [B,L] global latents.
    prior_adjacency:
        Optional [K,K] sparse clinical transition graph used as the initial
        disease graph.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        num_diseases: Optional[int] = None,
        prior_adjacency: Optional[torch.Tensor] = None,
        ode_steps: int = 4,
        time_scale: float = 180.0,
        n_time_frequencies: int = 8,
        max_residual_edge: float = 0.25,
        init_edge_logit: float = -6.0,
        self_loop_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_diseases = None if num_diseases is None else int(num_diseases)
        self.ode_steps = int(max(1, ode_steps))
        self.time_scale = float(time_scale)
        self.max_residual_edge = float(max_residual_edge)
        self.self_loop_weight = float(self_loop_weight)
        self.time_emb = TimeFourierEmbedding(n_frequencies=n_time_frequencies)

        self.global_field = MLP(
            latent_dim + self.time_emb.out_dim,
            hidden_dim,
            latent_dim,
            dropout=dropout,
            layers=3,
        )
        self.global_gate = nn.Sequential(
            nn.Linear(latent_dim + self.time_emb.out_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.global_norm = nn.LayerNorm(latent_dim)

        if self.num_diseases is not None:
            if prior_adjacency is None:
                prior_adjacency = torch.zeros(self.num_diseases, self.num_diseases, dtype=torch.float32)
            if prior_adjacency.shape != (self.num_diseases, self.num_diseases):
                raise ValueError(
                    f"prior_adjacency must have shape {(self.num_diseases, self.num_diseases)}, "
                    f"got {tuple(prior_adjacency.shape)}"
                )
            self.register_buffer("prior_adjacency", prior_adjacency.float().clamp_min(0.0), persistent=True)
            self.edge_logits = nn.Parameter(torch.full((self.num_diseases, self.num_diseases), float(init_edge_logit)))
            self.disease_emb = nn.Parameter(torch.randn(self.num_diseases, latent_dim) * 0.02)

            node_in = latent_dim * 3 + self.time_emb.out_dim + 2
            self.node_field = MLP(node_in, hidden_dim, latent_dim, dropout=dropout, layers=3)
            self.node_gate = nn.Sequential(nn.Linear(node_in, latent_dim), nn.Sigmoid())
            self.node_norm = nn.LayerNorm(latent_dim)
        else:
            self.register_buffer("prior_adjacency", torch.empty(0), persistent=False)
            self.edge_logits = None
            self.disease_emb = None
            self.node_field = None
            self.node_gate = None
            self.node_norm = None

    def _tau(self, delta_t_days: torch.Tensor) -> torch.Tensor:
        return torch.log1p(delta_t_days.float().clamp_min(0.0) / self.time_scale)

    def transition_adjacency(self) -> Optional[torch.Tensor]:
        if self.num_diseases is None or self.edge_logits is None:
            return None
        logits = 0.5 * (self.edge_logits + self.edge_logits.t())
        eye = torch.eye(self.num_diseases, device=logits.device, dtype=logits.dtype)
        residual = torch.sigmoid(logits) * self.max_residual_edge
        prior = self.prior_adjacency.to(logits.device, logits.dtype)
        adj = (prior + residual) * (1.0 - eye)
        adj = 0.5 * (adj + adj.t())
        return adj + eye * self.self_loop_weight

    def _global_vector_field(self, z: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        te = self.time_emb(tau).to(z.device, z.dtype)
        inp = torch.cat([z, te], dim=-1)
        return self.global_field(inp) * self.global_gate(inp)

    def _node_vector_field(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.num_diseases is None or self.node_field is None or self.node_gate is None:
            raise RuntimeError("Disease-level transition was called but num_diseases is not configured.")
        b, k, _ = z.shape
        if k != self.num_diseases:
            raise ValueError(f"Expected K={self.num_diseases} diseases, got K={k}")

        adj = adjacency if adjacency is not None else self.transition_adjacency()
        if adj is None:
            adj = torch.eye(k, device=z.device, dtype=z.dtype)
        adj = adj.to(z.device, z.dtype)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(b, -1, -1)
        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj_norm = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        graph_msg = torch.bmm(adj_norm, z)

        te = self.time_emb(tau).to(z.device, z.dtype).unsqueeze(1).expand(b, k, -1)
        disease_emb = self.disease_emb.to(z.device, z.dtype).unsqueeze(0).expand(b, k, -1)

        if current_state is None:
            state = torch.zeros(b, k, 1, device=z.device, dtype=z.dtype)
        else:
            state = torch.nan_to_num(current_state.float(), nan=0.0).to(z.device, z.dtype).view(b, k, 1)
        if current_state_mask is None:
            state_mask = torch.zeros(b, k, 1, device=z.device, dtype=z.dtype)
        else:
            state_mask = current_state_mask.float().to(z.device, z.dtype).view(b, k, 1)

        inp = torch.cat([z, graph_msg, disease_emb, te, state, state_mask], dim=-1)
        return self.node_field(inp) * self.node_gate(inp)

    def vector_field(
        self,
        z: torch.Tensor,
        tau: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z.dim() == 2:
            return self._global_vector_field(z, tau)
        if z.dim() == 3:
            return self._node_vector_field(
                z=z,
                tau=tau,
                adjacency=adjacency,
                current_state=current_state,
                current_state_mask=current_state_mask,
            )
        raise ValueError(f"z must be [B,L] or [B,K,L], got shape {tuple(z.shape)}")

    def forward(
        self,
        z: torch.Tensor,
        delta_t_days: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tau_total = self._tau(delta_t_days).to(z.device, z.dtype).view(-1)
        h = z
        for step in range(self.ode_steps):
            t0 = tau_total * (step / self.ode_steps)
            dt = tau_total / self.ode_steps
            view_shape = (-1,) + (1,) * (h.dim() - 1)
            dt_view = dt.view(*view_shape)
            f0 = self.vector_field(h, t0, adjacency, current_state, current_state_mask)
            h_euler = h + dt_view * f0
            f1 = self.vector_field(h_euler, t0 + dt, adjacency, current_state, current_state_mask)
            h = h + 0.5 * dt_view * (f0 + f1)
        if h.dim() == 2:
            return self.global_norm(h)
        return self.node_norm(h)

    def semigroup_loss(
        self,
        z: torch.Tensor,
        delta_t_days: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
        current_state: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-supervised consistency: T_dt(z) ~= T_dt2(T_dt1(z))."""
        dt = delta_t_days.float().clamp_min(0.0)
        dt1 = dt * 0.5
        dt2 = dt - dt1
        direct = self.forward(z, dt, adjacency, current_state, current_state_mask)
        middle = self.forward(z, dt1, adjacency, current_state, current_state_mask)
        composed = self.forward(middle, dt2, adjacency, current_state, current_state_mask)
        return F.mse_loss(composed, direct.detach())

    def graph_regularization(self) -> Dict[str, torch.Tensor]:
        if self.num_diseases is None or self.edge_logits is None:
            zero = torch.zeros((), device=self.global_norm.weight.device)
            return {"transition_graph_sparse": zero, "transition_graph_prior_alignment": zero}
        adj = self.transition_adjacency()
        assert adj is not None
        eye = torch.eye(self.num_diseases, device=adj.device, dtype=adj.dtype)
        off = 1.0 - eye
        prior = self.prior_adjacency.to(adj.device, adj.dtype)
        sparse = (adj * off).abs().mean()
        prior_alignment = ((adj * off - prior * off) ** 2).mean()
        return {
            "transition_graph_sparse": sparse,
            "transition_graph_prior_alignment": prior_alignment,
        }
